"""An output set's poses against a reference (training) set: :func:`compare_pose_sets`.

For every output image, the nearest usable reference pose under the mirror-invariant distance (none when the output
is unusable or comparable with no reference pose), flagged ``close`` when it is closer than ``threshold`` over the
whole limb skeleton; the same statistic inside the reference set (leave-one-out) as a baseline; and how far the two
sets' label mixes have drifted apart (Jensen-Shannon divergence of the coverage histograms). See ``docs/metrics.md``.

A close pose match means "the same pose", not "a copied frame". Independent people in a common stance match at the
default threshold most of the time, and the more reference frames there are the likelier it gets. The per-output
flag is a pointer to the training frame an image-level check (argus-proof's leak and identity checks) should look
at; the run-level signal is ``close_share`` well above ``baseline_close_share`` (``close_excess``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from argus_puppet.embedding import nearest_neighbours
from argus_puppet.metrics import ARMS, DEFAULT_PARAMS, FACINGS, STANCES, coverage, embed_set, shared_estimator
from argus_puppet.models import (
    Coverage,
    DistanceQuantiles,
    MetricParams,
    PoseMatch,
    PoseResult,
    PoseSetComparison,
)

# Distance under which two poses (sharing the whole limb skeleton) are a close pose match, in torso units.
# Re-estimating one pose with per-joint noise of 0.02 torso units (a few pixels on a typical subject) gives distances
# with a median of 0.031 and 84 % under 0.04 on the synthetic generators (all of them at 0.01 noise, p99 0.030).
# Independent draws of one stance with 8 degrees of limb jitter sit 0.05-0.06 from their nearest neighbour among
# 10-200 frames, so at 0.04 only 0-6 % of them match, and the reference set's own baseline absorbs that. At 0.08
# nearly every same-stance pose matched (and the baseline with it), which left no room to see copying. See
# docs/metrics.md.
DEFAULT_CLOSE_THRESHOLD = 0.04


def _quantiles(values: np.ndarray) -> DistanceQuantiles | None:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return None
    p01, p10, p50, p90 = (float(x) for x in np.quantile(v, (0.01, 0.10, 0.50, 0.90)))
    return DistanceQuantiles(n=len(v), p01=p01, p10=p10, p50=p50, p90=p90)


def js_divergence(p: Mapping[str, int] | Sequence[float], q: Mapping[str, int] | Sequence[float]) -> float:
    """Jensen-Shannon divergence, base 2, between two histograms (in [0, 1]).

    Mappings are aligned on the union of their keys. 0 for identical distributions, 1 for disjoint supports. When
    one histogram is empty the divergence is 1 (nothing in common) unless both are, which gives 0.
    """
    if isinstance(p, Mapping) or isinstance(q, Mapping):
        pm = dict(p) if isinstance(p, Mapping) else {}
        qm = dict(q) if isinstance(q, Mapping) else {}
        keys = sorted(set(pm) | set(qm))
        pa = np.array([pm.get(k, 0) for k in keys], dtype=np.float64)
        qa = np.array([qm.get(k, 0) for k in keys], dtype=np.float64)
    else:
        pa = np.asarray(p, dtype=np.float64)
        qa = np.asarray(q, dtype=np.float64)
    if pa.shape != qa.shape:
        raise ValueError(f"histograms differ in length: {pa.shape} vs {qa.shape}")
    if (pa < 0).any() or (qa < 0).any():
        raise ValueError("histogram counts must be >= 0")
    sp, sq = pa.sum(), qa.sum()
    if sp == 0 and sq == 0:
        return 0.0
    if sp == 0 or sq == 0:
        return 1.0
    pa, qa = pa / sp, qa / sq
    m = (pa + qa) / 2

    def kl(a: np.ndarray) -> float:
        nz = a > 0
        return float((a[nz] * np.log2(a[nz] / m[nz])).sum())

    return float(min(1.0, max(0.0, (kl(pa) + kl(qa)) / 2)))


def coverage_divergence(a: Coverage, b: Coverage) -> float:
    """Mean JS divergence (base 2) of the stance, arms and facing histograms of two coverages, in [0, 1]."""
    parts = (
        js_divergence({k: a.stance.get(k, 0) for k in STANCES}, {k: b.stance.get(k, 0) for k in STANCES}),
        js_divergence({k: a.arms.get(k, 0) for k in ARMS}, {k: b.arms.get(k, 0) for k in ARMS}),
        js_divergence({k: a.facing.get(k, 0) for k in FACINGS}, {k: b.facing.get(k, 0) for k in FACINGS}),
    )
    return float(np.mean(parts))


def compare_pose_sets(
    outputs: Sequence[PoseResult],
    reference: Sequence[PoseResult],
    *,
    threshold: float = DEFAULT_CLOSE_THRESHOLD,
    params: MetricParams | None = None,
) -> PoseSetComparison:
    """Each output's nearest reference pose, the close share against the reference set's own baseline, and the
    coverage drift.

    ``matches`` lists one entry per output image, sorted by image name. An output that is unusable, or shares too few
    joints with every usable reference pose, has ``nearest``, ``distance`` and ``shared_joints`` null and is not
    close; an unusable one also carries its ``unusable_reason`` (null for a usable but unmatched output). An output is close when some reference pose sharing at least ``params.min_close_joints`` of the 12
    non-head joints with it is closer than *threshold*; its match then names the nearest such pose, and otherwise the
    nearest comparable pose of any coverage. ``close_share`` is over the outputs that could be close
    (``close_eligible``); ``baseline_close_share`` is the same share inside the reference set, each reference pose
    against its nearest *other* reference pose. Ties go to the alphabetically first reference image. *params*
    (embedding and shared-joint rules) default to the suite's.
    """
    params = params or DEFAULT_PARAMS
    if threshold < 0:
        raise ValueError(f"threshold {threshold} must be >= 0")
    out_set = embed_set(outputs, params)
    ref_set = embed_set(reference, params)
    out_idx, ref_idx = out_set.usable_index, ref_set.usable_index
    oc, om = out_set.usable_arrays()
    rc, rm = ref_set.usable_arrays()
    rules = {
        "min_shared_joints": params.min_shared_joints,
        "min_shared_limb_joints": params.min_shared_limb_joints,
        "min_core_joints": params.min_close_joints,
    }
    nn = nearest_neighbours(oc, om, rc, rm, **rules)
    row_of = {i: row for row, i in enumerate(out_idx)}

    matches: list[PoseMatch] = []
    for i, image in enumerate(out_set.images):
        row = row_of.get(i)
        if row is None or nn.index[row] < 0:
            matches.append(PoseMatch(image=image, close=False, unusable_reason=out_set.reasons[i]))
            continue
        close = bool(nn.full_index[row] >= 0 and nn.full_distance[row] < threshold)
        if close:
            j, dj, shared = int(nn.full_index[row]), float(nn.full_distance[row]), int(nn.full_shared[row])
        else:
            j, dj, shared = int(nn.index[row]), float(nn.distance[row]), int(nn.shared[row])
        matches.append(
            PoseMatch(image=image, nearest=ref_set.images[ref_idx[j]], distance=dj, shared_joints=shared, close=close)
        )

    eligible = nn.full_index >= 0
    n_close = sum(1 for m in matches if m.close)
    close_share = n_close / int(eligible.sum()) if eligible.any() else 0.0

    baseline: float | None = None
    ref_quantiles: DistanceQuantiles | None = None
    n_base = 0
    if len(ref_idx) >= 2:
        loo = nearest_neighbours(rc, rm, rc, rm, exclude_self=True, **rules)
        base_ok = loo.full_index >= 0
        n_base = int(base_ok.sum())
        if n_base:
            baseline = float((loo.full_distance[base_ok] < threshold).mean())
            ref_quantiles = _quantiles(loo.full_distance[base_ok])

    out_cov = coverage([out_set.labels[i] for i in out_idx])
    ref_cov = coverage([ref_set.labels[i] for i in ref_idx])
    matched = sum(1 for m in matches if m.nearest is not None)
    return PoseSetComparison(
        output_count=len(matches),
        matched_count=matched,
        reference_count=len(ref_idx),
        output_unusable_reasons=out_set.reason_counts(),
        reference_unusable_reasons=ref_set.reason_counts(),
        matches=matches,
        close_eligible=int(eligible.sum()),
        close_share=close_share,
        baseline_eligible=n_base,
        baseline_close_share=baseline,
        close_excess=None if baseline is None else max(0.0, close_share - baseline),
        output_nn_distance=_quantiles(nn.full_distance[eligible]),
        reference_nn_distance=ref_quantiles,
        threshold=threshold,
        coverage_divergence=coverage_divergence(out_cov, ref_cov),
        output_coverage=out_cov,
        reference_coverage=ref_cov,
        params=params,
        estimator=shared_estimator([*outputs, *reference]),
    )
