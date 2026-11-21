"""An output set's poses against a reference (training) set: :func:`compare_pose_sets`.

For every output image, the nearest usable reference pose under the mirror-invariant distance (none when the output
is unusable or comparable with no reference pose), flagged ``memorised`` when it is closer than ``threshold``; and how far the two sets' label mixes have drifted apart
(Jensen-Shannon divergence of the coverage histograms). See ``docs/metrics.md``.

A pose match is evidence of memorisation, not proof: two different people can strike the same pose, and a pose
library or a pose-conditioned generator will reproduce reference poses by design. Proof should read it beside the
face-leak and image-similarity checks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from argus_puppet.embedding import cross_distances
from argus_puppet.metrics import ARMS, DEFAULT_PARAMS, FACINGS, STANCES, coverage, embed_set, shared_estimator
from argus_puppet.models import Coverage, MetricParams, PoseMatch, PoseResult, PoseSetComparison

# Distance under which an output pose counts as a near-duplicate of a reference frame, in torso units. Re-estimating
# one pose with per-joint noise of 0.02 torso units (a few pixels on a typical subject) gives distances whose 99th
# percentile is ~0.09 on the synthetic generators and whose median is ~0.04; the nearest two *different* presets
# (standing and hands-on-hips) are 0.12 apart and distinct stances 0.3-1.5. 0.08 flags a re-estimated copy of a frame
# in most cases while staying under every distinct pose. It also flags a pose within roughly +-10 degrees per limb of
# the frame, which a 2D pose cannot tell apart from a copy. See docs/metrics.md.
DEFAULT_MEMORISATION_THRESHOLD = 0.08


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
    threshold: float = DEFAULT_MEMORISATION_THRESHOLD,
    params: MetricParams | None = None,
) -> PoseSetComparison:
    """Each output's nearest reference pose, the memorised share, and the coverage drift.

    ``matches`` lists one entry per output image, sorted by image name. An output that is unusable, or shares too few
    joints with every usable reference pose, has ``nearest`` and ``distance`` null and is not memorised.
    ``output_count`` counts every output image, ``matched_count`` those with a nearest reference, and
    ``memorised_share`` is over the matched ones (0 when none matched). Ties for the nearest reference go to the
    alphabetically first reference image. *params* (embedding and shared-joint rules) default to the suite's.
    """
    params = params or DEFAULT_PARAMS
    if threshold < 0:
        raise ValueError(f"threshold {threshold} must be >= 0")
    out_set = embed_set(outputs, params)
    ref_set = embed_set(reference, params)
    out_idx, ref_idx = out_set.usable_index, ref_set.usable_index
    oc, om = out_set.usable_arrays()
    rc, rm = ref_set.usable_arrays()
    dist = cross_distances(
        oc,
        om,
        rc,
        rm,
        min_shared_joints=params.min_shared_joints,
        min_shared_limb_joints=params.min_shared_limb_joints,
    )
    row_of = {i: row for row, i in enumerate(out_idx)}

    matches: list[PoseMatch] = []
    for i, image in enumerate(out_set.images):
        row = row_of.get(i)
        d = dist[row] if row is not None and len(ref_idx) else np.zeros(0)
        finite = np.isfinite(d)
        if not finite.any():
            matches.append(PoseMatch(image=image, memorised=False))
            continue
        j = int(np.argmin(np.where(finite, d, np.inf)))  # first (alphabetical) on ties
        dj = float(d[j])
        matches.append(
            PoseMatch(image=image, nearest=ref_set.images[ref_idx[j]], distance=dj, memorised=dj < threshold)
        )

    out_cov = coverage([out_set.labels[i] for i in out_idx])
    ref_cov = coverage([ref_set.labels[i] for i in ref_idx])
    matched = sum(1 for m in matches if m.nearest is not None)
    memorised = sum(1 for m in matches if m.memorised)
    return PoseSetComparison(
        output_count=len(matches),
        matched_count=matched,
        reference_count=len(ref_idx),
        matches=matches,
        memorised_share=memorised / matched if matched else 0.0,
        threshold=threshold,
        coverage_divergence=coverage_divergence(out_cov, ref_cov),
        output_coverage=out_cov,
        reference_coverage=ref_cov,
        params=params,
        estimator=shared_estimator([*outputs, *reference]),
    )
