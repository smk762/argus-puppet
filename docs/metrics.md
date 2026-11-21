# Pose metrics: set stats, labels and set-vs-set comparison

Three library calls, all deterministic and independent of input order:

| call | module | returns |
|---|---|---|
| `pose_set_stats(results, params=None)` | `argus_puppet.metrics` | `PoseSetStats`: diversity, clusters, coverage, a 2D scatter |
| `compare_pose_sets(outputs, reference, threshold=0.08, params=None)` | `argus_puppet.compare` | `PoseSetComparison`: nearest training frame per output, memorised share, coverage drift |
| `pose_labels(keypoints, min_joint_score=0.3)` | `argus_puppet.labels` | `PoseLabels`: stance, arms, facing |

`load_sidecars(folder)` (`argus_puppet.sidecars`) reads a folder's `*.pose.json` sidecars, so
`pose_set_stats(load_sidecars(folder))` measures a folder that has already been through `argus-puppet pose`.

Every distance is the mirror-invariant pose distance of [pose-vector.md](pose-vector.md), in torso units.

## Parameters

| parameter | default | meaning |
|---|---|---|
| `min_joint_score` | 0.3 | a joint is confident at or above this score |
| `min_usable_joints` | 8 | confident body joints (of 17) a primary person needs to be usable |
| `cluster_threshold` | 0.25 | average-linkage cut: clusters merge while closer than this |
| `min_shared_joints` | 6 | body joints confident in both poses for their distance to be defined |
| `min_shared_limb_joints` | 2 | of those, how many must be limb joints (elbows, wrists, knees, ankles) |
| `diversity_scale` | 0.3 | the distance scale `s` of the diversity score below |
| memorisation `threshold` | 0.08 | `compare_pose_sets`: an output closer than this to a reference frame is flagged |

All but the memorisation threshold travel in every result as `MetricParams`, whose fields default to these values
(`MetricParams()`, also `metrics.DEFAULT_PARAMS`); a request may give any subset. The threshold is recorded on the
comparison itself.

## Set stats

Each image contributes its **primary person** (`PoseResult.primary`). An image is **usable** when it has one and
`embed()` accepts it (see [pose-vector.md](pose-vector.md#unusable-poses)). Unusable images are listed in `items`
with `usable: false`, keep whatever labels the rules can still decide (an upper-body shot still has arms), and are
left out of every number.

| field | definition |
|---|---|
| `count`, `usable` | images, and images with a usable primary pose |
| `mean_pairwise_distance` | mean `d` over all comparable usable pairs; 0 under two |
| `diversity` | see below |
| `clusters` | average-linkage clusters, largest first: `id`, `size`, `share = size / usable`, `medoid` |
| `largest_cluster_share` | share of the largest cluster; 0 when nothing is usable |
| `coverage` | per label, the number of usable poses carrying it; every label listed, zero or not |
| `items` | one per image, **sorted by image name**: `usable`, `cluster`, `labels`, `xy` |

### Clustering

Average linkage (UPGMA) on the pairwise distance matrix, merging while the two nearest clusters are closer than
`cluster_threshold` on average. Average linkage sits between single linkage (which chains a slow drift of poses
into one cluster) and complete linkage (which splits a cluster at its single worst pair). Undefined (NaN) distances
count as infinite, so a cluster never contains an incomparable pair.

The implementation is the nearest-neighbour-chain algorithm with the Lance-Williams update: O(N^2) time on one
N x N working copy, with no SciPy. It is checked against the textbook O(N^3) algorithm in the tests. Items are
sorted by image name before clustering and ties resolve by index, so the result is independent of input order.

Cluster **ids** run from 0 for the largest; equal sizes are ordered by the alphabetically first image in each. The
**medoid** is the member with the smallest summed distance to the other members (the first by name on a tie).

**Choosing 0.25.** Two renders of one pose with estimator-level noise and a few degrees of limb variation sit 0.03
to 0.15 apart. The closest distinct synthetic presets (standing vs hands on hips, 0.12; standing vs frontal seated,
0.22; T-pose vs arms up, 0.30) sit near or above the cut, and different stances sit 0.4 to 1.6 apart. So 0.25 merges
jittered copies of one pose and keeps "arms out" and "arms up" apart, but it does not split poses that differ only in
the angle of the elbows.

### Diversity

`diversity` is the number argus-proof's `--min-pose-diversity` gate reads. It is the mean, over all comparable pairs
of usable poses, of a saturating transform of their distance:

```
g(d)      = 1 - exp(-(d / s)^2)                         (s = diversity_scale, default 0.3)
diversity = mean_{i < j, d_ij defined} g(d_ij)          (0 under two usable poses or no defined pair)
```

Read it as the probability that two poses drawn at random from the set are "clearly different": `g` is 0 for
identical poses, 0.06 at 0.075 (estimator jitter), 0.63 at 0.3 (the cluster scale), and 0.98 at 0.6 (a different
stance).

Why this form:

- **0 for N identical poses** and for fewer than two usable poses.
- **Monotone in spread:** spreading the poses further apart raises every `g(d_ij)`.
- **Saturating:** each pair contributes at most 1, so the score is in [0, 1). A set of K equally common, mutually
  distinct poses scores about `1 - 1/K`.
- **Not dominated by one outlier:** a single odd pose among N touches only `N - 1` of the `N(N-1)/2` pairs and adds
  at most `2/N` (one lying pose added to 40 standing ones adds 0.05). A raw mean distance would let one far pose
  pull the whole set up.
- **Stable as N grows:** it is a U-statistic, an unbiased estimate of `E[g(d)]` for two random draws from the
  underlying distribution, so it does not drift with set size (see the table: N = 50 and N = 500 agree within 0.01).
- **A flat start:** the squared exponent gives `g` zero slope at `d = 0`, so estimator jitter on a repeated pose
  costs almost nothing, while a real change of pose counts in full.

A blend with an effective number of clusters was considered and left out: the cluster count depends on the cut, and
the mean of `g` already behaves like an effective-number measure (`1 / (1 - diversity)` is roughly the number of
equally common distinct poses).

#### Calibration

Synthetic stick figures (`tests/posegen.py`) at random scale, position and mirroring, with every limb angle jittered
uniformly by the given amount and every joint moved by Gaussian noise of the given standard deviation in torso
units. Default parameters.

| set | N | diversity | mean distance | clusters | largest share |
|---|---|---|---|---|---|
| one jump shot, tight (3 deg, 0.02) | 50 | 0.031 | 0.051 | 1 | 1.00 |
| one jump shot, tight (3 deg, 0.02) | 500 | 0.030 | 0.050 | 1 | 1.00 |
| one jump shot (5 deg, 0.03) | 50 | 0.070 | 0.078 | 1 | 1.00 |
| one jump shot (5 deg, 0.03) | 500 | 0.068 | 0.077 | 1 | 1.00 |
| one tuck jump (5 deg, 0.03) | 50 | 0.052 | 0.068 | 1 | 1.00 |
| one tuck jump (5 deg, 0.03) | 500 | 0.051 | 0.067 | 1 | 1.00 |
| one jump shot, loose (8 deg, 0.05) | 50 | 0.171 | 0.127 | 1 | 1.00 |
| one jump shot, loose (8 deg, 0.05) | 500 | 0.168 | 0.127 | 1 | 1.00 |
| 90% one jump shot, rest broad mix | 200 | 0.198 | 0.174 | 11 | 0.92 |
| 80% one jump shot, rest broad mix | 200 | 0.379 | 0.351 | 15 | 0.80 |
| 70% one jump shot, rest broad mix | 200 | 0.485 | 0.383 | 15 | 0.71 |
| two poses, 50/50 | 50 | 0.526 | 0.331 | 2 | 0.56 |
| two poses, 50/50 | 500 | 0.515 | 0.324 | 2 | 0.55 |
| standing, five arm positions | 50 | 0.573 | 0.288 | 4 | 0.32 |
| standing, five arm positions | 500 | 0.564 | 0.287 | 4 | 0.37 |
| 50% one jump shot, rest broad mix | 200 | 0.681 | 0.550 | 15 | 0.53 |
| broad mix, 16 presets (10 deg, 0.03) | 50 | 0.879 | 0.675 | 15 | 0.18 |
| broad mix, 16 presets (10 deg, 0.03) | 500 | 0.883 | 0.732 | 15 | 0.12 |

The scale 0.3 was chosen so that "the same jump shot with small jitter" lands at 0.05-0.15 and a broad mix of
stances and arm positions at 0.7 or more, with room between them for a gate. With 200 poses, scales of 0.25 / 0.3 /
0.35 give 0.10 / 0.07 / 0.05 for the jump shot, 0.25 / 0.18 / 0.14 for the loose jump shot, and 0.92 / 0.89 / 0.86
for the broad mix: 0.25 pushes a single loose pose toward the gate, 0.35 compresses the low end. The synthetic figures share one body's proportions; real people vary, which
adds a little distance to every pair, so expect real single-pose sets to score somewhat higher than the table.

#### Suggested gate for proof

**`--min-pose-diversity 0.3`, reject code `pose_collapse`.** On the calibration sets it rejects every single-pose
set (up to 0.17) and a set that is about 85% one pose or more. It passes a 50/50 two-pose set (0.52) and anything
broader. Proof should confirm the threshold on one real run, as it did for `--min-likeness`, and report
`largest_cluster_share` and the medoid beside the score, so a rejection says *which* pose the outputs collapsed onto.

## Labels

`pose_labels` derives three coarse labels from keypoint geometry. **It is a set of rules, not a classifier.** Every
length is in torso units (the robust scale of [pose-vector.md](pose-vector.md#normalisation)), every angle in
degrees, so labels do not change with translation or scale. Stance and arms do not change under a mirror; facing
swaps `left` and `right`. A label is `unknown` whenever the joints its rule needs are not confident: the rules
prefer `unknown` to a guess. The thresholds are the named constants at the top of `labels.py`.

### Stance

Gravity is assumed to point down the image (a roughly level camera). For each leg with a confident hip and knee:
the **thigh angle** from straight down, the visible **thigh length**, the **knee rise** above the hip, and with the
ankle, the interior **knee angle** (180 = straight), the **ankle drop** below the hip and the shin's drop below the
knee.

| stance | rule (first match wins) |
|---|---|
| unknown | no anchor; or the torso is foreshortened (visible torso under 0.5 of the scale, so its angle means nothing); or the torso is past 135 degrees from vertical (inverted) |
| lying | torso 60-135 degrees from vertical, unless every visible leg is a straight vertical standing leg (that is bending over, so it is *standing*) |
| unknown | upright torso but no whole leg (hip, knee and ankle): an upper-body shot, or a vertical thigh that could be standing or kneeling |
| jumping (star) | torso within 25 degrees of vertical; both legs whole and straight (knee 150 or more), each splayed 15 degrees or more to opposite sides, ankles 0.9 or more apart; both arms up or out |
| jumping (tuck) | torso within 25 degrees of vertical; both legs whole with knees bent (110 or less), both knees 0.25 or more above the hips, feet still 0.2 or more below the hips |
| standing | every visible leg: thigh within 35 degrees of down, visible thigh 0.55 or longer, knee 150 or straighter |
| seated | every visible leg is one of: a chair sit (thigh 55 degrees or more from down, knee no more than 0.25 above the hip, knee angle 70 or more, shin hanging 0.4 or more, feet 0.6 or more below the hips); legs out on the floor (thigh 55 or more, knee 140 or more, feet within 0.5 of hip height); thighs toward the camera (visible thigh under 0.45, knee not above the hip, shin hanging 0.5 or more, feet 0.7 or more below the hips) |
| crouching | every whole leg is bent (110 or less) with the hips within 0.8 of the ankles |
| unknown | anything else |

### Arms

In the torso's own frame ("up" is mid-hip to mid-shoulder, so a lying subject's raised arms are still "up"). Per arm
with a confident shoulder and wrist, `rise` is the wrist's height above the shoulder along the torso axis and
`lateral` its distance out to the side:

| per arm | rule |
|---|---|
| up | rise 0.25 or more |
| out | lateral 0.6 or more and rise -0.5 or more |
| down | anything else: hanging, hands on hips, arms folded, a hand at the face |

The set label is the arms' common class, `mixed` when the two visible arms differ, the one visible arm's class when
only one is visible, and `unknown` when neither is.

### Facing

The subject's facing as the camera sees it: `left` means they face the image's left.

- **Profile** when exactly one shoulder is confident, or both are and their separation across the body is under
  0.35 of the torso scale (frontal adults show about 0.7, a three-quarter view about 0.5). Then the nose decides: 0.05
  or more to the left of the ears (or the shoulders, without confident ears) is `left`, to the right is `right`. No
  nose, an offset under 0.05, or a torso more than 60 degrees from vertical (left and right of the image mean
  nothing for a lying subject) is `unknown`.
- Otherwise `front` when the face is visible (the nose and an eye are confident, or the 68 face landmarks average
  0.3 or more) **and** the subject's left shoulder is on the image's right; `back` when the face is not visible and
  the left shoulder is on the image's left. If the face and the shoulder order disagree, it is `unknown`.

### Limits of the label rules

What one 2D frame cannot decide, and how the rules behave:

- **Jumping has no ground reference.** Only the tuck and star silhouettes count. A star pose struck on the ground
  reads as jumping; a deep squat with an upright torso and the knees clearly above the hips reads as a tuck jump; a
  running or stepping jump reads as standing. The image's frame was considered as a ground cue and rejected: full
  length photos often show plenty of floor below standing feet.
- **The camera is assumed level.** A strongly tilted camera tilts every stance; a photo shot straight down at a
  lying person looks upright.
- **Cross-legged floor sitting** has bent knees with the feet near hip height, so it reads as crouching.
  **Kneeling** reads as unknown or crouching. Neither is a label.
- **Profile arms are measured in the image plane:** an arm reaching toward the camera foreshortens and can read as
  down.
- **One visible arm or leg decides alone** (the occluded one is assumed to agree), which can mislabel `mixed` arms
  in profile.
- **Facing trusts the estimator's left/right assignment.** Pose estimators sometimes swap sides on back views; the
  face-visibility cross-check turns most such swaps into `unknown` rather than a wrong label.
- **Synthetic tests only.** The rules are tested on parametric stick figures with jitter, dropout, scale,
  translation and mirroring; their accuracy on real photographs has not yet been measured.

## The 2D projection (`items[].xy`)

For the demo's pose-space scatter: the usable vectors are canonicalised jointly (`align_to_mean`), masked joints are
imputed with the set's mean for that joint, each joint is weighted by the square root of its distance weight, and
the two leading principal components are taken by SVD. Each component's sign is fixed so that its largest
loading is positive, which makes the plot reproducible. A single usable pose sits at (0, 0). The projection is a
picture, not a measurement: distances in it only approximate pose distances, and imputed joints pull sparse poses
toward the centre.

## Set-vs-set comparison

`compare_pose_sets(outputs, reference)` embeds both sets with the same parameters and, for every output, finds the
nearest usable reference pose under the mirror-invariant distance.

| field | definition |
|---|---|
| `matches` | one per output image, sorted by image name: `nearest`, `distance`, `memorised = distance < threshold`. An output that is unusable, or shares too few joints with every usable reference pose, has `nearest` and `distance` null and `memorised` false. Ties for nearest go to the alphabetically first reference image |
| `output_count` | every output image, usable or not (one `matches` entry each) |
| `matched_count` | outputs with a nearest reference pose |
| `reference_count` | usable reference poses |
| `memorised_share` | memorised outputs / `matched_count`; 0 when nothing matched. Unusable and unmatched outputs are left out of the denominator: they are no evidence either way, so check `matched_count` before reading the share |
| `output_coverage`, `reference_coverage` | label histograms of the usable poses of each set |
| `coverage_divergence` | Jensen-Shannon divergence (base 2) between the two coverages, averaged over stance, arms and facing; 0 = the same mix of poses, 1 = disjoint |

`unknown` counts as a label in the divergence, so a rise in unreadable bodies among the outputs registers as drift.
If one set has no usable poses and the other does, each histogram's divergence is 1; if neither does, 0.

### The memorisation threshold

`threshold` defaults to **0.08**: a near-duplicate pose, not merely the same stance. Re-estimating one pose with
per-joint noise of 0.02 torso units (a few pixels on a typical subject) gives distances with a median of about 0.04
and a 99th percentile of about 0.09 on the synthetic figures. The nearest two *different* presets (standing and
hands on hips) are 0.12 apart, and distinct stances 0.3 to 1.5. So 0.08 flags a re-estimated copy of a frame in most
cases while staying below every distinct pose.

Two caveats proof must carry into its report:

- **A pose match is evidence, not proof, of memorisation.** Different people strike the same pose; a pose within
  roughly 10 degrees per limb of a training frame scores like a copy (about 0.05), because a 2D pose cannot tell the
  two apart. A pose-conditioned generator, or one driven from a pose library, reproduces reference poses by design.
- It is strongest **beside** the identity-leak and image-similarity checks: a pose match to a frame whose face also
  matches that frame's person is a memorised frame. A pose match alone says the composition was copied.

## How argus-proof should use this

From smk762/argus-proof#122:

1. **Pose collapse gate.** Run `pose_set_stats` on the outputs (or `load_sidecars` on a folder `argus-puppet pose`
   has processed). Reject with `pose_collapse` when `diversity < --min-pose-diversity` (suggested default 0.3).
   Report `largest_cluster_share`, the largest cluster's `medoid`, and `usable / count`. A run with too few usable
   poses (say under 10) is "not measurable", not a pass.
2. **Training-distribution context.** Run `compare_pose_sets(outputs, training)`. A collapsed run with low
   `coverage_divergence` collapsed onto a pose that dominates the training set: a dataset-balance problem for
   curator's coverage panel. High divergence means the model drifted away from the training mix.
3. **Pose memorisation evidence.** The memorised matches (`matches[].memorised`, `nearest`) name the training
   frames an output copies in pose. Feed them to the leak check (1) as evidence; do not gate on pose alone.

## Cost

On a desktop CPU with synthetic poses, `pose_set_stats` takes about 0.4 s for 1000 images and 3.5 s for 5000 (of
which embedding and labelling are about 1.3 s, the distance matrix 1.4 s, clustering 0.4 s). Peak memory at 5000 is
about two N x N float32 matrices (100 MB each). `compare_pose_sets` on 2500 outputs against 2500 references takes
about 2 s.
