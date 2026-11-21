# Pose metrics: set stats, labels and set-vs-set comparison

Three library calls, all deterministic and independent of input order:

| call | module | returns |
|---|---|---|
| `pose_set_stats(results, params=None)` | `argus_puppet.metrics` | `PoseSetStats`: diversity, clusters, coverage, a 2D scatter |
| `compare_pose_sets(outputs, reference, threshold=0.04, params=None)` | `argus_puppet.compare` | `PoseSetComparison`: nearest training frame per output, close-match share against the training set's own baseline, coverage drift |
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
| `min_close_joints` | 12 | of the 12 non-head joints, how many two poses must share before they can be a close pose match (compare) or duplicates (library dedupe); 12 is the whole limb skeleton |
| close `threshold` | 0.04 | `compare_pose_sets`: a match closer than this over the whole limb skeleton is a close pose match |

All but the close threshold travel in every result as `MetricParams`, whose fields default to these values
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
| `comparable_pairs` | pairs of usable poses whose distance is defined (enough shared joints): the denominator of `diversity` and `mean_pairwise_distance` |
| `comparable_share` | `comparable_pairs / (usable (usable - 1) / 2)`; 0 under two usable. Low when many poses are partial (waist-up crops against full-body shots) |
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
- **Not dominated by one outlier:** one pose added to N touches N of the `N(N+1)/2` pairs, so it raises the score
  by at most `2 (1 - diversity) / (N + 1)` (one lying pose added to 40 standing ones adds 0.046, against a bound of
  0.047). The mean distance of the same set rises by 2.5 times: a raw mean lets one far pose pull the whole set up.
- **Stable as N grows:** it is a U-statistic, an unbiased estimate of `E[g(d)]` for two random draws from the
  underlying distribution, so it does not drift with set size: the mean over 40 sets of 6 broad-mix poses is within
  0.01 of one 400-pose set, where counting each pose against itself (a V-statistic) would give 5/6 of it.
- **A flat start:** the squared exponent gives `g` zero slope at `d = 0`, so estimator jitter on a repeated pose
  costs almost nothing, while a real change of pose counts in full.

A blend with an effective number of clusters was considered and left out: the cluster count depends on the cut, and
the mean of `g` already behaves like an effective-number measure (`1 / (1 - diversity)` is roughly the number of
equally common distinct poses).

#### Calibration

Synthetic stick figures (`tests/posegen.py`) at random scale, position and mirroring, with every limb angle jittered
uniformly by the given amount and every joint moved by Gaussian noise of the given standard deviation in torso
units. "Body proportions" varies each person's upper arm, forearm, thigh and shin by up to 8 %; "one hip lost" sets
one hip's score to 0 in half the frames (a waist-up crop losing a hip at the frame edge). Default parameters; every
set's comparable share is 1.00.

| set | N | diversity | mean distance | clusters | largest share |
|---|---|---|---|---|---|
| one jump shot, tight (3 deg, 0.02) | 50 | 0.032 | 0.052 | 1 | 1.00 |
| one jump shot, tight (3 deg, 0.02) | 500 | 0.032 | 0.052 | 1 | 1.00 |
| one jump shot (5 deg, 0.03) | 50 | 0.072 | 0.079 | 1 | 1.00 |
| one jump shot (5 deg, 0.03) | 500 | 0.069 | 0.078 | 1 | 1.00 |
| one tuck jump (5 deg, 0.03) | 50 | 0.051 | 0.068 | 1 | 1.00 |
| one tuck jump (5 deg, 0.03) | 500 | 0.053 | 0.068 | 1 | 1.00 |
| one jump shot, loose (8 deg, 0.05) | 50 | 0.160 | 0.123 | 1 | 1.00 |
| one jump shot, loose (8 deg, 0.05) | 500 | 0.175 | 0.129 | 1 | 1.00 |
| one jump shot, loose, body proportions | 200 | 0.185 | 0.133 | 1 | 1.00 |
| one standing pose, loose, body proportions | 200 | 0.166 | 0.126 | 1 | 1.00 |
| one jump shot, loose, one hip lost | 200 | 0.190 | 0.135 | 1 | 1.00 |
| one jump shot, loose, body proportions and one hip lost | 200 | 0.200 | 0.139 | 1 | 1.00 |
| 90% one jump shot, rest broad mix | 200 | 0.223 | 0.196 | 10 | 0.91 |
| 85% one jump shot, rest broad mix | 200 | 0.299 | 0.259 | 14 | 0.85 |
| 80% one jump shot, rest broad mix | 200 | 0.378 | 0.352 | 15 | 0.80 |
| 70% one jump shot, rest broad mix | 200 | 0.486 | 0.408 | 15 | 0.71 |
| two poses, 50/50 (standing / T-pose) | 200 | 0.398 | 0.211 | 2 | 0.58 |
| one standing pose, half frontal, half profile | 200 | 0.401 | 0.208 | 2 | 0.50 |
| two poses, 50/50 (jump shot / seated) | 50 | 0.544 | 0.520 | 2 | 0.54 |
| two poses, 50/50 (jump shot / seated) | 500 | 0.533 | 0.515 | 2 | 0.51 |
| standing, five arm positions | 50 | 0.565 | 0.288 | 4 | 0.42 |
| standing, five arm positions | 500 | 0.550 | 0.280 | 4 | 0.42 |
| 50% one jump shot, rest broad mix | 200 | 0.701 | 0.586 | 15 | 0.52 |
| broad mix, 16 presets (10 deg, 0.03) | 50 | 0.848 | 0.620 | 15 | 0.22 |
| broad mix, 16 presets (10 deg, 0.03) | 500 | 0.879 | 0.705 | 15 | 0.11 |

Before the anchor fix of [pose-vector.md](pose-vector.md#normalisation), losing one hip in half the frames lifted a
loose single pose from 0.18 to 0.35, over the gate; it now adds 0.01 to 0.04.

The scale 0.3 was chosen so that "the same jump shot with small jitter" lands at 0.05-0.2 and a broad mix of stances
and arm positions at 0.8 or more, with room between them for a gate. Scales of 0.25 / 0.3 / 0.35 give 0.094 / 0.066
/ 0.049 for 80 jump shots, 0.23 / 0.17 / 0.13 for 200 loose jump shots, and 0.91 / 0.88 / 0.85 for 120 broad-mix
poses: 0.25 pushes a single loose pose toward the gate, 0.35 compresses the low end. The non-squared `1 - exp(-d/s)`
would score the tight jump shot 0.22, with a steep slope at `d = 0` that turns estimator jitter into diversity. The
synthetic figures share one skeleton up to the 8 % proportion jitter; real people and real estimator noise add a
little distance to every pair, so expect real single-pose sets to score somewhat higher than the table.

**What counts as variety.** The vector lives in the image plane, so one stance shot frontal and in profile is two
poses to it (0.40 for a half/half set, like two different arm positions). That is composition variety, which is what
"not every output the same jump shot" asks about; it is not articulation variety, and a set that only varies the
camera angle will pass a diversity gate.

#### Suggested gate for proof

Read the score only when it measures enough of the set, then gate on two numbers:

1. **Measurable.** `usable >= 10`, `usable / count >= 0.5` and `comparable_share >= 0.8`. Otherwise the run is
   "not measurable", never a pass: 98 unusable images and two different poses score 0.90, and four poses of which
   one pair is comparable score whatever that one pair says.
2. **Reject `pose_collapse`** when `diversity < 0.3` **or** `largest_cluster_share >= 0.75`.

On the calibration sets every single-pose set, however loose (up to 0.20), fails both. Diversity alone rejects a
set about 87 % one pose or more (90 %: 0.22; 85 %: 0.30, on the line); the cluster share rejects 75 % or more,
because jitter, body proportions and lost joints inflate diversity but leave a collapsed set in one cluster (the 0.25
cluster cut is twice the jitter distances above). It passes a 70/30 set (0.49, 0.71), both 50/50 two-pose sets (0.40
and 0.53, largest 0.58) and anything broader. The gap between the loosest collapsed set (0.20) and 0.3 is narrower
than it looks once real noise is added, which is why the cluster-share test is there.

Proof should confirm both thresholds on one real run, as it did for `--min-likeness`, and report
`largest_cluster_share`, the largest cluster's medoid, `usable / count` and `comparable_share` beside the score, so a
rejection says *which* pose the outputs collapsed onto.

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
| standing | every visible leg: thigh within 35 degrees of down, visible thigh 0.55 or longer, knee 150 or straighter, visible shin 0.5 or longer (a short straight shin is a frontal view of someone kneeling upright) |
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

- **Profile** when exactly one shoulder is confident and the face is not seen from the front, or both are and their
  separation across the body is under 0.35 of the torso scale (frontal adults show about 0.7, a three-quarter view
  about 0.5). The face is seen from the front when both eyes are confident and the nose, if confident, lies between
  them across the body: then one shoulder is a frontal shot cropped at one side, and facing is `unknown` (with one
  shoulder the shoulder order cannot be checked). In a profile the nose decides: 0.05
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
  **Kneeling** reads as unknown or crouching (a frontal view of upright kneeling, with its short shin, is unknown).
  Neither is a label.
- **Sitting on a low stool** seen from the front, knees a little above the hips, reads as crouching.
- **Profile arms are measured in the image plane:** an arm reaching toward the camera foreshortens and can read as
  down.
- **One visible arm or leg decides alone** (the occluded one is assumed to agree), which can mislabel `mixed` arms
  in profile.
- **Facing trusts the estimator's left/right assignment.** Pose estimators sometimes swap sides on back views; the
  face-visibility cross-check turns most such swaps into `unknown` rather than a wrong label. A back view with
  swapped sides **and** face landmarks the estimator hallucinated at 0.3 or more passes both checks and reads
  `front`.
- **A frontal shot cropped through the face** (one shoulder, that side's ear and eye out of frame) looks like a
  profile to the rules and reads `left` or `right`; the frontal-face check needs both eyes.
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
nearest usable reference pose under the mirror-invariant distance. It then asks the same question inside the
reference set, so the answer can be read against what "no copying" looks like for that training set.

| field | definition |
|---|---|
| `matches` | one per output image, sorted by image name: `nearest`, `distance`, `shared_joints`, `close`. An output that is unusable, or shares too few joints with every usable reference pose, has `nearest`, `distance` and `shared_joints` null and `close` false. Ties for nearest go to the alphabetically first reference image |
| `matches[].shared_joints` | of the 12 non-head joints (shoulders, elbows, wrists, hips, knees, ankles), how many are confident in both poses, in the orientation compared |
| `matches[].close` | some reference pose sharing at least `min_close_joints` (12) of those joints is closer than `threshold`. The match then names the nearest such pose; otherwise it names the nearest comparable pose of any coverage, which is reported but never close. A waist-up output is never close to a full-body frame, however well the upper bodies agree |
| `output_count` | every output image, usable or not (one `matches` entry each) |
| `matched_count` | outputs with a nearest reference pose |
| `reference_count` | usable reference poses |
| `close_eligible` | outputs that share all 12 joints with at least one reference pose, so could be close |
| `close_share` | close outputs / `close_eligible`; 0 when none is eligible |
| `baseline_eligible`, `baseline_close_share` | the same statistic inside the reference set, leave-one-out: each reference pose against its nearest *other* reference pose, same rules. Null with fewer than two usable reference poses or none eligible |
| `close_excess` | `max(0, close_share - baseline_close_share)`; null when the baseline is |
| `output_nn_distance`, `reference_nn_distance` | quantiles (`n`, `p01`, `p10`, `p50`, `p90`) of the distance from each eligible output to its nearest whole-skeleton reference pose, and from each eligible reference pose to its nearest other one. Outputs whose quantiles sit well below the reference's are closer to the training frames than the frames are to each other |
| `output_coverage`, `reference_coverage` | label histograms of the usable poses of each set |
| `coverage_divergence` | Jensen-Shannon divergence (base 2) between the two coverages, averaged over stance, arms and facing; 0 = the same mix of poses, 1 = disjoint |

`unknown` counts as a label in the divergence, so a rise in unreadable bodies among the outputs registers as drift.
If one set has no usable poses and the other does, each histogram's divergence is 1; if neither does, 0.

### What a close pose match means

**A close pose match means "the same pose", not "a copied frame".** A 2D pose cannot tell a copy from another person
striking the same pose, and the more training frames there are, the likelier any common pose has a near twin among
them. So:

- a per-output `close` flag is a **pointer**: it names the training frame an image-level check (argus-proof's leak
  and identity checks) should compare the output against. On its own it is no evidence of copying;
- the **run-level signal** is `close_share` well above `baseline_close_share`: outputs sitting closer to training
  frames than the training frames sit to each other. `close_excess` is that difference, and the two
  `*_nn_distance` quantiles show it as a distribution.

### The close threshold

`threshold` defaults to **0.04** torso units. Re-estimating one pose with per-joint noise of 0.02 torso units (a
few pixels on a typical subject) gives distances with a median of 0.031 and 84 % under 0.04 on the synthetic
figures; at 0.01 noise every one is under 0.04 (99th percentile 0.030). Independent draws of one stance with 8 degrees
of limb jitter sit 0.05 to 0.06 from their nearest neighbour among 10 to 200 frames, so only 0 to 6 % of them come
under 0.04, and the baseline absorbs those.

The threshold was 0.08 before the baseline existed. At 0.08 independent draws of one common pose were flagged at
58 % against one reference frame and 100 % against 50, a broad mix against a broad mix at 87 %, and the baseline
saturated along with the outputs: even outputs that were all copies showed an excess of 0.02 in a same-stance set.
Measured on the synthetic figures (default parameters; copies are reference frames with 0.02 torso units of joint
noise at another scale and position):

| case | outputs vs reference | close_share | baseline | close_excess | output p50 | reference p50 |
|---|---|---|---|---|---|---|
| same stance (standing), independent draws, 8 deg / 0.03 | 200 vs 10 | 0.00 | 0.00 | 0.00 | 0.060 | 0.060 |
| same stance (standing), independent draws, 8 deg / 0.03 | 200 vs 200 | 0.03 | 0.03 | 0.00 | 0.050 | 0.049 |
| same stance (hands on hips), independent draws, 8 deg / 0.03 | 200 vs 200 | 0.06 | 0.00 | 0.06 | 0.051 | 0.050 |
| same stance (standing), independent draws, 3 deg / 0.02 | 200 vs 50 | 0.99 | 0.96 | 0.03 | | |
| broad mix vs broad mix, independent | 300 vs 200 | 0.00 | 0.00 | 0.00 | 0.067 | 0.066 |
| broad mix, 30 % of outputs copies | 300 vs 200 | 0.26 | 0.00 | 0.26 | 0.061 | 0.067 |
| broad mix, every output a copy | 300 vs 200 | 0.84 | 0.00 | 0.84 | 0.030 | 0.067 |
| same stance (standing), 30 % copies | 200 vs 50 | 0.24 | 0.00 | 0.24 | 0.050 | 0.054 |
| same stance (standing), every output a copy | 200 vs 50 | 0.84 | 0.00 | 0.84 | 0.031 | 0.055 |

The 3-degree row is the case the baseline exists for: jitter that small makes every independent draw look like a
copy (close share 0.99), and so does the reference set against itself (0.96), so the excess stays near zero. Without
copying the excess stayed at 0.06 or less in every run; 30 % copies gave 0.24 to 0.26. A copy only shows when the
output is re-estimated within the threshold: a copied frame regenerated with larger changes reads as the same pose
and is caught, if at all, by the image-level checks.

**Suggested use in proof** (not a reject on its own): with `close_eligible >= 20` and `baseline_eligible >= 20`, flag
the run for review (`pose_copying`) when `close_excess >= 0.1`, report both `*_nn_distance` quantiles beside it, and
send every close match's `nearest` frame to the leak and identity checks. A pose-conditioned generator, or one
driven from a pose library, reproduces reference poses by design, so expect a high excess there and do not gate it.

## How argus-proof should use this

From smk762/argus-proof#122:

1. **Pose collapse gate.** Run `pose_set_stats` on the outputs (or `load_sidecars` on a folder `argus-puppet pose`
   has processed). If `usable < 10`, `usable / count < 0.5` or `comparable_share < 0.8` the run is "not
   measurable", not a pass. Otherwise reject with `pose_collapse` when `diversity < --min-pose-diversity`
   (suggested 0.3) or `largest_cluster_share >= 0.75` (see [the suggested gate](#suggested-gate-for-proof)). Report
   `largest_cluster_share`, the largest cluster's `medoid`, `usable / count` and `comparable_share`.
2. **Training-distribution context.** Run `compare_pose_sets(outputs, training)`. A collapsed run with low
   `coverage_divergence` collapsed onto a pose that dominates the training set: a dataset-balance problem for
   curator's coverage panel. High divergence means the model drifted away from the training mix.
3. **Pose copying evidence.** `close_excess >= 0.1` (with at least 20 eligible poses on each side) says the
   outputs sit closer to the training frames than the frames sit to each other: flag the run for review. The close
   matches (`matches[].close`, `nearest`) name the training frame each output should be compared against at image
   level; feed them to the leak and identity checks. Do not gate on pose alone (see
   [what a close pose match means](#what-a-close-pose-match-means)).

## Cost

On a desktop CPU with synthetic poses, `pose_set_stats` takes about 0.4 s for 1000 images and 3.9 s for 5000 (of
which embedding and labelling are about 1.3 s, the distance matrix 1.4 s, clustering 0.4 s), with a peak of about
230 MB above the input at 5000 (two N x N float32 matrices). `compare_pose_sets` on 2500 outputs against 2500
references takes about 3.2 s and about 15 MB of working memory: it streams the nearest-neighbour search in row chunks
and never holds the M x N matrix, and about a third of the time is the reference set's leave-one-out baseline.
