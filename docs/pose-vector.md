# The pose vector and pose distance

`argus_puppet.embedding` turns one person's keypoints into a normalised pose vector and compares two vectors with a
mirror-invariant distance. Everything downstream (set diversity, clustering, the 2D scatter, the close-match check)
is built on this one distance, so this page states it exactly.

## Input

One person's 133 COCO-WholeBody keypoints `(x, y, score)` in image pixels, y pointing down (see the wire contract in
`models.py`). A joint is **confident** when its score is at least `min_joint_score` (default 0.3) and its
coordinates are finite and within 10^6 px of the origin (a coordinate further out is treated as unplaced, as the
renderer does; it would overflow the float32 distance kernel).

## Which joints

The vector uses the **17 COCO body joints** only: nose, eyes, ears, shoulders, elbows, wrists, hips, knees and
ankles.

- **Feet stay out.** The six foot points sit within a few centimetres of the ankle, so they add almost no lever arm
  to the leg's direction, and they are the least reliable points on generated images (shoes, cropped frames, motion
  blur). The ankles already say where the feet are.
- **Hands and face stay out.** Finger and face landmarks describe the hand and expression, not the body pose; 110
  extra points would swamp 17 body joints in any average.

## Normalisation

Let `p_j` be the pixel position of joint `j`.

- **Centre:** the mid-hip `h = (p_lhip + p_rhip) / 2`.
- **Mid-shoulder:** `s = (p_lsh + p_rsh) / 2`.

The four **anchor joints** (hips and shoulders) that place the centre and the scale are read more leniently than
the compared joints, because dropping one moves everything else. A frontal pose whose one hip dips from 0.30 to
0.29 would otherwise have its centre jump to the other hip, half a hip width, which moves the whole vector by 0.23
torso units; an anchor shoulder setting the scale of a foreshortened torso moved it by 0.18. So, per anchor joint:

1. confident: used as it is;
2. below `min_joint_score` but scoring at least 0.1 (`ANCHOR_MIN_SCORE`): its coordinate is still used for the
   centre and scale (the estimator still placed it), while the joint stays **masked** as a compared joint. This only
   completes a pair whose other joint is confident: a doubtful joint never makes a pose usable on its own;
3. missing altogether (score under 0.1, or unplaced) while the other pair is whole: estimated from it. A missing hip
   is the present hip plus 0.7 times the shoulder vector on that side (`HIP_TO_SHOULDER_WIDTH`: adult hip keypoints
   sit 0.45-0.55 torso lengths apart, shoulder keypoints about 0.7), a missing shoulder the present shoulder plus
   the hip vector divided by 0.7;
4. otherwise (one hip and one shoulder, nothing to estimate from): the one present joint of the pair stands in for
   the midpoint. That is right in profile, where the pair nearly coincides, and that is where it happens.

On the synthetic figures one anchor joint crossing the threshold now moves the vector by 0 (rule 2) and by at most
0.015 when it vanishes (rule 3; 95th percentile 0.05 with 10 degree limb jitter and 0.02 joint noise). The rules are
linear in the coordinates and left/right symmetric, so translation, scale and mirror invariance are exact.
- **Scale:** a robust torso length

  ```
  S = max(|s - h|, 1.2 * |p_lsh - p_rsh|, 1.5 * |p_lhip - p_rhip|)
  ```

  with each width counted when both of its anchor joints are known (rules 1-3 above). The torso foreshortens when the subject leans
  or lies toward the camera; the shoulder and hip widths do not, but they shrink in profile, where the torso does
  not. Adult shoulder keypoints sit about 0.7 torso lengths apart and hip keypoints about 0.45-0.55, so the
  multipliers keep both widths below the true torso length in an upright frontal pose (the torso itself is then the
  scale) and let them take over when the torso foreshortens. The maximum of three lower bounds is robust to either
  kind of foreshortening.

The vector is

```
v_j = (p_j - h) / S        for each confident body joint j
m_j = 1 if j is confident, else 0
```

stored as `coords` (17 x 2, torso units, image axes) and `mask` (17 booleans). A masked joint's coordinates are set
to 0 but carry no meaning: the distance never reads them. **Masking is not zero-filling**: a missing wrist does
not count as a wrist at the hip.

### Unusable poses

`embed()` returns `None` (the image is not usable) when one of these holds; `embed_or_reason()` returns the reason
instead, the first that applies, in this order (the set metrics put it on the wire, see
[metrics.md](metrics.md#why-a-pose-is-unusable)):

- no hip or no shoulder is confident (there is no anchor to centre or scale on; rules 2 and 3 above only complete a
  pair): `no_confident_torso` (neither), `no_confident_hip` or `no_confident_shoulder`;
- the scale `S` is under 4 pixels (noise, not a person): `scale_too_small`;
- fewer than `min_usable_joints` (default 8) body joints are confident: `too_few_joints`. Eight admits an upper-body
  shot with the hips in frame (head 5 + shoulders 2 + hips 2 = 9 before any limb) and refuses a head-and-shoulders
  portrait.

### Not rotation-invariant

The vector keeps the image's axes. A standing and a lying person, or an upright and an inverted one, are different
poses for diversity and close-match purposes, and stay far apart (standing vs lying is about 1.3 torso units on
the synthetic figures; the cluster threshold is 0.25).

## Mirroring

The mirror image of a vector negates x and swaps every left joint with its right counterpart, using the body part of
the skeleton's flip permutation (`skeleton.FLIP_PERMUTATION[:17]`):

```
mirror(v)_j = (-x_{f(j)}, y_{f(j)}),   mirror(m)_j = m_{f(j)}
```

where `f` swaps left and right eye, ear, shoulder, elbow, wrist, hip, knee and ankle. It is an involution, and
embedding a horizontally flipped image (`skeleton.flip_horizontal`) gives exactly `mirror` of the original vector.

## Distance

For two vectors `a`, `b` and one orientation, let `J = {j : m^a_j = m^b_j = 1}` be the joints confident in both.
With per-joint weights `w_j`:

```
D(a, b) = sum_{j in J} w_j |a_j - b_j|  /  sum_{j in J} w_j
```

the weighted mean Euclidean distance between corresponding joints, in torso units. The pose distance is the smaller
of the two orientations:

```
d(a, b) = min( D(a, b), D(a, mirror(b)) )
```

Properties (each one is a test in `tests/test_embedding.py`):

- `d(a, a) = 0` and `d(a, mirror(a)) = 0`: **mirror-invariant**.
- `d(a, b) = d(b, a)`: mirror is an isometry and the weights are left/right symmetric, so `D(b, mirror(a)) =
  D(mirror(b), a)`.
- Invariant to translation and uniform scale (the normalisation removes both).
- Grows with any joint displacement, and saturates nowhere: it is a mean of distances.
- Not a metric in the strict sense: masking means the triangle inequality can fail between poses with different
  visible joints.

### Too few shared joints

`d` is **not defined (NaN)** when, in both orientations, fewer than 6 joints are confident in both poses, or fewer
than 2 of the shared joints are limb joints (elbows, wrists, knees, ankles). Without the limb rule, two poses that
share only head and torso would look identical whatever their arms and legs were doing. Downstream, an undefined
pair is left out of means, can never join a cluster, and cannot be a nearest match.

### Weights

| joints | weight | why |
|---|---|---|
| nose, eyes, ears | 0.4 each | five points within a few centimetres; at full weight the head would count five times. 0.4 makes the head weigh two joints in all |
| shoulders, elbows, wrists, hips, knees, ankles | 1.0 | |

Limbs are **not** up-weighted. In torso units an extremity already moves several times as far as its proximal joint
for the same rotation (a 30 degree shoulder rotation moves the elbow by about 0.28 torso units and the wrist by about
0.54), so the lever arm emphasises wrists and ankles on its own. Extra weight on top would double-count them and
make the distance hostage to the two least reliable joints on a generated image. The hips are near the origin after
centring and contribute little; they stay in because they carry hip width and rotation.

## Canonical orientation

The distance handles mirroring by itself, but anything that averages vectors (a centroid, a PCA projection) needs
every pose in one orientation first.

- `canonicalize(v)` picks between `v` and `mirror(v)` the one whose **handedness** is not negative. Handedness is the
  weighted mean x of the confident joints (it changes sign under `mirror`); when that is zero the weighted mean of
  `x * y` decides. When both tie, the lexicographically smaller of the two (mask, then coordinates) is taken, so the
  choice never depends on which of the pair came in.
- `align_to_mean(set)` starts from `canonicalize` and then repeatedly flips each pose to whichever orientation is
  nearer the set's masked mean pose (up to five rounds). Near-symmetric poses can fall either side of the per-pose
  rule; aligning to the mean puts jittered copies of one pose on the same side. It depends only on the set, not its
  order.

## Matrices and cost

- `pairwise_distances(coords, mask)` gives the symmetric N x N matrix (float32 by default; N = 5000 is 100 MB). Only
  the upper triangle is computed, so the result is exactly symmetric.
- `cross_distances(a, b)` gives M x N.

Both work in row chunks whose temporary arrays stay around a few MB whatever N is, and compute in float32
(coordinates are O(1) torso units). On a desktop CPU the 5000 x 5000 matrix takes about 1.4 s.
