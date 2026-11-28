# Pose conditioning

argus-puppet draws **skeleton control images** for pose ControlNets and keeps a **pose library** to draw them
from, so a generator can produce a batch of images in varied, deliberately chosen poses.

Python API: `argus_puppet.render` (drawing), `argus_puppet.library` (the library) and `argus_puppet.sampling`
(the generic deduplication and diverse-sampling algorithms underneath).

## The control image

A control image is a black RGB picture of a stick figure. A pose ControlNet (SD 1.5 / SDXL "openpose" models,
ControlNet-union pose mode) reads the figure's colours to tell joints and limbs apart, and steers the generated
person into that pose. It carries no appearance: no face, clothing, body shape or background from the photo the
pose came from, only joint positions.

### The colour convention

Pose ControlNets were trained on figures drawn in one fixed convention: the one the OpenPose annotator introduced
and the DWPose / `controlnet_aux` annotators reproduce. A model only follows a figure drawn the same way, so
argus-puppet follows that convention as an interoperability spec. The drawing code is written from scratch; no
annotator code is used.

| Part | How it is drawn |
|---|---|
| Body | The 133 whole-body keypoints are converted to OpenPose's 18 joints; the neck, which the whole-body layout lacks, is the midpoint of the shoulders, scored as the lower of the two. Each of the 17 limbs whose two joints are both visible is a filled ellipse along the limb: semi-major half the limb length, semi-minor 4 px, in that limb's colour. When every limb is drawn, the limb layer is dimmed to 60 %. Each visible joint is then a radius-4 dot in its own colour at full intensity. The 18 colours run around the hue wheel from red (neck to right shoulder; the nose) to pink. |
| Hands | Per hand, 20 edges from the wrist out along each finger, each a 2 px line in its own hue (edge *i* of 20 is `hsv(i/20, 1, 1)`), then a radius-4 blue dot on each visible point. |
| Face | Each visible face landmark (all 68) is a white radius-3 dot. |
| Feet | Not drawn (the convention has no feet). |

Pixel sizes are at the annotator's 512 px working resolution; see [sizing](#sizing-and-aspect) for how they scale.

**Visibility** is the DWPose annotator's: a body joint is visible when its score is **strictly above** `min_score`
(default 0.3), so the synthesised neck appears only when both shoulders are above it; a hand or face point when its
score is **at or above** it. A hand or face point also needs both its coordinates, truncated to whole pixels at the
annotator's resolution, to be at least 1: one less than a pixel from the left or top edge (`k` px at stroke scale
`k`), or off the canvas there, is dropped with every hand edge touching it, as the annotator drops it. A point with a
non-finite coordinate, or one more than 10^6 px from the origin, is never drawn; anything else off the canvas is
clipped.

**Draw order** across people is also the annotator's: limb 0 of every person, then limb 1 of every person, up to
limb 16; the dimming; joint 0 of every person, up to joint 17; then every person's left hand, then every right hand
(each hand's edges, then its dots); then every face. Where two people overlap, the higher limb or joint index wins
whichever person is listed first, and a face dot lands on top of any hand. Later shapes overwrite earlier ones,
nothing is blended. Shapes have hard (aliased) edges as the annotators draw them, so every pixel is black or one of
the convention's colours.

### Options

`RenderOptions` (also the `options` of `POST /render`, and the flags of `argus-puppet render`):

| Field | Default | Effect |
|---|---|---|
| `body` | true | Draw the 18-joint body. |
| `hands` | true | Draw both hands. |
| `face` | true | Draw the face landmarks. |
| `width`, `height` | source size | Canvas size. Give one alone to keep the source aspect ratio. |
| `min_score` | 0.3 | Visibility threshold: body joints need a score above it, hand and face points at or above it; a limb needs both its joints. |
| `fit` | `contain` | How the source frame maps onto a canvas of another shape (see below). |
| `primary_only` | false | Draw only the primary subject instead of every person. |
| `stroke_scale` | `"auto"` | Multiplies every stroke width and radius; `"auto"` is `min(width, height) / 512`, a number fixes the factor (see below). |
| `margin` | 0.05 | `fit="skeleton"` only: the fraction of each side kept clear. |

The render functions also take `fit`, `margin`, `stroke_scale` and `primary_only` as keyword arguments, which
override the options when given.

Face dots are dense: some ControlNets give strong facial-expression control with them, others are confused by
them. Turn `face` off if the generated faces look distorted, and `hands` off for models trained on body-only
skeletons.

### Sizing and aspect

Keypoints are in the source image's pixel frame. The canvas is the source size unless another is requested; then
`fit` decides how the frame maps onto it:

| `fit` | Behaviour |
|---|---|
| `contain` (default) | One uniform scale so the whole source frame fits, centred, black bars on the long sides. Never distorts the skeleton. |
| `cover` | One uniform scale so the frame fills the canvas, centred; limbs near the edges may be cut off. |
| `stretch` | Separate x and y scales; distorts the skeleton when the aspect ratios differ. |
| `skeleton` | Ignore the source frame: fit the drawn points' bounding box into the canvas inside a margin (default 5 % of each side), uniformly scaled and centred. Library batches use this. |

**Stroke widths scale with the canvas.** The annotators resize the photo to a 512 px working resolution, draw 4 px
limb half-widths, radius-4 joints, 2 px hand lines and radius-3 face dots there, and resize the drawing to the output
size, so their strokes are proportional to the canvas. The default `stroke_scale="auto"` reproduces that with a
factor of `min(width, height) / 512`: twice the 512 px sizes on a 1024 x 1024 SDXL canvas, and less than them on a
canvas under 512 px, where the reference shrinks too (hand lines stay at least 1 px). A number fixes the factor;
`stroke_scale=1` draws the 512 px sizes on any canvas. Render at the generation resolution (e.g. 1024 x 1024 for
SDXL) rather than resizing a small control image afterwards, which would blur the hard edges.

### Checked against the reference, and the known deviations

The renderer was checked against the DWPose annotator by **pixel comparison**: controlnet_aux's `dwpose/util.py`
drawing functions (commit `b1aac318`, OpenCV 4.14; upstream IDEA-Research DWPose draws identically) rendered the
same canvas-space keypoints as puppet in 25 cases (two real photos at 256 to 2048 px, SD and SDXL canvases, each
part alone, two overlapping people, scores straddling the threshold, points on the canvas edge). The annotator was
run as its controlnet_aux pipeline (draw at short side 512, both sides rounded to 64, then a bilinear resize),
compared with puppet's `stroke_scale="auto"`, and drawn straight onto the canvas, compared with `stroke_scale=1`.
The tests pin each rule, with a handful of integer facts (pixel counts, boxes) taken from the annotator itself.

What matches exactly: the palette, the limb, joint and hand-edge assignment, the 60 % dimming, visibility at the
default threshold, draw order, face dots, and the **limbs**: centre, half-length and angle truncated to integers
on the annotator's grid as it does, then `cv2.ellipse2Poly`'s 1-degree polygon filled as `cv2.fillConvexPoly` fills
it (outline included). With body and face only, puppet's image is pixel-identical to the annotator's whenever the
annotator draws at the output size: a stroke scale of 1, or `"auto"` on a canvas whose short side is 512 (e.g.
512 x 768). A hand or face point is drawn only when its coordinates, truncated to integers on the annotator's grid,
are both at least 1 (the annotator's `eps` test compares that integer, so the band is one grid pixel, `k` px at
stroke scale `k`, not a share of the image), and a hand edge only when both its ends are; body joints have no such
test.

Share of pixels differing by more than 16/255 after these fixes (before them: 0.2 to 0.9 %):

| Case | Differing > 16 | Mask IoU |
|---|---|---|
| Drawn at the output size, body and face only | 0.000 % | 1.000 |
| Drawn at the output size, with hands | 0.004 to 0.05 % | 0.95 to 1.00 |
| `"auto"`, canvas upscaled from 512 (707 to 1536 px short side) | 0.58 to 0.82 % | 0.85 to 0.87 |
| `"auto"`, canvas downscaled from 512 (177 to 480 px short side) | 0.45 to 0.71 % | 0.82 to 0.88 |

Deviations that remain:

- **Hand lines (and their caps).** The annotator draws its 2 px hand lines with OpenCV (integer endpoints, round
  caps); puppet draws them with Pillow (float endpoints, square caps). This is all that still differs when drawing
  at the output size: from a few pixels per image at 512 px to about a thousand on a 2048 px canvas, along the
  finger lines, nearly all where the reference draws and puppet does not.
- **Resizing.** The annotator draws at 512 px and resizes with bilinear interpolation, which softens edges, adds
  in-between shades and snaps every shape to its 512 grid; puppet draws at the output size with hard edges, so the
  output holds only convention colours. On upscaled canvases nearly every remaining differing pixel is one of those
  blended rim shades (a 1 px rim round each shape, which puppet leaves black or full colour); on canvases under
  512 px the annotator's downscale also aliases, breaking thin lines and smudging dots. Puppet emulates the 512 grid
  for limb geometry and the near-edge band (limbs are the exact ellipse the polygon fill covers, grown half a grid
  pixel on both axes and scaled), not the resampling.
- **Anisotropic working size.** The annotator rounds both sides of its working size to multiples of 64, so for an
  aspect ratio that isn't a multiple of 64 its strokes come out a few per cent wider along one axis than the other
  (1280 x 884: 3.6 %); puppet scales strokes uniformly by `min(width, height) / 512`.
- **Canvas borders.** OpenCV clips a polygon's outline to the canvas before walking it, which can move a pixel on
  the border row or column where a limb leaves the canvas.

```python
from argus_puppet.models import PoseResult, RenderOptions
from argus_puppet.render import render_png

result = PoseResult.model_validate_json(open("photo.pose.json").read())
png = render_png(result, RenderOptions(width=1024, height=1024, face=False))   # strokes scale with the canvas
```

## The pose library

A pose library (`PoseLibrary`, saved as JSON) is a set of poses to sample from: one entry per source image,
holding the primary person's 133 keypoints, the source frame size, optional labels (stance, arms, facing) and a
cluster id, and its provenance: the source image's name and SHA-256.

### Building one

1. **Collect source images you are entitled to use**: public-domain or CC0 photos, or synthetic images (renders
   of posed 3-D figures work well and involve no real person). The library stores only keypoints, but it records
   which image each pose came from, and that image stays the pose's origin.
2. **Extract poses** with the estimator, producing a `.pose.json` sidecar per image.
3. **Build** from the sidecars. Each image contributes its primary person if their mean body-joint score is at
   least `min_body_score` (default 0.3) and the pose vector can embed them under the default metric params (a
   confident hip and shoulder to anchor on and 8 confident body joints; see
   [pose-vector.md](pose-vector.md#unusable-poses)). A pose the vector cannot embed could never be compared,
   deduplicated or clustered, so it would sit in a cluster of its own and stratified sampling would favour it; such
   images are counted as `unembeddable` (split by [reason](metrics.md#why-a-pose-is-unusable) in
   `unembeddable_reasons`) and skipped, as are images with nobody usable. Each entry is labelled
   (stance, arms, facing; `labels.pose_labels`). Entry ids are derived from the source image's hash and the
   keypoints, so rebuilding from the same inputs gives the same ids.
4. **Age check** (optional, `age_gate`; on with real-person sources): see below.
5. **Deduplicate** (`dedupe_threshold`, default 0.04; `None` turns it off): a pose closer than the threshold to a
   better-scoring pose already kept is dropped, so bursts of near-identical frames collapse to their best one. Two
   poses can only be duplicates when they share all 12 non-head joints (`MetricParams.min_close_joints`): a waist-up
   frame is never a duplicate of a full-body one, and a pose can only absorb others it was compared with over the
   whole limb skeleton.
6. **Cluster** (default: average linkage at the metrics' 0.25 cut; `clusterer=None` turns it off): each entry gets
   a cluster id, used to stratify sampling.
7. **Cap** (optional, `max_entries`): keep a diverse subset of that size (see below).

Steps 5 to 7 see the entries sorted by source image and id, so the library does not depend on the order the sidecars
were read in; the entries keep their input order.

```python
from argus_puppet.library import build_library, save_library
from argus_puppet.sidecars import load_sidecars

build = build_library(load_sidecars("poses/"), created_from="CC0 photos of adults, folder poses/", max_entries=500)
save_library(build.library, "poses.library.json")   # atomic: temp file, then rename
```

The CLI does the same: `argus-puppet library build poses/ --out poses.library.json` (add `--extract` to pose images
that have no sidecar yet). `build` also reports how many images were skipped and why. The library file is written
atomically with sorted keys, so it diffs cleanly and a crash never leaves a half-written file.

The pose distance used for deduplication, clustering and sampling is the suite's mirror-invariant pose distance
(`metrics.keypoint_distance_matrix`, in torso units; see [pose-vector.md](pose-vector.md)), so a pose and its mirror
image count as the same pose. Each step takes a replacement callable (`distance_matrix`, `labeler`, `clusterer`);
the whole-skeleton rule of the dedupe applies to a replacement distance too.

**Choosing 0.04.** The library's "same pose" is the comparison's close pose match (see
[metrics.md](metrics.md#the-close-threshold)), so the suite has one definition. On 60 synthetic poses plus two
re-estimated copies (another scale and position, per-joint noise in torso units) of 20 of them:

| threshold | copies left (0.01 noise) | copies left (0.02 noise) | broad mix lost (10 deg) | one stance lost (8 deg) | one stance lost (5 deg) |
|---|---|---|---|---|---|
| 0.03 | 2-4 of 40 | 21-24 of 40 | 0 of 60 | 0 of 60 | 3-4 of 60 |
| **0.04** | **0 of 40** | **9-11 of 40** | **0 of 60** | **1 of 60** | **25-32 of 60** |
| 0.05 | 0 of 40 | 3-6 of 40 | 0 of 60 | 5 of 60 | 42-51 of 60 |
| 0.08 (before) | 0 of 40 | 0 of 40 | 17-18 of 60 | 52-53 of 60 | 57-58 of 60 |

0.04 removes every copy re-estimated with a pixel or two of noise and three in four of the noisier ones, and keeps
distinct people in one stance once their limbs differ by about 8 degrees. The trade-off: people in one stance whose
limbs differ by 5 degrees or less are indistinguishable from re-estimated copies of each other in 2D, and the
dedupe merges about half of them. A copy that survives only costs one extra entry, which the cluster stratification
of sampling then down-weights, so the threshold errs on the side of keeping poses.

### The age check

The suite keeps any frame where a face **or a body** reads under 18 out of a pose library, background people
included. With real photos, build with the age check on: `argus-puppet library build poses/ --out lib.json
--age-check`, or `build_library(..., age_gate=AgeGate(HttpAgeEstimator(url), load_image))` from Python.

- Each candidate image (one that would otherwise become an entry) is posted to argus-curator's
  `POST /faces?body=true` at `ARGUS_PUPPET_AGE_URL` (else the suite-wide `ARGUS_CURATOR_URL`): InsightFace faces plus a body pass, each with a MiVOLO age
  estimate. The URL only ever comes from the settings, the call has a timeout and an overall deadline, and redirects
  are not followed. Nothing is sent unless the check is asked for.
- The image is kept only when **every** face and body reads 18 or over.
- It **fails closed**: an image is also excluded when the service is down or answers an error, when the answer
  cannot vouch for the image (a face or unmatched body without an estimate, a body pass that did not report `ok`),
  when nobody at all could be estimated, or when the image beside the sidecar cannot be read or no longer matches the
  SHA-256 its pose was extracted from.
- The build reports `age_passed`, `under_age` and `age_unchecked`, and each excluded image with the reason. The CLI
  exits 1 and writes no library when the check could not check a single image.
- The library records `age_checked: true`. A library built without the check says `false`, and its source images
  should be treated as unchecked.

### Diverse sampling

`sample(library, n)` picks *n* poses by **farthest-point sampling**: start from the most central pose, then
repeatedly take the pose whose nearest already-picked pose is farthest away. The batch spreads across the whole
range of poses instead of clustering around the most common one (people mostly stand facing the camera, so a
random draw is mostly standing poses).

- **Stratified by cluster**: when entries carry cluster ids, every cluster gets one pose before any gets a
  second, so rare pose types are represented in small batches.
- **Filters**: restrict to a stance, arms or facing label first.
- **Reproducible**: unseeded, the result depends only on the library. With a seed the first pick is random, so
  different seeds give different, still diverse, batches.
- Asking for at least as many poses as the library holds returns all of them, in diversity order. Poses the
  distance cannot compare (too few confident joints) are picked only after every comparable pose.

`render_entries(entries, (width, height))` renders a sampled batch at the generation size, re-fitting each
skeleton into the canvas with a margin and keeping its proportions, so poses from differently sized and framed
sources come out at a consistent size.

## Using the image in a generator

The control image goes to a ControlNet node next to the prompt: in ComfyUI, load the PNG (or receive it from the
render endpoint) and connect it to *Apply ControlNet* with an OpenPose / pose ControlNet model, no preprocessor
(the image is already a skeleton). Any ControlNet pipeline that accepts an OpenPose control image can take it
as the pose input of a generation. A strength around 0.6 to 1.0 holds the pose; lower values let the prompt override it.

Set the canvas to the generation resolution and aspect ratio. A skeleton drawn for a portrait canvas and fed to
a landscape generation is stretched by the generator, which is exactly the distortion `contain` and `skeleton`
fitting avoid.

**A pose image is an input like any other.** It is derived from a source photo, and the generator's input checks
apply to that source photo, not just to the stick figure: the library keeps each pose's source image name and
hash so the generator can run its checks on the source and record the skeleton with its output.

### argus-darkroom

argus-darkroom, the suite's guarded generator, takes a pose this way. A generation request carries the **source
photo** (`pose.data_base64`), never a control image or keypoints, so every skeleton it renders from comes from a
picture it checked:

1. The photo gets darkroom's full input scan (SingGuard, NudeNet, the age estimate) and goes on only when it passes.
2. Darkroom posts the checked photo to `POST /pose` (which writes nothing) and reads every person's keypoints.
3. It posts those keypoints to `POST /render` as a `BarePose` (`{keypoints: [person, ...], width, height}`) with
   `options` `{body, hands, face, width, height, fit}`: the canvas is the generation size, and `fit` is `contain`,
   `cover` or `skeleton`. For the photo's primary subject alone it sends that one person's keypoints.
4. It checks the PNG that comes back (the generation's size, at most 256 colours, something drawn) and feeds it to
   the ControlNet. Its record keeps the photo's hashes, the keypoints, and the estimator's version and model hashes
   from the `PoseResult`.

To generate from a library pose, send darkroom that entry's source image (`source_image`, verified by
`source_sha256`); `sample` picks which entries. Point darkroom at puppet with `ARGUS_DARKROOM_PUPPET_URL`, or the
suite-wide `ARGUS_PUPPET_URL`.
