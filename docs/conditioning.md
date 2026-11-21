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
| Body | The 133 whole-body keypoints are converted to OpenPose's 18 joints; the neck, which the whole-body layout lacks, is the midpoint of the shoulders, scored as the lower of the two. Each of the 17 limbs whose two joints are both confident is a filled ellipse along the limb: semi-major half the limb length, semi-minor 4 px, in that limb's colour. When every limb is drawn, the limb layer is dimmed to 60 %. Each confident joint is then a radius-4 dot in its own colour at full intensity. The 18 colours run around the hue wheel from red (neck to right shoulder; the nose) to pink. |
| Hands | Per hand, 20 edges from the wrist out along each finger, each a 2 px line in its own hue (edge *i* of 20 is `hsv(i/20, 1, 1)`), then a radius-4 blue dot on each confident point. |
| Face | Each confident face landmark (all 68) is a white radius-3 dot. |
| Feet | Not drawn (the convention has no feet). |

Draw order is body limbs, the dimming, body joints, then each person's hands and face; later shapes overwrite
earlier ones, nothing is blended. Shapes have hard (aliased) edges as the annotators draw them, so every pixel is
black or one of the convention's colours. A point is drawn only when its score is at least `min_score` (default
0.3) and its coordinates are finite; anything off the canvas is clipped.

### Options

`RenderOptions` (also the `options` of `POST /render`, and the flags of `argus-puppet render`):

| Field | Default | Effect |
|---|---|---|
| `body` | true | Draw the 18-joint body. |
| `hands` | true | Draw both hands. |
| `face` | true | Draw the face landmarks. |
| `width`, `height` | source size | Canvas size. Give one alone to keep the source aspect ratio. |
| `min_score` | 0.3 | Points below this score are not drawn; a limb needs both its joints. |
| `fit` | `contain` | How the source frame maps onto a canvas of another shape (see below). |
| `primary_only` | false | Draw only the primary subject instead of every person. |
| `stroke_scale` | 1.0 | Multiplies every stroke width and radius; `"auto"` is `min(width, height) / 512` (see below). |
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

**Stroke widths are fixed pixel sizes** (4 px limb half-width, radius-4 joints, 2 px hand lines, radius-3 face
dots) whatever the canvas size, as the annotators draw at their working resolution. The annotators usually draw
at a ~512 px detection resolution and the result is resized up to the generation size, so their strokes look
proportionally thicker on large canvases. To mimic that, pass `stroke_scale="auto"` (`min(width, height) / 512`)
or any positive factor. Render at the generation resolution (e.g. 1024 x 1024 for SDXL) rather than resizing a
small control image afterwards, which would blur the hard edges.

```python
from argus_puppet.models import PoseResult, RenderOptions
from argus_puppet.render import render_png

result = PoseResult.model_validate_json(open("photo.pose.json").read())
png = render_png(result, RenderOptions(width=1024, height=1024, face=False, stroke_scale="auto"))
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
   least `min_body_score` (default 0.3); images with nobody usable are skipped. Each entry is labelled (stance,
   arms, facing; `labels.pose_labels`). Entry ids are derived from the source image's hash and the keypoints, so
   rebuilding from the same inputs gives the same ids.
4. **Age check** (optional, `age_gate`; on with real-person sources): see below.
5. **Deduplicate** (`dedupe_threshold`, default 0.08; `None` turns it off): a pose closer than the threshold to a
   better-scoring pose already kept is dropped, so bursts of near-identical frames collapse to their best one.
6. **Cluster** (default: average linkage at the metrics' 0.25 cut; `clusterer=None` turns it off): each entry gets
   a cluster id, used to stratify sampling.
7. **Cap** (optional, `max_entries`): keep a diverse subset of that size (see below).

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
image count as the same pose. A pose the distance cannot use (too few confident joints) is never a duplicate and
sits in a cluster of its own. Each step takes a replacement callable (`distance_matrix`, `labeler`, `clusterer`).

**Choosing 0.08.** The library's "same pose" is the comparison's "memorised" (see
[metrics.md](metrics.md#the-memorisation-threshold)), so the suite has one definition. Re-estimating one pose sits
at a median 0.04 and a 99th percentile 0.09 on the synthetic figures, and the two closest *distinct* presets
(standing and hands on hips) are 0.12 apart. 0.08 removes nearly every re-estimated copy (a burst of frames, one
photo at two sizes) and stays a third below the closest distinct pair, so no real pose variety is lost; the rare
copy that survives costs one extra entry.

### The age check

The suite keeps any frame where a face **or a body** reads under 18 out of a pose library, background people
included. With real photos, build with the age check on: `argus-puppet library build poses/ --out lib.json
--age-check`, or `build_library(..., age_gate=AgeGate(HttpAgeEstimator(url), load_image))` from Python.

- Each candidate image (one that would otherwise become an entry) is posted to argus-curator's
  `POST /faces?body=true` at `ARGUS_PUPPET_AGE_URL`: InsightFace faces plus a body pass, each with a MiVOLO age
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
(the image is already a skeleton). argus-darkroom takes it as the pose input of a ControlNet generation. A
strength around 0.6 to 1.0 holds the pose; lower values let the prompt override it.

Set the canvas to the generation resolution and aspect ratio. A skeleton drawn for a portrait canvas and fed to
a landscape generation is stretched by the generator, which is exactly the distortion `contain` and `skeleton`
fitting avoid.

**A pose image is an input like any other.** It is derived from a source photo, and the generator's input checks
apply to that source photo, not just to the stick figure: the library keeps each pose's source image name and
hash so the generator can run its checks on the source and record the skeleton with its output.
