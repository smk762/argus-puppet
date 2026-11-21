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

`RenderOptions` (also the body of the render endpoint):

| Field | Default | Effect |
|---|---|---|
| `body` | true | Draw the 18-joint body. |
| `hands` | true | Draw both hands. |
| `face` | true | Draw the face landmarks. |
| `width`, `height` | source size | Canvas size. Give one alone to keep the source aspect ratio. |
| `min_score` | 0.3 | Points below this score are not drawn; a limb needs both its joints. |

Every person in the image is drawn by default; `primary_only` draws just the primary subject.

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
png = render_png(result, RenderOptions(width=1024, height=1024, face=False), stroke_scale="auto")
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
   least `min_body_score` (default 0.3); images with nobody usable are skipped. Entry ids are derived from the
   source image's hash and the keypoints, so rebuilding from the same inputs gives the same ids.
4. **Deduplicate** (optional, `dedupe_threshold`): a pose closer than the threshold to a better-scoring pose
   already kept is dropped, so bursts of near-identical frames collapse to their best one.
5. **Cluster** (optional): a clusterer assigns each entry a cluster id, used to stratify sampling.
6. **Cap** (optional, `max_entries`): keep a diverse subset of that size (see below).

```python
from argus_puppet.library import build_library, save_library

build = build_library(results, created_from="CC0 photos, folder poses/", dedupe_threshold=0.1, max_entries=500)
save_library(build.library, "poses.library.json")   # atomic: temp file, then rename
```

`build` also reports how many images were skipped and why. The library file is written atomically with sorted
keys, so it diffs cleanly and a crash never leaves a half-written file.

The pose distance used for deduplication, clustering and sampling is the suite's mirror-invariant pose distance,
so a pose and its mirror image count as the same pose. When it is not supplied, a simple stand-in is used:
hip-centred, torso-scaled mean body-joint distance (also mirror-invariant). Its units are torso lengths, which is
what the 0.1 threshold above suits; the real metric has its own scale.

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
