# `argus-puppet`

Body pose for the Argus suite: whole-body keypoints, pose metrics and ControlNet pose conditioning.

**Usage**:

```console
$ argus-puppet [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `-v, --verbose`: Log progress (to stderr)
* `--help`: Show this message and exit.

**Commands**:

* `version`: Print the package version and the...
* `schema`: Emit the wire-contract JSON Schema...
* `pose`: Write a &lt;stem&gt;.pose.json sidecar beside...
* `stats`: Pose diversity, clusters and coverage of a...
* `compare`: Each output&#x27;s nearest reference pose,...
* `render`: Draw OpenPose-convention skeleton control...
* `serve`: Run the HTTP service (needs the server...
* `models`: List, download and verify the pinned model...
* `library`: Build a pose library from a folder, and...

## `argus-puppet version`

Print the package version and the wire-contract version.

**Usage**:

```console
$ argus-puppet version [OPTIONS]
```

**Options**:

* `--help`: Show this message and exit.

## `argus-puppet schema`

Emit the wire-contract JSON Schema consumers codegen against.

**Usage**:

```console
$ argus-puppet schema [OPTIONS]
```

**Options**:

* `-o, --output <path>`: Where to write the JSON Schema  [default: schema/puppet-wire.schema.json]
* `--check`: Exit non-zero if the committed schema is stale (for CI)
* `--help`: Show this message and exit.

## `argus-puppet pose`

Write a &lt;stem&gt;.pose.json sidecar beside each image: every person&#x27;s 133 whole-body keypoints.

**Usage**:

```console
$ argus-puppet pose [OPTIONS] {target}
```

**Arguments**:

* `target <path>`: A folder of images, or one image file  [required]

**Options**:

* `-r, --recursive`: Include subfolders
* `--overwrite`: Re-pose images that already have a sidecar
* `--device <str>`: Where to run: auto, cpu, cuda (default: settings)
* `--pose-model <str>`: Pose model registry name (default: settings)
* `--detector <str>`: Person detector registry name (default: settings)
* `--json`: Print the run summary as JSON
* `--help`: Show this message and exit.

## `argus-puppet stats`

Pose diversity, clusters and coverage of a folder&#x27;s poses (from its sidecars).

**Usage**:

```console
$ argus-puppet stats [OPTIONS] {folder}
```

**Arguments**:

* `folder <path>`: A folder of images with &lt;stem&gt;.pose.json sidecars  [required]

**Options**:

* `--recursive / --no-recursive`: Include subfolders  [default: recursive]
* `--extract`: Pose the images that have no sidecar yet, first
* `--device <str>`: Where to run extraction: auto, cpu, cuda (default: settings)
* `--min-joint-score <float>`: A joint is confident at or above this score [0.3]
* `--min-usable-joints <int>`: Confident body joints a pose needs to count [8]
* `--cluster-threshold <float>`: Average-linkage cut, in torso units [0.25]
* `--min-shared-joints <int>`: Joints two poses must share to be compared [6]
* `-o, --out <path>`: Write the PoseSetStats JSON here
* `--json`: Print the PoseSetStats JSON instead of the summary
* `--min-diversity <float>`: Exit 1 when the set&#x27;s diversity is below this (for scripts)
* `--help`: Show this message and exit.

## `argus-puppet compare`

Each output&#x27;s nearest reference pose, close pose matches against the reference set&#x27;s own baseline, and how
far the pose mix drifted.

**Usage**:

```console
$ argus-puppet compare [OPTIONS] {outputs} {reference}
```

**Arguments**:

* `outputs <path>`: Folder of output images&#x27; sidecars (e.g. a LoRA&#x27;s samples)  [required]
* `reference <path>`: Folder of reference (training) images&#x27; sidecars  [required]

**Options**:

* `--threshold <float>`: Close pose match distance, torso units [0.08]
* `--recursive / --no-recursive`: Include subfolders  [default: recursive]
* `--extract`: Pose images that have no sidecar yet, first
* `--device <str>`: Where to run extraction: auto, cpu, cuda (default: settings)
* `--min-joint-score <float>`: A joint is confident at or above this score [0.3]
* `--min-usable-joints <int>`: Confident body joints a pose needs to count [8]
* `--min-shared-joints <int>`: Joints two poses must share to be compared [6]
* `--top <int>`: Closest matches to list  [default: 20]
* `-o, --out <path>`: Write the PoseSetComparison JSON here
* `--json`: Print the PoseSetComparison JSON instead of the summary
* `--help`: Show this message and exit.

## `argus-puppet render`

Draw OpenPose-convention skeleton control images (PNG) for pose ControlNets.

An image without a sidecar is posed first (in memory; --write-sidecar keeps the sidecar). Outputs are named
&lt;stem&gt;.openpose.png.

**Usage**:

```console
$ argus-puppet render [OPTIONS] {source}
```

**Arguments**:

* `source <path>`: A &lt;stem&gt;.pose.json sidecar, an image, or a folder of either  [required]

**Options**:

* `-o, --out <path>`: Output folder, or a .png file when the source is one pose  [required]
* `--body / --no-body`: Draw the body  [default: body]
* `--hands / --no-hands`: Draw the hands  [default: hands]
* `--face / --no-face`: Draw the face landmarks  [default: face]
* `--width <int range>`: Canvas width (default: the source&#x27;s)  [1&lt;=x&lt;=8192]
* `--height <int range>`: Canvas height (default: the source&#x27;s)  [1&lt;=x&lt;=8192]
* `--fit <str>`: How the source frame maps onto the canvas: contain, cover, stretch, skeleton  [default: contain]
* `--primary-only`: Draw only the primary person
* `--stroke-scale <str>`: Stroke width factor, or &#x27;auto&#x27; (min side / 512, the annotators&#x27; scaling)  [default: auto]
* `--min-score <float>`: Points scoring below this are not drawn  [default: 0.3]
* `-r, --recursive`: With a folder: include subfolders
* `--write-sidecar`: Save the sidecar of any image posed here
* `--device <str>`: Where to run extraction: auto, cpu, cuda (default: settings)
* `--help`: Show this message and exit.

## `argus-puppet serve`

Run the HTTP service (needs the server extra).

**Usage**:

```console
$ argus-puppet serve [OPTIONS]
```

**Options**:

* `--host <str>`: Interface to bind (loopback by default: there is no auth)  [default: 127.0.0.1]
* `--port <int>`: Port (default: ARGUS_PUPPET_PORT, else 8110)
* `--help`: Show this message and exit.

## `argus-puppet models`

List, download and verify the pinned model files.

**Usage**:

```console
$ argus-puppet models [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `list`: Show each pinned model: role, licence,...
* `fetch`: Download the pinned models into the model...
* `verify`: Check the model files against their pinned...

### `argus-puppet models list`

Show each pinned model: role, licence, size, and whether its file is present and verified.

**Usage**:

```console
$ argus-puppet models list [OPTIONS]
```

**Options**:

* `--verify / --no-verify`: Hash the files that are present  [default: verify]
* `--json`: Print as JSON
* `--help`: Show this message and exit.

### `argus-puppet models fetch`

Download the pinned models into the model directory, verifying each before it is written.

**Usage**:

```console
$ argus-puppet models fetch [OPTIONS] [names]...
```

**Arguments**:

* `names... <list[str]>`: Registry names (default: every pinned model)

**Options**:

* `--force`: Download even when a verified file is present
* `--help`: Show this message and exit.

### `argus-puppet models verify`

Check the model files against their pinned size and SHA-256; exit 1 if any is missing or wrong.

**Usage**:

```console
$ argus-puppet models verify [OPTIONS] [names]...
```

**Arguments**:

* `names... <list[str]>`: Registry names (default: every pinned model)

**Options**:

* `--help`: Show this message and exit.

## `argus-puppet library`

Build a pose library from a folder, and sample diverse poses from it.

**Usage**:

```console
$ argus-puppet library [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `build`: Build a pose library: one entry per...
* `sample`: Pick N mutually distant poses...

### `argus-puppet library build`

Build a pose library: one entry per image&#x27;s primary pose, labelled, deduplicated and clustered.

**Usage**:

```console
$ argus-puppet library build [OPTIONS] {folder}
```

**Arguments**:

* `folder <path>`: A folder of images with &lt;stem&gt;.pose.json sidecars  [required]

**Options**:

* `-o, --out <path>`: Where to write the library JSON  [required]
* `--dedupe-threshold <float>`: Drop poses closer than this to a better one (torso units; 0 = off)  [default: 0.08]
* `--max-entries <int range>`: Keep a diverse subset of this size  [x&gt;=1]
* `--min-body-score <float>`: Minimum mean body-joint score of a pose  [default: 0.3]
* `--created-from <str>`: Provenance note, e.g. &#x27;CC0 photos of adults&#x27;
* `--seed <int>`: Seed for the --max-entries subset (default: deterministic)
* `--recursive / --no-recursive`: Include subfolders  [default: recursive]
* `--extract`: Pose the images that have no sidecar yet, first
* `--age-check`: Send each source image to argus-curator (ARGUS_PUPPET_AGE_URL) and drop any frame where a face or body reads under 18, or that cannot be checked
* `--device <str>`: Where to run extraction: auto, cpu, cuda (default: settings)
* `--json`: Print the build counts as JSON
* `--help`: Show this message and exit.

### `argus-puppet library sample`

Pick N mutually distant poses (farthest-point sampling, every cluster before any twice).

**Usage**:

```console
$ argus-puppet library sample [OPTIONS] {library}
```

**Arguments**:

* `library <file>`: A library JSON written by `library build`  [required]

**Options**:

* `-n, --count <int range>`: How many poses to pick  [x&gt;=1; required]
* `--seed <int>`: Random first pick, reproducibly (default: the most central)
* `--stance <str>`: Only this stance: standing, jumping, crouching, seated, lying, unknown
* `--arms <str>`: Only these arms: up, out, down, mixed, unknown
* `--facing <str>`: Only this facing: front, back, left, right, unknown
* `--render-dir <path>`: Write a control PNG per picked pose here
* `--size <str>`: Control image size WIDTHxHEIGHT (with --render-dir)  [default: 1024x1024]
* `--hands / --no-hands`: Draw the hands  [default: hands]
* `--face / --no-face`: Draw the face landmarks  [default: face]
* `--stroke-scale <str>`: Stroke width factor, or &#x27;auto&#x27; (min side / 512)  [default: auto]
* `--json`: Print the picked entries as JSON
* `--help`: Show this message and exit.
