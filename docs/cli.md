# `argus-puppet`

Body pose for the Argus suite: whole-body keypoints, pose metrics and ControlNet pose conditioning.

**Usage**:

```console
$ argus-puppet [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `version`: Print the package version and the...
* `schema`: Emit the wire-contract JSON Schema...
* `pose`: Write a &lt;stem&gt;.pose.json sidecar beside...
* `serve`: Run the HTTP service (needs the ...
* `models`: List, download and verify the pinned model...

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

## `argus-puppet serve`

Run the HTTP service (needs the  extra).

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
