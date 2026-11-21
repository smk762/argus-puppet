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
