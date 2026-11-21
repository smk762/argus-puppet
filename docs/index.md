# argus-puppet

argus-puppet owns body pose for the Argus suite. It finds the people in an image and places 133 whole-body
keypoints on each: 17 body joints, 6 foot points, 68 face landmarks and 21 points per hand, in the COCO-WholeBody
layout. From those keypoints it measures sets of images (how varied the poses are, which pose dominates, whether an
output copies a training frame's pose) and draws skeleton control images that steer a pose ControlNet. Other
packages call it as a Python library, through the `argus-puppet` CLI, or over HTTP on port 8110. It needs no GPU:
everything runs on ONNX Runtime, on the CPU or on CUDA.

## The pipeline

```
images ──▶ extract ──▶ PoseResult (<stem>.pose.json)
                          │
                          ├──▶ pose vector ──▶ stats    ──▶ PoseSetStats       (diversity, clusters, coverage)
                          │                └─▶ compare  ──▶ PoseSetComparison  (nearest training pose per output)
                          │
                          └──▶ render ──▶ skeleton PNG for a pose ControlNet
                               library build ──▶ PoseLibrary ──▶ library sample ──▶ N diverse skeletons
```

| Stage | What happens | What it produces |
|---|---|---|
| Extract | A person detector (YOLOX) finds each person; a pose model (DWPose) places 133 keypoints in each box. One person is marked primary, by argus-curator's primary-face rule | `PoseResult`, saved beside the image as a `<stem>.pose.json` sidecar. It records the SHA-256 of both models |
| Pose vector | The primary person's 17 body joints are centred on the hips, scaled by torso length, with low-confidence joints masked. The distance between two vectors ignores left/right mirroring | A distance in torso units that every later stage uses |
| Stats | Pairwise distances over a set, clustered; geometry rules label each pose's stance, arms and facing | `PoseSetStats`: a 0 to 1 diversity score, cluster sizes and medoids, label coverage |
| Compare | Each output pose's nearest pose in a reference set | `PoseSetComparison`: matches, close-match share against the reference set's own baseline, coverage drift |
| Render | Keypoints drawn in the OpenPose colour convention most pose ControlNets were trained on | A PNG control image |
| Library | Primary poses from a folder, labelled, deduplicated, clustered; sampled so a batch spreads across the poses | `PoseLibrary` JSON, and control images for a sampled batch |

## Pages

| Page | What it covers |
|---|---|
| [cli.md](cli.md) | Every command and option, generated from the CLI |
| [server.md](server.md) | HTTP routes, request and response shapes, status codes, limits, the security model |
| [pose-vector.md](pose-vector.md) | The normalised pose vector and the mirror-invariant pose distance, stated exactly |
| [metrics.md](metrics.md) | Diversity, clustering, the stance/arms/facing labels, set-vs-set comparison, calibration and limits |
| [conditioning.md](conditioning.md) | Skeleton control images, the pose library, diverse sampling, use in a generator |
| [models.md](models.md) | The four pinned models, verification, measured cost, the licence position |
| [deploy.md](deploy.md) | Container images, environment variables, the data root, GPU sharing through argus-iris |

To install it and try every stage on a folder of photos, follow the
[README's quickstart](https://github.com/smk762/argus-puppet#quickstart). The wire types themselves are in `src/argus_puppet/models.py` and `schema/puppet-wire.schema.json`; the field
descriptions there are authoritative.

## Which page do I need

**argus-proof: the pose-diversity gate and pose-copying evidence.** Pose the outputs (and the training set, for
comparison) with `POST /pose/scan` or `argus-puppet pose`, then read `diversity` from `POST /stats` or
`pose_set_stats`. [metrics.md](metrics.md#how-argus-proof-should-use-this) gives the suggested gate (enough usable
and comparable poses first, then reject `pose_collapse` when `diversity < 0.3` or `largest_cluster_share >= 0.75`),
what to report beside it, and how to read
`POST /compare`. At the close threshold most same-stance poses match some training frame, so a per-output close
match only says which frame an image-level check (leak, identity) should look at; the run-level signal is
`close_share` well above the training set's own `baseline_close_share`.

**argus-curator: body-pose coverage.** `PoseSetStats.coverage` counts stance, arms and facing labels, and
`largest_cluster_share` with the cluster's `medoid` says when one pose dominates a set before training. The labels
are geometry rules, tested on synthetic figures only; read [metrics.md](metrics.md#labels) for what they can and
cannot tell apart. Puppet's primary person follows curator's primary-face rule, and `POST /pose?face=x,y,w,h`
takes curator's primary face as a hint ([server.md](server.md#the-primary-subject)).

**argus-darkroom: a pose input for ControlNet.** `POST /render` (or `render_png`) turns a sidecar into a control
image at the generation size; a pose library gives a diverse batch. [conditioning.md](conditioning.md) covers the
colour convention, sizing, and why the source photo, not the skeleton, is what the input checks apply to.

**Operators.** [deploy.md](deploy.md) for the images, the model volume and every environment variable;
[server.md](server.md#security-model) for what the service exposes; [models.md](models.md) for fetching and
verifying weights and for the licence position.

## Privacy and the age fail-safe

Puppet keeps keypoints, not pixels. A sidecar or library entry holds coordinates, scores, the image's name and its
SHA-256; nothing in it reproduces the image. The 68 face landmarks do describe the geometry of a face, so treat
sidecars of real people with the same care as the photos' metadata.

Images never leave the host during extraction, measurement or rendering. The outbound requests puppet makes are
`models fetch` (to the two model hosts), GPU lease calls to argus-iris when `ARGUS_GPU_COORDINATOR=broker` (these
carry no image data), and the age check below.

### `library build --age-check`

The suite keeps any frame where a face **or a body** reads under 18 out of a pose library, background people
included. With `--age-check` (or `build_library(..., age_gate=AgeGate(...))` in Python), each image that would
otherwise become an entry is sent to argus-curator's `POST /faces?body=true`, which returns a MiVOLO age estimate
for every face and every body it finds.

- An image is kept only when **every** face and body reads 18 or over.
- It **fails closed**. An image is also dropped when curator is unreachable or answers an error, when the answer
  cannot vouch for the image (a face without an estimate, a body pass that did not report `ok`), when nobody at all
  could be estimated, or when the image beside the sidecar cannot be read or no longer matches the hash its pose
  came from.
- The build reports how many images passed, read under age, or could not be checked, and names each excluded image
  with the reason. If not a single image could be checked, the CLI writes no library and exits 1.
- The library records `age_checked: true`. A library built without the check records `false`, and its sources should
  be treated as unchecked.

An estimate is a model's reading of an image, never a person's stated age.

**Nothing is sent anywhere without both the flag and a configured URL.** The URL comes only from
`ARGUS_PUPPET_AGE_URL` (curator's base URL), never from a request or an argument; `--age-check` without it is an
error. The call has a per-read timeout and an overall deadline, allows exactly the configured host, and does not
follow redirects. The HTTP service has no route that runs the age check.

Use the check whenever a library's sources are real photographs. Public-domain, CC0 or synthetic sources are still
the safer choice ([conditioning.md](conditioning.md#building-one)).

### Fixtures and demos

Test fixtures, documentation examples and demos use images of adults only: synthetic stick figures in the unit tests
and CC0 photographs of adults for end-to-end checks.
