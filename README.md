```
   █████╗ ███╗   ██╗██╗   ██╗██╗██╗
  ██╔══██╗████╗  ██║██║   ██║██║██║
  ███████║██╔██╗ ██║██║   ██║██║██║
  ██╔══██║██║╚██╗██║╚██╗ ██╔╝██║██║
  ██║  ██║██║ ╚████║ ╚████╔╝ ██║███████╗
  ╚═╝  ╚═╝╚═╝  ╚═══╝  ╚═══╝  ╚═╝╚══════╝
```

# Anvil — keep your scene true

A scene-consistency sentinel for robotics data collection. Pin a scene once;
every future session is checked against the pin. Drift gets flagged before it
gets baked into your dataset.

Sibling to [Forge](https://github.com/arpitg1304/forge): Forge shapes your data,
Anvil keeps your scene true.

> **Status:** v0.1.0 in active development. Schema is the only stable surface;
> CLI and internals will change until 0.1.0 ships.

## Install

```bash
# Bare install: Layer 1 fast path (histogram + ArUco + lighting), guard, and
# the inspector UI. CPU-only, works on a laptop with a webcam.
pip install anvil-robotics

# [full] adds the ML cascade: DINOv3 (scene embedding), DISK + LightGlue
# (marker-free camera pose), GroundingDINO + YOLO-World (named-object
# detection). Needs ~3GB of model weights and a GPU for reasonable speed.
pip install anvil-robotics[full]
```

**From source** (until v0.1.0 is published):

```bash
git clone https://github.com/arpitg1304/anvil && cd anvil
uv sync --group dev
uv run anvil --help
```

## Quick start

```bash
# 1. Pin a reference of your workspace. Optionally name objects to track.
anvil pin --camera 0 --name workspace --task "general workspace" \
  --object rack="test tube rack"

# 2. Re-check at any time. Returns exit code 0 / 1 / 2 (pass / warn / fail)
#    and writes an annotated diff PNG to .anvil/workspace/diffs/ on drift.
anvil check --against workspace

# 3. Wrap your recorder. Runs check first; on WARNING prompts you to
#    proceed; on FAILED blocks. Writes per-episode JSON sidecars under
#    <dataset>/anvil/ for every episode the recorder creates.
anvil guard --against workspace \
  --record-cmd "lerobot record --task ..." \
  --dataset-dir ./my_dataset

# 4. Browse pins, manifests, and diff history in a local browser tab.
anvil-inspect            # localhost:7777, auto-opens the browser
```

Add `--force` to `pin` to overwrite an existing reference. To validate the
pin → check loop end-to-end (including a deliberate camera nudge), follow
the sanity-check flow in [docs/aruco_setup.md](docs/aruco_setup.md#fix-in-place-then-sanity-check).

**Enable the `[full]` ML cascade (recommended).** The bare install gets
you ArUco-based pose drift, colour-histogram scene drift, lighting, and
the inspector — enough for catching gross changes on a CPU-only setup.
The `[full]` extra layers the real ML on top:

- **DINOv3-ViT-S/16** for semantic `scene_drift` (`1 - cosine` of the
  global CLS embedding) — catches object swaps and partial occlusion
  that histograms miss. Auto-cascades to DINOv2-Small if the DINOv3 HF
  gate hasn't been approved yet.
- **DISK + LightGlue** (via kornia) for marker-free camera pose drift —
  detects mount sag / twist / lateral slip without printed fiducials.
- **GroundingDINO** (with YOLO-World fallback) for **named-object
  tracking**: pin with `--object red_cube="red cube"`, and `check`
  reports per-object IoU + DINOv3 region cosine vs the pin.

```bash
uv sync --extra full
uv run --extra full hf auth login    # paste a read-scope HF token for DINOv3
```

DINOv3 weights are gated — accept the licence at
[facebook/dinov3-vits16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m)
first. GroundingDINO, DISK, LightGlue, DINOv2, and YOLO-World are all
ungated and download automatically on first use. `pin` and `check`
print which backbones loaded under the `embedder:`, `keypoints:`, and
`objects:` lines respectively.

## Finding your camera

Anvil takes a `--camera` argument that's an OpenCV index (`0`, `1`, …), a
device path (`/dev/video0`), or an RTSP URL. Use your OS tooling to figure
out which is which before pinning.

**Linux (V4L2):**

```bash
ls /dev/video*                          # enumerate device nodes
v4l2-ctl --list-devices                 # friendly names + node mapping (apt: v4l-utils)
v4l2-ctl -d /dev/video0 --list-formats-ext   # supported resolutions + framerates
ffplay /dev/video0                      # live preview (Ctrl-C to exit)
```

**macOS (AVFoundation):**

```bash
system_profiler SPCameraDataType                          # plain device list
ffmpeg -f avfoundation -list_devices true -i ""           # AVFoundation indices
ffplay -f avfoundation -framerate 30 -i 0                 # live preview from index 0
```

If multiple cameras show up, the OpenCV index usually matches enumeration
order — try `--camera 0` first, then `1`, etc.

**Optional but recommended for serious rigs:** print four ArUco fiducials
and stick them at the workspace corners. See [docs/aruco_setup.md](docs/aruco_setup.md)
for sizing, placement, and a printable generator one-liner. With markers
in place, Anvil's cheap Layer 1 pose check works without any GPU; with
the `[full]` extra installed, the more accurate DISK + LightGlue path
runs on top and supersedes ArUco automatically.

## Inspector UI

`anvil-inspect` serves a read-only browser view of your pins on
`http://127.0.0.1:7777`:

- Card grid of every pin under `.anvil/` with reference thumbnails and
  status badges.
- Per-pin detail: full reference image, manifest summary, named-object
  list, and the full diff-image history sorted newest-first.
- No DB, no auth, no actions — refreshes the filesystem on every page
  load. Safe to run alongside `pin` / `check` / `guard`.

```bash
anvil-inspect                          # localhost:7777, opens a tab
anvil-inspect --root /data/.anvil      # browse a different pin tree
anvil-inspect --port 8000 --no-browser # for SSH'd remote rigs
```

## Sidecars (the Forge contract)

When you wrap your recorder with `anvil guard --dataset-dir ./my_dataset`,
Anvil writes one schema-valid `EpisodeReport` JSON next to every episode
the recorder creates:

```
my_dataset/
├── data/, videos/, meta/      # your recorder's output
└── anvil/
    ├── manifest_hash.txt      # which pin this session was checked against
    ├── episode_000042.anvil.json
    └── episode_000043.anvil.json
```

The schema lives at [docs/metadata_schema.md](docs/metadata_schema.md) and
is the stable contract with [Forge](https://github.com/arpitg1304/forge):
filter episodes by drift after the fact (`forge filter ./my_dataset
./clean --max-pose-drift-deg 1.0`), correlate policy failures with
collection conditions, etc.

## How it works

A multi-layer cascade. Cheap checks gate expensive ones, so most checks
finish in well under a second:

1. **Layer 1** — histogram + lighting (CCT, luminance) + ArUco pose +
   DINOv3 / DINOv2 / histogram-fallback scene drift.
2. **Layer 2** — DISK + LightGlue marker-free camera pose drift +
   GroundingDINO / YOLO-World named-object IoU + DINOv3 region cosine.
3. **Layer 0** *(TBD)* — robot home-pose check via LeRobot joint state.
4. **Layer 3** *(TBD)* — Qwen3-VL agent loop for open-world drift the
   earlier layers couldn't resolve.

See [docs/architecture.md](docs/architecture.md) and the full plan in
[anvil_plan.md](anvil_plan.md).

## Troubleshooting

- **`pip install anvil-robotics` fails / not on PyPI:** v0.1.0 isn't published
  yet — use the from-source path above.
- **`pin` reports `aruco: 0 marker(s)` with markers visible:** markers need a
  white quiet zone for boundary contrast against the work surface. See
  [docs/aruco_setup.md](docs/aruco_setup.md#troubleshooting).
- **`pin` shows `embedder: histogram (fallback)` after installing `[full]`:**
  the DINOv3 weights couldn't be fetched. Most likely you haven't accepted the
  licence on [the HF model page](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m)
  or aren't authenticated. Run `uv run --extra full hf auth login` with a
  read-scope token. While waiting on the gate, the factory cascades to
  DINOv2 (`embedder: dinov2-small`) automatically.
- **`$VIRTUAL_ENV` empty after `source .venv/bin/activate`:** `pyenv
  virtualenv-init` in your shell rc overrides it on every prompt. Use `uv run
  <cmd>` instead, or `PROMPT_COMMAND= source .venv/bin/activate`.
- **`pytest` ImportError for `lark` from a `/opt/ros/.../python3.12` path:**
  ROS 2's `setup.bash` is leaking `PYTHONPATH` into the venv. Prefix the
  command with `PYTHONPATH= ` or stop sourcing ROS globally in `~/.bashrc`.

## License

MIT. See [LICENSE](LICENSE).
