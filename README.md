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
pip install anvil-robotics            # Layer 1 cascade only — works on CPU
pip install anvil-robotics[full]      # + DINOv3, SAM 3, VLM agent (needs GPU)
```

**From source** (until v0.1.0 is published):

```bash
git clone https://github.com/arpitg1304/anvil && cd anvil
uv sync --group dev
uv run anvil --help
```

## Quick start

```bash
anvil pin   --camera 0 --name pick_red_cube --task "pick the red cube"
anvil check --against pick_red_cube
anvil guard --against pick_red_cube --record-cmd "lerobot record ..."
```

Add `--force` to `pin` to overwrite an existing reference. To validate the
pin → check loop end-to-end (including a deliberate camera nudge), follow
the sanity-check flow in [docs/aruco_setup.md](docs/aruco_setup.md#fix-in-place-then-sanity-check).

**Enable the DINOv3 backbone (recommended).** The bare install computes
`scene_drift` from a colour histogram — fine for catching gross changes,
weak on subtle semantic drift (object swaps, partial occlusion). Install
the `[full]` extra and authenticate with Hugging Face to switch to
DINOv3-ViT-S/16 (`1 - cosine` of the global CLS embedding):

```bash
uv sync --extra full
uv run --extra full hf auth login    # paste a read-scope HF token
```

DINOv3 weights are gated — accept the licence at
[facebook/dinov3-vits16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m)
first. If the gate hasn't been approved yet, Anvil cascades to DINOv2-Small
(ungated) automatically; `pin` and `check` print which backbone is live as
the `embedder:` field. When no embedder loads, the histogram fallback runs
silently — pin output will say `embedder: histogram (fallback)`.

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
in place, Anvil can detect camera-mount sag/twist — the single most
common silent failure mode for ceiling-mounted webcams over weeks of use.

## How it works

A three-layer cascade. Cheap checks gate expensive ones, so most checks
finish in well under a second:

1. **Layer 0** — optional robot home-pose check (joint state diff)
2. **Layer 1** — histogram + global DINOv3 similarity + ArUco
3. **Layer 2** — SAM 3 per-object IoU + region embeddings + camera pose drift
4. **Layer 3** — Qwen3-VL-4B agent loop, only when 1–2 are ambiguous

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
