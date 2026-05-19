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

## Quick start

```bash
anvil pin   --camera 0 --name pick_red_cube --task "pick the red cube"
anvil check --against pick_red_cube
anvil guard --against pick_red_cube --record-cmd "lerobot record ..."
```

See [docs/quickstart.md](docs/quickstart.md) when it lands.

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

## How it works

A three-layer cascade. Cheap checks gate expensive ones, so most checks
finish in well under a second:

1. **Layer 0** — optional robot home-pose check (joint state diff)
2. **Layer 1** — histogram + global DINOv3 similarity + ArUco
3. **Layer 2** — SAM 3 per-object IoU + region embeddings + camera pose drift
4. **Layer 3** — Qwen3-VL-4B agent loop, only when 1–2 are ambiguous

See [docs/architecture.md](docs/architecture.md) and the full plan in
[anvil_plan.md](anvil_plan.md).

## License

MIT. See [LICENSE](LICENSE).
