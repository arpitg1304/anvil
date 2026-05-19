# Changelog

All notable changes to Anvil are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Project scaffold: `uv`-managed `pyproject.toml`, src layout, ruff + mypy + pytest config.
- `docs/metadata_schema.md` — v0.1 episode sidecar schema (the Forge contract).
- `anvil.schema` — Pydantic v2 models for the sidecar, with Literal enums
  for `status`, `severity`, `component`, `operator_action`, and the 9-flag
  vocabulary.
- `anvil.manifest` — `Manifest` model + YAML save/load + `compute_manifest_hash`
  (sha256 over every file in the pin directory, in sorted-path order).
- `anvil.cameras` — `Camera` abstract base + `WebcamCamera` (OpenCV V4L2 /
  AVFoundation) + `FileCamera` (test/replay driver).
- `anvil.errors` — `AnvilError` + `PinNotFound`, `ManifestVersionMismatch`,
  `CameraError`, `RobotError`.
- `anvil.cli` — Click app with the three locked verbs (`pin`, `check`,
  `guard`). `guard` is a clean "not yet" pointing to Week 3.
  - `pin` writes `manifest.yaml`, `reference.png`, `lighting.json`, and
    `pose/aruco.json`.
  - `check` loads pin references, runs Layer 1, derives `status` from
    finding severity, emits a schema-valid `EpisodeReport` JSON.
    Exit codes 0 / 1 / 2 / 3 for passed / warning / failed / error.
- `anvil.layers.layer1_fast` — Layer 1 fast path on CPU:
  - Per-channel BGR histograms + symmetric chi-square distance.
  - Mean luminance and McCamy-approximated correlated color temperature.
  - ArUco fiducial detection + rotation drift.
  - Composite `lighting_drift` ∈ [0, 1], histogram-derived `scene_drift`,
    and `max_camera_pose_drift_deg` from ArUco orientation deltas.
  - DINOv3 global cosine remains TODO behind the `[full]` extra.
- 73 tests across schema, manifest, cameras, Layer 1, and CLI. `mypy
  --strict` and `ruff` both clean across 15 source files.
