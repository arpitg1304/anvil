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
  - Composite `lighting_drift` ∈ [0, 1] and `max_camera_pose_drift_deg`
    from ArUco orientation deltas.
- `anvil.models` — pluggable global-embedder abstraction:
  - `GlobalEmbedder` ABC + `load_embedder()` factory.
  - `DINOv3Embedder` (`facebook/dinov3-vits16-pretrain-lvd1689m`, primary)
    and `DINOv2Embedder` (`facebook/dinov2-small`, fallback) concrete impls,
    lazy-loaded behind the `[full]` extra. Factory cascades DINOv3 → DINOv2,
    so users blocked on the DINOv3 HF gate still get a semantic
    `scene_drift` signal. CUDA when available, CPU otherwise.
  - `[full]` extra pulls `torch>=2.4`, `torchvision>=0.19`,
    `transformers>=4.45`, `pillow>=10`.
- `Layer1Output.scene_drift_source` exposes which path produced the
  number — `embedder:<name>` when DINOv3 is wired up, `histogram` on
  fallback. `run_layer1` accepts optional `embedding_ref` + `embedder`
  kwargs; `pin` saves `reference_features.npz`, `check` reuses it.
- `Manifest` gains `embedding_present: bool` and `embedder_name: str | None`
  for round-trip provenance.
- `anvil.models.keypoints` — pluggable keypoint detector + matcher
  abstraction:
  - `KeypointDetector` / `KeypointMatcher` ABCs + `KeypointSet` / `MatchResult`
    Pydantic types + `load_keypoint_pipeline()` factory.
  - `DISKDetector` and `LightGlueDISKMatcher` concrete impls via kornia.
    DISK substitutes for SuperPoint (kornia 0.8 dropped SuperPoint; DISK is
    the same family of CNN keypoint+descriptor method with weights kornia
    distributes directly — unrestricted, no HF gate).
  - `[full]` extra now also pulls `kornia>=0.8`.
- `anvil.layers.layer2_structural` — Layer 2 structural orchestrator:
  - `compute_keypoint_reference` / `save_keypoint_reference` /
    `load_keypoint_reference` persist N×128 DISK descriptors and keypoint
    coordinates to `pose/keypoints.npz`.
  - `run_layer2_keypoints` detects + matches keypoints, RANSAC-fits a
    rigid 2D transform via `cv2.estimateAffinePartial2D`, and reports
    rotation + translation + inlier counts. Emits its own
    `rotation_drift` finding (layer=2, `source: lightglue-disk`) when
    over threshold.
- `check` runs Layer 2 after Layer 1 when keypoints are pinned and the
  pipeline loads. Layer 2's rotation supersedes Layer 1's ArUco-derived
  pose score in the final report (more accurate, no fiducials needed).
- `Manifest` gains `keypoints_present: bool` and
  `keypoint_detector_name: str | None`.
- `pin` output now includes a `keypoints:` line; `<pin>/pose/keypoints.npz`
  is written alongside the other reference artifacts.
- 101 tests across schema, manifest, cameras, Layer 1, embedder,
  keypoints, Layer 2, CLI, and diff renderer. `mypy --strict` and `ruff`
  both clean across 22 source files.
