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
  - Also emits a `translation_drift` finding when the recovered
    translation magnitude exceeds `thresholds.max_camera_translation_px`
    (default 15 px). Catches the "camera mount slipped vertically"
    case that pure rotation thresholding misses — both findings raise
    the existing `camera_pose_drift` flag, no schema change required.
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
- `anvil.models.objects` — open-vocabulary object detector abstraction:
  - `ObjectDetector` ABC + `DetectedObject` Pydantic type +
    `load_object_detector()` factory.
  - `GroundingDINODetector` (`IDEA-Research/grounding-dino-tiny`,
    ~700MB unrestricted weights via HF transformers) — primary
    concrete impl. Designed for open-vocabulary "text → bbox" on
    arbitrary concepts (industrial prompts like "test tube rack" and
    "robot gripper" both return usable confidence). No HF gate.
  - `YOLOWorldDetector` (Ultralytics YOLO-World v8s, ~75MB
    unrestricted Tencent weights) — fallback. COCO-skewed training,
    less reliable on industrial vocabulary, kept as a smaller-install
    option when GroundingDINO can't load. Pinned to CPU to dodge an
    Ultralytics 8.4 `set_classes` CUDA tensor-device bug.
  - `[full]` extra now also pulls `ultralytics>=8.3`. (GroundingDINO
    rides the existing `transformers` dep — no new heavy package.)
- `anvil.layers.layer2_structural` — adds named-object workflow:
  - `compute_object_reference` / `save_object_reference` /
    `load_object_references` persist `objects/<name>.json` per the
    plan's manifest layout, including a DINOv3 region-cosine embedding
    of the bbox crop when an embedder is available.
  - `run_layer2_objects` re-detects each pinned object, computes per-
    object IoU + centroid delta + region cosine, and emits
    `object_missing`, `position_drift`, or `appearance_drift` findings.
    All three raise the existing `object_missing` or `object_moved`
    flags — no flag-vocabulary change.
- `pin --object NAME` (repeatable) names objects to track. Supports the
  `name=prompt` form for cases where the desired text prompt differs
  from the on-disk filename (e.g. `--object rack="test tube rack"`).
- `pin --force` now also wipes `objects/` and `pose/` subdirectories so
  re-pinning with a changed object set doesn't leave ghost JSONs. The
  `diffs/` subdir is preserved across re-pins.
- `ThresholdSpec` gains `min_object_iou` (0.5 default) and
  `min_object_appearance_cosine` (0.85 default).
- 115 tests across all layers, models, persistence, CLI, and diff
  renderer. `mypy --strict` and `ruff` clean across 24 source files.
- `anvil guard` — wraps an arbitrary recording command with a
  pre-flight `check` against a pinned reference. On WARNING, the
  default is to prompt the operator interactively (configurable via
  `--on-warning {prompt|tag|block}`); on FAILED, blocks by default
  (`--on-failed {block|prompt|proceed}`). When `--dataset-dir` is
  supplied, snapshots the directory before/after the record command
  and writes per-episode sidecars under `<dataset>/anvil/`.
- `anvil.sidecar` — LeRobotDataset v3 sidecar writer:
  - `snapshot_episodes()` lists `videos/*/episode_NNNNNN/` directories.
  - `write_sidecars()` writes one `EpisodeReport` JSON per new episode
    under `<dataset>/anvil/`.
  - `write_session_manifest_hash()` drops a `manifest_hash.txt` next to
    the sidecars naming the pin the session was checked against.
- `check_cmd` refactored to share a `_run_check` helper with
  `guard_cmd` — same pipeline, same diff rendering, no code duplication.
- 132 tests across all layers, models, persistence, CLI, sidecar, and
  diff renderer. `mypy --strict` and `ruff` clean across 26 source files.
- `anvil-inspect` — read-only localhost web UI for browsing pins.
  FastAPI + Jinja2 + Tailwind via CDN, no build step. Launched as a
  separate console script (the three CLI verbs stay locked at
  pin/check/guard).
  - Home (`/`): grid of all pins under `.anvil/` with reference
    thumbnails, pinned-at dates, and tag badges (aruco, embedder,
    keypoints, objects, diff count).
  - Pin detail (`/pin/<name>`): full reference image, manifest summary,
    named-object list with bboxes, raw `manifest.yaml` in a collapsible
    panel, and a diff-history gallery sorted newest-first.
  - Static-file routes for `reference.png` and individual diff PNGs,
    both with path-traversal protection.
- `anvil.server` package — FastAPI app factory in `app.py`, console
  script entry point in `main.py`, three Jinja2 templates. The
  inspector reads from disk on every request; no caching, no DB, no
  shared state.
- Main install now pulls `fastapi`, `uvicorn`, `jinja2` (~10MB
  combined) for the inspector. Layer-1-only users still don't need
  the `[full]` extra.
- 143 tests across all layers, models, persistence, CLI, sidecar,
  diff renderer, and server. `mypy --strict` and `ruff` clean across
  28 source files.
