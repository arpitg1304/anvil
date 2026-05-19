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
  `guard`). `pin` is end-to-end functional (writes `manifest.yaml` +
  `reference.png`). `check` loads the manifest and emits a schema-valid
  `EpisodeReport` JSON with placeholder zeros for the vision scores;
  exit codes 0 / 1 / 2 / 3 for passed / warning / failed / error.
  `guard` is a clean "not yet" pointing to Week 3.
- 58 tests across schema, manifest, cameras, and CLI. `mypy --strict` and
  `ruff` both clean.
