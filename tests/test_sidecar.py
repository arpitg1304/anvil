"""Tests for the per-episode sidecar writer."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from anvil.schema import (
    ANVIL_SCHEMA_VERSION,
    EpisodeReport,
    PinReference,
    Provenance,
    Scores,
)
from anvil.sidecar import (
    anvil_dir,
    manifest_hash_path,
    sidecar_path,
    snapshot_episodes,
    write_session_manifest_hash,
    write_sidecars,
)


def _make_report(episode_id: str = "manual_check") -> EpisodeReport:
    return EpisodeReport(
        anvil_schema_version=ANVIL_SCHEMA_VERSION,
        episode_id=episode_id,
        checked_at=datetime.now(tz=UTC),
        pin=PinReference(
            name="workspace",
            manifest_hash="sha256:" + "a" * 64,
            pinned_at=datetime.now(tz=UTC),
        ),
        status="passed",
        operator_action="proceeded",
        scores=Scores(
            scene_drift=0.05,
            lighting_drift=0.02,
            max_object_drift_cm=0.0,
            max_camera_pose_drift_deg=0.1,
            max_robot_joint_drift_deg=0.0,
        ),
        flags=[],
        findings=[],
        provenance=Provenance(anvil_version="0.1.0.dev0", hardware={}),
    )


def test_snapshot_returns_empty_for_missing_dataset_dir(tmp_path: Path) -> None:
    assert snapshot_episodes(tmp_path / "nope") == set()


def test_snapshot_returns_empty_for_dataset_without_videos(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    assert snapshot_episodes(tmp_path) == set()


def test_snapshot_finds_episode_dirs_under_videos(tmp_path: Path) -> None:
    base = tmp_path / "videos" / "chunk-000"
    (base / "episode_000001").mkdir(parents=True)
    (base / "episode_000042").mkdir()
    # Decoy: not matching the episode_NNNNNN pattern.
    (base / "not_an_episode").mkdir()
    assert snapshot_episodes(tmp_path) == {"episode_000001", "episode_000042"}


def test_snapshot_handles_multiple_chunks(tmp_path: Path) -> None:
    (tmp_path / "videos" / "chunk-000" / "episode_000001").mkdir(parents=True)
    (tmp_path / "videos" / "chunk-001" / "episode_000002").mkdir(parents=True)
    assert snapshot_episodes(tmp_path) == {"episode_000001", "episode_000002"}


def test_write_sidecars_creates_one_file_per_episode(tmp_path: Path) -> None:
    report = _make_report(episode_id="preflight")
    written = write_sidecars(tmp_path, ["episode_000001", "episode_000002"], report)
    assert len(written) == 2
    for ep_id, path in zip(
        ["episode_000001", "episode_000002"], written, strict=False
    ):
        assert path == sidecar_path(tmp_path, ep_id)
        payload = json.loads(path.read_text())
        assert payload["episode_id"] == ep_id
        # Ensure the pin reference + other fields survived the copy.
        assert payload["pin"]["name"] == "workspace"


def test_write_sidecars_creates_anvil_subdir(tmp_path: Path) -> None:
    write_sidecars(tmp_path, ["episode_000001"], _make_report())
    assert anvil_dir(tmp_path).exists()
    assert anvil_dir(tmp_path).is_dir()


def test_write_session_manifest_hash(tmp_path: Path) -> None:
    path = write_session_manifest_hash(tmp_path, "sha256:abc123")
    assert path == manifest_hash_path(tmp_path)
    assert path.read_text().strip() == "sha256:abc123"


def test_write_sidecars_with_empty_list_writes_nothing(tmp_path: Path) -> None:
    written = write_sidecars(tmp_path, [], _make_report())
    assert written == []
    # The anvil/ subdir is created either way (mkdir parents=True),
    # but no sidecar files inside.
    assert anvil_dir(tmp_path).exists()
    assert list(anvil_dir(tmp_path).iterdir()) == []
