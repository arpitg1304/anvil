"""Manifest model + I/O + hash determinism tests."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from anvil.errors import ManifestVersionMismatch, PinNotFound
from anvil.manifest import (
    MANIFEST_FILENAME,
    CameraSpec,
    Manifest,
    RobotSpec,
    ThresholdSpec,
    compute_manifest_hash,
    load_manifest,
    pin_directory,
    save_manifest,
)
from anvil.schema import ANVIL_SCHEMA_VERSION


def _make_manifest() -> Manifest:
    return Manifest(
        anvil_schema_version=ANVIL_SCHEMA_VERSION,
        name="pick_red_cube",
        task="pick the red cube",
        pinned_at=datetime(2026, 5, 19, 12, 0, 0, tzinfo=timezone.utc),
        camera=CameraSpec(driver="webcam", device="0", resolution=(640, 480)),
        robot=RobotSpec(enabled=True, driver="lerobot", home_pose_joints=[0.0, -1.57]),
        thresholds=ThresholdSpec(),
    )


# --- pin_directory -------------------------------------------------------


def test_pin_directory_default_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert pin_directory("foo") == tmp_path / ".anvil" / "foo"


def test_pin_directory_explicit_root(tmp_path: Path) -> None:
    assert pin_directory("foo", root=tmp_path) == tmp_path / "foo"


@pytest.mark.parametrize("name", ["", ".", "..", "a/b", "a\\b"])
def test_pin_directory_rejects_bad_names(name: str) -> None:
    with pytest.raises(ValueError):
        pin_directory(name)


# --- save / load round-trip ---------------------------------------------


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    manifest = _make_manifest()
    save_manifest(manifest, tmp_path)
    assert (tmp_path / MANIFEST_FILENAME).exists()
    reloaded = load_manifest(tmp_path)
    assert reloaded == manifest


def test_load_missing_dir_raises_pin_not_found(tmp_path: Path) -> None:
    with pytest.raises(PinNotFound):
        load_manifest(tmp_path / "does-not-exist")


def test_load_missing_manifest_raises_pin_not_found(tmp_path: Path) -> None:
    with pytest.raises(PinNotFound):
        load_manifest(tmp_path)


def test_load_rejects_future_major_version(tmp_path: Path) -> None:
    manifest = _make_manifest()
    save_manifest(manifest, tmp_path)
    # Tamper the on-disk version to a future major.
    path = tmp_path / MANIFEST_FILENAME
    text = path.read_text().replace(ANVIL_SCHEMA_VERSION, "1.0.0")
    path.write_text(text)
    with pytest.raises(ManifestVersionMismatch):
        load_manifest(tmp_path)


def test_robot_defaults_disabled() -> None:
    robot = RobotSpec()
    assert robot.enabled is False
    assert robot.driver == "none"
    assert robot.home_pose_joints == []


# --- compute_manifest_hash ----------------------------------------------


def test_hash_deterministic_for_same_content(tmp_path: Path) -> None:
    manifest = _make_manifest()
    save_manifest(manifest, tmp_path)
    (tmp_path / "reference.png").write_bytes(b"fake-png-bytes")
    h1 = compute_manifest_hash(tmp_path)
    h2 = compute_manifest_hash(tmp_path)
    assert h1 == h2
    assert h1.startswith("sha256:")
    assert len(h1) == len("sha256:") + 64


def test_hash_changes_when_file_content_changes(tmp_path: Path) -> None:
    manifest = _make_manifest()
    save_manifest(manifest, tmp_path)
    (tmp_path / "reference.png").write_bytes(b"v1")
    h1 = compute_manifest_hash(tmp_path)
    (tmp_path / "reference.png").write_bytes(b"v2")
    h2 = compute_manifest_hash(tmp_path)
    assert h1 != h2


def test_hash_changes_when_file_renamed(tmp_path: Path) -> None:
    save_manifest(_make_manifest(), tmp_path)
    (tmp_path / "a.bin").write_bytes(b"same-content")
    h_a = compute_manifest_hash(tmp_path)
    (tmp_path / "a.bin").rename(tmp_path / "b.bin")
    h_b = compute_manifest_hash(tmp_path)
    assert h_a != h_b


def test_hash_changes_when_new_file_added(tmp_path: Path) -> None:
    save_manifest(_make_manifest(), tmp_path)
    h1 = compute_manifest_hash(tmp_path)
    (tmp_path / "new.bin").write_bytes(b"x")
    h2 = compute_manifest_hash(tmp_path)
    assert h1 != h2


def test_hash_walks_subdirectories(tmp_path: Path) -> None:
    save_manifest(_make_manifest(), tmp_path)
    (tmp_path / "objects").mkdir()
    (tmp_path / "objects" / "cube.json").write_bytes(b"{}")
    h1 = compute_manifest_hash(tmp_path)
    (tmp_path / "objects" / "cube.json").write_bytes(b'{"updated": true}')
    h2 = compute_manifest_hash(tmp_path)
    assert h1 != h2


def test_hash_on_missing_pin_raises(tmp_path: Path) -> None:
    with pytest.raises(PinNotFound):
        compute_manifest_hash(tmp_path / "nope")
