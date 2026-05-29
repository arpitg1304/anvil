"""CLI smoke tests via Click's CliRunner.

Uses the file-backed camera so no real webcam is needed in CI. These tests
exercise the full pipe: argv → click → camera → manifest write → reload →
EpisodeReport JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from click.testing import CliRunner

from anvil.cli import _parse_resolution, cli
from anvil.manifest import (
    MANIFEST_FILENAME,
    REFERENCE_IMAGE_FILENAME,
    load_manifest,
)
from anvil.schema import ANVIL_SCHEMA_VERSION, EpisodeReport


def _write_image(path: Path, color: tuple[int, int, int] = (10, 20, 30)) -> Path:
    img = np.full((48, 64, 3), 0, dtype=np.uint8)
    img[:, :] = color
    assert cv2.imwrite(str(path), img)
    return path


# --- --resolution parsing ------------------------------------------------


def test_parse_resolution_valid() -> None:
    assert _parse_resolution("1920x1080") == (1920, 1080)
    assert _parse_resolution("1280X720") == (1280, 720)  # case-insensitive


def test_parse_resolution_empty_is_none() -> None:
    assert _parse_resolution("") is None
    assert _parse_resolution("   ") is None


def test_parse_resolution_malformed_raises() -> None:
    import click

    for bad in ["1920", "1920x", "axb", "1920x1080x720", "1920*1080"]:
        with pytest.raises(click.BadParameter):
            _parse_resolution(bad)


# --- top-level -----------------------------------------------------------


def test_help_lists_three_verbs() -> None:
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    for verb in ("pin", "check", "guard"):
        assert verb in result.output


def test_version_flag_works() -> None:
    result = CliRunner().invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "anvil" in result.output.lower()


# --- pin -----------------------------------------------------------------


def test_pin_creates_manifest_and_reference(tmp_path: Path) -> None:
    img = _write_image(tmp_path / "ref.png")
    pin_root = tmp_path / "pins"
    result = CliRunner().invoke(
        cli,
        [
            "pin",
            "--name", "pick_red_cube",
            "--camera-driver", "file",
            "--camera", str(img),
            "--task", "pick the red cube",
            "--root", str(pin_root),
        ],
    )
    assert result.exit_code == 0, result.output
    pin_dir = pin_root / "pick_red_cube"
    assert (pin_dir / MANIFEST_FILENAME).exists()
    assert (pin_dir / REFERENCE_IMAGE_FILENAME).exists()

    manifest = load_manifest(pin_dir)
    assert manifest.name == "pick_red_cube"
    assert manifest.task == "pick the red cube"
    assert manifest.camera.driver == "file"
    assert manifest.camera.resolution == (64, 48)
    assert manifest.robot.enabled is False


def test_pin_with_robot_flag_marks_enabled(tmp_path: Path) -> None:
    img = _write_image(tmp_path / "ref.png")
    pin_root = tmp_path / "pins"
    result = CliRunner().invoke(
        cli,
        [
            "pin",
            "--name", "p",
            "--camera-driver", "file",
            "--camera", str(img),
            "--robot",
            "--root", str(pin_root),
        ],
    )
    assert result.exit_code == 0, result.output
    manifest = load_manifest(pin_root / "p")
    assert manifest.robot.enabled is True


def test_pin_refuses_to_overwrite_without_force(tmp_path: Path) -> None:
    img = _write_image(tmp_path / "ref.png")
    pin_root = tmp_path / "pins"
    args = [
        "pin", "--name", "p",
        "--camera-driver", "file", "--camera", str(img),
        "--root", str(pin_root),
    ]
    first = CliRunner().invoke(cli, args)
    assert first.exit_code == 0
    second = CliRunner().invoke(cli, args)
    assert second.exit_code != 0
    assert "already exists" in second.output.lower()


def test_pin_force_overwrites(tmp_path: Path) -> None:
    img = _write_image(tmp_path / "ref.png")
    pin_root = tmp_path / "pins"
    args = [
        "pin", "--name", "p",
        "--camera-driver", "file", "--camera", str(img),
        "--root", str(pin_root),
    ]
    assert CliRunner().invoke(cli, args).exit_code == 0
    assert CliRunner().invoke(cli, [*args, "--force"]).exit_code == 0


# --- check ---------------------------------------------------------------


def test_check_emits_valid_episode_report_json(tmp_path: Path) -> None:
    img = _write_image(tmp_path / "ref.png")
    pin_root = tmp_path / "pins"
    runner = CliRunner()

    pin_result = runner.invoke(
        cli,
        [
            "pin", "--name", "p",
            "--camera-driver", "file", "--camera", str(img),
            "--root", str(pin_root),
        ],
    )
    assert pin_result.exit_code == 0

    check_result = runner.invoke(
        cli,
        [
            "check", "--against", "p",
            "--camera-driver", "file", "--camera", str(img),
            "--root", str(pin_root),
            "--episode-id", "episode_000001",
            "--json",
        ],
    )
    assert check_result.exit_code == 0, check_result.output

    payload = json.loads(check_result.output.strip())
    report = EpisodeReport.model_validate(payload)
    assert report.anvil_schema_version == ANVIL_SCHEMA_VERSION
    assert report.episode_id == "episode_000001"
    assert report.pin.name == "p"
    assert report.pin.manifest_hash.startswith("sha256:")
    assert report.status == "passed"
    assert report.flags == []
    assert report.findings == []


def test_check_missing_pin_exits_with_code_3(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli,
        [
            "check", "--against", "nope",
            "--root", str(tmp_path / "pins"),
            "--json",
        ],
    )
    assert result.exit_code == 3


def test_pin_writes_lighting_and_aruco_files(tmp_path: Path) -> None:
    img = _write_image(tmp_path / "ref.png")
    pin_root = tmp_path / "pins"
    result = CliRunner().invoke(
        cli,
        [
            "pin", "--name", "p",
            "--camera-driver", "file", "--camera", str(img),
            "--root", str(pin_root),
        ],
    )
    assert result.exit_code == 0, result.output
    pin_dir = pin_root / "p"
    assert (pin_dir / "lighting.json").exists()
    assert (pin_dir / "pose" / "aruco.json").exists()


def test_check_flags_lighting_shift_when_brightness_changes(tmp_path: Path) -> None:
    pinned_img = _write_image(tmp_path / "dark.png", color=(20, 20, 20))
    drifted_img = _write_image(tmp_path / "bright.png", color=(220, 220, 220))
    pin_root = tmp_path / "pins"
    runner = CliRunner()

    pin_result = runner.invoke(
        cli,
        [
            "pin", "--name", "p",
            "--camera-driver", "file", "--camera", str(pinned_img),
            "--root", str(pin_root),
        ],
    )
    assert pin_result.exit_code == 0, pin_result.output

    check_result = runner.invoke(
        cli,
        [
            "check", "--against", "p",
            "--camera-driver", "file", "--camera", str(drifted_img),
            "--root", str(pin_root),
            "--json",
        ],
    )
    # Warning exit code (drift detected but not critical).
    assert check_result.exit_code in (1, 2), check_result.output
    payload = json.loads(check_result.output.strip())
    assert "lighting_shift" in payload["flags"]
    assert payload["status"] in ("warning", "failed")
    lighting_findings = [f for f in payload["findings"] if f["component"] == "lighting"]
    assert len(lighting_findings) == 1
    assert lighting_findings[0]["evidence"]["mean_luminance_delta"] > 100


def test_check_passes_on_identical_frame(tmp_path: Path) -> None:
    img = _write_image(tmp_path / "ref.png")
    pin_root = tmp_path / "pins"
    runner = CliRunner()
    assert runner.invoke(
        cli,
        ["pin", "--name", "p",
         "--camera-driver", "file", "--camera", str(img),
         "--root", str(pin_root)],
    ).exit_code == 0

    result = runner.invoke(
        cli,
        ["check", "--against", "p",
         "--camera-driver", "file", "--camera", str(img),
         "--root", str(pin_root),
         "--json"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output.strip())
    assert payload["status"] == "passed"
    assert payload["flags"] == []
    assert payload["findings"] == []


def test_check_pin_hash_matches_filesystem(tmp_path: Path) -> None:
    img = _write_image(tmp_path / "ref.png")
    pin_root = tmp_path / "pins"
    runner = CliRunner()
    assert runner.invoke(
        cli,
        ["pin", "--name", "p", "--camera-driver", "file", "--camera", str(img),
         "--root", str(pin_root)],
    ).exit_code == 0

    a = runner.invoke(
        cli,
        ["check", "--against", "p", "--camera-driver", "file", "--camera", str(img),
         "--root", str(pin_root), "--json"],
    )
    b = runner.invoke(
        cli,
        ["check", "--against", "p", "--camera-driver", "file", "--camera", str(img),
         "--root", str(pin_root), "--json"],
    )
    hash_a = json.loads(a.output.strip())["pin"]["manifest_hash"]
    hash_b = json.loads(b.output.strip())["pin"]["manifest_hash"]
    assert hash_a == hash_b


# --- guard ---------------------------------------------------------------


def test_guard_help_lists_required_options() -> None:
    result = CliRunner().invoke(cli, ["guard", "--help"])
    assert result.exit_code == 0
    for flag in ("--against", "--record-cmd", "--dataset-dir", "--on-warning"):
        assert flag in result.output


def test_guard_missing_pin_exits_with_error(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli,
        [
            "guard",
            "--against", "no-such-pin",
            "--record-cmd", "true",
            "--root", str(tmp_path / "pins"),
        ],
    )
    assert result.exit_code == 3


def test_guard_runs_record_cmd_and_writes_sidecars(tmp_path: Path) -> None:
    # Pin a static image so the pre-flight check passes.
    img_path = tmp_path / "ref.png"
    _write_image(img_path)
    pin_root = tmp_path / "pins"
    pin_result = CliRunner().invoke(
        cli,
        [
            "pin", "--name", "p",
            "--camera-driver", "file", "--camera", str(img_path),
            "--root", str(pin_root),
        ],
    )
    assert pin_result.exit_code == 0, pin_result.output

    dataset = tmp_path / "dataset"
    (dataset / "videos" / "chunk-000").mkdir(parents=True)
    # Fake record command: just create a new episode dir + a video file.
    record_cmd = (
        f"sh -c 'mkdir -p {dataset}/videos/chunk-000/episode_000001 && "
        f"touch {dataset}/videos/chunk-000/episode_000001/cam.mp4'"
    )
    result = CliRunner().invoke(
        cli,
        [
            "guard",
            "--against", "p",
            "--camera-driver", "file", "--camera", str(img_path),
            "--root", str(pin_root),
            "--dataset-dir", str(dataset),
            "--record-cmd", record_cmd,
        ],
    )
    assert result.exit_code == 0, result.output
    sidecar = dataset / "anvil" / "episode_000001.anvil.json"
    assert sidecar.exists(), f"expected sidecar at {sidecar}"
    payload = json.loads(sidecar.read_text())
    assert payload["episode_id"] == "episode_000001"
    assert payload["pin"]["name"] == "p"
    assert (dataset / "anvil" / "manifest_hash.txt").exists()


def test_guard_blocks_on_failed_by_default(tmp_path: Path) -> None:
    # Pin one image, then point the pre-flight at a wildly different image
    # to force a FAILED status. --on-failed defaults to block.
    pinned = _write_image(tmp_path / "dark.png", color=(20, 20, 20))
    drifted = _write_image(tmp_path / "bright.png", color=(220, 220, 220))
    pin_root = tmp_path / "pins"
    CliRunner().invoke(
        cli,
        [
            "pin", "--name", "p",
            "--camera-driver", "file", "--camera", str(pinned),
            "--root", str(pin_root),
        ],
    )
    dataset = tmp_path / "dataset"
    record_cmd = f"sh -c 'mkdir -p {dataset}/videos/chunk-000/episode_000001'"
    result = CliRunner().invoke(
        cli,
        [
            "guard",
            "--against", "p",
            "--camera-driver", "file", "--camera", str(drifted),
            "--root", str(pin_root),
            "--dataset-dir", str(dataset),
            "--record-cmd", record_cmd,
        ],
    )
    # Blocked → exit code is the check's status code, not the record cmd's.
    assert result.exit_code in (1, 2), result.output
    # Record command should NOT have run.
    assert not (dataset / "videos" / "chunk-000" / "episode_000001").exists()
