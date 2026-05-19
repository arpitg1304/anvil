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
from click.testing import CliRunner

from anvil.cli import cli
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


def test_guard_is_explicit_not_implemented() -> None:
    result = CliRunner().invoke(
        cli,
        ["guard", "--against", "p", "--record-cmd", "echo hi"],
    )
    assert result.exit_code != 0
    assert "week 3" in result.output.lower()
