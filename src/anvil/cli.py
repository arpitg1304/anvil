"""Anvil command-line entry point.

Three top-level verbs are locked in v0.1: ``pin``, ``check``, ``guard``.

Exit codes (``check`` and ``guard`` only):
    0  passed
    1  warning  (drift detected, operator can choose to proceed)
    2  failed   (must abort)
    3  pin not found / version mismatch / system error
"""

from __future__ import annotations

import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import click
import cv2
from rich.console import Console

from anvil import __version__
from anvil.cameras import open_camera
from anvil.errors import AnvilError, PinNotFound
from anvil.manifest import (
    REFERENCE_IMAGE_FILENAME,
    CameraDriver,
    CameraSpec,
    Manifest,
    RobotSpec,
    ThresholdSpec,
    compute_manifest_hash,
    load_manifest,
    pin_directory,
    save_manifest,
)
from anvil.schema import (
    ANVIL_SCHEMA_VERSION,
    EpisodeReport,
    PinReference,
    Provenance,
    Scores,
)

_console = Console(stderr=True)

_EXIT_PASSED = 0
_EXIT_WARNING = 1
_EXIT_FAILED = 2
_EXIT_ERROR = 3


def _utcnow() -> datetime:
    return datetime.now(tz=UTC)


def _provenance() -> Provenance:
    hardware = {"platform": platform.platform(), "machine": platform.machine()}
    return Provenance(anvil_version=__version__, hardware=hardware)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="anvil")
def cli() -> None:
    """Anvil — pin a scene; verify every future session against it."""


@cli.command("pin")
@click.option("--name", required=True, help="Name for this pin (e.g. pick_red_cube).")
@click.option(
    "--camera",
    "camera_device",
    default="0",
    show_default=True,
    help=(
        "Camera index, /dev/video* path, RTSP URL, or image path "
        "(when using --camera-driver file)."
    ),
)
@click.option(
    "--camera-driver",
    type=click.Choice(["webcam", "file"]),
    default="webcam",
    show_default=True,
    help="Camera driver. Use 'file' for a fixed image (testing / replay).",
)
@click.option("--task", default="", help="Free-form description of what gets demonstrated.")
@click.option(
    "--robot/--no-robot",
    default=False,
    show_default=True,
    help="Mark the pin as robot-enabled (Layer 0 checks). Implementation lands later in Week 1.",
)
@click.option(
    "--root",
    "pin_root",
    type=click.Path(path_type=Path),
    default=None,
    help="Override pin storage root (default ./.anvil).",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Overwrite an existing pin with the same name.",
)
def pin_cmd(
    name: str,
    camera_device: str,
    camera_driver: str,
    task: str,
    robot: bool,
    pin_root: Path | None,
    force: bool,
) -> None:
    """Capture the current scene as a reference manifest."""
    try:
        pin_dir = pin_directory(name, root=pin_root)
        if pin_dir.exists() and any(pin_dir.iterdir()) and not force:
            raise click.ClickException(
                f"Pin already exists at {pin_dir}. "
                "Use --force to overwrite or pick a different --name."
            )
        if pin_dir.exists() and force:
            for child in pin_dir.iterdir():
                if child.is_file():
                    child.unlink()
        pin_dir.mkdir(parents=True, exist_ok=True)

        with open_camera(camera_driver, camera_device) as cam:
            frame = cam.grab()

        ref_path = pin_dir / REFERENCE_IMAGE_FILENAME
        if not cv2.imwrite(str(ref_path), frame):
            raise AnvilError(f"Failed to write reference frame to {ref_path}")

        height, width = frame.shape[:2]
        manifest = Manifest(
            anvil_schema_version=ANVIL_SCHEMA_VERSION,
            name=name,
            task=task,
            pinned_at=_utcnow(),
            camera=CameraSpec(
                driver=cast(CameraDriver, camera_driver),
                device=camera_device,
                resolution=(width, height),
            ),
            robot=RobotSpec(enabled=robot),
            thresholds=ThresholdSpec(),
        )
        save_manifest(manifest, pin_dir)
        manifest_hash = compute_manifest_hash(pin_dir)

        _console.print(f"[bold green]✓[/] Pinned [bold]{name}[/] at {pin_dir}")
        _console.print(f"  reference:    {ref_path}")
        _console.print(f"  resolution:   {width}x{height}")
        _console.print(f"  robot:        {'enabled' if robot else 'disabled'}")
        _console.print(f"  manifest_hash: {manifest_hash}")
    except AnvilError as exc:
        _console.print(f"[bold red]✗[/] {exc}")
        raise click.ClickException(str(exc)) from exc


@cli.command("check")
@click.option(
    "--against",
    "pin_name",
    required=True,
    help="Name of the pin to check against.",
)
@click.option(
    "--camera",
    "camera_device",
    default=None,
    help="Override camera device for this check (uses pin's camera if omitted).",
)
@click.option(
    "--camera-driver",
    type=click.Choice(["webcam", "file"]),
    default=None,
    help="Override camera driver for this check.",
)
@click.option(
    "--root",
    "pin_root",
    type=click.Path(path_type=Path),
    default=None,
    help="Override pin storage root.",
)
@click.option(
    "--episode-id",
    default="manual_check",
    show_default=True,
    help="Episode id stamped onto the sidecar report.",
)
@click.option(
    "--json",
    "json_only",
    is_flag=True,
    default=False,
    help="Emit only the sidecar JSON on stdout (no Rich output).",
)
def check_cmd(
    pin_name: str,
    camera_device: str | None,
    camera_driver: str | None,
    pin_root: Path | None,
    episode_id: str,
    json_only: bool,
) -> None:
    """Run a one-shot diff between the live scene and a pinned reference."""
    try:
        pin_dir = pin_directory(pin_name, root=pin_root)
        manifest = load_manifest(pin_dir)
        manifest_hash = compute_manifest_hash(pin_dir)

        # Open the camera. CLI flags override the pinned device if provided.
        driver = camera_driver or manifest.camera.driver
        device = camera_device or manifest.camera.device
        with open_camera(driver, device) as cam:
            _ = cam.grab()  # frame is used by Layer 1; not yet wired

        # Placeholder report: vision layers land in subsequent Week 1 commits.
        # The schema and pipe wiring are real; the scores are stub zeros.
        report = EpisodeReport(
            anvil_schema_version=ANVIL_SCHEMA_VERSION,
            episode_id=episode_id,
            checked_at=_utcnow(),
            pin=PinReference(
                name=manifest.name,
                manifest_hash=manifest_hash,
                pinned_at=manifest.pinned_at,
            ),
            status="passed",
            operator_action="proceeded",
            scores=Scores(
                scene_drift=0.0,
                lighting_drift=0.0,
                max_object_drift_cm=0.0,
                max_camera_pose_drift_deg=0.0,
                max_robot_joint_drift_deg=0.0,
            ),
            flags=[],
            findings=[],
            provenance=_provenance(),
        )

        if json_only:
            click.echo(report.model_dump_json())
        else:
            _console.print(f"[bold green]✓[/] Pin loaded: [bold]{manifest.name}[/]")
            _console.print(f"  status:  [green]{report.status}[/]")
            _console.print(f"  scores:  {report.scores.model_dump()}")
            _console.print("[yellow]Layer 1 vision not yet wired — placeholder report.[/]")

        sys.exit(
            {
                "passed": _EXIT_PASSED,
                "warning": _EXIT_WARNING,
                "failed": _EXIT_FAILED,
            }[report.status]
        )
    except PinNotFound as exc:
        _console.print(f"[bold red]✗[/] {exc}")
        sys.exit(_EXIT_ERROR)
    except AnvilError as exc:
        _console.print(f"[bold red]✗[/] {exc}")
        sys.exit(_EXIT_ERROR)


@cli.command("guard")
@click.option("--against", "pin_name", required=True)
@click.option("--record-cmd", required=True, help="Recording command to wrap.")
def guard_cmd(pin_name: str, record_cmd: str) -> None:
    """Wrap a recording command with a pre-episode check (Week 3)."""
    raise click.ClickException(
        "guard is not yet implemented — landing in Week 3 alongside LeRobot sidecar writing."
    )


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
