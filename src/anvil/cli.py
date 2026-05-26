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
from anvil.diff import render_check_diff
from anvil.errors import AnvilError, PinNotFound
from anvil.layers.layer1_fast import (
    compute_aruco_reference,
    compute_embedding_reference,
    compute_lighting_reference,
    load_aruco_reference,
    load_embedding_reference,
    load_lighting_reference,
    run_layer1,
    save_aruco_reference,
    save_embedding_reference,
    save_lighting_reference,
)
from anvil.layers.layer2_structural import (
    Layer2KeypointsOutput,
    Layer2ObjectsOutput,
    compute_keypoint_reference,
    compute_object_reference,
    load_keypoint_reference,
    load_object_references,
    run_layer2_keypoints,
    run_layer2_objects,
    save_keypoint_reference,
    save_object_reference,
)
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
from anvil.models import load_embedder
from anvil.models.keypoints import load_keypoint_pipeline
from anvil.models.objects import load_object_detector
from anvil.schema import (
    ANVIL_SCHEMA_VERSION,
    EpisodeReport,
    Finding,
    PinReference,
    Provenance,
    Scores,
    Status,
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


def _status_from_findings(findings: list[Finding]) -> Status:
    if any(f.severity == "critical" for f in findings):
        return "failed"
    if any(f.severity == "warning" for f in findings):
        return "warning"
    return "passed"


_STATUS_COLOR: dict[Status, str] = {
    "passed": "green",
    "warning": "yellow",
    "failed": "red",
}


def _render_check_summary(report: EpisodeReport, scene_drift_source: str) -> None:
    color = _STATUS_COLOR[report.status]
    _console.print(
        f"[bold {color}]{report.status.upper()}[/] — pin [bold]{report.pin.name}[/]"
    )
    scores = report.scores
    _console.print(
        f"  scene_drift:               {scores.scene_drift:.3f} "
        f"({scene_drift_source})\n"
        f"  lighting_drift:            {scores.lighting_drift:.3f}\n"
        f"  max_camera_pose_drift_deg: {scores.max_camera_pose_drift_deg:.2f}"
    )
    if report.flags:
        _console.print(f"  flags: {', '.join(report.flags)}")
    for finding in report.findings:
        sev_color = {"info": "blue", "warning": "yellow", "critical": "red"}[finding.severity]
        _console.print(
            f"  [bold {sev_color}]{finding.severity.upper()}[/] "
            f"[L{finding.layer} {finding.component}] {finding.detail}"
        )
        _console.print(f"     fix: {finding.fix}")


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
@click.option(
    "--object",
    "objects",
    multiple=True,
    help=(
        "Named object to track at check time (e.g. 'red_cube'). May be passed "
        "multiple times. Uses the name as the text prompt; for a different "
        "prompt write 'name=prompt' (e.g. 'rack=test tube rack')."
    ),
)
def pin_cmd(
    name: str,
    camera_device: str,
    camera_driver: str,
    task: str,
    robot: bool,
    pin_root: Path | None,
    force: bool,
    objects: tuple[str, ...],
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
            import shutil

            for child in pin_dir.iterdir():
                if child.is_file():
                    child.unlink()
                elif child.is_dir() and child.name != "diffs":
                    # Diff history survives a re-pin; everything else is
                    # regenerated and stale entries shouldn't persist.
                    shutil.rmtree(child)
        pin_dir.mkdir(parents=True, exist_ok=True)

        with open_camera(camera_driver, camera_device) as cam:
            frame = cam.grab()

        ref_path = pin_dir / REFERENCE_IMAGE_FILENAME
        if not cv2.imwrite(str(ref_path), frame):
            raise AnvilError(f"Failed to write reference frame to {ref_path}")

        lighting_ref = compute_lighting_reference(frame)
        save_lighting_reference(lighting_ref, pin_dir)
        aruco_ref = compute_aruco_reference(frame)
        save_aruco_reference(aruco_ref, pin_dir)

        embedder = load_embedder()
        embedder_name: str | None = None
        if embedder is not None:
            embedding_ref = compute_embedding_reference(frame, embedder)
            save_embedding_reference(embedding_ref, pin_dir)
            embedder_name = embedder.name

        keypoint_pipeline = load_keypoint_pipeline()
        keypoint_detector_name: str | None = None
        if keypoint_pipeline is not None:
            detector, _matcher = keypoint_pipeline
            keypoint_ref = compute_keypoint_reference(frame, detector)
            save_keypoint_reference(keypoint_ref, pin_dir)
            keypoint_detector_name = detector.name

        # Named-object refs (YOLO-World + DINOv3 region embedding).
        object_names: list[str] = []
        object_detector_name: str | None = None
        if objects:
            object_detector = load_object_detector()
            if object_detector is None:
                _console.print(
                    "[yellow]![/] --object was passed but no detector loaded; "
                    "install the [full] extra. Continuing without object refs."
                )
            else:
                object_detector_name = object_detector.name
                for raw in objects:
                    if "=" in raw:
                        obj_name, _, prompt = raw.partition("=")
                    else:
                        obj_name, prompt = raw, raw.replace("_", " ")
                    ref = compute_object_reference(
                        frame, obj_name, prompt, object_detector, embedder
                    )
                    if ref is None:
                        _console.print(
                            f"[yellow]![/] {obj_name!r} (prompt {prompt!r}) "
                            "not detected in pin frame; skipping."
                        )
                        continue
                    save_object_reference(ref, pin_dir)
                    object_names.append(obj_name)

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
            aruco_present=bool(aruco_ref.markers),
            embedding_present=embedder_name is not None,
            embedder_name=embedder_name,
            keypoints_present=keypoint_detector_name is not None,
            keypoint_detector_name=keypoint_detector_name,
            objects=object_names,
        )
        save_manifest(manifest, pin_dir)
        manifest_hash = compute_manifest_hash(pin_dir)

        marker_ids = [m.id for m in aruco_ref.markers]
        _console.print(f"[bold green]✓[/] Pinned [bold]{name}[/] at {pin_dir}")
        _console.print(f"  reference:     {ref_path}")
        _console.print(f"  resolution:    {width}x{height}")
        _console.print(f"  robot:         {'enabled' if robot else 'disabled'}")
        _console.print(
            f"  lighting:      mean lum={lighting_ref.mean_luminance:.1f}, "
            f"CCT={lighting_ref.cct_kelvin:.0f}K"
        )
        _console.print(
            f"  aruco:         {len(marker_ids)} marker(s)"
            + (f" {marker_ids}" if marker_ids else "")
        )
        _console.print(
            f"  embedder:      {embedder_name if embedder_name else 'histogram (fallback)'}"
        )
        keypoints_line = (
            f"{keypoint_detector_name} ({len(keypoint_ref.keypoints)} pts)"
            if keypoint_detector_name is not None
            else "disabled (no pipeline loaded)"
        )
        _console.print(f"  keypoints:     {keypoints_line}")
        if objects:
            objects_line = (
                f"{object_detector_name} — {len(object_names)}/{len(objects)} "
                "located"
                if object_detector_name is not None
                else "no detector loaded"
            )
            _console.print(f"  objects:       {objects_line} {object_names}")
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
        lighting_ref = load_lighting_reference(pin_dir)
        aruco_ref = load_aruco_reference(pin_dir)
        embedding_ref = load_embedding_reference(pin_dir)
        embedder = load_embedder() if embedding_ref is not None else None
        keypoint_ref = load_keypoint_reference(pin_dir)
        keypoint_pipeline = (
            load_keypoint_pipeline() if keypoint_ref is not None else None
        )
        object_refs = load_object_references(pin_dir)
        object_detector = load_object_detector() if object_refs else None

        driver = camera_driver or manifest.camera.driver
        device = camera_device or manifest.camera.device
        with open_camera(driver, device) as cam:
            frame = cam.grab()

        layer1 = run_layer1(
            current_frame=frame,
            lighting_ref=lighting_ref,
            aruco_ref=aruco_ref,
            thresholds=manifest.thresholds,
            embedding_ref=embedding_ref,
            embedder=embedder,
        )

        layer2: Layer2KeypointsOutput | None = None
        if keypoint_ref is not None and keypoint_pipeline is not None:
            kp_detector, kp_matcher = keypoint_pipeline
            layer2 = run_layer2_keypoints(
                current_frame=frame,
                keypoint_ref=keypoint_ref,
                detector=kp_detector,
                matcher=kp_matcher,
                thresholds=manifest.thresholds,
            )

        layer2_objects: Layer2ObjectsOutput | None = None
        if object_refs and object_detector is not None:
            layer2_objects = run_layer2_objects(
                current_frame=frame,
                object_refs=object_refs,
                detector=object_detector,
                embedder=embedder,
                thresholds=manifest.thresholds,
            )

        # Merge findings + flags from all layers; Layer 2 keypoints wins
        # the pose score when it ran successfully (more accurate than ArUco).
        merged_findings = list(layer1.findings)
        merged_flags = list(layer1.flags)
        pose_drift_score = layer1.max_camera_pose_drift_deg
        if layer2 is not None:
            merged_findings.extend(layer2.findings)
            for flag in layer2.flags:
                if flag not in merged_flags:
                    merged_flags.append(flag)
            if layer2.num_matches >= 10:
                pose_drift_score = layer2.abs_rotation_deg
        if layer2_objects is not None:
            merged_findings.extend(layer2_objects.findings)
            for flag in layer2_objects.flags:
                if flag not in merged_flags:
                    merged_flags.append(flag)

        status = _status_from_findings(merged_findings)
        report = EpisodeReport(
            anvil_schema_version=ANVIL_SCHEMA_VERSION,
            episode_id=episode_id,
            checked_at=_utcnow(),
            pin=PinReference(
                name=manifest.name,
                manifest_hash=manifest_hash,
                pinned_at=manifest.pinned_at,
            ),
            status=status,
            operator_action="proceeded",
            scores=Scores(
                scene_drift=layer1.scene_drift,
                lighting_drift=layer1.lighting_drift,
                max_object_drift_cm=0.0,
                max_camera_pose_drift_deg=pose_drift_score,
                max_robot_joint_drift_deg=0.0,
            ),
            flags=merged_flags,
            findings=merged_findings,
            provenance=_provenance(),
        )

        diff_path: Path | None = None
        if report.status != "passed":
            diff_path = render_check_diff(
                current_frame=frame,
                layer1=layer1,
                aruco_ref=aruco_ref,
                pin_dir=pin_dir,
                pin_name=manifest.name,
                threshold_deg=manifest.thresholds.max_camera_pose_drift_deg,
            )

        if json_only:
            click.echo(report.model_dump_json())
        else:
            _render_check_summary(report, layer1.scene_drift_source)
            if diff_path is not None:
                _console.print(f"  diff:          {diff_path}")

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
