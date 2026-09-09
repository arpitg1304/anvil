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
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import click
import cv2
from rich.console import Console

from anvil import __version__
from anvil.cameras import open_camera
from anvil.cameras.base import Frame
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
    compute_object_references,
    load_keypoint_reference,
    load_object_references,
    run_layer2_keypoints,
    run_layer2_objects,
    save_keypoint_reference,
    save_object_reference,
)
from anvil.manifest import (
    DEFAULT_PIN_ROOT,
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
from anvil.models.objects import DetectedObject, load_object_detector
from anvil.schema import (
    ANVIL_SCHEMA_VERSION,
    EpisodeReport,
    Finding,
    PinReference,
    Provenance,
    Scores,
    Status,
)
from anvil.sidecar import (
    anvil_dir,
    snapshot_episodes,
    write_session_manifest_hash,
    write_sidecars,
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
@click.option(
    "--resolution",
    default="1920x1080",
    show_default=True,
    help=(
        "Capture resolution as WIDTHxHEIGHT (webcam driver only). The pin "
        "records whatever the camera actually delivers; every future check "
        "is forced to match it. Pass e.g. '1280x720' for lower-res rigs."
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
    resolution: str,
) -> None:
    """Capture the current scene as a reference manifest."""
    try:
        requested_resolution = _parse_resolution(resolution)
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

        with open_camera(
            camera_driver, camera_device, resolution=requested_resolution
        ) as cam:
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
                object_pairs: list[tuple[str, str]] = []
                for raw in objects:
                    if "=" in raw:
                        obj_name, _, prompt = raw.partition("=")
                    else:
                        obj_name, prompt = raw, raw.replace("_", " ")
                    object_pairs.append((obj_name, prompt))
                # Single detector pass for all named objects — see
                # compute_object_references for the perf + correctness rationale.
                results = compute_object_references(
                    frame, object_pairs, object_detector, embedder
                )
                for obj_name, ref in results:
                    if ref is None:
                        prompt = dict(object_pairs)[obj_name]
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
@click.option(
    "--save",
    "save_record",
    is_flag=True,
    default=False,
    help=(
        "Keep this check so you can look at it later: the report, the frame, "
        "and a picture of what changed. See them with `anvil viewer`."
    ),
)
def check_cmd(
    pin_name: str,
    camera_device: str | None,
    camera_driver: str | None,
    pin_root: Path | None,
    episode_id: str,
    json_only: bool,
    save_record: bool,
) -> None:
    """Run a one-shot diff between the live scene and a pinned reference."""
    try:
        result = _run_check(
            pin_name,
            camera_device=camera_device,
            camera_driver=camera_driver,
            pin_root=pin_root,
            episode_id=episode_id,
        )
        saved: Path | None = None
        if save_record:
            saved = save_check_record(
                result.pin_dir, result.report, result.frame, result.diff_path
            )
        if json_only:
            click.echo(result.report.model_dump_json())
        else:
            _render_check_summary(result.report, result.scene_drift_source)
            if result.diff_path is not None:
                _console.print(f"  diff:          {result.diff_path}")
            if saved is not None:
                _console.print(f"  saved:         {saved}")
        sys.exit(_EXIT_FOR_STATUS[result.report.status])
    except PinNotFound as exc:
        _console.print(f"[bold red]✗[/] {exc}")
        sys.exit(_EXIT_ERROR)
    except AnvilError as exc:
        _console.print(f"[bold red]✗[/] {exc}")
        sys.exit(_EXIT_ERROR)


@dataclass
class _CheckResult:
    """Bundle of state from one ``_run_check`` invocation.

    ``check_cmd`` only needs the report; ``guard_cmd`` reuses the same
    object to drive its proceed-or-block decision and to write per-episode
    sidecars without rerunning the pipeline.
    """

    report: EpisodeReport
    diff_path: Path | None
    scene_drift_source: str
    pin_dir: Path
    manifest: Manifest
    frame: Frame


_EXIT_FOR_STATUS: dict[Status, int] = {
    "passed": _EXIT_PASSED,
    "warning": _EXIT_WARNING,
    "failed": _EXIT_FAILED,
}


def _parse_resolution(value: str) -> tuple[int, int] | None:
    """Parse a 'WIDTHxHEIGHT' string into (width, height).

    Returns ``None`` for an empty string (caller lets the camera pick).
    Raises ``click.BadParameter`` on a malformed value so the user gets a
    clear message instead of a downstream crash.
    """
    if not value.strip():
        return None
    parts = value.lower().split("x")
    if len(parts) != 2 or not all(p.strip().isdigit() for p in parts):
        raise click.BadParameter(
            f"--resolution must be WIDTHxHEIGHT (e.g. 1920x1080), got {value!r}"
        )
    return (int(parts[0]), int(parts[1]))


def _match_pin_resolution(
    frame: Frame,
    pin_resolution: tuple[int, int] | None,
) -> Frame:
    """Force ``frame`` to the pin's (width, height), warning if it differs.

    Cameras don't always honor a requested resolution — a webcam that came
    up at 640x480 this session can't be coerced to the 1920x1080 the pin
    was captured at. Every downstream comparison (ArUco positions,
    keypoints, object bboxes) assumes a shared pixel space, so we resize
    the live frame to the pin's resolution rather than silently producing
    garbage drift numbers (or crashing the side-by-side diff render).
    """
    if pin_resolution is None:
        return frame
    target_w, target_h = pin_resolution
    h, w = frame.shape[:2]
    if (w, h) == (target_w, target_h):
        return frame
    _console.print(
        f"[yellow]![/] Camera returned {w}x{h} but the pin is "
        f"{target_w}x{target_h}; resizing to match. Drift numbers are less "
        "reliable across a resolution change — prefer a camera that can "
        "deliver the pinned resolution, or repin at the current one."
    )
    return cast(
        Frame, cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
    )


CHECKS_SUBDIR = "checks"


_UNSAFE_IN_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


def _record_dir(pin_dir: Path, report: EpisodeReport) -> Path:
    """A fresh directory for one check: ``<pin>/checks/<stamp>_<episode>``.

    ``episode_id`` is operator-supplied and unconstrained by the schema, so it
    is reduced to one safe path segment — otherwise a value containing ``/``
    or ``..`` would write outside ``checks/``. A suffix is added rather than
    reusing a directory, so two checks in the same second keep both records
    instead of the second overwriting the first.
    """
    stamp = report.checked_at.strftime("%Y%m%dT%H%M%SZ")
    episode = _UNSAFE_IN_NAME.sub("_", report.episode_id).strip("._") or "check"
    base = pin_dir / CHECKS_SUBDIR
    for suffix in ("", *(f"-{n}" for n in range(2, 100))):
        candidate = base / f"{stamp}_{episode}{suffix}"
        if not candidate.exists():
            return candidate
    raise AnvilError(f"Too many checks saved this second under {base}")


def save_check_record(
    pin_dir: Path,
    report: EpisodeReport,
    frame: Frame,
    diff_path: Path | None = None,
) -> Path:
    """Persist one check under ``<pin_dir>/checks/<timestamp>_<episode>/``.

    Writes the report, the frame it was taken from, a difference heat map
    against the pinned reference, and anvil's annotated diff when one was
    rendered (only a non-passing check gets one). ``check`` is otherwise
    fire-and-forget: it prints a verdict and exits, leaving nothing to review
    or compare later. Returns the directory written.
    """
    import cv2
    import numpy as np

    out = _record_dir(pin_dir, report)
    try:
        out.mkdir(parents=True)
        (out / "report.json").write_text(
            report.model_dump_json(indent=2), encoding="utf-8"
        )
        wrote = cv2.imwrite(str(out / "frame.png"), frame)
    except OSError as exc:
        # A failed save is reported like any other anvil error, not as a
        # traceback — the check itself already ran and printed its verdict.
        raise AnvilError(f"Could not save the check record under {out}: {exc}") from exc
    if not wrote:
        raise AnvilError(f"Could not write the check frame to {out}")

    reference = pin_dir / REFERENCE_IMAGE_FILENAME
    if reference.exists():
        ref = cv2.imread(str(reference))
        if ref is not None:
            if ref.shape[:2] != frame.shape[:2]:
                ref = cv2.resize(ref, (frame.shape[1], frame.shape[0]))
            delta = cv2.absdiff(ref, frame).max(axis=2).astype(np.uint8)
            heat = cv2.applyColorMap(delta, cv2.COLORMAP_JET)
            # keep the scene readable underneath the glow
            cv2.imwrite(str(out / "heat.png"), cv2.addWeighted(heat, 0.75, frame, 0.25, 0))

    if diff_path is not None and diff_path.is_file():
        shutil.copyfile(diff_path, out / "diff.png")
    return out


def _run_check(
    pin_name: str,
    *,
    camera_device: str | None = None,
    camera_driver: str | None = None,
    pin_root: Path | None = None,
    episode_id: str = "manual_check",
) -> _CheckResult:
    """Run the full check pipeline and return everything callers might need.

    Writes an annotated diff PNG into ``<pin_dir>/diffs/`` when the report
    status is not ``passed`` — same behavior the CLI's ``check`` had
    inline before this was factored out.
    """
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
    pin_resolution = manifest.camera.resolution
    with open_camera(driver, device, resolution=pin_resolution) as cam:
        frame = cam.grab()
    frame = _match_pin_resolution(frame, pin_resolution)

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
        current_object_detections: list[DetectedObject] = []
        if object_refs and object_detector is not None:
            current_object_detections = object_detector.detect(
                frame, list({ref.prompt for ref in object_refs})
            )
        diff_path = render_check_diff(
            current_frame=frame,
            layer1=layer1,
            aruco_ref=aruco_ref,
            pin_dir=pin_dir,
            pin_name=manifest.name,
            threshold_deg=manifest.thresholds.max_camera_pose_drift_deg,
            object_refs=object_refs,
            current_detections=current_object_detections,
        )

    return _CheckResult(
        report=report,
        diff_path=diff_path,
        scene_drift_source=layer1.scene_drift_source,
        pin_dir=pin_dir,
        manifest=manifest,
        frame=frame,
    )


@cli.command("viewer")
@click.option(
    "--root",
    "pin_root",
    type=click.Path(path_type=Path),
    default=None,
    help="Where the pins live. Defaults to ./.anvil here.",
)
@click.option(
    "--checks",
    "checks_root",
    type=click.Path(path_type=Path),
    default=None,
    help=(
        "Extra directory of check records to include, laid out as "
        "<checks>/<pin>/<id>/report.json. Only needed if something other than "
        "`anvil check --save` wrote them."
    ),
)
@click.option("--host", default="127.0.0.1", show_default=True,
              help="Interface to bind. Localhost only by default.")
@click.option("--port", type=int, default=7788, show_default=True, help="Port to serve on.")
@click.option("--no-browser", is_flag=True, default=False,
              help="Don't auto-open a browser tab.")
def viewer_cmd(
    pin_root: Path | None,
    checks_root: Path | None,
    host: str,
    port: int,
    no_browser: bool,
) -> None:
    """Look at saved checks in a browser: the frames, side by side."""
    import webbrowser

    import uvicorn

    from anvil.server.viewer import create_app

    root = (pin_root or Path.cwd() / DEFAULT_PIN_ROOT).resolve()
    if not root.exists():
        _console.print(
            f"[bold red]✗[/] No pin directory at {root}. Run `anvil pin --name ...` "
            "first, or pass --root."
        )
        sys.exit(_EXIT_ERROR)

    app = create_app(root, checks_root)
    url = f"http://{host}:{port}"
    _console.print(f"[bold blue]►[/] Anvil viewer serving from {root}\n  → {url}")
    # Count both places records can live, so --checks is not reported as empty.
    n = sum(1 for _ in root.glob("*/checks/*/report.json"))
    if checks_root is not None:
        n += sum(1 for _ in checks_root.glob("*/*/report.json"))
    if n:
        _console.print(f"  [dim]{n} saved check(s)[/]")
    else:
        _console.print("  [dim]no saved checks yet — run `anvil check --save`[/]")
    if not no_browser:
        webbrowser.open(url)
    uvicorn.run(app, host=host, port=port, log_level="warning")


@cli.command("guard")
@click.option(
    "--against",
    "pin_name",
    required=True,
    help="Name of the pin to check against.",
)
@click.option(
    "--record-cmd",
    required=True,
    help="Recording command to wrap (e.g. 'lerobot record --task ...').",
)
@click.option(
    "--dataset-dir",
    type=click.Path(path_type=Path),
    default=None,
    help=(
        "LeRobot-style dataset directory to watch for new episodes. "
        "When set, anvil writes an EpisodeReport JSON under <dir>/anvil/ "
        "for each episode the record command creates."
    ),
)
@click.option(
    "--on-warning",
    type=click.Choice(["prompt", "tag", "block"]),
    default="prompt",
    show_default=True,
    help=(
        "What to do if the pre-flight check returns WARNING. "
        "'prompt' asks the operator interactively; 'tag' proceeds and "
        "stamps the deviation onto each episode's sidecar; 'block' aborts "
        "the record."
    ),
)
@click.option(
    "--on-failed",
    type=click.Choice(["block", "prompt", "proceed"]),
    default="block",
    show_default=True,
    help="What to do if the pre-flight check returns FAILED. Defaults to block.",
)
@click.option(
    "--camera",
    "camera_device",
    default=None,
    help="Override camera device for the pre-flight check (uses pin's camera if omitted).",
)
@click.option(
    "--camera-driver",
    type=click.Choice(["webcam", "file"]),
    default=None,
    help="Override camera driver for the pre-flight check.",
)
@click.option(
    "--root",
    "pin_root",
    type=click.Path(path_type=Path),
    default=None,
    help="Override pin storage root.",
)
@click.option(
    "--save",
    "save_record",
    is_flag=True,
    default=False,
    help=(
        "Keep the pre-flight check so you can look at it later — the same "
        "record `anvil check --save` writes. See it with `anvil viewer`."
    ),
)
def guard_cmd(
    pin_name: str,
    record_cmd: str,
    save_record: bool,
    dataset_dir: Path | None,
    on_warning: str,
    on_failed: str,
    camera_device: str | None,
    camera_driver: str | None,
    pin_root: Path | None,
) -> None:
    """Wrap a recording command with a pre-episode check.

    Workflow per invocation:

    1. Run ``check`` against ``--against`` as a pre-flight.
    2. Decide proceed/block from the report status and the ``--on-warning``
       / ``--on-failed`` policies.
    3. If proceeding, snapshot the dataset directory (if any), run the
       record command as a subprocess, then write per-episode sidecars
       for whatever new ``episode_NNNNNN/`` directories appeared.

    Exit code is the record command's own exit code on success, or the
    standard anvil exit code (0/1/2/3) when guard blocks before running.
    """
    import shlex
    import subprocess

    try:
        result = _run_check(
            pin_name,
            camera_device=camera_device,
            camera_driver=camera_driver,
            pin_root=pin_root,
            episode_id="guard_preflight",
        )
        _render_check_summary(result.report, result.scene_drift_source)
        if result.diff_path is not None:
            _console.print(f"  diff:          {result.diff_path}")
        if save_record:
            saved = save_check_record(
                result.pin_dir, result.report, result.frame, result.diff_path
            )
            _console.print(f"  saved:         {saved}")

        proceed = _decide_proceed(result.report.status, on_warning, on_failed)
        if not proceed:
            _console.print(
                "[bold red]✗[/] Aborting recording — re-pin or re-stage the rig "
                "and try again."
            )
            sys.exit(_EXIT_FOR_STATUS[result.report.status])

        before = (
            snapshot_episodes(dataset_dir) if dataset_dir is not None else set()
        )
        _console.print(f"[bold blue]►[/] Running: {record_cmd}")
        try:
            completed = subprocess.run(shlex.split(record_cmd), check=False)
        except FileNotFoundError as exc:
            _console.print(f"[bold red]✗[/] Record command not found: {exc}")
            sys.exit(_EXIT_ERROR)

        if dataset_dir is not None:
            new_eps = sorted(snapshot_episodes(dataset_dir) - before)
            if new_eps:
                written = write_sidecars(dataset_dir, new_eps, result.report)
                write_session_manifest_hash(
                    dataset_dir, result.report.pin.manifest_hash
                )
                _console.print(
                    f"  sidecars:     {len(written)} written under "
                    f"{anvil_dir(dataset_dir)}"
                )
            else:
                _console.print(
                    "  sidecars:     no new episode_NNNNNN/ directories detected"
                )

        sys.exit(completed.returncode)
    except PinNotFound as exc:
        _console.print(f"[bold red]✗[/] {exc}")
        sys.exit(_EXIT_ERROR)
    except AnvilError as exc:
        _console.print(f"[bold red]✗[/] {exc}")
        sys.exit(_EXIT_ERROR)


def _decide_proceed(
    status: Status, on_warning: str, on_failed: str
) -> bool:
    """Decide whether to proceed with the record command given the check status."""
    if status == "passed":
        return True
    if status == "warning":
        if on_warning == "tag":
            return True
        if on_warning == "block":
            return False
        return click.confirm(
            "WARNING: drift detected. Proceed with recording?", default=False
        )
    # status == "failed"
    if on_failed == "proceed":
        return True
    if on_failed == "prompt":
        return click.confirm(
            "FAILED: critical drift detected. Proceed anyway?", default=False
        )
    return False


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
