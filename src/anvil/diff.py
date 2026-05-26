"""Annotated diff images for ``check`` warning/failed runs.

When ``anvil check`` flags drift, this module writes a side-by-side PNG of
the pinned reference and the live frame with per-marker rotation deltas and
a header strip summarizing the Layer 1 scores. Saved under
``<pin_dir>/diffs/`` and deliberately excluded from ``manifest_hash`` so
recording a diff never invalidates the pin.

The renderer is composed as a sequence of independent annotators
(``_annotate_pose``, ``_annotate_header``, …) so Layer 2 (SAM3 masks,
per-object IoU) and Layer 3 (VLM caption) annotations slot in later as
new functions appended to the pipeline. No annotator should depend on
another; each consumes the canvas + its own inputs and returns the canvas.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import cv2
import numpy as np

from anvil.cameras.base import Frame
from anvil.layers.layer1_fast import ArucoReference, Layer1Output
from anvil.manifest import DIFFS_SUBDIR, REFERENCE_IMAGE_FILENAME

_HEADER_HEIGHT = 80
_HEADER_BG = (24, 24, 24)
_TEXT_COLOR = (255, 255, 255)
_LABEL_BG = (24, 24, 24)
_REF_PANEL_TAG = "REFERENCE"
_CUR_PANEL_TAG = "CURRENT"

# Per-marker outline colors. Stable picks (BGR).
_MARKER_OK_COLOR = (60, 220, 60)       # green
_MARKER_DRIFT_COLOR = (60, 60, 255)    # red — current marker exceeded threshold
_REF_OUTLINE_COLOR = (200, 200, 50)    # cyan-yellow, used on reference panel


def diff_dir(pin_dir: Path) -> Path:
    return pin_dir / DIFFS_SUBDIR


def render_check_diff(
    *,
    current_frame: Frame,
    layer1: Layer1Output,
    aruco_ref: ArucoReference,
    pin_dir: Path,
    pin_name: str,
    threshold_deg: float,
    timestamp: datetime | None = None,
) -> Path:
    """Write an annotated side-by-side PNG into ``<pin_dir>/diffs/``.

    Returns the path written to. Caller is responsible for deciding *when* to
    call this (the CLI only invokes it on non-passed checks).
    """
    ts = timestamp or datetime.now(tz=UTC)
    reference_frame = _load_reference(pin_dir)
    ref_corners = _detect_corners(reference_frame, aruco_ref.dictionary)
    cur_corners = _detect_corners(current_frame, aruco_ref.dictionary)

    # Per-marker rotation deltas (current minus pinned), keyed by marker id.
    # Reference orientation comes from aruco_ref (pinned); current orientation
    # is derived from the corners we already detected — no second detection.
    deltas = _per_marker_deltas(aruco_ref, cur_corners)

    ref_panel = _annotate_pose(
        reference_frame.copy(),
        ref_corners,
        deltas={},  # reference has no delta against itself
        threshold_deg=threshold_deg,
        outline_color=_REF_OUTLINE_COLOR,
    )
    cur_panel = _annotate_pose(
        current_frame.copy(),
        cur_corners,
        deltas=deltas,
        threshold_deg=threshold_deg,
        outline_color=_MARKER_OK_COLOR,
    )
    ref_panel = _stamp_panel_tag(ref_panel, _REF_PANEL_TAG)
    cur_panel = _stamp_panel_tag(cur_panel, _CUR_PANEL_TAG)

    body = cast(Frame, cv2.hconcat([ref_panel, cur_panel]))
    canvas = _prepend_header(body, layer1=layer1, pin_name=pin_name)

    out_dir = diff_dir(pin_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"diff_{ts.strftime('%Y%m%dT%H%M%SZ')}.png"
    out_path = out_dir / filename
    if not cv2.imwrite(str(out_path), canvas):
        raise OSError(f"Failed to write diff image to {out_path}")
    return out_path


# --- helpers --------------------------------------------------------------


def _load_reference(pin_dir: Path) -> Frame:
    path = pin_dir / REFERENCE_IMAGE_FILENAME
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Reference image missing or unreadable: {path}")
    return cast(Frame, img)


def _detect_corners(
    frame: Frame, dictionary_name: str
) -> dict[int, np.ndarray]:
    """Return ``{marker_id: 4x2 corner array}`` for the markers in ``frame``.

    Mirrors the subpixel-refined detection used by ``layer1_fast`` so corner
    coordinates match the values that drove the actual drift decision.
    """
    dict_id = int(getattr(cv2.aruco, dictionary_name))
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    corners, ids, _ = detector.detectMarkers(frame)
    if ids is None or len(ids) == 0:
        return {}
    out: dict[int, np.ndarray] = {}
    for marker_corners, marker_id in zip(corners, ids.flatten(), strict=False):
        out[int(marker_id)] = np.asarray(marker_corners, dtype=np.float32).reshape(4, 2)
    return out


def _orientation_from_corners(quad: np.ndarray) -> float:
    """Angle of the top edge (corner 0 → corner 1) in degrees."""
    dy = float(quad[1, 1] - quad[0, 1])
    dx = float(quad[1, 0] - quad[0, 0])
    return float(np.degrees(np.arctan2(dy, dx)))


def _per_marker_deltas(
    aruco_ref: ArucoReference, cur_corners: dict[int, np.ndarray]
) -> dict[int, float]:
    """Per-marker orientation delta (current minus pinned), wrapped to [-180, 180]."""
    deltas: dict[int, float] = {}
    for ref_m in aruco_ref.markers:
        quad = cur_corners.get(ref_m.id)
        if quad is None:
            continue
        cur_orientation = _orientation_from_corners(quad)
        raw = cur_orientation - ref_m.orientation_deg
        normalized = ((raw + 180.0) % 360.0) - 180.0
        deltas[ref_m.id] = float(normalized)
    return deltas


def _annotate_pose(
    panel: Frame,
    corners_by_id: dict[int, np.ndarray],
    deltas: dict[int, float],
    threshold_deg: float,
    outline_color: tuple[int, int, int],
) -> Frame:
    for marker_id, quad in corners_by_id.items():
        pts = quad.reshape(-1, 1, 2).astype(np.int32)
        delta = deltas.get(marker_id)
        color = (
            _MARKER_DRIFT_COLOR
            if delta is not None and abs(delta) > threshold_deg
            else outline_color
        )
        cv2.polylines(panel, [pts], isClosed=True, color=color, thickness=3)
        cx, cy = quad.mean(axis=0)
        label = f"ID {marker_id}"
        if delta is not None:
            label += f"  Δ{delta:+.2f}°"
        _draw_label(panel, label, (int(cx) - 80, int(cy) - 12))
    return panel


def _stamp_panel_tag(panel: Frame, tag: str) -> Frame:
    pad = 12
    text_size, _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    box_w = text_size[0] + pad * 2
    box_h = text_size[1] + pad * 2
    cv2.rectangle(panel, (0, 0), (box_w, box_h), _LABEL_BG, -1)
    cv2.putText(
        panel,
        tag,
        (pad, pad + text_size[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        _TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )
    return panel


def _draw_label(
    panel: Frame, text: str, origin: tuple[int, int]
) -> None:
    x, y = origin
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    pad = 6
    cv2.rectangle(
        panel,
        (x - pad, y - th - pad),
        (x + tw + pad, y + pad),
        _LABEL_BG,
        -1,
    )
    cv2.putText(
        panel,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        _TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )


def _prepend_header(body: Frame, *, layer1: Layer1Output, pin_name: str) -> Frame:
    w = body.shape[1]
    header = np.full((_HEADER_HEIGHT, w, 3), _HEADER_BG, dtype=np.uint8)
    line1 = (
        f"pin: {pin_name}   "
        f"scene_drift: {layer1.scene_drift:.3f} ({layer1.scene_drift_source})   "
        f"lighting_drift: {layer1.lighting_drift:.3f}   "
        f"pose_drift: {layer1.max_camera_pose_drift_deg:.2f}°"
    )
    flags_line = (
        f"flags: {', '.join(layer1.flags)}" if layer1.flags else "flags: (none)"
    )
    cv2.putText(
        header, line1, (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, _TEXT_COLOR, 1, cv2.LINE_AA,
    )
    cv2.putText(
        header, flags_line, (16, 62),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, _TEXT_COLOR, 1, cv2.LINE_AA,
    )
    return cast(Frame, cv2.vconcat([header, body]))


__all__ = ["diff_dir", "render_check_diff"]
