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
from anvil.layers.layer2_structural import ObjectReference
from anvil.manifest import DIFFS_SUBDIR, REFERENCE_IMAGE_FILENAME
from anvil.models.objects import DetectedObject

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

# Per-object outline colors.
_OBJECT_REF_COLOR = (255, 200, 0)      # cyan-blue on reference panel
_OBJECT_OK_COLOR = (60, 220, 60)       # green when matched + within threshold
_OBJECT_DRIFT_COLOR = (60, 60, 255)    # red when matched but over threshold
_OBJECT_MISSING_COLOR = (128, 128, 128)  # grey when no current match


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
    object_refs: list[ObjectReference] | None = None,
    current_detections: list[DetectedObject] | None = None,
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
    if object_refs:
        ref_panel = _annotate_object_refs(ref_panel, object_refs)
        cur_panel = _annotate_object_currents(
            cur_panel, object_refs, current_detections or []
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
            # cv2.putText with FONT_HERSHEY_SIMPLEX is ASCII-only — Unicode
            # glyphs like Delta or the degree sign render as '??'. Stick to
            # ASCII labels here; the JSON sidecar keeps the precise number.
            label += f"  d={delta:+.2f}deg"
        _draw_label(panel, label, (int(cx) - 80, int(cy) - 12))
    return panel


def _annotate_object_refs(panel: Frame, refs: list[ObjectReference]) -> Frame:
    """Draw each pinned object's bbox + name on the reference panel."""
    for ref in refs:
        x1, y1, x2, y2 = (round(v) for v in ref.bbox)
        cv2.rectangle(panel, (x1, y1), (x2, y2), _OBJECT_REF_COLOR, 2)
        size_w, size_h = x2 - x1, y2 - y1
        _draw_label(
            panel,
            f"{ref.name} ({size_w}x{size_h})",
            (x1 + 4, max(y1 - 8, 16)),
        )
    return panel


def _annotate_object_currents(
    panel: Frame,
    refs: list[ObjectReference],
    detections: list[DetectedObject],
) -> Frame:
    """Draw current-frame candidates for each pinned object.

    For each pinned ref we draw *all* candidates matching its prompt — the
    matcher's chosen one in green/red (depending on whether it would trip
    a finding), and any other candidates in a thin grey outline so the
    operator can see what alternatives the detector returned. This makes
    "the matcher picked a small candidate near the centroid over a
    properly-sized one offset" visible at a glance.
    """
    # Group detections by prompt for fast lookup.
    candidates_by_prompt: dict[str, list[DetectedObject]] = {}
    for d in detections:
        candidates_by_prompt.setdefault(d.name, []).append(d)

    for ref in refs:
        candidates = candidates_by_prompt.get(ref.prompt, [])
        # Other candidates first (so the chosen one draws on top).
        chosen = _pick_best_match(ref, candidates)
        for cand in candidates:
            if cand is chosen:
                continue
            x1, y1, x2, y2 = (round(v) for v in cand.bbox)
            cv2.rectangle(panel, (x1, y1), (x2, y2), _OBJECT_MISSING_COLOR, 1)
        if chosen is None:
            continue
        iou = _bbox_iou(ref.bbox, chosen.bbox)
        color = _OBJECT_OK_COLOR if iou >= 0.5 else _OBJECT_DRIFT_COLOR
        x1, y1, x2, y2 = (round(v) for v in chosen.bbox)
        cv2.rectangle(panel, (x1, y1), (x2, y2), color, 2)
        size_w, size_h = x2 - x1, y2 - y1
        _draw_label(
            panel,
            f"{ref.name} IoU={iou:.2f} ({size_w}x{size_h})",
            (x1 + 4, max(y1 - 8, 16)),
        )
    return panel


def _pick_best_match(
    ref: ObjectReference, candidates: list[DetectedObject]
) -> DetectedObject | None:
    """Mirror of ``layer2_structural._match_to_pinned`` so the diff shows the
    same candidate the orchestrator picked. Kept as a direct import to
    avoid drift.
    """
    from anvil.layers.layer2_structural import _match_to_pinned

    return _match_to_pinned(ref, candidates)


def _bbox_iou(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


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
        f"pose_drift: {layer1.max_camera_pose_drift_deg:.2f}deg"
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
