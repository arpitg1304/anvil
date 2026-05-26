"""Tests for the annotated check-diff renderer."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from anvil.diff import diff_dir, render_check_diff
from anvil.layers.layer1_fast import (
    DEFAULT_ARUCO_DICTIONARY,
    ArucoMarker,
    ArucoReference,
    Layer1Output,
)
from anvil.manifest import DIFFS_SUBDIR, REFERENCE_IMAGE_FILENAME, compute_manifest_hash


def _marker_frame(marker_id: int = 0, side: int = 200) -> np.ndarray:
    canvas_h, canvas_w = 480, 640
    marker = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), marker_id, side
    )
    frame = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    top = (canvas_h - side) // 2
    left = (canvas_w - side) // 2
    frame[top : top + side, left : left + side] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    return frame


def _seed_pin(pin_dir: Path) -> ArucoReference:
    """Lay down a reference.png + return an ArucoReference matching it."""
    pin_dir.mkdir(parents=True, exist_ok=True)
    frame = _marker_frame(marker_id=3)
    cv2.imwrite(str(pin_dir / REFERENCE_IMAGE_FILENAME), frame)
    return ArucoReference(
        dictionary=DEFAULT_ARUCO_DICTIONARY,
        markers=[ArucoMarker(id=3, center_px=(320.0, 240.0), orientation_deg=0.0)],
    )


def _layer1_with_pose_drift(deg: float) -> Layer1Output:
    return Layer1Output(
        scene_drift=0.02,
        scene_drift_source="embedder:dinov3-vits16",
        lighting_drift=0.04,
        max_camera_pose_drift_deg=deg,
        findings=[],
        flags=["camera_pose_drift"],
    )


def test_render_writes_png_under_diffs(tmp_path: Path) -> None:
    aruco_ref = _seed_pin(tmp_path)
    current = _marker_frame(marker_id=3)
    out = render_check_diff(
        current_frame=current,
        layer1=_layer1_with_pose_drift(1.5),
        aruco_ref=aruco_ref,
        pin_dir=tmp_path,
        pin_name="test_pin",
        threshold_deg=1.0,
    )
    assert out.exists()
    assert out.parent == diff_dir(tmp_path)
    assert out.parent.name == DIFFS_SUBDIR
    img = cv2.imread(str(out))
    assert img is not None
    # Side-by-side panels + header strip — should be wider than either input.
    assert img.shape[1] >= 2 * current.shape[1]
    assert img.shape[0] >= current.shape[0]  # header adds height


def test_render_is_excluded_from_manifest_hash(tmp_path: Path) -> None:
    aruco_ref = _seed_pin(tmp_path)
    # Compute the pre-diff hash. Including manifest.yaml isn't required for
    # this test — compute_manifest_hash hashes whatever files are present.
    hash_before = compute_manifest_hash(tmp_path)
    render_check_diff(
        current_frame=_marker_frame(marker_id=3),
        layer1=_layer1_with_pose_drift(1.5),
        aruco_ref=aruco_ref,
        pin_dir=tmp_path,
        pin_name="test_pin",
        threshold_deg=1.0,
    )
    # The diff is on disk now, but the hash must be stable.
    hash_after = compute_manifest_hash(tmp_path)
    assert hash_before == hash_after
    assert any(diff_dir(tmp_path).iterdir())


def test_render_handles_missing_marker_in_current(tmp_path: Path) -> None:
    # Reference had marker id 3, current frame has nothing — should still
    # render a diff without crashing (just no delta annotation on the marker).
    aruco_ref = _seed_pin(tmp_path)
    blank = np.full((480, 640, 3), 200, dtype=np.uint8)
    out = render_check_diff(
        current_frame=blank,
        layer1=_layer1_with_pose_drift(0.0),
        aruco_ref=aruco_ref,
        pin_dir=tmp_path,
        pin_name="test_pin",
        threshold_deg=1.0,
    )
    assert out.exists()


def test_render_filename_format(tmp_path: Path) -> None:
    aruco_ref = _seed_pin(tmp_path)
    out = render_check_diff(
        current_frame=_marker_frame(marker_id=3),
        layer1=_layer1_with_pose_drift(1.5),
        aruco_ref=aruco_ref,
        pin_dir=tmp_path,
        pin_name="test_pin",
        threshold_deg=1.0,
    )
    # Pattern: diff_<YYYYMMDDTHHMMSSZ>.png
    assert out.name.startswith("diff_")
    assert out.suffix == ".png"
    # Z suffix indicates UTC.
    assert out.stem.endswith("Z")
