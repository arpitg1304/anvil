"""Layer 1 primitives + orchestrator tests.

Uses synthetic frames (solid-color images, programmatically generated ArUco
markers) so the tests are fast, deterministic, and don't depend on real
camera output.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np
import pytest

from anvil.cameras.base import Frame
from anvil.layers.layer1_fast import (
    ARUCO_FILENAME,
    DEFAULT_ARUCO_DICTIONARY,
    EMBEDDING_FILENAME,
    HISTOGRAM_BINS,
    LIGHTING_FILENAME,
    ArucoMarker,
    ArucoReference,
    EmbeddingReference,
    LightingReference,
    aruco_path,
    compute_aruco_reference,
    compute_embedding_reference,
    compute_lighting_reference,
    detect_aruco_markers,
    embedding_path,
    lighting_path,
    load_aruco_reference,
    load_embedding_reference,
    load_lighting_reference,
    run_layer1,
    save_aruco_reference,
    save_embedding_reference,
    save_lighting_reference,
)
from anvil.manifest import ThresholdSpec
from anvil.models import GlobalEmbedder


class _FixedEmbedder(GlobalEmbedder):
    """Embedder that returns a caller-supplied vector. Lets tests force a
    known cosine distance between pin-time and check-time embeddings.
    """

    name: ClassVar[str] = "fixed-test"

    def __init__(self, vector: np.ndarray) -> None:
        v = vector.astype(np.float32)
        norm = float(np.linalg.norm(v))
        self._vec = v / norm if norm > 0 else v

    @property
    def dim(self) -> int:
        return int(self._vec.shape[0])

    def load(self) -> None:
        return None

    def embed(self, frame_bgr: Frame) -> np.ndarray:
        return self._vec.copy()


def _solid_frame(
    color: tuple[int, int, int], shape: tuple[int, int] = (240, 320)
) -> np.ndarray:
    height, width = shape
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = color
    return frame


def _marker_frame(
    marker_id: int = 0,
    side: int = 200,
    canvas: tuple[int, int] = (480, 640),
) -> np.ndarray:
    marker = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
        marker_id,
        side,
    )
    frame = np.full((canvas[0], canvas[1], 3), 255, dtype=np.uint8)
    top = (canvas[0] - side) // 2
    left = (canvas[1] - side) // 2
    frame[top : top + side, left : left + side] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    return frame


def _rotate_about_center(frame: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = frame.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(
        frame, matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)
    )


# --- lighting primitives -------------------------------------------------


def test_lighting_reference_histogram_shape() -> None:
    ref = compute_lighting_reference(_solid_frame((40, 80, 160)))
    assert len(ref.histogram_b) == HISTOGRAM_BINS
    assert len(ref.histogram_g) == HISTOGRAM_BINS
    assert len(ref.histogram_r) == HISTOGRAM_BINS
    # A solid-colour image puts every pixel in exactly one bin per channel.
    assert sum(1 for v in ref.histogram_b if v > 0) == 1
    assert sum(1 for v in ref.histogram_g if v > 0) == 1
    assert sum(1 for v in ref.histogram_r if v > 0) == 1


def test_mean_luminance_matches_grayscale_mean() -> None:
    frame = _solid_frame((50, 50, 50))
    ref = compute_lighting_reference(frame)
    # OpenCV BGR2GRAY of an isoluminant grey is exactly that grey value.
    assert ref.mean_luminance == pytest.approx(50.0, abs=1.0)


def test_cct_warmer_for_red_biased_near_white() -> None:
    # McCamy is only meaningful near the Planckian locus, so we use mildly
    # tinted near-whites rather than saturated primaries.
    warm = compute_lighting_reference(_solid_frame((180, 200, 220))).cct_kelvin  # R>B
    neutral = compute_lighting_reference(_solid_frame((200, 200, 200))).cct_kelvin
    cool = compute_lighting_reference(_solid_frame((220, 200, 180))).cct_kelvin  # B>R
    assert warm < neutral < cool
    # Neutral grey should land near daylight (D65 = 6504 K).
    assert 6000 < neutral < 7000


# --- ArUco primitives ----------------------------------------------------


def test_detect_aruco_on_synthetic_marker() -> None:
    frame = _marker_frame(marker_id=3)
    markers = detect_aruco_markers(frame, DEFAULT_ARUCO_DICTIONARY)
    assert len(markers) == 1
    assert markers[0].id == 3


def test_compute_aruco_reference_no_markers_in_empty_frame() -> None:
    ref = compute_aruco_reference(_solid_frame((128, 128, 128)))
    assert ref.markers == []
    assert ref.dictionary == DEFAULT_ARUCO_DICTIONARY


def test_aruco_orientation_changes_after_rotation() -> None:
    frame = _marker_frame(marker_id=0)
    rotated = _rotate_about_center(frame, 30.0)
    a = detect_aruco_markers(frame)[0]
    b = detect_aruco_markers(rotated)[0]
    # Orientation delta should be close to 30° (allow some marker-corner ordering slack).
    diff = ((b.orientation_deg - a.orientation_deg + 180) % 360) - 180
    assert abs(abs(diff) - 30.0) < 3.0


# --- run_layer1 orchestrator --------------------------------------------


def _build_refs(
    frame: np.ndarray,
) -> tuple[LightingReference, ArucoReference]:
    return compute_lighting_reference(frame), compute_aruco_reference(frame)


def test_run_layer1_clean_when_frames_identical() -> None:
    frame = _solid_frame((100, 100, 100))
    light_ref, aruco_ref = _build_refs(frame)
    out = run_layer1(
        current_frame=frame,
        lighting_ref=light_ref,
        aruco_ref=aruco_ref,
        thresholds=ThresholdSpec(),
    )
    assert out.flags == []
    assert out.findings == []
    assert out.scene_drift == pytest.approx(0.0, abs=1e-6)
    assert out.lighting_drift == pytest.approx(0.0, abs=1e-6)
    assert out.max_camera_pose_drift_deg == 0.0


def test_run_layer1_flags_lighting_shift_on_brightness_change() -> None:
    pinned = _solid_frame((40, 40, 40))
    current = _solid_frame((180, 180, 180))
    light_ref, aruco_ref = _build_refs(pinned)
    out = run_layer1(
        current_frame=current,
        lighting_ref=light_ref,
        aruco_ref=aruco_ref,
        thresholds=ThresholdSpec(),
    )
    assert "lighting_shift" in out.flags
    lighting_findings = [f for f in out.findings if f.component == "lighting"]
    assert len(lighting_findings) == 1
    assert lighting_findings[0].evidence is not None
    assert lighting_findings[0].evidence["mean_luminance_delta"] > 100.0


def test_run_layer1_flags_camera_pose_drift_after_rotation() -> None:
    pinned = _marker_frame(marker_id=0)
    current = _rotate_about_center(pinned, 10.0)
    light_ref, aruco_ref = _build_refs(pinned)
    out = run_layer1(
        current_frame=current,
        lighting_ref=light_ref,
        aruco_ref=aruco_ref,
        thresholds=ThresholdSpec(max_camera_pose_drift_deg=1.0),
    )
    assert "camera_pose_drift" in out.flags
    assert out.max_camera_pose_drift_deg > 5.0


def test_run_layer1_flags_missing_markers() -> None:
    pinned = _marker_frame(marker_id=0)
    current = _solid_frame((255, 255, 255), shape=(480, 640))  # marker removed
    light_ref, aruco_ref = _build_refs(pinned)
    out = run_layer1(
        current_frame=current,
        lighting_ref=light_ref,
        aruco_ref=aruco_ref,
        thresholds=ThresholdSpec(),
    )
    assert "camera_pose_drift" in out.flags
    missing_findings = [f for f in out.findings if f.issue == "markers_missing"]
    assert len(missing_findings) == 1
    assert missing_findings[0].evidence is not None
    assert missing_findings[0].evidence["missing_marker_ids"] == [0]


# --- persistence ---------------------------------------------------------


def test_lighting_reference_round_trip(tmp_path: Path) -> None:
    ref = compute_lighting_reference(_solid_frame((40, 80, 160)))
    saved = save_lighting_reference(ref, tmp_path)
    assert saved == lighting_path(tmp_path)
    assert saved.name == LIGHTING_FILENAME
    reloaded = load_lighting_reference(tmp_path)
    assert reloaded == ref


def test_aruco_reference_round_trip(tmp_path: Path) -> None:
    ref = ArucoReference(
        dictionary=DEFAULT_ARUCO_DICTIONARY,
        markers=[ArucoMarker(id=7, center_px=(100.0, 200.0), orientation_deg=12.5)],
    )
    saved = save_aruco_reference(ref, tmp_path)
    assert saved == aruco_path(tmp_path)
    assert saved.name == ARUCO_FILENAME
    assert saved.parent.name == "pose"
    reloaded = load_aruco_reference(tmp_path)
    assert reloaded == ref


# --- embedding-path scene_drift -----------------------------------------


def test_run_layer1_uses_embedder_when_ref_and_embedder_match() -> None:
    pinned = _solid_frame((128, 128, 128))
    ref_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    pin_embedder = _FixedEmbedder(ref_vec)
    embedding_ref = compute_embedding_reference(pinned, pin_embedder)

    # At check time, embedder returns a different unit vector — known cosine.
    check_vec = np.array([0.6, 0.8, 0.0, 0.0], dtype=np.float32)
    check_embedder = _FixedEmbedder(check_vec)
    light_ref, aruco_ref = _build_refs(pinned)
    out = run_layer1(
        current_frame=pinned,
        lighting_ref=light_ref,
        aruco_ref=aruco_ref,
        thresholds=ThresholdSpec(),
        embedding_ref=embedding_ref,
        embedder=check_embedder,
    )
    # cosine(ref, check) = 0.6 → drift = 0.4
    assert out.scene_drift == pytest.approx(0.4, abs=1e-5)
    assert out.scene_drift_source == "embedder:fixed-test"


def test_run_layer1_falls_back_to_histogram_when_no_embedder() -> None:
    frame = _solid_frame((128, 128, 128))
    light_ref, aruco_ref = _build_refs(frame)
    out = run_layer1(
        current_frame=frame,
        lighting_ref=light_ref,
        aruco_ref=aruco_ref,
        thresholds=ThresholdSpec(),
        embedding_ref=None,
        embedder=None,
    )
    # Identical frames → histogram chi^2 ~ 0, no embedder, source = histogram.
    assert out.scene_drift == pytest.approx(0.0, abs=1e-6)
    assert out.scene_drift_source == "histogram"


def test_run_layer1_falls_back_when_embedder_name_mismatches_ref() -> None:
    # Manifest pinned with embedder X, but current process loaded embedder Y.
    # The check should fall back to histogram rather than compare apples to
    # oranges across feature spaces.
    pinned = _solid_frame((128, 128, 128))
    ref = EmbeddingReference(
        embedder_name="some-other-backbone",
        dim=4,
        vector=[1.0, 0.0, 0.0, 0.0],
    )
    embedder = _FixedEmbedder(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
    light_ref, aruco_ref = _build_refs(pinned)
    out = run_layer1(
        current_frame=pinned,
        lighting_ref=light_ref,
        aruco_ref=aruco_ref,
        thresholds=ThresholdSpec(),
        embedding_ref=ref,
        embedder=embedder,
    )
    assert out.scene_drift_source == "histogram"


def test_embedding_reference_round_trip(tmp_path: Path) -> None:
    ref = EmbeddingReference(
        embedder_name="fixed-test",
        dim=4,
        vector=[0.5, 0.5, 0.5, 0.5],
    )
    saved = save_embedding_reference(ref, tmp_path)
    assert saved == embedding_path(tmp_path)
    assert saved.name == EMBEDDING_FILENAME
    reloaded = load_embedding_reference(tmp_path)
    assert reloaded is not None
    assert reloaded.embedder_name == ref.embedder_name
    assert reloaded.dim == ref.dim
    assert reloaded.vector == pytest.approx(ref.vector)


def test_load_embedding_reference_returns_none_when_missing(tmp_path: Path) -> None:
    assert load_embedding_reference(tmp_path) is None
