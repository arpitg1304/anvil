"""Tests for the Layer 2 keypoint orchestrator + persistence.

Uses a stub detector/matcher so the tests don't need kornia. Real DISK +
LightGlue end-to-end is validated via the smoke-test probe in the README's
quickstart, not in pytest.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from anvil.cameras.base import Frame
from anvil.layers.layer2_structural import (
    KEYPOINTS_FILENAME,
    KEYPOINTS_SUBDIR,
    KeypointReference,
    compute_keypoint_reference,
    keypoints_path,
    load_keypoint_reference,
    run_layer2_keypoints,
    save_keypoint_reference,
)
from anvil.manifest import ThresholdSpec
from anvil.models.keypoints import (
    KeypointDetector,
    KeypointMatcher,
    KeypointSet,
    MatchResult,
)


def _grid_keypoints(
    h: int = 480, w: int = 640, n: int = 50
) -> list[tuple[float, float]]:
    rng = np.random.default_rng(42)
    xs = rng.uniform(50, w - 50, n).astype(np.float32)
    ys = rng.uniform(50, h - 50, n).astype(np.float32)
    return [(float(x), float(y)) for x, y in zip(xs, ys, strict=False)]


class _RigidStubDetector(KeypointDetector):
    """Detector that returns a fixed keypoint pattern shifted by a known transform.

    Constructed with a rotation angle and translation; subsequent detect()
    calls produce keypoints transformed by that pose. Two instances let a
    test simulate a 'pin' state and a 'check' state with a known camera
    rotation between them.
    """

    name: ClassVar[str] = "rigid-stub"

    def __init__(
        self,
        base_points: list[tuple[float, float]],
        rotation_deg: float = 0.0,
        translation: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        self._base = np.asarray(base_points, dtype=np.float32)
        theta = np.deg2rad(rotation_deg)
        c, s = np.cos(theta), np.sin(theta)
        # Rotate around image center to mirror what cv2 does internally.
        self._rotation = np.array([[c, -s], [s, c]], dtype=np.float32)
        self._translation = np.array(translation, dtype=np.float32)
        self._center = np.array([320.0, 240.0], dtype=np.float32)

    @property
    def descriptor_dim(self) -> int:
        return 4

    def load(self) -> None:
        return None

    def detect(self, frame: Frame, *, max_keypoints: int = 1024) -> KeypointSet:
        centered = self._base - self._center
        rotated = centered @ self._rotation.T
        transformed = rotated + self._center + self._translation
        # Deterministic descriptors keyed to position so the matcher can pair
        # them across frames without ambiguity.
        descriptors = np.eye(len(self._base), self.descriptor_dim, dtype=np.float32)
        return KeypointSet(
            keypoints=[(float(x), float(y)) for x, y in transformed],
            descriptors=[d.tolist() for d in descriptors],
            image_hw=frame.shape[:2],
            detector_name=self.name,
        )


class _IdentityMatcher(KeypointMatcher):
    """Matcher that pairs index i in ref to index i in cur — perfect matching."""

    name: ClassVar[str] = "identity-stub"
    expected_detector: ClassVar[str] = "rigid-stub"

    def load(self) -> None:
        return None

    def match(self, ref: KeypointSet, cur: KeypointSet) -> MatchResult:
        n = min(ref.count, cur.count)
        return MatchResult(
            indices_ref=list(range(n)),
            indices_cur=list(range(n)),
        )


class _SparseMatcher(KeypointMatcher):
    """Matcher that returns fewer than ``_MIN_MATCHES_FOR_POSE`` pairs."""

    name: ClassVar[str] = "sparse-stub"
    expected_detector: ClassVar[str] = "rigid-stub"

    def load(self) -> None:
        return None

    def match(self, ref: KeypointSet, cur: KeypointSet) -> MatchResult:
        n = min(5, ref.count, cur.count)
        return MatchResult(indices_ref=list(range(n)), indices_cur=list(range(n)))


def _blank_frame() -> Frame:
    return np.zeros((480, 640, 3), dtype=np.uint8)


# --- persistence ---------------------------------------------------------


def test_keypoint_reference_round_trip(tmp_path: Path) -> None:
    ref = KeypointReference(
        detector_name="rigid-stub",
        image_hw=(480, 640),
        keypoints=[(10.0, 20.0), (30.0, 40.0)],
        descriptors=[[1.0, 0.0], [0.0, 1.0]],
    )
    saved = save_keypoint_reference(ref, tmp_path)
    assert saved == keypoints_path(tmp_path)
    assert saved.name == KEYPOINTS_FILENAME
    assert saved.parent.name == KEYPOINTS_SUBDIR

    reloaded = load_keypoint_reference(tmp_path)
    assert reloaded is not None
    assert reloaded.detector_name == ref.detector_name
    assert reloaded.image_hw == ref.image_hw
    assert reloaded.count == ref.count
    assert reloaded.keypoints[0] == pytest.approx(ref.keypoints[0])


def test_load_keypoint_reference_returns_none_when_missing(tmp_path: Path) -> None:
    assert load_keypoint_reference(tmp_path) is None


def test_compute_keypoint_reference_uses_detector_metadata() -> None:
    base = _grid_keypoints()
    det = _RigidStubDetector(base)
    ref = compute_keypoint_reference(_blank_frame(), det)
    assert ref.detector_name == "rigid-stub"
    assert ref.image_hw == (480, 640)
    assert ref.count == len(base)


# --- run_layer2_keypoints orchestrator ----------------------------------


def test_run_layer2_recovers_known_rotation() -> None:
    base = _grid_keypoints()
    pin_det = _RigidStubDetector(base, rotation_deg=0.0)
    check_det = _RigidStubDetector(base, rotation_deg=3.0)
    matcher = _IdentityMatcher()
    keypoint_ref = compute_keypoint_reference(_blank_frame(), pin_det)
    out = run_layer2_keypoints(
        current_frame=_blank_frame(),
        keypoint_ref=keypoint_ref,
        detector=check_det,
        matcher=matcher,
        thresholds=ThresholdSpec(max_camera_pose_drift_deg=1.0),
    )
    # Recovered rotation should be close to the applied 3°; sign convention
    # may flip depending on how affine matrix is interpreted. Compare abs.
    assert out.num_matches >= 10
    assert out.num_inliers >= 10
    assert abs(out.abs_rotation_deg - 3.0) < 0.5
    assert "camera_pose_drift" in out.flags
    assert any(f.issue == "rotation_drift" and f.layer == 2 for f in out.findings)


def test_run_layer2_clean_when_no_rotation() -> None:
    base = _grid_keypoints()
    det = _RigidStubDetector(base)
    matcher = _IdentityMatcher()
    keypoint_ref = compute_keypoint_reference(_blank_frame(), det)
    out = run_layer2_keypoints(
        current_frame=_blank_frame(),
        keypoint_ref=keypoint_ref,
        detector=det,
        matcher=matcher,
        thresholds=ThresholdSpec(max_camera_pose_drift_deg=1.0),
    )
    assert out.abs_rotation_deg < 0.1
    assert out.flags == []
    assert not any(f.issue == "rotation_drift" for f in out.findings)


def test_run_layer2_warns_when_match_count_too_low() -> None:
    base = _grid_keypoints()
    det = _RigidStubDetector(base, rotation_deg=5.0)
    keypoint_ref = compute_keypoint_reference(_blank_frame(), _RigidStubDetector(base))
    out = run_layer2_keypoints(
        current_frame=_blank_frame(),
        keypoint_ref=keypoint_ref,
        detector=det,
        matcher=_SparseMatcher(),
        thresholds=ThresholdSpec(),
    )
    sparse_findings = [f for f in out.findings if f.issue == "keypoint_match_sparse"]
    assert len(sparse_findings) == 1
    # Should not have attempted a rotation finding under sparse-match regime.
    assert not any(f.issue == "rotation_drift" for f in out.findings)
    assert out.num_matches < 10
