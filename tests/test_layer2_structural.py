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


def test_run_layer2_flags_translation_without_rotation() -> None:
    """Camera that slides without twisting should still trip camera_pose_drift.

    A lateral shift (no rotation) is the realistic 'loose mount slipped'
    case — and the regression that originally surfaced the gap: rotation
    stayed near 0 but the rig had visibly moved.
    """
    base = _grid_keypoints()
    pin_det = _RigidStubDetector(base, rotation_deg=0.0)
    check_det = _RigidStubDetector(base, rotation_deg=0.0, translation=(0.0, 25.0))
    keypoint_ref = compute_keypoint_reference(_blank_frame(), pin_det)
    out = run_layer2_keypoints(
        current_frame=_blank_frame(),
        keypoint_ref=keypoint_ref,
        detector=check_det,
        matcher=_IdentityMatcher(),
        thresholds=ThresholdSpec(
            max_camera_pose_drift_deg=1.0,
            max_camera_translation_px=15.0,
        ),
    )
    assert "camera_pose_drift" in out.flags
    translation_findings = [
        f for f in out.findings if f.issue == "translation_drift"
    ]
    assert len(translation_findings) == 1
    # Should NOT have raised a rotation finding (rotation was 0).
    assert not any(f.issue == "rotation_drift" for f in out.findings)
    assert translation_findings[0].evidence is not None
    assert translation_findings[0].evidence["translation_magnitude_px"] > 20.0


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


# --- named-object orchestrator -----------------------------------------


from anvil.layers.layer2_structural import (  # noqa: E402
    ObjectReference,
    compute_object_reference,
    load_object_references,
    object_reference_path,
    run_layer2_objects,
    save_object_reference,
)
from anvil.models.objects import DetectedObject, ObjectDetector  # noqa: E402


class _StaticObjectDetector(ObjectDetector):
    """Detector that returns a caller-supplied mapping ``prompt -> bbox``."""

    name: ClassVar[str] = "static-stub"

    def __init__(
        self, results: dict[str, tuple[float, float, float, float]] | None = None
    ) -> None:
        self._results = results or {}

    def load(self) -> None:
        return None

    def detect(
        self,
        frame: Frame,
        prompts: list[str],
        *,
        confidence_threshold: float = 0.1,
    ) -> list[DetectedObject]:
        out: list[DetectedObject] = []
        for p in prompts:
            bbox = self._results.get(p)
            if bbox is None:
                continue
            out.append(DetectedObject(name=p, bbox=bbox, confidence=0.95))
        return out


def test_object_reference_round_trip(tmp_path: Path) -> None:
    ref = ObjectReference(
        name="red_cube",
        prompt="red cube",
        bbox=(100.0, 200.0, 300.0, 400.0),
        confidence=0.9,
        image_hw=(480, 640),
        detector_name="static-stub",
        region_embedding=[0.5, 0.5, 0.5, 0.5],
        embedder_name="stub-embedder",
    )
    saved = save_object_reference(ref, tmp_path)
    assert saved == object_reference_path(tmp_path, "red_cube")
    reloaded = load_object_references(tmp_path)
    assert len(reloaded) == 1
    assert reloaded[0] == ref


def test_compute_object_reference_returns_none_when_not_detected() -> None:
    det = _StaticObjectDetector(results={})
    ref = compute_object_reference(
        _blank_frame(), name="rack", prompt="test tube rack", detector=det
    )
    assert ref is None


def test_compute_object_references_batches_single_detector_call() -> None:
    """The batch API must invoke ``detector.detect`` exactly once."""

    class _CountingDetector(_StaticObjectDetector):
        def __init__(self) -> None:
            super().__init__(
                results={
                    "red cube": (10.0, 20.0, 110.0, 120.0),
                    "blue plate": (300.0, 50.0, 500.0, 250.0),
                }
            )
            self.detect_calls = 0

        def detect(self, frame, prompts, *, confidence_threshold=0.1):  # type: ignore[override]
            self.detect_calls += 1
            return super().detect(frame, prompts)

    from anvil.layers.layer2_structural import compute_object_references

    det = _CountingDetector()
    pairs = [("cube", "red cube"), ("plate", "blue plate")]
    results = compute_object_references(_blank_frame(), pairs, det)
    assert det.detect_calls == 1
    names_to_refs = dict(results)
    assert names_to_refs["cube"] is not None
    assert names_to_refs["plate"] is not None
    assert names_to_refs["cube"].bbox == (10.0, 20.0, 110.0, 120.0)


def test_compute_object_references_dedupes_same_prompt() -> None:
    """Two object names sharing one prompt should bind to the same bbox."""

    class _Det(_StaticObjectDetector):
        def __init__(self) -> None:
            super().__init__(results={"robot gripper": (10.0, 10.0, 110.0, 110.0)})
            self.seen_prompts: list[list[str]] | None = None

        def detect(self, frame, prompts, *, confidence_threshold=0.1):  # type: ignore[override]
            # Capture what prompts the batch API passed in.
            self.seen_prompts = list(prompts)
            return super().detect(frame, prompts)

    from anvil.layers.layer2_structural import compute_object_references

    det = _Det()
    results = compute_object_references(
        _blank_frame(),
        [("left", "robot gripper"), ("right", "robot gripper")],
        det,
    )
    # Detector should only see the unique prompt once, not duplicated.
    assert det.seen_prompts == ["robot gripper"]
    by_name = dict(results)
    assert by_name["left"] is not None
    assert by_name["right"] is not None
    assert by_name["left"].bbox == by_name["right"].bbox


def test_compute_object_reference_captures_bbox() -> None:
    det = _StaticObjectDetector(
        results={"red cube": (10.0, 20.0, 110.0, 120.0)}
    )
    ref = compute_object_reference(
        _blank_frame(), name="red_cube", prompt="red cube", detector=det
    )
    assert ref is not None
    assert ref.bbox == (10.0, 20.0, 110.0, 120.0)
    assert ref.detector_name == "static-stub"
    assert ref.region_embedding == []  # no embedder passed


def test_run_layer2_objects_clean_when_object_in_same_place(tmp_path: Path) -> None:
    bbox = (100.0, 100.0, 300.0, 300.0)
    det = _StaticObjectDetector(results={"red cube": bbox})
    ref = compute_object_reference(
        _blank_frame(), name="red_cube", prompt="red cube", detector=det
    )
    assert ref is not None
    out = run_layer2_objects(
        current_frame=_blank_frame(),
        object_refs=[ref],
        detector=det,
        embedder=None,
        thresholds=ThresholdSpec(),
    )
    assert out.findings == []
    assert out.flags == []
    assert out.min_object_iou == pytest.approx(1.0)


def test_run_layer2_objects_flags_object_missing(tmp_path: Path) -> None:
    pin_det = _StaticObjectDetector(
        results={"red cube": (100.0, 100.0, 300.0, 300.0)}
    )
    check_det = _StaticObjectDetector(results={})  # nothing detected
    ref = compute_object_reference(
        _blank_frame(), name="red_cube", prompt="red cube", detector=pin_det
    )
    assert ref is not None
    out = run_layer2_objects(
        current_frame=_blank_frame(),
        object_refs=[ref],
        detector=check_det,
        embedder=None,
        thresholds=ThresholdSpec(),
    )
    assert "object_missing" in out.flags
    missing = [f for f in out.findings if f.issue == "object_missing"]
    assert len(missing) == 1
    assert missing[0].subject == "red_cube"


def test_run_layer2_objects_flags_position_drift(tmp_path: Path) -> None:
    pin_det = _StaticObjectDetector(
        results={"red cube": (100.0, 100.0, 300.0, 300.0)}
    )
    # Shift bbox so IoU drops below 0.5.
    check_det = _StaticObjectDetector(
        results={"red cube": (300.0, 100.0, 500.0, 300.0)}
    )
    ref = compute_object_reference(
        _blank_frame(), name="red_cube", prompt="red cube", detector=pin_det
    )
    assert ref is not None
    out = run_layer2_objects(
        current_frame=_blank_frame(),
        object_refs=[ref],
        detector=check_det,
        embedder=None,
        thresholds=ThresholdSpec(min_object_iou=0.5),
    )
    assert "object_moved" in out.flags
    pos = [f for f in out.findings if f.issue == "position_drift"]
    assert len(pos) == 1
    assert pos[0].evidence is not None
    assert pos[0].evidence["iou"] < 0.5
    assert out.max_object_drift_px > 100


def test_run_layer2_objects_no_refs_returns_empty() -> None:
    out = run_layer2_objects(
        current_frame=_blank_frame(),
        object_refs=[],
        detector=_StaticObjectDetector(),
        embedder=None,
        thresholds=ThresholdSpec(),
    )
    assert out.findings == []
    assert out.flags == []
    assert out.num_objects_pinned == 0
