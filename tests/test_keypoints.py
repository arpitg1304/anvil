"""Tests for the keypoint detector + matcher abstractions.

Concrete DISK + LightGlue isn't exercised here — it needs the [full] extra
and the kornia weight cache. Protocol contract is verified via stubs; the
factory's graceful-failure path is verified by forcing imports to fail.
"""

from __future__ import annotations

import sys
from typing import ClassVar

import numpy as np
import pytest

from anvil.cameras.base import Frame
from anvil.models.keypoints import (
    KeypointDetector,
    KeypointMatcher,
    KeypointSet,
    MatchResult,
    load_keypoint_pipeline,
)


class _StubDetector(KeypointDetector):
    name: ClassVar[str] = "stub"

    def __init__(self) -> None:
        self._loaded = False

    @property
    def descriptor_dim(self) -> int:
        return 4

    def load(self) -> None:
        self._loaded = True

    def detect(self, frame: Frame, *, max_keypoints: int = 1024) -> KeypointSet:
        if not self._loaded:
            self.load()
        h, w = frame.shape[:2]
        return KeypointSet(
            keypoints=[(10.0, 10.0), (50.0, 50.0)],
            descriptors=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            image_hw=(h, w),
            detector_name=self.name,
        )


class _StubMatcher(KeypointMatcher):
    name: ClassVar[str] = "stub-matcher"
    expected_detector: ClassVar[str] = "stub"

    def __init__(self) -> None:
        self._loaded = False

    def load(self) -> None:
        self._loaded = True

    def match(self, ref: KeypointSet, cur: KeypointSet) -> MatchResult:
        if ref.detector_name != self.expected_detector:
            raise ValueError(
                f"{self.name} expects detector "
                f"{self.expected_detector!r}, got {ref.detector_name!r}"
            )
        return MatchResult(indices_ref=[0, 1], indices_cur=[0, 1])


def test_keypoint_set_count_and_arrays() -> None:
    kp = KeypointSet(
        keypoints=[(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)],
        descriptors=[[1, 0], [0, 1], [1, 1]],
        image_hw=(480, 640),
        detector_name="stub",
    )
    assert kp.count == 3
    assert kp.keypoints_array().shape == (3, 2)
    assert kp.descriptors_array().shape == (3, 2)


def test_stub_detector_emits_set_with_correct_metadata() -> None:
    det = _StubDetector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    kp = det.detect(frame)
    assert kp.detector_name == "stub"
    assert kp.image_hw == (480, 640)
    assert kp.count == 2


def test_stub_matcher_rejects_wrong_detector() -> None:
    matcher = _StubMatcher()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    ref = _StubDetector().detect(frame)
    cur = KeypointSet(
        keypoints=ref.keypoints,
        descriptors=ref.descriptors,
        image_hw=ref.image_hw,
        detector_name="other-detector",
    )
    with pytest.raises(ValueError, match="expects detector"):
        matcher.match(cur, ref)


def test_load_keypoint_pipeline_returns_none_when_load_fails(monkeypatch) -> None:
    def boom(self: object) -> None:
        raise RuntimeError("simulated [full] missing")

    monkeypatch.setattr(
        "anvil.models.disk_lightglue.DISKDetector.load", boom, raising=True
    )
    assert load_keypoint_pipeline() is None


def test_load_keypoint_pipeline_returns_none_when_module_import_fails(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "anvil.models.disk_lightglue", None)
    assert load_keypoint_pipeline() is None
