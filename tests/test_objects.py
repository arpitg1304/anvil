"""Tests for the open-vocabulary object detector abstraction."""

from __future__ import annotations

import sys
from typing import ClassVar

import numpy as np

from anvil.cameras.base import Frame
from anvil.models.objects import (
    DetectedObject,
    ObjectDetector,
    load_object_detector,
)


class _StubObjectDetector(ObjectDetector):
    """Returns one detection per prompt, with deterministic bbox + confidence."""

    name: ClassVar[str] = "stub-detector"

    def __init__(self, present: set[str] | None = None) -> None:
        # If ``present`` is None all prompts are detected; otherwise only
        # prompts in the set come back.
        self._present = present
        self._loaded = False

    def load(self) -> None:
        self._loaded = True

    def detect(
        self,
        frame: Frame,
        prompts: list[str],
        *,
        confidence_threshold: float = 0.1,
    ) -> list[DetectedObject]:
        if not self._loaded:
            self.load()
        h, w = frame.shape[:2]
        out: list[DetectedObject] = []
        for i, p in enumerate(prompts):
            if self._present is not None and p not in self._present:
                continue
            # Spread bboxes across the frame deterministically by prompt index.
            x0 = (50 + i * 60) % (w - 100)
            y0 = (40 + i * 40) % (h - 80)
            out.append(
                DetectedObject(
                    name=p,
                    bbox=(float(x0), float(y0), float(x0 + 80), float(y0 + 60)),
                    confidence=0.9,
                )
            )
        return out


def test_detected_object_geometry_properties() -> None:
    obj = DetectedObject(
        name="cube", bbox=(10.0, 20.0, 110.0, 80.0), confidence=0.5
    )
    assert obj.centroid == (60.0, 50.0)
    assert obj.width == 100.0
    assert obj.height == 60.0
    assert obj.area == 6000.0


def test_stub_detector_returns_one_box_per_prompt() -> None:
    det = _StubObjectDetector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = det.detect(frame, ["red_cube", "blue_plate"])
    assert {d.name for d in out} == {"red_cube", "blue_plate"}
    assert all(d.confidence > 0.0 for d in out)


def test_stub_detector_respects_present_set() -> None:
    det = _StubObjectDetector(present={"red_cube"})
    out = det.detect(np.zeros((480, 640, 3), dtype=np.uint8), ["red_cube", "missing"])
    assert [d.name for d in out] == ["red_cube"]


def test_empty_prompts_returns_empty() -> None:
    det = _StubObjectDetector()
    out = det.detect(np.zeros((480, 640, 3), dtype=np.uint8), [])
    assert out == []


def test_load_object_detector_returns_none_when_load_fails(monkeypatch) -> None:
    def boom(self: object) -> None:
        raise RuntimeError("simulated [full] missing")

    monkeypatch.setattr(
        "anvil.models.grounding_dino.GroundingDINODetector.load",
        boom,
        raising=True,
    )
    monkeypatch.setattr(
        "anvil.models.yolo_world.YOLOWorldDetector.load", boom, raising=True
    )
    assert load_object_detector() is None


def test_load_object_detector_returns_none_when_module_import_fails(
    monkeypatch,
) -> None:
    monkeypatch.setitem(sys.modules, "anvil.models.grounding_dino", None)
    monkeypatch.setitem(sys.modules, "anvil.models.yolo_world", None)
    assert load_object_detector() is None


def test_load_object_detector_falls_back_to_yolo_world(monkeypatch) -> None:
    """When GroundingDINO fails to load, factory should still return YOLOWorld."""

    def boom(self: object) -> None:
        raise RuntimeError("simulated GroundingDINO failure")

    monkeypatch.setattr(
        "anvil.models.grounding_dino.GroundingDINODetector.load",
        boom,
        raising=True,
    )
    monkeypatch.setattr(
        "anvil.models.yolo_world.YOLOWorldDetector.load",
        lambda self: setattr(self, "_loaded", True),
        raising=True,
    )
    det = load_object_detector()
    assert det is not None
    assert det.name == "yolo-world-v8s"


