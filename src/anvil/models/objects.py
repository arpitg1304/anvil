"""Open-vocabulary object detector abstraction for Layer 2.

Layer 2's named-object workflow needs a text-promptable detector — "red
cube", "test tube rack" → bounding boxes. The bare install ships no
detector; ``load_object_detector()`` returns ``None`` and the named-object
path is silently skipped, same fallback pattern as the embedder and
keypoint pipelines.

Today's concrete impl is YOLO-World via Ultralytics (unrestricted Tencent
weights, no HF gate). Per-pixel masks via SAM 3 would slot in here as a
second concrete impl, with the choice driven by what's available at
``load`` time.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from anvil.cameras.base import Frame

_FORWARD_COMPAT = ConfigDict(extra="ignore")


class DetectedObject(BaseModel):
    """One detection from a single frame.

    ``name`` is the text prompt that fired (e.g. ``"red cube"``). ``bbox`` is
    in image pixel coordinates, ``(x1, y1, x2, y2)`` — top-left to
    bottom-right — matching Ultralytics / OpenCV convention.
    """

    model_config = _FORWARD_COMPAT

    name: str
    bbox: tuple[float, float, float, float]
    confidence: float

    @property
    def centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float:
        return max(self.width, 0.0) * max(self.height, 0.0)


class ObjectDetector(ABC):
    """Text-promptable open-vocabulary detector."""

    name: ClassVar[str]
    """Stable id of the backbone (e.g. ``yolo-world-v8s``)."""

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def detect(
        self, frame: Frame, prompts: list[str], *, confidence_threshold: float = 0.1
    ) -> list[DetectedObject]:
        """Detect every prompt in one pass.

        For each prompt the detector returns its highest-confidence box
        (or none, if it can't find that concept). Implementations should
        match returned objects' ``name`` field to the input ``prompts``
        exactly so callers can index by prompt string.
        """


def load_object_detector() -> ObjectDetector | None:
    """Return the best available detector, or ``None`` if nothing loads.

    Today only YOLO-World v8s is wired up. Adding a SAM 3 concrete
    later is a single ``try`` block here.
    """
    try:
        from anvil.models.yolo_world import YOLOWorldDetector
    except ImportError:
        return None
    detector = YOLOWorldDetector()
    try:
        detector.load()
    except Exception:
        return None
    return detector


__all__ = ["DetectedObject", "ObjectDetector", "load_object_detector"]
