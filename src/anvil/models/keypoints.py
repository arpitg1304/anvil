"""Keypoint detection + matching abstractions for Layer 2.

Layer 2's keypoint path estimates camera pose drift from natural texture in
the scene — no ArUco markers required. The work splits cleanly into two
roles:

* ``KeypointDetector`` — given a frame, return up to N salient keypoints
  with descriptors. Concrete impls today: DISK via kornia.
* ``KeypointMatcher`` — given two ``KeypointSet`` references, return the
  pairs of indices that correspond to the same physical point. Concrete
  impls today: LightGlue (DISK variant) via kornia.

Like the embedder abstraction, both live behind the ``[full]`` extra.
``load_keypoint_pipeline()`` returns a compatible detector+matcher pair, or
``None`` when nothing usable can be loaded. Layer 2 falls back to "no
keypoint signal available" silently when that happens — same pattern as
``load_embedder``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict

from anvil.cameras.base import Frame

_FORWARD_COMPAT = ConfigDict(extra="ignore")


class KeypointSet(BaseModel):
    """Detector output: ``N`` keypoints in pixel coords + their descriptors."""

    model_config = _FORWARD_COMPAT

    # (x, y) in image pixel coordinates, one per keypoint.
    keypoints: list[tuple[float, float]]
    # NxD float descriptors. Detector-specific dimensionality (DISK = 128,
    # SuperPoint would be 256).
    descriptors: list[list[float]]
    # Image shape the keypoints were detected from, (height, width). Some
    # matchers need this for positional encoding.
    image_hw: tuple[int, int]
    # Stable name of the detector that produced this set. The matcher
    # validates this before running so we never feed DISK descriptors into a
    # SuperPoint-trained LightGlue head.
    detector_name: str

    @property
    def count(self) -> int:
        return len(self.keypoints)

    def keypoints_array(self) -> np.ndarray:
        return np.asarray(self.keypoints, dtype=np.float32)

    def descriptors_array(self) -> np.ndarray:
        return np.asarray(self.descriptors, dtype=np.float32)


class MatchResult(BaseModel):
    """Output of ``KeypointMatcher.match``: paired indices into the two sets."""

    model_config = _FORWARD_COMPAT

    # Same length. ``indices_ref[i]`` matches ``indices_cur[i]``.
    indices_ref: list[int]
    indices_cur: list[int]

    @property
    def count(self) -> int:
        return len(self.indices_ref)


class KeypointDetector(ABC):
    name: ClassVar[str]
    """Stable id, mixed into KeypointSet so a matcher can sanity-check."""

    @property
    @abstractmethod
    def descriptor_dim(self) -> int: ...

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def detect(self, frame: Frame, *, max_keypoints: int = 1024) -> KeypointSet: ...


class KeypointMatcher(ABC):
    name: ClassVar[str]
    """Stable id of the matcher (e.g. ``lightglue-disk``)."""

    expected_detector: ClassVar[str]
    """Name of the detector this matcher was trained against. ``match`` should
    raise ``ValueError`` if a ``KeypointSet`` carries a different ``detector_name``.
    """

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def match(self, ref: KeypointSet, cur: KeypointSet) -> MatchResult: ...


# A compatible detector+matcher pair. Returned as a tuple so callers can hold
# both with one name.
KeypointPipeline = tuple[KeypointDetector, KeypointMatcher]


def load_keypoint_pipeline() -> KeypointPipeline | None:
    """Return a usable ``(detector, matcher)`` pair, or ``None`` if nothing loads.

    Today only DISK + LightGlue (DISK variant) is implemented. Adding new
    pairs (SuperPoint, KeyNet, …) is a single ``try`` block here.
    """
    try:
        from anvil.models.disk_lightglue import DISKDetector, LightGlueDISKMatcher
    except ImportError:
        return None
    detector = DISKDetector()
    matcher = LightGlueDISKMatcher()
    try:
        detector.load()
        matcher.load()
    except Exception:
        return None
    return (detector, matcher)


__all__ = [
    "KeypointDetector",
    "KeypointMatcher",
    "KeypointPipeline",
    "KeypointSet",
    "MatchResult",
    "load_keypoint_pipeline",
]
