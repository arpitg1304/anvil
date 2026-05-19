"""Filesystem-backed camera — reads from an image file or a directory.

Used by the test suite and by ``anvil check`` / ``anvil pin`` runs against
saved reference data. Production rigs use :class:`WebcamCamera`.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import cv2

from anvil.cameras.base import Camera, Frame
from anvil.errors import CameraError

_IMAGE_GLOBS: tuple[str, ...] = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")


class FileCamera(Camera):
    """Read frames from disk.

    A single file path returns that frame on every ``grab()``. A directory
    cycles through its image files in sorted filename order, wrapping at the
    end (so the camera never "runs out").
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._frames: list[Path] = []
        self._idx = 0

    def open(self) -> None:
        if not self._path.exists():
            raise CameraError(f"FileCamera path does not exist: {self._path}")
        if self._path.is_file():
            self._frames = [self._path]
        else:
            self._frames = sorted(
                p for pattern in _IMAGE_GLOBS for p in self._path.glob(pattern)
            )
            if not self._frames:
                raise CameraError(f"No image files in {self._path}")
        self._idx = 0

    def grab(self) -> Frame:
        if not self._frames:
            raise CameraError("FileCamera not opened or empty")
        path = self._frames[self._idx % len(self._frames)]
        self._idx += 1
        frame = cv2.imread(str(path))
        if frame is None:
            raise CameraError(f"Failed to decode image at {path}")
        # cv2.imread returns uint8 BGR; cast to satisfy the typed Frame alias.
        return cast(Frame, frame)

    def close(self) -> None:
        self._frames = []
        self._idx = 0
