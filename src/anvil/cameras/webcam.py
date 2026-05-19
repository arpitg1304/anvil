"""USB / V4L2 / AVFoundation webcam driver, via OpenCV."""

from __future__ import annotations

from typing import cast

import cv2

from anvil.cameras.base import Camera, Frame
from anvil.errors import CameraError


class WebcamCamera(Camera):
    """OpenCV VideoCapture-backed driver.

    ``device`` accepts an integer index (passed straight to ``VideoCapture``),
    a stringified integer (``"0"``), or a path/URI (``/dev/video0``,
    ``rtsp://...``). Numeric strings are coerced to int for cross-platform
    consistency.
    """

    def __init__(
        self,
        device: int | str,
        resolution: tuple[int, int] | None = None,
    ) -> None:
        self._device = device
        self._resolution = resolution
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        raw = self._device
        if isinstance(raw, str) and raw.isdigit():
            target: int | str = int(raw)
        else:
            target = raw
        cap = cv2.VideoCapture(target)
        if not cap.isOpened():
            cap.release()
            raise CameraError(f"Failed to open camera {self._device!r}")
        if self._resolution is not None:
            width, height = self._resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        self._cap = cap

    def grab(self) -> Frame:
        if self._cap is None:
            raise CameraError("Camera not opened — call .open() or use as a context manager")
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise CameraError(f"Failed to grab frame from {self._device!r}")
        # OpenCV's VideoCapture returns uint8 BGR; cast to satisfy the typed Frame alias.
        return cast(Frame, frame)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
