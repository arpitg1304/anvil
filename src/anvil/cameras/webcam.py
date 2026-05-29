"""USB / V4L2 / AVFoundation webcam driver, via OpenCV."""

from __future__ import annotations

import platform
from typing import cast

import cv2

from anvil.cameras.base import Camera, Frame
from anvil.errors import CameraError

# Many UVC webcams (Logitech BRIO in particular) return empty frames for the
# first few reads while the V4L2 buffer pipeline spins up — and the first 10+
# valid frames are themselves not stable because auto-exposure / white
# balance / auto-focus are still converging. 15 frames ≈ 500ms at 30fps,
# enough for AE to settle on most UVC cameras without making check feel
# noticeably slower.
_WARMUP_FRAMES = 15

# A requested resolution can take several frames to actually take effect —
# cap.set() returns immediately but the camera reconfigures asynchronously,
# so the first grabs may still be at the old size. Read up to this many
# extra frames waiting for the actual frame size to match the request
# before giving up and letting the caller deal with the mismatch.
_RESOLUTION_SETTLE_FRAMES = 30

# After a resolution change, continuous-autofocus webcams (the BRIO) hunt:
# sharpness craters as the lens searches, then recovers when AF re-locks
# (~1-1.5s). Grabbing during the trough yields a blurry reference. We watch
# the Laplacian-variance sharpness and wait for AF to dip and recover, or for
# sharpness to hold steady near its peak (fixed-focus cameras never dip).
_FOCUS_MIN_FRAMES = 25  # always read ≥ this, so a delayed hunt can manifest
_FOCUS_MAX_FRAMES = 75  # hard cap (~2.5s at 30fps) so we never block forever
_FOCUS_DIP_RATIO = 0.6  # sharpness below this fraction of peak = AF hunting
_FOCUS_RECOVER_RATIO = 0.85  # back above this fraction of peak = re-locked


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
        # On Linux force the V4L2 backend. The default CAP_ANY auto-selection
        # sometimes picks GStreamer/FFMPEG, which silently ignore
        # CAP_PROP_FRAME_WIDTH/HEIGHT — the resolution request never takes and
        # the camera stays at its default (often 640x480).
        if platform.system() == "Linux":
            cap = cv2.VideoCapture(target, cv2.CAP_V4L2)
        else:
            cap = cv2.VideoCapture(target)
        if not cap.isOpened():
            cap.release()
            raise CameraError(f"Failed to open camera {self._device!r}")
        if self._resolution is not None:
            width, height = self._resolution
            # USB webcams can only deliver >720p in MJPG mode — the raw YUYV
            # path is USB-bandwidth-capped (a BRIO falls back to 640x480).
            # The codec must be set *before* the resolution to take effect.
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        for _ in range(_WARMUP_FRAMES):
            cap.read()
        if self._resolution is not None:
            self._settle_resolution(cap, self._resolution)
        self._settle_focus(cap)
        self._cap = cap

    @staticmethod
    def _settle_focus(cap: cv2.VideoCapture) -> None:
        """Read frames until autofocus stops hunting (sharpness plateaus).

        Tracks the peak sharpness seen. Once sharpness has dipped (AF
        searching) and recovered to near peak, the lens has re-locked and we
        return. Fixed-focus cameras never dip, so we also return when
        sharpness simply holds steady near peak. Capped at
        ``_FOCUS_MAX_FRAMES`` so a camera that never stabilizes can't block
        the open indefinitely.
        """
        peak = 0.0
        saw_dip = False
        recent: list[float] = []
        for i in range(_FOCUS_MAX_FRAMES):
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            peak = max(peak, sharp)
            recent.append(sharp)
            if len(recent) > 6:
                recent.pop(0)
            if peak > 0 and sharp < _FOCUS_DIP_RATIO * peak:
                saw_dip = True
            if i + 1 < _FOCUS_MIN_FRAMES:
                continue
            if peak <= 0:
                continue
            if saw_dip and sharp >= _FOCUS_RECOVER_RATIO * peak:
                return  # AF hunted and re-locked
            if (
                not saw_dip
                and len(recent) == 6
                and min(recent) >= _FOCUS_RECOVER_RATIO * max(recent)
            ):
                return  # steady near peak — fixed focus or already locked

    @staticmethod
    def _settle_resolution(
        cap: cv2.VideoCapture, resolution: tuple[int, int]
    ) -> None:
        """Read frames until the actual frame size matches ``resolution``.

        ``cap.set`` reconfigures the camera asynchronously, so a grab right
        after open can still return the old resolution (the BRIO is prone to
        handing back a cached 640x480 for the first few frames). Spin until
        the size matches or we exhaust the settle budget — at which point the
        caller's own resolution check decides what to do.
        """
        target_w, target_h = resolution
        for _ in range(_RESOLUTION_SETTLE_FRAMES):
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            h, w = frame.shape[:2]
            if (w, h) == (target_w, target_h):
                return

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
