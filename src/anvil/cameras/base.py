"""Camera driver abstraction.

Frames are returned as ``numpy.ndarray`` in OpenCV's native BGR uint8 layout
(``H x W x 3``). Drivers are context managers — open and close are explicit
because real hardware can fail to acquire, and the user should see that
failure as a clean ``CameraError`` rather than a half-initialized state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from types import TracebackType

import numpy as np
from numpy.typing import NDArray

Frame = NDArray[np.uint8]


class Camera(AbstractContextManager["Camera"], ABC):
    """Abstract camera. Implementations MUST return BGR uint8 frames."""

    @abstractmethod
    def open(self) -> None:
        """Acquire the underlying device. Raises :class:`CameraError` on failure."""

    @abstractmethod
    def grab(self) -> Frame:
        """Read one frame. Raises :class:`CameraError` if the device is not open."""

    @abstractmethod
    def close(self) -> None:
        """Release the device. Idempotent."""

    def __enter__(self) -> Camera:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
