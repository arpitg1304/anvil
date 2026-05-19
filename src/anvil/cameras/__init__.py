"""Camera drivers + factory."""

from __future__ import annotations

from anvil.cameras.base import Camera, Frame
from anvil.cameras.file import FileCamera
from anvil.cameras.webcam import WebcamCamera


def open_camera(
    driver: str,
    device: str,
    resolution: tuple[int, int] | None = None,
) -> Camera:
    """Resolve a driver name to a Camera instance. Caller must ``.open()``
    or use the returned object as a context manager.
    """
    if driver == "webcam":
        return WebcamCamera(device, resolution)
    if driver == "file":
        return FileCamera(device)
    raise ValueError(f"Unknown camera driver: {driver!r}")


__all__ = ["Camera", "FileCamera", "Frame", "WebcamCamera", "open_camera"]
