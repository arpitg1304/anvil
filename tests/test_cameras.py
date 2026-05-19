"""Camera driver tests (file-backed only — webcam is hardware).

The webcam driver is exercised via type-check and the CLI integration
test below. Live webcam open/grab is not tested in CI.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from anvil.cameras import FileCamera, open_camera
from anvil.cameras.webcam import WebcamCamera
from anvil.errors import CameraError


def _write_image(path: Path, color: tuple[int, int, int] = (10, 20, 30)) -> Path:
    img = np.full((48, 64, 3), 0, dtype=np.uint8)
    img[:, :] = color  # BGR
    assert cv2.imwrite(str(path), img)
    return path


# --- FileCamera ----------------------------------------------------------


def test_file_camera_single_image(tmp_path: Path) -> None:
    img_path = _write_image(tmp_path / "ref.png")
    with FileCamera(img_path) as cam:
        frame = cam.grab()
    assert frame.shape == (48, 64, 3)
    assert frame.dtype == np.uint8


def test_file_camera_single_image_repeats(tmp_path: Path) -> None:
    img_path = _write_image(tmp_path / "ref.png")
    with FileCamera(img_path) as cam:
        a = cam.grab()
        b = cam.grab()
    assert np.array_equal(a, b)


def test_file_camera_directory_cycles(tmp_path: Path) -> None:
    _write_image(tmp_path / "a.png", color=(1, 2, 3))
    _write_image(tmp_path / "b.png", color=(4, 5, 6))
    with FileCamera(tmp_path) as cam:
        a = cam.grab()
        b = cam.grab()
        a_again = cam.grab()  # wraps back to a
    # a and a_again must match
    assert np.array_equal(a, a_again)
    # a and b are visibly different
    assert not np.array_equal(a, b)


def test_file_camera_missing_path_raises() -> None:
    with pytest.raises(CameraError):
        FileCamera("/nonexistent/path.png").open()


def test_file_camera_empty_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(CameraError):
        FileCamera(tmp_path).open()


def test_file_camera_grab_before_open_raises(tmp_path: Path) -> None:
    img_path = _write_image(tmp_path / "ref.png")
    cam = FileCamera(img_path)
    with pytest.raises(CameraError):
        cam.grab()


def test_file_camera_close_is_idempotent(tmp_path: Path) -> None:
    img_path = _write_image(tmp_path / "ref.png")
    cam = FileCamera(img_path)
    cam.open()
    cam.close()
    cam.close()  # must not raise


# --- factory -------------------------------------------------------------


def test_open_camera_returns_file_camera(tmp_path: Path) -> None:
    img_path = _write_image(tmp_path / "ref.png")
    cam = open_camera("file", str(img_path))
    assert isinstance(cam, FileCamera)


def test_open_camera_returns_webcam() -> None:
    cam = open_camera("webcam", "0", resolution=(640, 480))
    assert isinstance(cam, WebcamCamera)
    # Don't actually .open() — that would attach to a real device.


def test_open_camera_rejects_unknown_driver() -> None:
    with pytest.raises(ValueError):
        open_camera("realsense", "0")
