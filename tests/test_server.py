"""Tests for the Anvil inspector FastAPI app.

Builds a fake .anvil/ tree in tmp_path and exercises the routes via
FastAPI's TestClient — no real network, no real browser.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from anvil.layers.layer1_fast import (
    compute_aruco_reference,
    compute_lighting_reference,
    save_aruco_reference,
    save_lighting_reference,
)
from anvil.manifest import (
    DIFFS_SUBDIR,
    CameraSpec,
    Manifest,
    save_manifest,
)
from anvil.schema import ANVIL_SCHEMA_VERSION
from anvil.server.app import create_app


def _write_pin(root: Path, name: str, color: tuple[int, int, int] = (20, 40, 60)) -> Path:
    pin_dir = root / name
    pin_dir.mkdir(parents=True, exist_ok=True)
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    frame[:, :] = color
    cv2.imwrite(str(pin_dir / "reference.png"), frame)
    save_lighting_reference(compute_lighting_reference(frame), pin_dir)
    save_aruco_reference(compute_aruco_reference(frame), pin_dir)
    manifest = Manifest(
        anvil_schema_version=ANVIL_SCHEMA_VERSION,
        name=name,
        task=f"task for {name}",
        pinned_at=datetime.now(tz=UTC),
        camera=CameraSpec(driver="file", device="ref.png", resolution=(64, 48)),
    )
    save_manifest(manifest, pin_dir)
    return pin_dir


def _write_diff(pin_dir: Path, filename: str = "diff_20260526T044827Z.png") -> Path:
    diffs = pin_dir / DIFFS_SUBDIR
    diffs.mkdir(parents=True, exist_ok=True)
    target = diffs / filename
    frame = np.full((48, 64, 3), 200, dtype=np.uint8)
    cv2.imwrite(str(target), frame)
    return target


def test_index_lists_pins(tmp_path: Path) -> None:
    root = tmp_path / ".anvil"
    _write_pin(root, "pick_red_cube")
    _write_pin(root, "workspace")
    client = TestClient(create_app(root))
    res = client.get("/")
    assert res.status_code == 200
    assert "pick_red_cube" in res.text
    assert "workspace" in res.text


def test_index_handles_empty_root(tmp_path: Path) -> None:
    root = tmp_path / ".anvil"
    root.mkdir()
    client = TestClient(create_app(root))
    res = client.get("/")
    assert res.status_code == 200
    assert "No pins found" in res.text


def test_index_handles_missing_root(tmp_path: Path) -> None:
    # Server should not crash if the root directory doesn't exist;
    # treats it as no pins.
    client = TestClient(create_app(tmp_path / "nope"))
    res = client.get("/")
    assert res.status_code == 200


def test_pin_detail_renders(tmp_path: Path) -> None:
    root = tmp_path / ".anvil"
    _write_pin(root, "workspace")
    client = TestClient(create_app(root))
    res = client.get("/pin/workspace")
    assert res.status_code == 200
    assert "workspace" in res.text
    # Manifest YAML toggle should appear.
    assert "manifest.yaml" in res.text


def test_pin_detail_returns_404_for_unknown(tmp_path: Path) -> None:
    root = tmp_path / ".anvil"
    root.mkdir()
    client = TestClient(create_app(root))
    res = client.get("/pin/no-such-pin")
    assert res.status_code == 404


def test_pin_detail_rejects_path_traversal(tmp_path: Path) -> None:
    root = tmp_path / ".anvil"
    root.mkdir()
    client = TestClient(create_app(root))
    res = client.get("/pin/..%2Fsecret")
    # Either 400 (sanitizer rejects) or 404 (no such pin) — both acceptable.
    assert res.status_code in (400, 404)


def test_reference_image_served(tmp_path: Path) -> None:
    root = tmp_path / ".anvil"
    _write_pin(root, "workspace")
    client = TestClient(create_app(root))
    res = client.get("/pin/workspace/reference.png")
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/png"
    assert len(res.content) > 0


def test_reference_image_404_for_missing(tmp_path: Path) -> None:
    root = tmp_path / ".anvil"
    (root / "workspace").mkdir(parents=True)
    client = TestClient(create_app(root))
    res = client.get("/pin/workspace/reference.png")
    assert res.status_code == 404


def test_diff_image_served(tmp_path: Path) -> None:
    root = tmp_path / ".anvil"
    pin_dir = _write_pin(root, "workspace")
    _write_diff(pin_dir)
    client = TestClient(create_app(root))
    res = client.get("/pin/workspace/diff/diff_20260526T044827Z.png")
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/png"


def test_diff_route_rejects_arbitrary_filenames(tmp_path: Path) -> None:
    root = tmp_path / ".anvil"
    _write_pin(root, "workspace")
    client = TestClient(create_app(root))
    res = client.get("/pin/workspace/diff/anything.png")
    assert res.status_code == 400


def test_pin_detail_includes_diff_gallery(tmp_path: Path) -> None:
    root = tmp_path / ".anvil"
    pin_dir = _write_pin(root, "workspace")
    _write_diff(pin_dir, "diff_20260526T040000Z.png")
    _write_diff(pin_dir, "diff_20260526T050000Z.png")
    client = TestClient(create_app(root))
    res = client.get("/pin/workspace")
    assert res.status_code == 200
    # Newest diff first.
    text = res.text
    assert text.index("050000Z") < text.index("040000Z")
