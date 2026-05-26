"""FastAPI app factory for the Anvil inspector.

Reads from a ``.anvil/`` directory at request time — no caching, no DB.
Every page is server-rendered HTML via Jinja2 + Tailwind via CDN, no
build step. Read-only on purpose: the inspector doesn't run checks,
doesn't write pins, doesn't accept input. It shows what's already there.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.templating import Jinja2Templates

from anvil.layers.layer2_structural import load_object_references
from anvil.manifest import (
    DEFAULT_PIN_ROOT,
    DIFFS_SUBDIR,
    MANIFEST_FILENAME,
    REFERENCE_IMAGE_FILENAME,
    load_manifest,
)

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_DIFF_FILENAME_PATTERN = re.compile(r"^diff_\d{8}T\d{6}Z\.png$")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


def create_app(root: Path | None = None) -> FastAPI:
    """Build the FastAPI app rooted at ``root`` (defaults to ``./.anvil``).

    The root is resolved at app-creation time; callers can rebuild the app
    against a different directory by calling ``create_app`` again.
    """
    pin_root = (root or Path.cwd() / DEFAULT_PIN_ROOT).resolve()
    app = FastAPI(title="Anvil Inspector", version="0.1.0")
    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> Response:
        return templates.TemplateResponse(
            request,
            "index.html",
            {"pins": _list_pins(pin_root), "root": pin_root},
        )

    @app.get("/pin/{name}", response_class=HTMLResponse)
    def pin_detail(request: Request, name: str) -> Response:
        _check_name(name)
        view = _load_pin_view(pin_root, name)
        if view is None:
            raise HTTPException(status_code=404, detail=f"Pin {name!r} not found")
        return templates.TemplateResponse(request, "pin_detail.html", {"pin": view})

    @app.get("/pin/{name}/reference.png")
    def pin_reference(name: str) -> FileResponse:
        _check_name(name)
        path = pin_root / name / REFERENCE_IMAGE_FILENAME
        if not path.exists():
            raise HTTPException(status_code=404, detail="reference image missing")
        return FileResponse(path, media_type="image/png")

    @app.get("/pin/{name}/diff/{filename}")
    def pin_diff(name: str, filename: str) -> FileResponse:
        _check_name(name)
        if not _DIFF_FILENAME_PATTERN.match(filename):
            raise HTTPException(status_code=400, detail="invalid diff filename")
        path = pin_root / name / DIFFS_SUBDIR / filename
        if not path.exists():
            raise HTTPException(status_code=404, detail="diff not found")
        return FileResponse(path, media_type="image/png")

    return app


# --- helpers --------------------------------------------------------------


def _check_name(name: str) -> None:
    """Reject pin names that aren't safe path segments.

    The same rule ``anvil.manifest.pin_directory`` enforces — keeps a
    malicious user from sneaking ``..`` or absolute paths into URLs.
    """
    if not _SAFE_NAME.match(name) or name in {".", ".."}:
        raise HTTPException(status_code=400, detail=f"invalid pin name {name!r}")


def _list_pins(root: Path) -> list[dict[str, Any]]:
    """Return one summary dict per pin under ``root``, newest pinned first."""
    if not root.exists():
        return []
    out: list[dict[str, Any]] = []
    for pin_dir in sorted(root.iterdir()):
        if not pin_dir.is_dir():
            continue
        manifest_path = pin_dir / MANIFEST_FILENAME
        if not manifest_path.exists():
            continue
        try:
            manifest = load_manifest(pin_dir)
        except Exception:
            continue
        diff_count = sum(
            1
            for p in (pin_dir / DIFFS_SUBDIR).glob("diff_*.png")
            if p.is_file()
        ) if (pin_dir / DIFFS_SUBDIR).exists() else 0
        out.append(
            {
                "name": manifest.name,
                "task": manifest.task,
                "pinned_at": manifest.pinned_at.isoformat(),
                "aruco_present": manifest.aruco_present,
                "embedding_present": manifest.embedding_present,
                "keypoints_present": manifest.keypoints_present,
                "objects": list(manifest.objects),
                "diff_count": diff_count,
            }
        )
    out.sort(key=lambda d: d["pinned_at"], reverse=True)
    return out


def _load_pin_view(root: Path, name: str) -> dict[str, Any] | None:
    """Full per-pin payload for the detail template."""
    pin_dir = root / name
    manifest_path = pin_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        return None
    manifest = load_manifest(pin_dir)
    objects = load_object_references(pin_dir)
    manifest_yaml = yaml.safe_dump(
        manifest.model_dump(mode="json"),
        sort_keys=False,
        default_flow_style=False,
    )
    diffs = _list_diffs(pin_dir)
    return {
        "name": manifest.name,
        "task": manifest.task,
        "pinned_at": manifest.pinned_at.isoformat(),
        "resolution": (
            f"{manifest.camera.resolution[0]}x{manifest.camera.resolution[1]}"
            if manifest.camera.resolution
            else "unknown"
        ),
        "camera_driver": manifest.camera.driver,
        "camera_device": manifest.camera.device,
        "aruco_present": manifest.aruco_present,
        "embedder_name": manifest.embedder_name or "histogram (fallback)",
        "keypoint_detector_name": manifest.keypoint_detector_name or "none",
        "objects": [
            {
                "name": o.name,
                "prompt": o.prompt,
                "bbox": [round(v) for v in o.bbox],
                "confidence": round(o.confidence, 3),
                "detector_name": o.detector_name,
            }
            for o in objects
        ],
        "manifest_yaml": manifest_yaml,
        "diffs": diffs,
    }


def _list_diffs(pin_dir: Path) -> list[dict[str, str]]:
    """Diff PNGs in ``<pin_dir>/diffs/`` sorted newest-first."""
    diffs_dir = pin_dir / DIFFS_SUBDIR
    if not diffs_dir.exists():
        return []
    out: list[dict[str, str]] = []
    for path in diffs_dir.glob("diff_*.png"):
        if not _DIFF_FILENAME_PATTERN.match(path.name):
            continue
        timestamp = _parse_diff_timestamp(path.name)
        out.append(
            {
                "filename": path.name,
                "timestamp": timestamp,
            }
        )
    out.sort(key=lambda d: d["filename"], reverse=True)
    return out


def _parse_diff_timestamp(filename: str) -> str:
    """Pretty-print the ISO timestamp encoded in a diff filename."""
    # diff_20260526T044827Z.png → 2026-05-26 04:48:27 UTC
    stem = filename.removeprefix("diff_").removesuffix(".png")
    try:
        dt = datetime.strptime(stem, "%Y%m%dT%H%M%SZ")
    except ValueError:
        return stem
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


__all__ = ["create_app"]
