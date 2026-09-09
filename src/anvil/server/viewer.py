"""FastAPI app factory for the Anvil viewer — ``anvil viewer``.

A display layer over what anvil already writes. It reads pins through the same
loaders the rest of the package uses, renders them, and writes nothing.

One page, ``/``: saved check records, newest first — the verdict, the scores
against the pin's thresholds, the findings, and the frames. ``anvil check``
only writes a record when given ``--save``; without any, the page says so.

The inspector at ``anvil-inspect`` shows how a pin was *configured*. This shows
what the pin *sees*.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates

from anvil.manifest import (
    DEFAULT_PIN_ROOT,
    REFERENCE_IMAGE_FILENAME,
    load_manifest,
)

_TEMPLATES_DIR = Path(__file__).parent / "templates"
def _safe_path(base: Path, *parts: str) -> Path:
    """``base`` joined with ``parts``, refusing anything that escapes it.

    Pin names come from directory names on disk and anvil does not restrict
    them, so an alphabet whitelist would 400 on a perfectly valid unicode pin.
    What actually matters is that the resolved path stays inside ``base`` —
    which also covers a symlink pointing out of the tree.
    """
    for part in parts:
        if not part or part in {".", ".."} or {"/", "\\", "\x00"} & set(part):
            raise HTTPException(status_code=400, detail=f"invalid name {part!r}")
    root = base.resolve()
    path = (root / Path(*parts)).resolve()
    if not path.is_relative_to(root):
        raise HTTPException(status_code=400, detail="path escapes the root")
    return path


def _measures(root: Path, pin: str) -> dict[str, bool]:
    """Which layers this pin can actually report on.

    A score of 0.00 means "no drift" only if something measured it. On a bare
    install `max_camera_pose_drift_deg` is 0.00 because no layer ran, and the
    page must not read that as "the camera hasn't moved".
    """
    try:
        m = load_manifest(root / pin)
    except Exception:
        return {"pose": False, "scene": False}
    return {
        "pose": m.aruco_present or m.keypoints_present,
        "scene": m.embedding_present,
    }


def _thresholds(root: Path, pin: str) -> dict[str, float]:
    try:
        t = load_manifest(root / pin).thresholds
    except Exception:
        return {}
    return {
        "scene": t.scene_drift,
        "lighting": t.lighting_drift,
        "pose_deg": t.max_camera_pose_drift_deg,
    }


_STATUSES = frozenset({"PASSED", "WARNING", "FAILED"})


def _status(value: object) -> str:
    """One of the known verdicts, or ``UNREADABLE`` for anything else."""
    text = str(value).upper() if value is not None else ""
    return text if text in _STATUSES else "UNREADABLE"


def _checked_at(value: object, cid: str) -> str:
    """An ISO timestamp the page can parse, from the report or the directory."""
    if isinstance(value, str):
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            pass
        else:
            return value
    return _stamp_from_id(cid)


def _stamp_from_id(cid: str) -> str:
    """``20260101T000000Z_trial_01`` -> an ISO timestamp, or the id unchanged."""
    head = cid.split("_", 1)[0]
    try:
        return datetime.strptime(head, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC).isoformat()
    except ValueError:
        return cid


def _record_dirs(root: Path, checks_root: Path | None) -> list[tuple[str, Path]]:
    """(pin name, record directory) for every saved check we can find.

    Two places, so this works whatever wrote the records:

    * ``<root>/<pin>/checks/<id>/`` — where ``anvil check --save`` puts them.
    * ``<checks_root>/<pin>/<id>/`` — an external tree, for callers that keep
      their own (``--checks``).
    """
    found: dict[tuple[str, str], Path] = {}
    for base, sub in ((root, "checks"), (checks_root, None)):
        if base is None or not base.is_dir():
            continue
        for pin_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            holder = pin_dir / sub if sub else pin_dir
            if not holder.is_dir():
                continue
            for d in holder.iterdir():
                # Keyed by (pin, id) so a record reachable from both roots is
                # listed once; the pin's own copy is read first and wins.
                if d.is_dir():
                    found.setdefault((pin_dir.name, d.name), d)
    return [(pin, d) for (pin, _), d in found.items()]


def _checks(root: Path, checks_root: Path | None) -> list[dict[str, Any]]:
    """Saved check records, newest first. Empty when nothing saved any."""
    items: list[dict[str, Any]] = []
    thresholds: dict[str, dict[str, float]] = {}
    measures: dict[str, dict[str, bool]] = {}
    for pin, d in _record_dirs(root, checks_root):
        report = d / "report.json"
        if not report.is_file():
            continue
        if pin not in thresholds:
            thresholds[pin] = _thresholds(root, pin)
            measures[pin] = _measures(root, pin)
        thr = thresholds[pin]
        has_ref = (root / pin / REFERENCE_IMAGE_FILENAME).exists()
        try:
            r = json.loads(report.read_text())
        except Exception as exc:
            r = {"error": str(exc)}
        if not isinstance(r, dict):
            # Valid JSON, wrong shape — a list or a bare null would otherwise
            # crash every field lookup below and take the whole page with it.
            r = {"error": "report.json is not an object"}

        def url(fn: str, _pin: str = pin, _d: Path = d) -> str | None:
            return f"/checks/{_pin}/{_d.name}/{fn}" if (_d / fn).is_file() else None

        items.append({
            "pin": pin,
            "id": d.name,
            # Constrained to the values the page styles; a report is a file on
            # disk and anything unexpected in it is data, not a class name.
            "status": _status(r.get("status")),
            # The page hands this to Date(); anything it cannot parse would
            # render as "Invalid Date". The directory name carries a usable
            # timestamp, so fall back to it.
            "checked_at": _checked_at(r.get("checked_at"), d.name),
            "episode_id": r.get("episode_id", ""),
            "scores": r.get("scores", {}),
            "findings": r.get("findings", []),
            "operator_action": r.get("operator_action", ""),
            "frame": url("frame.png"),
            "diff": url("diff.png"),
            "heat": url("heat.png") or url("heat_m1.png"),
            "reference": f"/pins/{pin}/reference.png" if has_ref else None,
            "thresholds": thr,
            "measures": measures[pin],
        })
    items.sort(key=lambda c: c["id"], reverse=True)
    return items


def create_app(root: Path | None = None, checks: Path | None = None) -> FastAPI:
    """Build the viewer app. ``root`` is a pin root; ``checks`` is optional."""
    pin_root = (root or Path.cwd() / DEFAULT_PIN_ROOT).resolve()
    checks_root = checks.resolve() if checks else None
    app = FastAPI(title="Anvil Viewer", version="0.1.0")
    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> Response:
        return templates.TemplateResponse(
            request, "viewer.html", {"checks": _checks(pin_root, checks_root)}
        )

    @app.get("/api")
    def api(latest: int = 0) -> Response:
        if latest:
            # The page polls this every few seconds only to notice a new check.
            # Reading every report to answer it would scale with history; the
            # newest record's directory name is enough.
            ids = [d.name for _, d in _record_dirs(pin_root, checks_root)
                   if (d / "report.json").is_file()]
            return JSONResponse({"latest": max(ids) if ids else ""})
        return JSONResponse(_checks(pin_root, checks_root))

    @app.get("/pins/{pin}/reference.png")
    def reference(pin: str) -> FileResponse:
        path = _safe_path(pin_root, pin, REFERENCE_IMAGE_FILENAME)
        if not path.is_file():
            raise HTTPException(status_code=404, detail="no reference image")
        return FileResponse(path, media_type="image/png")

    @app.get("/checks/{pin}/{cid}/{filename}")
    def check_image(pin: str, cid: str, filename: str) -> FileResponse:
        if not filename.endswith(".png"):
            raise HTTPException(status_code=404, detail="not found")
        # Records live in either place — see _record_dirs.
        candidates = [_safe_path(pin_root, pin, "checks", cid, filename)]
        if checks_root is not None:
            candidates.append(_safe_path(checks_root, pin, cid, filename))
        for path in candidates:
            if path.is_file():
                return FileResponse(path, media_type="image/png")
        raise HTTPException(status_code=404, detail="not found")

    return app
