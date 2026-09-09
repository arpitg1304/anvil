"""Tests for the Anvil viewer FastAPI app and the check-record writer.

Builds a fake .anvil/ tree in tmp_path and exercises the routes via
FastAPI's TestClient — no real network, no real browser.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from anvil.cli import CHECKS_SUBDIR, save_check_record
from anvil.layers.layer1_fast import (
    compute_aruco_reference,
    compute_lighting_reference,
    save_aruco_reference,
    save_lighting_reference,
)
from anvil.manifest import CameraSpec, Manifest, save_manifest
from anvil.schema import (
    ANVIL_SCHEMA_VERSION,
    EpisodeReport,
    Finding,
    PinReference,
    Provenance,
    Scores,
)
from anvil.server.viewer import create_app

_SHAPE = (48, 64, 3)


def _write_pin(root: Path, name: str, *, can_measure_pose: bool = False) -> Path:
    pin_dir = root / name
    pin_dir.mkdir(parents=True, exist_ok=True)
    frame = np.full(_SHAPE, 40, dtype=np.uint8)
    cv2.imwrite(str(pin_dir / "reference.png"), frame)
    # `guard` runs a real check against this pin, which needs the layer-1
    # references the pin command would have written.
    save_lighting_reference(compute_lighting_reference(frame), pin_dir)
    save_aruco_reference(compute_aruco_reference(frame), pin_dir)
    save_manifest(
        Manifest(
            anvil_schema_version=ANVIL_SCHEMA_VERSION,
            name=name,
            task=f"task for {name}",
            pinned_at=datetime.now(tz=UTC),
            camera=CameraSpec(driver="file", device="ref.png", resolution=(64, 48)),
            keypoints_present=can_measure_pose,
        ),
        pin_dir,
    )
    return pin_dir


def _report(pin: str, episode: str = "trial_01", status: str = "passed") -> EpisodeReport:
    return EpisodeReport(
        anvil_schema_version=ANVIL_SCHEMA_VERSION,
        episode_id=episode,
        checked_at=datetime.now(tz=UTC),
        pin=PinReference(
            name=pin,
            manifest_hash="sha256:" + "0" * 64,
            pinned_at=datetime.now(tz=UTC),
        ),
        status=status,  # type: ignore[arg-type]
        operator_action="proceeded",
        scores=Scores(
            scene_drift=0.01,
            lighting_drift=0.02,
            max_object_drift_cm=0.0,
            max_camera_pose_drift_deg=0.0,
            max_robot_joint_drift_deg=0.0,
        ),
        findings=[],
        provenance=Provenance(anvil_version="0.1.0.dev0"),
    )


def _write_record(pin_dir: Path, cid: str, *, status: str = "passed") -> Path:
    """A record as `anvil check --save` writes it: report + frame + heat."""
    out = pin_dir / CHECKS_SUBDIR / cid
    out.mkdir(parents=True, exist_ok=True)
    report = _report(pin_dir.name, episode=cid, status=status)
    (out / "report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    for fn in ("frame.png", "heat.png"):
        cv2.imwrite(str(out / fn), np.full(_SHAPE, 90, dtype=np.uint8))
    return out


# --- the page --------------------------------------------------------------


def test_index_renders_with_no_pins(tmp_path: Path) -> None:
    """A viewer pointed at nothing still serves a page that explains itself."""
    client = TestClient(create_app(tmp_path / ".anvil"))
    res = client.get("/")
    assert res.status_code == 200
    assert "No saved checks" in res.text


def test_index_says_how_to_save_when_pins_have_no_records(tmp_path: Path) -> None:
    root = tmp_path / ".anvil"
    _write_pin(root, "cell")
    res = TestClient(create_app(root)).get("/")
    assert res.status_code == 200
    assert "--save" in res.text, "the empty state must say how to record a check"


def test_records_in_the_pin_dir_are_found_without_a_checks_flag(tmp_path: Path) -> None:
    """`check --save` writes inside the pin, so no --checks should be needed."""
    root = tmp_path / ".anvil"
    _write_record(_write_pin(root, "cell"), "20260101T000000Z_trial_01")
    body = TestClient(create_app(root)).get("/api").json()
    assert [c["pin"] for c in body] == ["cell"]
    assert body[0]["frame"] is not None
    assert body[0]["heat"] is not None
    assert body[0]["reference"] == "/pins/cell/reference.png"


def test_external_checks_directory_is_merged_in(tmp_path: Path) -> None:
    """A caller that keeps its own records can point --checks at them."""
    root = tmp_path / ".anvil"
    _write_pin(root, "cell")
    external = tmp_path / "checks" / "cell" / "20260101T000000Z_trial_09"
    external.mkdir(parents=True)
    (external / "report.json").write_text(_report("cell", "trial_09").model_dump_json())
    body = TestClient(create_app(root, tmp_path / "checks")).get("/api").json()
    assert [c["episode_id"] for c in body] == ["trial_09"]


def test_checks_are_newest_first(tmp_path: Path) -> None:
    pin_dir = _write_pin(tmp_path / ".anvil", "cell")
    for cid in ("20260101T000000Z_a", "20260102T000000Z_b", "20260103T000000Z_c"):
        _write_record(pin_dir, cid)
    body = TestClient(create_app(tmp_path / ".anvil")).get("/api").json()
    assert [c["id"][-1] for c in body] == ["c", "b", "a"]


def test_latest_endpoint_reports_the_newest_id(tmp_path: Path) -> None:
    """The page polls this to decide whether to reload."""
    pin_dir = _write_pin(tmp_path / ".anvil", "cell")
    _write_record(pin_dir, "20260101T000000Z_a")
    _write_record(pin_dir, "20260102T000000Z_b")
    client = TestClient(create_app(tmp_path / ".anvil"))
    assert client.get("/api?latest=1").json()["latest"] == "20260102T000000Z_b"


def test_latest_is_empty_with_no_records(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path / ".anvil"))
    assert client.get("/api?latest=1").json()["latest"] == ""


def test_unreadable_report_does_not_break_the_page(tmp_path: Path) -> None:
    """One corrupt record must not hide every other check."""
    pin_dir = _write_pin(tmp_path / ".anvil", "cell")
    _write_record(pin_dir, "20260101T000000Z_good")
    bad = pin_dir / CHECKS_SUBDIR / "20260102T000000Z_bad"
    bad.mkdir(parents=True)
    (bad / "report.json").write_text("{not json")
    body = TestClient(create_app(tmp_path / ".anvil")).get("/api").json()
    assert len(body) == 2
    assert body[0]["status"] == "UNREADABLE"


# --- images ----------------------------------------------------------------


def test_serves_reference_and_record_images(tmp_path: Path) -> None:
    root = tmp_path / ".anvil"
    _write_record(_write_pin(root, "cell"), "20260101T000000Z_trial_01")
    client = TestClient(create_app(root))
    for url in (
        "/pins/cell/reference.png",
        "/checks/cell/20260101T000000Z_trial_01/frame.png",
        "/checks/cell/20260101T000000Z_trial_01/heat.png",
    ):
        res = client.get(url)
        assert res.status_code == 200, url
        assert res.headers["content-type"] == "image/png"


def test_missing_image_is_404_not_500(tmp_path: Path) -> None:
    root = tmp_path / ".anvil"
    _write_record(_write_pin(root, "cell"), "20260101T000000Z_trial_01")
    client = TestClient(create_app(root))
    assert client.get("/checks/cell/20260101T000000Z_trial_01/diff.png").status_code == 404
    assert client.get("/pins/nope/reference.png").status_code == 404


def test_non_png_is_refused(tmp_path: Path) -> None:
    """Only images are served — never a report or a manifest."""
    root = tmp_path / ".anvil"
    _write_record(_write_pin(root, "cell"), "20260101T000000Z_trial_01")
    res = TestClient(create_app(root)).get("/checks/cell/20260101T000000Z_trial_01/report.json")
    assert res.status_code in (400, 404)


def test_path_traversal_is_rejected(tmp_path: Path) -> None:
    """Same rule the inspector enforces: a pin name is one safe path segment."""
    root = tmp_path / ".anvil"
    _write_pin(root, "cell")
    secret = tmp_path / "secret.png"
    cv2.imwrite(str(secret), np.zeros(_SHAPE, dtype=np.uint8))
    client = TestClient(create_app(root))
    for url in (
        "/pins/..%2F..%2Fsecret/reference.png",
        "/checks/../../x/y/frame.png",
        "/checks/cell/..%2F..%2F..%2Fsecret/frame.png",
    ):
        assert client.get(url).status_code in (400, 404), url


# --- the record writer -----------------------------------------------------


def test_save_check_record_writes_report_frame_and_heat(tmp_path: Path) -> None:
    pin_dir = _write_pin(tmp_path / ".anvil", "cell")
    frame = np.full(_SHAPE, 120, dtype=np.uint8)
    out = save_check_record(pin_dir, _report("cell"), frame)

    assert out.parent == pin_dir / CHECKS_SUBDIR
    assert (out / "frame.png").is_file()
    assert (out / "heat.png").is_file(), "a heat map is what makes a PASS reviewable"
    saved = json.loads((out / "report.json").read_text())
    assert saved["episode_id"] == "trial_01"
    assert saved["anvil_schema_version"] == ANVIL_SCHEMA_VERSION


def test_save_check_record_copies_the_annotated_diff_when_there_is_one(
    tmp_path: Path,
) -> None:
    """anvil only renders a diff for a non-passing check; keep it with the record."""
    pin_dir = _write_pin(tmp_path / ".anvil", "cell")
    diff = tmp_path / "diff_20260101T000000Z.png"
    cv2.imwrite(str(diff), np.full(_SHAPE, 200, dtype=np.uint8))
    out = save_check_record(pin_dir, _report("cell"), np.zeros(_SHAPE, np.uint8), diff)
    assert (out / "diff.png").is_file()


def test_save_check_record_survives_a_missing_reference(tmp_path: Path) -> None:
    """No reference image means no heat map — but still a usable record."""
    pin_dir = tmp_path / ".anvil" / "cell"
    pin_dir.mkdir(parents=True)
    out = save_check_record(pin_dir, _report("cell"), np.zeros(_SHAPE, np.uint8))
    assert (out / "report.json").is_file()
    assert not (out / "heat.png").exists()


def test_saved_record_is_visible_to_the_viewer(tmp_path: Path) -> None:
    """The end-to-end contract: what --save writes is what the page reads."""
    root = tmp_path / ".anvil"
    pin_dir = _write_pin(root, "cell")
    save_check_record(pin_dir, _report("cell", "trial_42"), np.full(_SHAPE, 77, np.uint8))
    body = TestClient(create_app(root)).get("/api").json()
    assert len(body) == 1
    assert body[0]["episode_id"] == "trial_42"
    assert TestClient(create_app(root)).get(body[0]["heat"]).status_code == 200


def test_findings_reach_the_payload(tmp_path: Path) -> None:
    """The page turns findings into its plain-language lines."""
    pin_dir = _write_pin(tmp_path / ".anvil", "cell")
    report = _report("cell", status="warning")
    report = report.model_copy(
        update={
            "findings": [
                Finding(
                    id="f_001",
                    severity="warning",
                    layer=1,
                    component="lighting",
                    subject="scene",
                    issue="lighting_shift",
                    detail="Lighting drifted",
                    fix="Check the overhead lamp.",
                    evidence={"color_temp_delta_k": -1200.0},
                )
            ]
        }
    )
    save_check_record(pin_dir, report, np.zeros(_SHAPE, np.uint8))
    body = TestClient(create_app(tmp_path / ".anvil")).get("/api").json()
    assert body[0]["status"] == "WARNING"
    assert body[0]["findings"][0]["component"] == "lighting"
    assert body[0]["findings"][0]["evidence"]["color_temp_delta_k"] == -1200.0


def test_a_record_in_both_roots_is_listed_once(tmp_path: Path) -> None:
    """Migrating from an external tree to --save must not double every check."""
    root = tmp_path / ".anvil"
    pin_dir = _write_pin(root, "cell")
    cid = "20260101T000000Z_trial_01"
    _write_record(pin_dir, cid)
    external = tmp_path / "checks" / "cell" / cid
    external.mkdir(parents=True)
    (external / "report.json").write_text(_report("cell", "trial_01").model_dump_json())
    body = TestClient(create_app(root, tmp_path / "checks")).get("/api").json()
    assert len(body) == 1


def test_report_without_a_timestamp_still_renders(tmp_path: Path) -> None:
    """The id carries a time; hand the page something Date() can parse."""
    pin_dir = _write_pin(tmp_path / ".anvil", "cell")
    out = pin_dir / CHECKS_SUBDIR / "20260101T000000Z_trial_01"
    out.mkdir(parents=True)
    (out / "report.json").write_text('{"status": "passed"}')
    body = TestClient(create_app(tmp_path / ".anvil")).get("/api").json()
    assert body[0]["checked_at"].startswith("2026-01-01T00:00:00")


def test_hostile_text_in_a_report_cannot_break_out_of_the_script(tmp_path: Path) -> None:
    """Check payloads are rendered by `tojson`, never spliced into the page."""
    pin_dir = _write_pin(tmp_path / ".anvil", "cell")
    out = pin_dir / CHECKS_SUBDIR / "20260101T000000Z_trial_01"
    out.mkdir(parents=True)
    (out / "report.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "checked_at": "2026-01-01T00:00:00Z",
                "episode_id": "</script><img src=x onerror=alert(1)>",
                "scores": {},
            }
        )
    )
    text = TestClient(create_app(tmp_path / ".anvil")).get("/").text
    assert "</script><img" not in text


def test_a_pin_that_cannot_measure_pose_says_so(tmp_path: Path) -> None:
    """A bare install scores pose 0.00 because nothing ran — never claim it held still."""
    root = tmp_path / ".anvil"
    _write_record(_write_pin(root, "cell"), "20260101T000000Z_trial_01")
    body = TestClient(create_app(root)).get("/api").json()
    assert body[0]["measures"]["pose"] is False


def test_a_pin_with_keypoints_reports_pose_as_measured(tmp_path: Path) -> None:
    root = tmp_path / ".anvil"
    pin_dir = _write_pin(root, "cell", can_measure_pose=True)
    _write_record(pin_dir, "20260101T000000Z_trial_01")
    body = TestClient(create_app(root)).get("/api").json()
    assert body[0]["measures"]["pose"] is True


def test_a_hostile_episode_id_cannot_escape_the_checks_directory(tmp_path: Path) -> None:
    """`episode_id` is operator-supplied and unconstrained by the schema."""
    pin_dir = _write_pin(tmp_path / ".anvil", "cell")
    report = _report("cell").model_copy(update={"episode_id": "../../../escaped"})
    out = save_check_record(pin_dir, report, np.zeros(_SHAPE, np.uint8))
    assert (pin_dir / CHECKS_SUBDIR) in out.parents
    assert not (tmp_path / "escaped").exists()
    assert ".." not in out.name


def test_two_checks_in_the_same_second_keep_both_records(tmp_path: Path) -> None:
    """The default episode id is constant, so the stamp alone is not unique."""
    pin_dir = _write_pin(tmp_path / ".anvil", "cell")
    stamped = _report("cell", "manual_check")
    first = save_check_record(pin_dir, stamped, np.zeros(_SHAPE, np.uint8))
    second = save_check_record(pin_dir, stamped, np.full(_SHAPE, 9, np.uint8))
    assert first != second
    assert first.is_dir() and second.is_dir()
    body = TestClient(create_app(tmp_path / ".anvil")).get("/api").json()
    assert len(body) == 2


def test_an_empty_episode_id_still_produces_a_usable_name(tmp_path: Path) -> None:
    pin_dir = _write_pin(tmp_path / ".anvil", "cell")
    report = _report("cell").model_copy(update={"episode_id": "..."})
    out = save_check_record(pin_dir, report, np.zeros(_SHAPE, np.uint8))
    assert out.name.endswith("_check")


def test_an_unknown_status_cannot_become_a_class_name(tmp_path: Path) -> None:
    """A report is a file on disk; its `status` must never reach the DOM raw."""
    pin_dir = _write_pin(tmp_path / ".anvil", "cell")
    out = pin_dir / CHECKS_SUBDIR / "20260101T000000Z_trial_01"
    out.mkdir(parents=True)
    (out / "report.json").write_text(
        json.dumps(
            {
                "status": 'passed" onmouseover="alert(1)',
                "checked_at": "2026-01-01T00:00:00Z",
                "episode_id": "e",
                "scores": {},
            }
        )
    )
    client = TestClient(create_app(tmp_path / ".anvil"))
    assert client.get("/api").json()[0]["status"] == "UNREADABLE"
    assert "onmouseover" not in client.get("/").text


def test_latest_does_not_depend_on_reading_every_report(tmp_path: Path) -> None:
    """The poll runs every few seconds; it must not scale with history."""
    pin_dir = _write_pin(tmp_path / ".anvil", "cell")
    for cid in ("20260101T000000Z_a", "20260103T000000Z_c", "20260102T000000Z_b"):
        _write_record(pin_dir, cid)
    client = TestClient(create_app(tmp_path / ".anvil"))
    assert client.get("/api?latest=1").json()["latest"] == "20260103T000000Z_c"


def test_a_unicode_pin_name_still_serves_its_images(tmp_path: Path) -> None:
    """anvil does not restrict pin names, so the viewer must not either."""
    root = tmp_path / ".anvil"
    _write_record(_write_pin(root, "café_日本"), "20260101T000000Z_trial_01")
    client = TestClient(create_app(root))
    assert client.get("/pins/café_日本/reference.png").status_code == 200
    assert (
        client.get("/checks/café_日本/20260101T000000Z_trial_01/frame.png").status_code
        == 200
    )


def test_a_symlink_out_of_the_root_is_refused(tmp_path: Path) -> None:
    """Containment is checked after resolving, so a link cannot escape."""
    root = tmp_path / ".anvil"
    pin_dir = _write_pin(root, "cell")
    out = pin_dir / CHECKS_SUBDIR / "20260101T000000Z_trial_01"
    out.mkdir(parents=True)
    secret = tmp_path / "secret.png"
    cv2.imwrite(str(secret), np.zeros(_SHAPE, dtype=np.uint8))
    (out / "leak.png").symlink_to(secret)
    res = TestClient(create_app(root)).get(
        "/checks/cell/20260101T000000Z_trial_01/leak.png"
    )
    assert res.status_code == 400


def test_a_report_that_is_not_an_object_does_not_break_the_page(tmp_path: Path) -> None:
    """Valid JSON of the wrong shape must not take every other check with it."""
    pin_dir = _write_pin(tmp_path / ".anvil", "cell")
    _write_record(pin_dir, "20260101T000000Z_good")
    for cid, text in (("20260102T000000Z_list", "[]"), ("20260103T000000Z_null", "null")):
        bad = pin_dir / CHECKS_SUBDIR / cid
        bad.mkdir(parents=True)
        (bad / "report.json").write_text(text)
    client = TestClient(create_app(tmp_path / ".anvil"))
    assert client.get("/").status_code == 200
    body = client.get("/api").json()
    assert len(body) == 3
    assert sum(1 for c in body if c["status"] == "UNREADABLE") == 2


def test_an_unparseable_checked_at_falls_back_to_the_record_id(tmp_path: Path) -> None:
    """The page calls Date() on this; garbage would render as "Invalid Date"."""
    pin_dir = _write_pin(tmp_path / ".anvil", "cell")
    out = pin_dir / CHECKS_SUBDIR / "20260101T000000Z_trial_01"
    out.mkdir(parents=True)
    (out / "report.json").write_text(
        json.dumps({"status": "passed", "checked_at": "not-a-date", "scores": {}})
    )
    body = TestClient(create_app(tmp_path / ".anvil")).get("/api").json()
    assert body[0]["checked_at"].startswith("2026-01-01T00:00:00")


def _echo(word: str) -> str:
    """A record command that exists on any machine running the test suite."""
    return f'{sys.executable} -c "print(\'{word}\')"'


def test_guard_still_runs_without_the_save_flag(tmp_path: Path) -> None:
    """`guard` shares _run_check with `check`; adding --save must not break it."""
    from click.testing import CliRunner

    from anvil.cli import cli

    root = tmp_path / ".anvil"
    pin_dir = _write_pin(root, "cell")
    frame = tmp_path / "f.png"
    cv2.imwrite(str(frame), cv2.imread(str(pin_dir / "reference.png")))
    res = CliRunner().invoke(
        cli,
        ["guard", "--against", "cell", "--root", str(root),
         "--camera-driver", "file", "--camera", str(frame),
         "--record-cmd", _echo("recorded")],
    )
    assert "recorded" in res.output
    assert not (pin_dir / CHECKS_SUBDIR).exists()


def test_guard_save_writes_the_same_record_as_check(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from anvil.cli import cli

    root = tmp_path / ".anvil"
    pin_dir = _write_pin(root, "cell")
    frame = tmp_path / "f.png"
    cv2.imwrite(str(frame), cv2.imread(str(pin_dir / "reference.png")))
    CliRunner().invoke(
        cli,
        ["guard", "--against", "cell", "--root", str(root),
         "--camera-driver", "file", "--camera", str(frame),
         "--record-cmd", _echo("recorded"), "--save"],
    )
    records = list((pin_dir / CHECKS_SUBDIR).iterdir())
    assert len(records) == 1
    assert (records[0] / "report.json").is_file()
    assert TestClient(create_app(root)).get("/api").json()[0]["frame"] is not None
