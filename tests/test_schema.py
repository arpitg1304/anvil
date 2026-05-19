"""Schema round-trip + validation tests.

These pin the v0.1 contract documented in docs/metadata_schema.md. Breaking
any of these without bumping `ANVIL_SCHEMA_VERSION` is a bug.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest
from pydantic import ValidationError

from anvil.schema import (
    ANVIL_SCHEMA_VERSION,
    AgentTrace,
    EpisodeReport,
    Finding,
    PinReference,
    Provenance,
    Scores,
)

VALID_HASH = "sha256:" + "a" * 64


def _minimal_report() -> EpisodeReport:
    return EpisodeReport(
        anvil_schema_version=ANVIL_SCHEMA_VERSION,
        episode_id="episode_000042",
        checked_at=datetime(2026, 5, 19, 14, 32, 18, tzinfo=timezone.utc),
        pin=PinReference(
            name="pick_red_cube",
            manifest_hash=VALID_HASH,
            pinned_at=datetime(2026, 5, 12, 9, 14, 3, tzinfo=timezone.utc),
        ),
        status="passed",
        operator_action="proceeded",
        scores=Scores(
            scene_drift=0.05,
            lighting_drift=0.02,
            max_object_drift_cm=0.5,
            max_camera_pose_drift_deg=0.1,
            max_robot_joint_drift_deg=0.0,
        ),
        provenance=Provenance(anvil_version="0.1.0.dev0"),
    )


def _full_report() -> EpisodeReport:
    """Mirror the §1 example in docs/metadata_schema.md."""
    return EpisodeReport(
        anvil_schema_version="0.1.0",
        episode_id="episode_000042",
        checked_at=datetime(2026, 5, 19, 14, 32, 18, 421000, tzinfo=timezone.utc),
        pin=PinReference(
            name="pick_red_cube",
            manifest_hash=(
                "sha256:7f3a1b9c4e8d2a6f9b3c5e7d1a8f4c2e"
                "6b9d3a5f7c1e8b4a6d2f9c3e5b7a1d8f"
            ),
            pinned_at=datetime(2026, 5, 12, 9, 14, 3, tzinfo=timezone.utc),
        ),
        status="warning",
        operator_action="proceeded",
        scores=Scores(
            scene_drift=0.27,
            lighting_drift=0.41,
            max_object_drift_cm=5.8,
            max_camera_pose_drift_deg=0.4,
            max_robot_joint_drift_deg=1.2,
        ),
        flags=["lighting_shift", "robot_home_drift"],
        findings=[
            Finding(
                id="f_001",
                severity="warning",
                layer=2,
                component="object",
                subject="red_cube",
                issue="position_drift",
                detail="Moved ~5.8cm toward camera and 2.1cm right of pinned position",
                fix="Slide back to the X mark on the mat",
                evidence={
                    "iou": 0.62,
                    "region_cosine": 0.89,
                    "centroid_delta_px": [42, -89],
                },
            ),
            Finding(
                id="f_002",
                severity="warning",
                layer=1,
                component="lighting",
                subject="global",
                issue="warmer_color_temperature",
                detail="Mean color temp shifted from ~5200K to ~4100K",
                fix="Check the overhead LED, may have switched modes",
                evidence={
                    "histogram_chi2": 0.34,
                    "mean_luminance_delta": -18.2,
                    "color_temp_delta_k": -1100,
                },
            ),
            Finding(
                id="f_003",
                severity="critical",
                layer=0,
                component="robot",
                subject="joint_3",
                issue="joint_drift",
                detail="Joint 3 is 1.2° off home (expected 1.5707, got 1.5916 rad)",
                fix="Re-run robot calibration, or check if the base clamp slipped",
                evidence={
                    "expected_rad": 1.5707,
                    "measured_rad": 1.5916,
                    "delta_deg": 1.197,
                    "tolerance_deg": 0.5,
                },
            ),
        ],
        agent_trace=AgentTrace(
            vlm_invoked=True,
            vlm_model="Qwen/Qwen3-VL-4B-Instruct",
            tool_calls=4,
            questions_asked_operator=0,
        ),
        provenance=Provenance(
            anvil_version="0.1.0",
            models={
                "dinov3": "facebook/dinov3-vits16-pretrain-lvd1689m",
                "sam3": "facebook/sam3-base@3.1",
                "vlm": "Qwen/Qwen3-VL-4B-Instruct",
                "keypoints": "lightglue/superpoint_v1",
            },
            hardware={"gpu": "NVIDIA GeForce RTX 4090", "cuda": "12.4"},
        ),
    )


# --- round trip ----------------------------------------------------------


def test_schema_version_constant():
    assert ANVIL_SCHEMA_VERSION == "0.1.0"


def test_minimal_round_trip():
    report = _minimal_report()
    payload = report.model_dump_json()
    reloaded = EpisodeReport.model_validate_json(payload)
    assert reloaded == report


def test_full_round_trip_matches_docs_example():
    report = _full_report()
    payload = report.model_dump_json()
    reloaded = EpisodeReport.model_validate_json(payload)
    assert reloaded == report
    # Evidence dicts must survive intact.
    assert reloaded.findings[0].evidence == {
        "iou": 0.62,
        "region_cosine": 0.89,
        "centroid_delta_px": [42, -89],
    }


def test_dict_round_trip_preserves_enums():
    report = _full_report()
    dumped = report.model_dump(mode="json")
    reloaded = EpisodeReport.model_validate(dumped)
    assert reloaded == report


# --- enum validation -----------------------------------------------------


def _bad(payload: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        EpisodeReport.model_validate(payload)


def test_bad_status_rejected():
    payload = _minimal_report().model_dump(mode="json")
    payload["status"] = "ok"
    _bad(payload)


def test_bad_operator_action_rejected():
    payload = _minimal_report().model_dump(mode="json")
    payload["operator_action"] = "skipped"
    _bad(payload)


def test_bad_severity_rejected():
    with pytest.raises(ValidationError):
        Finding(
            id="f_x",
            severity="catastrophic",  # type: ignore[arg-type]
            layer=1,
            component="object",
            subject="x",
            issue="y",
            detail="d",
            fix="f",
        )


def test_bad_component_rejected():
    with pytest.raises(ValidationError):
        Finding(
            id="f_x",
            severity="warning",
            layer=1,
            component="camera",  # type: ignore[arg-type] # not in v0.1 enum (use 'pose')
            subject="x",
            issue="y",
            detail="d",
            fix="f",
        )


def test_invalid_flag_rejected():
    payload = _minimal_report().model_dump(mode="json")
    payload["flags"] = ["not_a_real_flag"]
    _bad(payload)


# --- numeric constraints -------------------------------------------------


def test_score_above_one_rejected():
    with pytest.raises(ValidationError):
        Scores(
            scene_drift=1.5,
            lighting_drift=0.0,
            max_object_drift_cm=0.0,
            max_camera_pose_drift_deg=0.0,
            max_robot_joint_drift_deg=0.0,
        )


def test_score_below_zero_rejected():
    with pytest.raises(ValidationError):
        Scores(
            scene_drift=-0.1,
            lighting_drift=0.0,
            max_object_drift_cm=0.0,
            max_camera_pose_drift_deg=0.0,
            max_robot_joint_drift_deg=0.0,
        )


def test_negative_physical_unit_rejected():
    with pytest.raises(ValidationError):
        Scores(
            scene_drift=0.0,
            lighting_drift=0.0,
            max_object_drift_cm=-1.0,
            max_camera_pose_drift_deg=0.0,
            max_robot_joint_drift_deg=0.0,
        )


def test_layer_out_of_range_rejected():
    with pytest.raises(ValidationError):
        Finding(
            id="f",
            severity="info",
            layer=5,
            component="object",
            subject="x",
            issue="y",
            detail="d",
            fix="f",
        )


def test_negative_tool_calls_rejected():
    with pytest.raises(ValidationError):
        AgentTrace(vlm_invoked=True, tool_calls=-1)


# --- string-pattern constraints ------------------------------------------


def test_bad_manifest_hash_rejected():
    with pytest.raises(ValidationError):
        PinReference(
            name="x",
            manifest_hash="not-a-hash",
            pinned_at=datetime.now(timezone.utc),
        )


def test_short_manifest_hash_rejected():
    with pytest.raises(ValidationError):
        PinReference(
            name="x",
            manifest_hash="sha256:abc",
            pinned_at=datetime.now(timezone.utc),
        )


def test_bad_schema_version_rejected():
    payload = _minimal_report().model_dump(mode="json")
    payload["anvil_schema_version"] = "not-semver"
    _bad(payload)


# --- datetime constraints -----------------------------------------------


def test_naive_datetime_rejected():
    with pytest.raises(ValidationError):
        PinReference(
            name="x",
            manifest_hash=VALID_HASH,
            pinned_at=datetime.now(),
        )


# --- forward compatibility ----------------------------------------------


def test_unknown_top_level_field_ignored():
    """Per schema doc §9: consumers MUST ignore unknown fields, not error."""
    payload = _minimal_report().model_dump(mode="json")
    payload["future_v0_2_field"] = "ignored"
    payload["pin"]["future_subfield"] = 42
    reloaded = EpisodeReport.model_validate(payload)
    # extras are dropped, so equality with the original holds
    assert reloaded == _minimal_report()


def test_agent_trace_optional():
    report = _minimal_report()
    assert report.agent_trace is None
    payload = report.model_dump_json()
    assert EpisodeReport.model_validate_json(payload).agent_trace is None
