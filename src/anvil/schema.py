"""Anvil sidecar metadata schema (v0.1.0).

This module is the typed implementation of `docs/metadata_schema.md`.
That document is the source of truth; if the two ever disagree, the doc wins
and this file is the bug.

Forge (and any other consumer) implements against the JSON contract — not this
module — so backwards compatibility is enforced by the JSON shape, not by
import-level stability.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

ANVIL_SCHEMA_VERSION: Literal["0.1.0"] = "0.1.0"

# --- Enums (Literal per project convention) -------------------------------

Status = Literal["passed", "warning", "failed"]
OperatorAction = Literal["proceeded", "fixed_and_retried", "aborted"]
Severity = Literal["info", "warning", "critical"]
Component = Literal["object", "lighting", "pose", "robot", "scene"]

Flag = Literal[
    "scene_drift",
    "lighting_shift",
    "object_moved",
    "object_missing",
    "object_added",
    "camera_pose_drift",
    "robot_home_drift",
    "robot_base_moved",
    "vlm_uncertain",
]

# --- Constrained scalar aliases -------------------------------------------

SchemaVersion = Annotated[str, Field(pattern=r"^\d+\.\d+\.\d+$")]
ManifestHash = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
LayerIndex = Annotated[int, Field(ge=0, le=3)]
UnitInterval = Annotated[float, Field(ge=0.0, le=1.0)]
NonNegFloat = Annotated[float, Field(ge=0.0)]
NonNegInt = Annotated[int, Field(ge=0)]


# Default model config: ignore unknown fields on input (forward compat per §9
# of the schema doc). Producer code populates only the documented fields, so
# nothing is silently dropped from our own output.
_FORWARD_COMPAT = ConfigDict(extra="ignore")


class PinReference(BaseModel):
    """Reference to the pin a check was run against."""

    model_config = _FORWARD_COMPAT

    name: str
    manifest_hash: ManifestHash
    pinned_at: AwareDatetime


class Scores(BaseModel):
    """Aggregate numerics downstream tools can threshold on."""

    model_config = _FORWARD_COMPAT

    scene_drift: UnitInterval
    lighting_drift: UnitInterval
    max_object_drift_cm: NonNegFloat
    max_camera_pose_drift_deg: NonNegFloat
    max_robot_joint_drift_deg: NonNegFloat


class Finding(BaseModel):
    """One localized issue raised by a cascade layer."""

    model_config = _FORWARD_COMPAT

    id: str
    severity: Severity
    layer: LayerIndex
    component: Component
    subject: str
    issue: str
    detail: str
    fix: str
    # Layer-specific raw numbers — schema is open by layer, so we keep it as
    # an untyped mapping. Producers should still emit consistent keys per
    # (component, issue) pair so dashboards can rely on them.
    evidence: dict[str, Any] | None = None


class AgentTrace(BaseModel):
    """Layer 3 trace; present only when the VLM agent ran."""

    model_config = _FORWARD_COMPAT

    vlm_invoked: bool
    vlm_model: str | None = None
    tool_calls: NonNegInt = 0
    questions_asked_operator: NonNegInt = 0


class Provenance(BaseModel):
    """Reproducibility — what code/models produced this report."""

    model_config = _FORWARD_COMPAT

    anvil_version: str
    models: dict[str, str] = Field(default_factory=dict)
    hardware: dict[str, str] = Field(default_factory=dict)


class EpisodeReport(BaseModel):
    """Top-level sidecar written next to each LeRobotDataset episode."""

    model_config = _FORWARD_COMPAT

    anvil_schema_version: SchemaVersion
    episode_id: str
    checked_at: AwareDatetime
    pin: PinReference
    status: Status
    operator_action: OperatorAction
    scores: Scores
    flags: list[Flag] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    agent_trace: AgentTrace | None = None
    provenance: Provenance


__all__ = [
    "ANVIL_SCHEMA_VERSION",
    "AgentTrace",
    "Component",
    "EpisodeReport",
    "Finding",
    "Flag",
    "LayerIndex",
    "ManifestHash",
    "NonNegFloat",
    "NonNegInt",
    "OperatorAction",
    "PinReference",
    "Provenance",
    "SchemaVersion",
    "Scores",
    "Severity",
    "Status",
    "UnitInterval",
]
