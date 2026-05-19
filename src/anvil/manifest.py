"""Pin manifest model + filesystem I/O + content hashing.

A pin lives in a directory like ``.anvil/pick_red_cube/`` and contains:

    manifest.yaml            # this file's model
    reference.png            # canonical RGB frame
    reference_features.npz   # DINOv3 features (Week 1, not yet written)
    objects/*.json           # SAM3 masks (Week 2)
    pose/keypoints.json      # SuperPoint descriptors (Week 1)
    ...

``manifest_hash`` is sha256 of every file in the pin directory, in
deterministic (sorted) path order, with paths mixed into the digest so two
files with swapped content produce a different hash.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from anvil.errors import ManifestVersionMismatch, PinNotFound
from anvil.schema import NonNegFloat, SchemaVersion, UnitInterval

MANIFEST_FILENAME = "manifest.yaml"
REFERENCE_IMAGE_FILENAME = "reference.png"
DEFAULT_PIN_ROOT = Path(".anvil")

CameraDriver = Literal["webcam", "file"]
RobotDriver = Literal["lerobot", "none"]


_FORWARD_COMPAT = ConfigDict(extra="ignore")


class CameraSpec(BaseModel):
    model_config = _FORWARD_COMPAT

    driver: CameraDriver
    device: str
    # (width, height); optional because some drivers honor whatever the device gives.
    resolution: (
        tuple[Annotated[int, Field(ge=1)], Annotated[int, Field(ge=1)]] | None
    ) = None


class RobotSpec(BaseModel):
    """Layer 0 configuration. Defaults to disabled / `none` driver."""

    model_config = _FORWARD_COMPAT

    enabled: bool = False
    driver: RobotDriver = "none"
    home_pose_joints: list[float] = Field(default_factory=list)
    tolerance_deg: Annotated[float, Field(gt=0.0)] = 0.5
    include_visual_base_check: bool = True


class ThresholdSpec(BaseModel):
    """Per-check pass/fail bounds. Defaults are intentionally loose for v0.1;
    auto-tuning lands in Week 3.
    """

    model_config = _FORWARD_COMPAT

    scene_drift: UnitInterval = 0.3
    lighting_drift: UnitInterval = 0.3
    max_object_drift_cm: NonNegFloat = 5.0
    max_camera_pose_drift_deg: NonNegFloat = 1.0


class Manifest(BaseModel):
    """Top-level pin manifest, serialized as ``manifest.yaml``."""

    model_config = _FORWARD_COMPAT

    anvil_schema_version: SchemaVersion
    name: str
    task: str = ""
    pinned_at: AwareDatetime
    camera: CameraSpec
    robot: RobotSpec = Field(default_factory=RobotSpec)
    thresholds: ThresholdSpec = Field(default_factory=ThresholdSpec)
    # Filled in Week 2 when SAM3 lands; empty for Layer-1-only pins.
    objects: list[str] = Field(default_factory=list)
    aruco_present: bool = False


# --- filesystem helpers ---------------------------------------------------


def pin_directory(name: str, *, root: Path | None = None) -> Path:
    """Resolve the directory for a named pin.

    ``root`` defaults to ``./.anvil``. The pin name is appended as-is; it
    should be safe for use as a single path segment.
    """
    if "/" in name or "\\" in name or name in {"", ".", ".."}:
        raise ValueError(f"Invalid pin name: {name!r}")
    base = root if root is not None else Path.cwd() / DEFAULT_PIN_ROOT
    return base / name


def save_manifest(manifest: Manifest, pin_dir: Path) -> Path:
    """Write ``manifest.yaml`` into ``pin_dir``, creating the dir if needed."""
    pin_dir.mkdir(parents=True, exist_ok=True)
    payload = manifest.model_dump(mode="json")
    target = pin_dir / MANIFEST_FILENAME
    with target.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, default_flow_style=False)
    return target


def load_manifest(pin_dir: Path) -> Manifest:
    """Read ``manifest.yaml`` from ``pin_dir`` and validate version."""
    if not pin_dir.exists():
        raise PinNotFound(f"No pin directory at {pin_dir}")
    path = pin_dir / MANIFEST_FILENAME
    if not path.exists():
        raise PinNotFound(f"No {MANIFEST_FILENAME} at {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    manifest = Manifest.model_validate(data)
    if not manifest.anvil_schema_version.startswith("0.1."):
        raise ManifestVersionMismatch(
            f"Manifest at {path} has version "
            f"{manifest.anvil_schema_version}; this Anvil expects 0.1.x"
        )
    return manifest


def compute_manifest_hash(pin_dir: Path) -> str:
    """sha256 of every file in ``pin_dir``, in sorted path order.

    The relative path is mixed into the digest before the content so renaming
    a file (without changing content) still changes the hash.
    """
    if not pin_dir.exists():
        raise PinNotFound(f"No pin directory at {pin_dir}")
    h = hashlib.sha256()
    for path in sorted(pin_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(pin_dir).as_posix()
        h.update(rel.encode("utf-8"))
        h.update(b"\x00")
        h.update(path.read_bytes())
        h.update(b"\x00")
    return "sha256:" + h.hexdigest()


__all__ = [
    "DEFAULT_PIN_ROOT",
    "MANIFEST_FILENAME",
    "REFERENCE_IMAGE_FILENAME",
    "CameraDriver",
    "CameraSpec",
    "Manifest",
    "RobotDriver",
    "RobotSpec",
    "ThresholdSpec",
    "compute_manifest_hash",
    "load_manifest",
    "pin_directory",
    "save_manifest",
]
