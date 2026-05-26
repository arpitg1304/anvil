"""Layer 2 — structural path.

Today owns the **keypoint-based camera pose drift** check (DISK + LightGlue
+ rigid-2D fit). SAM3 per-object IoU lands here too in Week 2; that work
adds new functions alongside the keypoint orchestrator below, it doesn't
replace it.

Flow on `check` when the keypoint pipeline is available:

1. Load reference keypoints from ``pose/keypoints.npz``.
2. Detect keypoints on the live frame.
3. Match them against the reference with LightGlue.
4. RANSAC-fit a rigid 2D transform (rotation + translation + uniform
   scale) via ``cv2.estimateAffinePartial2D``; extract the rotation.
5. Compare against ``thresholds.max_camera_pose_drift_deg``.

When the pipeline is unavailable — bare install, no GPU, weights
unreachable, or the pin pre-dates this feature — Layer 2 keypoints simply
doesn't run and the caller falls back to Layer 1's ArUco pose signal.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from anvil.cameras.base import Frame
from anvil.manifest import ThresholdSpec
from anvil.models.keypoints import (
    KeypointDetector,
    KeypointMatcher,
    KeypointSet,
    MatchResult,
)
from anvil.schema import Finding, Flag, Severity

KEYPOINTS_FILENAME = "keypoints.npz"
KEYPOINTS_SUBDIR = "pose"

# Below this match count the rigid-2D fit is too noisy to trust. We surface
# the situation as a warning finding rather than reporting a bogus rotation.
_MIN_MATCHES_FOR_POSE = 10

# A rotation >= this is treated as critical rather than warning.
_KEYPOINT_POSE_CRITICAL_DEG = 5.0

_FORWARD_COMPAT = ConfigDict(extra="ignore")


class KeypointReference(BaseModel):
    """Persisted keypoint state captured at pin time."""

    model_config = _FORWARD_COMPAT

    detector_name: str
    image_hw: tuple[int, int]
    # Nx2 keypoints, NxD descriptors. Kept as lists for round-trip ergonomics
    # in Pydantic; persisted on disk as compact npz (see save/load below).
    keypoints: list[tuple[float, float]]
    descriptors: list[list[float]]

    @property
    def count(self) -> int:
        return len(self.keypoints)

    def to_set(self) -> KeypointSet:
        return KeypointSet(
            keypoints=self.keypoints,
            descriptors=self.descriptors,
            image_hw=self.image_hw,
            detector_name=self.detector_name,
        )


class Layer2KeypointsOutput(BaseModel):
    """Result of one Layer 2 keypoint pass."""

    model_config = _FORWARD_COMPAT

    # Camera rotation in degrees, recovered from the rigid-2D fit. Positive
    # follows OpenCV's convention (counter-clockwise in image space).
    rotation_deg: float
    # Translation in pixels recovered from the fit, (tx, ty).
    translation_px: tuple[float, float]
    # Diagnostic counters — small enough to dump straight into evidence.
    num_keypoints_ref: int
    num_keypoints_cur: int
    num_matches: int
    num_inliers: int
    findings: list[Finding] = Field(default_factory=list)
    flags: list[Flag] = Field(default_factory=list)

    @property
    def abs_rotation_deg(self) -> float:
        return abs(self.rotation_deg)


# --- persistence ---------------------------------------------------------


def keypoints_path(pin_dir: Path) -> Path:
    return pin_dir / KEYPOINTS_SUBDIR / KEYPOINTS_FILENAME


def compute_keypoint_reference(
    frame: Frame, detector: KeypointDetector, *, max_keypoints: int = 1024
) -> KeypointReference:
    kp_set = detector.detect(frame, max_keypoints=max_keypoints)
    return KeypointReference(
        detector_name=kp_set.detector_name,
        image_hw=kp_set.image_hw,
        keypoints=kp_set.keypoints,
        descriptors=kp_set.descriptors,
    )


def save_keypoint_reference(ref: KeypointReference, pin_dir: Path) -> Path:
    target = keypoints_path(pin_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        keypoints=np.asarray(ref.keypoints, dtype=np.float32),
        descriptors=np.asarray(ref.descriptors, dtype=np.float32),
        detector_name=np.array(ref.detector_name),
        image_hw=np.asarray(ref.image_hw, dtype=np.int32),
    )
    return target


def load_keypoint_reference(pin_dir: Path) -> KeypointReference | None:
    target = keypoints_path(pin_dir)
    if not target.exists():
        return None
    data = np.load(target, allow_pickle=False)
    keypoints = data["keypoints"].astype(np.float32)
    descriptors = data["descriptors"].astype(np.float32)
    hw_arr = data["image_hw"].astype(np.int32)
    return KeypointReference(
        detector_name=str(data["detector_name"]),
        image_hw=(int(hw_arr[0]), int(hw_arr[1])),
        keypoints=[(float(p[0]), float(p[1])) for p in keypoints],
        descriptors=[d.tolist() for d in descriptors],
    )


# --- pose recovery -------------------------------------------------------


def _rotation_from_affine(matrix: np.ndarray) -> float:
    """Extract rotation in degrees from a 2x3 ``estimateAffinePartial2D`` result."""
    # ``estimateAffinePartial2D`` produces [[s·cosθ, -s·sinθ, tx],
    #                                       [s·sinθ,  s·cosθ, ty]].
    # atan2(sinθ, cosθ) recovers θ without ambiguity.
    a, b = float(matrix[0, 0]), float(matrix[1, 0])
    return float(np.degrees(np.arctan2(b, a)))


def _match_points(
    ref: KeypointSet, cur: KeypointSet, matches: MatchResult
) -> tuple[np.ndarray, np.ndarray]:
    pts_ref = np.array(
        [ref.keypoints[i] for i in matches.indices_ref], dtype=np.float32
    )
    pts_cur = np.array(
        [cur.keypoints[i] for i in matches.indices_cur], dtype=np.float32
    )
    return pts_ref, pts_cur


def _next_id(findings: list[Finding]) -> str:
    return f"f_{len(findings) + 1:03d}"


def run_layer2_keypoints(
    *,
    current_frame: Frame,
    keypoint_ref: KeypointReference,
    detector: KeypointDetector,
    matcher: KeypointMatcher,
    thresholds: ThresholdSpec,
) -> Layer2KeypointsOutput:
    """Run keypoint-based pose drift end-to-end.

    Returns a sentinel-style output (rotation 0, empty findings) when the
    match count is below ``_MIN_MATCHES_FOR_POSE`` — caller can detect that
    via ``num_matches`` and choose how to surface it.
    """
    findings: list[Finding] = []
    flags: list[Flag] = []

    cur_set = detector.detect(current_frame, max_keypoints=len(keypoint_ref.keypoints))
    matches = matcher.match(keypoint_ref.to_set(), cur_set)

    pts_ref, pts_cur = _match_points(keypoint_ref.to_set(), cur_set, matches)

    rotation_deg = 0.0
    translation = (0.0, 0.0)
    inliers = 0

    if matches.count < _MIN_MATCHES_FOR_POSE:
        # Too few matches — could be lighting flip, partial occlusion, or
        # genuinely a different scene. Surface as a warning so the operator
        # knows the keypoint signal is degraded; the Layer 1 ArUco signal
        # remains authoritative in the report.
        findings.append(
            Finding(
                id=_next_id(findings),
                severity="warning",
                layer=2,
                component="pose",
                subject="camera",
                issue="keypoint_match_sparse",
                detail=(
                    f"Only {matches.count} keypoint matches against the pin "
                    f"(min {_MIN_MATCHES_FOR_POSE}). Scene may have changed "
                    "substantially or lighting is too dim."
                ),
                fix=(
                    "Check the lighting and that the rig hasn't been "
                    "significantly reconfigured."
                ),
                evidence={
                    "num_matches": matches.count,
                    "num_keypoints_ref": keypoint_ref.count,
                    "num_keypoints_cur": cur_set.count,
                },
            )
        )
        return Layer2KeypointsOutput(
            rotation_deg=0.0,
            translation_px=translation,
            num_keypoints_ref=keypoint_ref.count,
            num_keypoints_cur=cur_set.count,
            num_matches=matches.count,
            num_inliers=0,
            findings=findings,
            flags=flags,
        )

    affine, inlier_mask = cv2.estimateAffinePartial2D(
        pts_ref,
        pts_cur,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
    )
    if affine is None:
        # RANSAC didn't converge. Treat like sparse-matches case but with a
        # distinct issue tag.
        findings.append(
            Finding(
                id=_next_id(findings),
                severity="warning",
                layer=2,
                component="pose",
                subject="camera",
                issue="keypoint_pose_fit_failed",
                detail=(
                    "Rigid-2D RANSAC could not fit a consistent transform "
                    "across the matched keypoints."
                ),
                fix="Check for motion blur, severe lighting change, or partial occlusion.",
                evidence={"num_matches": matches.count},
            )
        )
        return Layer2KeypointsOutput(
            rotation_deg=0.0,
            translation_px=translation,
            num_keypoints_ref=keypoint_ref.count,
            num_keypoints_cur=cur_set.count,
            num_matches=matches.count,
            num_inliers=0,
            findings=findings,
            flags=flags,
        )

    affine = cast(np.ndarray, affine)
    rotation_deg = _rotation_from_affine(affine)
    translation = (float(affine[0, 2]), float(affine[1, 2]))
    inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0

    abs_rotation = abs(rotation_deg)
    if abs_rotation > thresholds.max_camera_pose_drift_deg:
        severity: Severity = (
            "critical" if abs_rotation >= _KEYPOINT_POSE_CRITICAL_DEG else "warning"
        )
        findings.append(
            Finding(
                id=_next_id(findings),
                severity=severity,
                layer=2,
                component="pose",
                subject="camera",
                issue="rotation_drift",
                detail=(
                    f"Camera rotated ~{rotation_deg:+.2f}° vs pin "
                    f"(keypoint-derived from {inliers}/{matches.count} inlier "
                    "matches)."
                ),
                fix="Re-level the camera mount, or repin if intentional.",
                evidence={
                    "rotation_delta_deg": rotation_deg,
                    "translation_px": list(translation),
                    "num_matches": matches.count,
                    "num_inliers": inliers,
                    "source": "lightglue-disk",
                },
            )
        )
        flags.append("camera_pose_drift")

    return Layer2KeypointsOutput(
        rotation_deg=rotation_deg,
        translation_px=translation,
        num_keypoints_ref=keypoint_ref.count,
        num_keypoints_cur=cur_set.count,
        num_matches=matches.count,
        num_inliers=inliers,
        findings=findings,
        flags=flags,
    )


__all__ = [
    "KEYPOINTS_FILENAME",
    "KEYPOINTS_SUBDIR",
    "KeypointReference",
    "Layer2KeypointsOutput",
    "compute_keypoint_reference",
    "keypoints_path",
    "load_keypoint_reference",
    "run_layer2_keypoints",
    "save_keypoint_reference",
]
