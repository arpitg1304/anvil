"""Layer 2 — structural path.

Two independent checks both live here:

1. **Keypoint-based camera pose drift** — DISK + LightGlue + rigid-2D fit.
   Estimates rotation and translation against the pin without needing
   ArUco fiducials. Runs when ``pose/keypoints.npz`` exists.
2. **Named-object drift** — YOLO-World text-promptable detection on a
   list of objects the operator named at pin time. Per-object bbox IoU
   + DINOv3 region cosine. Runs when ``objects/*.json`` exist.

Either can fall back silently when its dependency stack isn't installed
([full] extra missing, weights unreachable, etc.) — the caller just
doesn't get that part of the signal.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from anvil.cameras.base import Frame
from anvil.manifest import ThresholdSpec
from anvil.models.embedder import GlobalEmbedder
from anvil.models.keypoints import (
    KeypointDetector,
    KeypointMatcher,
    KeypointSet,
    MatchResult,
)
from anvil.models.objects import DetectedObject, ObjectDetector
from anvil.schema import Finding, Flag, Severity

KEYPOINTS_FILENAME = "keypoints.npz"
KEYPOINTS_SUBDIR = "pose"
OBJECTS_SUBDIR = "objects"

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
    translation_magnitude = float(np.hypot(translation[0], translation[1]))

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
        if "camera_pose_drift" not in flags:
            flags.append("camera_pose_drift")

    if translation_magnitude > thresholds.max_camera_translation_px:
        # Translation is reported separately from rotation: a camera that
        # slips down without twisting (loose ceiling mount, bumped tripod
        # leg) trips this but not the rotation finding.
        findings.append(
            Finding(
                id=_next_id(findings),
                severity="warning",
                layer=2,
                component="pose",
                subject="camera",
                issue="translation_drift",
                detail=(
                    f"Camera shifted ~{translation_magnitude:.1f}px vs pin "
                    f"(dx={translation[0]:+.1f}, dy={translation[1]:+.1f}; "
                    f"keypoint-derived, {inliers}/{matches.count} inliers)."
                ),
                fix="Check the camera mount — clamp, tripod, or arm may have slipped.",
                evidence={
                    "translation_px": list(translation),
                    "translation_magnitude_px": translation_magnitude,
                    "rotation_delta_deg": rotation_deg,
                    "num_matches": matches.count,
                    "num_inliers": inliers,
                    "source": "lightglue-disk",
                },
            )
        )
        if "camera_pose_drift" not in flags:
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


class ObjectReference(BaseModel):
    """Persisted state for one named object at pin time."""

    model_config = _FORWARD_COMPAT

    # The operator-supplied name (used as a filename and the text prompt).
    name: str
    prompt: str
    bbox: tuple[float, float, float, float]
    confidence: float
    image_hw: tuple[int, int]
    detector_name: str
    # Optional DINOv3 (or whichever embedder ran) features of the bbox crop.
    # Empty list when no embedder was available at pin time — appearance
    # comparison is skipped at check time in that case.
    region_embedding: list[float] = Field(default_factory=list)
    embedder_name: str | None = None

    @property
    def centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class Layer2ObjectsOutput(BaseModel):
    """Result of one Layer 2 named-object pass."""

    model_config = _FORWARD_COMPAT

    findings: list[Finding] = Field(default_factory=list)
    flags: list[Flag] = Field(default_factory=list)
    # max_*_drift fields surface raw numbers for callers / Forge filtering.
    max_object_drift_px: float = 0.0
    min_object_iou: float = 1.0
    min_object_appearance_cosine: float = 1.0
    num_objects_pinned: int = 0
    num_objects_detected: int = 0


def objects_dir(pin_dir: Path) -> Path:
    return pin_dir / OBJECTS_SUBDIR


def object_reference_path(pin_dir: Path, name: str) -> Path:
    return objects_dir(pin_dir) / f"{name}.json"


def _crop_bbox(frame: Frame, bbox: tuple[float, float, float, float]) -> Frame | None:
    """Return the BGR crop for ``bbox``, or ``None`` if degenerate / out of bounds."""
    h, w = frame.shape[:2]
    x1 = max(0, round(bbox[0]))
    y1 = max(0, round(bbox[1]))
    x2 = min(w, round(bbox[2]))
    y2 = min(h, round(bbox[3]))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()


def _build_object_reference(
    frame: Frame,
    name: str,
    prompt: str,
    match: DetectedObject,
    detector_name: str,
    embedder: GlobalEmbedder | None,
) -> ObjectReference:
    region_embedding: list[float] = []
    embedder_name: str | None = None
    if embedder is not None:
        crop = _crop_bbox(frame, match.bbox)
        if crop is not None:
            region_embedding = embedder.embed(crop).astype(np.float32).tolist()
            embedder_name = embedder.name
    return ObjectReference(
        name=name,
        prompt=prompt,
        bbox=match.bbox,
        confidence=match.confidence,
        image_hw=frame.shape[:2],
        detector_name=detector_name,
        region_embedding=region_embedding,
        embedder_name=embedder_name,
    )


def compute_object_reference(
    frame: Frame,
    name: str,
    prompt: str,
    detector: ObjectDetector,
    embedder: GlobalEmbedder | None = None,
) -> ObjectReference | None:
    """Detect ``prompt`` in ``frame`` and return a pinnable reference.

    Returns ``None`` if the detector saw no match (caller should warn the
    operator that the named object isn't actually visible in the frame).
    """
    detections = detector.detect(frame, [prompt])
    match: DetectedObject | None = next(
        (d for d in detections if d.name == prompt), None
    )
    if match is None:
        return None
    return _build_object_reference(
        frame, name, prompt, match, detector.name, embedder
    )


def compute_object_references(
    frame: Frame,
    objects: list[tuple[str, str]],
    detector: ObjectDetector,
    embedder: GlobalEmbedder | None = None,
) -> list[tuple[str, ObjectReference | None]]:
    """Detect every named object in a single detector pass.

    ``objects`` is a list of ``(name, prompt)`` pairs. Returns one entry per
    input pair, in order, with ``None`` for objects the detector didn't find.

    Two named objects sharing the same prompt will bind to the *same* detected
    bbox (the best-confidence instance of that prompt). For multi-instance
    tracking — two grippers, three test tubes — use distinct prompts.

    Calling the detector once with all prompts (instead of once per object)
    is both a perf win (single CLIP text encode) and dodges an Ultralytics
    8.4 bug where repeated set_classes on a CUDA model leaves text tokens
    on CPU.
    """
    if not objects:
        return []
    unique_prompts = list(dict.fromkeys(prompt for _name, prompt in objects))
    detections = detector.detect(frame, unique_prompts)
    # Pin-time: pick the single highest-confidence detection per prompt.
    # Detectors return all candidates above their threshold; multiple
    # candidates per prompt are normal on cluttered scenes.
    best_by_prompt: dict[str, DetectedObject] = {}
    for d in detections:
        current = best_by_prompt.get(d.name)
        if current is None or d.confidence > current.confidence:
            best_by_prompt[d.name] = d

    out: list[tuple[str, ObjectReference | None]] = []
    for name, prompt in objects:
        match = best_by_prompt.get(prompt)
        if match is None:
            out.append((name, None))
            continue
        out.append(
            (
                name,
                _build_object_reference(
                    frame, name, prompt, match, detector.name, embedder
                ),
            )
        )
    return out


def save_object_reference(ref: ObjectReference, pin_dir: Path) -> Path:
    target = object_reference_path(pin_dir, ref.name)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(ref.model_dump_json(indent=2), encoding="utf-8")
    return target


def load_object_references(pin_dir: Path) -> list[ObjectReference]:
    """Load every ``objects/*.json`` in ``pin_dir``, sorted by name."""
    dir_ = objects_dir(pin_dir)
    if not dir_.exists():
        return []
    refs: list[ObjectReference] = []
    for path in sorted(dir_.glob("*.json")):
        refs.append(ObjectReference.model_validate_json(path.read_text("utf-8")))
    return refs


def _bbox_iou(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def _match_to_pinned(
    ref: ObjectReference, candidates: list[DetectedObject]
) -> DetectedObject | None:
    """Pick the candidate that best matches the pinned bbox.

    Open-vocab detectors return multiple candidates per prompt on cluttered
    scenes — and the same physical object often shows up at several
    *scales* (a full bbox around the rack, a tight box around its slats,
    etc.). Scoring on centroid distance alone can pick a tiny/skinny
    sub-detection right next to the pin's center over a properly-sized
    candidate slightly offset.

    Two-tier score: **highest IoU first**, centroid distance as
    tiebreaker. IoU directly measures "same object, same place, similar
    scale" so it's the primary signal. When all candidates have IoU=0
    (the object truly moved out of its pinned region), the centroid
    tiebreaker falls back to "where did it most likely go".

    Returns ``None`` if no candidates were supplied (caller treats as
    ``object_missing``).
    """
    if not candidates:
        return None
    rx, ry = ref.centroid

    def score(d: DetectedObject) -> tuple[float, float]:
        iou = _bbox_iou(ref.bbox, d.bbox)
        dist_sq = (d.centroid[0] - rx) ** 2 + (d.centroid[1] - ry) ** 2
        # min() picks the smallest tuple → max IoU first (via negation),
        # then min distance.
        return (-iou, dist_sq)

    return min(candidates, key=score)


def run_layer2_objects(
    *,
    current_frame: Frame,
    object_refs: list[ObjectReference],
    detector: ObjectDetector,
    embedder: GlobalEmbedder | None,
    thresholds: ThresholdSpec,
) -> Layer2ObjectsOutput:
    """Re-detect each pinned object and compare against the reference.

    Emits per-object findings and aggregate worst-case stats. Three kinds
    of finding land here:

    * ``object_missing`` — pinned object wasn't found this run.
    * ``position_drift`` — detected, but its IoU vs the pin is below
      ``thresholds.min_object_iou`` (the object moved).
    * ``appearance_drift`` — detected, but DINOv3 region cosine fell
      below ``thresholds.min_object_appearance_cosine`` (likely swapped
      for a different instance, or relit drastically).
    """
    findings: list[Finding] = []
    flags: list[Flag] = []
    if not object_refs:
        return Layer2ObjectsOutput()

    prompts = [ref.prompt for ref in object_refs]
    detections = detector.detect(current_frame, prompts)
    # Group all candidates per prompt — open-vocab detectors typically
    # return several per class on cluttered scenes, and the "best-conf"
    # candidate isn't stable across runs even on identical frames. We
    # match each pinned object to whichever candidate is *closest* to
    # where it was pinned.
    candidates_by_prompt: dict[str, list[DetectedObject]] = {}
    for d in detections:
        candidates_by_prompt.setdefault(d.name, []).append(d)

    worst_drift_px = 0.0
    worst_iou = 1.0
    worst_cosine = 1.0

    for ref in object_refs:
        candidates = candidates_by_prompt.get(ref.prompt, [])
        det = _match_to_pinned(ref, candidates)
        if det is None:
            findings.append(
                Finding(
                    id=_next_id(findings),
                    severity="warning",
                    layer=2,
                    component="object",
                    subject=ref.name,
                    issue="object_missing",
                    detail=(
                        f"Pinned object {ref.name!r} (prompt {ref.prompt!r}) not "
                        "detected in current frame."
                    ),
                    fix=(
                        f"Verify {ref.name} is still in the rig and visible to "
                        "the camera; if removed intentionally, repin."
                    ),
                    evidence={"prompt": ref.prompt, "detector": ref.detector_name},
                )
            )
            if "object_missing" not in flags:
                flags.append("object_missing")
            worst_iou = 0.0
            worst_cosine = 0.0
            continue

        iou = _bbox_iou(ref.bbox, det.bbox)
        rx, ry = ref.centroid
        cx, cy = det.centroid
        centroid_delta_px = float(np.hypot(cx - rx, cy - ry))
        worst_drift_px = max(worst_drift_px, centroid_delta_px)
        worst_iou = min(worst_iou, iou)

        cosine: float | None = None
        if ref.region_embedding and embedder is not None:
            crop = _crop_bbox(current_frame, det.bbox)
            if crop is not None:
                cur_vec = embedder.embed(crop)
                ref_vec = np.asarray(ref.region_embedding, dtype=np.float32)
                cosine = float(np.dot(cur_vec, ref_vec))
                worst_cosine = min(worst_cosine, cosine)

        if iou < thresholds.min_object_iou:
            findings.append(
                Finding(
                    id=_next_id(findings),
                    severity="warning",
                    layer=2,
                    component="object",
                    subject=ref.name,
                    issue="position_drift",
                    detail=(
                        f"{ref.name!r} moved: IoU vs pin = {iou:.2f}, "
                        f"centroid shifted ~{centroid_delta_px:.0f}px."
                    ),
                    fix=f"Move {ref.name} back to its pinned position, or repin.",
                    evidence={
                        "iou": iou,
                        "centroid_delta_px": [cx - rx, cy - ry],
                        "ref_bbox": list(ref.bbox),
                        "current_bbox": list(det.bbox),
                        "confidence": det.confidence,
                    },
                )
            )
            if "object_moved" not in flags:
                flags.append("object_moved")

        if cosine is not None and cosine < thresholds.min_object_appearance_cosine:
            findings.append(
                Finding(
                    id=_next_id(findings),
                    severity="warning",
                    layer=2,
                    component="object",
                    subject=ref.name,
                    issue="appearance_drift",
                    detail=(
                        f"{ref.name!r} looks different: region cosine = "
                        f"{cosine:.2f} (threshold "
                        f"{thresholds.min_object_appearance_cosine})."
                    ),
                    fix=(
                        f"Check if {ref.name} was swapped, repainted, or is being "
                        "viewed under very different lighting."
                    ),
                    evidence={
                        "region_cosine": cosine,
                        "iou": iou,
                        "embedder": ref.embedder_name,
                    },
                )
            )
            if "object_moved" not in flags:
                flags.append("object_moved")

    return Layer2ObjectsOutput(
        findings=findings,
        flags=flags,
        max_object_drift_px=worst_drift_px,
        min_object_iou=worst_iou,
        min_object_appearance_cosine=worst_cosine,
        num_objects_pinned=len(object_refs),
        num_objects_detected=len(detections),
    )


__all__ = [
    "KEYPOINTS_FILENAME",
    "KEYPOINTS_SUBDIR",
    "OBJECTS_SUBDIR",
    "KeypointReference",
    "Layer2KeypointsOutput",
    "Layer2ObjectsOutput",
    "ObjectReference",
    "compute_keypoint_reference",
    "compute_object_reference",
    "compute_object_references",
    "keypoints_path",
    "load_keypoint_reference",
    "load_object_references",
    "object_reference_path",
    "objects_dir",
    "run_layer2_keypoints",
    "run_layer2_objects",
    "save_keypoint_reference",
    "save_object_reference",
]
