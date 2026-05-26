"""Layer 1 — fast vision path.

Runs on CPU or a small GPU. Computes:

* Per-channel BGR histograms + symmetric chi-square distance.
* Mean luminance and McCamy-approximated correlated colour temperature (CCT).
* ArUco fiducial detection and rotation drift (if markers were pinned).
* Global image embedding (DINOv3 by default) → ``scene_drift = 1 - cosine``.
  Falls back to the histogram chi-square when no embedder is available
  (bare install, no GPU, weights unreachable, etc.).

The module is self-contained: it owns the on-disk reference shapes
(``lighting.json``, ``pose/aruco.json``, ``reference_features.npz``) and
exposes save/load helpers for each. ``run_layer1`` is the orchestrator the
CLI calls.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, cast

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from anvil.cameras.base import Frame
from anvil.manifest import ThresholdSpec
from anvil.models import GlobalEmbedder
from anvil.schema import (
    Finding,
    Flag,
    NonNegFloat,
    Severity,
)

HISTOGRAM_BINS = 64
LIGHTING_FILENAME = "lighting.json"
ARUCO_FILENAME = "aruco.json"
ARUCO_SUBDIR = "pose"
EMBEDDING_FILENAME = "reference_features.npz"
DEFAULT_ARUCO_DICTIONARY = "DICT_4X4_50"

# Normalization knobs for the composite lighting_drift score. They map a
# raw delta onto [0, 1] before mixing. 60 luminance units (~24% of full
# range) and 2000 K are the rough "this is obviously different" anchors.
_LUMINANCE_FULLSCALE = 60.0
_CCT_FULLSCALE_K = 2000.0

_HIST_WEIGHT = 0.5
_LUMINANCE_WEIGHT = 0.25
_CCT_WEIGHT = 0.25

_LIGHTING_CRITICAL_THRESHOLD = 0.6
_CAMERA_POSE_CRITICAL_DEG = 5.0


_FORWARD_COMPAT = ConfigDict(extra="ignore")

HistogramChannel = Annotated[
    list[float], Field(min_length=HISTOGRAM_BINS, max_length=HISTOGRAM_BINS)
]


class LightingReference(BaseModel):
    """Per-channel BGR histogram + summary stats for the pinned frame."""

    model_config = _FORWARD_COMPAT

    bins: int = HISTOGRAM_BINS
    histogram_b: HistogramChannel
    histogram_g: HistogramChannel
    histogram_r: HistogramChannel
    mean_luminance: NonNegFloat
    cct_kelvin: NonNegFloat


class ArucoMarker(BaseModel):
    model_config = _FORWARD_COMPAT

    id: int
    center_px: tuple[float, float]
    orientation_deg: float


class ArucoReference(BaseModel):
    model_config = _FORWARD_COMPAT

    dictionary: str = DEFAULT_ARUCO_DICTIONARY
    markers: list[ArucoMarker] = Field(default_factory=list)


class EmbeddingReference(BaseModel):
    """L2-normalized global embedding captured at pin time.

    Persisted as ``reference_features.npz`` (compressed). The ``embedder_name``
    field lets ``check`` detect that the user has swapped backbones since the
    pin was made.
    """

    model_config = _FORWARD_COMPAT

    embedder_name: str
    dim: int
    vector: list[float]


class Layer1Output(BaseModel):
    """Result of one Layer 1 pass. Score fields feed `EpisodeReport.scores`."""

    model_config = _FORWARD_COMPAT

    scene_drift: float
    scene_drift_source: str  # "embedder:<name>" or "histogram"
    lighting_drift: float
    max_camera_pose_drift_deg: float
    findings: list[Finding] = Field(default_factory=list)
    flags: list[Flag] = Field(default_factory=list)


# --- lighting primitives -------------------------------------------------


def _compute_histogram(channel: np.ndarray) -> list[float]:
    hist = cv2.calcHist([channel], [0], None, [HISTOGRAM_BINS], [0, 256])
    return hist.flatten().astype(np.float32).tolist()


def _estimate_cct_kelvin(frame: Frame) -> float:
    """McCamy's approximation of CCT from BGR uint8.

    Returns 0.0 if the frame has effectively no chrominance (degenerate
    denominator); callers should treat that as "unknown" rather than "0 K".
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    mean_r, mean_g, mean_b = cast(
        tuple[float, float, float], tuple(linear.reshape(-1, 3).mean(axis=0).tolist())
    )
    x_val = 0.4124564 * mean_r + 0.3575761 * mean_g + 0.1804375 * mean_b
    y_val = 0.2126729 * mean_r + 0.7151522 * mean_g + 0.0721750 * mean_b
    z_val = 0.0193339 * mean_r + 0.1191920 * mean_g + 0.9503041 * mean_b
    s = x_val + y_val + z_val
    if s <= 1e-9:
        return 0.0
    x = x_val / s
    y = y_val / s
    denom = 0.1858 - y
    if abs(denom) < 1e-9:
        return 0.0
    n = (x - 0.3320) / denom
    cct = 437 * n**3 + 3601 * n**2 + 6861 * n + 5517
    return max(0.0, float(cct))


def compute_lighting_reference(frame: Frame) -> LightingReference:
    b_channel, g_channel, r_channel = cv2.split(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return LightingReference(
        bins=HISTOGRAM_BINS,
        histogram_b=_compute_histogram(b_channel),
        histogram_g=_compute_histogram(g_channel),
        histogram_r=_compute_histogram(r_channel),
        mean_luminance=float(gray.mean()),
        cct_kelvin=_estimate_cct_kelvin(frame),
    )


def _histogram_chi2(ref_hist: list[float], cur_hist: list[float]) -> float:
    """Symmetric chi-square between two histograms after PDF normalization.

    Returns 0 for identical, 1.0 for disjoint distributions.
    """
    a = np.asarray(ref_hist, dtype=np.float64)
    b = np.asarray(cur_hist, dtype=np.float64)
    a = a / max(a.sum(), 1e-9)
    b = b / max(b.sum(), 1e-9)
    diff_sq = (a - b) ** 2
    summed = a + b + 1e-12
    return float(0.5 * np.sum(diff_sq / summed))


def _avg_channel_chi2(ref: LightingReference, cur: LightingReference) -> float:
    return (
        _histogram_chi2(ref.histogram_b, cur.histogram_b)
        + _histogram_chi2(ref.histogram_g, cur.histogram_g)
        + _histogram_chi2(ref.histogram_r, cur.histogram_r)
    ) / 3.0


# --- ArUco primitives ----------------------------------------------------


def _resolve_aruco_dict(name: str) -> int:
    try:
        return int(getattr(cv2.aruco, name))
    except AttributeError as exc:
        raise ValueError(f"Unknown ArUco dictionary: {name!r}") from exc


def detect_aruco_markers(
    frame: Frame, dictionary_name: str = DEFAULT_ARUCO_DICTIONARY
) -> list[ArucoMarker]:
    dictionary = cv2.aruco.getPredefinedDictionary(_resolve_aruco_dict(dictionary_name))
    params = cv2.aruco.DetectorParameters()
    # Default corner detection has ~0.5px jitter. Subpixel refinement drops it
    # to ~0.05px, which directly cuts the orientation-delta noise floor — the
    # difference between false-positive 1° drift on a static scene and real
    # sag detection at 0.5°. Costs <1ms per frame.
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    corners, ids, _ = detector.detectMarkers(frame)
    if ids is None or len(ids) == 0:
        return []
    markers: list[ArucoMarker] = []
    for marker_corners, marker_id in zip(corners, ids.flatten(), strict=False):
        pts = np.asarray(marker_corners).reshape(4, 2)
        center = pts.mean(axis=0)
        # Orientation = angle of the top edge (corner 0 → corner 1) in degrees.
        dy = float(pts[1, 1] - pts[0, 1])
        dx = float(pts[1, 0] - pts[0, 0])
        orientation = float(np.degrees(np.arctan2(dy, dx)))
        markers.append(
            ArucoMarker(
                id=int(marker_id),
                center_px=(float(center[0]), float(center[1])),
                orientation_deg=orientation,
            )
        )
    return sorted(markers, key=lambda m: m.id)


def compute_aruco_reference(
    frame: Frame, dictionary_name: str = DEFAULT_ARUCO_DICTIONARY
) -> ArucoReference:
    return ArucoReference(
        dictionary=dictionary_name,
        markers=detect_aruco_markers(frame, dictionary_name),
    )


def _aruco_pose_drift(
    reference: list[ArucoMarker], current: list[ArucoMarker]
) -> tuple[float, list[int], dict[int, float]]:
    """Aggregate per-marker rotation deltas into a single drift score.

    Returns ``(score, missing_ids, per_marker)``:

    * ``score`` — **median** absolute rotation delta across visible markers,
      in degrees. Median (not max) so a single jittery marker — common when
      one tag is partially shadowed or near the image edge — doesn't
      dominate. Real camera rotation moves all markers together, so for
      genuine drift median ≈ max.
    * ``missing_ids`` — pinned marker ids not seen in the current frame.
    * ``per_marker`` — full ``{id: signed_delta_deg}`` map for diagnostics.
      Surfaced in the finding's ``evidence`` so the operator can spot
      which marker is the outlier.
    """
    current_by_id = {m.id: m for m in current}
    missing = sorted(m.id for m in reference if m.id not in current_by_id)
    per_marker: dict[int, float] = {}
    for ref_marker in reference:
        cur_marker = current_by_id.get(ref_marker.id)
        if cur_marker is None:
            continue
        raw = cur_marker.orientation_deg - ref_marker.orientation_deg
        normalized = ((raw + 180.0) % 360.0) - 180.0
        per_marker[ref_marker.id] = float(normalized)
    if not per_marker:
        return (0.0, missing, per_marker)
    abs_deltas = [abs(v) for v in per_marker.values()]
    score = float(np.median(abs_deltas))
    return (score, missing, per_marker)


# --- orchestrator --------------------------------------------------------


def _next_id(findings: list[Finding]) -> str:
    return f"f_{len(findings) + 1:03d}"


def _lighting_fix_hint(lum_delta: float, cct_delta: float) -> str:
    if abs(cct_delta) > 500:
        warmer = cct_delta < 0
        direction = "warmer" if warmer else "cooler"
        return (
            f"Lighting color temperature shifted {direction} — check the overhead lamp."
        )
    if abs(lum_delta) > 20:
        return "Brightness changed — check the overhead lamp or for auto-exposure drift."
    return "Lighting changed — verify the lighting setup matches the pin."


def run_layer1(
    *,
    current_frame: Frame,
    lighting_ref: LightingReference,
    aruco_ref: ArucoReference,
    thresholds: ThresholdSpec,
    embedding_ref: EmbeddingReference | None = None,
    embedder: GlobalEmbedder | None = None,
) -> Layer1Output:
    findings: list[Finding] = []
    flags: list[Flag] = []

    # --- lighting / scene ---
    current_lighting = compute_lighting_reference(current_frame)
    hist_chi2 = _avg_channel_chi2(lighting_ref, current_lighting)
    lum_delta = current_lighting.mean_luminance - lighting_ref.mean_luminance
    cct_delta = current_lighting.cct_kelvin - lighting_ref.cct_kelvin

    lum_norm = min(abs(lum_delta) / _LUMINANCE_FULLSCALE, 1.0)
    cct_norm = min(abs(cct_delta) / _CCT_FULLSCALE_K, 1.0)
    lighting_drift = float(
        np.clip(
            _HIST_WEIGHT * hist_chi2 + _LUMINANCE_WEIGHT * lum_norm + _CCT_WEIGHT * cct_norm,
            0.0,
            1.0,
        )
    )

    if lighting_drift > thresholds.lighting_drift:
        severity: Severity = (
            "critical" if lighting_drift >= _LIGHTING_CRITICAL_THRESHOLD else "warning"
        )
        findings.append(
            Finding(
                id=_next_id(findings),
                severity=severity,
                layer=1,
                component="lighting",
                subject="global",
                issue="lighting_changed",
                detail=(
                    f"Lighting drifted: histogram chi^2 {hist_chi2:.3f}, "
                    f"luminance delta {lum_delta:+.1f}, "
                    f"CCT delta {cct_delta:+.0f}K"
                ),
                fix=_lighting_fix_hint(lum_delta, cct_delta),
                evidence={
                    "histogram_chi2": hist_chi2,
                    "mean_luminance_delta": lum_delta,
                    "color_temp_delta_k": cct_delta,
                },
            )
        )
        flags.append("lighting_shift")

    # --- ArUco / camera pose ---
    pose_drift_deg = 0.0
    per_marker_deltas: dict[int, float] = {}
    if aruco_ref.markers:
        current_markers = detect_aruco_markers(current_frame, aruco_ref.dictionary)
        pose_drift_deg, missing_ids, per_marker_deltas = _aruco_pose_drift(
            aruco_ref.markers, current_markers
        )
        if missing_ids:
            findings.append(
                Finding(
                    id=_next_id(findings),
                    severity="warning",
                    layer=1,
                    component="pose",
                    subject="aruco",
                    issue="markers_missing",
                    detail=f"ArUco markers not seen this run: {missing_ids}",
                    fix="Check that workspace fiducials are still in view of the camera.",
                    evidence={"missing_marker_ids": missing_ids},
                )
            )
            if "camera_pose_drift" not in flags:
                flags.append("camera_pose_drift")
        if pose_drift_deg > thresholds.max_camera_pose_drift_deg:
            pose_severity: Severity = (
                "critical" if pose_drift_deg >= _CAMERA_POSE_CRITICAL_DEG else "warning"
            )
            abs_per_marker = {mid: abs(v) for mid, v in per_marker_deltas.items()}
            max_marker_delta = (
                max(abs_per_marker.values()) if abs_per_marker else 0.0
            )
            findings.append(
                Finding(
                    id=_next_id(findings),
                    severity=pose_severity,
                    layer=1,
                    component="pose",
                    subject="camera",
                    issue="rotation_drift",
                    detail=(
                        f"Camera rotated ~{pose_drift_deg:.2f}° vs pin (median "
                        f"across {len(per_marker_deltas)} markers; worst "
                        f"marker {max_marker_delta:.2f}°)."
                    ),
                    fix="Re-level the camera mount, or repin if intentional.",
                    evidence={
                        "rotation_delta_deg": pose_drift_deg,
                        "per_marker_delta_deg": per_marker_deltas,
                        "max_marker_delta_deg": max_marker_delta,
                    },
                )
            )
            if "camera_pose_drift" not in flags:
                flags.append("camera_pose_drift")

    # --- scene_drift ---
    # Prefer the embedder cosine path; fall back to the histogram chi^2 when
    # no embedder is wired up or the pin pre-dates this feature. Only one path
    # contributes to scene_drift — blending muddies thresholds.
    scene_drift, scene_drift_source, scene_evidence = _compute_scene_drift(
        current_frame=current_frame,
        hist_chi2=hist_chi2,
        embedding_ref=embedding_ref,
        embedder=embedder,
    )
    if scene_drift > thresholds.scene_drift and "lighting_shift" not in flags:
        findings.append(
            Finding(
                id=_next_id(findings),
                severity="warning",
                layer=1,
                component="scene",
                subject="global",
                issue="scene_changed",
                detail=(
                    f"Scene drift {scene_drift:.3f} above threshold "
                    f"({scene_drift_source})."
                ),
                fix="Layer 2 (SAM3) will localize the changed region once it lands.",
                evidence=scene_evidence,
            )
        )
        flags.append("scene_drift")

    return Layer1Output(
        scene_drift=scene_drift,
        scene_drift_source=scene_drift_source,
        lighting_drift=lighting_drift,
        max_camera_pose_drift_deg=pose_drift_deg,
        findings=findings,
        flags=flags,
    )


def _compute_scene_drift(
    *,
    current_frame: Frame,
    hist_chi2: float,
    embedding_ref: EmbeddingReference | None,
    embedder: GlobalEmbedder | None,
) -> tuple[float, str, dict[str, object]]:
    """Returns (drift in [0, 1], source label, evidence dict).

    Cosine path runs only when both the reference and a live embedder are
    available *and* their names match. A name mismatch is a soft warning, not
    an error — we fall back to histogram so the check still produces a
    sensible number.
    """
    if embedding_ref is not None and embedder is not None:
        if embedder.name == embedding_ref.embedder_name:
            current_vec = embedder.embed(current_frame)
            ref_vec = np.asarray(embedding_ref.vector, dtype=np.float32)
            # Both vectors are L2-normalized by contract; cosine = dot product.
            cosine = float(np.dot(current_vec, ref_vec))
            drift = float(np.clip(1.0 - cosine, 0.0, 1.0))
            return (
                drift,
                f"embedder:{embedder.name}",
                {"cosine": cosine, "embedder": embedder.name},
            )
    drift = float(np.clip(hist_chi2, 0.0, 1.0))
    return (drift, "histogram", {"histogram_chi2": hist_chi2})


# --- persistence ---------------------------------------------------------


def lighting_path(pin_dir: Path) -> Path:
    return pin_dir / LIGHTING_FILENAME


def aruco_path(pin_dir: Path) -> Path:
    return pin_dir / ARUCO_SUBDIR / ARUCO_FILENAME


def save_lighting_reference(ref: LightingReference, pin_dir: Path) -> Path:
    target = lighting_path(pin_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(ref.model_dump_json(), encoding="utf-8")
    return target


def load_lighting_reference(pin_dir: Path) -> LightingReference:
    return LightingReference.model_validate_json(
        lighting_path(pin_dir).read_text(encoding="utf-8")
    )


def save_aruco_reference(ref: ArucoReference, pin_dir: Path) -> Path:
    target = aruco_path(pin_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(ref.model_dump_json(), encoding="utf-8")
    return target


def load_aruco_reference(pin_dir: Path) -> ArucoReference:
    return ArucoReference.model_validate_json(
        aruco_path(pin_dir).read_text(encoding="utf-8")
    )


def embedding_path(pin_dir: Path) -> Path:
    return pin_dir / EMBEDDING_FILENAME


def compute_embedding_reference(
    frame: Frame, embedder: GlobalEmbedder
) -> EmbeddingReference:
    vec = embedder.embed(frame)
    return EmbeddingReference(
        embedder_name=embedder.name,
        dim=int(vec.shape[0]),
        vector=vec.astype(np.float32).tolist(),
    )


def save_embedding_reference(ref: EmbeddingReference, pin_dir: Path) -> Path:
    target = embedding_path(pin_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        embedding=np.asarray(ref.vector, dtype=np.float32),
        embedder_name=np.array(ref.embedder_name),
    )
    return target


def load_embedding_reference(pin_dir: Path) -> EmbeddingReference | None:
    target = embedding_path(pin_dir)
    if not target.exists():
        return None
    data = np.load(target, allow_pickle=False)
    vector = data["embedding"].astype(np.float32)
    embedder_name = str(data["embedder_name"])
    return EmbeddingReference(
        embedder_name=embedder_name,
        dim=int(vector.shape[0]),
        vector=vector.tolist(),
    )


__all__ = [
    "ARUCO_FILENAME",
    "DEFAULT_ARUCO_DICTIONARY",
    "EMBEDDING_FILENAME",
    "HISTOGRAM_BINS",
    "LIGHTING_FILENAME",
    "ArucoMarker",
    "ArucoReference",
    "EmbeddingReference",
    "Layer1Output",
    "LightingReference",
    "aruco_path",
    "compute_aruco_reference",
    "compute_embedding_reference",
    "compute_lighting_reference",
    "detect_aruco_markers",
    "embedding_path",
    "lighting_path",
    "load_aruco_reference",
    "load_embedding_reference",
    "load_lighting_reference",
    "run_layer1",
    "save_aruco_reference",
    "save_embedding_reference",
    "save_lighting_reference",
]
