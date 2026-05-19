# Anvil sidecar metadata schema — v0.1.0

> **Status:** v0.1. May break in 0.x. Will lock at v1.0.0 (additive changes only after that).
> **Consumer contract:** unknown fields MUST be ignored, not error. Unknown
> flag values MUST be ignored, not error. This is the forward-compatibility
> default both Anvil itself and downstream tools (Forge, etc.) follow.

Anvil writes one JSON file per LeRobotDataset v3 episode:

```
my_dataset/anvil/episode_NNNNNN.anvil.json
my_dataset/anvil/manifest_hash.txt        # pin reference for the whole session
```

`anvil check` writes the same schema standalone (no `episode_id` enforcement
required by downstream tooling). The `pin` command also reuses many of the
nested types listed below for its own manifest output, but the manifest's
top-level structure is documented separately.

This document is the **source of truth** for [`anvil.schema`](../src/anvil/schema.py).
Forge and any other consumer can implement against this file without depending
on the Anvil Python package.

---

## 1. Example

```json
{
  "anvil_schema_version": "0.1.0",
  "episode_id": "episode_000042",
  "checked_at": "2026-05-19T14:32:18.421Z",
  "pin": {
    "name": "pick_red_cube",
    "manifest_hash": "sha256:7f3a1b9c4e8d2a6f9b3c5e7d1a8f4c2e6b9d3a5f7c1e8b4a6d2f9c3e5b7a1d8f",
    "pinned_at": "2026-05-12T09:14:03.000Z"
  },
  "status": "warning",
  "operator_action": "proceeded",
  "scores": {
    "scene_drift": 0.27,
    "lighting_drift": 0.41,
    "max_object_drift_cm": 5.8,
    "max_camera_pose_drift_deg": 0.4,
    "max_robot_joint_drift_deg": 1.2
  },
  "flags": ["lighting_shift", "robot_home_drift"],
  "findings": [
    {
      "id": "f_001",
      "severity": "warning",
      "layer": 2,
      "component": "object",
      "subject": "red_cube",
      "issue": "position_drift",
      "detail": "Moved ~5.8cm toward camera and 2.1cm right of pinned position",
      "fix": "Slide back to the X mark on the mat",
      "evidence": {
        "iou": 0.62,
        "region_cosine": 0.89,
        "centroid_delta_px": [42, -89]
      }
    },
    {
      "id": "f_002",
      "severity": "warning",
      "layer": 1,
      "component": "lighting",
      "subject": "global",
      "issue": "warmer_color_temperature",
      "detail": "Mean color temp shifted from ~5200K to ~4100K",
      "fix": "Check the overhead LED, may have switched modes",
      "evidence": {
        "histogram_chi2": 0.34,
        "mean_luminance_delta": -18.2,
        "color_temp_delta_k": -1100
      }
    },
    {
      "id": "f_003",
      "severity": "critical",
      "layer": 0,
      "component": "robot",
      "subject": "joint_3",
      "issue": "joint_drift",
      "detail": "Joint 3 is 1.2° off home (expected 1.5707, got 1.5916 rad)",
      "fix": "Re-run robot calibration, or check if the base clamp slipped",
      "evidence": {
        "expected_rad": 1.5707,
        "measured_rad": 1.5916,
        "delta_deg": 1.197,
        "tolerance_deg": 0.5
      }
    }
  ],
  "agent_trace": {
    "vlm_invoked": true,
    "vlm_model": "Qwen/Qwen3-VL-4B-Instruct",
    "tool_calls": 4,
    "questions_asked_operator": 0
  },
  "provenance": {
    "anvil_version": "0.1.0",
    "models": {
      "dinov3": "facebook/dinov3-vits16-pretrain-lvd1689m",
      "sam3": "facebook/sam3-base@3.1",
      "vlm": "Qwen/Qwen3-VL-4B-Instruct",
      "keypoints": "lightglue/superpoint_v1"
    },
    "hardware": {
      "gpu": "NVIDIA GeForce RTX 4090",
      "cuda": "12.4"
    }
  }
}
```

## 2. Top-level fields

| Field | Type | Required | Notes |
|---|---|---|---|
| `anvil_schema_version` | semver string (`^\d+\.\d+\.\d+$`) | ✓ | Bump major on breaking changes. Consumers should accept any version with the same major. |
| `episode_id` | string | ✓ | Matches the LeRobotDataset episode naming (e.g. `episode_000042`). |
| `checked_at` | ISO-8601 UTC datetime (timezone-aware) | ✓ | When `check` ran. Naive datetimes MUST be rejected. |
| `pin` | object | ✓ | Reference the check ran against. See §3. |
| `status` | enum | ✓ | `passed` \| `warning` \| `failed` |
| `operator_action` | enum | ✓ | `proceeded` \| `fixed_and_retried` \| `aborted` |
| `scores` | object | ✓ | Aggregate numerics Forge can threshold on. See §4. |
| `flags` | list of strings | ✓ | Categorical flags from the v0.1 vocabulary. See §6. May be empty. |
| `findings` | list of objects | ✓ | Per-issue detail with evidence. See §5. May be empty. |
| `agent_trace` | object | optional | Present only when Layer 3 ran. See §7. |
| `provenance` | object | ✓ | Reproducibility — what code/models produced this. See §8. |

## 3. `pin` object

| Field | Type | Required | Notes |
|---|---|---|---|
| `name` | string | ✓ | The pinned scene name (`pick_red_cube`, etc). |
| `manifest_hash` | string matching `^sha256:[0-9a-f]{64}$` | ✓ | Pins the exact reference state. Lets Forge detect re-pins. |
| `pinned_at` | ISO-8601 UTC datetime (timezone-aware) | ✓ | When the reference was originally captured. |

## 4. `scores` object

All fields required. The first two are dimensionless `[0.0, 1.0]`; the rest are
physical units and MUST be `>= 0.0`.

| Field | Type | Range | Notes |
|---|---|---|---|
| `scene_drift` | float | `[0.0, 1.0]` | Global Layer 1 similarity-derived drift (1.0 = totally different scene). |
| `lighting_drift` | float | `[0.0, 1.0]` | Composite of histogram + color-temp + luminance deltas. |
| `max_object_drift_cm` | float ≥ 0 | cm | Largest per-object positional delta in Layer 2. |
| `max_camera_pose_drift_deg` | float ≥ 0 | degrees | SuperPoint+LightGlue camera pose delta. |
| `max_robot_joint_drift_deg` | float ≥ 0 | degrees | Largest joint deviation from pinned home pose. `0.0` when the robot is disabled. |

## 5. `findings[]` object

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | string | ✓ | Stable per-report identifier (`f_001`, `f_002`, ...). |
| `severity` | enum | ✓ | `info` \| `warning` \| `critical` |
| `layer` | int `[0, 3]` | ✓ | Which cascade layer raised this finding. |
| `component` | enum | ✓ | `object` \| `lighting` \| `pose` \| `robot` \| `scene` |
| `subject` | string | ✓ | Free-form name (object name, joint name, `global`, ...). |
| `issue` | string | ✓ | Short slug like `position_drift`, `joint_drift`, `warmer_color_temperature`. |
| `detail` | string | ✓ | Human-readable description. |
| `fix` | string | ✓ | Concrete actionable instruction. |
| `evidence` | object | optional | Layer-specific raw numbers for debugging. Schema is open by layer. |

`component` notes:
- `object` — a SAM-tracked named object (`red_cube`, `blue_plate`, ...).
- `lighting` — global lighting metric.
- `pose` — camera-pose drift.
- `robot` — robot home-pose / joint / base-clamp issue.
- `scene` — global scene-similarity issue not localized to a specific object.

## 6. Flag vocabulary (v0.1, locked)

Additive only — new values come in minor versions, removals require a major
bump. Unknown values MUST be ignored by consumers (forward compat).

| Flag | Raised when |
|---|---|
| `scene_drift` | Layer 1 global similarity below threshold |
| `lighting_shift` | Color temp, luminance, or histogram delta beyond bounds |
| `object_moved` | At least one tracked object exceeded position tolerance |
| `object_missing` | A tracked object could not be located |
| `object_added` | VLM/SAM detected an unexpected object |
| `camera_pose_drift` | Keypoint-based camera pose exceeded angular tolerance |
| `robot_home_drift` | Any joint exceeded `tolerance_deg` |
| `robot_base_moved` | Visual check on `robot_base` failed |
| `vlm_uncertain` | Layer 3 ran but couldn't reach a confident verdict |

## 7. `agent_trace` object (optional)

Present only when Layer 3 (VLM agent) fired.

| Field | Type | Notes |
|---|---|---|
| `vlm_invoked` | bool | Always `true` when this object exists. |
| `vlm_model` | string | HF model id (e.g. `Qwen/Qwen3-VL-4B-Instruct`). |
| `tool_calls` | int ≥ 0 | Smolagents tool invocations during this check. |
| `questions_asked_operator` | int ≥ 0 | `ask_operator` tool invocations. |

## 8. `provenance` object

| Field | Type | Required | Notes |
|---|---|---|---|
| `anvil_version` | string | ✓ | Package version. |
| `models` | `{string: string}` | ✓ | Map of role → model checkpoint id. May be empty when no models ran. |
| `hardware` | `{string: string}` | ✓ | Map of key → value (e.g. `gpu`, `cuda`, `cpu`, `ram`). May be empty. |

## 9. Compatibility policy

- `0.x` versions are MVP and may break without notice. Pin Anvil version in
  integration code.
- `1.0.0` will lock the schema. After that, only additive changes (new
  optional fields, new flag values) without a major bump.
- Consumers MUST ignore unknown fields and unknown flag values rather than
  erroring. This is the forward-compatibility default.

## 10. Storage layout

```
my_dataset/
├── data/chunk-000/file_000.parquet
├── videos/chunk-000/episode_000042/cam_high.mp4
├── meta/
└── anvil/                              # added by `anvil guard`
    ├── manifest_hash.txt               # pin reference for the whole session
    ├── episode_000041.anvil.json
    ├── episode_000042.anvil.json
    └── episode_000043.anvil.json
```

The sidecars live in their own subdirectory next to the LeRobotDataset to
avoid touching parquet files (which are shared across episodes in v3) and to
keep Anvil non-invasive.
