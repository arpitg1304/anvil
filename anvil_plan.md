# Anvil — keep your scene true

*A scene-consistency sentinel for robotics data collection. Sibling tool to [Forge](https://github.com/arpitg1304/forge).*

*Pin a scene once. Every future session is checked against the pin. Drift gets flagged before it gets baked into your dataset.*

> **Forge** shapes your data. **Anvil** keeps your scene true. Together they form a complete data-quality pipeline: Anvil prevents bad data at collection time, Forge cleans and converts it after.

---

## 1. Why this exists

Every team training diffusion VLAs, world models, or imitation policies hits the same wall: their policy generalizes worse than it should, and they can't tell whether the cause is the model, the demonstrations, or the **conditions under which the demos were collected**. Lighting drifted. The mat got nudged 4cm. Someone replaced the red cup with a slightly different red cup. The camera mount sagged 2 degrees over a week of bumping.

This stochasticity is invisible at collection time and only shows up as a 15% drop in success rate three weeks later. By then, you can't tell which demos are contaminated.

The two existing options are both bad:
1. **Be religious about setup.** Doesn't scale, doesn't survive personnel changes, doesn't catch slow drift.
2. **Train through it with massive data and augmentation.** Works for well-funded labs, doesn't help anyone else.

**Anvil is a third option:** treat the physical scene like source code. Pin a known-good version. Every session, run a cheap automated check against the pin. Block collection (or just tag the episodes) if drift exceeds threshold. Use a VLM agent to tell the operator *specifically* what's off.

The killer downstream feature: every demo gets stamped with the deviation it was collected under, so when your policy fails you can run a correlation and see "the 23 worst-performing rollouts came from sessions with lighting Δ > 0.4."

## 2. What it does — three commands

```
anvil pin --camera 0 --name pick_red_cube --task "pick the red cube"
anvil check --against pick_red_cube
anvil guard --against pick_red_cube --record-cmd "lerobot record ..."
```

- **`pin`** captures the current scene as a reference manifest.
- **`check`** runs a one-shot diff and reports pass/fail with diagnostics.
- **`guard`** wraps a recording command, runs a check before each episode, and stamps deviation metadata onto the resulting LeRobotDataset.

A small FastAPI web UI on `localhost:7777` shows the pinned scene, current view, and any flagged diffs.

## 3. Architecture

A pre-check followed by a three-layer vision cascade. Cheap checks gate expensive ones. The VLM only fires when it's actually needed.

```
┌─────────────────────────────────────────────────────────┐
│ Layer 0 — Robot pre-check (~100ms, opt-in)              │
│   • Read joint state, compare to pinned home pose       │
│   • Visual robot-base check rolled into Layer 2 SAM 3   │
│   → Skipped entirely if no robot is configured          │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│ Layer 1 — Fast vision path (~200ms, CPU or tiny GPU)   │
│   • Histogram + color stats vs reference                │
│   • Global DINOv3 cosine similarity                     │
│   • ArUco fiducial pose (if markers present)            │
│   → Pass cleanly? Done. Most checks end here.          │
└─────────────────────────────────────────────────────────┘
                            ↓ fail or borderline
┌─────────────────────────────────────────────────────────┐
│ Layer 2 — Structural path (~1-2s, GPU)                  │
│   • SAM 3 re-segmentation of named objects             │
│     (incl. robot_base if robot enabled)                 │
│   • Per-object IoU + DINOv3 region embedding cosine    │
│   • Camera pose drift via SuperPoint + LightGlue       │
│   → Localized report: which object moved, by how much  │
└─────────────────────────────────────────────────────────┘
                            ↓ still ambiguous or fail
┌─────────────────────────────────────────────────────────┐
│ Layer 3 — VLM agent path (~5-10s, GPU)                  │
│   • Qwen3-VL-4B structured scene diff vs reference     │
│   • Agent loop: re-segment, probe, ask operator        │
│   • Natural-language fix instructions                  │
└─────────────────────────────────────────────────────────┘
```

The cascade matters because Layer 3 is the expensive one. In a well-set-up rig 95% of checks should resolve at Layer 1 in well under a second.

**On Layer 0:** Robot home-pose drift is a sneakier failure mode than scene drift because it contaminates *every demo* through proprioception, and the operator can't catch it by eye. Anvil handles it with two cheap checks: a joint-state diff against the pinned home pose (catches motor zero drift, calibration shifts, bumped base), and a visual check on `robot_base` as a SAM 3 named object (catches a slipped clamp without any robot SDK integration). Strictly opt-in — set `robot.driver: none` in the manifest and the whole layer is bypassed, keeping Anvil useful for non-robot scene-capture work.

## 4. Open-source model stack

All recommendations verified against current (May 2026) releases.

| Role | Model | Why | Size |
|---|---|---|---|
| **Dense feature backbone** | **DINOv3** (Meta, Aug 2025) | Frozen backbone hits 66.1 mAP on COCO and 64.4 NAVI recall — SOTA for self-supervised dense features. Specifically validated for scene-change detection. | ViT-S/16 (~22M) is plenty; ViT-B for harder scenes |
| **Object segmentation** | **SAM 3** (Meta, Nov 2025, SAM 3.1 Mar 2026) | Text-promptable concept segmentation — you say "red cube" and "blue plate" once, it tracks them forever. 2× the accuracy of SAM 2 on open-vocab. Integrated into Ultralytics. | 848M, runs on a 4090 |
| **Fast first-pass VLM** | **Moondream 3** (or Moondream 2B fallback) | Lowest measured per-request cost of any open VLM, runs on edge hardware. Good enough for "is anything obviously wrong" yes/no. | ~2B |
| **Judge / agent VLM** | **Qwen3-VL-4B-Instruct** (with 8B Thinking as upgrade path) | Explicit advanced spatial perception, 2D + 3D grounding, native agent capabilities, strong structured output. Best open VLM for this role as of May 2026. GLM-4.5V is a credible alternative with 3D spatial reasoning. | 4B (8GB VRAM in FP16) |
| **Keypoint matching** | **SuperPoint + LightGlue** | Standard for sub-degree camera pose drift detection | Tiny |
| **Fiducials (optional)** | **ArUco / ChArUco via OpenCV** | Ground-truth pose anchor if user can place markers | Free |

**Serving:** Ollama for the dev/MVP path (one-line install for Qwen3-VL and Moondream). vLLM or SGLang for anyone wanting production throughput. Both supported by Qwen3-VL out of the box.

**Hardware floor:** RTX 4090 (24GB) runs everything comfortably. RTX 3090 works. An RTX 3060 12GB can run the fast path + Moondream and skip Qwen3-VL judge calls. CPU-only works for the Layer 1 cascade.

## 5. The reference manifest

A `pin` produces a directory like:

```
.anvil/pick_red_cube/
├── manifest.yaml              # version, timestamps, thresholds, robot config
├── reference.png              # canonical RGB frame
├── reference_features.npz     # DINOv3 dense features (compressed)
├── objects/
│   ├── red_cube.json          # SAM 3 mask, prompt, region embedding
│   ├── blue_plate.json
│   ├── workspace_mat.json
│   └── robot_base.json        # optional, present if robot enabled
├── lighting.json              # histogram, color temp estimate, mean luminance
├── pose/
│   ├── keypoints.json         # SuperPoint descriptors of reference frame
│   └── aruco.json             # optional fiducial positions
├── robot.json                 # optional: home joint pose, driver, tolerances
├── description.txt            # VLM-generated canonical scene description
└── thresholds.yaml            # per-check pass/fail bounds (auto-tuned, user-editable)
```

The manifest is git-friendly (small text + a few binary blobs). Teams should commit `.anvil/` to their data repo.

The robot subsection of `manifest.yaml` looks like:

```yaml
robot:
  enabled: true                    # default false — tool works without a robot
  driver: lerobot                  # lerobot | ros2 | none | custom
  home_pose_joints: [0.0, -1.57, 1.57, 0.0, 1.57, 0.0]
  tolerance_deg: 0.5
  include_visual_base_check: true  # adds robot_base as a SAM 3 tracked object
```

## 6. The agent loop

Layer 3 is where it gets fun. Built on **smolagents** (HF) — small, multi-step, native tool use, ReAct-style. Tools exposed to the VLM:

- `re_segment(object_name)` — re-runs SAM 3 with the pinned prompt, returns mask + IoU vs reference
- `measure_pose_drift()` — returns translation/rotation delta in cm/deg via keypoint matching
- `probe_lighting()` — returns histogram diff, color-temp shift, mean-luminance delta
- `compare_region(bbox)` — DINOv3 cosine sim on a specific image region
- `check_robot_home()` — reads joint state, returns per-joint deltas vs pinned home (only when robot enabled)
- `ask_operator(question, options)` — surfaces a yes/no/multi-choice prompt in the web UI

Output is a structured report:

```yaml
status: drift_detected
severity: medium
findings:
  - object: red_cube
    issue: position_drift
    detail: "Moved ~6cm toward camera and 3cm right of pinned position"
    fix: "Slide it back to the X mark on the mat"
  - global: lighting
    issue: warmer_color_temperature
    detail: "Mean color temp shifted from ~5200K to ~4100K — likely a different lamp"
    fix: "Check the overhead LED, may have switched modes"
  - component: robot_home_pose
    issue: joint_drift
    detail: "Joint 3 is 1.2° off home (expected 1.57, got 1.59 rad)"
    fix: "Re-run robot calibration, or check if the base clamp slipped"
agent_questions: []
proceed: false
```

This report gets serialized into the LeRobot dataset metadata for every episode collected during the session, regardless of whether the operator chose to proceed.

## 7. LeRobot integration

LeRobotDataset v3 is the standard format (parquet shards + MP4, episode-based, metadata-rich). Anvil integrates two ways:

1. **`guard` wraps `lerobot record`** — runs a check before each episode, can block or just tag.
2. **Sidecar metadata file** — `episode_NNNNNN.anvil.json` next to each episode, with the full deviation report. Standard LeRobot tooling can ingest it as an extra column.

The killer use-case unlocks here: load any LeRobot dataset, filter episodes by anvil deviation score, retrain. Suddenly your dataset has a quality dimension that wasn't there.

## 8. Forge integration

Anvil is positioned as a sibling tool to [Forge](https://github.com/arpitg1304/forge), not a subcommand. Different lifecycle phases:

- **Anvil** runs *before/during* collection — verifies the rig is consistent with a pinned reference. Vision + robot proprioception.
- **Forge** runs *after* collection — inspects, converts, quality-scores, filters, segments, visualizes existing datasets. Proprioception + format I/O.

Together they cover both halves of data quality:
- Forge's `quality` command scores episodes on **proprioception signals** (smoothness, dead actions, gripper chatter, static periods, timestamp regularity, action saturation, action entropy, path length).
- Anvil scores sessions on **visual and environmental signals** (lighting drift, object position, camera pose, robot home, scene semantics).

**Shared metadata contract.** Anvil writes a per-episode JSON sidecar in a schema Forge (or any tool) can ingest. The intended Forge UX:

```bash
# After collection, with Anvil sidecars present:
forge filter ./my_dataset ./clean --max-scene-drift 0.3 --max-pose-drift-deg 1.0
forge filter ./my_dataset ./clean --exclude-anvil-flags lighting_shift,robot_home_drift
forge quality ./my_dataset --include-anvil   # blend Anvil signals into the 0-10 score
```

### Schema example — `episode_000042.anvil.json`

Concrete v0.1 schema. Claude Code should generate Pydantic models matching this shape in `anvil/schema.py` and treat it as the source of truth for both `pin` output and `check` output (the manifest reuses many of the same nested types).

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

### Field reference

| Field | Type | Required | Notes |
|---|---|---|---|
| `anvil_schema_version` | semver string | ✓ | Bump major on breaking changes |
| `episode_id` | string | ✓ | Matches LeRobotDataset episode naming |
| `checked_at` | ISO-8601 UTC | ✓ | When `check` ran |
| `pin.name` | string | ✓ | The pinned scene this was checked against |
| `pin.manifest_hash` | sha256 hex | ✓ | Pins the exact reference state; lets Forge detect re-pins |
| `pin.pinned_at` | ISO-8601 UTC | ✓ | When the reference was captured |
| `status` | enum | ✓ | `passed` \| `warning` \| `failed` |
| `operator_action` | enum | ✓ | `proceeded` \| `fixed_and_retried` \| `aborted` |
| `scores.*` | float 0.0-1.0 or cm/deg | ✓ | Aggregate numerics Forge can threshold on |
| `flags` | list of strings | ✓ | Categorical flags; stable vocabulary (see below) |
| `findings` | list of objects | ✓ | Per-issue detail with evidence |
| `findings[].severity` | enum | ✓ | `info` \| `warning` \| `critical` |
| `findings[].layer` | int 0-3 | ✓ | Which cascade layer raised it |
| `findings[].component` | enum | ✓ | `object` \| `lighting` \| `pose` \| `robot` \| `scene` |
| `findings[].evidence` | object | optional | Layer-specific raw numbers for debugging |
| `agent_trace` | object | optional | Present only when Layer 3 ran |
| `provenance` | object | ✓ | Reproducibility — what code/models produced this |

### Flag vocabulary (v0.1)

Stable set of categorical flags. New flags are additive (no removals without a major version bump):

- `scene_drift` — Layer 1 global similarity below threshold
- `lighting_shift` — color temp, luminance, or histogram delta beyond bounds
- `object_moved` — at least one tracked object exceeded position tolerance
- `object_missing` — a tracked object could not be located
- `object_added` — VLM/SAM detected an unexpected object
- `camera_pose_drift` — keypoint-based camera pose exceeded angular tolerance
- `robot_home_drift` — any joint exceeded `tolerance_deg`
- `robot_base_moved` — visual check on `robot_base` failed
- `vlm_uncertain` — Layer 3 ran but couldn't reach a confident verdict

### Storage layout

Sidecars live in a dedicated subdirectory alongside the LeRobot v3 dataset, keyed by episode ID. This avoids touching the parquet files (which are shared across episodes in v3) and keeps Anvil non-invasive:

```
my_dataset/
├── data/chunk-000/file_000.parquet
├── videos/chunk-000/episode_000042/cam_high.mp4
├── meta/
└── anvil/                              # added by anvil guard
    ├── manifest_hash.txt               # pin reference for the whole session
    ├── episode_000041.anvil.json
    ├── episode_000042.anvil.json
    └── episode_000043.anvil.json
```

### Compatibility policy

- `0.x` versions are MVP and may break. Pin Anvil version in Forge integration code.
- `1.0` will lock the schema. Additive changes only after that (new fields, new flag values).
- Forge should ignore unknown fields and unknown flag values rather than erroring — forward compatibility default.

Anvil ships this schema as `docs/metadata_schema.md` in the repo so Forge (or any other tool) can implement against it without depending on the Anvil Python package.

**Why sibling, not subcommand:** Anvil's dependency footprint is heavy (DINOv3 + SAM 3 + a local VLM + smolagents — easily 10GB of weights and a GPU). Folding that into Forge would either force every Forge user to deal with it or push it into an extras-pile that nobody installs. As a sibling, Anvil iterates independently while Forge stays lean.

## 9. MVP scope — 3 weeks, one person

Hard scoping. Anything tagged **OUT** is post-MVP.

### Week 1 — vector cascade, no VLM, no UI

- [ ] CLI scaffold (`pin`, `check`, `guard`) with Click
- [ ] Camera abstraction (USB/V4L2 + RealSense + IP cam via OpenCV)
- [ ] Manifest schema (Pydantic) + save/load
- [ ] Layer 1: histogram + global DINOv3 cosine + ArUco (if present)
- [ ] Layer 2 keypoints: SuperPoint + LightGlue pose drift
- [ ] Layer 0 robot home-pose check (opt-in, LeRobot driver first; `none` driver for camera-only mode)
- [ ] JSON output, exit-code semantics for shell scripting
- [ ] Tests on a fixed pair of "pinned" and "drifted" image folders (no live camera needed for CI)

**End-of-week deliverable:** `anvil pin` then `anvil check` works end-to-end, vector-only, on a USB cam, with optional robot home-pose verification.

### Week 2 — structural + VLM judge

- [ ] SAM 3 integration via Ultralytics, named-object workflow
- [ ] `robot_base` auto-added to tracked objects when robot is enabled
- [ ] Per-object IoU and DINOv3 region cosine
- [ ] Qwen3-VL-4B via Ollama for the judge call
- [ ] Smolagents loop with the 6 tools listed above (including `check_robot_home`)
- [ ] Structured YAML/JSON report
- [ ] Cascade orchestration (only run Layer N+1 if Layer N is borderline/fail)

**End-of-week deliverable:** Move an object 5cm, run check, get a report that says "red cube moved ~5cm in direction X." Bump the robot base, get "robot_base region embedding drift + joint 3 offset" in the same report.

### Week 3 — LeRobot integration + minimal UI

- [ ] `guard` subcommand wrapping arbitrary record commands
- [ ] Sidecar metadata writer matching LeRobotDataset v3
- [ ] FastAPI + HTMX inspector: pinned image, live image, latest report, history timeline
- [ ] Auto-threshold tuning from a small "known-good variation" calibration set
- [ ] README, install instructions, one-camera demo notebook
- [ ] Public repo + MIT license + initial release

**End-of-week deliverable:** Open-sourceable v0.1.0 with a 2-minute demo video.

### Explicitly OUT of MVP

- Multi-camera setups (single camera only)
- Stereo depth or 3D scene reconstruction
- Auto-correction (only flag, never act on the rig)
- Cloud sync / hosted version / multi-user
- Real-time mid-episode monitoring (only pre-episode checks)
- Anything beyond English language prompts
- Custom fine-tuning of any model
- **Robot-side**: end-effector pose checks via FK (needs URDF, per-robot), hand-eye calibration verification, repeatability checks (move-to-home N times, measure variance), trajectory replay sanity checks, non-LeRobot robot drivers

## 10. Hardware assumptions

Aligned with your basement rig:
- One ceiling-mounted camera (USB or IP, doesn't matter)
- A workstation with at least 12GB VRAM (RTX 3060 12GB is the floor for the full stack; 4090 is comfortable)
- Optional: a few printed ArUco markers on the workspace edges (huge accuracy win for pose drift detection, ~$0 to add)
- Optional: a robot reachable via LeRobot (enables Layer 0 home-pose check; tool works fine without it)

Stretch: integrate with `lerobot record` running on the same machine.

## 11. Risks and what could go wrong

Honest assessment.

| Risk | Mitigation |
|---|---|
| **DINOv3 features too coarse to catch small object moves** | Layer 2 uses SAM 3 masks to get *per-object* embeddings, which is much sharper than global features. Validated by the DINOv2-cross-attention scene-change paper. |
| **VLM hallucinates differences that aren't there** | Structured output schema with grounded fields (must reference a specific object or region from Layer 2); never let the VLM make a claim that wasn't first localized by a non-VLM tool |
| **Threshold tuning is the actual hard problem** | Ship auto-tuning: collect 10 "known-good" frames after pinning, fit a normal distribution to each metric, set threshold at 3σ. User can override. |
| **Camera mounting sags over weeks → false positives** | This is a feature, not a bug — that's exactly the drift you want caught. But provide a `repin` command to re-baseline when the change is intentional. |
| **Operator habituation: people start clicking past warnings** | Layer 3 generates *specific* fix instructions ("slide cube 5cm right toward the tape mark") rather than vague warnings, which research on alarm fatigue suggests is the only thing that actually works. |
| **Hardware floor too high to be a useful OSS tool** | Layers 1-2 work without the VLM. Make the VLM strictly optional in config. A laptop-only mode is realistic. |

## 12. Open-sourcing strategy

- **License:** MIT.
- **Repo:** `anvil/anvil` on GitHub.
- **Discoverability:** post to LeRobot community Discord, r/robotics, HN Show. Hugging Face Spaces demo with a webcam.
- **First-month goal:** five external users who pin a scene. Talk to all of them.
- **What earns adoption:** make it work in 5 minutes from `pip install anvil` to a working pin on a laptop webcam. The Layer 3 VLM stuff is for the README screenshots; the Layer 1 path is what gets people in the door.

## 13. Why this is a good portfolio project specifically

It checks all the boxes you mentioned:
- **Robotics foundation models adjacent** — uses DINOv3, SAM 3, Qwen3-VL exactly the way papers in this space use them.
- **Solves a real pain** — every team building diffusion VLAs / world models eats this problem.
- **Agentic flavor** — the Layer 3 loop is a clean, well-scoped agent use case.
- **Small enough to ship** — 3 weeks, one person, no novel research required.
- **Compounds** — each downstream feature (dataset filtering, failure correlation, automated curriculum) is a paper-shaped extension.
- **Ecosystem leverage** — pairs with Forge to form a complete robotics data-quality stack. Cross-promotes naturally (Forge users are exactly the audience for Anvil, and vice versa) and the shared metadata schema is a defensible piece of design.

If you decide to push it further after MVP, the most interesting next step isn't more features — it's **correlating deviation metadata with downstream policy success rate** across a public dataset like Open-X-Embodiment. That's the kind of result that turns a tool into a paper.

---

## Naming — decided

**Name:** `Anvil`. Sibling to Forge in the robotics-smithy metaphor.

**Why:** Forge shapes the data, Anvil holds the work true. The pairing is visceral and the metaphor maps exactly: an anvil is a known-good rigid reference surface that doesn't move — which is precisely what the tool verifies about your collection rig.

**PyPI namespace handling:** the bare `anvil` package name is taken on PyPI (abandoned 2014 Jinja template generator), and the Anvil.works web framework owns adjacent names (`anvil-app-server`, `anvil-uplink`, `python-anvil`). Same problem you solved for Forge — follow the same pattern:

| Surface | Name |
|---|---|
| GitHub repo | `arpitg1304/anvil` |
| PyPI package | `anvil-robotics` |
| CLI binary | `anvil` |
| Python import | `import anvil` |

```bash
pip install anvil-robotics
anvil pin --camera 0 --name pick_red_cube
anvil check --against pick_red_cube
anvil guard --against pick_red_cube --record-cmd "lerobot record ..."
```
