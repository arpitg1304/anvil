# ArUco marker setup

ArUco fiducials are **optional** — Layer 1 works without them — but if you
print and place four of them, Anvil gets a much sharper signal on a
specific class of failure that's otherwise invisible to histogram-based
checks.

## What the markers buy you

| Anvil signal | Without markers | With markers |
|---|---|---|
| `max_camera_pose_drift_deg` | Always 0 | Real angular delta from marker rotation |
| `camera_pose_drift` flag | Never raised | Raised when the camera mount sags, twists, or shifts |
| `markers_missing` finding | Never raised | Raised if something occludes the workspace or the camera moved drastically |
| Pin sanity check | Limited | `anvil pin` reports exactly how many markers it can see |

**Primary use: catching camera pose drift.** A ceiling-mounted webcam
loosens, sags, or gets bumped over weeks of bench work. That's the
classic "my policy started failing and I don't know why" cause. Four
corner markers turn that invisible drift into a single number
(`max_camera_pose_drift_deg`) you can threshold on.

**What they do NOT help with** (different layers handle these):

- Object position drift → Layer 2 SAM3 (Week 2)
- Robot home pose → Layer 0 LeRobot driver
- Lighting changes → Layer 1 histogram + CCT (already wired)

## Recommended placement

Four markers, one at each corner of the work surface — flat, fixed,
always in the camera's field of view, outside the robot's reach.

**Overhead view** (looking down through the ceiling camera):

```
+-----+---------------------------------+-----+
|  0  |                                 |  1  |
+-----+                                 +-----+
        ┌─────────────────────────────┐
        │                             │
        │      robot working area     │
        │       (no markers here)     │
        │                             │
        │      [ robot ]              │
        │                             │
        └─────────────────────────────┘
+-----+                                 +-----+
|  3  |                                 |  2  |
+-----+---------------------------------+-----+
            ← workspace mat / table →
```

**Why the corners specifically:** when the camera rotates slightly
around its optical axis, opposite-corner markers move in opposite
directions in the image. Spreading the markers maximally across the
frame gives the rotation-delta math the strongest signal it can get.
Clustered markers, or markers in the center, are far less sensitive.

The IDs themselves (`0`/`1`/`2`/`3`) are just identifiers — Anvil
matches by ID, not by spatial layout, so you can pick any four IDs from
the `DICT_4X4_50` range (0–49). The numbering above is just convention
(clockwise from top-left).

## Specs

- **Dictionary:** `DICT_4X4_50` — hardcoded in Anvil's
  [layer1_fast.py](../src/anvil/layers/layer1_fast.py). Any other dictionary
  will detect zero markers silently.
- **IDs:** 0, 1, 2, 3 (or any four unique IDs in 0–49).
- **Physical size:** **~10 cm** per marker for an overhead webcam at
  1–1.5 m. 8 cm is the minimum; larger is more robust to blur and low
  light.
- **Quiet zone:** Leave ~5 mm of white around the black border (print
  with margins; don't trim flush to the edge).
- **Paper:** Matte if possible — glossy reflects overhead LEDs and
  confuses detection.

## Generate

One-liner using Anvil's own OpenCV install, so the markers are
guaranteed to match what Anvil decodes:

```bash
uv run --project <path-to-anvil> python3 -c "
import cv2
d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
for i in range(4):
    cv2.imwrite(f'marker_{i}.png', cv2.aruco.generateImageMarker(d, i, 1200))
"
```

Writes `marker_0.png` … `marker_3.png` at 1200×1200 px. In your printer
dialog, set the size to "fit to 10 cm wide" (or scale to whatever
physical size you chose).

## Fix in place, then sanity-check

1. Glue or double-stick-tape the markers flat. Any curl or shift will
   later be read as drift.
2. Run a sanity pin:

   ```bash
   anvil pin --camera 0 --name marker_test --task "marker sanity check"
   ```

3. Confirm the output line:

   ```
   aruco:         4 marker(s) [0, 1, 2, 3]
   ```

   If you see fewer than 4, see Troubleshooting.

4. Move the camera 1° on purpose, then:

   ```bash
   anvil check --against marker_test
   ```

   You should see a `camera_pose_drift` flag and a finding like:

   ```
   WARNING [L1 pose] Camera rotated ~1.20 degrees vs pin (ArUco-derived).
   ```

   Re-pin once everything is set the way you want it for production.

## Troubleshooting

- **Fewer than 4 markers detected at pin time:**
  - A marker is partially out of frame → move them inward slightly.
  - Glossy paper reflecting the overhead LED → reprint on matte.
  - Quiet zone trimmed too tight → reprint with margins.
  - Markers too small relative to camera distance → reprint larger.
- **Detection flickers between runs (sometimes 4, sometimes 3):**
  - Marginal lighting or motion blur → enlarge markers, or stiffen the
    overhead lighting.
- **`camera_pose_drift` keeps firing when you haven't moved the camera:**
  - One marker has shifted physically (tape gave way, paper curled).
    Check each marker, re-glue, re-pin.
- **`camera_pose_drift` doesn't fire when you obviously moved the
  camera:**
  - Check `manifest.thresholds.max_camera_pose_drift_deg` (default
    `1.0`). Lower it if you want tighter tolerance, or repin.

## Once you have markers, what changes downstream

`EpisodeReport.scores.max_camera_pose_drift_deg` becomes a meaningful
number rather than a constant 0. Forge (the sibling tool) will be able
to filter on it, e.g.:

```bash
forge filter ./my_dataset ./clean --max-camera-pose-drift-deg 0.5
```

so the discipline of putting markers up at the start of a project pays
dividends every time you re-cut your dataset later.
