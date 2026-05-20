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

## Sample markers (printable)

Four print-ready PNGs are checked in next to this doc:

- [marker_0.png](marker_0.png)
- [marker_1.png](marker_1.png)
- [marker_2.png](marker_2.png)
- [marker_3.png](marker_3.png)

Each is a ~13 cm × 13 cm canvas at 305 DPI containing:

- a 10 cm `DICT_4X4_50` marker (centered)
- a 5 mm white quiet zone around the marker
- a thin black **cut line** at the 11 cm boundary
- ~1 cm of printer-safe white margin outside the cut line

```
┌──────────────────────────────┐  ← page edge (~13 cm)
│                              │
│   ┌──────────────────────┐   │  ← cut line (11 cm)
│   │                      │   │
│   │   ███████████████    │   │
│   │   ██   ArUco    ██   │   │  ← marker (10 cm)
│   │   ██   marker   ██   │   │
│   │   ███████████████    │   │
│   │                      │   │
│   └──────────────────────┘   │
│                              │
└──────────────────────────────┘
```

Cut along the printed black line — not the page edge — and you get an
11 cm square with the marker correctly inset by 5 mm of white on every
side.

## Regenerating from scratch

If you want different IDs, a different physical size, or you've lost
the sample PNGs, this one-liner reproduces them. Uses Anvil's own
OpenCV install so the marker bits are guaranteed to match what Anvil
decodes:

```bash
uv run --project <path-to-anvil> python3 -c "
import cv2, numpy as np
DPI = 305
PX_PER_MM = DPI / 25.4
MARKER_PX = int(round(100 * PX_PER_MM))   # 10 cm marker
QUIET_PX  = int(round(5   * PX_PER_MM))   # 5 mm quiet zone
SAFETY_PX = int(round(10  * PX_PER_MM))   # 10 mm printer-safe margin outside cut line
CANVAS_PX = MARKER_PX + 2*(QUIET_PX + SAFETY_PX)
d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
for i in range(4):
    marker = cv2.aruco.generateImageMarker(d, i, MARKER_PX)
    canvas = np.full((CANVAS_PX, CANVAS_PX), 255, dtype=np.uint8)
    off = SAFETY_PX + QUIET_PX
    canvas[off:off+MARKER_PX, off:off+MARKER_PX] = marker
    cv2.rectangle(canvas, (SAFETY_PX, SAFETY_PX),
                  (CANVAS_PX-SAFETY_PX-1, CANVAS_PX-SAFETY_PX-1), 0, 2)
    cv2.imwrite(f'marker_{i}.png', canvas)
print(f'wrote marker_0..3.png at {CANVAS_PX}px square (~13cm at 305 DPI)')
" && sips -s dpiHeight 305 -s dpiWidth 305 marker_*.png
```

Change `100` (the marker side in mm) to resize. The `sips` step at the
end embeds the DPI metadata so macOS Preview prints at the correct
physical size without per-page percentage math.

## Print on macOS

1. `open docs/marker_0.png` (or whichever you're printing).
2. **⌘P** → click **"Show Details"** if you see the compact dialog.
3. **Scale: 100%** — *do not* use "Scale to fit". With the embedded
   305 DPI metadata, 100% is the correct physical size.
4. Paper: US Letter or A4, orientation either way (one marker per page).
5. Print, then **cut along the printed black line** (not the page edge).
6. **Verify with a ruler** that the cut square is ~11 cm and the marker
   black border sits ~5 mm inside it.

If your printer driver ignores the DPI metadata (the cut square comes
out the wrong size at 100%), fall back to **Custom Scale: 24%** —
Preview otherwise treats the PNG as 72 DPI and that percentage maps the
1561 px canvas to ~13 cm.

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
