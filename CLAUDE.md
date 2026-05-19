# CLAUDE.md

Working context for Claude Code in the Anvil repo. Read this first when
starting a session in this directory.

## Project

Anvil is a scene-consistency sentinel for robotics data collection. Pin a
scene once; every future session is checked against the pin. Drift gets
flagged before it gets baked into your dataset.

Sibling to [Forge](https://github.com/arpitg1304/forge): Forge shapes data
post-collection; Anvil keeps the rig consistent pre-collection.

## Authoritative documents (read in this order)

1. **[anvil_plan.md](anvil_plan.md)** — full project specification. The
   *what*. Source of truth on goals, architecture, schema, scope.
2. **[prompt.txt](prompt.txt)** — working instructions. The *how*. Order
   of operations, non-negotiables, scope guards, tooling defaults. Read
   end-to-end every fresh session.
3. **[docs/metadata_schema.md](docs/metadata_schema.md)** — v0.1 sidecar
   schema (the Forge contract). Additive changes only after 1.0.

Anything in this file (CLAUDE.md) defers to those three.

## Non-negotiables (lifted from prompt.txt §"NON-NEGOTIABLES")

- The v0.1 schema in `docs/metadata_schema.md`. Forge will read against it.
- The 9-flag vocabulary in schema §6. No new flags in v0.1.
- The three CLI verbs: `pin`, `check`, `guard`. No new top-level verbs.
- The 3-layer cascade architecture. Don't invent Layer 1.5.
- The MVP "Explicitly OUT" list in plan §9. Surface, don't just add.

## Where things are

```
src/anvil/
├── schema.py         # Pydantic v2 sidecar models (the Forge contract)
├── manifest.py       # Pin manifest + YAML save/load + sha256 hashing
├── errors.py         # AnvilError hierarchy
├── cli.py            # Click app: pin / check / guard
├── cameras/          # Camera abstraction (webcam + file driver)
├── layers/           # Vision cascade (layer1_fast.py done; 0/2/3 TBD)
├── robots/           # Robot drivers (TODO: lerobot, none)
├── models/           # Model wrappers (TODO: dinov3, sam3, vlm)
├── agent/            # Smolagents loop (TODO, Week 2)
└── server/           # FastAPI + HTMX inspector (TODO, Week 3)
```

See [ROADMAP.md](ROADMAP.md) for status; [CHANGELOG.md](CHANGELOG.md) for
what landed when.

## Working style (lifted from prompt.txt §"WORKING STYLE")

- **Tooling:** `uv` for package management (no poetry / setup.py /
  requirements.txt). Ruff for lint+format. Mypy `--strict` on
  `src/anvil/`. Pytest for tests. Click for CLI. Rich for terminal
  output. Pydantic v2 for schemas. FastAPI + HTMX (Week 3) for UI —
  Tailwind via CDN, no build step. Ollama for the VLM dev path; vLLM
  documented but not required.
- **Type-hint everything.** `from __future__ import annotations` at the
  top. Absolute imports inside `anvil`, never relative. No bare `Any`
  in `src/` without a justification comment.
- **Tests run after every meaningful change.** If you generate >50 lines
  of code without running tests, slow down.
- **Commit-sized work units.** Conventional prefixes (feat, fix,
  refactor, test, docs). Each session ends green.
- **Default to no comments.** Only when the *why* is non-obvious.

## Getting started on a fresh checkout

```bash
uv sync --group dev          # installs runtime + dev deps
uv run pytest                # all tests should pass
uv run ruff check src tests
uv run mypy                  # --strict on src/anvil

# Smoke test
anvil pin --camera 0 --name workspace --task "general workspace pin"
anvil check --against workspace
```

If you don't know your camera index, see the "Finding your camera"
section in [README.md](README.md).

## Layer 1 quirks worth knowing

- `scene_drift` is **histogram-derived** in the current code; the plan
  calls for `1 - DINOv3 cosine`. Swapping that in is the natural next
  step on a GPU machine — add `torch` + `transformers` to the `[full]`
  extra, wrap in `anvil.models.dinov3`, gracefully fall back to
  histograms when the model can't load.
- McCamy's CCT approximation is **only valid near the Planckian locus**.
  For real indoor lighting it's accurate to a few hundred K, which is
  plenty for drift detection. Highly saturated frames produce garbage
  CCTs — there's a regression test fixing this.
- ArUco uses `DICT_4X4_50` and OpenCV 4.7+'s `ArucoDetector` API.
  Without camera calibration we report the **rotation delta of the
  top edge** as the pose-drift proxy, not a true 3D pose.

## Next session priorities

1. **Layer 1 DINOv3 global cosine** — see "Layer 1 quirks". Behind `[full]`.
2. **Layer 0 LeRobot driver** — `anvil.robots.{base,lerobot,none}`. Wire
   into `check` when `manifest.robot.enabled`. Joint-state diff against
   pinned home; tolerance from `RobotSpec.tolerance_deg`.
3. **Layer 2** (Week 2): SAM3 named-object workflow + SuperPoint+LightGlue
   for true camera pose drift.
4. **Layer 3** (Week 2): Qwen3-VL agent via smolagents + Ollama.
5. **Week 3**: `guard` subcommand + sidecar writer + FastAPI inspector.

## When stuck or when reality diverges

The plan was written quickly. If it conflicts with how a dep actually
behaves (smolagents API drift, SAM 3 release shifts, etc.), surface the
gap explicitly — don't silently work around it.

If you find a flaw in the v0.1 schema, **flag it and wait**. Forge code
is being written against it; don't change it unilaterally.
