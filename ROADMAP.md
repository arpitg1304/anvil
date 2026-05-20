# Roadmap

Living document. Full context: [anvil_plan.md](anvil_plan.md).

## v0.1.0 — MVP (3 weeks)

### Week 1 — vector cascade, no VLM, no UI
- [x] Schema + Pydantic models + tests (the Forge contract)
- [x] CLI scaffold (`pin`, `check`, `guard`) with `pin` end-to-end
- [x] Camera abstraction (USB/V4L2 webcam + file driver for tests)
- [x] Manifest schema + save/load + content hashing
- [x] JSON output, exit-code semantics (0 / 1 / 2 / 3)
- [x] Layer 1: histogram + ArUco + composite drift scores
- [x] Layer 1: DINOv3 global cosine (gated behind `[full]` extra)
- [ ] Layer 2 keypoints: SuperPoint + LightGlue
- [ ] Layer 0: LeRobot driver home-pose check (opt-in)

### Week 2 — structural + VLM judge
- [ ] SAM 3 via Ultralytics
- [ ] `robot_base` as a tracked object when robot enabled
- [ ] Per-object IoU + DINOv3 region cosine
- [ ] Qwen3-VL-4B via Ollama
- [ ] Smolagents loop (6 tools)
- [ ] Cascade orchestration

### Week 3 — LeRobot integration + minimal UI
- [ ] `guard` wraps arbitrary record commands
- [ ] Sidecar writer matching LeRobotDataset v3
- [ ] FastAPI + HTMX inspector (localhost:7777)
- [ ] Auto-threshold tuning
- [ ] README, install, demo notebook
- [ ] Public repo + initial release

## Post-MVP (not committed)

- Multi-camera + stereo depth / 3D
- Auto-correction on the rig (currently flag-only)
- Non-LeRobot robot drivers (ROS2 etc)
- End-effector FK pose checks
- Cloud sync / hosted version
- Real-time mid-episode monitoring
- Non-English prompts
- Custom model fine-tuning
