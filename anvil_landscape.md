# Anvil — strategic landscape

*A survey of the robotics data infrastructure and evaluation ecosystem as of May 2026, and where Anvil fits commercially.*

> **TL;DR:** The robotics data infrastructure space is on fire — $40M Series Bs, NVIDIA partnerships, YC companies, multiple Bessemer portfolio plays. Every player operates *post-collection* (validating data after it's been recorded) or *in the cloud* (storing, searching, visualizing). The *pre-collection, on-rig physical verification* layer is essentially empty. That's Anvil. The natural commercial extension is "evals-as-a-service" certification — and there are already 2-3 companies who would buy that on day one.

---

## 1. The market reality

Robotics data infrastructure is now a real venture category with serious capital flowing in.

- **Foxglove** raised **$40M Series B** in Nov 2025 (Bessemer-led, with Eclipse and Amplify). Customers include Amazon, Anduril, NVIDIA, Shield AI, Dexterity, Wayve. Pitched as "the data stack and ML platform that every other robotics startup can use, so they don't have to reinvent the wheel." Open-sourced the MCAP file format (adopted by ROS 2 and NVIDIA Isaac).
- **Voxel51** is in Bessemer's portfolio and is now actively positioning into robotics data infrastructure, with a recent NVIDIA partnership through the Physical AI Data Factory Blueprint.
- **Pitchbook**: robotics and physical AI startups raised **$27.6B across 1,009 deals in 2025**, more than 2× the 2024 total.
- **Bessemer's 2026 thesis** says it directly: *"The defining advantage in physical AI will not be model novelty, but the quality of the data infrastructure behind it. As models converge, the companies that win will be the ones with the strongest data flywheel."*

This is the relevant tailwind. Anvil ships into a market that's already convinced it needs better data tooling.

## 2. Map of adjacent players

| Player | What they do | Where they sit | Relevance to Anvil |
|---|---|---|---|
| **Foxglove** | Multimodal data platform: ingest, store, search, visualize, curate robot logs. $40M Series B. SOC 2 Type II. Customers: Amazon, Anduril, NVIDIA, Shield AI. | Post-collection cloud platform | Category leader for the "after the data exists" half. Anvil complements them — *they don't do pre-collection rig verification*. Their "Data Search and Curation" launch in April 2026 hints they want this layer; could be acquirer or partner. |
| **Voxel51 / FiftyOne** | Versioned dataset catalog, custom workspaces, integration with labeling and simulation. Pivoting toward mobility/robotics. | Dataset cataloging + curation | Strong in the "build a queryable view over your data" layer. Doesn't touch the rig. |
| **Roboto AI** | Seattle, 2022. "Data platform for robotics and embodied AI": search, transform, analyze multimodal logs. | Robotics log analytics | Direct competitor to Foxglove. Same post-collection layer. |
| **Encord** | Computer vision data labeling platform with robotics extension. | Annotation | Different layer (labeling), adjacent to data quality. |
| **Formant / InOrbit** | Fleet management for deployed robots. | Production fleet ops | Different lifecycle phase (deployed, not training). |
| **Traceplane** | Brand new (April 2026 launch blog post, waitlist mode). Automated QA at *dataset ingest* time. Validates structural integrity, metadata correctness, timestamp alignment, schema consistency. | Post-collection cloud QA | **Closest thesis match.** Same belief that "robotics needs the data validation layer every other industry has." But they validate *after* the data exists, on the data itself. Anvil validates the *physical conditions of collection*, before/during. Genuinely complementary — they could ship Anvil sidecars as part of their ingest report. Likely candidate for partnership or integration. |
| **Claru** | Commercial "benchmark-grade data collection across 100+ real-world environments with calibrated sensor rigs, standardized object sets, millimeter-precise initial state placement using custom alignment jigs." | Eval-grade data collection as a service | **Active player in the exact eval-rig market the user identified.** Sells managed collection in RLDS/HDF5 with reproducibility docs. Already has the customer pain Anvil solves — they need to *prove* their rigs are consistent across 100+ sites. Likely Anvil customer. |
| **Cortex AI** (YC) | "Human-in-the-Loop Rollouts & Evals — real-world deployments with remote operators who recover robots when they fail." Cortex Marketplace where workplaces get paid to host data-collection and evaluation sessions. | Eval-as-a-service marketplace | Building exactly the company the user predicted. Different bet (humans-in-the-loop everywhere); needs rig consistency to scale. Likely Anvil customer or partner. |
| **NVIDIA Isaac Lab-Arena** | Open-source framework for simulation-based policy evaluation. Co-developed with Lightwheel ("physical AI infrastructure company"). Feb 2026. | Simulation-based eval | Different bet entirely (eval in sim, not on physical rigs). Sets the standard for sim-eval, but doesn't solve the real-world eval problem. |
| **NVIDIA Physical AI Data Factory Blueprint** | Open NVIDIA + Nebius pipeline for generating training data via simulation + Cosmos world models. Early users: Voxel51, FieldAI, Hexagon, Skild AI, Uber. | Synthetic data generation | Different stack (sim → train), but signals that NVIDIA is treating "data infrastructure" as a category. |
| **AGIBOT WORLD 2026** | Open-source heterogeneous dataset. Hundreds of hours of real-world data. Tasks, subtasks, error-recovery trajectories annotated. | Open data corpus | Demand-side signal that big labs are publishing data that downstream tools need to validate. |

## 3. The empty quadrant

Plot the space on two axes — *when* the tool operates (pre-collection / post-collection) and *what* it operates on (data / physical rig):

```
                        ┌──────────────────────────────┬──────────────────────────────┐
                        │   PRE-COLLECTION             │   POST-COLLECTION            │
                        │                              │                              │
        PHYSICAL RIG    │   ★ Anvil (open)             │   (mostly empty —            │
                        │                              │    Claru ships QA reports    │
                        │                              │    with delivered data)      │
                        │                              │                              │
                        ├──────────────────────────────┼──────────────────────────────┤
                        │                              │                              │
        DATA            │   (RoboArena, SimplerEnv      │   Foxglove, Voxel51,         │
                        │    for eval-time data)       │   Traceplane, Forge,         │
                        │                              │   Roboto AI                  │
                        │                              │                              │
                        └──────────────────────────────┴──────────────────────────────┘
```

Three observations:

1. **The top-left quadrant has no one.** Nobody is selling "verify your physical collection rig is in the state it was the last time you used it." It's a category waiting to be named.
2. **Adjacent players treat the rig as out-of-scope.** Foxglove, Voxel51, Traceplane all assume "data exists, do something with it." They don't go upstream to ask whether the data should have been collected at all.
3. **The bottom-left exists in academic form but isn't productized.** RoboArena and SimplerEnv are research artifacts. No one has built a tool that gets used during every collection session by every lab in the field.

## 4. The "evals-as-a-service" thesis — sharpened

The user's instinct: there's a bigger company around this. Here's why it's defensible.

### The problem nobody has solved yet

Every robotics paper that reports real-world results has a reproducibility footnote. Lynnerup et al. (CoRL 2020) literally surveyed this — RL on real robots is "notoriously hard to reproduce" because of environment stochasticity and unreported hyperparameters. The robomimic team found that *validation loss doesn't predict policy performance* — best validation checkpoint is 50-100% worse than best overall. SIMPLER and RoboArena were both built explicitly to address the reproducibility crisis in real-world robot eval.

**The current state of the art for "I trust this evaluation":**
- For sim eval: use SIMPLER or Isaac Lab-Arena.
- For real eval: ship robots to a centralized challenge (DARPA, Amazon Picking), or distribute via RoboArena and use Elo aggregation.
- For commercial labs: pay Claru, Cortex AI, or similar to run evals on your behalf.

What's missing in all three: **a portable, machine-verifiable certificate that "the rig under which policy X was evaluated is the same rig under which policy Y was evaluated."** RoboArena explicitly chose to *embrace* environment variation rather than control it. The commercial players (Claru, Cortex) have to take their consistency claims on faith.

### Where Anvil becomes a moat

Three plays, in order of ambition:

**Play 1 — The standard.** Anvil's `manifest_hash` becomes how people cite eval conditions in papers. "Evaluated on rig `sha256:7f3a...` over 50 episodes." Suddenly "what rig did you use" has a checkable answer. Open-source, free, becomes infrastructure. Like git commit hashes for physical setups.

**Play 2 — The certificate.** Commercial layer: "Anvil-certified" eval rig. Means the rig has passed cross-rig consistency checks against a reference manifest, deviation tagged on every collected episode, audit trail immutable. Sellable to eval-as-a-service companies (Claru, Cortex) as a trust signal they can pass to their customers. Sellable to research labs as a publication-grade reproducibility artifact.

**Play 3 — The eval company.** Run an actual eval service that uses Anvil as the QA spine. Labs submit a policy + a target task. You evaluate it on a network of Anvil-certified rigs across multiple environments. Output is an Elo-style ranking with full deviation provenance per episode. Effectively RoboArena + Claru, but with cryptographic proof of consistency baked into every result.

The acquirers and partners for each:

| Play | Natural acquirer / partner | Why |
|---|---|---|
| Standard | Foxglove, NVIDIA, HuggingFace | Becomes part of an existing platform's data quality story. |
| Certificate | Claru, Cortex AI, Traceplane | Direct upsell — they need this and can't easily build it themselves. |
| Eval company | Standalone — no clean exit, but big TAM | Every robotics startup needs third-party evals before customer deployments. |

### Why now

Three concurrent forces:

1. **VLA policies are getting good enough to deploy.** Physical Intelligence's π0.7 (April 2026), OpenVLA, Octo — these are commercially relevant. Customers buying robots want some assurance they work. Third-party evaluation becomes a real market.
2. **Data infrastructure is hot capital.** $27.6B into robotics in 2025. The category "robotics data tooling" has been validated by Foxglove's Series B. A new tool in an adjacent slot of the stack can ride that.
3. **The reproducibility problem is finally biting commercially.** Traceplane's audit showed every major open-source dataset has known bugs. As labs increasingly fine-tune from these datasets, "where did my drift come from" becomes a problem with real economic cost — not just a research nuisance.

## 5. Relevant research / prior art on the *technical* side

Worth knowing what's been done that Anvil's implementation builds on or competes with.

**Data quality scoring (already used by Forge):**
- Belkhale, Cui, Sadigh — *Data Quality in Imitation Learning* (2023). Foundational paper. Argues data curation matters more than model design in robotics IL.
- DemInf (Belkhale et al., 2025) — mutual-information-based demo quality scoring. Used by Forge's `quality` module.
- Re-Mix (Hejna et al., 2024) — data curation across the Open X-Embodiment dataset, 38% improvement over uniform weights. Strong empirical signal that data weighting matters at scale.

**Scene change / visual consistency:**
- Lin et al. — *Robust Scene Change Detection Using Visual Foundation Models and Cross-Attention Mechanisms* (2024). Uses frozen DINOv2 + cross-attention to handle viewpoint and lighting variation. Anvil's Layer 2 borrows directly from this.
- DINOv3 (Meta, Aug 2025) — successor to DINOv2 with substantially better dense features. Anvil's recommended backbone.
- SAM 3 / SAM 3.1 (Meta, Nov 2025 / Mar 2026) — text-promptable concept segmentation. Replaces the SAM 2 + manual prompting workflow Anvil would otherwise need.

**Real-world robot evaluation:**
- *A Survey on Reproducibility by Evaluating Deep RL Algorithms on Real-World Robots* (Lynnerup et al., CoRL 2020). The canonical reproducibility-crisis paper. Worth citing whenever explaining why Anvil exists.
- *RoboArena: Distributed Real-World Evaluation of Generalist Robot Policies* (June 2025). Crowd-sourced pairwise eval on DROID. Decentralization-as-strategy.
- *RobotArena ∞* (Oct 2025). Real-to-sim translation for scalable benchmarking. Sim eval framing.
- *SIMPLER / SimplerEnv* (CoRL 2024). Established that sim-eval can rank policies in ways that correlate with real performance.
- TOTO benchmark (CMU). Physical benchmark for in-the-wild robotic offline learning.

**Robotics data audit:**
- Traceplane's "We Audited 10 Popular Open-Source Robot Datasets" (April 2026). Excellent prior-art article. Best source for talking about why this market exists. **Worth reading in full** before any commercial pitch.

## 6. What this means for the project trajectory

Concrete recommendations.

**Near-term (build phase, weeks 1-3):** Don't change the MVP. The 3-week scope is right. The whole strategic thesis depends on Anvil existing first and being usable — without that, it's a pitch deck, not a tool.

**At v0.1 release:** Position the launch narrative around reproducibility, not "scene consistency." Reproducibility is the word that resonates with paper writers, eval companies, and grant committees. Scene consistency sounds like a feature; reproducibility sounds like a category.

**Specific actions for the launch post:**
- Lead with the Traceplane finding ("every major dataset has known quality bugs") to establish the problem.
- Cite RoboArena and SIMPLER as evidence that the field is converging on this need.
- Position Anvil + Forge as "the open-source robotics data quality stack" — same author, complementary tools, shared schema.
- Quietly note that the schema is designed to support an audit-trail / certificate flow, without making it the headline.

**Within 30 days of launch:** Reach out to Traceplane, Claru, Cortex AI, Foxglove. Not for partnerships — for *conversations*. Each of them has either built or wanted exactly the post-collection version of this. Find out what they'd pay for. That's the v1.0 roadmap.

**Within 90 days:** A "powered by Anvil" badge in a Claru or Traceplane workflow is the v1.0 milestone that proves the category. Or, alternatively, a Foxglove blog post citing Anvil as the pre-collection complement to their post-collection platform. Either one validates the strategic positioning.

**Within 6 months, if signal is strong:** Decide between (a) continuing as a beloved OSS tool with a small consultancy around it, (b) building Play 2 (certification service — small team, focused product, sellable to existing eval companies), or (c) raising for Play 3 (full eval-as-a-service company — bigger team, more capital, more risk). The data from months 3-6 tells you which.

## 7. Open questions worth sitting with

Honest unknowns that the data doesn't resolve:

1. **Will the eval-as-a-service category consolidate or fragment?** If Cortex AI absorbs Claru, that's 1-2 buyers, not 5. Concentration risk for Play 2.
2. **Does NVIDIA care?** Their Physical AI Data Factory plus Isaac Lab-Arena suggests they want the entire data and eval stack. Anvil could be a feature they bundle, not a company they buy.
3. **Will Foxglove move upstream?** Their April 2026 "Data Search and Curation" launch is one step away. A pre-collection feature is 6-12 months further. Race condition.
4. **Does the schema travel?** Anvil's value depends on the metadata contract being adopted. If LeRobot v4 ships its own incompatible schema, Anvil has to re-spec. Worth tracking the HF roadmap.

None of these are reasons not to build. They're reasons to ship fast.

---

## Appendix: links worth keeping handy

- Traceplane audit blog: https://traceplane.ai/blog/we-audited-10-robotics-datasets
- Foxglove product: https://foxglove.dev/product
- Bessemer 2026 robotics thesis: https://www.bvp.com/atlas/bessemer-predicts-robotics-and-physical-ai
- RoboArena paper: https://arxiv.org/abs/2506.18123
- SimplerEnv: https://simpler-env.github.io/
- Data Quality in Imitation Learning (Belkhale): https://arxiv.org/abs/2306.02437
- Lin et al. scene change detection: https://arxiv.org/pdf/2409.16850
- Claru: https://claru.ai/
- Cortex AI: via Y Combinator company directory
- NVIDIA Physical AI Data Factory Blueprint: https://nvidianews.nvidia.com/news/nvidia-announces-open-physical-ai-data-factory-blueprint
- Voxel51 mobility workshop: https://voxel51.com/events/designing-data-infrastructures-for-multimodal-mobility-datasets-january-13-2026
