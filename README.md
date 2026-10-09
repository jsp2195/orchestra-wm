# ORCHESTRA-WM

ORCHESTRA-WM is a synthetic research testbed for studying whether action-conditioned multi-agent world models can support counterfactual reasoning and centralized coordination in partially observed traffic systems.

**ORCHESTRA-WM is a synthetic research testbed. It is not intended or validated for control of real vehicles or traffic infrastructure.**

The experiment asks whether one learned belief of a joint traffic scene can imagine the consequences of several coordinated interventions, rank those futures, and choose useful actions. Trajectory accuracy, counterfactual fidelity and decision fidelity are measured separately.

## Run it

Python 3.11+ and [uv](https://docs.astral.sh/uv/) are required. No paid APIs, credentials or external simulation service are used. From this checkout:

```sh
uv sync --frozen
uv run python scripts/run_pipeline.py --config configs/smoke.yaml
```

This runs tests, the simulator action audit, balanced trajectory collection, all learned baselines and ORCHESTRA training, paired evaluations, plots, a self-contained animated HTML demo, a GIF, and measured reports. CPU smoke uses 64 episodes, ten vehicles, four connected AVs, and a 1M-parameter model. It is an integration experiment, not conclusive scientific evidence.

```sh
# Rebuild from clean output, preserving the old directory under /tmp
uv run python scripts/run_pipeline.py --config configs/smoke.yaml --clean
# Test contracts only
uv run pytest -q
# GPU-scale configuration (refuses to run without a GPU)
uv run python scripts/run_pipeline.py --config configs/small.yaml
```

The pipeline resumes completed stages only when its source/config fingerprint matches and required artifacts exist. Data/checkpoints/TensorBoard logs are git-ignored; a fresh clone retrains them. Small is configured for 2,048 episodes and a larger model. Full specifies 20,000 episodes but is deliberately blocked by the automatic pipeline. Neither was executed on the CPU-only development machine. Linux defaults to an official CPU PyTorch wheel; on a CUDA host install the matching official CUDA-enabled PyTorch wheel before GPU training. Runtime device selection supports CUDA, MPS and CPU. For a CUDA host, use `uv pip install --torch-backend=auto --reinstall torch` after syncing, then run `uv run --no-sync python scripts/run_pipeline.py --config configs/small.yaml`; `--no-sync` prevents the CPU lockfile from replacing the CUDA wheel. This CUDA installation path is documented but untested on this machine.

## Inspect the experiment

- [Animated HTML demo](outputs/smoke/demo.html) — open the downloaded file in a browser; no server needed.
- [Animated GIF](outputs/smoke/demo.gif) — four hypothetical joint plans from the same belief.
- [Experiment summary](outputs/EXPERIMENT_SUMMARY.md) — actual checks, failures, limitations and hardware.
- [Decision-sufficiency comparison](outputs/smoke/DECISION_SUFFICIENCY.md).
- [Claim ledger](docs/CLAIM_LEDGER.md), [evaluation protocol](docs/EVALUATION_PROTOCOL.md), [design decisions](DECISIONS.md).
- Raw measurements: `outputs/smoke/*.csv`; aggregate metrics: `outputs/smoke/metrics.json`; figures: `outputs/smoke/figures/`.

<!-- MEASURED_RESULTS -->
| Model | Rollout error h10 (m) | Action gap (m) | Counterfactual Pearson | Plan-rank Spearman | MPC true objective |
| --- | --- | --- | --- | --- | --- |
| persistence | 17.6138 | 0.0000 | undefined | undefined | 10.9442 |
| constant_velocity | 8.9479 | 0.0000 | undefined | undefined | 10.9442 |
| independent | 7.3548 | -0.0009 | 0.4157 | 0.5938 | 4.8759 |
| no_actions | 7.3781 | 0.0000 | undefined | undefined | 10.9442 |
| orchestra | 7.1687 | 1.2444 | 0.6155 | 0.5960 | 4.8759 |
| privileged | 8.5781 | -0.4223 | 0.7373 | 0.5860 | 4.8759 |
| oracle | 0 (simulator reference) | N/A | 1.0000 | 1.0000 | 4.4432 |

CPU smoke: one training seed and three paired evaluation seeds. ORCHESTRA has 1012878 parameters, trained for 192 updates. Correct-action gap at ten steps: 1.2444 m. Reappearance error: 8.4288 m intact vs 13.1360 m reset. These are integration-scale measurements; see the claim ledger for failed criteria.
<!-- END_MEASURED_RESULTS -->

## Synthetic world and observations

Four procedural road families are implemented: four-way intersection, merge, short corridor and 2×2 urban grid. A directed lane is a polyline route with public geometry and conflict metadata. Vehicles move along centerlines with bounded acceleration and jerk, car-following, background intersection yielding, collision/conflict detection, departures and optional respawning. This simplified simulator has no photorealistic vision, steering control, lane changes or turning decisions.

Background agents follow simulator policies. Connected agents accept abstract `-1 = yield`, `0 = maintain`, `+1 = proceed` speed advisories. A joint action assigns one advisory per connected vehicle; other slots are masked. Variable vehicle counts are supported and tested through 40. Signals are optional in the design and not implemented in this version.

The primary model receives visible noisy vehicle tracks, AV self-telemetry, delayed lane occupancy/queue measurements and public lane geometry. Detection masks, camera blind spots, missed detections, sensor outage and source withholding create partial observability. Stable track IDs isolate modeling from association. Simulator desired speed, driver parameters and hidden positions are not primary-model inputs. Exact state is available for supervised targets, diagnostics and the explicitly labeled privileged baseline.

## World model

```text
partial agent + lane tokens
          ↓
shared scene transformer
          ↓
per-entity recurrent world memory
          ↓  hypothetical joint action tokens
shared dynamics transformer
          ↓
future agent states + lane states + objective components
```

The compact model is trained from scratch with agent-state, speed, lane occupancy/queue, collision/conflict component and detached latent-consistency losses over autonomous five-step rollouts. Future observations supply targets only. A constant-velocity residual prior stabilizes training; the action-shuffle experiment checks whether learned action responses add value beyond that prior.

```python
state = model.initial_state(batch_size=1, num_agents=10, num_lanes=4)
state = model.observe(state, observation_tokens, observation_mask, previous_action)
next_state, heads = model.imagine_step(state, hypothetical_joint_action)
trajectory = model.imagine(state, hypothetical_action_sequence)
```

The step returns both next learned state and decoded heads. Imagination accepts no observations and cannot access the simulator. Padding masks and observation masks have different meanings: a padded slot never participates; a temporarily invisible real agent retains its recurrent belief. Tests enforce these boundaries.

Baselines include persistence, constant velocity, independent per-agent learned dynamics, an interaction model with action inputs removed, a memoryless model, and privileged-state learned dynamics. All evaluated forecasts use the same simulator trajectories. The privileged model is diagnostic and not the primary world model.

## Counterfactuals and planning

Categorical CEM samples joint advisory sequences, imagines them entirely inside the learned model, scores predicted objective components and refits categorical probabilities to elites. Only the first chosen joint action is executed. New partial observations update the belief before replanning. The orchestrator holds no simulator object; a test patches simulator stepping to fail if it is queried internally.

The synthetic objective combines delay, queue, progress, pairwise collision/conflict counts and acceleration smoothness, with explicit configurable weights. A separate finite-budget oracle uses simulator clones with the same CEM budget. It tests whether the task is amenable to planning; it is not a proof of optimality or safety.

Counterfactual evaluations branch the exact same inferred state and independently clone the corresponding true state. They measure pairwise outcome-distance correlations, cost rank correlations, top-k agreement, selected true rank and regret. Closed-loop comparisons include random, maintain, local reactive control, heuristic coordination, ORCHESTRA MPC, independent learned MPC, no-action MPC, privileged MPC and oracle MPC. This distinguishes shared-model coordination from gains available to a simpler action-conditioned cost predictor.

## Scientific limits

Three paired evaluation simulator seeds support descriptive mean/SD/SE. Only one training initialization is used; this does not establish robustness to optimization randomness. Smoke has short horizons and small planning budgets. Long-horizon errors, weak action conditioning or missing coordination advantages must remain visible in the claim ledger. Correlation alone is not calibrated uncertainty, and a favorable objective can conceal collisions unless its components are examined.

OOD density, behavior, geometry, topology, incident and sensing tests are reported separately. The same checkpoint is used for sensor/source ablations. Active sensing, uncertainty calibration, anonymous tracking and full-scale training remain unimplemented or unexecuted as documented. No real-world claim is made.

See [AGENTS.md](AGENTS.md) for module boundaries, extension guidance and stage commands.

<!-- PHASE2_RESULTS -->
# Phase 2 — Joint Interaction and Coordination

ORCHESTRA remains a functioning multi-agent world model, but the experiment did not establish a measurable centralized coordination advantage over an independent learned-controller baseline.

**CENTRALIZED ORCHESTRATION BENEFIT: NO.** C coordination gain = -0.132126, SD 0.228848, SE 0.132126; Student-t 95% CI [-0.700616, 0.436365]. The unchanged acceptance criterion is lower bound >0.1. Independent MPC objective 7.356933; C objective 7.489058; oracle objective 4.899767 (lower is better). The oracle is finite-budget CEM, not a proof of global optimality.

## Definitive comparison

| Model | Interaction Error (m) | Joint CF Corr. | Joint Shuffle Gap (m) | Plan Rank Spearman | MPC Objective | Coordination Gain |
|---|---|---|---|---|---|---|
| A: Original data | 11.0024 | 0.5546 | 3.2748 | 0.5027 | 7.4735 | -0.1166 |
| B: Factorial data | 6.2602 | 0.5711 | 6.9734 | 0.5061 | 7.4228 | -0.0659 |
| C: Factorial + CF | 6.0671 | 0.5738 | 7.1894 | 0.4822 | 7.4891 | -0.1321 |
| Independent | 9.7012 | 0.5936 | 4.2250 | 0.5484 | 7.3569 | 0.0000 |
| Privileged | 6.6720 | 0.5804 | 6.6865 | 0.4820 | 7.3569 | 0.0000 |
| D: Pairwise + CF | 6.4552 | 0.5790 | 6.8590 | 0.4885 | 7.3569 | 0.0000 |
| Oracle | 0.0000 | 1.0000 | N/A | 1.0000 | 4.8998 | 2.4572 |

Interaction Error is final-step-10 Euclidean position error averaged over the two interacting AVs, nine siblings, 16 held-out families and three training seeds. Joint CF Corr. is the mean within-family nine-plan objective Pearson. Joint shuffle gap is absolute pair-error degradation in metres; INT-E instead uses each seed's ratio of mean errors. Oracle forecast/rank entries are self-comparison references, not learned predictions; its shuffle gap is undefined. A is the original-data Phase-1 architecture retrained for the matched Phase-2 budget; the frozen Phase-1 checkpoint appears only in audits. D and privileged are secondary diagnostics; neither can replace C.

## Preregistered gates

| Gate | Status | Measurements |
|---|---|---|
| INT-A | PASS | {"fraction_above_005": 0.9166666666666666, "interaction_rms": 1.2609545525031485} |
| INT-B | PASS | {"training_families": 24, "training_siblings": 216} |
| INT-C | FAIL | {"ci_high": 14.069780035204575, "ci_low": -6.801566742689738, "mean": 3.6341066462574187, "n_seeds": 3, "sd": 4.200923861662304, "se": 2.4254045223758536} |
| INT-D | PASS | {"objective_pearson": 0.573818161385601, "outcome_distance_pearson": 0.9894158588023917} |
| INT-E | PASS | {"ci_high": 1.2835438915224346, "ci_low": 1.087524938967835, "mean": 1.1855344152451348, "n_seeds": 3, "sd": 0.03945412358328637, "se": 0.02277884887145115} |
| INT-F | FAIL | {"regret": 1.4804249196965251, "selected_percentile": 18.01470588235294, "spearman": 0.4821551427588579, "top10_overlap": 0.625} |
| INT-G | FAIL | {"command_frequencies": {"-1": 0.0, "0": 0.006944444444444444, "1": 0.9930555555555556}, "joint_strategies": 4} |
| INT-H | FAIL | {"bootstrap_high": 0.0, "bootstrap_low": -0.5276421015057714, "bootstrap_replicates": 5000, "ci_high": 0.4363647136906334, "ci_low": -0.7006157563509592, "mean": -0.13212552133016292, "model": "C_difference", "n_seeds": 3, "sd": 0.22884811592036763, "se": 0.13212552133016295, "task": "all"} |

## Paired primary result by task

| task | mean | sd | se | ci_low | ci_high | n_seeds | bootstrap_low | bootstrap_high | bootstrap_replicates |
|---|---|---|---|---|---|---|---|---|---|
| all | -0.1321 | 0.2288 | 0.1321 | -0.7006 | 0.4364 | 3 | -0.5276 | 0.0000 | 5000 |
| intersection | 0.0005 | 0.0009 | 0.0005 | -0.0018 | 0.0029 | 3 | 0.0000 | 0.0022 | 5000 |
| merge | -0.2648 | 0.4586 | 0.2648 | -1.4041 | 0.8745 | 3 | -1.0389 | 0.0000 | 5000 |

Confidence intervals use three training-seed means (df=2), not 48 independent trained models. The secondary hierarchical bootstrap resamples training seeds and matched episode pairs. No seed or checkpoint was selected after evaluation. All models receive 240 updates, batch 18, ten-step autonomous rollouts; factorial models use 75% sibling batches and 25% natural batches. A uses original data only. Training curves remain noisy, and this fixed integration-scale budget does not establish convergence.

## Audits and data

Phase-1 action frequency was 100% proceed in both audited tasks, including the recorded conflict/relevance strata. Numeric reproduction differences are zero for all three audited tables. 71 original artifact hashes remain unchanged; README and claim-ledger Phase-1 prefixes are checked separately from their appended Phase-2 sections. Simulator interaction RMS is 1.260955; fraction above 0.05 is 0.916667. Each of 24 training and six validation families has nine siblings, giving 270 episodes / 3240 future transitions. All 16 test families are disjoint. Manifests record seeds, state hashes, config, and compressed data hash. Hidden states and sibling metadata remain targets or diagnostics, not primary model inputs.

D was triggered before Phase-2 training: frozen model mean cross-agent influence was 0.718145 m/action on interacting targets versus 0.724232 on distant targets. This audit did not demonstrate spatial specificity; it is not a precise estimate of structural impossibility. The operational direction rule is recorded in PHASE2_EXECUTION_NOTES.md. D adds a small relative-geometry action-message channel.

The independent baseline has separate per-agent dynamics and an additive cost head. Both learned controllers optimize joint candidate tensors, but the independent model cannot represent cross-agent effects. CEM uses identical balanced 18-plan initial pools, two iterations, horizons, weights and proposal randomness. Learned candidate scores use only imagination; real simulator clones belong solely to oracle and post-hoc evaluation. No action-label penalties were added.

## Scientific interpretation

Trajectory fidelity, counterfactual fidelity, and decision fidelity are distinct. Positive counterfactual correlation can reflect correct marginal action effects while missing nonlinear interaction residuals. Joint-pairing shuffle also harms an independent action-conditioned predictor and alone does not prove joint reasoning. The interaction-residual errors, rank results and closed-loop paired costs are therefore reported separately. The preregistered C result alone adjudicates centralized benefit. C's objective interaction-residual RMSE is 1.243167, versus 1.225667 for the structurally additive independent model. C predicts a mean objective interaction RMS of 0.052795 against true 1.225667, and its residual Pearson is -0.280523 (secondary diagnostics). The independent residual RMS of about 6e−8 is floating-point roundoff, so its residual correlation is not scientifically interpretable. The centered sibling difference loss can primarily fit marginal effects without recovering the double-centered non-additive residual. This is the specific failure targeted by the next experiment.

OOD is UNTESTED: INT-H failed, so no broad generalization experiments were run. Hidden-conflict memory is a secondary executed comparison in memory_coordination.csv; it cannot override INT-H. The four additional cases per seed give the following means (synthetic conflicts are cumulative simulator event counts):

| memory | objective | conflicts |
|---|---|---|
| intact | 4.7804 | 4.7500 |
| reset | 6.4391 | 6.8333 |
 No real-world safety or infrastructure-control claim is made.

## Reproduction

From the repository root, with preserved Phase-1 local dataset/checkpoints available:

```sh
uv sync --frozen
uv run pytest -q
uv run python scripts/run_phase2.py --config configs/phase2.yaml
```

A fresh clone needs the original ignored Phase-1 dataset/checkpoints and TensorBoard files restored from the original workspace/archive before exact preservation verification can pass. The preservation manifest includes timestamped logs that cannot be reconstructed byte-for-byte by retraining. Do not run the Phase-1 pipeline over the preserved outputs or replace the manifest to bypass a mismatch. This is an archival portability limitation of the existing WIP manifest; the current workspace contains and verifies all originals. Checkpoints and dataset arrays are intentionally not in Git. Phase-2 stages reuse matching checkpoints and immutable dataset manifests. All 18 trained models, optimizer steps, parameter counts, hardware versions, stage timing and metrics are recorded in training_metadata.json / metrics.json. Aggregate CSVs retain every seed; no best-seed selection is supported.

## Figures

- [01_phase1_action_choices](outputs/phase2/figures/01_phase1_action_choices.png)
- [02_true_factorial](outputs/phase2/figures/02_true_factorial.png)
- [03_factorial_coverage](outputs/phase2/figures/03_factorial_coverage.png)
- [04_cross_agent_sensitivity](outputs/phase2/figures/04_cross_agent_sensitivity.png)
- [05_interaction_error](outputs/phase2/figures/05_interaction_error.png)
- [06_true_predicted_matrices](outputs/phase2/figures/06_true_predicted_matrices.png)
- [07_joint_shuffle](outputs/phase2/figures/07_joint_shuffle.png)
- [08_counterfactual_correlation](outputs/phase2/figures/08_counterfactual_correlation.png)
- [09_plan_rank_scatter](outputs/phase2/figures/09_plan_rank_scatter.png)
- [10_phase2_action_choices](outputs/phase2/figures/10_phase2_action_choices.png)
- [11_centralized_independent_objective](outputs/phase2/figures/11_centralized_independent_objective.png)
- [12_coordination_gain_ci](outputs/phase2/figures/12_coordination_gain_ci.png)
- [13_oracle_gap](outputs/phase2/figures/13_oracle_gap.png)
- [14_hidden_conflict_memory](outputs/phase2/figures/14_hidden_conflict_memory.png)
- [15_three_seed_summary](outputs/phase2/figures/15_three_seed_summary.png)
- [16_training_curves](outputs/phase2/figures/16_training_curves.png)

NEXT EXPERIMENT: Keep the same factorial families, architecture, and update budget; replace the centered sibling loss with a loss on the double-centered interaction residual J, and test whether reduced objective-residual error improves plan ranking and the preregistered coordination gain.

<!-- PHASE3_I24 -->

## Phase 3 — ORCHESTRA-I24

A separately namespaced passive, stochastic highway world-model extension. Official I-24 MOTION data access is currently **BLOCKED**; only a conspicuously labeled synthetic fixture has been trained and evaluated. Phase 2 remains a negative centralized-coordination result. No causal AV action, real safety, real wave skill or Cavnue compatibility claim is made.

```sh
uv run python scripts/run_i24_pipeline.py --config configs/i24_smoke.yaml
uv run python scripts/run_i24_pipeline.py --config configs/i24_pilot.yaml --data-root /path/to/I24MOTION_PUBLIC
```

[Fixture demo](outputs/i24/demo/index.html) · [Measured report](outputs/i24/EXPERIMENT_SUMMARY.md) · [Data acquisition](docs/I24_DATA_ACQUISITION.md) · [Evaluation protocol](docs/I24_EVALUATION_PROTOCOL.md) · [Model card](docs/I24_MODEL_CARD.md) · [Claim ledger](docs/I24_CLAIM_LEDGER.md). The research config is gated and never launched automatically. The demo must be downloaded/opened in a browser; GitHub does not execute HTML previews.

<!-- I24_MSD -->

## ORCHESTRA-I24-MSD — real short-horizon traffic pilot

**Real Harvard I24-MSD data has now been acquired and used for training.** This is separate from the continuous-I24 adapter and its synthetic fixture. One checksum-verified 1,369,499,074-byte ZIP supplied 349 inspected records, including 182 eligible multivehicle scenes; the pilot uses 57/19/19 grouped train/validation/test scenes. Source-defined context is 1 second with 8 seconds of recorded future. We evaluate 1/2/4/6-second autonomous forecasts.

Three compact models (84,514 parameters each) trained for 500 CPU updates, seed 101. Four-second FDE: ORCHESTRA **2.272 m**, constant velocity **2.219 m**, independent learned **2.246 m**. The primary joint energy-score reduction versus independent is **0.0095 m**, provisional 95% interval **[−0.1111, 0.0813]**. This pilot **does not establish an interaction-model advantage**. Nominal 90% position coverage is only **54.6%**. One source date and one training seed do not establish generalization or convergence.

[Real interactive demo](outputs/i24_msd/demo/index.html) · [Measured report](outputs/i24_msd/EXPERIMENT_SUMMARY.md) · [Data card](docs/I24_MSD_DATA_CARD.md) · [Protocol](docs/I24_MSD_EVALUATION_PROTOCOL.md) · [Claim ledger](docs/I24_MSD_CLAIM_LEDGER.md) · [Run/resume details](docs/I24_MSD_IMPLEMENTATION.md).

```sh
uv sync --frozen
uv run python scripts/run_i24_msd_pipeline.py --config configs/i24_msd_pilot.yaml --data-root data/i24_msd/raw
uv run pytest -q
```

The command reuses verified raw files, extracted shards and matching checkpoints. Raw data/checkpoints/full inference caches stay ignored; the standalone demo embeds exact checkpoint-generated evaluation arrays and hashes. It shows recorded truth separately. Download the HTML and open it in a modern browser; GitHub does not execute HTML previews.

I24-MSD uses Scenario protocol buffers, not continuous I24 JSON. The original stochastic recurrent interaction architecture is reused, with population macro feedback disabled because curated vehicle subsets are not traffic censuses. No long-horizon waves, learned inflow/outflow, causal AV intervention or Cavnue hardware validation is claimed. Prior Phase-2 negative coordination results and continuous-I24 fixture claims remain unchanged.
