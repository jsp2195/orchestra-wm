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
