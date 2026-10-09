# Working on ORCHESTRA-WM

This is synthetic research software, never a real vehicle or infrastructure controller. Work in the existing checkout; cloud tasks are already isolated. Do not create another worktree unless explicitly requested. Preserve experiment integrity: never invent results, cherry-pick seeds, change metrics to pass a claim, or use simulator truth inside primary-model planning.

## Layout and architecture

- `sim/`: directed route-lanes, vehicles, jerk-limited dynamics, yielding policy, sensing, renderer.
- `envs/traffic_env.py`: Gymnasium simulator, clone, reset, step, diagnostic hidden state.
- `data/`: balanced policy/scenario collection, stable IDs, masks, episode-split batches, diagnostics.
- `models/world_model.py`: agent/lane/action embeddings, transformer scene encoder, per-entity GRU memory, action-conditioned transformer rollout, state/lane/cost heads. `training.py` owns autonomous rollout loss and held-out validation. `baselines.py` supplies trivial forecasts.
- `planning/`: generic categorical CEM, model-only `Orchestrator`, explicitly separate simulator oracle.
- `evaluation/`: paired prediction, action audit/shuffle, nonlinear interactions, occlusion, counterfactual/rank fidelity, feedback planning, sensing/OOD/scaling, reporting, figures and demo.
- `pipeline.py`: restartable stage orchestration and artifact validation.
- `configs/`: smoke, small, full, partial observation and OOD overlays.
- `scripts/`: executable entry points. `tests/`: behavioral simulator/model/planner/data contracts.

## Commands

Run from the repository root:

```sh
uv sync --frozen
uv run pytest -q
uv run python scripts/run_pipeline.py --config configs/smoke.yaml
uv run python scripts/run_pipeline.py --config configs/smoke.yaml --clean
uv run python scripts/collect_data.py --config configs/smoke.yaml
uv run python scripts/train_world_model.py --config configs/smoke.yaml --all
uv run python scripts/evaluate_prediction.py --config configs/smoke.yaml
uv run python scripts/evaluate_counterfactuals.py --config configs/smoke.yaml
uv run python scripts/evaluate_planning.py --config configs/smoke.yaml
uv run python scripts/make_demo.py --config configs/smoke.yaml
```

The evaluation_prediction entry point currently runs the full evaluation bundle so all baselines share contexts. Individual evaluation functions are available in their named modules. `small` requires a real CUDA/MPS device. Never automatically launch full. CPU smoke is the default. Linux uv resolves a CPU PyTorch wheel for practical cloud installation; a CUDA host must install its matching official PyTorch CUDA wheel into the environment before GPU training. Device selection itself supports CUDA, MPS and CPU. On a CUDA host, install with `uv pip install --torch-backend=auto --reinstall torch`, then use `uv run --no-sync` so the CPU lock does not replace the CUDA build. That installation path was not executed on this CPU-only machine.

## Simulation and extensions

A Lane is a complete directed polyline route through one or more conflict zones. Vehicles progress longitudinally along segments, then depart; optional respawn assigns a new generation ID. There are no lane changes, turning decisions, or real steering controls. Background vehicles use speed preference, following gaps and intersection yielding; connected vehicles receive {-1,0,+1} speed advisories with the same smooth low-level integrator. Commands on background slots are ignored. Sensors use a separate RNG, so sensor ablations preserve physical trajectories.

To add a road topology, extend `make_road`, supplying polylines and conflict nodes, then add deterministic path/pose and interaction tests. Public geometry is available to lane tokens. To add an agent type, extend Vehicle metadata and the driver policy, then define which fields are observable and which controls are allowed. To add an action, update the Gym space, advisory mapping, collection policies, CEM categories, action tokenizer, audit and action-mask tests together. Rebuild datasets/checkpoints when semantics change.

Default stable IDs map directly to entity slots. Padding is separate from visibility: `valid=False` removes a padding entity from attention, while invisible real entities retain memory. Missing detections never contain true state. `track_loss` is temporary detection loss with stable identity on reacquisition; anonymous association is not implemented.

## Training and planning invariants

`observe` alone may consume real observations. `imagine_step` returns `(next_state, prediction_heads)`; `imagine` loops without observations. Hidden simulator states provide supervised targets and diagnostic privileged inputs only. The ORCHESTRA/independent/no-action/memoryless primary pathways use partial observations. Future observations in the consistency loss are detached targets, never rollout inputs. The independent model has no cross-agent state-prediction attention; the no-action model zeros advisory input.

The learned planner holds a model and config, never a TrafficEnv. It scores predicted cost components, chooses a joint action sequence, and executes only the first action externally. Oracle clone queries belong only in diagnostic code. Keep simulator-query exclusion tests intact.

## Outputs and reproducibility

Outputs default to `outputs/smoke/`. CSVs are individual paired measurements; seed-summary CSVs contain mean, SD, SE and count. `metrics.json`, `EXPERIMENT_SUMMARY.md`, and `docs/CLAIM_LEDGER.md` derive from actual CSVs. `demo.html` is self-contained; `demo.gif` is portable. Checkpoints/data/TensorBoard logs are retained locally but git-ignored; reproduce them with the pipeline. Source/config fingerprints invalidate stale stages. `--clean` archives old outputs under the OS temporary directory instead of discarding them.

Three evaluation simulator seeds do not mean three independently trained models. Label smoke claims accordingly. See `docs/EVALUATION_PROTOCOL.md` before modifying an experiment, and `DECISIONS.md` for tradeoffs and failed approaches.

## Phase 2

`uv run python scripts/run_phase2.py --config configs/phase2.yaml` executes the frozen campaign, reusing matching completed stages. Read `docs/PHASE2_PLAN.md` and `docs/PHASE2_EXECUTION_NOTES.md` before changing Phase-2 code. C is primary; D is diagnostic. Never change gates based on outcomes. Three independently trained seeds (101/202/303), family-disjoint siblings, and seed-level confidence intervals are mandatory. Phase-1 hashes and documentation prefixes are verified before/after reporting. Phase-2 arrays and checkpoints remain ignored; CSVs, manifests, figures, metadata and reports are versioned. Cached evaluation semantics are fingerprinted and mismatches fail closed. OOD is forbidden when INT-H fails.

## Phase 3 — passive I-24 extension

Keep all new code under `orchestra_wm/i24`, configs `i24_*`, tests `tests/i24`, outputs `outputs/i24`. Preserve Phase-1/2 findings and hashes; do not reuse coordination controls/weights for passive forecasting. Read `docs/I24_EVALUATION_PROTOCOL.md` before scientific comparisons. `uv run python scripts/run_i24_pipeline.py --config configs/i24_smoke.yaml` trains only a prominently labeled synthetic fixture. Official access is blocked until authorized JSON arrays and acquisition metadata arrive; never substitute one-vehicle CAN/GPS or I24-3D. Real pilot: `--config configs/i24_pilot.yaml --data-root /path/to/I24MOTION_PUBLIC`. Research is explicitly gated. Use two CPU threads, zero loader workers; no automatic GPU selection. Data/source/split fingerprints and atomic checkpoints govern resumption. Genuine raw files, canonical partitions, caches and checkpoints stay ignored. The compressed browser payload contains selected exact evaluation arrays and provenance; never replace them with replay. Updating the fixture does not establish real-data science or cross-corridor validation.

<!-- I24_MSD -->

## Harvard I24-MSD campaign

Work on `phase3-i24-msd`; retain the existing verified ZIP and three extracted shards under ignored `data/i24_msd/raw`. New code/results are under `orchestra_wm/i24_msd` and `outputs/i24_msd`. Read docs/I24_MSD_DATA_CARD.md, I24_MSD_EVALUATION_PROTOCOL.md and I24_MSD_IMPLEMENTATION.md. The source cutoff is 1 s in 9 s records; never reuse continuous-I24 10/20 s evaluation. The existing pilot is REAL, 57/19/19 supplied-track-grouped scenes, three 500-update seed-101 models; uncertainty/generalization remain limited. Use the exact run/resume command in README; cached bytes and fingerprints are verified, never silently replaced. Do not reinterpret the uncertain energy-score result as interaction superiority. `preserve()` protects 381 earlier tracked files, allowing only appended notes in README/AGENTS/DECISIONS/.gitignore. Full tests include actual records when present and explicitly skip real-data tests on an unmounted checkout. The research config remains gated. Do not automatically run larger or unrelated campaigns, and do not publish raw data, temporary transfers, full checkpoints or signed download URLs.
