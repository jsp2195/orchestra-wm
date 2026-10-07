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
