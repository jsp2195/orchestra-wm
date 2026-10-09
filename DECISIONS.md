# Design and experiment decisions

| Decision | Alternatives | Rationale | Consequences |
|---|---|---|---|
| Self-contained structured simulator | SUMO, raw video, photorealistic simulation | Keep compute focused on world modeling and joint interventions | Synthetic rules and thresholds are not traffic realism or safety validation |
| Complete directed route-lanes | Fine-grained lane segments with turn routing | Preserve the initial working simulator and deterministic motion | Route progression follows polyline segments; no lane changes or route decisions; successor lists are reserved |
| Conservative lane conflict metadata plus geometric conflict nodes | Detailed swept-path graph | Compact procedural scenarios | Some declared lane conflicts are conservative; runtime yielding uses actual proximity |
| Stable entity slots | Hungarian matching / learned association | Isolate world modeling from identity errors | Temporary missed detections supported, anonymous track loss deferred |
| Structured observations plus public lane geometry | Camera backbone | Reduces MVP compute and removes visual-pretraining dependence | Hidden simulator attributes are supervised labels only, except explicit privileged baseline |
| Recurrent per-agent latent tokens and shared transformers | Full RSSM or large video transformer | Compact, variable-agent model with persistent memory | Approximately 1M smoke parameters; no probabilistic belief calibration |
| Bounded residual around constant velocity | Fully unconstrained coordinate decoding | Stabilize short training and enforce finite rollouts | Good one-step predictions can come from the prior; action-shuffle test is essential |
| Cost heads trained against simulator components | Fully geometric analytic costs from forecasts | Learn delay/conflict response while exposing interpretable components | Planner can exploit cost-head errors; independent true-cost rank/regret evaluation is mandatory |
| Single smoke initialization, three held-out simulator seeds | Three full training runs or five seeds | End-to-end CPU integration budget | Scientific conclusions remain provisional; SD/SE do not cover training variation |
| Six-step categorical CEM | Continuous CEM, exhaustive search | Actions are categorical and jointly optimized | Finite-budget oracle is a diagnostic reference, not guaranteed global optimum |
| Fixed objective, fixed primary horizon | Tune weights/horizon after results | Preserve scientific integrity | Failed criteria remain failures rather than being redefined |
| Explicit CPU uv wheel on Linux | Install default multi-GB CUDA runtime | Current machine has no GPU; portable lightweight smoke | CUDA machines need matching official PyTorch build; MPS/CPU auto-selection is implemented |
| Deferred uncertainty and active sensing | Add unvalidated variance head now | User requested these only after deterministic planning works | No calibration, coverage or active-query claim |

## Integration problems found and corrected

1. Initial collection used `episode % 2` for scenario and `episode % 6` for policy. This confounded policies with road families. The final collector cycles policies across scenario blocks; a test verifies every family receives all six behaviors. Preliminary measurements were archived, and all final training/evaluation was rerun from clean outputs.
2. Initial lane-token padding could become nonzero after the first imagined step, allowing padded lanes to influence later dynamics. Padded lane tokens now stay zero, entity padding is masked, and a padding-invariance test protects real predictions.
3. The first trainer lacked a validation split and timings. Final eight episodes are held out, validation batches are fixed across epochs, and both train/validation losses, steps, parameter counts and wall time are recorded. No evaluation seed is used for training or model selection.
4. Large intersection populations initially clamped multiple same-lane starts to position zero. Spacing/jitter now adapt to count; tests verify non-overlapping same-lane starts through 40 agents on the intersection topology. Dense merge layouts remain a stress test, not realistic demand modeling.
5. The cloud machine has a read-only home directory. Matplotlib/fontconfig and uv caches now use writable temporary/project cache directories; no TLS or package integrity checks were disabled.

6. The independent dynamics baseline initially pooled latent tokens before a nonlinear cost head. Although its state predictions were independent, its costs could express interactions. Final independent costs are an additive average of per-agent heads; an explicit factorial-additivity test enforces the null hypothesis. Action-shuffle permutations are also identical across compared models.

## Failed or incomplete scientific approaches

Short residual training may rely heavily on constant-velocity priors and may learn action-dependent cost rankings without accurate action-dependent trajectories. Preliminary runs exhibited that pattern. No reward/metric tuning was performed to make those results pass. The final measured direction is recorded in `outputs/EXPERIMENT_SUMMARY.md`; do not substitute preliminary values.

Current-observation trivial baselines and the reset-memory ablation are deliberately limited. The stronger separately trained memoryless model and independent learned MPC baseline are also reported. Improvement over local reactive control alone cannot isolate the value of centralized interaction modeling. Short planning episodes limit throughput evidence. These are Phase-1 research limitations, not production readiness claims.

## Phase 2 continuation

- Preserve the preregistered plan/config and original Phase-1 artifact hashes. Repair the reproduction comparator by semantic row keys; all numerical differences were exactly zero.
- Add the pre-authorized D channel only because the frozen Phase-1 spatial-specificity audit failed its pre-training directional test. Treat the small sensitivity difference cautiously, and retain C as primary.
- Use a balanced initial CEM pool with all nine factorial choices and nine distinct temporal plans. Replace the duplicate all-maintain temporal plan with a staggered exchange. Apply the same candidates, objective, random proposals and budgets to oracle and learned controllers; do not reward command diversity.
- Preserve all three final checkpoints per model and the fixed 240-update budget. Resumption checks source/config signatures; final results never select a best seed or checkpoint.
- The independent learned baseline retains the original additive per-agent objective predictor. Its shared CEM machinery optimizes an additive score without a cross-agent predictive pathway. This matches the committed Phase-2 plan rather than silently substituting a different local optimizer.
- Treat trajectory error, factorial structure, and closed-loop coordination separately. A shuffle intervention can hurt even an independent predictor, so it cannot alone establish joint reasoning. Report negative primary findings without moving thresholds.

## Phase 3 — ORCHESTRA-I24

- Create a separate branch and passive research namespace from completed Phase 2. Preserve the negative coordination result and original ledger byte-for-byte; Phase-3 claims have their own ledger.
- Official multivehicle data is inaccessible here: no mounted files, data-site network 403, tutorial requires account credentials. Only official documentation/code were retrieved. Do not count accessible documentation or I24-3D validation samples as data access.
- Implement the inspected v1.0 westbound JSON-array contract, not an invented universal I-24 schema. Use official PDF roadway coordinates/approximate lane bounds; reject unsupported versions/directions. Keep global-geographic calibration unavailable rather than fabricate it.
- Start with a compact graph variational recurrent state-space model and learned macro decoder. Shared latent prior innovations generate coupled futures; the training-only posterior never becomes an imagination input. No action conditioning on passive data.
- Fixed-cohort identity roster comes from history only. Visible-track fields are inputs; full reconstructed fields are targets. Empty observed bins do not certify empty road. Open-road Poisson entry from historical appearances is explicitly an unvalidated inflow assumption.
- Six matched-update fixture variants, proper energy/CRPS scores, memory/graph/sensing controls, and exact demo/evaluation-array checks provide engineering evidence. Fixture results do not satisfy real-data, wave, information-value or Cavnue gates. Small fixed-budget training is not convergence.
- Preregister the 10-second micro-information macro proper-score experiment before held-out comparisons. Distinguish information advantage from model architecture advantage and count independent sessions, not vehicles/windows.
- Stream bounded Parquet conversion with source/acquisition/partition hashes; fail on mismatch. Resumption must not silently mix stale partitions with changed data. Split windows are session-grouped or chronologically buffered.

<!-- I24_MSD -->

## Real I24-MSD pilot decisions

- Resume from Phase-3 commit 1635c9b on the requested `phase3-i24-msd` branch; preserve all prior science. Acquisition/protocol milestone is 3f3e9a3.
- Harvard access initially failed at the proxy. Supported environment domain additions enabled metadata and the official public S3 redirect. Downloaded one smallest ZIP (1,369,499,074 bytes), verified Harvard MD5, and extracted only three of 325 members (26,768,195 bytes), within the bounded disk budget. No authentication bypass or substitute source.
- Actual payloads decode as official Scenario protos, not tf.train.Example. Pin/vendor the Apache-2.0 schema subset and generated Python modules; avoid TensorFlow. Native 91-step 10 Hz records supply a 1 s cutoff; use causal 5 Hz decimation and 1/2/4/6 s forecasts.
- Retain RoadsideScene and the original stochastic graph/GRU architecture. Disable population macro feedback/losses for curated cohorts. Preserve original polylines; use lane medians for neighbor relations and score finite-source-edge violations only where supported. Do not invent lane population counts, confidence probabilities or source splits.
- Group shared supplied identities before deterministic split selection; one date and missing absolute origin times limit leakage/generalization claims. Preserve immutable manifests. Primary comparison and thresholds were written before held-out scoring; no favorable seed or checkpoint selection.
- The first optimizer attempt exposed empty emulated conditioning and stopped after one update. Its checkpoint remains archived locally. Dense fallback for an empty training observation was implemented before completing all fixed-budget models. A later missing-demo-module integration error was resolved by completing reporting; trained checkpoints were reused, not retrained.
- Training completed at 500 updates/model. Full validation optimization loss fell 0.0698→0.0130, but convergence is not established. Four-second ORCHESTRA FDE is slightly worse than CV/independent; the primary energy advantage has an interval crossing zero. Report negative/uncertain results and inadequate 90% calibration plainly.
- Real browser/GIF output must use saved evaluated generated arrays, with truth kept separate. No fabricated macro fields, road geometry or scientific claims. Cavnue remains an interface pending authorized live schema/calibration/coverage data.
