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

## Failed or incomplete scientific approaches

Short residual training may rely heavily on constant-velocity priors and may learn action-dependent cost rankings without accurate action-dependent trajectories. Preliminary runs exhibited that pattern. No reward/metric tuning was performed to make those results pass. The final measured direction is recorded in `outputs/EXPERIMENT_SUMMARY.md`; do not substitute preliminary values.

Current-observation trivial baselines and the reset-memory ablation are deliberately limited. The stronger separately trained memoryless model and independent learned MPC baseline are also reported. Improvement over local reactive control alone cannot isolate the value of centralized interaction modeling. Short planning episodes limit throughput evidence. These are Phase-1 research limitations, not production readiness claims.
