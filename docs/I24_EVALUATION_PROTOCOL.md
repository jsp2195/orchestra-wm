# ORCHESTRA-I24 preregistered evaluation protocol v1

Written before held-out Phase-3 comparisons. Fixture tests are engineering evidence only. All real-data scientific outcomes remain BLOCKED until authorized multivehicle MOTION records pass provenance and QC. No threshold, baseline or best seed is selected using test results.

## Primary result: information value

At 10 physical seconds, compare full stochastic micro–macro and aggregate-only predicted lane-bin speed/density/flow distributions on identical fixed regions and held-out time windows. Use energy score on the same vector of targets, scaled by fixed physical scales [30 m/s, 100 veh/km/lane, 3000 veh/hour/lane], never test-fitted normalization. Positive paired score reduction favors micro+macro. Primary support requires lower bound of a paired cluster-bootstrap 95% interval >0; at least five independent recording sessions/days are required for a research-level claim. With fewer independent sessions, report preliminary interval and do not claim confirmed generalization. Statistical units are sessions/days, or explicitly labeled buffered contiguous blocks if one session only. 2000 bootstrap draws, seed 7103. Pilot may use one training seed; research uses 101/202/303 and reports all.

## Architecture question (separate)

Compare full stochastic model with independent learned dynamics using identical per-agent observations, and deterministic relational and no-cross-scale variants. Report inputs and parameter counts; same optimizer-update budgets. Full vs macro-only changes information, not just architecture. Full vs independent retains observed agents but removes relational messages, shared innovations and field feedback. Graph-disabled and residual-input permutation controls are secondary; they are not causal interventions on traffic.

## Protocol and baselines

5 Hz target grid, 5 s context; physical horizons 5/10/20 s when the file span supports them. No shorter horizon is relabeled 20 s. Fixed-cohort tracks must exist by cutoff; future entrants are excluded from micro error. Full future reconstructed fields are macro targets, never input. Exits use valid-target masks; report counts and coverage. Open-road generation separately uses an explicit past-estimated inflow assumption and predicted boundary exits, never realized future arrivals. Neither mode establishes real causal control.

Baselines: constant position/velocity first; independent learned, deterministic relational, aggregate-only, micro-only, no-cross-scale, full variational stochastic model; intact/reset memory and graph-off controls. IDM is applicable only with verified leaders, adequate following spans and calibration on training records, otherwise NOT APPLICABLE with reason. All variants use fixed final-step checkpoints; no test selection.

Mask regimes are EMULATED unless authentic source detections exist: dense, sparse (50%/20% tracks), blind spatial segment, contiguous final-history outage. Recompute input fields solely from visible historical agents. Same trained checkpoint, matched scenes, fixed mask seeds. No bidirectional smoothing or interpolation across cutoff. Reconstructed source tracks may themselves incorporate offline future information; disclose that limitation.

## Metrics and statistical safeguards

Micro ADE/FDE/speed error at each supported horizon; per-step drift; valid counts, lane-change and class strata. Proper joint energy score, marginal CRPS, 90% interval coverage/width, sample diversity and minADE (secondary only). Macro component MAE/energy score/coverage; decoder-vs-generated-micro consistency. Physics: overlaps, acceleration >8 m/s², road departures and longitudinal ordering changes, with masks and denominators. Report violations; do not silently clip forecasts to improve results.

For matched-macro analysis, match history-only lane-bin means within fixed normalized Euclidean distance 0.1, and report headway/relative-speed differences. Never match using outcomes. Input permutation moves complete per-agent historical kinematic residual sequences between fixed spatial slots, preserving residual marginals while disrupting their relational placement; label it a diagnostic distribution shift. Score all eligible pairs, not selected examples.

Native source frequency, speed/density regimes, lane changes, continuity and possible waves are audited before training. Wave-onset metrics are gated on independently verified annotated events. No detector is tuned on test data; no precision/recall/lead-time claim from merely slow traffic or synthetic sinusoidal fixtures. 30–60 s wave analysis and cross-day OOD are UNTESTED if adequate records are absent. Known wave reconstruction and analysis is not itself the novel contribution.

Splits group sessions where >=3 exist. Otherwise use chronological 60/20/20 blocks and context+forecast guards; no shared time support across partitions. Window stride >= context+forecast; source IDs preserved. Exclude incomplete windows rather than borrowing future tracks for input initialization. Splits and source hashes are immutable. Fail if an adequate train/validation/test split cannot be built.

## Stop gates

G3 can pass on fixture integration only. G4 requires a REAL checkpoint and finite autonomous validation loss; finite values alone do not imply plausibility or convergence. G5 scientific comparison requires identical held-out support and honest cluster counts. G6 real demo requires an actual REAL checkpoint and recorded scene. G7 requires an explicitly supplied pilot-passed artifact and resource audit; never auto-run the research configuration.
