# Phase 2: joint interaction and coordination

Phase 1 at `dec08b6` is preserved. All new artifacts go under `outputs/phase2`; the original dataset, checkpoints, reports and results are read-only inputs.

## Fixed sequence

1. Re-execute Phase-1 interaction and closed-loop planning measurements and compare numeric columns (excluding wall timings).
2. Audit each AV's executed actions, relevant pair coverage, true 3×3 factorial outcomes, and cross-agent action sensitivity before training.
3. Build same-initial-state sibling families on intersections/merges; retain natural-traffic training batches. Entire family IDs are disjoint across train/validation/test. Hidden interaction labels and sibling outcomes are targets/metadata only.
4. Train A (original data), B (factorial data), C (factorial data + difference loss), independent factorial baseline, and privileged factorial diagnostic with three independent seeds (101,202,303). If the Phase-1 action pathway lacks spatial specificity, add D, a small explicit relative-geometry message channel; do not change Phase 1.
5. Evaluate held-out interaction states and matched closed-loop episodes. Generalization is gated on the primary coordination criterion.

## Fixed compute and comparisons

Use the Phase-1 128-wide, two-layer architecture (~1M parameters), 240 optimizer updates, batch 18, ten-step autonomous training, identical learning rate, same three seeds and a fixed family-level validation set. Models B/C/D share exactly the same data schedule. A uses the original episode-split dataset. The independent and privileged controls also train on factorial/natural mixtures for the same update budget. C/D reuse the ordinary rollout predictions for sibling losses, so they do not receive additional target batches. No early stopping or test-set model selection. Report that a fixed update budget does not imply convergence or equal optimization difficulty.

Factorial families have nine constant pair commands; other connected vehicles are absent (background traffic remains). For every family, use a cloned simulator and identical observation history. A 25% natural-trajectory batch fraction preserves ordinary dynamics. Training, validation and evaluation simulator seeds use disjoint ranges. Validation never splits siblings across partitions.

## Definitions and gates (before retraining)

For each scalar or vector 3×3 outcome matrix F, define the additive projection A(a,b)=row_mean(a)+column_mean(b)-grand_mean. The interaction residual is J=F−A. Report RMS true J and RMS prediction error on J; this isolates non-additive effects from marginal action effects.

- INT-A: mean true objective interaction RMS exceeds 0.05 and at least one third of audited families have RMS >0.05.
- INT-B: every training family contains all nine joint choices; relevant pair coverage is reported separately from ordinary background traffic.
- INT-C: primary ten-step position error on the two interacting AVs improves over the matched independent model, with a seed-level paired 95% interval above zero for the error reduction. Also report all-agent errors, speed, conflict proxy, and residual errors.
- INT-D: positive held-out factorial objective Pearson and pairwise outcome-geometry correlation; inspect full matrices, not only averages.
- INT-E: destroying B pairings while retaining A's sequence and both marginal action histograms increases interaction forecast error by at least 5%; report a seed-level interval. This intervention can also affect an independent action-conditioned model, so shuffle degradation alone does not prove non-additivity.
- INT-F: mean interaction-plan Spearman >0.5 and top-10% overlap above chance (2/18 for the diverse 18-plan pool); report regret and selected true percentile.
- INT-G: at least two different joint strategies occur and no individual command exceeds 95% across matched geometries. Diversity alone is not success and commands are never penalized for their labels.
- INT-H (primary): paired cost(independent learned MPC)−cost(centralized C) has a 95% Student-t interval across three independent training-seed means whose lower bound exceeds 0.1 objective units. Also report seed mean/SD/SE, episode-paired hierarchical bootstrap intervals, per-task results and D as a secondary comparison. Do not substitute the best model/seed after seeing results. The independent learned baseline uses the Phase-1 additive per-agent cost predictor with identical CEM budget; it has no cross-agent predictive pathway.

Use 16 held-out families (8 per task), all evaluated under each independently trained model. Candidate pools must cover all nine factorial commands and diverse temporal alternations. For the Phase-2 CEM experiment, use the same initial balanced candidate set and proposal updates for learned and oracle planners. Keep objective weights unchanged. Compare the old CEM/short horizon against the balanced candidates/long horizon during the pre-training oracle audit; changes are applied equally, never selected to reward action diversity.

Confidence intervals treat training seeds as independent units. Many episodes within one trained model are not independent model replications. Three seeds give wide intervals; a negative result is an acceptable outcome. If INT-H fails, do not run OOD or strengthen the headline claim.

## Required outputs

Action distributions and coverage CSVs; simulator factorial outcomes; sensitivity audit; immutable sibling metadata and split manifests; train/validation curves and checkpoint metadata; interaction, shuffle, rank and MPC measurements; 15 figures; confidence intervals and gate results; an appended Phase-2 claim ledger and README section. Exactly one next scientific experiment will follow the measured result.
