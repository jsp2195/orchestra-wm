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

- [01_phase1_action_choices](figures/01_phase1_action_choices.png)
- [02_true_factorial](figures/02_true_factorial.png)
- [03_factorial_coverage](figures/03_factorial_coverage.png)
- [04_cross_agent_sensitivity](figures/04_cross_agent_sensitivity.png)
- [05_interaction_error](figures/05_interaction_error.png)
- [06_true_predicted_matrices](figures/06_true_predicted_matrices.png)
- [07_joint_shuffle](figures/07_joint_shuffle.png)
- [08_counterfactual_correlation](figures/08_counterfactual_correlation.png)
- [09_plan_rank_scatter](figures/09_plan_rank_scatter.png)
- [10_phase2_action_choices](figures/10_phase2_action_choices.png)
- [11_centralized_independent_objective](figures/11_centralized_independent_objective.png)
- [12_coordination_gain_ci](figures/12_coordination_gain_ci.png)
- [13_oracle_gap](figures/13_oracle_gap.png)
- [14_hidden_conflict_memory](figures/14_hidden_conflict_memory.png)
- [15_three_seed_summary](figures/15_three_seed_summary.png)
- [16_training_curves](figures/16_training_curves.png)

NEXT EXPERIMENT: Keep the same factorial families, architecture, and update budget; replace the centered sibling loss with a loss on the double-centered interaction residual J, and test whether reduced objective-residual error improves plan ranking and the preregistered coordination gain.
