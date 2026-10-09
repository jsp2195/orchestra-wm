# Phase-2 implementation notes

The committed `PHASE2_PLAN.md` and `configs/phase2.yaml` remain unchanged. No acceptance threshold is selected after evaluating Phase-2 models.

Before retraining:

- The unfinished reproduction check compared CSV row positions although model iteration orders differed. It now aligns semantic keys (`seed`, model/controller, scenario/plan) before numerical comparison. Existing reproduction CSVs are reused only after comparison against unchanged Phase-1 artifacts. This repairs validation, not results.
- D eligibility is determined only from the pre-training audit: the measured mean action-A influence on B must fail to be larger for interacting than distant targets. A nonpositive interacting-minus-distant mean triggers the already-planned small pairwise channel. This directional operational rule is fixed before executing the remaining audit; it does not change INT-A–H.
- Identical sibling families form every nine-example training group. Configured batch 18 contains two complete families. All models share factorial validation families; A trains only on the original data. The pre-registered 25% natural fraction is implemented deterministically as one in every four optimizer batches.
- Joint-pairing shuffle independently permutes B's complete action sequences across the nine siblings while holding A fixed. The B marginal is unchanged; we also report an ordinary full-pair shuffle as a comparator. Primary interaction error is the mean Euclidean error of AVs 0 and 1 at step 10.
- C remains the primary model for all gates; D, if eligible, cannot replace C for INT-H. The original-data retrain A is the table's Phase-1 architecture baseline, distinguished from the frozen Phase-1 checkpoint used in audits.

Execution details:

- The evaluated horizon is ten autonomous steps; prediction tables report final-step pair/all-agent position, speed, conflict-proxy and total-objective errors. Separate CSVs report factorial objective residuals and their agreement, diverse-pool ranks, command/joint-strategy frequencies, and closed-loop task components.
- Confidence intervals use training-seed paired means with Student-t df=2. The secondary bootstrap uses 5000 draws, seed 1701, sampling training seeds and then paired families within each selected seed. This secondary interval never replaces INT-H.
- The optional hidden-conflict experiment uses four additional disjoint families. AV B is visible in the four-frame context, then loses direct state and telemetry for six replanning steps. Intact and reset-memory C use identical scenarios and planning proposals; this is secondary to the primary 16-family experiment.
- The original preservation manifest also hashes ignored timestamped TensorBoard files. Exact preservation works in the current workspace; a fresh clone requires those original archived inputs. Regenerating the ignored Phase-2 dataset was explicitly tested against all three immutable manifests and reproduced their bytes exactly.
