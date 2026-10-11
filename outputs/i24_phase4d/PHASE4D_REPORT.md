# Phase 4D: residual forecasting and matched information controls

**Executed result: a substantially better validation joint macro forecast, but no
demonstrated microscopic-information or interaction advantage. Independent-date
generalization was not executed.** All new results use November 21 training and
validation only. The examined developmental-test partition and November 28/29
prediction outcomes were not opened. Previous negative findings are unchanged.

At ten seconds, residual full macro energy was **0.090313 / 0.090657** for seeds
101/202, versus **0.151269** for aggregate persistence and **0.119389** for a fitted
stochastic persistence control. This is approximately 40% lower than persistence,
on validation used for checkpoint selection. Speed point error remains worse than
persistence. The matched aggregate-only models slightly outperform full in both
seeds. These are developmental engineering results, not independent test estimates.

## 1. Preservation and execution scope

Started from `3eca2217ee2d106f96b53524f483f842db757f68`, branch
`phase4-i24-continuous`. The prospective implementation/protocol was committed and
pushed as `901302cae28d50242550e49513479db8b566ac3c`. Final delivery commit is the
commit containing this report; use `git rev-parse HEAD` on that checkout.

All 719 inventoried historical tracked files remained byte-identical. All six
original 100-update pilot checkpoints, both 300-update macro controls, and all
three Phase 4C joint checkpoints were SHA256-verified against their recorded
hashes. See [historical_preservation.json](historical_preservation.json). Existing
raw/nested source archives, canonical data, checkpoints, recovery bundles, reports,
negative results and demonstrations were retained. No original experiment was
retrained or overwritten. The genuine Phase 4B preservation gate passed before
new training. No restricted arrays or weights are published to GitHub.

New private recovery verification is recorded separately in
[preservation_status.json](preservation_status.json), created only after remote
readback succeeds. The private manifest contains archive/member/part SHA256,
byte ranges, ordered Drive IDs, and recovery instructions. It includes new model
optimizer/RNG checkpoints, the feature cache and saved prediction arrays. The
existing prepared canonical dataset remains in its previously verified backup;
the new cache is not a substitute for those original canonical files.

This overlay filesystem has no established cross-task durability guarantee.
Git preserves code/results, and the authenticated private recovery artifacts
preserve model/data bytes. No files were deleted to gain storage.

## 2. Quantitative failure analysis

Analysis used the old models' **validation** predictions and the existing train/
validation sources, never their developmental-test outcomes. Detailed evidence:
[diagnosis](diagnosis/diagnosis.json), [bias and drift](diagnosis/validation_summary.csv),
[optimization](diagnosis/optimization.csv).

| Diagnostic | Training | Validation |
|---|---:|---:|
| Mean supported-bin speed, m/s | 6.069 | 14.113 |
| Mean density, veh/km/lane | 35.190 | 22.131 |
| Mean flow, veh/h/lane | 622.726 | 1040.539 |
| Fraction supported bins below 10 m/s | 83.33% | 19.53% |
| Fraction above 20 m/s | 0.32% | 7.58% |
| Supported-bin fraction | 97.63% | 95.76% |

These are scene-equal summaries of reconstructed populations, not a complete
roadway census. A substantial regime shift is established. Its isolated causal
contribution to model error has not been measured.

The old absolute macro decoder predicted 8.228 m/s after 0.2 seconds against
14.118 m/s observed truth; its ten-second bias was -7.025 m/s and speed MAE
7.054 m/s. Its raw-zero softplus reference is 5.199 m/s, not the most recent
observed speed. This is an initialization reference, not a hard speed cap. The
immediate discontinuity and persistent low-speed bias support using an observed-
state residual reference. Physical normalization [30,100,3000] was consistent;
no new feet/metres or 5 Hz conversion defect was demonstrated.

After the residual change, full's two-seed mean speed bias is -0.0068 m/s at
0.2 s, -0.665 at 10 s, and -1.360 at 20 s. Mean departure from the reference
is 0.264, 1.480 and 2.031 m/s respectively. Longer rollout bias remains; the
absolute decoder's immediate collapse has been substantially reduced. See
[residual_drift.csv](analysis/residual_drift.csv). This architectural comparison
also changes optimization and training duration, so it does not isolate one
causal factor.

The old macro controls had finite gradients, median norms about 0.078/0.079 and
no clipping. Their failure was not evidence of exploding gradients. The old
joint full model had median norm 12.75 and clipping on 96.7% of its 30 updates;
independent clipped 83.3%, graph-off 100%. This supports separating the macro
information experiment from the imbalanced joint objective; it does not prove
that clipping alone caused poor forecasts. The old macro full had 36,850
gradient-connected parameters versus 17,870 for macro-only despite equal
allocation. That comparison could not isolate additional microscopic input.

Prior/graph sensitivity previously existed, but sensitivity did not establish
benefit. The new matched controls and one-at-a-time diagnostics below directly
test narrower versions of those questions. No negative historical finding has
been reinterpreted as positive.

## 3. Authentic data actually used and access limitation

No new source date was ingested. The prepared source contains 60 minutes, 30,170
source records and 10,368,782 native samples with 94/23/23 train/validation/
developmental-test windows. **Phase 4D used only the first two partitions:**

| Partition | Nov 21 intervals (CST) | Minutes | Source track records | Native samples | Windows |
|---|---|---:|---:|---:|---:|
| Train | 06:15 and 07:15, 20 min each | 40 | 23,783 | 8,458,052 | 94 |
| Validation | 08:15, 10 min | 10 | 4,332 | 1,321,549 | 23 |

Source counts precede filtering; they are not counts of unique vehicles or all
usable canonical samples. The existing train intervals excluded 525 boundary/
guard track records and 274,674 corresponding samples; validation excluded 193
and 44,271. Native unusable sample counts were 305,778 train and 71,850 validation.
Keep these source audit categories distinct rather than assuming disjoint losses.
Track-fragment identities, reconstruction gaps and selection-boundary losses
limit population completeness. No new identity truncation or trajectory overlap
was introduced; the inherited manifest reports zero shared source IDs across
partitions. Stable source IDs do not establish perfect physical identity through
reconstruction fragments.

Scenes cover approximately 1 km westbound, four lanes and eight longitudinal
bins per lane, with five-second observation history and twenty-second autonomous
forecasting at 5 Hz. Native reconstruction is 25 Hz. The cache has 351 entries:
117 unique windows times dense/sparse50/outage emulations, **not 351 independent
recordings**. Canonical and cached file hashes are checked on load.

Targets retain the existing causal trailing-one-second exposure definitions:
density from vehicle-time per lane-length, speed from distance/time exposure,
and flow from distance per lane-length/time. The sample identity is q=3.6kv
for m/s, veh/km/lane and veh/h/lane. Scoring support represents available
reconstruction, not proof that unsupported cells are empty. Future support is
used only in scoring/loss, never in inference. Source trajectories themselves
are offline reconstructions; causal model inputs do not undo source smoothing.

The configured direct bulk-source credential was tested once: **HTTP 401,
`authError`**, despite the binding being configured. The response does not
distinguish expiration, revocation or another credential mismatch. It was not
retried. Connected Drive can preserve compact artifacts, but its supported
materialization path is limited to 32 MiB; no multi-GB source stream was run.
At the storage check 12.60 GB was free, below the smallest new assigned archive
plus the existing 10 GB reserve. Existing verified archives were retained.

[multidate_selection.json](multidate_selection.json) preserves training Nov 21/22/23,
validation Nov 25, test Nov 28/29, and source-only 06:15/07:15/08:15/09:15 ten-minute
slots with QC exclusions reported rather than outcome-driven replacements.
The existing streaming ingestor still needs per-date acquisition/session metadata
parameterization and verification. It was not represented as already working on
new dates. Two independent test days would still fail the five-group confirmation
gate. No independent-date performance or confidence interval is available.

## 4. Residual models and prospective comparison

[Protocol](../../docs/I24_PHASE4D_PROTOCOL.md) was fixed before new validation
evaluation. The main model is a **new residual roadway-field graph forecaster**,
not a claim that the old vehicle-graph model has been repaired in place. Fixed
lane-bin nodes use 32 hidden dimensions, graph messages, history and autonomous
GRUs, local stochastic diffusion and shared four-dimensional innovations.
Speed/density evolve as log residuals around latest supported observations;
flow is derived from each generated sample. A 0.05 positive floor is disclosed
and controlled by separately scoring identically floored physical persistence.

Both primary models execute the same layers, graph, recurrent computation and
21,100 scalar parameters. **All 21,100 received nonzero gradients in each run.**
Aggregate inputs contain identical fields, support/age, roadway position and an
eight-feature deterministic aggregate basis. Only full adds eight causal vehicle
statistics: speed/variance, acceleration, lateral velocity, length, within-bin
position, observed history age and braking fraction. Zero additions make the
same-weight models bitwise identical. This tests these microscopic summaries,
not every possible representation of vehicle histories. Equal computation and
observed gradients do not guarantee equal optimization difficulty or statistical
effective complexity.

Both use the same proper normalized joint macro energy objective over all 100
autonomous forecast steps, four training particles, batch four, AdamW 0.001,
weight decay 0.0001 and gradient clipping at one. Paired runs share initialization,
minibatches, observation regimes, noise and LR schedule. No future-conditioned
posterior, teacher forcing, micro objective or cohort-to-total conservation penalty
enters this information comparison. Validation selects one common checkpoint
for each pair using mean 5/10/20-second energy. The ten-second primary metric
and its sign remain unchanged.

The separate microscopic residual trial predicts the historical cohort around
CV, using learned acceleration with a fixed two-second exponential decay.
Both 24-dimensional models have 5,692 parameters and identical local noise;
graph versus independent changes only historical proximity messages versus
self-messages. They do not generate arrivals or model future interaction graphs
recursively. Complete macroscopic traffic and surviving historical vehicles
remain distinct targets; no false conservation equality is imposed.

## 5. Actual optimization and convergence

| Run | Updates per model | Common selected update | Stop/status | Recorded wall seconds | Peak RSS MiB |
|---|---:|---:|---|---:|---:|
| Residual full + aggregate, seed 101 | 600 each | 350 | Five nonimproving checks; stability heuristic false | 565.46 | 583.86 |
| Residual full + aggregate, seed 202 | 800 each | 750 | Update cap; stability heuristic true | 616.78 | 585.77 |
| Micro graph + independent, seed 101 | 250 each | 150 | Four nonimproving checks | 33.23 | 680.21 |
| Smaller aggregate, 5,692 parameters | 600 | 350 | Replayed seed101 budget/LR/selection index | 294.44 | 518.10 |
| Two-parameter diffusion persistence | 500 | 500 | Fixed train-only fit; no validation selection | 37.78 | See execution logs |

There were **4,400 actual optimizer updates across eight trained instances**.
Pair wall times cover both models plus validation, not each model separately.
These recorded times sum to about 25.8 minutes; they are not end-to-end task
latency. Runtime had two CPU quota cores, 8 GiB RAM and no GPU, using two Torch
threads and zero data-loader workers. New training stayed below 0.7 GiB recorded
peak process RSS. Numerical smoke: finite loss/gradients, approximately 0.418 s
per warmed primary update at batch four/four samples.

Primary median gradient norms were about 0.128–0.129; only 0.33%/0.625% of steps
clipped for seeds101/202. No nonfinite training or scored prediction was accepted.
First/last fifty-step training losses decreased from about 0.120 to 0.0815
(seed101) and 0.1195 to 0.0787 (seed202). Training averages cover all times and
mixed observation regimes and should not be directly equated to dense endpoint
validation scores. The best shared validation objectives were 0.088879 and
0.089008. Last-four-check ranges were 0.002564 and 0.001204. No-improvement
stopping is not convergence proof; seed101 and the micro trial did not establish
stable convergence, and seed202 only passed the declared finite-run heuristic.

Atomic checkpoints retain model, optimizer, RNG, progress and source/config/data
fingerprints. Completed namespaces reject configuration changes. Best and last
weights remain available; validation-selected checkpoints, not final or favorable
test weights, were evaluated. The small-capacity control inherits update350,
even though its later validation loss continued decreasing.

## 6. Matched validation forecasting results

All values below are scene-equal dense validation means. Eight independent
samples are used for stochastic primary scoring; both seeds are reported.
Exact historical target/support equality was verified for 46 scene/regime pairs
and 138 horizon scores; persistence score differences were zero. Historical
Phase4C ran dense/outage, not sparse50. See
[comparability audit](analysis/historical_comparability.json).

| Model | Seed | Macro ES 5s | ES 10s | ES 20s |
|---|---:|---:|---:|---:|
| Aggregate persistence | — | 0.126706 | 0.151269 | 0.155651 |
| Fitted diffusion persistence | 101 | 0.093851 | 0.119389 | 0.141107 |
| Residual full | 101 | 0.080176 | 0.090313 | 0.093188 |
| Residual aggregate | 101 | 0.080180 | 0.090115 | 0.092663 |
| Residual full | 202 | 0.080421 | 0.090657 | 0.095146 |
| Residual aggregate | 202 | 0.080275 | 0.090003 | 0.094706 |

At 10s, physical persistence scores 0.151242, so the floor does not explain the
improvement. The old matched macro full/macro-only score 0.151485/0.152572 on
this validation partition; the old joint full/independent score 0.134111/0.128366.
The new small aggregate scores 0.097197, versus 0.090115 for its larger control.
Capacity matters in this limited comparison; it is not evidence for microscopic
information. The stronger two-parameter stochastic control shows that the new
score gain is not explained entirely by adding uncertainty to a point forecast.

| Ten-second model | Speed MAE m/s | Density MAE veh/km/lane | Flow MAE veh/h/lane |
|---|---:|---:|---:|
| Persistence | 1.848 | 9.843 | 494.941 |
| Diffusion persistence | 3.811 | 10.056 | 552.304 |
| Full101 | 2.423 | 8.678 | 425.817 |
| Aggregate101 | 2.415 | 8.629 | 424.763 |
| Full202 | 2.351 | 8.402 | 420.674 |
| Aggregate202 | 2.366 | 8.456 | 418.780 |

Speed mean error is worse, despite improved density/flow and joint energy.
For full101, speed/density/flow marginal CRPS at10s is 1.832/6.209/295.007,
versus persistence 1.848/9.843/494.941. The post-selection 64-particle precision
audit reduces full101 speed MAE to **1.936**, still worse than persistence1.848.
Its density/flow MAE is8.540/413.635. Finite eight-particle averaging contributes
to point error but does not eliminate the remaining speed weakness. This audit
does not replace primary scores or choose checkpoints.

The preregistered sign remains **aggregate-only minus full**:

| Seed | 10s paired difference | Windows favoring full |
|---|---:|---:|
| 101 | -0.000198163 | 9/23 |
| 202 | -0.000654659 | 8/23 |

The two-seed mean is -0.000426411. Both favor aggregate-only slightly. Correlated
windows from one validation recording, selected on that validation, do not justify
an independent confidence interval or a confirmed information-value claim.
No microscopic-information advantage was established. This cannot establish that
all vehicle histories are intrinsically useless; it limits these features/models.

One-at-a-time full101 inference diagnostics at10s:

| Intervention | Energy | Difference from intact full101 |
|---|---:|---:|
| Intact | 0.090313 | 0 |
| Remove microscopic additions | 0.091021 | +0.000708 |
| Replace graph messages with self-messages | 0.090457 | +0.000144 |
| Reset history GRU each frame | 0.091924 | +0.001611 |
| Remove shared 4D latent, retain local diffusion | 0.090266 | -0.000047 |
| Remove all stochastic innovations | 0.128611 | +0.038298 |

These change one mechanism at a frozen checkpoint but introduce an inference
distribution shift. They are not retrained controls or causal evidence. History
reset still retains engineered historical features. Learned stochasticity is
useful for this score; shared latent and graph benefit are not convincingly
established. The trained aggregate control is the appropriate information
comparison, not the micro-removed inference result.

Under emulated outage, full101/full202 ES10 is0.116979/0.117355, versus persistence
0.232587 and diffusion0.165188. Sparse50 gives0.125881/0.124747 versus0.208929 and
0.150100. These emulations use the same corrupted history and future scoring
support for every model; they are not measured real sensor failures.

All23 dense scene-average historical speeds fall in10–20m/s. This dataset does
not validate broad scene-level congested/free-flow generalization. A separately
labeled post-selection bin-history diagnostic has139 congested,498 intermediate,
and56 free-flow supported bin endpoints at10s, spread over22/23/16 correlated
windows. Full101 improves density/flow MAE within each stratum but worsens speed
MAE (congested1.630 vs1.387, intermediate2.304 vs1.596, free2.956 vs2.701).
See [bin-regime errors](precision_audit/bin_regime_summary.csv),
[all component scores](validation/summary_table.csv) and
[historical comparisons](analysis/macro_comparisons.csv).

## 7. Microscopic forecasting, calibration and physical validity

| Model | ADE 5/10/20s, m | FDE 5/10/20s, m |
|---|---|---|
| Constant velocity | 1.885 / 5.106 / 11.550 | 5.467 / 16.985 / 49.071 |
| New graph residual | 1.899 / 5.098 / 11.479 | 5.482 / 16.854 / 48.719 |
| New independent residual | 1.899 / 5.098 / 11.479 | 5.482 / 16.855 / 48.709 |
| Old Phase4C full | 1.942 / 5.327 / 12.227 | 5.665 / 17.876 / 53.177 |
| Old Phase4C independent | 1.886 / 5.105 / 11.581 | 5.462 / 16.990 / 49.368 |

The small10/20s improvements over CV are validation-only, with no convincing
graph advantage. Five-second FDE is slightly worse. Endpoint denominators fall
from1,189 to693 to251 across the23 scenes; these metrics concern surviving
observed-cohort tracks, not arriving vehicles or the full traffic population.
Graph model10s position CRPS is8.548m versus CV9.323m; scaled position energy
is0.465567 versus0.648346. Eight-particle coordinate coverage is only42.6%.
See [micro evaluation](micro_seed101/validation/summary_table.csv).

Eight-sample full macro nominal90% coverage at10s is69.5%/65.3% for101/202.
Small ensembles themselves bias empirical quantile coverage; therefore a
separate64-particle audit also uses randomized predictive ranks:

| Model101, 10s | Speed coverage | Density coverage | Flow coverage |
|---|---:|---:|---:|
| Full, empirical5–95% quantiles | 96.60% | 63.54% | 78.19% |
| Aggregate, empirical5–95% quantiles | 96.74% | 64.27% | 77.91% |
| Diffusion persistence | 98.02% | 30.15% | 86.09% |

Full's interval widths are15.15m/s,21.39veh/km/lane and1327veh/h/lane. Its
randomized rank central90% coverage is97.18%/67.08%/81.64%. Speed uncertainty
is too broad while density and flow remain undercovered. Nonzero diversity and
proper scores do not demonstrate calibration. Correlated bins are not used for
iid calibration p-values. See [calibration audit](calibration64/summary.json).

Macro generated fields were finite and nonnegative; sample flow identity error
was zero at reported precision. This algebraic identity is not a complete
conservation or traffic-feasibility guarantee. For full across both seeds,
0.088% of scored sampled speeds exceeded55m/s at10s and0.193% at20s; none of
these scored densities exceeded200veh/km/lane. These are heuristic extreme-value
screens, not universal physical thresholds; no evaluation clipping was applied.

For the micro graph model at10s, reverse-speed rate was0.218%, lateral departure
8.10%, and endpoint overlapping-pair fraction0.229%. At20s lateral departure
rose to14.37%. CV has5.55% lateral departure and0.256% overlapping pairs at10s.
No sampled acceleration exceeded8m/s². Overlap/departure diagnostics depend on
reconstructed dimensions and roadway support; exits are not automatically
violations. The trial is neither collision-free nor physically validated as a
complete world simulator. No new congestion-wave propagation or causal vehicle
control claim was tested or demonstrated.

## 8. Scientific artifacts, tests and continuation

Only a small set of diagnostic figures was produced. Existing interactive and
historical visual demonstrations were left unchanged because this is not a
validated improvement of the complete joint world model.

- [Learning curves and gradients](analysis/learning_curves.png)
- [Macro and micro baseline comparisons](analysis/baseline_comparisons.png)
- [Calibration, observation loss and historical-bin errors](analysis/calibration_and_regimes.png)
- [First chronological checkpoint-backed forecast example](analysis/checkpoint_example.png)
- [Figure/array provenance](analysis/figure_provenance.json)

The example uses the first validation scene, not a favorable selected outcome.
Its recorded and generated series are distinct. The displayed spatial average
uses scoring support solely to compare outcomes; that future support is not a
model input. Raw trajectory/forecast payloads remain private.

Regression suite: **119 passed, 8 skipped** using importlib mode. Six focused
Phase4D tests cover aggregate input isolation, full micro sensitivity, zero-input
equivalence, future-label exclusion, physical identities, correct energy support,
observation carry-forward and independent vehicle isolation. Actual completed
evaluation additionally asserted finite arrays, checkpoint/source fingerprints,
exact target/support comparability and saved-array integrity.

### Exact reproducible commands in the preserved environment

```bash
cd /workspace/orchestra-wm
git switch phase4-i24-continuous
uv run pytest --import-mode=importlib -q
uv run python scripts/train_i24_phase4d.py --output outputs/i24_phase4d/residual_seed101 --seed 101 --max-updates 800 --wall-seconds 1800
uv run python scripts/train_i24_phase4d.py --output outputs/i24_phase4d/residual_seed202 --seed 202 --max-updates 800 --wall-seconds 1800
uv run python scripts/train_i24_phase4d_micro.py --wall-seconds 900
uv run python scripts/train_i24_phase4d_capacity.py
```

Primary commands verify completed results and return without extending training;
the micro/capacity commands refuse completed namespaces. These are also the
identical-config resume commands for a recovered incomplete atomic run. Do not
increase update limits in these namespaces or bypass fingerprints. All runs in
this delivery are complete; none requires further updates to reproduce its state.

To reproduce evaluation in a **new**, unused output namespace without touching
completed arrays:

```bash
uv run python scripts/evaluate_i24_phase4d.py --runs outputs/i24_phase4d/residual_seed101 outputs/i24_phase4d/residual_seed202 outputs/i24_phase4d/capacity16_seed101 --output outputs/i24_phase4d/validation_reproduction --ablations
```

Historical-scoring comparison and figures are reproduced by
`uv run python scripts/summarize_i24_phase4d.py` after restoring the evaluated
arrays and precision audit. Restore private recovery parts in manifest order,
verify part and concatenated SHA256 before extracting in the repository, and
retain the separately backed canonical dataset/Phase4B receipt. A Git checkout
alone cannot recreate the preserved trained weights.

**Highest-value next experiment:** train the matched residual pair on the frozen
Nov21/22/23 source-only intervals and select/check calibration on Nov25, before
opening Nov28/29. First restore the secure bulk-source authorization and storage
headroom, then parameterize and test the existing staged ingestor's per-date
metadata. No valid executable new-date ingestion command exists yet; inventing
one would conceal unfinished work. Freeze that dataset/config in a new namespace.
Prioritize speed-point accuracy and density calibration on independent validation
over more visualizations or more seeds of the present single-day trial.

The independent-date engineering objective remains untested. The present work
establishes actual bounded optimization, a stronger probabilistic macro baseline,
and a better-controlled negative microscopic-information result on selected
single-day validation. It does not establish a competitive, calibrated,
open-population joint traffic world model.
