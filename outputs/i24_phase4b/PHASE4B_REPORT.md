# Phase 4B: preservation, audit and expanded authentic development data

**Achieved:** original recovery verification, source-hashed local bundles,
scientific audit, isolated regression-tested corrections, one-hour authentic
streaming expansion, supplementary metrics and a checkpoint-backed visual gallery.

**Blocked:** durable remote backup and source-date listing. **New training updates:
0. New held-out Phase4B model comparisons: 0.** No retraining, convergence,
multi-date generalization or positive information-value result is claimed.

## Preservation and resources

- Six original checkpoints match recorded SHA256 hashes, alongside unchanged
  pilot reports/evaluations and all prior tracked scientific files.
- Original archive and nested ZIP remain present; no source download repeated.
- Original local recovery bundle: 293,723,563 bytes,
  SHA256 `2087b14b8b92b2c0b22ba034b1ee8925597a2d55ede4897bca8ca6baf35d3e8d`.
- Expanded canonical recovery bundle: 106,007,156 bytes,
  SHA256 `f5002c23bc752124adcb7432df1e081a301be920be947cae4c6644e0b07234dd`. Every bundled file was verified.
- Cloud overlay persistence is not guaranteed across future tasks. Both bundles
  are local, **not remote backups**. No writable artifact identity is configured.
- Drive metadata listing returned HTTP401 once; no retries and no credential
  exposure. Refresh the secure source token and publish. It remains read-only.
- Three CPUs, about 9 GB initially available RAM, no CUDA. Source conversion took
  456.82 seconds; this is ingestion time, not training time. About
  16.67 GB remained free after preparation and preservation.
- Safe setup and recovery instructions: [recovery guide](../../docs/I24_PHASE4B_RECOVERY.md).

## Scientific audit

[Full audit](../../docs/I24_PHASE4B_AUDIT.md) distinguishes correct inference
causality and horizon arithmetic from offline reconstruction limitations.

Confirmed defects: macro-only discarded its learned latent prior; its training
consistency term used microscopic observations; a pseudo-acceleration likelihood
penalized rollout drift as physical acceleration; soft aggregation admitted lateral
departures; the reported departure rate conflated lawful exits with off-road motion.
The prospective correction keeps the layer topology and removes posterior/micro
penalties from a common macro-only proper-score objective. It is an information
control, not an implemented joint-motion convergence result. Active-capacity,
open-population consistency and latent-utility controls remain unresolved.

The full checkpoint's weighted acceleration NLL was 0.05133 versus macro MSE
0.01630 on a declared training-scene diagnostic. This reveals unequal loss pressure,
not proof that it caused every score difference. Original results were untouched.

## Expanded data

Date: **2022-11-21 only**; already-examined development day. The four fixed intervals
are 06:15–06:35 and 07:15–07:35 CST training, 08:15–08:25 validation and 09:15–09:25
developmental test. Bounds were frozen before evaluating any new model. Same
999.744 m WB region and approximate four-lane roadway coordinates.

- **30,170 tracks, 10,368,782 native samples** in 60 minutes.
- **94 / 23 / 23** complete train/validation/development-test windows at 5 Hz,
  five-second context and twenty-second future.
- Maximum window population 464, no truncation; cap 2048 fails closed.
- Independent checks: no shared track identities, no cross-split temporal overlap,
  finite canonical states and exact recomputation of every field/support array.
- Nested member CRC passed during streaming. The full 20.13 GB JSON was not extracted.
- 778 boundary-crossing tracks / 327,070 samples omitted,
  **3.06%** of retained-plus-omitted samples. This is
  less thinning than the original 30.25%, but is still incomplete reconstructed
  population coverage. Twelve short tracks and two adapter-rejected tracks were
  excluded. 417,640 retained native samples were marked unusable by the adapter.

| Block | Windows | Median window speed (m/s) | Minimum concurrent usable tracks |
|---|---:|---:|---:|
| 0615_train | 47 | 3.54 | 10 |
| 0715_train | 47 | 4.43 | 3 |
| 0815_validation | 23 | 13.41 | 6 |
| 0915_development_test | 23 | 31.58 | 3 |

The chronological test block is predominantly free-flow while training is mostly
congested. It tests a single-day regime shift, not an IID test or date generalization.
Whole regional tracks crossing extraction boundaries were omitted to retain the
strict identity rule. This causes thinning near block edges, visible in the low
minimum concurrency. Passing basic QC does not establish census completeness.
Before a census-target study, include non-shared boundary identities with adequate
past derivative support or explicitly bound the edge exposure bias.

Prospective date-disjoint roles are frozen: November 21/22/23 training, November 25
validation, November 28/29 test. Availability is **unverified** because of HTTP401;
no reserved-date outcomes were read. Two test days alone would still fall short
of the preregistered five-session/day confirmation requirement.

## Training and held-out results

Original six models: **100 updates each, seed 101, 38,298 allocated parameters**.
Convergence is unestablished. Original training horizon was 10 seconds; 20-second
scores are autonomous extrapolation beyond that training horizon. No new optimizer
updates were run before durable preservation was established, as requested.

The following remain **historical pilot** results, not expanded-data results:

| Dense 10-second measure | Result |
|---|---:|
| Full macro energy | 0.114990 |
| Macro-only macro energy | 0.109700 |
| Micro-only macro energy | 0.092011 |
| Primary macro-only minus full | -0.005290 |
| Full / CV micro FDE | 12.280 / 11.977 m |
| Full marginal 90% coverage | 20.1% |
| Full CRPS speed / density / flow | 1.978 / 12.970 / 270.322 |
| Full lateral departures / reverse speed | 3.69% / 5.97% |
| Known cohort / target density | 40.02% |

CRPS units are m/s, veh/km/lane, veh/hour/lane. Rates average the three test windows
on target-valid sample-time support. No independent-day confidence interval exists.
Micro-only uses an aggregated micro output and different losses, so its winning
score is not a controlled explanation of the micro-information hypothesis.
The negative primary result remains negative. Calibration and physical validity
are not achieved. Congestion propagation, wave onset and causal vehicle control
are **not demonstrated**.

## Visual experiment and validation

[Visual gallery](gallery/VISUAL_GALLERY.md): four small GIFs, nine scientific figure
families and a standalone interactive viewer. It contains exact arrays for three
historical held-out windows × two observation regimes × eight prior samples.
Positions, fields and masks were checked against every source NPZ. Chromium tested
scenario, regime, sample, truth toggle, time scrubber and play controls with no
JavaScript errors. No server or credential is needed to open the downloaded HTML.
No new Phase4B trained model is implied by these visuals.

## Exact continuation

Preserve this worker until both private bundles are backed up and readback hashes
recorded. See the secure setup guide; never paste credentials. After backup, put a
non-secret receipt at `work/phase4b/private_backup_receipt.json` containing actual
`remote_object_id`, `private: true`, original `sha256` and `readback_sha256`,
`expanded_dataset_object_id`, `expanded_dataset_readback_sha256`, and
`expanded_data_manifest_sha256`. Populate from verified upload/readback evidence,
not invented declarations. The expected expanded manifest hash is
`a9ba486be5bdf5ca17aeebf804c16b409d6aa13ba29eec0439a4f7bfd4666480`.

```sh
cd /workspace/orchestra-wm
uv run python scripts/preserve_i24_phase4b.py
uv run python scripts/prepare_i24_phase4b.py
uv run python scripts/train_i24_phase4b.py --data-output outputs/i24_phase4b/development_v1 --output outputs/i24_phase4b/matched_macro_seed101_u300 --preservation-receipt work/phase4b/private_backup_receipt.json --updates 300 --seed 101 --variants full macro_only --wall-seconds 1800
```

The preparation command verifies/reuses completed scenes. Training is bounded,
uses twenty-second autonomous proper macro-score objectives and all validation
windows, saves atomic optimizer/RNG/gradient records every update, and never opens
test scenes. It deliberately raises on missing backup evidence. It is prospective
code with fixture gradient tests, **not an executed training experiment**.
Rerun the same command after a wall-budget interruption. Wall allowance can change;
scientific budget/source changes require a new namespace and fingerprint. Completed
runs are immutable. No automatic multi-seed or multi-date campaign is launched.

Highest-value next experiment: after verified private backup, run the bounded
common-objective full versus aggregate control on the expanded data, assess
validation convergence and capacity fairness, and only then authorize a new final
held-out comparison. Do not claim joint microscopic fidelity from this macro-only
training objective.
