# Run Phase 4E in Google Colab

Open `notebooks/ORCHESTRA_Phase4E_Colab.ipynb` in Colab and choose **Runtime → Run all**.
Authorize the built-in Google Drive mount. No access token, refresh token, client
secret, environment publisher, or Drive API configuration is used by this workflow.
The public GitHub checkout requires no GitHub credential.

The default pipeline restores the verified completed November21/22 datasets,
processes November23/25, runs combined QC, builds and backs up the train/validation
cache, performs the numerical pilot, trains both paired seeds, fits stochastic
persistence, and evaluates November25 validation and 64-particle diagnostics.
It processes November28/29 and evaluates their forecasts only after validation
promotion, training-readiness checks, and an immutable test specification pass.

## What must be visible in the mounted Drive

The notebook automatically searches **My Drive** for the original ZIP names and
existing backup manifests/parts. If the source folder is in **Shared with me**, add
a shortcut to it in My Drive once. If duplicate archive names exist, set the
optional `SOURCE_ROOT` field to their one intended directory. This field selects
source ZIPs only; backup discovery still searches My Drive.

Required existing recovery manifests (and their previously verified parts):

| Dataset | Manifest | Expected manifest SHA256 |
|---|---|---|
| Nov21 | `ORCHESTRA_PHASE4E_COMPLETED_NOV21_RECOVERY_MANIFEST.json` | `7370299e46ec2c9362e4741301b10b9dcf665521d8d0a4dc5ef5a18dc1ac5c75` |
| Nov22 | `ORCHESTRA_PHASE4E_COMPLETED_NOV22_RECOVERY_MANIFEST.json` | `aebdcd943f82f76c9555b1cd35e0af52df9f8f89448224791538de2ea1ddba2e` |

Required new sources are `11-23-2022.zip` and `11-25-2022.zip`. The files
`11-28-2022.zip` and `11-29-2022.zip` are needed only after promotion. November21
and22 original ZIPs are not needed and are never used as a silent reingestion
fallback. Missing backups stop with a specific message; incomplete historical
shards are not substituted. No Drive IDs or private links are embedded in code.

## Resumption and private preservation

The default new folder is `MyDrive/ORCHESTRA_WM_PHASE4E_COLAB/`:

- `workspace/outputs/i24_phase4e/` holds completed/recovered data, caches, checkpoints,
  QC, curves, validation and gated test outputs.
- `workspace/work/phase4e/` holds private recovery, data-preservation and stage receipts.
- `backups/` holds immutable, content-addressed compressed snapshots and inventories.

If Colab disconnects, reopen **the same notebook revision**, keep the same private
folder and choose Run all again. Code/configuration hashes are locked; a different
implementation cannot silently resume this experiment. Do not run two sessions
against the same private folder at the same time.

Completed date files are checked against their verified inventory and reused.
During new ingestion, every30,000 source records the converter flushes native
shards and commits a gzip JSON state containing counters, IDs, native periods and
shard hashes. Resumption replays the compressed JSON prefix (ZIP is sequential)
but skips conversion of committed records. Uncommitted shards move to a preserved
interruption directory, never into accepted data. Canonical preparation restarts
from the verified complete native manifest. An interrupted feature cache is moved
aside and deterministically rebuilt; completed caches are verified, not replaced.

Each paired optimizer update writes model, optimizer, NumPy/Torch RNG, schedule,
capacity masks, and validation history to an atomic Drive checkpoint and verifies
its bytes after reopening. Both models are in the same checkpoint. Selected best
checkpoints and every100-update checkpoint also trigger a separate full-run backup.
An interruption between writing last and best is repaired from the selected last
state before another update. The diffusion control similarly checkpoints every
update and backs up each100. A55-minute training time slice resumes automatically
with exactly the same configuration until the frozen stopping rule/cap is reached.

Completed dates, complete training data/cache before substantive training, QC,
pilot, model runs, validation/calibration, frozen test inputs and final predictions
receive compressed backups with full per-member SHA256 readback. Receipt filenames
and contents survive restart. Files are under a new My Drive folder; the notebook
performs no sharing operation. **Readback is through the mounted filesystem, not
an independent Drive API download, and ACLs are not independently queried.** Keep
this folder private. FUSE/server synchronization and Colab runtime limits still
apply; the notebook cannot guarantee an uninterrupted session.

A verified snapshot can be restored without overwriting differing files using
`storage.extract_verified(archive, receipt['files'], destination)`. Keep the
receipt and archive together. Historic original backup parts are never deleted.
New archives/checkpoints and private source bytes never enter the Git checkout.

## Preserved scientific controls

This adaptation starts at commit `2cf2330533f6630425116b636f1d7ba878862653`.
All pre-existing tracked scientific files are preserved byte-for-byte. The new
`orchestra_wm/i24_phase4e_colab/` adapters port the existing conversion/training
logic and add filesystem transport, durable state, and preserved restart behavior.
The Phase4D model, causal feature builder, cache format, QC, primary evaluation,
calibration and test-freeze scripts remain unchanged.

Date roles stay train Nov21/22/23, validation Nov25, independent test Nov28/29.
The four America/Chicago600-second slots (06:15/07:15/08:15/09:15), roadway geometry,
25Hz native sampling, causal5Hz conversion, support/exclusion rules, five-second
history and20-second targets are unchanged. Original mounted ZIPs must match the
acquired full outer size/MD5; both selected ZIP CRCs and nested/JSON SHA256 are
verified. No skipped/rejected slot is replaced and no population is truncated.

The full/aggregate21,100-parameter models use paired seeds101/202 from scratch,
identical minibatches/regimes/noise, the original optimizer/objective and joint
validation selection, maximum1600 updates, minimum500, and the frozen stopping
rule. Diffusion fits training only for500 updates. CPU with two threads and zero
loader workers is intentional; the notebook does not auto-select a GPU or change
precision/budgets. A GPU runtime is unnecessary for this frozen implementation.

Promotion still requires all four learned primary models to beat ordinary
persistence on dense November25 scene-equal mean5/10/20-second energy. Full rows,
identical cache/windows/support, both seeds and matched active capacity are checked.
Missing results fail closed. In addition, the notebook conservatively stops for
validation-only review if the existing last-four-check stability heuristic is
false; it does not equate that heuristic with convergence, extend the cap, retune
architecture, or open tests to resolve it. Calibration is diagnostic, not a new
selection objective. Test sources are processed only after this readiness decision;
all-six-date source QC and verified test backup precede forecasting. A source/code/
checkpoint hash freeze binds the exact evaluator and selected weights. Subsequent
runs cannot tune on test outcomes or replace completed results.

Independent reporting includes paired seed/day effects and equal-day pooled effects.
Two test dates remain below the five-independent-group confirmation gate. No
confirmed microscopic interaction benefit or complete joint world-model claim is
implied by macro prediction improvements.

## Verification and execution status

Local behavioral tests cover mounted ZIP identity/integrity without network access,
private multipart restoration, malicious tar members, corrupted backups, native
resume equivalence to the frozen converter, exact numerical model/optimizer/RNG
resume after a disconnect, and failed/missing/unstable validation gates. The notebook
also runs the focused tests before accessing data. Dependency setup uses the
existing `uv.lock`, Python3.11, and pinned streaming requirements.

The actual Colab mount and full data/training campaign require the user's browser
authorization and have **not** been executed in Codex. Fixture tests are not research
results. No new Phase4E training or independent-test result is claimed by delivering
this notebook. The run writes its actual status to
`workspace/work/phase4e/colab_execution.json` and prints progress and gate decisions.
