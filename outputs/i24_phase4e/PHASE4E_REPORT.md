# Current continuation status

**COMPLETED:** Nov21 ingestion and source QC, 92 training windows, verified private
backups of data/QC/metadata and all recoverable partial work. **BLOCKED:** remaining
multi-date ingestion after HTTP401. **NOT EXECUTED:** Phase4E training, validation,
and independent forecasts. The user requested preservation and a new-task handoff.

See **Continuation: one complete source date** below for this task's results and
`continuation_backup_status.json` for public verification evidence. The original
handoff is retained below as historical context.

# Phase 4E — preservation and blocked research handoff

**Multi-date training: BLOCKED. Independent testing: BLOCKED.**
No accepted complete Phase4E dataset, optimizer updates, pilot results, validation
forecasts, or independent-test results were produced. Cross-date generalization
and microscopic-information advantage remain unanswered. The user will refresh
the secure source credential and resume in a new task.

## Recovery and access evidence

Work started from `bcdfeb3f3f8c5c508c4fe8a44d0cd92aeb145d56` on
`phase4-i24-continuous`. The existing Phase4D eight-part recovery archive and all
part hashes were verified; only four primary best/last checkpoint files were
restored. Historical prediction arrays were not opened. See `recovery_status.json`.
A preservation scan found 799 historical tracked files unchanged, except allowed
append-only `.gitignore` additions; see `historical_preservation.json`.

At 2026-10-11 18:51:58 UTC, the source API returned metadata HTTP200 and an exact
HTTP206 1KiB ZIP range; subsequent real multi-GB streaming yielded authentic Nov21
records. A later request at 19:21:46 UTC returned HTTP401 before archive restoration
bytes were written. Expiration versus revocation is undetermined; no error reason
was retained. This was not a missing-file diagnosis. See `source_access.json` and
`source_failure.json`. After the user's stop instruction, the token was not retried.
The independent working Drive connector completed all final private backups.

## Ingestion, failure handling and storage

The frozen assignment remains training Nov21/22/23, validation Nov25, independent
testing Nov28/29. Each date has four 600-second slots beginning 06:15, 07:15, 08:15,
09:15 America/Chicago. No outcome-driven date/window changes were made.

| Attempt | Native shards | Written native rows | Bytes | Acceptance |
|---|---:|---:|---:|---|
| data_v1, Nov21 | 119 | 5,990,819 | 350,770,503 | Incomplete, rejected |
| data_v2, Nov21 | 109 | 5,488,974 | 321,635,143 | Incomplete, rejected |

These are repeated reads of the same recording, not independent populations or
unique vehicle counts. No completed native/canonical dataset manifest or cache
exists. The other five dates were listed but not streamed. All six dates therefore
still require complete, verified ingestion. No partial shards can enter training.

The first attempt exposed a ZIP64 data-descriptor case: the streaming library
reports an unknown member size until completion. Our redundant comparison rejected
`None`; the fix now compares known sizes only, retaining the library's CRC and
size validation. A forced-ZIP64/nonseekable regression fixture passes. The second
process disappeared before completion without an application error or completion
receipt; its termination cause is unknown. A later local nested-archive restoration
attempt encountered HTTP401. These ingestion issues are separate from authentication.

The environment had an 8 GiB memory limit, roughly two CPU quota cores, no available
NVIDIA utility, and about 31.6 GB initial free storage. The implementation reserves
10 GB free disk. Storage did not cause a documented failure. No unbacked data was
deleted. The next namespace is data_v3; local Nov21 nested-archive recovery and
source-only QC helpers were added but have not completed real-source execution.

## Implementation and verification

The isolated Phase4E code implements per-date metadata and geometry, bounded exact
HTTP ranges, standard ZIP/streaming ZIP validation, frozen selections, native QC,
canonical preparation, train/validation caches, private preservation gates,
paired residual training, diffusion persistence, validation diagnostics, and
fail-closed test freezing. Historical model semantics were retained. Source
metadata stability and selected-member integrity are checked; whole outer ZIP
MD5 verification is not claimed. Local fixture coverage does not certify a
completed multi-date dataset or the full research pipeline.

The full regression suite passed **126 tests, 8 skipped** in the final 9.44-second suite run.
The final local backup restoration helper also completed an actual full archive
recovery drill, checking ordered parts, archive hash, safe member inventory and
every extracted file SHA256. No scientific training/evaluation was substituted
for missing data. No new trained weights or prediction arrays exist.

## Scientific status and limitations

The prospective protocol is `docs/I24_PHASE4E_PROTOCOL.md`. Full and aggregate-only
models retain matched 21,100 parameters, paired seeds101/202, from-scratch training,
a 1,600-update cap, minimum500, validation every100, patience6, and the frozen
macro energy objective. Those are planned settings, not achieved convergence.
Completed updates are **zero**. There are no learning curves or selected Phase4E
checkpoints. Train-only diffusion fitting and 64-particle calibration are unrun.

Completed-dataset source QC, split certification, population/support statistics,
regime coverage and exclusions are **BLOCKED**. Ordinary/physical/stochastic
persistence comparisons, speed/density/flow MAE and bias, CRPS, coverage, sharpness,
energy, diversity, matched microscopic-information effects, ablations, and per-day
or traffic-regime effects are **BLOCKED**, with no metrics to report.

Validation promotion is **NOT EVALUATED**, not a performance failure. Independent
forecast testing remains **BLOCKED and unopened**. The pipeline requires all four
paired learned variants to beat ordinary persistence on the preregistered dense
Nov25 mean5/10/20-second energy before freezing test artifacts. Even successful
Nov28/29 evaluation would give only two independent dates, below the five-group
confirmation gate; windows and paired seeds do not create independent days.

Prior Phase4D Nov21 validation improvements remain historical, separately scoped
results; they do not establish new-date accuracy or a microscopic-information
advantage. Field forecasts and emulated missingness do not establish a complete
joint traffic world model: state reconstruction, useful vehicle interactions,
arrival generation, long-horizon fidelity, traffic-wave dynamics, actions and
closed-loop autonomous-vehicle behavior remain unvalidated. Positivity and the
flow identity alone do not prove physical realism. The smallest next milestone is
completion of this frozen cross-date experiment; no new architecture campaign or
larger seed campaign was launched.

## Verified preservation and exact continuation

All **228 native shards**, their frozen selections, private source metadata/listing
and ingestion logs are in 27 owner-only connector backup parts. Every remote part
was downloaded and SHA256-verified, and their ordered reconstruction matched:

- Archive bytes: **665410723**.
- Archive SHA256: `cd317cb28ef6fb191eac9ef0dc6e26f4ef87589f4cd314b1bc490ba450e5d85e`.
- Private recovery manifest: `ORCHESTRA_PHASE4E_INCOMPLETE_RECOVERY_MANIFEST.json`.
- Manifest bytes: **63234**.
- Manifest SHA256: `f0fcefa1659a59ae982c3d01b2d8f4582523cfc7548a7d24269b3b9ac45d76f0`.

The manifest itself passed authenticated readback and owner-only checks. Its
private Drive link is supplied in the final handoff, not committed. Public evidence
is `partial_backup_status.json`; private per-part receipts stay under ignored
`work/phase4e/partial_preservation/`. This backup preserves incomplete artifacts;
it is not the completed-data receipt required to authorize training.

Follow `docs/I24_PHASE4E_RECOVERY.md` for precise download, hash verification,
non-overwriting restoration, refreshed-credential probe, all-six-date ingestion,
QC, complete-data backup, pilot/training, validation and gated testing commands.
Do not retry the old credential. Preserve v1/v2 and restart accepted preparation
in a fresh namespace. Keep all raw data, canonical/prediction arrays, weights,
private IDs and receipts out of Git.

Earlier pushed implementation commits are `59c7164` (preregistration/streaming)
and `656663e` (ZIP64 fix/preserved attempts). The final handoff commit includes
this report, recovery instructions, ingestion helpers, restore helper and public
status manifests; its exact SHA is supplied in chat after remote push verification.

---

# Continuation: one complete source date; multi-date experiment still BLOCKED

The preceding text is the preserved prior-task handoff. This continuation supersedes
its statement that no complete Phase4E dataset exists. It does **not** supersede
any historical scientific finding.

## COMPLETED

- Fetched `phase4-i24-continuous` and started from preservation commit
  `ad49250`, not the Phase4D starting commit. Restored all 27 prior Phase4E parts,
  checked their recorded SHA256 hashes, reconstructed the 665,410,723-byte archive,
  and verified every extracted member. All 228 prior native shards remain staged
  as incomplete artifacts; none was relabeled as an accepted training dataset.
- Verified the separate eight-part Phase4D archive and restored the four primary
  seed101/202 best/last checkpoints against their historical hashes. Historical
  prediction arrays were not opened.
- The refreshed source credential returned metadata HTTP200 and exact HTTP206
  bytes 0–1023 on Nov21. Its identity, size, MD5 and version matched the recovered
  metadata. Downloaded only the 5,958,100,914-byte Nov21 nested ZIP; its SHA256
  matched `dbabe7b3e665a7d48b27d89d9f55fa0190e2cbcd7f9953beae59a8663b6ec3b7`.
- Completed Nov21 in `outputs/i24_phase4e/data_v4/2022-11-21`: 830,366 source
  records, 280,265,948 source samples, complete inner CRC/SHA verification,
  historical nested-ZIP SHA verification, and unchanged remote metadata after
  ingestion. Whole outer archive MD5 verification is not claimed.
- Produced 92 canonical training windows, 23 in each frozen 06:15/07:15/08:15/09:15
  America/Chicago slot. Source-only QC passed: geometry, finite usable samples,
  0.2-second canonical timing, >=2 concurrent tracks, capacity without truncation,
  field identity, and frozen role checks. Maximum window population was 456.
  There were no excluded candidate windows, source duplicate IDs, shared slot IDs,
  or recorded native gaps over 0.5 seconds. Native median sampling interval was
  checked against 0.04 seconds. Track-boundary and invalid-sample exclusions remain
  explicitly recorded in native/QC manifests.
- Mean per-window supported-bin fraction was 0.917939. Scene-equal supported-field
  means were speed 14.394651 m/s, density 26.083976 veh/km/lane, and flow
  910.301478 veh/hour/lane. Mean fractions of supported bins were 0.461827 congested,
  0.267505 intermediate, and 0.270668 free-flow. These describe the reconstructed
  retained population, not unbiased roadway census or forecasting performance.

## Authentication and interrupted attempt

The token supported both source reads and a small private recovery-receipt upload
with authenticated readback. After completed Nov21 ingestion and a successful
source metadata check at 19:56:22 UTC on 2026-10-11, the first completed-data backup
upload returned HTTP401. No token-based request was retried. Expiration versus
revocation is undetermined. The environment exposes only GOOGLE_DRIVE_ACCESS_TOKEN;
no securely configured automatic refresh mechanism was available. The independent
connected Drive workflow was used for preservation after that failure.

A detached data_v3 job disappeared without an exit receipt, leaving 24 native
shards and a frozen selection (25 files total). The cgroup reported zero OOM kills;
the termination cause is unknown. This attempt is preserved separately and is not
accepted. Running data_v4 in a tool-managed foreground session completed the stream.
Use that execution pattern in continuation. No historic files were deleted for
space. The old v1/v2 shards were verified and retained, but lack completion/integrity
receipts and cannot replace a fully verified stream. The completed v4 dataset must
be reused; Nov21 does not need another archive download or ingestion pass.

## BLOCKED / NOT EXECUTED

| Stage | Status | Exact remaining work |
|---|---|---|
| Training/validation ingestion | BLOCKED | Nov22, Nov23, Nov25 not started in this continuation |
| Multi-date source QC and feature cache | BLOCKED | Require all three remaining assigned dates |
| Actual Phase4E training | NOT EXECUTED | Zero optimizer updates; pilot and seeds101/202 unrun |
| Fitted stochastic persistence | NOT EXECUTED | Train-only fit requires complete training cache |
| Nov25 validation and promotion | NOT EXECUTED | No forecasts, metrics, or model-selection decisions |
| Nov28/29 source ingestion | NOT EXECUTED | Both dates still need source-only processing |
| Independent forecast evaluation | NOT EXECUTED, UNOPENED | Requires frozen validation promotion and test protocol |

No energy scores, speed/density/flow MAEs, CRPS, forecast coverage/calibration,
forecast diversity, forecast physical-validity scores, missingness robustness,
or microscopic-information effects were produced in this continuation. Source
field identity is not a forecast physical-validity result. Falling losses cannot
be cited because no training ran. Cross-date generalization remains unanswered.
The primary information difference remains aggregate-only minus full energy;
there is no new evidence for microscopic benefit. Even eventual Nov28/29 success
would provide only two independent dates, below the preserved five-group gate.

## Continuation requirements

Refresh the source credential through secure environment settings and republish;
never paste credentials into chat. Restore the new completed-Nov21 private backup
and verify all hashes before reuse. Keep the earlier incomplete backup and Phase4D
checkpoint archive. Use `data_v4` consistently for remaining preparation,
verification, QC and cache commands. Process Nov22, Nov23 and Nov25 sequentially,
backing up each completed date. Then build and privately preserve the complete
train/validation cache before pilot and substantive paired training. Keep original
protocols, horizons, seeds, QC, slots, model capacity and promotion gate unchanged.
See the appended continuation commands in `docs/I24_PHASE4E_RECOVERY.md`.

The final pushed report commit SHA is supplied in the handoff and downloadable
report; within Git, identify the report-bearing commit using
`git log -1 --format=%H -- outputs/i24_phase4e/PHASE4E_REPORT.md`.

## Verified continuation backups

All 21 new backup parts passed authenticated SHA256 readback and owner-only checks.
The completed Nov21 bundle contains 403 files (native/canonical data and integrity
manifests); its 17-part archive is 414,119,241 bytes, SHA256
`e949bd3ed5f1b1bf1a98c5466591096b5625b5da9a408047bbd17c37a525e96a`.
The separate 14-file audit bundle preserves QC CSVs/audit, original private source
metadata/listing, access/recovery receipts and execution logs. The 25-file
interrupted v3 attempt is preserved in three parts. All three archive restorations
passed inventory and per-file SHA256 verification.

The consolidated private recovery index is
`ORCHESTRA_PHASE4E_CONTINUATION_RECOVERY_INDEX.json`, SHA256
`e4313e94b89beec0c8aa2fd6f1a06a2bf891e44b90a69be58587fc1cf52dd804`. Each manifest and the index also passed authenticated
readback and owner-only checks. Public hashes/counts are in
`continuation_backup_status.json`; private IDs/links remain outside Git.
Earlier 228 partial shards and four primary Phase4D checkpoints remain recoverable
through the reverified original manifests referenced by the private index.
No files were deleted and no completed Nov21 ingestion needs repeating.

All 799 historical tracked files passed preservation checks, with only previously
allowed append-only `.gitignore` additions. The user explicitly requested final
preservation and handoff; refreshed authorization will be supplied in a new task.

Final regression suite: **126 passed, 8 skipped** in 8.58 seconds with
`uv run --no-sync pytest -q --import-mode=importlib`. The generalized recovery helper
was also exercised by successful full restoration drills for the completed data,
audit supplement, and interrupted-attempt bundles.

---

# Continuation on 2026-10-11: Nov21 restored, Nov22 completed locally

## COMPLETED

- Checked out the required starting commit `9fd6f20754d36cdfae30d4567b06c0dfab081a9d` on `phase4-i24-continuous`. No November 21 ingestion was rerun.
- Downloaded the completed-Nov21 recovery index, 17-part data archive, and one-part audit supplement from private Drive. The index and both manifests matched the public SHA256 values in `continuation_backup_status.json`. All parts, combined archives, and 403 data files plus 14 audit files passed the restoration helper's size and SHA256 checks. Installed only absent or byte-identical files in `data_v4`; restored the original private source listing and metadata into ignored `work/phase4e/`.
- A new source probe (`source_recheck_03.json`) passed metadata comparison and exact HTTP 206, 1 KiB byte-range access for the Nov21 archive. The managed environment exposes an access-token secret and no configured outbound identity or renewable OAuth capability. No credential was printed or stored in Git.
- Ingested November 22 only, using the frozen four slots and `data_v4`. Preparation produced 92 train windows and nested-ZIP SHA256 `3f81287f188f7781622a8d362f9d87f2e6d381a0034e6be1299e1416e7d99748`. `verify_i24_phase4e_sources.py` accepted stable source metadata and selected-stream integrity. Whole outer archive MD5 is not claimed verified.
- Offline Nov22 source QC passed: 92 windows, 771,946 source records scanned, median native interval approximately 0.04 s, zero recorded source gaps over 0.5 s, zero excluded windows, and zero source-ID duplication. See `source_qc_nov22_continuation/audit.json` for field, geometry, population and selection details. No forecasting conclusions follow from source QC.
- The preservation script created a local Nov22 archive at ignored `work/phase4e/nov22_backup_private.tar.gz` (SHA256 `d9be9a7cf0b77532c702f8a69fd493777e76f751005c4c093444ff9a1ff5741a`) and inventory. These are local recovery aids only, not verified private remote backups.

## BLOCKED / NOT EXECUTED

| Stage | Status | Reason or remaining action |
|---|---|---|
| Nov22 private backup | BLOCKED | First upload returned HTTP 401; no retry, completion receipt, or remote SHA256 readback. Refresh the secure authorization and preserve `data_v4/2022-11-22` plus its QC and manifests before another date. |
| Nov23 and Nov25 ingestion | NOT EXECUTED | Stop-on-authentication-failure rule. |
| Combined four-date QC/cache and preservation | BLOCKED | Nov23/25 missing and Nov22 remote backup unverified. |
| Phase4E pilot and paired seed-101/202 training | NOT EXECUTED | Complete-data preservation gate not met; zero new optimizer updates. |
| Nov25 validation, promotion and freeze | NOT EXECUTED | No model results or promotion decision. |
| Nov28/29 ingestion and independent forecasts | NOT EXECUTED, UNOPENED | Validation gate not reached. |

No new macro energy, MAE, CRPS, calibration, physical-validity, robustness, or microscopic-information result exists. The primary aggregate-only minus full-model energy comparison remains unevaluated. No convergence or cross-date generalization claim is made.

## Exact continuation

Keep this checkout and the existing verified Nov21/Nov22 `data_v4` directories. Preserve the ignored local Nov22 archive and inventory until a private upload passes authenticated SHA256 readback. Refresh `GOOGLE_DRIVE_ACCESS_TOKEN` through secure environment configuration; probe with a new receipt filename. If the probe passes, rerun `scripts/preserve_i24_phase4e.py` for the **completed Nov22** directory and include its QC/metadata in a verified private backup. Do not reingest Nov21 or Nov22. Then run the documented per-date prepare, verify, QC and private-preserve steps for Nov23 and Nov25, stopping at the first authorization error. Only after combined QC, cache creation, and full dataset readback should the frozen training and validation protocol resume. The original Phase4D checkpoints and historical incomplete shards remain recoverable through their existing private manifests; they were not restored because they are not required for this data-stage continuation.

Regression suite after restoration and Nov22 QC: **126 passed, 8 skipped** with `uv run --no-sync pytest -q --import-mode=importlib`. This report and the public source-only QC/probe receipts are the only new files intended for Git; canonical data and local preservation archives stay private.
