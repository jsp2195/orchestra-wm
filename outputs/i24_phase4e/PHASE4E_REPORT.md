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
