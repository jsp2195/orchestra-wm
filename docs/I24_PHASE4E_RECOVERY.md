# Phase 4E recovery handoff — BLOCKED

Multi-date training and independent testing are **BLOCKED**, not completed.
No Phase4E optimizer updates, pilot, validation forecasts, or independent forecasts ran.
The user instructed us not to retry the invalid source credential. Do not run any
source/API command below until a new task has a refreshed secure credential.
Never paste credentials into chat or Git. Preserve all incomplete namespaces.

## Recover the private partial artifacts

Use the private Drive manifest link delivered with this handoff, named
`ORCHESTRA_PHASE4E_INCOMPLETE_RECOVERY_MANIFEST.json`. Its 63,234 bytes have SHA256
`f0fcefa1659a59ae982c3d01b2d8f4582523cfc7548a7d24269b3b9ac45d76f0`.
All 27 parts and the manifest were downloaded through the authenticated connector
and checked byte-for-byte/SHA256. Permissions were owner-only. The reconstructed
665,410,723-byte archive has SHA256
`cd317cb28ef6fb191eac9ef0dc6e26f4ef87589f4cd314b1bc490ba450e5d85e`.
Private IDs remain in that manifest, never in Git.

1. Check out `phase4-i24-continuous` at the final commit SHA supplied in chat.
2. Download the manifest using the working Drive connector. Verify its SHA above.
3. Download each ordered manifest `parts` entry via its `url` or `file_id` into
   `work/phase4e/received/ORCHESTRA_PHASE4E_incomplete_ingestion.part000` through
   `part026`. The manifest's old `readback_path` values are session-local evidence,
   not usable locations in the new environment. Parts are at most 24 MiB.
4. Run this local-only restoration; it refuses an existing output directory:

```bash
python scripts/restore_i24_phase4e_backup.py \
  --manifest work/phase4e/received/ORCHESTRA_PHASE4E_INCOMPLETE_RECOVERY_MANIFEST.json \
  --parts work/phase4e/received --output work/phase4e/recovered
```

This verifies every part, ordered reconstruction, member inventory, and extracted
file hash. A full local recovery drill passed in the originating environment.
The archive contains all 228 native shards from data_v1/data_v2, frozen selections,
private source listing/metadata, and ingestion logs. These are incomplete repeated
Nov21 attempts, **not a valid training dataset**. Keep them in staging. Copy only
`source_listing_private.json` and `source_metadata_private.json` from
`work/phase4e/recovered/restored/work/phase4e/` to `work/phase4e/` when those targets
are absent; if present, compare and preserve both if different. No deletion needed.

The previous Phase4D private recovery manifest remains separately valid (SHA256
`80099dd88b4876f4c9fa7b44fbbcd7ce8c198c15d3144a6e13b237606a9979e5`).
Its eight-part archive SHA is
`7fe7b8261888be815ef6957aa8a81ede448ca5fa66735a2b743bead06119f66b`.
Four historical primary best/last checkpoints were recovered and verified locally;
no prior prediction arrays were opened. Phase4E training starts from scratch.

## Resume ingestion only after the credential refresh

```bash
uv sync --frozen
uv pip install -r configs/i24_phase4e_requirements.txt
uv run --no-sync python scripts/probe_i24_phase4e_source.py \
  --metadata work/phase4e/source_metadata_private.json \
  --output outputs/i24_phase4e/source_recheck_01.json
```

Require successful metadata identity/version comparison and exact HTTP206 1KiB
range verification. On 401 stop immediately; do not loop or troubleshoot the old
credential. Existing receipt paths are immutable; choose a fresh receipt name.

With adequate disk space (always retain 10 GB free), first recover just the Nov21
nested ZIP using the known historical SHA, not the outer archive and diagnostics:

```bash
uv run --no-sync python scripts/restore_i24_phase4e_nov21.py \
  --metadata work/phase4e/source_metadata_private.json
```

This new helper has not completed a real download. It preserves partial bytes;
resumption replays and checks the prefix rather than trusting a partial ZIP.
Run long ingestion in a durable job/session and retain its exit status and log.
The second previous ingestion process disappeared without a completion receipt;
the termination cause is unknown. Do not assume a foreground tool session is durable.

All six dates still need complete accepted ingestion. Keep the four frozen
America/Chicago 600-second slots at 06:15, 07:15, 08:15, 09:15 unchanged. Start with:

```bash
set -e
for source_date in 2022-11-21 2022-11-22 2022-11-23 2022-11-25; do
  uv run --no-sync python scripts/prepare_i24_phase4e.py \
    --listing work/phase4e/source_listing_private.json --dates "$source_date" \
    --output outputs/i24_phase4e/data_v3
  uv run --no-sync python scripts/verify_i24_phase4e_sources.py \
    --listing work/phase4e/source_listing_private.json --dates "$source_date" \
    --data outputs/i24_phase4e/data_v3
done
uv run --no-sync python scripts/qc_i24_phase4e.py \
  --dates 2022-11-21 2022-11-22 2022-11-23 2022-11-25 \
  --output outputs/i24_phase4e/source_qc_train_validation
uv run --no-sync python scripts/cache_i24_phase4e.py
```

Use a new namespace if data_v3 is incomplete; never merge shards across attempts.
If changing namespace, pass it to preparation, verification, QC and cache, and
update the test-cache source in the new freeze implementation before freezing.
Selected-member CRC/SHA and stable source metadata must pass. Whole outer archive
MD5 is not claimed verified. Check geometry, native timing, track IDs/splits,
population, support, exclusions and regimes in completed QC before training.
Process each date separately; a short-lived token may not cover all six archives.
Nov22/23/25/28/29 have not been streamed. No date is accepted yet.

## Remaining scientific work (prospective, not executed)

Read `docs/I24_PHASE4E_PROTOCOL.md`. Before substantive training, privately back up
all completed train/validation data and cache, verify remote readback, and create
the required `work/phase4e/data_backup_private.json` receipt. The partial recovery
manifest does not meet this training gate. With refreshed authorized write access:

```bash
uv run --no-sync python scripts/preserve_i24_phase4e.py \
  --roots outputs/i24_phase4e/data_v3/2022-11-21 outputs/i24_phase4e/data_v3/2022-11-22 \
  outputs/i24_phase4e/data_v3/2022-11-23 outputs/i24_phase4e/data_v3/2022-11-25 \
  outputs/i24_phase4e/cache_v1 --receipt work/phase4e/data_backup_private.json \
  --label train_validation --cache outputs/i24_phase4e/cache_v1
uv run --no-sync python scripts/pilot_i24_phase4e.py
uv run --no-sync python scripts/train_i24_phase4e.py --seed 101 \
  --output outputs/i24_phase4e/residual_seed101 --max-updates 1600 --wall-seconds 3600
uv run --no-sync python scripts/train_i24_phase4e.py --seed 202 \
  --output outputs/i24_phase4e/residual_seed202 --max-updates 1600 --wall-seconds 3600
uv run --no-sync python scripts/fit_i24_phase4e_diffusion.py
uv run --no-sync python scripts/evaluate_i24_phase4e.py \
  --runs outputs/i24_phase4e/residual_seed101 outputs/i24_phase4e/residual_seed202 \
  --output outputs/i24_phase4e/validation
uv run --no-sync python scripts/calibrate_i24_phase4e.py \
  --run outputs/i24_phase4e/residual_seed101 --output outputs/i24_phase4e/calibration64_seed101
uv run --no-sync python scripts/calibrate_i24_phase4e.py \
  --run outputs/i24_phase4e/residual_seed202 --output outputs/i24_phase4e/calibration64_seed202
```

Review pilot timing/memory before the full bounded runs. Check completion receipts,
convergence, matched active capacity, baseline scores and calibration; resume
interrupted training with identical configuration instead of declaring completion.
Prepare and verify Nov28/29 using the same per-date commands above, then run source
QC over all six dates. This is source QC only; keep independent forecasts unopened.
Back up these new complete test datasets as well. Only after validation promotion:

```bash
uv run --no-sync python scripts/freeze_i24_phase4e.py \
  --runs outputs/i24_phase4e/residual_seed101 outputs/i24_phase4e/residual_seed202
uv run --no-sync python scripts/evaluate_i24_phase4e.py \
  --runs outputs/i24_phase4e/residual_seed101 outputs/i24_phase4e/residual_seed202 \
  --cache outputs/i24_phase4e/test_cache_v1 --split test \
  --freeze outputs/i24_phase4e/independent_test_protocol.json \
  --output outputs/i24_phase4e/independent_test
```

If validation fails, stop before independent evaluation. Report paired day effects
and regimes, not IID window confidence. Two independent dates cannot satisfy the
five-independent-group confirmation gate. Preserve and readback-verify all new
checkpoints, optimizer/RNG states, predictions and manifests before ending the next
task. Update the report and push only safe code and public summaries.

---

# Continuation after ad49250: reuse completed Nov21

The preceding instructions describe the original all-incomplete handoff. A later
continuation completed and source-QC-verified Nov21 in `data_v4`: 92 windows across
all four frozen slots. **Do not rerun Nov21 ingestion.** The source token returned
HTTP401 during its first completed-data backup upload, after a successful source
metadata comparison. No token retry occurred. Nov22/23/25 and Nov28/29 remain
unprocessed. Multi-date training and all Phase4E forecasts remain unexecuted.

The new private recovery manifest is named
`ORCHESTRA_PHASE4E_COMPLETED_NOV21_RECOVERY_MANIFEST.json`. Its exact SHA256,
archive hash, byte counts and verification status are in
`outputs/i24_phase4e/continuation_backup_status.json`; its private link is supplied
in the handoff. The manifest includes the private source listing/metadata and
references to the earlier recovery manifests. It contains no credentials. The
prior 228 incomplete shards remain in their original verified backup, and the
additional interrupted v3 attempt has its own verified private backup.

1. Download the new manifest through the connected Drive app and compare its SHA256
   with the public status. Download its 17 ordered parts using the recorded private
   IDs/URLs; compare each hash. Retain names
   `ORCHESTRA_PHASE4E_COMPLETED_NOV21.part000` through `part016`.
2. Restore into a fresh staging directory:

```bash
python scripts/restore_i24_phase4e_backup.py \
  --manifest work/phase4e/received/ORCHESTRA_PHASE4E_COMPLETED_NOV21_RECOVERY_MANIFEST.json \
  --parts work/phase4e/received \
  --part-prefix ORCHESTRA_PHASE4E_COMPLETED_NOV21 \
  --output work/phase4e/completed_nov21_recovered
```

3. Copy the verified staged `outputs/i24_phase4e/data_v4/2022-11-21` files into the
   matching checkout path. Existing files must match recorded hashes; never
   overwrite differing bytes. Restore `source_listing_private.json` and
   `source_metadata_private.json` under ignored `work/phase4e/` from the manifest's
   `source_listing` and `source_metadata` JSON objects. Restore historical Phase4D
   checkpoints and earlier partial artifacts as required using their original
   manifests; do not open historical forecast arrays.
4. The nested Nov21 archive remains verified in this worker but is not duplicated
   in the new private compact backup; its existing private original source remains.
   No archive download is needed to reuse the completed native/canonical dataset.
5. Refresh authorization through secure environment settings and republish. Only
   after confirmed refreshed authorization, use the existing probe with a **new**
   output receipt name. Stop at the first401, with no retries. No automatic refresh
   mechanism was configured in this continuation.
6. Use tool-managed foreground sessions with captured log/exit status. A detached
   job disappeared without an exit receipt; the cause is unknown. Do not rely on
   `nohup` alone for this environment's job lifetime.

```bash
uv sync --frozen
uv pip install -r configs/i24_phase4e_requirements.txt
uv run --no-sync python scripts/probe_i24_phase4e_source.py \
  --metadata work/phase4e/source_metadata_private.json \
  --output outputs/i24_phase4e/source_recheck_03.json

# Execute one assigned date at a time; keep 10GB free and back up before the next.
# Repeat with 2022-11-23 and 2022-11-25 only after the prior date is verified.
uv run --no-sync python scripts/prepare_i24_phase4e.py \
  --listing work/phase4e/source_listing_private.json --dates 2022-11-22 \
  --output outputs/i24_phase4e/data_v4
uv run --no-sync python scripts/verify_i24_phase4e_sources.py \
  --listing work/phase4e/source_listing_private.json --dates 2022-11-22 \
  --data outputs/i24_phase4e/data_v4
uv run --no-sync python scripts/preserve_i24_phase4e.py \
  --roots outputs/i24_phase4e/data_v4/2022-11-22 \
  --receipt work/phase4e/nov22_backup_private.json --label completed_nov22

# Only once all four assigned train/validation dates exist and pass source checks:
uv run --no-sync python scripts/qc_i24_phase4e.py \
  --data outputs/i24_phase4e/data_v4 \
  --dates 2022-11-21 2022-11-22 2022-11-23 2022-11-25 \
  --output outputs/i24_phase4e/source_qc_train_validation
uv run --no-sync python scripts/cache_i24_phase4e.py \
  --data outputs/i24_phase4e/data_v4 --output outputs/i24_phase4e/cache_v1
uv run --no-sync python scripts/preserve_i24_phase4e.py \
  --roots outputs/i24_phase4e/data_v4/2022-11-21 outputs/i24_phase4e/data_v4/2022-11-22 \
  outputs/i24_phase4e/data_v4/2022-11-23 outputs/i24_phase4e/data_v4/2022-11-25 \
  outputs/i24_phase4e/cache_v1 --receipt work/phase4e/data_backup_private.json \
  --label train_validation --cache outputs/i24_phase4e/cache_v1
```

Then run the original bounded numerical pilot, seed101/202 paired training,
train-only diffusion fit, Nov25 validation and 64-particle diagnostics from above.
The new one-date backup is **not** the complete-cache training receipt. Preserve
all checkpoints, optimizer/RNG and forecasts with authenticated remote readback.
Only after validation promotion and frozen decisions may independent forecasts run.
The prospective freeze script currently names `data_v3` for test-cache preparation;
parameterize its source path to verified `data_v4` **before** freezing or opening
any test outcomes. Do not change metrics, roles, slots, budgets or scientific gates.
If a new date is interrupted, preserve its partial namespace; reuse all completed
dates and avoid treating incomplete native shards as accepted data.

## Restore the QC/metadata supplement too

Start from the private `ORCHESTRA_PHASE4E_CONTINUATION_RECOVERY_INDEX.json` linked
in the handoff. Verify its SHA against `continuation_backup_status.json`. It links
the completed-Nov21 manifest, interrupted-v3 manifest, audit manifest, and original
Phase4E/Phase4D recovery manifests. Download the audit manifest and its single part:

```bash
python scripts/restore_i24_phase4e_backup.py \
  --manifest work/phase4e/received/ORCHESTRA_PHASE4E_NOV21_AUDIT_RECOVERY_MANIFEST.json \
  --parts work/phase4e/received --part-prefix ORCHESTRA_PHASE4E_NOV21_AUDIT \
  --output work/phase4e/nov21_audit_recovered
```

This restores the QC CSVs, audit JSON, original private source metadata/listing,
continuation integrity/access receipts and logs into isolated staging. Copy only
absent or byte-identical files to their recorded relative paths. The public
`source_qc_nov21_continuation` files are also versioned in Git. If desired, rerun
**QC only** in a fresh output namespace after restoring data; do not rerun ingestion.
The interrupted-v3 bundle uses prefix `ORCHESTRA_PHASE4E_INTERRUPTED_V3` and must
remain staged as incomplete, alongside the old incomplete v1/v2 artifacts.

## Exact staged installation and remaining-date loop

After the two Nov21 restore commands above pass, install only missing or identical
verified files; abort on any differing existing file:

```bash
python - <<'PY_RESTORE'
from pathlib import Path
import hashlib, shutil
for staging in [Path('work/phase4e/completed_nov21_recovered/restored'),
                Path('work/phase4e/nov21_audit_recovered/restored')]:
    assert staging.is_dir()
    for source in staging.rglob('*'):
        if not source.is_file():
            continue
        target = Path(source.relative_to(staging))
        if target.exists():
            with source.open('rb') as a, target.open('rb') as b:
                assert hashlib.file_digest(a, 'sha256').digest() == hashlib.file_digest(b, 'sha256').digest(), target
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
PY_RESTORE

# Only after the refreshed-authorization probe passes:
set -e
for source_date in 2022-11-22 2022-11-23 2022-11-25; do
  uv run --no-sync python scripts/prepare_i24_phase4e.py \
    --listing work/phase4e/source_listing_private.json --dates "$source_date" \
    --output outputs/i24_phase4e/data_v4
  uv run --no-sync python scripts/verify_i24_phase4e_sources.py \
    --listing work/phase4e/source_listing_private.json --dates "$source_date" \
    --data outputs/i24_phase4e/data_v4
  uv run --no-sync python scripts/preserve_i24_phase4e.py \
    --roots "outputs/i24_phase4e/data_v4/$source_date" \
    --receipt "work/phase4e/completed_${source_date}_backup_private.json" \
    --label "completed_${source_date}"
done
```

Stop on the first source or backup error. Completed dates and their receipts stay
valid; do not repeat their ingestion. Continue with the all-four-date QC/cache
commands above only when the three remaining assigned dates are complete.
