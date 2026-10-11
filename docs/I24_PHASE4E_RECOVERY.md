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
