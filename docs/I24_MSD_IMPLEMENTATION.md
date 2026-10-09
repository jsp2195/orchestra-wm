# I24-MSD execution and resumption

Milestones: repository preservation **COMPLETE**; public metadata/download/checksum **COMPLETE**; authentic multivehicle Scenario decoding **COMPLETE**; short-horizon preregistration and supplied-ID-disjoint splits **COMPLETE**; real CPU pilot **COMPLETE**; autonomous evaluation and checkpoint-backed demo **COMPLETE**. Independent-session generalization, calibrated uncertainty, macro census reconstruction, learned arrivals/exits, long-horizon waves and Cavnue deployment are **NOT ESTABLISHED**.

Use the existing checkout and `phase3-i24-msd` branch. Do not redownload the verified ZIP or delete extracted shards. `uv sync --frozen` installs the locked runtime. No running service is needed for training; the HTML is self-contained.

```sh
uv run pytest -q
uv run python scripts/run_i24_msd_pipeline.py --config configs/i24_msd_pilot.yaml --data-root data/i24_msd/raw
uv run python scripts/run_i24_msd_pipeline.py --config configs/i24_msd_smoke.yaml --data-root data/i24_msd/raw --metadata-json outputs/i24_msd/dataverse_metadata.json
```

The pilot command rehashes existing raw files, verifies extracted-member hashes and immutable splits, and resumes exact fingerprint-matching checkpoints. With no local source, it fetches official metadata and the configured real file ID under the 1.5 GB transfer / 2 GiB disk budgets, then extracts only the three selected shards. Byte completion never implies semantic scene validation. The smoke command trains a separate 20-update engineering check; its checkpoint is not used for pilot metrics. Research is gated and never auto-run. The current pipeline deliberately supports this inspected 2022-11-24 archive/schema, not arbitrary uninspected release variants.

Stages and budgets:

```sh
uv run python scripts/run_i24_msd_pipeline.py --stage discover --config configs/i24_msd_pilot.yaml
uv run python scripts/run_i24_msd_pipeline.py --stage acquire --config configs/i24_msd_pilot.yaml --file-id 11835291 --max-download-bytes 1500000000 --max-disk-bytes 2147483648
uv run python scripts/run_i24_msd_pipeline.py --stage inspect --config configs/i24_msd_pilot.yaml
```

Interrupted file transfers restart that incomplete file; no HTTP Range support is assumed. Completed files are rehashed and reused. HTTP 403 is not retried indefinitely; 429/5xx use bounded backoff. HTML/login/error bodies, restricted files, checksum/size failures and unsafe archive paths/links are rejected. The raw ZIP plus selected extracted files use about 1.397 GB. Generic full `--extract` will reject this archive under 2 GiB because all members expand to 4.76 GB; the pilot's selective extraction is the intended path.

If network settings change, allow `doi.org`, `dataverse.harvard.edu`, `ct135.github.io`, and Harvard's verified `dvn-cloud-iqss.s3.amazonaws.com` redirect host. Retain existing package/GitHub destinations. No API token was needed. Manual authorized import remains available: obtain the exact ZIP and Native API metadata JSON for DOI 10.7910/DVN/DQOWQI, preserve original filename and terms, then run:

```sh
uv run python scripts/run_i24_msd_pipeline.py --stage acquire --config configs/i24_msd_pilot.yaml --metadata-json /path/to/dataverse_metadata.json --import-dir /path/to/downloads --file-id 11835291 --data-root data/i24_msd/raw
uv run python scripts/run_i24_msd_pipeline.py --config configs/i24_msd_pilot.yaml --data-root data/i24_msd/raw
```

Metadata exported by a user is a provenance declaration, not an independent authentication of arbitrary bytes; exact public checksum and release identity remain mandatory.

The 500-step pilot uses the preserved Phase-3 recurrent graph/prior model through an MSD subclass, with unknown-population macro feedback disabled. Training uses autonomous prior states plus a training-only posterior auxiliary likelihood/KL; future targets never become recurrent states. Validation loss includes that auxiliary objective and is not itself a held-out forecast metric. A sampled sparse/outage mask with no historically visible agent falls back to dense conditioning during training. Episodes with no future training targets are excluded from the training sampler; evaluation keeps explicit missing-target denominators. One initial one-step implementation failure is archived under ignored `checkpoints/failed_mask_attempt/`; all final models use the same fixed seed 101 and 500-update budget, with no test-selected checkpoint.

Scientific metrics are regenerated from exact final checkpoints; inference caches/arrays and checkpoints stay ignored. Published HTML embeds selected exact evaluated arrays, hashes and source geometry; `verify_demo` and a regeneration test prevent ground-truth replay. Downstream reporting is namespaced here and never calls the original continuous-I24 reporter, which has incompatible 10-second primary horizons.
