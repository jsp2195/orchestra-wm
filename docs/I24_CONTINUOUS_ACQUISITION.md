# Original continuous I24: access gate and secure import

Status: **BLOCKED**. The resumed session received HTTP **503 Service Unavailable** from both https://i24motion.org/access_data and https://i24motion.org/data. Earlier probes received proxy tunnel 403. `i24motion.org` is saved in the cloud environment's custom allowlist, retaining Harvard/GitHub/package destinations. A saved draft is not proof of published or active network configuration. Review/save the existing environment settings and publish the environment if needed; no credentials were requested or used. No authenticated listing, redirect download host, filename or size could be verified. Do not invent these details.

Rotate the previously exposed account password. Never paste passwords, cookies, authorization headers or signed URLs into this repository or chat.

## Required user action

1. In your own authenticated browser, visit the official access page above, log in with a newly rotated credential and accept the applicable agreement. Codex does not inherit that browser session.
2. Select **original I24MOTION_PUBLIC_v1.0 westbound, corrected-timestamp, multivehicle JSON-array trajectories**, preferably a continuous congested 10–30 minute recording. Download only if the actual listed size fits the **3,000,000,000-byte** pilot budget; keep at least **10,000,000,000 bytes** free. If no suitable subset exists, provide the official listing's non-secret filename/size/release details before attempting a larger transfer. No larger transfer is authorized by this implementation.
3. **Actual current filename and size: unknown, blocked by inaccessible listing.** The official tutorial documents `trajectory_data_by_date/11-22-2022/data_demo/INCEPTION.22-11-22.tutorial.json`, but its current availability, size and continuous coverage have not been verified. It is an example to locate, not a promise of a usable download. Do not substitute Harvard TFRecords, I24-3D, CAN logs or fixtures.
4. Transfer the authorized extracted JSON using your workspace's approved file-upload/mount mechanism into `/workspace/orchestra-wm/data/i24_continuous/raw/`, or mount a permitted read-only data directory and pass it as `--data-root`. This chat has no verified large-file upload/mount UI, so no button sequence is asserted. Never transfer your browser profile/cookies. Do not upload restricted data to a third-party host. Preserve the supplied terms locally.
5. Create `acquisition.json` beside the JSON files using actual source metadata, local byte counts and SHA256 from `sha256sum FILE` and `stat -c %s FILE`. Fields below are a schema description, not a fabricated acquisition:

```text
dataset: "I-24 MOTION"
release: "I24MOTION_PUBLIC_v1.0"
format: "json-array"
direction: -1
terms_accepted: true
corrected_timestamps: true
source_url: "https://i24motion.org/access_data"
x_origin_ft: actual upstream roadway-coordinate reference (number)
session: actual source recording/session identifier (string)
recording_date: actual YYYY-MM-DD (string)
start_unix_s / end_unix_s: actual corrected interval endpoints (numbers)
files: [{"path": "actual-relative-name.json", "bytes": actual_integer, "sha256": "actual_hex_digest"}]
```

No signed URL, credential, arbitrary metadata fields or future target data belongs in this manifest. The adapter accepts only the documented WB format; a different release requires source inspection first.

## Executable resume command

```sh
cd /workspace/orchestra-wm
uv run python scripts/run_i24_continuous_pipeline.py \
  --config configs/i24_continuous_pilot.yaml \
  --data-root /workspace/orchestra-wm/data/i24_continuous/raw \
  --report outputs/i24_continuous/import_after_mount.json
```

This implemented command performs bounded, read-only hash/schema/chronology QC, **not the unimplemented continuous training campaign**. It exits 2 while downstream science is gated, including when local schema QC passes but provenance remains unreviewed. It never copies/deletes original files or trains on fixtures. Existing reports are immutable; use a distinct `--report` for changed inputs. Per-second multivehicle support is only an initial coverage check, not proof of uninterrupted frame-level reconstruction or origin. Continue implementation after review of authentic mounted records: streaming conversion, subsecond QC, corridor-aware macro exposure, leakage-safe splits, then real smoke/pilot. No automatic fallback to the old fixture pipeline.

The source hash checks authenticate consistency with the local declaration, not publisher identity. The caller must establish authorized official provenance. Parsing is record-streamed but one vehicle record is materialized; oversized individual records need separate review. Scratch output is metadata only at this gate; 6 GB scratch is reserved for later conversion, not exercised now.

## User-supplied private Drive route (subsequent session)

The user has supplied `11-21-2022.zip` (6,248,968,002 bytes) and `auxiliary_information.zip` (4,917,485 bytes) in private Drive. These are user-declared names/sizes, not yet cloud-verified. See [secure Drive transfer and authorization](I24_CONTINUOUS_DRIVE_TRANSFER.md). This new route does not depend on the unavailable official portal listing. No raw bytes have arrived; source chronology remains unvalidated.
