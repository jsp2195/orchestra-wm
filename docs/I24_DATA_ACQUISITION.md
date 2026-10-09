# Official I-24 MOTION acquisition and source contract

**Current status: BLOCKED.** No genuine multivehicle MOTION source files are mounted in this workspace. Zero source files / zero source bytes. This environment's request to https://i24motion.org/data returned `Tunnel connection failed: 403 Forbidden`. The official tutorial explicitly says to download with account credentials. Accessible GitHub documentation is not trajectory-data access. See [access manifest](../outputs/i24/DATA_ACCESS_MANIFEST.json) for document commit IDs and hashes.

## Manual access and exact resume

1. Open https://i24motion.org/data in your browser. Create/sign into an authorized account using the site's current instructions and accept the public data-use agreement. This session has not seen the authenticated portal and cannot verify its current button labels.
2. Select the official **I24MOTION_PUBLIC_v1.0** multivehicle trajectory release. The official tutorial identifies `trajectory_data_by_date/11-22-2022/data_demo/INCEPTION.22-11-22.tutorial.json`. Prefer a modest official subset; do not substitute one-vehicle CAN/GPS, I24-3D annotations, or the postprocessing simulator/validation files.
3. Download/extract the selected authorized JSON-array files locally, preserving the release/run/session metadata. Check size and available disk first. Put them under `/path/to/I24MOTION_PUBLIC`; do not put credentials in Git.
4. Create `acquisition.json` in that directory, following the example below. Supply actual session identifiers, source URL and a declared straightened-road region. `x_origin_ft` is the upstream WB reference in the native roadway coordinates, **not** a latitude/longitude or TN state-plane coordinate. Do not infer exact geography from approximate mile markers. The implementation currently accepts only WB v1.0; other releases require an explicitly inspected adapter.
5. Run:

```sh
uv run python scripts/run_i24_pipeline.py --config configs/i24_pilot.yaml --data-root /path/to/I24MOTION_PUBLIC
```

```json
{
  "dataset": "I-24 MOTION",
  "release": "I24MOTION_PUBLIC_v1.0",
  "format": "json-array",
  "direction": -1,
  "terms_accepted": true,
  "source_url": "https://i24motion.org/data",
  "x_origin_ft": 327360,
  "s_min_m": 0,
  "s_max_m": 800,
  "files": [{"path": "YOUR_AUTHORIZED_SUBSET.json", "session": "ACTUAL_RECORDING_SESSION_ID"}]
}
```

This is a schema example, not supplied real data or verified site geometry. Acquisition metadata is a user-supplied provenance declaration; schema checks alone cannot authenticate the origin of arbitrary files. File bytes are SHA-256 hashed and saved with conversion metadata. The pilot caps conversion at two million source samples, 128 agents/window, 48 windows and 2 GB generated scratch. It fails rather than silently treating a truncated/incomplete population as the full release. Choose a genuinely bounded source subset; a full tutorial file may exceed this budget. At least three usable buffered partitions are required; a very short excerpt may be inadequate. Original raw data is never committed.

## Verified source semantics

Sources inspected: official I24M_documentation `v1.x_I-24MOTION_data_documentation.pdf` and data_tutorial `intro_trajectory_data.ipynb`, pinned by commit/hash in the manifest. The notebook reads a top-level JSON array with `ijson.items(file, 'item')`. Each vehicle record has `_id` (possibly MongoDB `$oid` wrapper), timestamp arrays, x_position/y_position arrays, scalar dimensions, direction and coarse_vehicle_class. It is not a per-frame CAN log. The adapter deliberately does not guess unrelated JSONL/nested release formats.

- Timestamps: corrected Unix seconds. Native interval is measured per record and reported; documentation examples are about 0.04 s. The model grid is 0.2 s (5 Hz).
- Roadway x/y are curvilinear feet. x increases EB; y is positive WB and negative EB. `_id` is anonymized within-session identity.
- Positions are **back-center**. WB travel-oriented center coordinate: `s = -(x_position - x_origin_ft) * 0.3048 + length_m/2`; `d = y_position * 0.3048`.
- These are roadway feet, not the US-survey-foot TN projected system. The adapter does not fabricate global XY or geographic coordinates. Official centerline spline control points would be required for a map/geodetic adapter.
- Official approximate WB lateral boundaries: [12,24,36,48,60] ft; use 0.5 map confidence. This is not surveyed lane precision. EB/ramp/lane-change boundary ambiguity is not silently guessed.
- Vehicle classes 0–5 are retained; unavailable class is −1. Native flags are retained as opaque internal flags, not calibrated sensor confidence. Fine class, road_segment_ids and configuration_id are documented as not implemented in v1.0 and are not treated as valid topology/sensor metadata.

The postprocessing-lite README uses a different illustrative lateral origin. It was not used to override the release-specific PDF.

## QC and causal conversion

Iterate vehicle records; flush zstd Parquet batches at 50,000 rows. Reject non-increasing timestamps, invalid shapes/dimensions, nonfinite positions; report gaps, rejected kinematics and lane jumps. Backward differences only; first two derivative samples are invalid. Resample with last-sample extrapolation for at most 0.25 s, never interpolation across cutoff. Thresholds flag/filter preprocessing artifacts, not model outputs: speed 65 m/s, lateral speed 10 m/s, acceleration 15 m/s², native gaps >0.5 s. Duplicate within-session IDs fail for explicit reconciliation rather than being concatenated blindly. Discontinuity proxies cannot certify tracking swaps; no raw imagery is available to validate identities.

Known official artifacts include missing poles, overpass occlusion, homography duplicates/gaps and packet loss. Offline stitching/reconciliation may exploit future imagery. That limitation persists even with a strictly causal model-input protocol. No measured camera outage histories are supplied here; sparse/blind/outage regimes are **EMULATED**.

Versioned canonical scenes preserve source hashes, sessions, SI units, existence vs detection vs observation vs padding, confidence provenance, static map, classes and IDs. Full-density future fields are targets. Empty observed bins mean no visible support, not proof of zero traffic. Reconstructed source density is coverage-dependent, not an unbiased vehicle census.

## Terms and attribution

The official public data-use agreement (19 September 2023) permits academic/commercial use and derivative sharing while retaining terms, prohibits re-identification and harm, requires attribution, and provides data as-is. Do not confuse software BSD licenses with data terms. Required citation:

Gloudemans, D., Wang, Y., Ji, J., Zachár, G., Barbour, W., Hall, E., Cebelak, M., Smith, L. and Work, D.B. (2023). **I-24 MOTION: An instrument for freeway traffic science.** Transportation Research Part C, 155, 104311.

Conversion restart behavior: a completed `conversion.json` records acquisition metadata, source hashes and Parquet hashes. A mismatch fails closed. An interrupted conversion without its final manifest rebuilds only generated `part-*.parquet` files. Model checkpoints are replaced atomically and include optimizer and RNG states. Scene budgets are allocated across sessions and partitions, and region bounds are applied before rejecting oversized agent rosters.

The current real adapter's track-confidence value is a constant 0.5 placeholder on usable samples, **not** a calibrated probability or an authentic per-camera confidence. The release does not supply the required per-sample sensor uncertainty here. Visibility/existence masks are separate. Replace this placeholder only when an inspected release/partner stream supplies defensible quality metadata; do not infer confidence from internal feasibility flags.
