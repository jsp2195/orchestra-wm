# I24-MSD data card — executed real-data pilot

## Identity, access and provenance

**REAL_ACCESSED.** Harvard Dataverse DOI [10.7910/DVN/DQOWQI](https://doi.org/10.7910/DVN/DQOWQI), dataset ID 11720287, released version **1.1**, version ID 499823. The API lists 12 public ZIP files. This campaign downloaded exactly **one**, file **11835291**, `data_2022_11_24.zip`, **1,369,499,074 bytes**. Published MD5 `224ed625f7454776cceca807fbb916cb` matches. SHA-256 `42604d863de9d30a86c0fdd4c00b48d276fae914d82a2bcc1e3d8cfc0b29964f`.

The verified ZIP is retained at `data/i24_msd/raw/11835291-data_2022_11_24.zip`, ignored by Git. It has 325 TFRecord members, 4,756,225,464 uncompressed bytes. Only members **00000, 00162, 00324** were extracted, totaling **26,768,195 bytes**, under `data/i24_msd/raw/pilot_members/`. No second download was performed after verification. The public API redirects normally to `dvn-cloud-iqss.s3.amazonaws.com`; initial proxy denials were resolved by adding the DOI/Harvard/author/S3 hosts to the environment configuration. Signed redirect URLs/headers are not committed.

[DOWNLOAD_MANIFEST](../outputs/i24_msd/DOWNLOAD_MANIFEST.json) distinguishes visible metadata from actual downloaded bytes; [extraction manifest](../outputs/i24_msd/extraction_manifest.json) identifies selected members and hashes. Other date files remain metadata-only. The filename gives a source date, not independently verified recording/session IDs. No official train/validation/test partition is specified in the acquired archive/API.

## Verified native schema

Authentic payloads are **Waymo-compatible Scenario protocol buffers inside TFRecord framing**, not `tf.train.Example`, not continuous `I24MOTION_PUBLIC_v1.0` JSON, not Waymo vehicle-collected data. Each inspected record has CRC32C-verified length and payload, 91 timestamps at ~0.0999999 s, `current_time_index=10`, and static map features. Payload fields decode with zero unknown bytes against the pinned official schema. [Schema provenance](../orchestra_wm/i24_msd/protos/UPSTREAM.json) pins Waymo commit `99a4cb3ff07e2fe06c2ce73da001f850f628e45a`; Apache-2.0 definitions/license are preserved. Only unused camera/lidar definitions are omitted. Generated Python modules require the existing locked protobuf runtime; no TensorFlow dependency.

The three shards contain **349 records**. The adapter retains **182 scenes with at least two vehicles visible somewhere in the native observation history**. Single-agent or history-empty records are not the target multivehicle benchmark. Inspected scenes have variable numbers of vehicles (up to 10 in this sample), unique source track IDs (including signed IDs), per-step validity, center XY, velocity XY, heading, dimensions, object types, lane centers, road lines and road edges. The source description permits up to 32, but this pilot does not claim to have sampled all 32-agent cases.

The native observation context is **0–1 s**, with **8 s future**, total **9 s**. Causal factor-two decimation gives a 5 Hz model grid: six history states including cutoff, followed by up to 40 future states. Evaluated physical horizons are **1, 2, 4, 6 s**. No independent scenes are concatenated.

## Units, coordinates and masks

The official protobuf contract specifies center positions/dimensions in metres, velocities in m/s, heading in radians. Actual vehicle dimensions and ~3.6 m lane spacing are consistent with SI; position/velocity difference checks are saved in `data_qc.csv`. Native large longitudinal coordinates are retained in provenance/transform, not mislabeled latitude/longitude. The model uses `s = sign(median historical vx) * (native x - median historical x)` and `d = native y`. Origin and travel sign use history only; positions/velocities transform consistently. Native heading and original static polylines are preserved. Accelerations use backward differences at valid adjacent native states. Missing kinematics are not invented. Source dimensions are repeated/default-like in these clips; do not claim measured per-vehicle size fidelity.

Source validity is the usable reconstructed-track/existence mask, not a directly observed camera detection probability. Observation dropout is **EMULATED**. Unknown confidence is represented by zero plus `confidence_available=false`, not a calibrated probability. Invisibility is separate from existence and padding. Historically unseen future entrants never initialize the model; exits/gaps only mask scoring targets, and future validity never controls rollout evolution. The fixed cohort persists in imagination; learned entrance/exit hazards and open-system population generation are **UNTESTED**.

Static lane medians approximate the supplied lane geometry for neighbor relations. The numerical model extent (±1000 m about history origin) is not a real road boundary. Population lane density/flow is **N/A** because curated scenarios are not verified vehicle censuses. Macro heads are inherited but their inputs, feedback and losses are disabled for this pilot. Road-edge violation scoring uses only the common longitudinal support of the two actual finite road-edge polylines, never extrapolated boundaries; report coverage and denominator. Velocity direction is a heading proxy for generated boxes, not a separately learned yaw head.

## Splits and limitations

Build connected components of shared source track IDs across all 182 eligible scenes before allocation. Deterministic seed 7301 partitions groups 60/20/20, then caps the pilot to **57 train / 19 validation / 19 test scenes**. Record hashes, scenario IDs, source files and groups are immutable in `split_manifest.json`. Supplied identities never cross splits; duplicate record hashes and identical observed geometries are checked. Tests mutate future labels and verify unchanged model observations/predictions.

Relative timestamps and anonymized IDs do not prove independent origins or exclude every alias/near-duplicate. We cannot certify independent recording sessions or independent-day OOD from this single date. The 4-second primary comparison has 18 eligible test scenes in six supplied-track groups; the nineteenth has no valid final target at that horizon. Group-bootstrap intervals are provisional. Do not treat these short clips as 18 independent recordings. Source reconstruction may use offline information even though our inputs are causal.

## Terms, attribution and transfer

The release permits academic/commercial use and derivatives with retained terms and attribution; it prohibits re-identification, harm and privacy-law violations and is provided as-is. Preserve the complete fields in `dataverse_metadata.json` (terms, restrictions, conditions, citation requirements and disclaimer), plus the authors' official page terms. Required citations:

- Jayawardana, V., Tang, C., Ji, J., Philion, J., Peng, X. and Wu, C. (2025). **Noise-Aware Generative Microscopic Traffic Simulation.** arXiv:2508.07453.
- Gloudemans, D. et al. (2023). **I-24 MOTION: An instrument for freeway traffic science.** Transportation Research Part C 155, 104311.

The original paper already studies generative I24-MSD simulation using SMART/noise-aware objectives and IDM/CV comparisons. Our pilot does not establish novelty or a comparable benchmark win. CavnueAdapter remains the preserved authorization/schema interface: live use would require permissioned detections/tracks, synchronized clocks, coordinate calibration, actual sensor footprint and lane geometry, stable identities, confidence/quality flags, and privacy/use rights. Curated clips do not validate live Cavnue hardware, causal AV control, traffic-wave emergence or cross-corridor transfer.
