# Cavnue portability: schema interface, not a deployed integration

No Cavnue data, sensor API, contract or infrastructure was accessed. `CavnueAdapter` validates a future authorized `RoadsideScene` and intentionally raises `NotImplementedError` for raw ingestion until an actual partner format is supplied. Synthetic fixture examples validate the common schema but are never labeled Cavnue recordings. SH 130/I-94 are potential instrumented-corridor contexts, not demonstrated deployments of this code.

## RoadsideScene 1.0 field requirements

| Field | Requirement / interpretation |
|---|---|
| scene/session/provenance | Stable recording session, schema version, source file hashes, permission/publication reference |
| clock/time | Calibrated common timestamp in seconds, synchronization error and reference |
| position | Calibrated coordinate reference, metre units, lane-relative s/d; explicit transform to global XY if available |
| track identity | Anonymized session-local IDs; continuity and reassociation semantics; no re-identification |
| kinematics | Position/velocity/acceleration/heading with measured-vs-derived provenance; heading can be derived from atan2(vd,vs) |
| dimensions/class | Dimensions in metres, uncertainty, optional class and unknown value |
| masks | Existence, valid reconstructed sample/detection, currently visible observation, and padding are separate states |
| map | Lane centerlines/driving-space bounds, map confidence, direction and boundary context |
| coverage | Physical sensor footprints, fusion vs raw detection distinction, authentic source IDs only if supplied |
| quality | Confidence origin, quality flags, gaps, duplicate/swap provenance; no invented calibrated confidence |
| aggregates | Optional independently measured speed/density/flow with denominator and coverage; otherwise derive visible input fields only |

The current I24-specific projection is a straightened curvilinear road frame, not a geographic lane map. The core model uses SI coordinates and fixed lane/bin metadata; a partner adapter must supply the actual calibration rather than pretending this projection transfers unchanged.

## Questions for a partner

1. Are persistent anonymized tracks retained, or only aggregate detections? Are raw and fused versions separately available?
2. What are sampling rate, clock accuracy, gaps and continuity across sensors? How are identity resets/reacquisition represented?
3. What lane-map/calibration versions, lane accuracy and dimension/class uncertainty exist?
4. Are authentic per-source visibility/coverage and quality flags retained?
5. Which incident/breakdown annotations are independently verified, and at what temporal accuracy?
6. What research access, privacy, publication and derivative-sharing permissions apply?

## Transfer protocol (unexecuted on real partner data)

Validate schema, units and time; plot real geometry before training; audit coverage and IDs; freeze session-level splits. Run the same checkpoint under coordinate/map/coverage-shift controls and report physical-unit prediction, calibration and violations. Separate normalization/adapter bugs from corridor distribution shift. Retraining/fine-tuning, if authorized, uses training sessions only. Report zero-shot and adapted performance separately. No live traffic control or safety claim follows from schema compatibility.
