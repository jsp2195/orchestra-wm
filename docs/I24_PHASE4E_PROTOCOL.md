# Phase 4E prospective multi-date generalization

Starting commit: bcdfeb3f3f8c5c508c4fe8a44d0cd92aeb145d56. This specification is
written before any Phase4E optimization or prediction outcomes. Earlier code,
models, arrays and findings remain immutable. No architecture campaign.

## Source and split lock

Train 2022-11-21/22/23; validation 2022-11-25; independent test 2022-11-28/29.
Each date uses the inherited 06:15, 07:15, 08:15, 09:15 America/Chicago slots,
600 seconds each. Keep the inherited WB 316800–320080 feet region, four lanes,
eight bins/lane, back-center to center transformation, 25 Hz native timestamps,
backward derivatives, <=0.08-second causal resampling to 5 Hz, five-second history,
20-second targets and trailing-one-second exposure fields. Empty support is unknown.
Exclude source tracks crossing a slot boundary, invalid samples and windows with
fewer than two concurrent usable tracks. Fail on duplicate source IDs, unexpected
schema/date/native period or a population exceeding 2048; never truncate vehicles.
Report missing/rejected slots; do not replace them. Source reconstruction is an
observed retained population, not a complete roadway census; it can use future imagery.
Historical Nov21 partitions were already examined. Their new training role follows
the predeclared date assignment, not an independent-test claim.

Use tested exact HTTP ranges for the outer ZIP, standard zipfile for its selected
nested member, and stream-unzip 0.0.99 for sequential inner JSON parsing. Verify
selected member CRCs at both layers and hash the complete nested ZIP and JSON.
For Nov21, additionally require the historical nested-ZIP SHA256. Do not claim a
full outer-archive MD5 check when unselected diagnostic images are skipped.
Compare remote metadata before/after ingestion. Keep 10,000,000,000 bytes free.
Do not retain multiple full source archives or delete historical artifacts.
Canonical/native files, feature and prediction arrays, and weights remain private.

## Matched model training

Reuse Phase4D ResidualFields unchanged: 32 hidden units, 21,100 active parameters,
roadway-bin graph, residual log speed/density anchored on the last supported state,
shared/local stochastic innovations, flow q=3.6kv. Full adds the inherited eight
causal microscopic statistics; aggregate-only receives the same aggregate basis.
Both use the same macro-only energy objective, all 100 rollout steps, physical
scales [30,100,3000], AdamW lr0.001, weight decay0.0001, batch4, four samples and
gradient clipping1. Paired seeds101/202 share initial weights, minibatches,
observation regimes, innovations, schedules and update counts. Train from scratch;
restored historical weights are preservation/reference artifacts, not new training.

Run a bounded numerical pilot before substantive training. Primary maximum1600
updates/model; minimum500; validate at update1 and every100 on all dense Nov25
windows, eight particles, mean5/10/20-second energy. Select one shared update for
each full/aggregate pair by their mean validation score. Improvement threshold
0.0001; halve LR at two/four nonimproving checks; stop after six checks without
improvement. Retain atomic model/optimizer/RNG/update/config/source checkpoints.
Report last-four-check range <=max(0.002,2%mean) as a heuristic, never convergence
proof. A completed cap does not imply sufficient training. Diagnose cap/instability
using train/validation only before any independent evaluation.

Fit unchanged two-parameter DiffusionPersistence on training only, 500 updates,
Adam0.01, batch4/four samples, fixed final checkpoint. Ordinary aggregate persistence
and identically floored physical persistence retain the same historical support.
Preserve new data privately with authenticated SHA256 readback before substantial
training. Two CPU threads, zero loader workers; no automatic GPU selection.

## Validation and independent-test lock

Score every model on identical targets/support at5/10/20 seconds, dense/sparse50/
outage histories, eight samples, common sample seeds7103+scene number. Report
energy, component MAE/bias/CRPS, empirical90% coverage/width, diversity, negatives,
flow identity, speeds>55 m/s and densities>200 veh/km/lane. These thresholds are
diagnostics, not automatic claims of physical impossibility. Scene-equal primary
means; provide bin support, window counts, date and slot denominators. Regimes use
historical supported-bin speed<10,10–20,>=20 m/s. Correlated windows are not days.
Missingness is simulated, not observed real sensor failure.

Separately label64-particle dense validation diagnostics of mean bias, spread,
point-error Monte Carlo effects and calibration; do not replace the primary score
or select checkpoints on them. Point/calibration shortcomings must remain visible.

Promotion requires each selected learned primary to beat ordinary aggregate
persistence on dense Nov25 mean5/10/20-second energy; also report the stochastic
baseline comparison. If promotion fails or prerequisites are incomplete, test
predictions stay unopened. Before opening them, write a versioned specification
with exact data/checkpoint/source hashes, seeds, horizons, samples and metrics.
After opening Nov28/29 outcomes, no tuning, checkpoint changes or new selection.

Primary information value remains aggregate-only minus full10-second macro energy,
paired within seed/date/window; positive favors microscopic input. Report both
seeds, per-day effects and equal-day pooled effects. With two independent test
days, provide the two day effects and their range, not a misleading window-level
confirmatory interval. The inherited >=5 independent-day/session gate and positive
cluster-bootstrap lower95% bound remain necessary for confirmation. No claim of
vehicle-interaction benefit follows from this fixed-roadway graph experiment.
