# Phase 4D prospective residual experiment

This protocol precedes new model validation and any independent testing. Preserve
all earlier research. Source access returned HTTP401 authError once; do not retry
the failed credential. Continue with the existing 94 train and 23 validation
November21 windows. Never reopen the examined developmental-test partition for
tuning. Independent November28/29 outcomes remain unopened.

Frozen date roles remain train November21/22/23, validation November25 and test
November28/29. Multi-date transfer requires working source authorization through
the supported streaming path. Select clock-stratified intervals at 06:15, 07:15,
08:15 and 09:15 CST (ten minutes each where present), before model outcomes. Use
source-only usability criteria: >=2 simultaneously reconstructed tracks, no
population truncation, existing trajectory/coordinate QC and supported fields.
Report every rejected interval and speed/density regime; do not choose favorable
forecast outcomes. Process one archive/date at a time with the existing streaming
ingestor and preserve all archive/checkpoint bytes. Two test dates cannot satisfy
the five-independent-session/day confirmation gate.

## Anchored and capacity-matched macro experiment

Predict residual log-speed/density evolution from each bin's latest supported
historical field. Derive sample flow as 3.6kv. A 0.05 floor admits unknown/zero
bins; report ordinary persistence and identically floored physical persistence
separately so this detail cannot artificially favor the learned model. Field
scales remain [30,100,3000]. Use five seconds of observations and 100 autonomous
0.2-second steps. No future support, roster or field enters inference.

The graph consists of fixed roadway lane-bin tokens, not invented vehicles.
Both full and aggregate controls execute identical active encoders, graph messages,
GRUs and stochastic residual heads with 32 hidden dimensions. Aggregate inputs
include current/carried fields, support/age, position and a deterministic eight-
feature aggregate basis. Full additionally adds eight past-only observed-vehicle
statistics: instantaneous speed/variance, acceleration, lateral velocity, length,
within-bin position, observed history age and braking fraction. Zeroing these
additions makes identical-weight full and aggregate forecasts exactly equal.
This tests these microscopic summaries, not all possible vehicle information.
Both primary controls retain the same graph and recurrent mechanisms. Check all
scalar parameters for observed nonzero training gradients, not just allocation.
Identical computational capacity does not guarantee identical optimization difficulty.

Train independently initialized seeds101/202, each with paired models, common
scene/regime/minibatch/noise seeds, AdamW lr0.001, batch4 and four prior samples.
Use the same normalized proper macro energy objective averaged across all100
forecast times. There is no micro, posterior or conservation term in this control.
The separate historical-cohort micro experiment must not alter its objective.

Maximum800 updates/model/seed; evaluate all23 dense validation windows every50
updates using eight samples and the mean5/10/20-second energy. Select one common
checkpoint for the pair minimizing their mean validation objective. Both models
always receive equal updates. Reduce LR by half after two/four nonimproving checks.
Stop after >=250 updates and five checks without shared improvement >=0.0001.
Report the last-four-check range (stability heuristic <=max(0.002,2%mean)) separately
from no-improvement stopping; do not call an arbitrary cap or overfit plateau
proof of convergence. Wall budgets are operational and resumable with atomic
optimizer/RNG/source/config checkpoints. No held-out test selection.

Validation diagnostics at5/10/20s include energy, component MAE/CRPS/bias, empirical
90% coverage/width, diversity, negative fields, flow identity, historical regime,
and dense/sparse50/outage robustness on matched support. Use16 or more particles
for a separately labeled calibration sensitivity check if feasible; the main
comparison uses eight, consistent with saved prior controls. Report both seeds,
not the better seed. Validation improvements alone are developmental engineering
evidence, not independent-date generalization.

## Mechanism controls and promotion gate

Primary full-versus-aggregate changes only microscopic input additions. Secondary
graph-off substitutes self-messages for inter-bin messages with all layers intact;
memoryless resets history memory each observed frame; latent-off removes shared
four-dimensional innovations while retaining local stochastic diffusion. Evaluate
each change separately at the frozen checkpoint and label it an inference
distribution shift, not retrained or causal evidence. A smaller aggregate control
changes capacity only and must be separately labeled if trained. Never conflate
these controls with the primary information comparison.

The preregistered primary independent metric remains ten-second macro-only minus
full energy, equal scene weights and [30,100,3000] scales. Do not open independent
test outcomes unless validation decisions are frozen and the learned candidates
beat ordinary persistence on validation. Capacity matching, numerical finiteness,
nonzero stochasticity and positive fields alone do not pass this gate. Keep the
interactive demonstration unchanged in this phase unless a genuinely improved
model has sufficient evidence; no new visualization campaign.

## Additional strong controls, specified before their validation evaluation

Fit a two-parameter lognormal diffusion-persistence baseline on training only,
500 fixed Adam updates at lr0.01, common four-particle proper score and training
regimes. Its arithmetic mean reference stays at observed speed/density, with flow
derived from samples. This distinguishes adding forecast uncertainty from learning
context-dependent mean evolution. No validation calibration or test fitting.

The separate kinematic-residual trial uses seed101, identical 24-dimensional
history encoders and message/head layers for graph and independent models. Graph
uses a historical-cohort proximity matrix; independent substitutes self-messages,
with identical local stochastic innovations in both. Forecasts integrate learned
acceleration residuals with a fixed two-second exponential decay around CV.
This is a finite-cohort baseline, not an open-population interacting simulator.
Train up to500 updates, validate every25, minimum150 and patience4 (delta0.0001),
with a common pair-selected checkpoint and shared budgets. Micro energy uses
[100,3.6576,30,5] scales for position and velocity. No macro term or total-versus-
cohort conservation penalty. Compare supported ADE/FDE to CV at5/10/20s and report
survival counts, physical violations, probabilistic scores and calibration.

If resources permit, a 16-dimensional aggregate control will replay seed101's
primary minibatch and noise schedule, learning-rate changes, completed update
budget and selected checkpoint index. Only hidden capacity changes. Its checkpoint
index is inherited from the primary pair, not separately optimized on validation.
This is a capacity control, not another microscopic information comparison.
