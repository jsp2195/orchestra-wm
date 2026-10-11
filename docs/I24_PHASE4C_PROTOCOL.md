# Phase 4C prospective execution protocol

The private original and expanded recovery bundles have authenticated SHA256
readback receipts. This supersedes the historical Phase 4B backup blocker without
rewriting its report. The existing Phase 4B implementation and original pilot
remain unchanged. Starting source commit: `22c5846e9d08a303432a5b7dba5651779beda406`.

## Matched macro objective

Use seed 101, 300 updates per variant, full and macro-only, 32 hidden dimensions,
four latent dimensions, 25 historical frames and 100 autonomous forecast steps
at 5 Hz. AdamW learning rate 0.001; gradient norm clipped at 1. Two independently
sampled prior rollouts estimate the same proper macro energy objective. No
microscopic, posterior or cross-population consistency term enters this control.
The existing trainer checkpoints optimizer and RNG state atomically after every
update. The wall allowance is operational; scientific fingerprints cannot change
on resume. Final checkpoints, not the lowest validation points, are evaluated.

Data are the already frozen 60-minute November 21 developmental selection,
94/23/23 train/validation/test windows. No new scene selection is performed.
Validation uses every window. Developmental test opens only after both final
checkpoints and a hashed validation decision exist. This is a single-day regime
shift study, not independent-date confirmation. No developmental-test tuning.

## Evaluation frozen before developmental-test access

Eight autonomous prior samples; dense and emulated final-half-history outage;
identical mask and rollout seeds across model variants. Five-, ten- and twenty-
second indices are 24, 49 and 99 after initialization at historical index 24.
Future masks enter scoring only. Score each supported lane-bin target using the
preregistered physical scales [30 m/s, 100 veh/km/lane, 3000 veh/h/lane]. Average
scene scores equally. Report the paired 10-second macro-only minus full energy
score, without treating adjacent windows as independent recording sessions.
No independent-session confidence interval is available for one recording day.

Report macro component MAE, unbiased ensemble CRPS, empirical marginal coverage,
90% width, pairwise diversity and negative generated fields. Eight-sample
quantiles are coarse empirical intervals, not certified 90% predictive coverage.
Report the decoder's q−3.6kv discrepancy without conflating total fields and the
historically observed cohort. Constant velocity aggregates only its observed
cohort; aggregate persistence has the correct historical aggregate input but
does not model future population turnover. Both are explicitly labeled.

Full-model micro ADE/FDE and physical rates are diagnostics of macro-trained
motion, not evidence of joint trajectory training. Longitudinal exits are separate
from lateral departures and reverse speed. Do not clip invalid predictions.
Saved NPZ forecasts and source/checkpoint hashes back every score and visual.
Arrays remain private; safe aggregate metrics and figures may be versioned.

## Limits on interpretation

Both models allocate 38,298 parameters, but macro-only does not use the agent
modules. Identical allocated size and optimizer budget do not imply matched
effective capacity. A favorable score alone cannot establish that microscopic
information is responsible. A future architecture-matched information restriction
control and independent recording sessions remain necessary.

Fixed-cohort motion and retained-population fields have different populations.
Future entrants and disappearing reconstructed tracks cannot be fabricated as
observations. No full-population conservation loss will be imposed on the cohort.
The source contains offline reconstructed trajectories, approximate lane geometry,
boundary thinning and missing coverage; it is not a complete roadway census.

Joint-model follow-on work must keep cohort and residual-population targets
separate, use past-only initialization, and live in a new namespace. It cannot
retroactively change this comparison. November 21/22/23 remain training dates,
November 25 validation, and November 28/29 independent-test assignments. Data
availability does not alter those frozen roles. No causal control or wave-onset
claim follows from these passive recordings.

## Bounded joint follow-on, frozen before new test evaluation

After the control completes, a separate seed-101 trial may use 30 additional
updates per model: full, independent microscopic dynamics, and graph-disabled.
All start from the identical final full macro-control weights and reset AdamW
for the new objective. Graph-disabled retains aggregate feedback, whereas the
independent vehicle pathway removes graph messages, shared microscopic innovations
and macro feedback. Its separate aggregate decoder still predicts total fields,
so it is not scored against total fields using a cohort-only aggregation head.
An inference regression verifies that changing neighbors and macro state cannot
change a fixed independent vehicle's rollout. This is a common warm-start trial;
the shared full-model warm start can favor the intact model and is not a substitute
for independently converged from-scratch controls.
It does not replace the preregistered macro-information comparison.

The new objective averages proper micro and macro scores with weights 0.5/0.5.
Microscopic scales are [100 m longitudinal position, 3.6576 m lateral position,
30 m/s longitudinal velocity, 5 m/s lateral velocity]. Macro scales remain
[30,100,3000]. Both use two autonomous prior samples, 20-second rollouts, the
same sampled train windows and sensing regimes, and all validation windows.
There is no future-conditioned posterior, pseudo-acceleration likelihood or
cross-population conservation penalty. Final update 30 is fixed before test.

Density and flow partition additively into the historically observed cohort and
the residual population; speed does not. The residual includes future entrants,
historically missed vehicles and source identity fragmentation. It must not be
interpreted as measured inflow. The new objective supervises cohort trajectories
and total retained-population fields separately. It is population-aware target
accounting, not a complete open-population microscopic generator: individual
arrivals and a nonnegative residual-population decoder remain future work.
This short trial tests whether direct trajectory supervision can improve joint
behavior; it cannot establish convergence or a general interaction advantage.

The joint trial also corrects an independent physical defect of the old total-
field decoder: independent Gaussian speed/density/flow heads can emit negative
fields and violate the defining exposure identity q=3.6kv. The graph/GRU layers
and checkpoint tensor shapes stay intact, but the new total-field head samples
positive lognormal speed and density and derives flow. Two old flow-decoder rows
are inactive. This constraint concerns one total-population field target; it is
not a conservation equality between total fields and the historical cohort.
Microscopic motion remains unclipped. This change belongs only to the separate
joint trial and never alters the matched macro-control models or their metrics.
