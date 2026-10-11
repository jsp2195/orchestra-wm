"""Compile completed Phase 4C results without fitting or selecting models."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path('outputs/i24_phase4c')
MAIN = ROOT/'matched_macro_seed101_u300'
JOINT = ROOT/'joint_seed101_u30'

def table(frame, columns):
    frame = frame[columns].copy()
    rows = ['| '+' | '.join(columns)+' |', '| '+' | '.join(['---']*len(columns))+' |']
    for row in frame.itertuples(index=False, name=None):
        rows.append('| '+' | '.join(f'{x:.6f}' if isinstance(x, float) else str(x) for x in row)+' |')
    return '\n'.join(rows)

main = {s:pd.read_csv(MAIN/s/'baseline_table.csv') for s in ['validation','development_test']}
joint = {s:pd.read_csv(JOINT/'evaluation'/s/'baseline_table.csv') for s in ['validation','development_test']}
training = []
for namespace in [Path('outputs/i24_phase4b/matched_macro_seed101_u300'), JOINT]:
    for row in json.loads((namespace/'completed.json').read_text()):
        training.append(dict(experiment=namespace.name, variant=row['variant'], updates=row['updates'],
                             allocated_parameters=row['parameters'], wall_seconds=row['training_wall_seconds']))

text = '''# Phase 4C: actual training, preliminary results, preserved evidence

The requested seed-101 full-versus-macro-only experiment completed **300 updates
per model**, followed by a separately specified joint trial with **30 additional
updates each** for full, independent and graph-disabled models. Total new research
optimizer updates: **690**, excluding disposable numerical smoke tests. These are
authentic November 21 trajectories, not Harvard scenarios or synthetic fixtures.

The main comparison does **not establish a microscopic information advantage**.
Its developmental 10-second macro-only minus full energy difference is positive
but tiny; aggregate persistence is substantially better, calibration is poor,
effective capacity is unequal, and there is only one examined recording day and
one seed. The joint trial is a short warm-start experiment, not a converged or
complete open-population traffic world model. Negative historical findings remain.

## Preservation and reproducibility

The 603 previously tracked scientific files, six original 100-update checkpoints,
and both original recovery bundles passed SHA256 rechecks. No source archive,
prepared dataset or old checkpoint was deleted or replaced. The original private
backup receipt passed the real training gate. See
[preservation verification](preservation_verification.json). A local filesystem
surviving this task is not a guarantee for a new cloud task; verified private
backups, source fingerprints and receipts are the recovery mechanism.

New checkpoints retain optimizer, NumPy and Torch RNG states, configuration,
source hashes, update counts and curves. Writes are atomic; completed results and
evaluation protocols are immutable. The initial 1,800-second allowance stopped
the main full model at update 154; the identical command resumed to 300 and then
trained macro-only. No fingerprint was bypassed and no budget was silently changed.

## Data and physical interpretation

The frozen expanded data contain **60 minutes, 30,170 source track records and
10,368,782 native trajectory samples**, with **94 train / 23 validation / 23
developmental-test windows**. The source is November 21, 2022, westbound, roughly
1 km. Train intervals are 06:15–06:35 and 07:15–07:35 CST, validation 08:15–08:25,
and development test 09:15–09:25. Training is more congested; the later test has
faster traffic. All 23 held-out windows are scored, with no performance selection.
The largest union scene has 464 track records. Scene history is 5 seconds and
the autonomous objective and evaluation extend to 20 seconds at 5 Hz.

The native cadence is about 25 Hz. Conversion uses Unix seconds, 0.3048 m/foot,
westbound travel coordinates, vehicle-center offsets and causal resampling.
Publisher timestamps are documented as synchronization corrected; reconstruction
and smoothing are offline processing, not independently verified online sensing.
No future identities, values, validity masks or fields initialize inference.
Future support masks are scoring/supervision only. Lane bounds are the existing
approximate 12/24/36/48/60-foot roadway-coordinate boundaries, not surveyed maps.

Splits share no source IDs or observation/forecast support. Expanded boundary
thinning is about 3.06%; the historical pilot discarded 1,291 guard/crossing tracks
and 655,055 samples (30.25% of its relevant in-interval denominator). Source gaps,
boundary thinning and track fragmentation mean these are **retained reconstructed
populations, not complete freeway censuses**. Empty support is unknown, not zero.
See [source limitations](../../docs/I24_PHASE4C_SOURCE_LIMITATIONS.md).

Fields use trailing one-second sampled vehicle-time/distance exposure: density
in veh/km/lane, speed in m/s, flow in veh/h/lane. The approximation satisfies
q=3.6kv for the same population. The historical cohort accounts for only about
30.3% of total developmental-test target density at 10 seconds, 1.3% at 20 seconds.
Residual population includes arrivals, missed observations and fragmented tracks;
it is not an identified inflow process. Long-horizon micro scores are survivor-
conditioned, not errors over every future vehicle.

## What was trained

'''+table(pd.DataFrame(training),['experiment','variant','updates','allocated_parameters','wall_seconds'])+'''

The CPU-only worker used two Torch threads and zero data workers (three CPU quota,
about 9.7 GiB RAM). Largest-scene smoke updates were finite: main full 23.4 s and
about 706 MiB peak RSS; joint 10.9 s and about 631 MiB. Observed main training RSS
was about 1 GiB. Wall times above include each trainer's validation/checkpoint
work, not standalone evaluations or rendering. Gradient norms and validation
curves are saved, with clipping at 1 and fail-closed nonfinite checks.

The architecture retains graph messages, agent/field GRUs with 32 hidden units,
and a four-dimensional autonomous Gaussian latent prior. AdamW learning rate is
0.001. Main variants use identical proper macro energy objectives, two prior
samples and identical update/window/regime budgets; no microscopic, posterior,
pseudo-acceleration NLL or cross-population consistency objective is included.
The full model's microscopic output in this experiment is therefore a diagnostic
of macro-trained motion, not directly supervised trajectory forecasting.

Allocated parameters are equal (38,298), but a local three-step gradient diagnostic
finds 36,850 connected parameters for full versus 17,870 for macro-only. This is
not an architecture-matched information restriction. Macro-only demonstrably
ignores microscopic values/roster; posterior modules are unused in both. Full
validation macro objective fell from 0.22162 to 0.15272, and macro-only from 0.22008
to 0.15330. Full reached about 0.13285 at update 240, then worsened: convergence
is not established. The prospectively fixed final update 300 was evaluated.

The joint trial starts all three models from the same full-300 weights and resets
AdamW for the new objective. Each receives 30 updates; this is not three separate
330-update training campaigns. The warm start may favor intact full dynamics.
Micro and macro proper scores receive 0.5/0.5 weights, with micro physical scales
[100 m, 3.6576 m, 30 m/s, 5 m/s] and macro [30,100,3000]. Cohort motion and total
fields are supervised separately; no invalid cohort-total conservation loss is
imposed. Positive lognormal speed/density and derived flow repair the Gaussian
head's negative-field and flow-identity defects only in this new joint namespace.
Two old flow-output rows are inactive. Individual arrivals are not generated.

The independent vehicle branch removes neighbor and aggregate feedback and shared
microscopic noise; a multistep invariance test verifies the restriction. It has a
separate total-field decoder, avoiding a cohort-versus-total target mismatch.
Graph-off only removes graph messages and retains aggregate feedback. Neither is
a pure information control. No new deterministic, micro-only or no-cross-scale
model, extra seed or independent-date model was trained in this bounded campaign.

### Audit corrections and unresolved comparison defects

The Phase 4B audit had already repaired macro-only's discarded learned prior,
removed microscopic consistency supervision from the aggregate-only control,
removed the drift-contaminated acceleration pseudo-target from the common loss,
and rejected laterally off-road agents during aggregation. This execution tested
and trained that existing corrected implementation without rewriting old results.
New Phase 4C evaluation distinguishes legal exits, adds CRPS/diversity/coverage,
scores aggregate persistence from the last supported history rather than treating
outage bins as observed zeros, and saves explicit endpoint/support denominators.
The separate joint head repairs negative speed/density/flow and q-versus-kv
inconsistency. Regression checks exercise those demonstrated failure modes.

The historical micro-only variant was not a pure no-macro-supervision control:
it retained field NLL and used a different aggregation head. Its earlier good
macro score therefore cannot establish interaction benefit. Historical full also
faced a much larger suspect acceleration term and unmatched losses. The new common
macro objective removes that loss imbalance, but still leaves effective capacity,
optimization, source completeness and population-turnover limitations unresolved.
The joint 0.5/0.5 physical-scale weights are prospective, not established optimal
or automatically balanced gradient contributions. Full convergence and useful
stochastic diversity remain to be demonstrated, not assumed from finite gradients.

## Main preregistered information comparison

Lower energy is better; the paired difference is macro-only minus full, positive
favoring full. Physical scales remain [30,100,3000]. Eight prior samples, common
scene/sensing seeds, supported bins and equal scene weights were used. Validation
decisions were hashed and frozen before a single developmental-test pass. The
test was not used to choose the checkpoint, training duration or architecture.

'''
for split, frame in main.items():
    selected=frame[(frame.regime=='dense') & (frame.horizon_seconds==10)]
    scores=selected.set_index('variant').macro_energy
    text+=f'### {split}: 10 seconds, dense observations\n\n'
    text+=table(selected,['variant','macro_energy','speed_mae','density_mae','flow_mae','macro_coverage90'])
    text+=f'\n\nPaired mean macro-only minus full: **{scores["macro_only"]-scores["full"]:+.9f}**.\n\n'
text+='''Aggregate persistence beats both learned macro models on the developmental
test. CV's macro fields only contain its historical cohort, so it is a secondary
population-mismatched macro baseline, although a valid cohort-motion baseline.
There is no independent-session confidence interval: 23 adjacent windows are
not 23 independent recordings. The preregistered confirmatory group requirements
are unmet. Eight-sample quantiles are coarse empirical predictive intervals,
not confidence intervals on the information-value effect.

### Actual autonomous horizon and microscopic diagnostics

'''+table(main['development_test'].query("regime == 'dense' and variant in ['full','macro_only','constant_velocity']"),['variant','horizon_seconds','macro_energy','ade_m','fde_m','micro_xy_coverage90','valid_micro_endpoint'])+'''

ADE/FDE above use the ensemble mean; expected-sample ADE is separately recorded.
Endpoint counts above are averages per scene, not total independent vehicles.
Only 10 total cohort endpoints across seven of 23 dense test scenes survive to
20 seconds; no outage cohort endpoint survives there. Missing FDE is unavailable,
never a zero. This severely limits long-horizon microscopic generalization claims.
Physical departures are evaluated over target-valid cohort sample-times and can
understate failures outside source support. Pair overlaps are geometric diagnostics,
not crash probabilities. Legal longitudinal exits are reported separately.

## Separately supervised joint trial

Final update 30 was fixed before its new test pass. Validation trajectories improve
substantially over their first joint update, but fluctuate and do not establish
convergence. This trial was specified before inspecting its new developmental
outcomes; the day had already been examined by the main experiment and is not
pristine independent evidence. The changed head/loss means this cannot replace
the main full-versus-macro-only hypothesis test.

'''
for split, frame in joint.items():
    text+=f'### Joint {split}\n\n'
    text+=table(frame.query("regime == 'dense'"),['variant','horizon_seconds','macro_energy','ade_m','fde_m','macro_coverage90','micro_xy_coverage90'])+'\n\n'
text+='''### Calibration, physical validity and missing observations

'''+table(joint['development_test'].query("horizon_seconds == 10 and variant in ['full','independent','graph_off']"),['variant','regime','macro_energy','macro_coverage90','macro_diversity','macro_negative_fraction','flow_identity_mae_veh_h','lateral_departure_rate','reverse_speed_rate','endpoint_vehicle_overlap_fraction'])+'''

On developmental test, joint full's 10-second FDE is **11.302 m**, versus **10.362 m**
for CV and **10.469 m** for independent dynamics. Full improves over its macro-
objective-only motion (23.609 m) but does not beat these baselines. Full's macro
energy is **0.411892**, independent **0.380185**, graph-off **0.413243**, and aggregate
persistence **0.294397**. The small graph-off minus full difference, +0.001351,
is descriptive within this warm-start trial, not a confirmed interaction gain.
Independent remains better overall, despite the common full-model warm start.
The joint trial's nominal 90% macro coverage is only **24.97%** for full, worse
than the main control's **29.20%**. It is neither calibrated nor reliably physical
at the vehicle level, despite the constrained total-field decoder.

### Joint developmental component errors at 10 seconds, dense observations

'''+table(joint['development_test'].query("horizon_seconds == 10 and regime == 'dense'"),['variant','speed_mae','density_mae','flow_mae','speed_crps','density_crps','flow_crps'])+'''

All component CRPS, widths, coverage, speed/density/flow MAE, acceleration rates,
overlap denominators, sample counts and scoring support are in the paired CSVs.
Zero negative fields and a near-zero flow-identity residual in the joint trial
are structural constraints, not evidence of accurate motion or calibrated traffic
uncertainty. The microscopic trajectories remain unclipped and may be invalid.
Dense/outage comparisons emulate missing history observations, not real sensor
failure ground truth. Past-only speed/density regime tables are exploratory and
do not change primary scoring or select scenes.

The learned prior has nonzero gradients and changed weights in both main models.
Removing its learned mean while retaining variance/noise changes predictions,
but improves mean validation energy on the fixed first three diagnostic windows
by about 0.00391 (full) and 0.01713 (macro-only). Removing full's graph messages
changes forecasts only slightly and improves energy there by about 0.00021.
Memory reset also changes forecasts. These are distribution-shifting sensitivity
checks, not retrained causal tests or proof that the latent state, graph or memory
improves predictive skill. Useful latent representation remains unestablished.

## Recorded disturbances and independent dates

The training-only [recorded kymographs](recorded_dynamics/observed_training_kymographs.png)
and [lag diagnostic](recorded_dynamics/analysis.json) show adjacent-bin correlations
consistent with upstream propagation: peak delays 14–28 seconds, apparent speeds
−4.46 to −8.93 m/s and correlations about 0.32–0.62. This exploratory lag search
uses offline detrending and dependent data, with no significance test or validated
wave annotation. It does not identify wave onset, causality, or model reproduction;
many lags exceed the 20-second forecast. **Generated congestion propagation was
not demonstrated.** This passive model supports no vehicle-control intervention claim.

Connected Drive metadata verified ten dated archives, including all frozen dates:
train November 21/22/23, validation November 25, independent test November 28/29.
See [available dates](available_recording_dates.json). Other dates remain unassigned.
No new archive was downloaded or test-day outcome inspected. The direct environment
source/upload binding returned one HTTP401 and was not retried; the separately
authorized connected Drive app supports metadata and verified compact backups.
Its local materialization tool is limited to 32 MiB, so a large-archive streaming
transfer still requires a supported route. Remaining storage and this execution's
bounded compute were also reasons to defer multi-date ingestion. This is not a
claim that the source files are inaccessible or that another date cannot fit with
a more efficient staging implementation. The original archives remain intact.

## Visual evidence and tests

[Main gallery](gallery/VISUAL_GALLERY.md) preserves the macro-objective experiment.
[Joint gallery](joint_gallery/VISUAL_GALLERY.md) shows the newer directly supervised
joint forecasts. Each has real traffic, eight independent autonomous futures,
collective fields, interaction geometry and emulated-outage GIFs, plus learning,
calibration, physical-consistency and baseline figures. First chronological scenes
are shown, not favorable scenes. Straightened approximate roadway geometry is
explicitly labeled. Recorded orange futures and generated blue futures remain distinct.

Interactive HTML payloads remain private because they embed exact trajectory
arrays. Browser verification checks them byte-for-value against saved evaluation
arrays and exercises scenario, variant, sample, outage, truth, playback and time
controls, including 20-second endpoints. Download the privately supplied HTML
and open it in a browser; a Drive preview is not hosted JavaScript execution.
Public figures and aggregate CSVs do not include raw source records or checkpoints.

Run `uv run pytest -q --import-mode=importlib` for the full suite. The import mode
avoids pre-existing duplicate test basenames. Focused tests cover future-input
exclusion, information restrictions, probabilistic scores, population decomposition,
independent rollouts, physical head constraints, resumable upload offsets and
authentication failure handling. Final test and browser receipts accompany this report.

## Exact continuation and next experiment

The requested runs are completed; identical commands verify/reuse completed
artifacts rather than extending incompatible fingerprints:

```sh
cd /workspace/orchestra-wm
uv run python scripts/train_i24_phase4b.py --data-output outputs/i24_phase4b/development_v1 --output outputs/i24_phase4b/matched_macro_seed101_u300 --preservation-receipt work/phase4b/private_backup_receipt.json --updates 300 --seed 101 --variants full macro_only --wall-seconds 1800
uv run python scripts/train_i24_phase4c_joint.py --preservation-receipt work/phase4b/private_backup_receipt.json --updates 30 --seed 101 --wall-seconds 900
uv run python scripts/evaluate_i24_phase4c.py --training-output outputs/i24_phase4c/joint_seed101_u30 --output outputs/i24_phase4c/joint_seed101_u30/evaluation --split development_test
```

Restore the original and expanded private bundles plus new verified artifacts
at their recorded relative paths if moving workers. Changing update budgets
requires a new namespace with explicit warm-start provenance, not editing the
fingerprint or overwriting completed checkpoints.

The highest-value next experiment is a **prospectively frozen, effective-capacity-
matched full-versus-aggregate information test across the assigned recording
dates**, retaining persistence and independent dynamics baselines and auditing
source population coverage. Establish validation convergence and calibration
before opening independent test dates. More updates on this single day's regime-
shift split alone cannot establish the intended scientific claim. A residual
population/birth-death model and reliable long-lived identities remain prerequisites
for a complete open-population microscopic generator and cross-scale conservation.
'''
(ROOT/'PHASE4C_REPORT.md').write_text(text)
print('Wrote completed-results report:', ROOT/'PHASE4C_REPORT.md')
