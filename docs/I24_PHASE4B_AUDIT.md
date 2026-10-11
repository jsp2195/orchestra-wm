# Phase 4B scientific audit

This is an audit and developmental data expansion, not a new positive scientific
result. All original pilot code, checkpoints, metrics and figures remain unchanged.
Starting commit: `d9eb80aeb16cd15aef954ad7d53a986d5acb979b`.

## Confirmed correct

- `i24_continuous/data.py:observations` selects identities from historical visible
  detections only, zeros hidden values, and recomputes historical fields from
  visible samples. A perturbation test changes all future values and future
  detection/existence masks without changing inference inputs.
- The recurrent rollout receives its own predicted state and an autonomous prior.
  The future-conditioned posterior is used in a separate training-loss branch;
  its returned state is never the next recurrent inference state. Targets and
  target masks enter losses/scoring, not the history initialization.
- This model-level causality does **not** remove future information used by the
  original offline camera reconstruction. Track association, smoothing and
  timestamp corrections are publisher processing, not independently verified raw
  sensing. History-wide identity allocation is available at forecast cutoff.
- Unix seconds, feet-to-metres factor 0.3048, westbound sign, back-center to center
  offset and backward derivatives are explicit. Source cadence is approximately
  25 Hz; 5 Hz uses past-only extrapolation up to 0.08 s. The source documentation
  identifies synchronization-corrected timestamps. No claim of raw synchronization
  re-estimation or verified identity correspondence is made.
- The documented approximate WB lane bounds are 12/24/36/48/60 feet, not a newly
  surveyed road boundary. Graphics show straightened roadway coordinates, not a
  verified geographic aerial map. Global calibration remains unvalidated.
- Temporal splits have no shared history/forecast support or source IDs across
  train/validation/test. Original 25-second scenes use five seconds of history
  (indices 0–24) and predictions 25–124. Indices 49/74/124 are 5/10/20 seconds
  after the last observed time at index 24; the physical horizon arithmetic is
  correct. Adjacent scenes can share the unused final endpoint, but splits cannot.
- Macro labels use trailing one-second sampled vehicle-time and vehicle-distance
  exposure: density = vehicle-seconds/(length × time), flow = vehicle-metres/
  (length × time), speed = distance/exposure. This is a 5 Hz Riemann approximation,
  not exact continuous Edie integration. Unknown empty support is masked.
- Joint energy score uses fixed physical scales [30,100,3000], RMS-normalized
  Euclidean distances and off-diagonal ensemble pairs. All compared models use
  identical supported macro targets per scene/regime/horizon. The three test
  windows are one contiguous block; no independent-day CI is defensible.
- Atomic optimizer/RNG checkpoints and data/source/config fingerprints were
  present. The six recovered hashes match their original recorded values.

## Confirmed defects and corrections isolated in `i24_phase4b`

| Defect | Consequence | Phase 4B correction / status |
|---|---|---|
| Macro-only computes a learned prior sample, then replaces `field_z` with fresh standard-normal noise | Its learned latent prior does not drive field generation; stochastic pathways differ from full | `AuditedWorldModel` uses its learned macro prior; a test verifies prior sensitivity and independence from microscopic values and roster-sized random draws |
| Macro-only retains the consistency penalty against CV-propagated microscopic observations | Its training objective includes microscopic information despite the aggregate-only label | Prospective matched-macro control removes all micro/consistency/posterior terms **from both variants**, with an identical proper macro-energy objective |
| “Acceleration” NLL target uses `(true_future_velocity - imagined_previous_velocity)/dt` | Accumulated velocity drift is misinterpreted as one-step physical acceleration | The prospective prior-only control removes this pseudo-acceleration likelihood. A future joint microscopic likelihood must use a correctly defined transition target; it is not claimed implemented here |
| Soft spatial aggregation does not reject lateral departures | Off-road agents can still contribute lane density/flow | New bounded temporal aggregation applies documented lateral bounds; regression test demonstrates the original failure |
| Original physical departure rate mixes lawful longitudinal exits with lateral departures | It is not solely an off-road violation rate | Supplementary audit separates exits, lateral departures and reverse speed; historical table remains untouched |
| CRPS, diversity, interval widths and decoder/micro consistency omitted | Original evaluation did not cover the full probabilistic specification | Added separately on exact saved forecasts with explicit support/denominators; no original primary metric is replaced |

The corrected model keeps the 32-hidden/4-latent topology and 38,298 allocated
parameters. Equal allocated parameter counts are **not** proof of equal effective
capacity: aggregate-only and independent variants have inactive branches. The
prospective macro-only branch deliberately excludes microscopic state from its
field computation. A clean information-value conclusion still needs capacity and
optimization controls (including an appropriately matched aggregate-capacity
control), and adequately trained validated models. This audit does not claim that
those controls have already passed.

## Objective balance and the negative result

On the first **training** scene, dense history, original full checkpoint, the
50-step loss decomposition gives micro MSE 0.02461, macro MSE 0.01630, weighted
acceleration NLL 0.05133, weighted consistency 0.00094, field NLL 0.00105 and KL
0.00090. Thus the suspect acceleration term exceeds the macro MSE by about 3.15×
on this diagnostic scene. Its acceleration target differs from a successive-truth
velocity difference by 4.46 m/s² on average. This is direct evidence of unequal
objective pressures; it does not establish their causal contribution across the
whole training distribution. No optimizer steps or test tuning were performed.

Micro-only and independent use generated-vehicle aggregation, whereas full,
macro-only, deterministic and no-cross-scale use a learned macro decoder. The
micro-only implementation disables macro MSE and consistency, **but still retains
its weighted field NLL** in the stochastic term. It is therefore neither a
no-macro-supervision control nor a matched decoder comparison. Micro-only has no
observed-field input to its predictive branch; that input restriction is distinct
from its supervision. Its better energy score cannot by itself demonstrate
superior microscopic dynamics or an interaction-information advantage.

Full vs macro-only also changes loss terms, latent utilization and effective
capacity. Full vs independent removes messages, shared noise and field feedback;
it is not exclusively a graph ablation. No-cross-scale does not answer the pure
information question either. Memory-reset, residual-permutation, graph-only and
aggregate-capacity controls remain unrun. Proper-score training is prospective;
no corrected checkpoint has been fitted or evaluated.

The original MSE on stochastic draws includes a variance penalty, which can
encourage low diversity. The small likelihood/KL terms do not establish latent
utility. Diversity is nonzero, but a prior-mean/latent ablation and validation
calibration study are still needed. Gaussian field noise permits negative values;
none were clipped to improve the audit. Physical validity is not guaranteed.

## Population and masking limitations

The original split discarded 1,291 crossing/guard records and 655,055 samples:
30.25% of retained-or-omitted in-interval samples (short exclusions outside that
denominator). This is substantial population thinning, especially near split
boundaries. Fields represent retained reconstructed regional tracks, not a
freeway census. Source spatial thinning near s=870–890 m is independently noted.

At dense 10 seconds, the historical cohort accounts for only 40.02% of target
density on average. Future unseen tracks include actual arrivals, reconstruction
fragments and losses/reacquisitions; this is not a calibrated physical inflow
estimate. Full macro targets contain these tracks, while the micro generator has
no birth process. A consistency penalty demanding equality with a fixed cohort
is thus scientifically mismatched. The audit compares hard aggregation of saved
generated vehicles with the decoder separately, rather than declaring agreement.
Full scaled decoder/micro RMSE is 0.16572 at 10 seconds.

The original ADE/FDE evaluate the **ensemble mean trajectory**, not mean sample
ADE/FDE. The audit adds expected-sample ADE with its own label. Target masks are
valid for scoring but cannot justify drawing future predictions only where truth
exists. The viewer uses predicted validity for blue vehicles and truth validity
only for orange recorded comparisons. Predicted cohort persistence despite track
fragmentation is visible, not hidden. Overlap rates are per unordered pair and
must not be described as collision probabilities. No leader ordering, class or
lane-change stratification result is claimed.

Marginal coverage uses empirical 5th/95th ensemble quantiles from only eight
samples. It is a forecast interval, not a confidence interval on model skill.
Original full dense 10-second coverage is 20.1% versus nominal 90%. Audit mean
widths are 2.17 m/s, 8.42 veh/km/lane and 210.44 veh/h/lane. Full lateral departure
rate is 3.69% and reverse-speed rate 5.97% over target-valid micro sample-times;
longitudinal exit rate is reported separately. These are averaged scene rates,
not pooled population estimates. The generator is not physically validated.

## Training horizon, convergence, and claims

Original training used 10-second autonomous objectives and evaluated actual
20-second rollouts. The latter are out-of-training-horizon tests, not evidence of
20-second-trained dynamics. The new bounded control is configured for 20-second
prior rollouts; it has **zero completed training updates** at this stage.

Original full validation objective continued falling from update 80 to 100 by
0.00972. Only 13 training windows and one seed were used. No convergence claim is
supported; different variant validation objectives are not comparable scores.
The primary historical macro-only minus full energy difference remains
**−0.005290**. Macro-only is better; a positive information-value result is absent.
Full micro FDE 12.280 m also trails constant velocity 11.977 m at dense 10 seconds.

No confirmed congestion propagation or onset result is reported. Observed
space-time plots are evidence of reconstructed traffic, not annotated propagation
speed or causal wave generation. No passive observational result identifies an
AV/control intervention effect.

## Gates before scaling

1. Complete verified private durable backup of compact data and checkpoints.
   Current storage is an overlay; read-only Drive credentials cannot upload.
2. Refresh source access securely to verify date availability (one listing failed
   HTTP401; not retried). November 21 remains examined development data. Frozen
   prospective date assignments are in `configs/i24_phase4b_development.json`.
3. Validate expanded scenes and capacity/memory behavior. Do not truncate high
   populations or optimize repeatedly on the old 13 windows.
4. Run the bounded common-objective full/aggregate control, compare validation
   convergence without test selection, and address effective-capacity controls.
   It isolates the macro objective, not joint microscopic fidelity.
5. Define a scientifically consistent open-population or cohort-target experiment
   before claiming a coherent micro–macro generative model. Keep the original
   primary target and comparison unchanged for historical reporting.
6. Only then perform new held-out evaluation and, if warranted, independent seeds
   and dates. Fewer than five independent sessions/days cannot meet the existing
   confirmatory gate. Two reserved test dates alone are insufficient.
