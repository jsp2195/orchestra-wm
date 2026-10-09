# ORCHESTRA-I24 — SYNTHETIC FIXTURE — NOT AN I-24-TRAINED MODEL

Real-data autonomous generation, wave modeling, microscopic information value, and real-Cavnue transfer are BLOCKED.

Protected Phase-1/2 checks: 268. Phase-2 coordination gain remains −0.132126, CI [−0.700616, 0.436365]; no reinterpretation.

Fixture/full 10 s FDE: 16.364200 m; macro energy score: 0.086164; trajectory 90% coverage: 0.297643; overlap pair fraction: 0.007802. These numbers do not measure I-24 skill.

## Training and evaluation

6 model/seed runs, 100 optimizer updates each, seeds [101], CPU threads 2. Full model parameters: [84514]. Total measured training time: 74.47 seconds. Updates sample scenes; they are not full dataset epochs. Validation curves and all final checkpoint hashes are recorded. Convergence is not established.

The six trained variants are independent, deterministic relational, macro-only, micro-only, no-cross-scale, and full. Kinematic baselines run before training. All forecasts are autonomous. Information-value and architecture-value comparisons remain distinct. Stochastic prior innovations and decoder variances are trained, not arbitrary post-hoc noise. The training-only posterior has no inference interface.

10 s paired macro energy-score reduction (macro-only minus full): 0.011856; cluster interval [-0.007345774476586433, 0.024534865379762263], 4 clusters. Fixture scene seeds are not real recording days; one training seed is preliminary engineering evidence.

Mode definitions: fixed cohort uses only historically detected identities. Macro targets include the full bounded source population; this is harder than reconstructing visible fields. Open-road generation uses a separately labeled past-appearance-rate Poisson inflow assumption, with no future identities. It is not validated inflow learning. Unknown sampling coverage means reconstructed density is not an unbiased census of real traffic.

Partial observations are EMULATED. The same checkpoint is evaluated under dense, sparse50/sparse20, blind-zone, outage, reset-memory, graph-off and kinematic-residual permutation controls. Macro-only agent outputs are explicitly N/A. Zero valid-agent denominators are recorded and must not be mistaken for successful forecasts. No real wave events or cross-day OOD were evaluated. IDM calibration is not applicable to this fixture campaign.

## Artifacts

- Interactive demo: [outputs/i24/demo/index.html](demo/index.html)
- Numerical forecasts: rollouts/*.npz (ignored from Git); demo payload retains the displayed evaluation arrays and provenance.
- Metrics: metrics.json; evaluation.csv; baseline_table.csv; paired_ablations.csv; confidence_intervals.json; stratified_metrics.csv.
- QC/split hashes: data_qc/; split_manifest.json; run_manifest.json.
- Figures: figures/ (fixture labels on plots).

The fixture demo fails the real-data showcase gate by design. HTML controls select saved autonomous checkpoint predictions; revealing truth only affects the evaluation view.

## Resume with authorized real data

Prepare acquisition.json as documented in docs/I24_DATA_ACQUISITION.md and place the official JSON files under the same directory, then run:

```sh
uv run python scripts/run_i24_pipeline.py --config configs/i24_pilot.yaml --data-root /path/to/I24MOTION_PUBLIC
```

No login bypass, substitute one-vehicle CAN/GPS dataset, I24-3D subset, or invented Cavnue data was used. Real-source adapter execution and real clip geometry/QC remain blocked until the files arrive.

NEXT EXPERIMENT: Run the preregistered 10-second macro energy-score comparison on an authorized congested I-24 multivehicle subset, with matched macro histories and measured micro headway differences.
