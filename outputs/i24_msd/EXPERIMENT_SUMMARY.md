# ORCHESTRA-I24-MSD — real-data pilot

**REAL_ACCESSED: authentic Harvard trajectories were parsed, trained and autonomously evaluated.** This establishes functioning real-data engineering, not a confirmed interaction advantage.

Source: DOI 10.7910/DVN/DQOWQI, release 1.1, file 11835291. Downloaded 1,369,499,074 bytes; checksum verified. Three retained TFRecord shards total 26,768,195 bytes. 349 native records inspected; 182 history-eligible multivehicle scenes; split {'test': 19, 'train': 57, 'validation': 19}.

Native timing: 91 × ~0.1 s; source cutoff index 10 (1 s); 8 s available future. Model observes six 5 Hz history states and evaluates 1/2/4/6 s futures. No 10/20/60 s or traffic-wave claim.

## Executed training

Device CPU, AMD EPYC 7763 64-Core Processor; Python 3.12.14; PyTorch 2.14.1+cpu; 2 threads, zero data-loader workers. Seed 101, 84,514 parameters/model, 500 optimizer updates/model. Three-model pilot training time 91.465 s (smoke uses only full). Checkpoint hashes and curves: training_metadata.json / training.csv. Fixed final checkpoints; no test selection.

Training uses autonomous prior recurrence with a training-only posterior auxiliary loss. Curated-cohort macro feedback/losses are disabled because population coverage is unknown. The validation curve improves but the fixed budget does not establish convergence; optimization loss includes an auxiliary term and is not a forecast score. No architecture or metric was changed to improve a test result.

## Four-second held-out results

| Model | ADE m | FDE m | Speed MAE m/s | Joint energy m | 90% coverage |
|---|---:|---:|---:|---:|---:|
| constant_velocity | 0.9190 | 2.2195 | 0.6577 | 1.8652 | 0.0000 |
| deterministic | 0.9114 | 2.4879 | 0.7045 | 2.0370 | 0.0000 |
| full | 0.9139 | 2.2715 | 0.6840 | 1.4675 | 0.5464 |
| independent | 0.8923 | 2.2459 | 0.6952 | 1.4770 | 0.7494 |
| persistence | 63.6780 | 130.7577 | 0.6577 | 92.6528 | 0.0003 |

Primary paired energy reduction (independent − full): **0.009499202171961466 m**, provisional 95% interval **[-0.10767094355076551, 0.07927718013525009]**. 18 eligible final-target scenes; 6 supplied-track groups, but only ONE source date. Confirmatory support is UNTESTED under the preregistered five-recording-group requirement. The interval crosses zero; this pilot does not establish an interaction-model advantage.

## Calibration, physical diagnostics and ablations

Full-model 4 s marginal coverage is 0.5464 for nominal 0.90; joint samples are underdispersed. Position CRPS 0.4246 m; RMS sample diversity 0.4878 m. Oriented-box overlap fraction 0.000000; acceleration >8 m/s² fraction 0.000000. These are diagnostics on a small low-density cohort, not safety evidence.
Finite-source-map road-edge footprint violation fraction: 0.1295, coverage 0.9937. Recorded-truth violation under the SAME edge diagnostic: 0.1396. Source noise, map selection and proxy headings limit physical interpretation; no clipping or boundary extrapolation is used.

| Full checkpoint, 4 s | FDE m | Joint energy m | Coverage90 |
|---|---:|---:|---:|
| dense | 2.2715 | 1.4675 | 0.5464 |
| graph_off | 2.6013 | 1.6623 | 0.5063 |
| outage | 2.8419 | 1.8513 | 0.3282 |
| reset_memory | 2.3764 | 1.5240 | 0.5825 |
| sparse50 | 2.1761 | 1.1754 | 0.5438 |

Graph-disabled/reset-memory comparisons use the same checkpoint and scenes. Sparse sensing may change the historically observed cohort; its aggregate error is not a directly matched all-agent benefit. Empty observation/target cases are recorded in skipped_evaluations.json.

## Artifacts and reproducibility

- [Real interactive showcase](demo/index.html), exact evaluated joint samples and checkpoint/source hashes. Browser verification and screenshot are in demo/.
- figures/: training/validation curves, horizon errors/proper scores/calibration/overlap and ablations.
- DOWNLOAD_MANIFEST.json, extraction_manifest.json, split_manifest.json, data_qc.json/csv, run_manifest.json, metrics.json, evaluation.csv, primary_result.json.
- Raw ZIP/shards, all checkpoints and full rollout arrays are retained locally and Git-ignored. The standalone HTML contains selected exact evaluated arrays.

```sh
uv run python scripts/run_i24_msd_pipeline.py --config configs/i24_msd_pilot.yaml --data-root data/i24_msd/raw
uv run pytest -q
```

## Limitations

Only one independently trained pilot seed and one dated archive; unknown absolute source times/aliases prevent exhaustive overlap certification. Supplied track IDs are disjoint across splits. Static geometry is retained, but curated clips do not establish unbiased density/flow. Fixed-cohort dynamics are evaluated; learned arrivals/exits, long-range waves, cross-day/corridor OOD, causal AV control and Cavnue hardware transfer are UNTESTED. Basic I24-MSD generation is prior work, not a new claim. The source/dataset and original instrument citations/terms are retained in docs/I24_MSD_DATA_CARD.md.

Phase 2 remains negative: coordination gain −0.132126, 95% CI [−0.700616, 0.436365]. Continuous-I24 fixture results remain fixture-only. Neither is rewritten by this real-data pilot.

NEXT EXPERIMENT: Run the preregistered four-second full-versus-independent joint-energy comparison on congested, independently dated I24-MSD scenes, grouping shared source tracks and matching observed speed/headway regimes to test residual interaction information value.
