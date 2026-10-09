"""Generate restrained summaries from executed pilot measurements."""
from pathlib import Path
import json
import pandas as pd


def write_summary(out):
    out=Path(out);m=json.loads((out/'metrics.json').read_text());q=m['qc'];t=m['training'];p=m['results']['primary']
    f=pd.read_csv(out/'baseline_table.csv');dense=f[(f.regime=='dense')&(f.horizon_seconds==4)].set_index('variant')
    full=dense.loc['full'];resource=json.loads((out/'resource_audit.json').read_text())
    lines=['# ORCHESTRA-I24-MSD — real-data pilot','',
        '**REAL_ACCESSED: authentic Harvard trajectories were parsed, trained and autonomously evaluated.** This establishes functioning real-data engineering, not a confirmed interaction advantage.','',
        f"Source: DOI 10.7910/DVN/DQOWQI, release 1.1, file 11835291. Downloaded {q['downloaded_bytes']:,} bytes; checksum verified. Three retained TFRecord shards total 26,768,195 bytes. 349 native records inspected; {q['multiagent_context_scenes']} history-eligible multivehicle scenes; split {q['selected_split_counts']}.",'',
        'Native timing: 91 × ~0.1 s; source cutoff index 10 (1 s); 8 s available future. Model observes six 5 Hz history states and evaluates 1/2/4/6 s futures. No 10/20/60 s or traffic-wave claim.','',
        '## Executed training','',
        f"Device CPU, {resource['cpu']}; Python {resource['python']}; PyTorch {resource['torch']}; 2 threads, zero data-loader workers. Seed 101, {t[0]['parameters']:,} parameters/model, {t[0]['updates']} optimizer updates/model. Three-model pilot training time {sum(x['seconds'] for x in t):.3f} s (smoke uses only full). Checkpoint hashes and curves: training_metadata.json / training.csv. Fixed final checkpoints; no test selection.",'',
        'Training uses autonomous prior recurrence with a training-only posterior auxiliary loss. Curated-cohort macro feedback/losses are disabled because population coverage is unknown. The validation curve improves but the fixed budget does not establish convergence; optimization loss includes an auxiliary term and is not a forecast score. No architecture or metric was changed to improve a test result.','',
        '## Four-second held-out results','',
        '| Model | ADE m | FDE m | Speed MAE m/s | Joint energy m | 90% coverage |',
        '|---|---:|---:|---:|---:|---:|']
    for name,row in dense.iterrows():lines.append(f'| {name} | {row.ade_m:.4f} | {row.fde_m:.4f} | {row.speed_mae_mps:.4f} | {row.joint_energy_m:.4f} | {row.coverage90:.4f} |')
    lines+=['',f"Primary paired energy reduction (independent − full): **{p['paired_energy_reduction_m']} m**, provisional 95% interval **{p['ci95_provisional']}**. {p['eligible_test_scenarios']} eligible final-target scenes; {p['track_disjoint_groups']} supplied-track groups, but only ONE source date. Confirmatory support is UNTESTED under the preregistered five-recording-group requirement. The interval crosses zero; this pilot does not establish an interaction-model advantage.",'',
        '## Calibration, physical diagnostics and ablations','',
        f"Full-model 4 s marginal coverage is {full.coverage90:.4f} for nominal 0.90; joint samples are underdispersed. Position CRPS {full.position_crps_m:.4f} m; RMS sample diversity {full.diversity_rms_m:.4f} m. Oriented-box overlap fraction {full.overlap_fraction:.6f}; acceleration >8 m/s² fraction {full.acceleration_over_8_fraction:.6f}. These are diagnostics on a small low-density cohort, not safety evidence.",
        f"Finite-source-map road-edge footprint violation fraction: {full.road_boundary_violation:.4f}, coverage {full.road_boundary_coverage:.4f}. Recorded-truth violation under the SAME edge diagnostic: {full.truth_road_boundary_violation:.4f}. Source noise, map selection and proxy headings limit physical interpretation; no clipping or boundary extrapolation is used.",'',
        '| Full checkpoint, 4 s | FDE m | Joint energy m | Coverage90 |','|---|---:|---:|---:|']
    for _,r in f[(f.variant=='full')&(f.horizon_seconds==4)].iterrows():lines.append(f'| {r.regime} | {r.fde_m:.4f} | {r.joint_energy_m:.4f} | {r.coverage90:.4f} |')
    lines+=['','Graph-disabled/reset-memory comparisons use the same checkpoint and scenes. Sparse sensing may change the historically observed cohort; its aggregate error is not a directly matched all-agent benefit. Empty observation/target cases are recorded in skipped_evaluations.json.','',
        '## Artifacts and reproducibility','',
        '- [Real interactive showcase](demo/index.html), exact evaluated joint samples and checkpoint/source hashes. Browser verification and screenshot are in demo/.',
        '- figures/: training/validation curves, horizon errors/proper scores/calibration/overlap and ablations.',
        '- DOWNLOAD_MANIFEST.json, extraction_manifest.json, split_manifest.json, data_qc.json/csv, run_manifest.json, metrics.json, evaluation.csv, primary_result.json.',
        '- Raw ZIP/shards, all checkpoints and full rollout arrays are retained locally and Git-ignored. The standalone HTML contains selected exact evaluated arrays.',
        '', '```sh','uv run python scripts/run_i24_msd_pipeline.py --config configs/i24_msd_pilot.yaml --data-root data/i24_msd/raw','uv run pytest -q','```','',
        '## Limitations','',
        'Only one independently trained pilot seed and one dated archive; unknown absolute source times/aliases prevent exhaustive overlap certification. Supplied track IDs are disjoint across splits. Static geometry is retained, but curated clips do not establish unbiased density/flow. Fixed-cohort dynamics are evaluated; learned arrivals/exits, long-range waves, cross-day/corridor OOD, causal AV control and Cavnue hardware transfer are UNTESTED. Basic I24-MSD generation is prior work, not a new claim. The source/dataset and original instrument citations/terms are retained in docs/I24_MSD_DATA_CARD.md.','',
        'Phase 2 remains negative: coordination gain −0.132126, 95% CI [−0.700616, 0.436365]. Continuous-I24 fixture results remain fixture-only. Neither is rewritten by this real-data pilot.','',
        'NEXT EXPERIMENT: Run the preregistered four-second full-versus-independent joint-energy comparison on congested, independently dated I24-MSD scenes, grouping shared source tracks and matching observed speed/headway regimes to test residual interaction information value.']
    (out/'EXPERIMENT_SUMMARY.md').write_text('\n'.join(lines)+'\n')
    if out==Path('outputs/i24_msd'):
        ledger=['# I24-MSD claim ledger','', 'Separate campaign; prior Phase-1/2/continuous-I24 claim ledgers are preserved. Config i24_msd_pilot.yaml; training seed 101; split seed 7301; bootstrap seed 7103.','',
                '| Claim | Status | Executed evidence |','|---|---|---|',
                '| Authentic multivehicle data acquired and decoded | SUPPORTED | 1,369,499,074-byte Harvard ZIP, MD5 verified; 349 Scenario records, 182 eligible multivehicle scenes |',
                '| Real recurrent model trains and imagines autonomously | SUPPORTED (engineering) | 500 updates per variant; exact checkpoint regeneration and future-mutation leakage tests |',
                f'| Joint model improves over independent dynamics | NOT SUPPORTED in this pilot | 4 s energy reduction {p["paired_energy_reduction_m"]:.6f} m; provisional CI {p["ci95_provisional"]}; confirmatory five-recording-group gate UNTESTED |',
                f'| Full model beats constant velocity point forecasts | NOT SUPPORTED | 4 s FDE {full.fde_m:.6f} vs {dense.loc["constant_velocity"].fde_m:.6f} m |',
                f'| Generative calibration is adequate | NOT SUPPORTED | 4 s 90% marginal coverage {full.coverage90:.6f} |',
                '| Graph and recurrent state carry predictive information | PARTIALLY SUPPORTED | Same-checkpoint graph-off/reset errors increase; small one-date sample, no independent-session confirmation |',
                '| Real demo uses generated trajectories | SUPPORTED (engineering) | Exact saved evaluation arrays, checkpoint/source hashes, browser and anti-replay/regeneration tests |',
                '| Population micro–macro information value | UNTESTED | Curated cohorts are not a census; macro feedback/loss/scientific fields disabled |',
                '| Open-system arrivals/exits | UNTESTED | Fixed historical cohort, target-validity-masked evaluation only |',
                '| Independent-day/corridor generalization | UNTESTED | One source date; missing absolute origin times and possible aliases |',
                '| Long-horizon waves, AV orchestration, Cavnue control | UNTESTED | Nine-second passive clips; no intervention or authorized partner data |']
        Path('docs/I24_MSD_CLAIM_LEDGER.md').write_text('\n'.join(ledger)+'\n')
