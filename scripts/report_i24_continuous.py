"""Render measured original-I24 pilot stages and exact reproducible commands."""
import argparse
import json
from pathlib import Path
import shutil

import pandas as pd


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.output)
    source=json.loads((root/'source_audit.json').read_text());qc=json.loads((root/'authentic_qc.json').read_text())
    data=json.loads((root/'data_manifest.json').read_text());training=json.loads((root/'training_metadata.json').read_text())
    summary=json.loads((root/'scientific_summary.json').read_text());archives=json.loads((root/'archive_inventory.json').read_text())
    table=pd.read_csv(root/'baseline_table.csv');dense=table[(table.regime=='dense')&(table.horizon_seconds==10)]
    lines=['# Original continuous I-24: authentic preliminary pilot','',
           'The private original archive is downloaded and integrity-verified. A fixed congested regional recording passed authentic-data QC and supplied all training and evaluation data. No Harvard data or synthetic fixture was substituted.','',
           '| Stage | Achieved |','|---|---|',
           '| Download | Drive metadata/media HTTP 200; exact byte count and Drive MD5; SHA256 recorded; outer and nested member CRCs verified |',
           '| Authentic source | Original INCEPTION JSON array, 2022-11-21, both directions, one session, about four hours, native 25 Hz |',
           '| Bounded selection | 07:00–07:10 CST westbound, 999.744 m regional extent; only 121,778,906 JSON bytes extracted |',
           '| QC | 6,078 tracks, 2,180,391 samples; all tracks accepted, 72,210 samples marked unusable; minimum 120 usable concurrent tracks per 5 Hz frame |',
           '| Training data | 13/2/3 whole 25 s train/validation/test windows; track-disjoint, 25 s guards; no full-source RAM load |',
           '| Real models | Six variants, seed 101, exactly 100 updates each, CPU/two threads; final checkpoints hashed |',
           '| Evaluation | 5/10/20 physical seconds; two kinematic and six learned baselines; dense, sparse50, sparse20, blind and outage |',
           '| Visualization | Native real-traffic congestion diagrams, actual checkpoint forecast GIF, standalone interactive saved-array demo |',
           '| Confirmatory research | NOT ACHIEVED: only one recording day/block, 100-update optimization, no independently labeled wave events |','',
           '## Archive integrity','', '| File | Bytes | MD5 | SHA256 |','|---|---:|---|---|']
    for archive in archives['archives']:
        lines.append(f"| {archive['name']} | {archive['bytes']} | {archive['md5']} | {archive['sha256']} |")
    lines+=['','The nested JSON would occupy 20,130,434,353 bytes. It was streamed through CRC validation instead of fully extracted. The outer ZIP, the nested trajectory ZIP, auxiliary ZIP, and measured acquisition/provenance manifests remain under ignored `data/i24_continuous/raw/`. No credentials are stored.','',
            '## Source and scope','',
            f"Session `{qc['session']}`. Full source: {source['counts']['source_tracks']:,} records and {source['counts']['source_samples']:,} samples, {source['source_duration_s']:.3f} seconds. Timestamp correction semantics come from public v1.x documentation; raw synchronization corrections are not independently reproducible from these trajectories.",'',
            f"Median usable speed: {qc['usable_speed_m_s_quantiles'][2]:.3f} m/s. {qc['tracks_with_25s_usable_span']} tracks have a usable span of at least 25 s. Roadway-coordinate inverse checks gave x error {qc['coordinate_roundtrip_max_error_ft']['x']} ft and y error {qc['coordinate_roundtrip_max_error_ft']['y']:.3g} ft. Coordinates are curvilinear SI, not geographic XY; no unsupported WGS84 calibration is asserted.",'',
            'Observed support is strongly thinned around s=870–890 m (only 6–7% of frames contain reconstructed positions in those 10 m bins). Internal source flags include Overlap/Lost/Anomalous state and are not calibrated sensor confidence. Offline reconstruction can use future imagery; identity accuracy and unbiased census coverage are unverified.','',
            f"Track-disjoint preparation omitted {data['counts']['crossing_or_guard_tracks_omitted']:,} crossing/guard tracks and {data['counts']['crossing_or_guard_samples_omitted']:,} samples; 44 empty/short in-interval tracks were excluded. Macro targets describe the retained regional reconstructed population, not full unbiased freeway occupancy.",'',
            'Speed, density and flow use causal trailing one-second vehicle-time/distance exposure at 5 Hz. Empty support remains UNKNOWN. Model consistency uses soft spatial assignment with the same temporal exposure; it is not current-count times current-speed. Binary usable-sample indicators replace the old fixed 0.5 confidence. No future roster or realized arrivals initialize forecasts; microscopic scores use the past-visible fixed cohort.','',
            '## Measured pilot results','',
            '| Model | Updates | Final autonomous validation objective | Parameters |','|---|---:|---:|---:|']
    for model in training:
        lines.append(f"| {model['variant']} | {model['updates']} | {model['final_validation_loss']:.6f} | {model['parameters']} |")
    lines+=['','Variant-specific objectives differ; these validation objectives are not direct comparative forecast scores. Finite loss establishes execution, not convergence.','',
            '| Dense, 10 s model | Joint macro energy score | Micro FDE (m) |','|---|---:|---:|']
    for _,row in dense.iterrows():
        fde='N/A' if pd.isna(row.fde_m) else f'{row.fde_m:.3f}'
        lines.append(f'| {row.variant} | {row.macro_energy:.6f} | {fde} |')
    full=dense[dense.variant=='full'].iloc[0];cv=dense[dense.variant=='constant_velocity'].iloc[0]
    lines+=['',f"Full-model dense 10 s macro 90% interval coverage was only {full.macro_coverage90:.1%}; calibration is not achieved. Its mean micro FDE was {full.fde_m:.3f} m versus constant-velocity {cv.fde_m:.3f} m. This pilot does not demonstrate microscopic forecast superiority either."]
    lines+=['',f"Primary descriptive mean (macro-only minus full, 10 s): **{summary['mean_difference']:.6f}**. Positive favors full; negative favors macro-only. Paired window differences: {summary['paired_differences']}. No session-bootstrap confidence interval or confirmed information-value claim is reported from one contiguous held-out block.",'',
            'Physical score scales remain [30 m/s, 100 veh/km/lane, 3000 veh/h/lane]. Eight autonomous samples per stochastic forecast; valid counts, coverage and physical violations are in `evaluation.csv`. One training seed and three test windows are preliminary. Wave-onset skill, cross-day generalization, IDM leader calibration, memory-reset and residual-permutation diagnostics remain untested in this bounded campaign.','',
            '## Exact resume in the current preserved workspace','',
            'These commands verify/reuse completed checkpoints and immutable evaluation. They do not repeat the large download.','',
            '```sh','cd /workspace/orchestra-wm',
            'uv run python scripts/train_i24_continuous.py --data-output outputs/i24_continuous/pilot_0700_wb --output outputs/i24_continuous/pilot_0700_wb --updates 100 --seed 101 --wall-seconds 1800',
            'uv run python scripts/evaluate_i24_continuous.py --data-output outputs/i24_continuous/pilot_0700_wb --output outputs/i24_continuous/pilot_0700_wb --samples 8',
            'uv run python scripts/visualize_i24_continuous.py --data-root data/i24_continuous/raw/pilot_0700_wb --output outputs/i24_continuous/pilot_0700_wb --with-forecast',
            'uv run python scripts/report_i24_continuous.py --output outputs/i24_continuous/pilot_0700_wb','```','',
            '## Rebuild if a fresh task has no ignored source files','',
            'The worker uses an overlay filesystem with no identified persistent dataset mount. Files persisted within this environment, but cross-task persistence is not established. Git stores implementation/reports, not raw data, canonical arrays or checkpoints. Earlier Harvard archives/checkpoints were already absent from this workspace at task start; all prior tracked scientific files remain preserved. Preserve/mount this worker directory for checkpoint continuation. A fresh checkout alone cannot recover trained weights.','',
            'The downloader preserves/reuses existing archives and identified partials. It requests HTTP206 with exact Content-Range to resume. HTTP401 stops without retries; refresh the secure GOOGLE_DRIVE_ACCESS_TOKEN binding and publish, then rerun. Never paste the credential. If files are absent, the transfer must start from zero; do not discard present bytes.','',
            '```sh',
            'uv run python scripts/acquire_i24_continuous_drive.py --data-root data/i24_continuous/raw --max-download-bytes 6300000000',
            'uv run python scripts/inspect_i24_continuous_archives.py --data-root data/i24_continuous/raw',
            'uv run python scripts/prepare_i24_continuous_source.py --archive data/i24_continuous/raw/637b023440527bf2daa5932f__post1.zip --destination data/i24_continuous/raw/pilot_0700_wb --start 1669035600 --seconds 600 --x-min-ft 316800 --x-max-ft 320080',
            'uv run python scripts/qc_i24_continuous_source.py --data-root data/i24_continuous/raw/pilot_0700_wb --archive-root data/i24_continuous/raw --session 637b023440527bf2daa5932f__post1 --documentation-commit 9f8a9158ba0bcbfdd525f4be771f8d45ad1bd4df',
            'uv run python scripts/prepare_i24_continuous_training.py --data-root data/i24_continuous/raw/pilot_0700_wb --output outputs/i24_continuous/pilot_0700_wb_rebuilt --max-agents 512','```','',
            'Use the separately rebuilt data namespace and a new model-output namespace if original checkpoints are missing. Source/split/checkpoint fingerprints fail closed rather than overwriting different work. Do not treat regenerated arrays as recovered trained weights. The legacy import-only CLI still exits 2 after passing local schema QC; actual training uses the new explicitly real-data commands above.','',
            '## Artifacts','',
            '- `archive_inventory.json`, `source_audit.json`, `acquisition.json`, `provenance.json`, `authentic_qc.json`, `fine_roadway_coverage.json`: measured access/source/QC evidence.',
            '- `data_manifest.json`, `run_manifest.json`, `training.csv`, `training_metadata.json`: immutable preparation and real optimization evidence.',
            '- `evaluation.csv`, `baseline_table.csv`, `evaluation_manifest.json`, `scientific_summary.json`: matched checkpoint predictions and honest limitations.',
            '- `figures/real_traffic_time_space.png`, `figures/real_congestion_fields.png`: actual native-source diagrams.',
            '- `demo/index.html`, `demo/real_checkpoint_forecast.gif`, `demo/provenance.json`: saved forecast arrays with checkpoint hashes; no simulator/replay substitute.','',
            'Source citation: Gloudemans et al. (2023), I-24 MOTION: An instrument for freeway traffic science, Transportation Research Part C 155, 104311. Data use agreement and source documentation are preserved in the original archives / cited public documentation.']
    (root/'ORIGINAL_PILOT_REPORT.md').write_text('\n'.join(lines)+'\n')
    print(root/'ORIGINAL_PILOT_REPORT.md')


if __name__=='__main__':main()
