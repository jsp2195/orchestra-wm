from pathlib import Path
import argparse,json,hashlib,time,platform,os,subprocess,signal
import numpy as np
import pandas as pd
import torch
from orchestra_wm.utils.config import load_config
from orchestra_wm.i24.data import SyntheticFixtureAdapter,real_scenes,split_scenes,save_scene,load_scene,fields_numpy,write_json,sha,observations
from orchestra_wm.i24.training import train_models,load_checkpoint
from orchestra_wm.i24.evaluation import cheap_baselines,evaluate,history_matching
from orchestra_wm.i24.reporting import make_figures,make_demo,verify_demo,make_gif
from orchestra_wm.i24.open_road import generate_open_road


def preserve():
    path=Path('outputs/i24/preservation.json')
    if not path.exists():raise ValueError('Missing protected-history manifest')
    checks=json.loads(path.read_text())
    for name,expected in checks.items():
        data=Path(name).read_bytes()
        if name=='README.md':data=data.split(b'\n<!-- PHASE3_I24 -->')[0]
        assert hashlib.sha256(data).hexdigest()==expected, f'Protected Phase 1/2 changed: {name}'
    return len(checks)


def qc(scenes,out,cfg):
    rows=[]
    for scene in scenes:
        valid=scene.existence;v=scene.values;speed=v[:,:,2][valid];f,_=fields_numpy(v,valid,scene.road)
        rows.append({'scene':scene.scene_id,'session':scene.session_id,'source_kind':scene.provenance['kind'],'tracks':len(scene.track_ids),'samples':int(valid.sum()),'speed_mean_mps':float(speed.mean()),'speed_p10_mps':float(np.quantile(speed,.1)),'congested_fraction_below_10mps':float((speed<10).mean()),'freeflow_fraction_above_20mps':float((speed>20).mean()),'density_mean_veh_km_lane':float(f[:,:,1].mean()),'lane_change_tracks':int((np.ptp(v[:,:,1],axis=0)>scene.road.lane_width*.5).sum()),'valid_track_fraction':float(valid.mean()),'verified_wave_events':0})
    pd.DataFrame(rows).to_csv(out/'data_qc/regimes.csv',index=False)
    write_json(out/'data_qc/wave_status.json',{'status':'BLOCKED' if cfg['source_kind']=='SYNTHETIC_FIXTURE' else 'UNTESTED','verified_onset_events':0,'reason':'No independently verified real onset/event annotations; slow-speed prevalence is not a wave-onset label.'})


def report(cfg,out,metadata,frame,figures,demo,timing,preserved):
    fake=cfg['source_kind']=='SYNTHETIC_FIXTURE';status='BLOCKED' if fake else 'PRELIMINARY';primary=json.loads((out/'confidence_intervals.json').read_text());summary=frame[(frame.regime=='dense')&(frame.horizon_seconds==10)].groupby('variant').mean(numeric_only=True)
    summary.to_csv(out/'baseline_table.csv');paired=frame[(frame.variant=='full')&(frame.horizon_seconds==10)].groupby('regime').mean(numeric_only=True);paired.to_csv(out/'paired_ablations.csv')
    metrics={'source_kind':cfg['source_kind'],'real_data_status':status,'primary_information_value':primary,'models':metadata,'baseline_10s':summary.reset_index().replace({np.nan:None}).to_dict('records'),'ablation_10s':paired.reset_index().replace({np.nan:None}).to_dict('records'),'hardware':{'cpu':next((l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')),platform.machine()),'python':platform.python_version(),'torch':torch.__version__,'threads':cfg['threads'],'device':'cpu','cuda_available':torch.cuda.is_available()},'runtime_seconds':timing,'config':cfg,'preserved_files':preserved,'figures':figures,'demo':demo,'wave_onset_status':'BLOCKED' if fake else 'UNTESTED','cross_day_generalization':'BLOCKED' if fake else 'UNTESTED','convergence':'Not established; fixed-budget fixture optimization is integration evidence only.' if fake else 'See validation curve; no automatic convergence claim.'}
    write_json(out/'metrics.json',metrics)
    full=summary.loc['full'];kind='SYNTHETIC FIXTURE — NOT AN I-24-TRAINED MODEL' if fake else 'REAL I-24 preliminary pilot'
    lines=[f'# ORCHESTRA-I24 — {kind}', '', 'Real-data autonomous generation, wave modeling, microscopic information value, and real-Cavnue transfer are '+status+'.', '',f'Protected Phase-1/2 checks: {preserved}. Phase-2 coordination gain remains −0.132126, CI [−0.700616, 0.436365]; no reinterpretation.', '',f'Fixture/full 10 s FDE: {full.fde_m:.6f} m; macro energy score: {full.macro_energy:.6f}; trajectory 90% coverage: {full.trajectory_coverage90:.6f}; overlap pair fraction: {full.overlap_pair_fraction:.6f}. These numbers do not measure I-24 skill.', '', '## Training and evaluation', '',f'{len(metadata)} model/seed runs, {cfg["training_steps"]} optimizer updates each, seeds {cfg["training_seeds"]}, CPU threads {cfg["threads"]}. Full model parameters: {[m["parameters"] for m in metadata if m["variant"]=="full"]}. Total measured training time: {sum(m["seconds"] for m in metadata):.2f} seconds. Updates sample scenes; they are not full dataset epochs. Validation curves and all final checkpoint hashes are recorded. Convergence is not established.', '', 'The six trained variants are independent, deterministic relational, macro-only, micro-only, no-cross-scale, and full. Kinematic baselines run before training. All forecasts are autonomous. Information-value and architecture-value comparisons remain distinct. Stochastic prior innovations and decoder variances are trained, not arbitrary post-hoc noise. The training-only posterior has no inference interface.', '',f'10 s paired macro energy-score reduction (macro-only minus full): {primary["mean"]:.6f}; cluster interval [{primary["ci_low"]}, {primary["ci_high"]}], {primary["clusters"]} clusters. Fixture scene seeds are not real recording days; one training seed is preliminary engineering evidence.', '', 'Mode definitions: fixed cohort uses only historically detected identities. Macro targets include the full bounded source population; this is harder than reconstructing visible fields. Open-road generation uses a separately labeled past-appearance-rate Poisson inflow assumption, with no future identities. It is not validated inflow learning. Unknown sampling coverage means reconstructed density is not an unbiased census of real traffic.', '', 'Partial observations are EMULATED. The same checkpoint is evaluated under dense, sparse50/sparse20, blind-zone, outage, reset-memory, graph-off and kinematic-residual permutation controls. Macro-only agent outputs are explicitly N/A. Zero valid-agent denominators are recorded and must not be mistaken for successful forecasts. No real wave events or cross-day OOD were evaluated. IDM calibration is not applicable to this fixture campaign.', '', '## Artifacts', '',f'- Interactive demo: [{demo}](demo/index.html)', '- Numerical forecasts: rollouts/*.npz (ignored from Git); demo payload retains the displayed evaluation arrays and provenance.', '- Metrics: metrics.json; evaluation.csv; baseline_table.csv; paired_ablations.csv; confidence_intervals.json; stratified_metrics.csv.', '- QC/split hashes: data_qc/; split_manifest.json; run_manifest.json.', '- Figures: figures/ (fixture labels on plots).', '', 'The fixture demo fails the real-data showcase gate by design. HTML controls select saved autonomous checkpoint predictions; revealing truth only affects the evaluation view.', '', '## Resume with authorized real data', '', 'Prepare acquisition.json as documented in docs/I24_DATA_ACQUISITION.md and place the official JSON files under the same directory, then run:', '', '```sh', 'uv run python scripts/run_i24_pipeline.py --config configs/i24_pilot.yaml --data-root /path/to/I24MOTION_PUBLIC', '```', '', 'No login bypass, substitute one-vehicle CAN/GPS dataset, I24-3D subset, or invented Cavnue data was used. Real-source adapter execution and real clip geometry/QC remain blocked until the files arrive.', '', 'NEXT EXPERIMENT: Run the preregistered 10-second macro energy-score comparison on an authorized congested I-24 multivehicle subset, with matched macro histories and measured micro headway differences.']
    (out/'EXPERIMENT_SUMMARY.md').write_text('\n'.join(lines)+'\n')
    if out==Path('outputs/i24'):
        ledger='# I24 claim ledger\n\nAll current numbers are SYNTHETIC_FIXTURE results. Phase-1/2 ledger is unchanged.\n\n| Claim | Status | Evidence |\n|---|---|---|\n'
        for claim in ['Real autonomous multi-agent generation','Micro information improves real macro prediction','Relational architecture advantage on real data','Real probabilistic calibration','Real traffic-wave/onset modeling','Real partial-roadside robustness','Real cross-day / Cavnue transfer']:
            ledger+=f'| {claim} | BLOCKED | No authentic source files; outputs/i24/DATA_ACCESS_MANIFEST.json |\n'
        ledger+='| Autonomous stochastic fixture integration | SUPPORTED (engineering only) | i24_smoke.yaml; seed 101; metrics.json, training_metadata.json, demo/provenance.json; no real-data claim |\n'
        Path('docs/I24_CLAIM_LEDGER.md').write_text(ledger)
        base=Path('README.md').read_bytes().split(b'\n<!-- PHASE3_I24 -->')[0]
        addition='\n<!-- PHASE3_I24 -->\n\n## Phase 3 — ORCHESTRA-I24\n\nA separately namespaced passive, stochastic highway world-model extension. Official I-24 MOTION data access is currently **BLOCKED**; only a conspicuously labeled synthetic fixture has been trained and evaluated. Phase 2 remains a negative centralized-coordination result. No causal AV action, real safety, real wave skill or Cavnue compatibility claim is made.\n\n```sh\nuv run python scripts/run_i24_pipeline.py --config configs/i24_smoke.yaml\nuv run python scripts/run_i24_pipeline.py --config configs/i24_pilot.yaml --data-root /path/to/I24MOTION_PUBLIC\n```\n\n[Fixture demo](outputs/i24/demo/index.html) · [Measured report](outputs/i24/EXPERIMENT_SUMMARY.md) · [Data acquisition](docs/I24_DATA_ACQUISITION.md) · [Evaluation protocol](docs/I24_EVALUATION_PROTOCOL.md) · [Model card](docs/I24_MODEL_CARD.md) · [Claim ledger](docs/I24_CLAIM_LEDGER.md). The research config is gated and never launched automatically. The demo must be downloaded/opened in a browser; GitHub does not execute HTML previews.\n'
        Path('README.md').write_bytes(base+addition.encode())
    return metrics


def run(config,data_root=None,pilot_gate=None):
    cfg=load_config(config);out=Path(cfg['output']);assert out.is_relative_to('outputs/i24');out.mkdir(parents=True,exist_ok=True);(out/'data_qc').mkdir(exist_ok=True);start=time.perf_counter();preserved=preserve()
    existing_checkpoints=len(list((out/'checkpoints').glob('*.pt'))) if (out/'checkpoints').exists() else 0
    torch.set_num_threads(cfg['threads']);torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
    if cfg.get('research_gate_required'):
        if not pilot_gate:raise ValueError('Research gated: supply --pilot-gate after real pilot QC/scientific review and resource headroom audit')
        gate=json.loads(Path(pilot_gate).read_text())
        if gate.get('source_kind')!='REAL_I24' or not gate.get('approved_resource_audit') or not gate.get('pilot_passed'):raise ValueError('Insufficient real pilot gate')
    if cfg['source_kind']=='REAL_I24' and not data_root:
        write_json(out/'BLOCKED.json',{'status':'BLOCKED','reason':'Authorized official multivehicle files and acquisition.json required','resume':'uv run python scripts/run_i24_pipeline.py --config configs/i24_pilot.yaml --data-root /path/to/I24MOTION_PUBLIC'})
        raise ValueError('REAL data gate blocked. Supply --data-root with acquisition.json; no fixture substitution.')
    def deadline(signum,frame):raise TimeoutError('I24 wall-time limit reached; resume from atomic checkpoints')
    signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,cfg['max_wall_seconds'])
    if cfg['source_kind']=='SYNTHETIC_FIXTURE':scenes=SyntheticFixtureAdapter().generate(cfg['fixture_scenes'],cfg['dt'],cfg['fixture_duration'],cfg['seed'])
    else:
        scenes,conversion=real_scenes(data_root,cfg,out/'converted')
        write_json(out/'DATA_ACCESS_MANIFEST.json',{'status':'REAL ACCESSED','release':'I24MOTION_PUBLIC_v1.0','files':conversion['sources'],'file_count':len(conversion['sources']),'total_bytes':sum(x['bytes'] for x in conversion['sources']),'citation':conversion['citation'],'provenance':'User-declared authorized official acquisition; content schema and hashes validated'})
    # Columnar serialization is exercised in smoke too; training consumes the canonical roundtrip.
    paths=[]
    for i,scene in enumerate(scenes):
        path=out/'canonical'/f'scene-{i:05}.parquet';save_scene(scene,path);paths.append(path)
    scenes=[load_scene(p) for p in paths];splits=split_scenes(scenes);qc(scenes,out,cfg)
    splitmanifest={k:[{'scene':s.scene_id,'session':s.session_id,'start':float(s.time[0]),'end':float(s.time[-1]),'source_hash':s.provenance['source_hash']} for s in v] for k,v in splits.items()}
    path=out/'split_manifest.json'
    if path.exists() and json.loads(path.read_text())!=splitmanifest:raise ValueError('Immutable split changed; use a new output namespace')
    write_json(path,splitmanifest)
    sources=[Path('orchestra_wm/i24')/p for p in ['schema.py','data.py','model.py','training.py']]
    fingerprint=hashlib.sha256(b''.join(p.read_bytes() for p in sources)+json.dumps(cfg,sort_keys=True).encode()+path.read_bytes()).hexdigest()
    write_json(out/'run_manifest.json',{'fingerprint':fingerprint,'source_hashes':{str(p):sha(p) for p in sources},'evaluation_source_hashes':{str(p):sha(p) for p in Path('orchestra_wm/i24').glob('*') if p.is_file()},'split_sha256':sha(path),'source_kind':cfg['source_kind'],'config':cfg,'epochs':'scene-sampling updates, not full epochs','resources':{'device':'cpu','threads':cfg['threads'],'workers':0,'scratch_cap_bytes':2*1024**3,'max_wall_seconds':cfg['max_wall_seconds']}})
    baseline_rows=cheap_baselines(cfg,splits['test'],out,splits['train'])
    metadata=train_models(cfg,splits,out,fingerprint,start)
    frame=evaluate(cfg,splits['test'],out,baseline_rows);matches=history_matching(splits['test'],cfg,out)
    model,_=load_checkpoint(out/'checkpoints'/f'full-{cfg["training_seeds"][0]}.pt');scene=splits['test'][0];obs,_=observations(scene,round(cfg['context_seconds']/cfg['dt']));_,open_meta=generate_open_road(model,obs,scene.road,round(5/cfg['dt']));write_json(out/'open_road_assumptions.json',open_meta)
    figures=make_figures(out,splits['test'],cfg);demo=make_demo(out,cfg);make_gif(out,cfg);entries=verify_demo(out)
    metrics=report(cfg,out,metadata,frame,figures,demo,time.perf_counter()-start,preserved)
    # N/A macro-only micro metrics are explicit null; every defined metric must be finite.
    numeric=frame.select_dtypes(include='number');assert np.isfinite(numeric.fillna(0)).all().all();assert all(Path(p).is_file() for p in figures);assert preserve()==preserved
    write_json(out/'validation.json',{'pass':True,'preserved_files':preserved,'demo_entries_exactly_match_evaluation':entries,'defined_metrics_finite':True,'split_counts':{k:len(v) for k,v in splits.items()},'history_matched_pairs':matches,'real_data_gate':'BLOCKED' if cfg['source_kind']=='SYNTHETIC_FIXTURE' else 'REAL_SOURCE_LOADED'})
    print(json.dumps({'completed':cfg['name'],'real_data_status':metrics['real_data_status'],'seconds':metrics['runtime_seconds'],'demo':demo}),flush=True)
    history_path=out/'execution_history.json'
    history=json.loads(history_path.read_text()) if history_path.exists() else []
    history.append({'runtime_seconds':metrics['runtime_seconds'],'existing_checkpoints_at_start':existing_checkpoints,'model_count':len(metadata),'training_seconds_in_checkpoints':sum(m['seconds'] for m in metadata),'fingerprint':fingerprint,'source_kind':cfg['source_kind']})
    write_json(history_path,history)
    signal.setitimer(signal.ITIMER_REAL,0)
    return metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/i24_smoke.yaml');p.add_argument('--data-root');p.add_argument('--pilot-gate');a=p.parse_args();run(a.config,a.data_root,a.pilot_gate)
