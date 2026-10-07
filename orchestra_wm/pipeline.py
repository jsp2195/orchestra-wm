"""Restartable experiment orchestration with source/config fingerprints."""
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import torch
import yaml
from orchestra_wm.utils.config import load_config
from orchestra_wm.utils.seed import seed_everything
from orchestra_wm.evaluation.common import write_json


def fingerprint(cfg):
    digest=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode())
    for root in ['orchestra_wm','scripts','tests','configs']:
        for path in sorted(Path(root).rglob('*')):
            if path.suffix in ['.py','.yaml']: digest.update(path.read_bytes())
    return digest.hexdigest()


def validate_artifacts(out):
    required=['data.npz','action_signal_audit.csv','data_diagnostics.json','training.csv','prediction.csv',
        'action_shuffle.csv','interactions.csv','memory.csv','rank_fidelity.csv','planning.csv','generalization.csv',
        'scaling.csv','demo.html','demo.gif','metrics.json','EXPERIMENT_SUMMARY.md','DECISION_SUFFICIENCY.md']
    for name in required:
        assert (out/name).is_file() and (out/name).stat().st_size>0,name
    assert len(list((out/'figures').glob('*.png')))>=20
    metrics=json.loads((out/'metrics.json').read_text())
    assert metrics['training']['orchestra']['steps']>0
    assert metrics['data']['transitions']>0
    return {'artifacts_checked':len(required),'figures':len(list((out/'figures').glob('*.png')))}


def run(config_path,clean=False,output=None):
    cfg=load_config(config_path)
    if output: cfg['output']=output
    if cfg['name']=='full': raise SystemExit('Full is intentionally disabled in the automated pipeline. Use explicit individual stage commands for a deliberate full experiment.')
    if cfg['name']=='small' and not (torch.cuda.is_available() or torch.backends.mps.is_available()):
        raise SystemExit('small requires an available GPU; use smoke on this CPU machine.')
    out=Path(cfg['output']);started=time.perf_counter()
    if clean and out.exists():
        archive=Path(tempfile.mkdtemp(prefix='orchestra-previous-'))/out.name
        shutil.move(str(out),archive);print('Previous outputs preserved at',archive,flush=True)
    out.mkdir(parents=True,exist_ok=True);seed_everything(cfg['seed'],cfg['threads'])
    (out/'resolved_config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    source=fingerprint(cfg);manifest_path=out/'pipeline_manifest.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if manifest.get('fingerprint')!=source: manifest={'fingerprint':source,'stages':{}}
    runtime={'python':platform.python_version(),'torch':torch.__version__,'hardware':platform.platform(),
        'cpu_count':os.cpu_count(),'threads':cfg['threads'],'cuda':torch.cuda.is_available(),'mps':torch.backends.mps.is_available(),
        'clean_output_run':bool(clean or not manifest.get('stages')),'source_fingerprint':source}
    test=subprocess.run([sys.executable,'-m','pytest','-q'],capture_output=True,text=True)
    (out/'tests.log').write_text(test.stdout+test.stderr);print(test.stdout,flush=True)
    if test.returncode: raise RuntimeError('Tests failed; see tests.log')
    runtime['test_result']=test.stdout.strip().splitlines()[-1]
    def stage(name,fn,artifacts):
        if name in manifest['stages'] and all((out/a).exists() for a in artifacts):
            print('Reuse completed stage:',name,flush=True);return
        print('Stage:',name,flush=True);now=time.perf_counter();fn()
        for artifact in artifacts:
            if not (out/artifact).exists(): raise RuntimeError(f'Missing stage artifact: {artifact}')
        manifest['stages'][name]={'seconds':time.perf_counter()-now,'artifacts':artifacts}
        write_json(manifest_path,manifest)
    from orchestra_wm.evaluation.action_sensitivity import action_audit
    from orchestra_wm.data.collection import collect
    from orchestra_wm.data.diagnostics import diagnose
    from orchestra_wm.models.training import train,VARIANTS
    from orchestra_wm.evaluation.run import evaluate
    from orchestra_wm.evaluation.plots import generate_plots
    from orchestra_wm.evaluation.demo import make_demo
    from orchestra_wm.evaluation.report import report
    stage('audit',lambda:action_audit(cfg,out/'action_signal_audit.csv'),['action_signal_audit.csv'])
    shutil.copyfile(out/'action_signal_audit.csv',Path('outputs/action_signal_audit.csv'))
    def data():
        collect(cfg,out/'data.npz');diagnose(out/'data.npz',out/'data_diagnostics.json')
    stage('data',data,['data.npz','data_diagnostics.json'])
    stage('train',lambda:train(cfg,out/'data.npz',out),['training.csv','training_metadata.json']+[f'checkpoints/{v}.pt' for v in VARIANTS])
    stage('evaluate',lambda:evaluate(cfg,out),['prediction.csv','action_shuffle.csv','rank_fidelity.csv','planning.csv','generalization.csv','scaling.csv','memory.csv','interactions.csv'])
    stage('figures',lambda:generate_plots(cfg,out),['figures/performance_summary.png'])
    stage('demo',lambda:make_demo(cfg,out),['demo.html','demo.gif','demo_data.json'])
    runtime['elapsed_seconds']=time.perf_counter()-started;runtime['stage_timings']=manifest['stages']
    # Retain the complete run timing when resuming a finished experiment.
    if manifest.get('completed_runtime') and not clean:
        runtime['resume_seconds']=runtime['elapsed_seconds'];runtime['elapsed_seconds']=manifest['completed_runtime']['elapsed_seconds']
        runtime['clean_output_run']=manifest['completed_runtime']['clean_output_run']
    metrics=report(cfg,out,runtime)
    manifest['validation']=validate_artifacts(out);manifest['completed_runtime']=runtime
    write_json(manifest_path,manifest)
    print('Complete:',out/'EXPERIMENT_SUMMARY.md',flush=True)
    return metrics
