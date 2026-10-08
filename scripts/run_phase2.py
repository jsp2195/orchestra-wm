"""Restartable execution of the fixed Phase-2 campaign."""
import argparse
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import time
from orchestra_wm.utils.config import load_config
from orchestra_wm.utils.seed import seed_everything
from orchestra_wm.phase2.audits import preserve_manifest,verify_preservation,reproduce,action_choices,coverage,factorial_audit
from orchestra_wm.phase2.data import collect_factorial
from orchestra_wm.phase2.training import train_phase2
from orchestra_wm.phase2.evaluation import evaluate_models,evaluate_mpc,evaluate_memory
from orchestra_wm.phase2.statistics import compute_statistics,supplemental_statistics
from orchestra_wm.phase2.reporting import report
from orchestra_wm.evaluation.common import write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/phase2.yaml');args=parser.parse_args()
    cfg=load_config(args.config);out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True);seed_everything(cfg['seed'],cfg['threads'])
    times={};start=time.perf_counter()
    def stage(name,fn):
        now=time.perf_counter();result=fn();times[name]=time.perf_counter()-now;write_json(out/'pipeline_timing.json',times);return result
    stage('tests',lambda:subprocess.run([sys.executable,'-m','pytest','-q'],check=True))
    preserve_manifest(out)
    def audits():
        models=reproduce(out);action_choices(models,out);coverage(out);factorial_audit(cfg,models,out)
    stage('audits',audits);decision=json.loads((out/'architecture_decision.json').read_text())
    stage('data',lambda:collect_factorial(cfg,out))
    names=stage('training',lambda:train_phase2(cfg,out,decision['explicit_variant_eligible']))
    # Fail closed rather than silently reusing outcomes after evaluation semantics change.
    sources=['orchestra_wm/phase2/evaluation.py','orchestra_wm/phase2/planning.py','orchestra_wm/phase2/scenarios.py']
    digest=hashlib.sha256(b''.join(Path(p).read_bytes() for p in sources)+json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    manifest=out/'evaluation_manifest.json'
    checkpoints={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((out/'checkpoints').glob('*.pt'))}
    if manifest.exists():
        old=json.loads(manifest.read_text())
        assert old['source_config_sha256']==digest,'Evaluation changed: archive old evaluation artifacts explicitly before rerunning'
        assert old.get('checkpoint_sha256',checkpoints)==checkpoints,'Evaluation checkpoint mismatch'
    write_json(manifest,{'source_config_sha256':digest,'checkpoint_sha256':checkpoints,'config':cfg,'seeds':cfg['training_seeds']})
    stage('prediction',lambda:evaluate_models(cfg,out,names));stage('planning',lambda:evaluate_mpc(cfg,out,names))
    stage('memory',lambda:evaluate_memory(cfg,out) if not (out/'memory_coordination.csv').exists() else None)
    gates=stage('statistics',lambda:compute_statistics(cfg,out));supplemental_statistics(out)
    if gates['INT-H']['pass']:
        raise RuntimeError('INT-H passed: execute the gated limited OOD extension before publishing final reports')
    else:write_json(out/'generalization.json',{'status':'UNTESTED','reason':'Preregistered gate INT-H failed; OOD not executed.'})
    times['total_this_invocation']=time.perf_counter()-start
    stage('report',lambda:report(cfg,out,times))
    stage('validation',lambda:validate(out,cfg,names))
    times['total_this_invocation']=time.perf_counter()-start
    write_json(out/'pipeline_timing.json',times)
    metrics=json.loads((out/'metrics.json').read_text());metrics['runtime']=times;write_json(out/'metrics.json',metrics)
    print('Phase-2 pipeline complete; INT-H:',gates['INT-H'],flush=True)


def validate(out,cfg,names):
    import numpy as np
    import pandas as pd
    hashes=verify_preservation(out)
    for filename in ['interaction_prediction','interaction_rank','joint_shuffle','factorial_plan_costs','mpc_results','mpc_actions','coordination_gain','seed_summary']:
        frame=pd.read_csv(out/f'{filename}.csv');assert np.isfinite(frame.select_dtypes(include='number')).all().all(),filename
    pred=pd.read_csv(out/'interaction_prediction.csv');assert len(pred)==len(names)*len(cfg['training_seeds'])*cfg['evaluation_families']
    metrics=json.loads((out/'metrics.json').read_text());assert all(Path(p).is_file() for p in metrics['figures'])
    split=json.loads((out/'dataset_v2_factorial/test_split.json').read_text());assert len(split)==cfg['evaluation_families'] and all(x['interaction_relevant'] for x in split)
    write_json(out/'validation.json',{'pass':True,'phase1_files_preserved':hashes,'finite_measurements':True,'prediction_rows':len(pred),'figures':len(metrics['figures']),'test_families':len(split)})

if __name__=='__main__':main()
