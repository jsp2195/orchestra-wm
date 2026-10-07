import argparse
from pathlib import Path
from orchestra_wm.utils.config import load_config
from orchestra_wm.evaluation.action_sensitivity import action_audit
from orchestra_wm.data.collection import collect
from orchestra_wm.data.diagnostics import diagnose

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/smoke.yaml');a=p.parse_args()
    c=load_config(a.config);out=Path(c['output']);out.mkdir(parents=True,exist_ok=True)
    action_audit(c,out/'action_signal_audit.csv')
    collect(c,out/'data.npz');print(diagnose(out/'data.npz',out/'data_diagnostics.json'))
