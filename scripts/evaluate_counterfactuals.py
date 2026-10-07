import argparse
from pathlib import Path
from orchestra_wm.utils.config import load_config
from orchestra_wm.models.training import load_model,VARIANTS
from orchestra_wm.evaluation.counterfactuals import evaluate_counterfactuals
from orchestra_wm.utils.seed import seed_everything
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/smoke.yaml');a=p.parse_args();c=load_config(a.config)
    seed_everything(c['seed'],c['threads']);out=Path(c['output'])
    evaluate_counterfactuals(c,{v:load_model(out/'checkpoints'/f'{v}.pt') for v in VARIANTS},out)
