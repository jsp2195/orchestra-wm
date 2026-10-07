import argparse
from pathlib import Path
from orchestra_wm.utils.config import load_config
from orchestra_wm.utils.seed import seed_everything
from orchestra_wm.evaluation.demo import make_demo
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/smoke.yaml');a=p.parse_args()
    c=load_config(a.config);seed_everything(c['seed'],c['threads']);print(make_demo(c,Path(c['output'])))
