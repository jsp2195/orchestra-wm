import argparse
from pathlib import Path
from orchestra_wm.utils.config import load_config
from orchestra_wm.models.training import train
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/smoke.yaml');p.add_argument('--all',action='store_true');a=p.parse_args()
    c=load_config(a.config);out=Path(c['output'])
    if a.all: train(c,out/'data.npz',out)
    else: train(c,out/'data.npz',out,['orchestra'])
