import argparse
from pathlib import Path
from orchestra_wm.utils.config import load_config
from orchestra_wm.models.training import train
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/smoke.yaml');a=p.parse_args()
    c=load_config(a.config);out=Path(c['output'])
    train(c,out/'data.npz',out,['independent','no_actions','privileged','memoryless'])
