"""Train bounded authentic-data passive models; no fixture fallback."""
import argparse
import json
from orchestra_wm.i24_continuous.training import train


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-output',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--updates',type=int,default=100)
    p.add_argument('--seed',type=int,default=101)
    p.add_argument('--wall-seconds',type=int,default=1800)
    p.add_argument('--variants',nargs='+',default=['full','independent','deterministic','macro_only','micro_only','no_crossscale'])
    a=p.parse_args()
    print(json.dumps(train(a.data_output,a.output,a.updates,tuple(a.variants),a.seed,a.wall_seconds),indent=2))


if __name__=='__main__':main()
