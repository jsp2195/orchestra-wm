import argparse,json
from orchestra_wm.i24_phase4d.training import train
p=argparse.ArgumentParser();p.add_argument('--cache',default='outputs/i24_phase4d/cache_v1');p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=101);p.add_argument('--max-updates',type=int,default=800);p.add_argument('--wall-seconds',type=int,default=1800);a=p.parse_args()
print(json.dumps(train(a.cache,a.output,a.seed,a.max_updates,a.wall_seconds),indent=2))
