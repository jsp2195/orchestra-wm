import argparse,json
from orchestra_wm.i24_phase4c.training import train_joint
p=argparse.ArgumentParser()
p.add_argument('--data-output',default='outputs/i24_phase4b/development_v1')
p.add_argument('--initial-output',default='outputs/i24_phase4b/matched_macro_seed101_u300')
p.add_argument('--output',default='outputs/i24_phase4c/joint_seed101_u30')
p.add_argument('--preservation-receipt',required=True)
p.add_argument('--updates',type=int,default=30)
p.add_argument('--seed',type=int,default=101)
p.add_argument('--wall-seconds',type=int,default=900)
a=p.parse_args()
print(json.dumps(train_joint(a.data_output,a.initial_output,a.output,a.preservation_receipt,a.updates,a.seed,a.wall_seconds),indent=2))
