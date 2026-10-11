import argparse,json
from orchestra_wm.i24_phase4b.training import train
p=argparse.ArgumentParser()
p.add_argument('--data-output',default='outputs/i24_phase4b/development_v1')
p.add_argument('--output',default='outputs/i24_phase4b/matched_macro_seed101_u300')
p.add_argument('--preservation-receipt',required=True)
p.add_argument('--updates',type=int,default=300)
p.add_argument('--wall-seconds',type=int,default=1800)
p.add_argument('--seed',type=int,default=101)
p.add_argument('--variants',nargs='+',default=['full','macro_only'])
a=p.parse_args()
print(json.dumps(train(a.data_output,a.output,a.preservation_receipt,a.updates,tuple(a.variants),a.seed,a.wall_seconds),indent=2))
