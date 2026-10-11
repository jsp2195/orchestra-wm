import argparse
import json
from orchestra_wm.i24_phase4b.data import prepare

p=argparse.ArgumentParser()
p.add_argument('--config',default='configs/i24_phase4b_development.json')
p.add_argument('--output',default='outputs/i24_phase4b/development_v1')
a=p.parse_args()
r=prepare(a.config,a.output)
print(json.dumps({k:v for k,v in r.items() if k not in ('scenes','artifacts')},indent=2))
