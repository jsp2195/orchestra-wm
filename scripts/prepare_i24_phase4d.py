import argparse,json
from orchestra_wm.i24_phase4d.data import prepare
p=argparse.ArgumentParser();p.add_argument('--data-root',default='outputs/i24_phase4b/development_v1');p.add_argument('--output',default='outputs/i24_phase4d/cache_v1');a=p.parse_args()
r=prepare(a.data_root,a.output);print(json.dumps(dict(entries=len(r['entries']),config=r['config']),indent=2))
