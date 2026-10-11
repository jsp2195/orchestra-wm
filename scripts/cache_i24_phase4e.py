import argparse,json
from orchestra_wm.i24_phase4e.cache import prepare
p=argparse.ArgumentParser();p.add_argument('--data',default='outputs/i24_phase4e/data_v2');p.add_argument('--output',default='outputs/i24_phase4e/cache_v1')
a=p.parse_args();m=prepare(a.data,a.output);print(json.dumps({'entries':len(m['entries'])}))
