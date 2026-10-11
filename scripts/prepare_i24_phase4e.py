"""Prepare explicit frozen dates; private Drive IDs stay in the ignored listing."""
import argparse,json
from pathlib import Path
from orchestra_wm.i24_phase4e.data import prepare,ROLES
p=argparse.ArgumentParser()
p.add_argument('--listing',required=True)
p.add_argument('--dates',nargs='+',required=True,choices=list(ROLES))
p.add_argument('--output',default='outputs/i24_phase4e/data_v1')
a=p.parse_args();listing=json.loads(Path(a.listing).read_text())['files']
for date in a.dates:
    name=date[5:7]+'-'+date[8:]+'-'+date[:4]+'.zip'
    matches=[m for m in listing if m['name']==name]
    if len(matches)!=1:raise ValueError('Missing or ambiguous assigned source date: '+date)
    result=prepare(date,matches[0],Path(a.output)/date)
    print(json.dumps({'date':date,'scene_counts':result['scene_counts'],'source_hash':result['nested_zip_sha256']}),flush=True)
