import argparse,json
from orchestra_wm.i24_phase4e.preservation import backup
p=argparse.ArgumentParser();p.add_argument('--roots',nargs='+',required=True);p.add_argument('--receipt',required=True);p.add_argument('--label',required=True);p.add_argument('--cache')
a=p.parse_args();r=backup(a.roots,a.receipt,a.label,a.cache)
print(json.dumps({'status':r['status'],'archive_sha256':r['archive_sha256'],'parts':len(r['parts']),'files':len(r['files'])}))
