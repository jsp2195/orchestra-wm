"""Verify remote identity after streaming, without any prediction evaluation."""
import argparse,datetime,json
from pathlib import Path
from orchestra_wm.i24_phase4e.preservation import request
from orchestra_wm.i24_continuous.source import write_immutable
from orchestra_wm.i24.data import sha
p=argparse.ArgumentParser();p.add_argument('--listing',required=True);p.add_argument('--data',default='outputs/i24_phase4e/data_v1');p.add_argument('--dates',nargs='+',required=True);a=p.parse_args()
listing=json.loads(Path(a.listing).read_text())['files']
for date in a.dates:
    root=Path(a.data)/date;frozen=json.loads((root/'frozen_selection.json').read_text());source=json.loads((root/'native_manifest.json').read_text())
    old=next(x for x in listing if x['name']==frozen['source_metadata']['name'])
    current=json.loads(request('drive/v3/files/'+old['id']+'?fields=id,name,size,md5Checksum,version'))
    if any(current[k]!=old[k] for k in ['id','name','size','md5Checksum','version']):raise ValueError('Remote source changed since acquisition listing')
    if not source['nested_members_crc_verified'] or not source['outer_selected_member_crc_verified']:raise ValueError('Incomplete source stream')
    result=dict(status='SOURCE_METADATA_STABLE_AND_SELECTED_STREAM_INTEGRITY_VERIFIED',checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                metadata=frozen['source_metadata'],native_manifest_sha256=sha(root/'native_manifest.json'),
                nested_zip_sha256=source['nested_zip_sha256'],whole_outer_md5_verified=False)
    target=root/'source_verification.json'
    if not target.exists():write_immutable(target,result)
    else:
        previous=json.loads(target.read_text())
        if previous['native_manifest_sha256']!=result['native_manifest_sha256']:raise ValueError('Source verification changed')
    print(date,result['status'],flush=True)
