"""Self-contained browser view of exact evaluated autonomous arrays."""
import base64
import gzip
import json
from pathlib import Path
import numpy as np
from .acquisition import atomic_json, hashes


def make_demo(out, saved):
    out=Path(out);directory=out/'demo';directory.mkdir(exist_ok=True)
    # Display selection uses historical agent count only, not forecast quality.
    dense=[m for m in saved if m['regime']=='dense']
    ranked=sorted(dense,key=lambda m:(-len(np.load(m['path'])['track_ids']),m['scene']))[:3]
    ids={m['scene'] for m in ranked};entries=[]
    for meta in saved:
        if meta['scene'] not in ids:continue
        with np.load(meta['path']) as data:
            arrays={k:data[k].tolist() for k in data.files}
        entries.append({'meta':meta,'arrays':arrays})
    payload={'label':'I24-MSD REAL 9-SECOND SCENARIOS','doi':'10.7910/DVN/DQOWQI','entries':entries}
    encoded=gzip.compress(json.dumps(payload,separators=(',',':'),allow_nan=False).encode(),mtime=0)
    (directory/'data.json.gz').write_bytes(encoded)
    template=Path('orchestra_wm/i24_msd/demo_template.html').read_text()
    trained=json.loads((out/'training_metadata.json').read_text())
    counts=json.loads((out/'split_manifest.json').read_text())
    template=template.replace('__TRAINING_UPDATES__',str(trained[0]['updates'])).replace('__TRAINING_SCENES__',str(len(counts['train'])))
    (directory/'index.html').write_text(template.replace('__PAYLOAD__',base64.b64encode(encoded).decode()))
    atomic_json(directory/'provenance.json',{'source_kind':'REAL_I24_MSD','payload_sha256':hashes(directory/'data.json.gz')['SHA-256'],
        'entries':[{'scene':e['meta']['scene'],'regime':e['meta']['regime'],'array_sha256':e['meta']['array_sha256'],
                    'checkpoint_sha256':e['meta']['checkpoint_sha256'],'source_hash':e['meta']['source_hash']} for e in entries],
        'replay':False,'selected_by':'largest historical observed cohort, then scenario ID','macro_population_fields':'N/A, unknown cohort sampling coverage'})
    verify_demo(out)
    return str(directory/'index.html')


def verify_demo(out):
    out=Path(out);payload=gzip.decompress((out/'demo/data.json.gz').read_bytes());data=json.loads(payload)
    assert data['label']=='I24-MSD REAL 9-SECOND SCENARIOS'
    for entry in data['entries']:
        meta=entry['meta'];assert hashes(meta['path'])['SHA-256']==meta['array_sha256']
        with np.load(meta['path']) as actual:
            for key,value in entry['arrays'].items():assert np.array_equal(actual[key],np.array(value)),key
            mask=actual['valid'];pred=actual['prediction'][0,:,:,:2];truth=actual['truth'][:,:,:2]
            assert mask.any() and not np.array_equal(pred[mask],truth[mask]),'Generated sample must not be ground-truth replay'
    return len(data['entries'])
