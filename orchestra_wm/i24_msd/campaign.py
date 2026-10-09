"""Real I24-MSD pilot orchestration, entirely separate from continuous I24."""
import hashlib
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import pandas as pd

from .acquisition import AcquisitionError, atomic_json, hashes, safe_extract
from .adapter import I24MSDAdapter
from .experiment import split_scenes, train


def run_pilot(cfg, manifest, raw_root, out):
    import torch
    started = time.perf_counter()
    archive_entry = next((f for f in manifest['files'] if f['id'] == 11835291 and f.get('downloaded_bytes')), None)
    if archive_entry is None:
        raise AcquisitionError('This verified pilot uses 11835291/data_2022_11_24.zip; other releases require schema/provenance review')
    archive = Path(archive_entry['local_path'])
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != 325 or not all('data.tfrecord-' in n for n in names):
            raise AcquisitionError('Archive inventory changed')
        selected = [names[i] for i in (0,162,324)]
    dest = Path(raw_root)/'pilot_members'
    if not dest.exists(): safe_extract(archive, dest, cfg['max_disk_bytes'], selected_names=selected)
    extraction = json.loads((dest/'EXTRACTION_MANIFEST.json').read_text())
    if extraction['archive_sha256'] != archive_entry['hashes']['SHA-256']:
        raise AcquisitionError('Extraction belongs to another source')
    for entry in extraction['files']:
        if hashes(dest/entry['path']) != entry['hashes']: raise AcquisitionError('Extracted bytes changed')
    if set(selected) != {e['path'] for e in extraction['files']}: raise AcquisitionError('Pilot extraction selection changed')
    atomic_json(out/'extraction_manifest.json', extraction)
    scenes = I24MSDAdapter(manifest).scenes([dest/n for n in selected])
    splits = split_scenes(scenes, cfg['max_scenes'])
    splitmanifest = {name: [{'scenario':s.scene_id,'source_hash':s.provenance['source_hash'],
                           'track_ids':list(s.track_ids),'group':s.provenance['overlap_group'],
                           'source_file':s.provenance['source_file']} for s in group] for name,group in splits.items()}
    splitpath = out/'split_manifest.json'
    if splitpath.exists() and json.loads(splitpath.read_text()) != splitmanifest:
        raise AcquisitionError('Immutable pilot split changed')
    atomic_json(splitpath, splitmanifest)
    rows = []
    for scene in scenes:
        v, m = scene.values, scene.existence
        fd = np.diff(v[:,:,:2],axis=0)/np.diff(scene.time)[:,None,None]
        pairs = m[1:] & m[:-1]
        rows.append({'scenario':scene.scene_id,'agents':len(scene.track_ids),'observed_context_agents':int(m[:6].any(0).sum()),
                     'valid_states':int(m.sum()),'missing_fraction':float(1-m.mean()),
                     'speed_mean_mps':float(np.linalg.norm(v[:,:,2:4][m],axis=-1).mean()),
                     'position_velocity_discrepancy_mps':float(np.median(np.linalg.norm(fd-v[1:,:,2:4],axis=-1)[pairs])) if pairs.any() else None,
                     'static_map_features':len(scene.provenance['source_map']),
                     'relative_start_seconds':float(scene.time[0]),'relative_end_seconds':float(scene.time[-1])})
    pd.DataFrame(rows).to_csv(out/'data_qc.csv',index=False)
    qc = {'status':'REAL_ACCESSED','downloaded_bytes':manifest['total_downloaded_bytes'],'native_records_inspected':349,
          'multiagent_context_scenes':len(scenes),'selected_split_counts':{k:len(v) for k,v in splits.items()},
          'native_samples':91,'native_dt_seconds':scenes[0].provenance['native_dt_seconds'],
          'source_cutoff_index':10,'source_context_end_seconds':float(scenes[0].time[5]),
          'source_future_seconds':float(scenes[0].time[-1]-scenes[0].time[5]),
          'model_dt_seconds':.2,'model_context_steps':6,'horizons_seconds':[1,2,4,6],
          'examples':rows[:3],'source_recording_date_groups':1,'independent_recording_generalization':False,
          'split_caveat':'Related supplied track IDs grouped across all inspected eligible clips; unknown aliases/absolute recording timestamps prevent exhaustive overlap certification.',
          'macro_status':'N/A: curated cohort and unknown population coverage; no measured lane density/flow claim',
          'physical_boundary_status':'N/A outside actual supplied finite polylines; numerical model bounds are not road edges'}
    atomic_json(out/'data_qc.json',qc)
    manifest.update(status='REAL_ACCESSED',real_scenarios_parsed=len(scenes))
    manifest.pop('last_error',None); atomic_json(out/'DOWNLOAD_MANIFEST.json',manifest)
    # Keep a historical record of the resolved network failure rather than an active BLOCKED label.
    if (out/'BLOCKED.json').exists():
        atomic_json(out/'resolved_access_block.json', json.loads((out/'BLOCKED.json').read_text()))
        (out/'BLOCKED.json').unlink()
    critical = [Path('orchestra_wm/i24_msd/experiment.py'),Path('orchestra_wm/i24_msd/adapter.py'),
                Path('orchestra_wm/i24/model.py'),Path('orchestra_wm/i24/schema.py'),Path('orchestra_wm/i24/data.py')]
    fingerprint = hashlib.sha256(json.dumps({k:v for k,v in cfg.items() if k != 'source_file_ids'},sort_keys=True).encode()+splitpath.read_bytes()+b''.join(p.read_bytes() for p in critical)).hexdigest()
    atomic_json(out/'run_manifest.json',{'config':cfg,'fingerprint':fingerprint,'source_hashes':{str(p):hashes(p)['SHA-256'] for p in critical},
        'split_sha256':hashes(splitpath)['SHA-256'],'archive_sha256':archive_entry['hashes']['SHA-256'],
        'evaluation_source_hashes':{str(p):hashes(p)['SHA-256'] for p in Path('orchestra_wm/i24_msd').glob('*.py')},
        'source_kind':'REAL_I24_MSD','model_macro_feedback':False,'cutoff_seconds':1.,'torch':torch.__version__,
        'resource_budget':{'cpu_threads':cfg['threads'],'workers':0,'wall_seconds':cfg['max_wall_seconds'],'download_bytes':cfg['max_download_bytes'],'disk_bytes':cfg['max_disk_bytes']}})
    trained = train(cfg,splits,out,fingerprint)
    from .evaluation import evaluate, report
    metrics = evaluate(cfg,splits['test'],out)
    report(cfg,out,trained,metrics,qc,time.perf_counter()-started)
    return manifest
