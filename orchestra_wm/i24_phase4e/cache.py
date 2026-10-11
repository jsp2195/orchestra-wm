"""Frozen multi-date feature cache using the unchanged Phase4D causal features."""
import json
from pathlib import Path
import numpy as np
import torch
from orchestra_wm.i24.data import load_scene,sha
from orchestra_wm.i24_continuous.data import observations
from orchestra_wm.i24_continuous.source import write_immutable
from orchestra_wm.i24_phase4d.data import features,adjacency
from .data import ROLES

FEATURE_SOURCE=Path('orchestra_wm/i24_phase4d/data.py')


def prepare(data_root,output,splits=('train','validation')):
    data_root,output=Path(data_root),Path(output);output.mkdir(parents=True,exist_ok=True)
    sources={};entries=[];ids={split:set() for split in splits}
    for date,role in ROLES.items():
        if role not in splits:continue
        root=data_root/date;manifest_path=root/'data_manifest.json'
        manifest=json.loads(manifest_path.read_text())
        verification=json.loads((root/'source_verification.json').read_text())
        if verification['status']!='SOURCE_METADATA_STABLE_AND_SELECTED_STREAM_INTEGRITY_VERIFIED' or verification['native_manifest_sha256']!=sha(root/'native_manifest.json'):
            raise ValueError('Source integrity/metadata verification missing')
        if manifest['source']!='REAL_I24' or manifest['source_dates']!=[date] or manifest['shared_source_track_ids']!=0:
            raise ValueError('Source/split identity failure')
        for name,digest in manifest['artifacts'].items():
            if sha(root/name)!=digest:raise ValueError('Canonical source hash mismatch')
        if manifest['scene_counts'].get(role,0)==0:raise ValueError('Assigned date has no usable windows: '+date)
        sources[str(manifest_path)]=sha(manifest_path)
        for scene in manifest['scenes']:
            if scene['split']!=role:raise ValueError('Date role mismatch')
            p=root/scene['path'];meta=json.loads(p.with_suffix('.json').read_text());ids[role].update(meta['track_ids'])
    for a in ids:
        for b in ids:
            if a<b and ids[a]&ids[b]:raise ValueError('Cross-split source track-ID overlap')
    cfg=dict(source_manifests=sources,feature_source_sha256=sha(FEATURE_SOURCE),cache_source_sha256=sha(Path(__file__)),
             splits=list(splits),regimes=['dense','sparse50','outage'],roles=ROLES,
             observation_seed='7301 + chronological scene number within split',shared_source_track_ids=0,
             scope='Frozen recording-date roles; test cache constructed only after validation freeze')
    write_immutable(output/'config.json',cfg)
    numbers={split:0 for split in splits}
    for manifest_path in sources:
        root=Path(manifest_path).parent;manifest=json.loads(Path(manifest_path).read_text());date=manifest['source_dates'][0]
        for item in manifest['scenes']:
            split=item['split'];number=numbers[split];numbers[split]+=1
            p=root/item['path'];scene=load_scene(p)
            with np.load(p.with_suffix('.fields.npz')) as f:target=f['fields'][25:125].copy();support=f['support'][25:125].copy()
            for regime in cfg['regimes']:
                obs,_=observations(scene,25,regime,7301+number)
                arrays=features(obs,scene.road);arrays.update(target=target,support=support,adjacency=adjacency(scene.road))
                path=output/f'{split}-{number}-{regime}.npz'
                if path.exists():
                    with np.load(path) as old:
                        if set(old.files)!=set(arrays) or any(not np.array_equal(old[k],v) for k,v in arrays.items()):
                            raise ValueError('Existing feature cache differs; preserved')
                else:np.savez_compressed(path,**arrays)
                entries.append(dict(path=path.name,sha256=sha(path),split=split,number=number,regime=regime,
                                    scene=scene.scene_id,source_path=str(p),date=date,block=item['block']))
        print('Cached',date,flush=True)
    result=dict(config=cfg,entries=entries);write_immutable(output/'manifest.json',result);return result


def load_cache(root,allowed_splits=('train','validation')):
    root=Path(root);m=json.loads((root/'manifest.json').read_text());items=[]
    if m['config']['feature_source_sha256']!=sha(FEATURE_SOURCE) or m['config']['cache_source_sha256']!=sha(Path(__file__)):
        raise ValueError('Feature implementation fingerprint changed')
    for path,digest in m['config']['source_manifests'].items():
        if sha(path)!=digest:raise ValueError('Source manifest fingerprint changed')
        manifest=json.loads(Path(path).read_text())
        for name,h in manifest['artifacts'].items():
            if sha(Path(path).parent/name)!=h:raise ValueError('Canonical artifact changed')
    for e in m['entries']:
        if e['split'] not in allowed_splits or ROLES[e['date']]!=e['split']:
            raise ValueError('Forbidden split in cache')
        path=root/e['path']
        if sha(path)!=e['sha256']:raise ValueError('Cached array hash mismatch')
        with np.load(path) as f:arrays={k:torch.from_numpy(f[k].copy()) for k in f.files}
        items.append((e,arrays))
    return m,items
