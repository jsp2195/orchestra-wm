"""One archive pass into bounded private shards, then one scene at a time."""
import collections
import json
from pathlib import Path
import shutil
import time
import zipfile

import ijson
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from orchestra_wm.i24.data import I24Adapter, sha, save_scene
from orchestra_wm.i24.schema import RoadMap, RoadsideScene
from orchestra_wm.i24_continuous.data import exposure_fields
from orchestra_wm.i24_continuous.source import write_immutable


def prepare(config_path, output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(Path(config_path).read_text())
    write_immutable(output/'frozen_selection.json',cfg)
    if (output/'data_manifest.json').exists():
        result=json.loads((output/'data_manifest.json').read_text())
        for name,h in result['artifacts'].items():
            if sha(output/name)!=h:raise ValueError('Cached artifact mismatch: '+name)
        return result
    blocks=cfg['blocks'];names=[b['name'] for b in blocks]
    native_manifest=output/'native_manifest.json'
    if not native_manifest.exists():
        if list((output/'native').rglob('*.parquet')):
            raise ValueError('Incomplete shards preserved; select a new output to restart source conversion')
        archive=Path(cfg['archive'])
        if sha(archive)!=cfg['archive_sha256']:raise ValueError('Source archive hash mismatch')
        meta=json.loads(Path('data/i24_continuous/raw/pilot_0700_wb/acquisition.json').read_text())
        adapter=I24Adapter(meta);counts=collections.Counter();buffers={n:[] for n in names}
        sizes=collections.Counter();parts=collections.Counter();ids={n:set() for n in names}
        def flush(n):
            if not buffers[n]:return
            if shutil.disk_usage(output).free<cfg['min_free_bytes']:raise ValueError('Disk reserve reached; shards preserved')
            directory=output/'native'/n;directory.mkdir(parents=True,exist_ok=True)
            p=directory/f'part-{parts[n]:05}.parquet'
            pq.write_table(pa.Table.from_pandas(pd.concat(buffers[n],ignore_index=True),preserve_index=False),p,compression='zstd')
            parts[n]+=1;buffers[n]=[];sizes[n]=0
        began=time.monotonic();last=began
        with zipfile.ZipFile(archive) as z:
            member=next(i for i in z.infolist() if i.filename.endswith('.json'))
            with z.open(member) as f:
                for r in ijson.items(f,'item',use_float=True):
                    counts['source_records_scanned']+=1
                    if time.monotonic()-last>30:
                        print('Streamed',counts['source_records_scanned'],'records; retained',sum(v for k,v in counts.items() if k.endswith('_tracks')),flush=True);last=time.monotonic()
                    if int(r.get('direction',0))!=-1:continue
                    t=np.asarray(r['timestamp'],float);x=np.asarray(r['x_position'],float)
                    if not len(t):continue
                    spatial=(x>=cfg['x_min_ft'])&(x<cfg['x_max_ft'])
                    if not spatial.any():continue
                    tt=t[spatial]
                    for b in blocks:
                        start=b['start'];end=start+b['seconds'];n=b['name']
                        if tt[-1]<start or tt[0]>=end:continue
                        # Retain a whole regional identity within one temporal block.
                        # This excludes only block-crossing records, never an internal window boundary.
                        if tt[0]<start or tt[-1]>=end:
                            counts[n+'_boundary_tracks_omitted']+=1
                            counts[n+'_boundary_samples_omitted']+=int((spatial&(t>=start)&(t<end)).sum());continue
                        if spatial.sum()<3:counts[n+'_short_tracks']+=1;continue
                        chosen=dict(r)
                        for key in ('timestamp','x_position','y_position'):
                            chosen[key]=np.asarray(r[key])[spatial].tolist()
                        frame,q=adapter.record(chosen,meta['session'],cfg['archive_sha256'])
                        if frame is None:counts[n+'_invalid_tracks']+=1;continue
                        ident=str(frame.track.iloc[0])
                        if any(ident in v for v in ids.values()):raise ValueError('Shared identity across blocks')
                        ids[n].add(ident);counts[n+'_tracks']+=1;counts[n+'_samples']+=len(frame)
                        counts[n+'_unusable_samples']+=q['rejected_samples'];counts[n+'_gaps']+=q['gaps']
                        buffers[n].append(frame);sizes[n]+=len(frame)
                        if sizes[n]>=50000:flush(n)
                while f.read(1024*1024):pass
        for n in names:flush(n)
        write_immutable(native_manifest,dict(counts=dict(counts),source_crc_verified=True,wall_seconds=time.monotonic()-began,
                                             shared_track_ids=0,artifacts={str(p.relative_to(output)):sha(p) for p in (output/'native').rglob('*.parquet')}))
    native=json.loads(native_manifest.read_text())
    for name,digest in native['artifacts'].items():
        if sha(output/name)!=digest:raise ValueError('Native shard integrity mismatch: '+name)
    road=RoadMap(0,(cfg['x_max_ft']-cfg['x_min_ft'])*.3048,(18*.3048,30*.3048,42*.3048,54*.3048),12*.3048,8,1.)
    scenes=[];excluded=[]
    for b in blocks:
        dataset=ds.dataset(str(output/'native'/b['name']),format='parquet')
        for begin in np.arange(b['start']+1,b['start']+b['seconds']-25,25):
            group=dataset.to_table(filter=(ds.field('time')>=begin-.5)&(ds.field('time')<=begin+25)).to_pandas()
            identities=tuple(sorted(group.track.unique()))
            if len(identities)>cfg['max_agents']:raise ValueError('Population cap exceeded; no truncation allowed')
            grid=begin+np.arange(126)*.2;values=np.zeros((126,len(identities),8),np.float32);exists=np.zeros(values.shape[:2],bool);classes=[]
            for j,ident in enumerate(identities):
                track=group[group.track==ident].sort_values('time');tt=track.time.to_numpy()
                ix=np.searchsorted(tt,grid,side='right')-1;use=ix>=0;ix=np.maximum(ix,0);age=grid-tt[ix]
                use&=(age<=.08)&track.usable.to_numpy()[ix]
                v=track[['s','d','vs','vd','as','ad','length','width']].to_numpy()[ix]
                v[:,:2]+=v[:,2:4]*np.maximum(age,0)[:,None];use&=(v[:,0]>=0)&(v[:,0]<road.s_max)
                values[:,j]=v;exists[:,j]=use;classes.append(int(track['class'].iloc[0]))
            if not len(identities) or exists.sum(1).min()<2:
                excluded.append(dict(block=b['name'],start=float(begin),reason='insufficient concurrent reconstruction'));continue
            provenance=dict(kind='REAL_I24',source_hash=cfg['archive_sha256'],split=b['split'],cluster=b['name'],
                            recording_date=cfg['source_date'],sensing='EMULATED',population='Retained regional reconstruction; block-boundary omissions; not census',
                            timestamp='documented corrected Unix seconds',kinematics='backward derivatives; <=0.08s causal extrapolation')
            scene=RoadsideScene(f'{b["name"]}:{begin}',cfg['source_date'],provenance,grid,identities,values,exists,exists.copy(),exists.astype(float),road,np.array(classes)).validate()
            path=output/'canonical'/f'{b["name"]}-{int(begin)}.parquet';save_scene(scene,path)
            fields,support=exposure_fields(values,exists,road)
            np.savez_compressed(path.with_suffix('.fields.npz'),fields=fields,support=support)
            scenes.append(dict(path=str(path.relative_to(output)),split=b['split'],block=b['name'],start=float(begin),end=float(begin+25),
                               tracks=len(identities),usable_samples=int(exists.sum()),min_concurrent=int(exists.sum(1).min()),
                               median_speed_m_s=float(np.median(values[:,:,2][exists])),field_support_fraction=float(support.mean())))
        print('Prepared block',b['name'],flush=True)
    result=dict(source='REAL_I24',source_sha256=cfg['archive_sha256'],source_dates=[cfg['source_date']],
                frozen_selection_sha256=sha(output/'frozen_selection.json'),counts=native['counts'],scenes=scenes,
                scene_counts=dict(collections.Counter(s['split'] for s in scenes)),shared_source_track_ids=0,independent_sessions=1,
                excluded_windows=excluded,maximum_window_tracks=max(s['tracks'] for s in scenes),
                macro_definition='Causal trailing 1s vehicle-time/distance exposure; unknown empty support',
                limitation='Single-day development only; examined day; reconstructed subset with boundary omissions, not census',
                artifacts={str(p.relative_to(output)):sha(p) for p in (output/'canonical').rglob('*') if p.is_file()})
    write_immutable(output/'data_manifest.json',result)
    return result
