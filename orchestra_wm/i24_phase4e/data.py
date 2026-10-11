"""Frozen per-date source ingestion; inherited Phase4B geometry and resampling."""
import collections,datetime,hashlib,json,shutil,time
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from orchestra_wm.i24.data import I24Adapter,sha,save_scene
from orchestra_wm.i24.schema import RoadMap,RoadsideScene
from orchestra_wm.i24_continuous.data import exposure_fields
from orchestra_wm.i24_continuous.source import write_immutable
from .source import remote_records,SESSIONS

ROLES={'2022-11-21':'train','2022-11-22':'train','2022-11-23':'train','2022-11-25':'validation','2022-11-28':'test','2022-11-29':'test'}
SLOTS=['06:15','07:15','08:15','09:15']

def configuration(date,metadata):
    split=ROLES[date]
    public_metadata={k:metadata[k] for k in ['name','size','md5Checksum','version']}
    fingerprint=hashlib.sha256(json.dumps(public_metadata,sort_keys=True).encode()).hexdigest()
    blocks=[dict(name=date+'_'+clock.replace(':',''),split=split,start=datetime.datetime.fromisoformat(date+'T'+clock).replace(tzinfo=ZoneInfo('America/Chicago')).timestamp(),seconds=600) for clock in SLOTS]
    return dict(source_date=date,source_metadata=public_metadata,archive_sha256=fingerprint,
                source_fingerprint_kind='SHA256 of remote name/size/MD5/version; actual nested ZIP SHA256 recorded after complete stream',
                session=SESSIONS[date],x_min_ft=316800,x_max_ft=320080,max_agents=2048,min_free_bytes=10_000_000_000,
                blocks=blocks,recording_timezone='America/Chicago',selection='Frozen Phase4D clocks; source-only QC; no replacement slots')

def prepare(date,metadata,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    cfg=configuration(date,metadata);write_immutable(output/'frozen_selection.json',cfg)
    if (output/'data_manifest.json').exists():
        result=json.loads((output/'data_manifest.json').read_text())
        for name,h in result['artifacts'].items():
            if sha(output/name)!=h:raise ValueError('Cached artifact mismatch: '+name)
        return result
    blocks=cfg['blocks'];names=[b['name'] for b in blocks];native_manifest=output/'native_manifest.json'
    if not native_manifest.exists():
        if list((output/'native').rglob('*.parquet')):raise ValueError('Partial native shards preserved; require new output namespace')
        meta=json.loads(Path('outputs/i24_continuous/pilot_0700_wb/acquisition.json').read_text())
        meta.update(session=cfg['session'],recording_date=date,files=[{'path':cfg['session']+'.json','session':cfg['session']}])
        adapter=I24Adapter(meta);counts=collections.Counter();buffers={n:[] for n in names}
        sizes=collections.Counter();parts=collections.Counter();ids={n:set() for n in names};all_ids=set();audit={};native_dt=[]
        first=float('inf');last_timestamp=float('-inf')
        day_start=datetime.datetime.fromisoformat(date).replace(tzinfo=ZoneInfo('America/Chicago')).timestamp()
        def flush(n):
            if not buffers[n]:return
            if shutil.disk_usage(output).free<cfg['min_free_bytes']:raise ValueError('Disk reserve reached; shards preserved')
            directory=output/'native'/n;directory.mkdir(parents=True,exist_ok=True)
            p=directory/f'part-{parts[n]:05}.parquet'
            pq.write_table(pa.Table.from_pandas(pd.concat(buffers[n],ignore_index=True),preserve_index=False),p,compression='zstd')
            parts[n]+=1;buffers[n]=[];sizes[n]=0
        began=time.monotonic();last=began
        for r in remote_records(metadata,date,audit):
            counts['source_records_scanned']+=1
            if time.monotonic()-last>30:
                print(date,'streamed',counts['source_records_scanned'],'retained',sum(v for k,v in counts.items() if k.endswith('_tracks')),flush=True);last=time.monotonic()
            for key in ['_id','timestamp','x_position','y_position','length','width','direction']:
                if key not in r:raise ValueError('Unrecognized source schema')
            ident=r['_id'].get('$oid') if isinstance(r['_id'],dict) else str(r['_id'])
            if not ident or ident in all_ids:raise ValueError('Duplicate/invalid source identity; reconcile before training')
            all_ids.add(ident)
            t=np.asarray(r['timestamp'],float);x=np.asarray(r['x_position'],float);y=np.asarray(r['y_position'],float)
            counts['source_samples']+=len(t)
            if len(t)<3 or x.shape!=t.shape or y.shape!=t.shape:counts['invalid_array_tracks']+=1;continue
            if not np.isfinite(np.stack([t,x,y])).all():counts['nonfinite_tracks']+=1;continue
            if np.any(np.diff(t)<=0):counts['nonmonotonic_tracks']+=1;continue
            first=min(first,float(t[0]));last_timestamp=max(last_timestamp,float(t[-1]))
            if t[0]<day_start or t[-1]>=day_start+86400:raise ValueError('Record timestamp outside documented recording date')
            native_dt.append(float(np.median(np.diff(t))));counts['source_gaps_over_0_5s']+=int((np.diff(t)>.5).sum())
            counts['direction_'+str(int(r['direction']))+'_records']+=1
            if int(r['direction'])!=-1:continue
            spatial=(x>=cfg['x_min_ft'])&(x<cfg['x_max_ft'])
            if not spatial.any():continue
            tt=t[spatial]
            for b in blocks:
                start=b['start'];end=start+b['seconds'];n=b['name']
                if tt[-1]<start or tt[0]>=end:continue
                if tt[0]<start or tt[-1]>=end:
                    counts[n+'_boundary_tracks_omitted']+=1;counts[n+'_boundary_samples_omitted']+=int((spatial&(t>=start)&(t<end)).sum());continue
                if spatial.sum()<3:counts[n+'_short_tracks']+=1;continue
                chosen=dict(r)
                for key in ('timestamp','x_position','y_position'):chosen[key]=np.asarray(r[key])[spatial].tolist()
                frame,q=adapter.record(chosen,meta['session'],cfg['archive_sha256'])
                if frame is None:
                    counts[n+'_invalid_tracks']+=1
                    for k,v in q.items():counts[n+'_'+k]+=v
                    continue
                if any(ident in v for v in ids.values()):raise ValueError('Shared identity across blocks')
                ids[n].add(ident);counts[n+'_tracks']+=1;counts[n+'_samples']+=len(frame)
                for key in ['rejected_samples','gaps','lane_jumps']:counts[n+'_'+key]+=q[key]
                buffers[n].append(frame);sizes[n]+=len(frame)
                if sizes[n]>=50000:flush(n)
        for n in names:flush(n)
        audit.update(source_start_unix_s=first,source_end_unix_s=last_timestamp,native_dt_median_s=float(np.median(native_dt)),
                     counts=dict(counts),shared_track_ids=0,source_duplicate_ids=0,wall_seconds=time.monotonic()-began,
                     artifacts={str(p.relative_to(output)):sha(p) for p in (output/'native').rglob('*.parquet')})
        if abs(audit['native_dt_median_s']-.04)>.001:raise ValueError('Unexpected native sampling period')
        write_immutable(native_manifest,audit)
    native=json.loads(native_manifest.read_text())
    for name,digest in native['artifacts'].items():
        if sha(output/name)!=digest:raise ValueError('Native shard integrity mismatch: '+name)
    road=RoadMap(0,(cfg['x_max_ft']-cfg['x_min_ft'])*.3048,(18*.3048,30*.3048,42*.3048,54*.3048),12*.3048,8,1.)
    scenes=[];excluded=[]
    for b in blocks:
        if not (output/'native'/b['name']).exists():
            excluded.append(dict(block=b['name'],reason='no retained reconstruction in frozen slot'));continue
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
                excluded_windows=excluded,maximum_window_tracks=max((s['tracks'] for s in scenes),default=0),
                macro_definition='Causal trailing 1s vehicle-time/distance exposure; unknown empty support',
                limitation='One date in frozen multi-date roles; reconstructed subset with boundary omissions, not census',native_manifest_sha256=sha(native_manifest),nested_zip_sha256=native['nested_zip_sha256'],
                artifacts={str(p.relative_to(output)):sha(p) for p in (output/'canonical').rglob('*') if p.is_file()})
    write_immutable(output/'data_manifest.json',result)
    return result
