"""Streaming canonical data and track-disjoint buffered chronological scenes."""
import collections
import hashlib
import json
from pathlib import Path

import ijson
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from orchestra_wm.i24.data import I24Adapter, sha, save_scene, load_scene
from orchestra_wm.i24.schema import RoadMap, RoadsideScene, Observation
from .source import write_immutable


def exposure_fields(values,visible,road,dt=.2,window_seconds=1.):
    """Causal Riemann exposure over trailing 1 s, not instantaneous count*speed.

    Each observed endpoint represents dt vehicle-seconds and v*dt metres.
    k=vehicle-seconds/(L*T), q=vehicle-metres/(L*T), v=metres/seconds.
    Only jointly visible, usable samples contribute; empty bins are UNKNOWN.
    """
    t,n=visible.shape;k=road.field_count
    count=np.zeros((t,k),np.float32);distance=np.zeros_like(count)
    length=(road.s_max-road.s_min)/road.bins
    for i in range(t):
        x=values[i];lane=np.argmin(abs(x[:,1,None]-np.asarray(road.lane_centers)),axis=1)
        bins=np.floor((x[:,0]-road.s_min)/length).astype(int)
        use=visible[i]&(bins>=0)&(bins<road.bins)&(abs(x[:,1]-np.asarray(road.lane_centers)[lane])<=road.lane_width/2)
        index=lane[use]*road.bins+bins[use]
        np.add.at(count[i],index,dt)
        np.add.at(distance[i],index,x[use,2]*dt)
    steps=round(window_seconds/dt)
    cumulative=np.vstack([np.zeros((1,k)),np.cumsum(count,axis=0)])
    travelled=np.vstack([np.zeros((1,k)),np.cumsum(distance,axis=0)])
    stop=np.arange(1,t+1);begin=np.maximum(0,stop-steps)
    exposure=cumulative[stop]-cumulative[begin];metres=travelled[stop]-travelled[begin]
    span=(stop-begin)*dt
    speed=np.divide(metres,exposure,out=np.zeros_like(exposure),where=exposure>0)
    density=exposure/(length*span[:,None])*1000
    flow=metres/(length*span[:,None])*3600
    return np.stack([speed,density,flow],axis=-1).astype(np.float32),exposure>0


def prepare(root,output,max_agents=256):
    root,output=Path(root),Path(output);output.mkdir(parents=True,exist_ok=True)
    meta=json.loads((root/'acquisition.json').read_text())
    qc=json.loads((root/'authentic_qc.json').read_text())
    if qc['status']!='REGIONAL_SUBSECOND_SUPPORT_PASSED':raise ValueError('Authentic subsecond QC must pass first')
    sourcehash=meta['files'][0]['sha256'];manifest_path=output/'data_manifest.json'
    if manifest_path.exists():
        old=json.loads(manifest_path.read_text())
        if old['source_sha256']!=sourcehash or old['max_agents']!=max_agents:raise ValueError('Immutable prepared source/config changed')
        for name,digest in old['artifacts'].items():
            if sha(output/name)!=digest:raise ValueError('Prepared artifact integrity mismatch')
        return old
    if list(output.glob('native/*/*.parquet')) or list(output.glob('canonical/*.parquet')):
        raise ValueError('Uncommitted preparation exists; preserve it and choose a new output')
    adapter=I24Adapter(meta);start,end=meta['start_unix_s'],meta['end_unix_s'];duration=end-start
    boundaries=[start,start+.6*duration,start+.8*duration,end]
    guard=25.;names=['train','validation','test']
    intervals={name:[boundaries[i]+(guard if i else 0),boundaries[i+1]-(guard if i<2 else 0)] for i,name in enumerate(names)}
    counts=collections.Counter();track_sets={name:set() for name in names};buffers={name:[] for name in names};rows={name:0 for name in names};parts={name:0 for name in names}
    def flush(name):
        if not buffers[name]:return
        directory=output/'native'/name;directory.mkdir(parents=True,exist_ok=True)
        table=pa.Table.from_pandas(pd.concat(buffers[name],ignore_index=True),preserve_index=False)
        pq.write_table(table,directory/f'part-{parts[name]:05}.parquet',compression='zstd')
        parts[name]+=1;buffers[name]=[];rows[name]=0
    for entry in meta['files']:
        path=root/entry['path']
        if path.stat().st_size!=entry['bytes'] or sha(path)!=sourcehash:raise ValueError('Selected source integrity mismatch')
        with path.open('rb') as handle:
            for record in ijson.items(handle,'item',use_float=True):
                try:frame,detail=adapter.record(record,meta['session'],sourcehash)
                except (ValueError,TypeError,KeyError):counts['schema_rejected_tracks']+=1;continue
                if frame is None:counts.update(detail);continue
                frame=frame[(frame.time>=start)&(frame.time<end)]
                if len(frame)<3:counts['empty_or_short_in_interval']+=1;continue
                # Whole selected source tracks crossing a split/guard are omitted.
                partition=None
                for name,(lower,upper) in intervals.items():
                    if float(frame.time.min())>=lower and float(frame.time.max())<upper:partition=name;break
                if partition is None:
                    counts['crossing_or_guard_tracks_omitted']+=1;counts['crossing_or_guard_samples_omitted']+=len(frame);continue
                ident=str(frame.track.iloc[0])
                if any(ident in ids for ids in track_sets.values()):raise ValueError('Shared source identity across partitions')
                track_sets[partition].add(ident);counts[partition+'_tracks']+=1;counts[partition+'_samples']+=len(frame)
                buffers[partition].append(frame);rows[partition]+=len(frame)
                if rows[partition]>=50000:flush(partition)
    for name in names:flush(name)
    audit=json.loads((root/'source_audit.json').read_text());sel=audit['selection']
    road=RoadMap(0,(sel['x_max']-sel['x_min'])*.3048,(18*.3048,30*.3048,42*.3048,54*.3048),12*.3048,8,1.)
    scenes=[];rejected_windows=[];maximum_population=0
    for name,(lower,upper) in intervals.items():
        paths=list((output/'native'/name).glob('*.parquet'))
        if not paths:raise ValueError('Insufficient track-disjoint support for '+name)
        dataset=ds.dataset(paths,format='parquet')
        for begin in np.arange(lower+1,upper-25,25.):
            stop=begin+25
            # Arrow predicate scans at most one bounded window; no full-source RAM load.
            group=dataset.to_table(filter=(ds.field('time')>=begin-.5)&(ds.field('time')<=stop)).to_pandas()
            ids=tuple(sorted(group.track.unique()));maximum_population=max(maximum_population,len(ids))
            if not ids:rejected_windows.append(dict(split=name,start=float(begin),reason='no tracks'));continue
            if len(ids)>max_agents:raise ValueError('Regional population exceeds max_agents; explicit source-region review required')
            grid=begin+np.arange(126)*.2
            values=np.zeros((len(grid),len(ids),8),np.float32);exists=np.zeros(values.shape[:2],bool);classes=[]
            for j,ident in enumerate(ids):
                track=group[group.track==ident].sort_values('time');tt=track.time.to_numpy();ix=np.searchsorted(tt,grid,side='right')-1
                use=ix>=0;ix=np.maximum(ix,0);age=grid-tt[ix]
                use&=(age<=.08)&track.usable.to_numpy()[ix]
                v=track[['s','d','vs','vd','as','ad','length','width']].to_numpy()[ix]
                v[:,:2]+=v[:,2:4]*np.maximum(age,0)[:,None]
                use&=(v[:,0]>=road.s_min)&(v[:,0]<road.s_max)
                values[:,j]=v;exists[:,j]=use;classes.append(int(track['class'].iloc[0]))
            if np.min(exists.sum(1))<2:
                rejected_windows.append(dict(split=name,start=float(begin),reason='fewer than two usable tracks in a 5Hz frame'));continue
            provenance=dict(kind='REAL_I24',source_hash=sourcehash,kinematics='strict backward derivatives; past-only extrapolation <=0.08s',
                            split=name,cluster=name,release=adapter.release,sensing='EMULATED',
                            population='Track-disjoint retained regional population; crossing/guard tracks omitted and counted',
                            confidence='Binary usable-sample indicator, not calibrated sensor confidence',
                            macro='Trailing 1s causal vehicle-time/distance exposure; empty support UNKNOWN')
            scene=RoadsideScene(f'{meta["session"]}:{begin:.3f}',meta['session'],provenance,grid,ids,values,exists,exists.copy(),exists.astype(float),road,np.array(classes)).validate()
            path=output/'canonical'/f'{name}-{len(scenes):05}.parquet';save_scene(scene,path)
            fields,support=exposure_fields(values,exists,road)
            np.savez_compressed(path.with_suffix('.fields.npz'),fields=fields,support=support)
            scenes.append(dict(path=str(path.relative_to(output)),split=name,start=float(begin),end=float(stop),tracks=len(ids)))
    scene_counts=collections.Counter(s['split'] for s in scenes)
    if any(scene_counts[name]<1 for name in names):raise ValueError('No complete 25s scenes in one or more partitions; training blocked')
    result=dict(source='REAL_I24',source_sha256=sourcehash,max_agents=max_agents,maximum_window_tracks=maximum_population,
                split_intervals=intervals,guard_seconds=guard,counts=dict(counts),scene_counts=dict(scene_counts),
                scenes=scenes,rejected_windows=rejected_windows,
                shared_source_track_ids=0,independent_sessions=1,
                artifacts={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()},
                limitation='Single-day regional retained-track preliminary pilot; omitted cross-boundary tracks mean fields are not a full traffic census',
                macro_definition='Causal trailing 1s Riemann vehicle-time/distance exposure at 5Hz; observed-sample support only')
    write_immutable(manifest_path,result)
    return result


def observations(scene,context_steps,regime='dense',seed=0):
    history=scene.detection[:context_steps];cohort=np.flatnonzero(history.any(0));rng=np.random.default_rng(seed)
    values=scene.values[:context_steps,cohort].copy();visible=history[:,cohort].copy();keep=rng.random(len(cohort))
    if regime in ('sparse50','sparse20'):visible&=keep[None]<(0.5 if regime=='sparse50' else .2)
    if regime=='blind':visible&=~((values[:,:,0]>.15*scene.road.s_max)&(values[:,:,0]<.35*scene.road.s_max))
    if regime=='outage':visible[context_steps//2:]=False
    seen=visible.any(0);cohort=cohort[seen];values=values[:,seen];visible=visible[:,seen]
    if not len(cohort):raise ValueError('No historical visible cohort')
    values[~visible]=0;fields,support=exposure_fields(values,visible,scene.road)
    ids=tuple(scene.track_ids[i] for i in cohort)
    return [Observation(values[i],visible[i],ids,visible[i].astype(np.float32),fields[i],support[i]) for i in range(context_steps)],cohort
