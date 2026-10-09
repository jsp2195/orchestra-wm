"""Explicit v1.x MOTION JSON-array adapter, causal resampling, fixture and splits."""
from pathlib import Path
from dataclasses import asdict
import hashlib,json,os
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import ijson
from orchestra_wm.i24.schema import RoadMap,RoadsideScene,Observation

FT=.3048
CITATION='Gloudemans et al. 2023, Transportation Research Part C 155, 104311.'

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def write_json(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)

def fields_numpy(values,visible,road):
    shape=values.shape[:-2];result=np.zeros((*shape,road.field_count,3),np.float32);support=np.zeros((*shape,road.field_count),bool)
    flat=values.reshape(-1,values.shape[-2],8);mask=visible.reshape(-1,visible.shape[-1]);target=result.reshape(-1,road.field_count,3);sup=support.reshape(-1,road.field_count)
    length=(road.s_max-road.s_min)/road.bins
    for t,(v,m) in enumerate(zip(flat,mask)):
        lane=np.argmin(np.abs(v[:,1,None]-np.asarray(road.lane_centers)),axis=1);bins=np.floor((v[:,0]-road.s_min)/length).astype(int)
        valid=m&(bins>=0)&(bins<road.bins)&(np.abs(v[:,1]-np.asarray(road.lane_centers)[lane])<=road.lane_width/2)
        for k in range(road.field_count):
            selected=valid&(lane*road.bins+bins==k);n=selected.sum()
            if n:
                speed=float(v[selected,2].mean());density=n/length*1000
                target[t,k]=[speed,density,density*speed*3.6];sup[t,k]=True
    return result,support

class I24Adapter:
    release='I24MOTION_PUBLIC_v1.0'
    def __init__(self,acquisition):
        self.meta=acquisition
        if acquisition.get('dataset')!='I-24 MOTION' or acquisition.get('release')!=self.release:raise ValueError('Only verified multivehicle MOTION v1.0 JSON-array is supported')
        if acquisition.get('format')!='json-array' or acquisition.get('direction')!=-1:raise ValueError('Verified adapter currently supports official JSON array, westbound only')
        if not acquisition.get('terms_accepted') or not acquisition.get('source_url','').startswith('https://i24motion.org'):raise ValueError('Authorized official acquisition provenance required')
        if not acquisition.get('files'):raise ValueError('No authentic source files declared')
        self.origin_ft=float(acquisition['x_origin_ft'])
    def record(self,r,session,source):
        required=['_id','timestamp','x_position','y_position','length','width','direction']
        if any(k not in r for k in required):raise ValueError('Not the documented MOTION vehicle-trajectory schema')
        if int(r['direction'])!=-1:return None,{'other_direction':1}
        t=np.asarray(r['timestamp'],float);x=np.asarray(r['x_position'],float);y=np.asarray(r['y_position'],float)
        if len(t)<3 or len(t)>200000 or x.shape!=t.shape or y.shape!=t.shape:raise ValueError('Invalid trajectory-array lengths')
        if not np.isfinite(np.stack([t,x,y])).all():return None,{'nonfinite_track':1}
        if np.any(np.diff(t)<=0):return None,{'invalid_time_order_track':1}
        ident=r['_id'];ident=ident.get('$oid') if isinstance(ident,dict) else str(ident)
        if not ident:raise ValueError('Invalid anonymous track ID')
        length=float(r['length'])*FT;width=float(r['width'])*FT
        if not (1<length<30 and .5<width<4):return None,{'invalid_dimensions_track':1}
        # Source is back-center, x increases EB. Convert to WB travel-increasing center s.
        s=-(x-self.origin_ft)*FT+length/2;d=y*FT
        dt=np.diff(t);v=np.zeros((len(t),2));v[1:]=np.diff(np.stack([s,d],1),axis=0)/dt[:,None]
        a=np.zeros_like(v);a[2:]=np.diff(v[1:],axis=0)/dt[1:,None]
        usable=np.ones(len(t),bool);usable[:2]=False;usable[1:] &= dt<=.5
        usable &= (np.abs(v[:,0])<65)&(np.abs(v[:,1])<10)&(np.abs(a[:,0])<15)&(np.abs(a[:,1])<15)
        lane=np.floor((y-12)/12).astype(int);usable &= (lane>=0)&(lane<4)
        quality={'samples':len(t),'rejected_samples':int((~usable).sum()),'gaps':int((dt>.5).sum()),'lane_jumps':int((np.abs(np.diff(lane))>1).sum()),'native_dt_median':float(np.median(dt)),'tracking_swaps_verified':0}
        f=pd.DataFrame({'session':session,'track':str(ident),'time':t,'s':s,'d':d,'vs':v[:,0],'vd':v[:,1],'as':a[:,0],'ad':a[:,1],'length':length,'width':width,'class':int(r.get('coarse_vehicle_class',-1)),'usable':usable,'source':source,'flags':json.dumps(r.get('flags',[]))})
        return f,quality
    def convert(self,root,destination,max_rows=2000000):
        destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
        manifest_path=destination/'conversion.json'
        if manifest_path.exists():
            old=json.loads(manifest_path.read_text())
            if old.get('acquisition')!=self.meta:raise ValueError('Immutable conversion acquisition mismatch')
            for source in old['sources']:
                if sha(Path(root)/source['path'])!=source['sha256']:raise ValueError('Source content changed')
            for name,digest in old['parts'].items():
                if sha(destination/name)!=digest:raise ValueError('Converted partition changed')
            return old
        # An interrupted conversion has no committed manifest; rebuild only our generated partitions.
        for stale in destination.glob('part-*.parquet'):stale.unlink()
        rows=0;part=0;buffer=[];count=0;stats=[];sources=[];seen=set()
        for item in self.meta['files']:
            file=(Path(root)/item['path']).resolve()
            if not file.is_relative_to(Path(root).resolve()):raise ValueError('Source path outside data root')
            sourcehash=sha(file);sources.append({'path':item['path'],'session':item['session'],'sha256':sourcehash,'bytes':file.stat().st_size})
            with open(file,'rb') as handle:
                for r in ijson.items(handle,'item',use_float=True):
                    frame,q=self.record(r,item['session'],sourcehash);stats.append(q)
                    if frame is None:continue
                    key=(item['session'],frame.track.iloc[0])
                    if key in seen:raise ValueError('Duplicate within-session track ID across source records; reconcile explicitly')
                    seen.add(key);rows+=len(frame)
                    if rows>max_rows:raise ValueError('Preprocessing row budget exceeded; select a smaller authorized subset')
                    buffer.append(frame);count+=len(frame)
                    if count>=50000:
                        pq.write_table(pa.Table.from_pandas(pd.concat(buffer,ignore_index=True)),destination/f'part-{part:05}.parquet',compression='zstd');part+=1;buffer=[];count=0
            if sum(p.stat().st_size for p in destination.glob('*.parquet'))>2*1024**3:raise ValueError('Scratch cap exceeded')
        if buffer:pq.write_table(pa.Table.from_pandas(pd.concat(buffer,ignore_index=True)),destination/f'part-{part:05}.parquet',compression='zstd')
        summary={'acquisition':self.meta,'parts':{p.name:sha(p) for p in destination.glob('part-*.parquet')},'schema':'I24-v1.0-json-array','sources':sources,'rows':rows,'tracks':len(seen),'qc':stats,'citation':CITATION,'coordinate_transform':'WB s=-(x_ft-x_origin_ft)*0.3048+length_m/2; d=y_ft*0.3048; straightened curvilinear map','lane_mapping':'WB documented approximate [12,24,36,48,60] ft bounds; confidence 0.5','derivatives':'strict backward differences; no future interpolation; first two samples invalid','reconstruction_limitation':'Official offline tracking/reconciliation can use future imagery; no raw camera outages available','swap_qc':'Discontinuity/acceleration/lane-jump proxies only; identities cannot be verified without source imagery'}
        write_json(destination/'conversion.json',summary);return summary

class SyntheticFixtureAdapter:
    def generate(self,count=18,dt=.2,duration=30,seed=7301):
        scenes=[];road=RoadMap(0,800,(5.4864,9.144,12.8016,16.4592),3.6576,8,1)
        for k in range(count):
            rng=np.random.default_rng(seed+k);n=int(rng.integers(8,17));t=np.arange(0,duration+dt/2,dt);v=np.zeros((len(t),n,8),np.float32);exists=np.ones((len(t),n),bool)
            speed=float(rng.uniform(8,22));phase=rng.uniform(0,6);v[:,:,6]=4.8;v[:,:,7]=1.9
            for i in range(n):
                lane=i%4;base=50+(i//4)*55+rng.uniform(0,8)
                vel=speed+2*np.sin(.2*t+phase+.08*i);s=base+np.cumsum(vel)*dt
                lateral=np.full(len(t),road.lane_centers[lane])
                if i==0 and k%3==0:lateral+=3.6576/(1+np.exp(-(t-10)))
                v[:,i,0]=s;v[:,i,1]=lateral;v[:,i,2]=vel;v[:,i,3]=np.r_[0,np.diff(lateral)/dt];v[:,i,4]=np.r_[0,np.diff(vel)/dt];v[:,i,5]=np.r_[0,np.diff(v[:,i,3])/dt]
                if i==n-1:exists[:int(7/dt),i]=False
                exists[:,i] &= s<road.s_max
            provenance={'kind':'SYNTHETIC_FIXTURE','source_hash':hashlib.sha256(v.tobytes()+exists.tobytes()).hexdigest(),'kinematics':'analytic synthetic fixture; NOT I-24','seed':seed+k,'sensing':'EMULATED'}
            scene=RoadsideScene(f'fixture-{k}',f'fixture-session-{k}',provenance,t,tuple(f'v{i}' for i in range(n)),v,exists,exists.copy(),exists.astype(float),road,np.zeros(n,int)).validate();scenes.append(scene)
        return scenes


def real_scenes(root,cfg,destination):
    meta=json.loads((Path(root)/'acquisition.json').read_text());adapter=I24Adapter(meta)
    manifest=adapter.convert(root,destination,max_rows=cfg['max_source_rows'])
    frame=pd.concat([pd.read_parquet(p) for p in sorted(Path(destination).glob('*.parquet'))],ignore_index=True)
    road=RoadMap(float(meta['s_min_m']),float(meta['s_max_m']),(18*FT,30*FT,42*FT,54*FT),12*FT,cfg['bins'],.5)
    scenes=[];span=cfg['context_seconds']+max(cfg['horizons_seconds']);dt=cfg['dt'];guard=span
    sessions=sorted(frame.session.unique());session_split={s:['train','validation','test'][min(2,int(i/len(sessions)*3))] for i,s in enumerate(sessions)} if len(sessions)>=3 else None
    if len(sessions)>cfg['max_scenes']:raise ValueError('Too many sessions for the declared scene budget')
    partition_counts={'train':0,'validation':0,'test':0}
    for session,group in frame.groupby('session'):
        session_count=0
        lower=float(group.time.min());upper=float(group.time.max());duration=upper-lower
        for start in np.arange(lower+1,upper-span,span+guard):
            end=start+span
            if session_split:split=session_split[session]
            elif end<lower+.6*duration-guard:split='train'
            elif start>lower+.6*duration+guard and end<lower+.8*duration-guard:split='validation'
            elif start>lower+.8*duration+guard:split='test'
            else:continue
            if partition_counts[split]>=max(1,cfg['max_scenes']//3):continue
            window=group[(group.time>=start-.5)&(group.time<=end)&(group.s>=road.s_min)&(group.s<road.s_max)];ids=sorted(window.track.unique())
            if not ids or len(ids)>cfg['max_agents']:continue
            time=start+np.arange(round(span/dt)+1)*dt;v=np.zeros((len(time),len(ids),8),np.float32);exists=np.zeros((len(time),len(ids)),bool);det=exists.copy();classes=[]
            for i,ident in enumerate(ids):
                track=window[window.track==ident].sort_values('time');tt=track.time.to_numpy();indices=np.searchsorted(tt,time,side='right')-1;valid=indices>=0;indices=np.maximum(indices,0);age=time-tt[indices]
                valid &= (age<=.25)&track.usable.to_numpy()[indices]
                x=track[['s','d','vs','vd','as','ad','length','width']].to_numpy()[indices];x[:,:2]+=x[:,2:4]*np.maximum(age,0)[:,None]
                valid &= (x[:,0]>=road.s_min)&(x[:,0]<road.s_max)
                v[:,i]=x;exists[:,i]=valid;det[:,i]=valid;classes.append(int(track['class'].iloc[0]))
            provenance={'kind':'REAL_I24','source_hash':hashlib.sha256(json.dumps(manifest['sources'],sort_keys=True).encode()).hexdigest(),'kinematics':'backward-derived; zero-order causal extrapolation <=0.25 s','release':adapter.release,'x_origin_ft':adapter.origin_ft,'coordinate_transform':manifest['coordinate_transform'],'global_xy_available':False,'split':split,'cluster':session if session_split else f'{session}:{split}','sensing':'EMULATED','reconstruction':'offline source may use future imagery'}
            scenes.append(RoadsideScene(f'{session}:{start:.3f}',session,provenance,time,tuple(ids),v,exists,det,exists.astype(float)*.5,road,np.array(classes)).validate())
            partition_counts[split]+=1;session_count+=1
            if session_count>=max(1,cfg['max_scenes']//len(sessions)):break
    if not scenes:raise ValueError('No complete multivehicle windows within declared bounds')
    return scenes,manifest


def split_scenes(scenes):
    split={k:[] for k in ['train','validation','test']};n=len(scenes)
    for i,s in enumerate(scenes):
        label=s.provenance.get('split') or ('train' if i<int(.6*n) else 'validation' if i<int(.8*n) else 'test');split[label].append(s)
    if any(not v for v in split.values()):raise ValueError('Insufficient independent support for all three partitions')
    for a in split:
        for b in split:
            if a>=b:continue
            for s in split[a]:
                for t in split[b]:
                    if s.session_id==t.session_id and max(s.time[0],t.time[0])<=min(s.time[-1],t.time[-1]):raise ValueError('Shared raw time support across splits')
    return split


def observations(scene,context_steps,regime='dense',seed=0):
    # Identity roster is derived ONLY from pre-cutoff detections, never from future IDs/counts.
    past=scene.detection[:context_steps];cohort=np.flatnonzero(past.any(0));ids=tuple(scene.track_ids[i] for i in cohort)
    rng=np.random.default_rng(seed);keep=rng.random(len(cohort));result=[]
    for t in range(context_steps):
        values=scene.values[t,cohort].copy();visible=past[t,cohort].copy()
        if regime in ['sparse50','sparse20']:visible &= keep<(0.5 if regime=='sparse50' else .2)
        if regime=='blind':visible &= ~((values[:,0]>scene.road.s_min+.15*(scene.road.s_max-scene.road.s_min))&(values[:,0]<scene.road.s_min+.35*(scene.road.s_max-scene.road.s_min)))
        if regime=='outage' and t>=context_steps//2:visible[:]=False
        values[~visible]=0;conf=scene.confidence[t,cohort]*visible;fields,support=fields_numpy(values,visible,scene.road)
        result.append(Observation(values,visible,ids,conf,fields,support))
    # Permanently unseen agents are not supplied as privileged identity hints.
    seen=np.any([o.visible for o in result],axis=0);cohort=cohort[seen];ids=tuple(i for i,m in zip(ids,seen) if m)
    result=[Observation(o.values[seen],o.visible[seen],ids,o.confidence[seen],o.fields,o.field_support) for o in result]
    if not len(cohort):raise ValueError('No historical visible tracks; cannot initialize a fixed cohort')
    return result,cohort


def save_scene(scene,path):
    """Canonical compressed columnar roundtrip; raw source remains separate."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);t,n=scene.values.shape[:2]
    data={'time':np.repeat(scene.time,n),'track_index':np.tile(np.arange(n),t),'existence':scene.existence.ravel(),'detection':scene.detection.ravel(),'confidence':scene.confidence.ravel()}
    for i,key in enumerate(['s','d','vs','vd','as','ad','length','width']):data[key]=scene.values[:,:,i].ravel()
    pq.write_table(pa.Table.from_pydict(data),path,compression='zstd')
    write_json(path.with_suffix('.json'),{'scene_id':scene.scene_id,'session_id':scene.session_id,'provenance':scene.provenance,'track_ids':scene.track_ids,'vehicle_class':scene.vehicle_class.tolist(),'road':asdict(scene.road),'shape':[t,n],'parquet_sha256':sha(path)})


def load_scene(path):
    path=Path(path);meta=json.loads(path.with_suffix('.json').read_text());assert sha(path)==meta['parquet_sha256'];f=pd.read_parquet(path);t,n=meta['shape'];v=f[['s','d','vs','vd','as','ad','length','width']].to_numpy().reshape(t,n,8);road=meta['road'];road['lane_centers']=tuple(road['lane_centers'])
    return RoadsideScene(meta['scene_id'],meta['session_id'],meta['provenance'],f.time.to_numpy().reshape(t,n)[:,0],tuple(meta['track_ids']),v,f.existence.to_numpy().reshape(t,n),f.detection.to_numpy().reshape(t,n),f.confidence.to_numpy().reshape(t,n),RoadMap(**road),np.array(meta['vehicle_class'])).validate()


def collate_observations(items):
    """Explicit padding, distinct from invisible known agents. Stable IDs stay in metadata."""
    n=max(len(o.track_ids) for o in items);values=np.zeros((len(items),n,8),np.float32);visible=np.zeros((len(items),n),bool);padding=np.ones((len(items),n),bool)
    for i,o in enumerate(items):
        k=len(o.track_ids);values[i,:k]=o.values;visible[i,:k]=o.visible;padding[i,:k]=False
    return {'values':values,'visible':visible,'padding':padding,'track_ids':[o.track_ids for o in items]}
