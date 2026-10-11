"""History-only fixed roadway-token features, with no future roster allocation."""
import json
from pathlib import Path
import numpy as np
from orchestra_wm.i24.data import load_scene,sha
from orchestra_wm.i24_continuous.data import observations
from orchestra_wm.i24_continuous.source import write_immutable

SCALES=np.array([30.,100.,3000.],np.float32)

def features(obs,road):
    k=road.field_count;latest=np.zeros((k,3),np.float32);age=np.ones(k,np.float32)*5
    aggregate=[];micro=[];seen_count=np.zeros(len(obs[0].visible))
    pos=road.tokens()/np.array([road.s_max,max(road.lane_centers)],np.float32)
    for o in obs:
        prev=latest.copy();age+=.2;latest[o.field_support]=o.fields[o.field_support];age[o.field_support]=0
        v,d,q=(latest/SCALES).T
        basis=np.stack([v,d,q,v*d,v*v,d*d,(latest[:,0]-prev[:,0])/30,np.minimum(age/5,1)],-1)
        agg=np.concatenate([latest/SCALES,o.field_support[:,None],np.minimum(age[:,None]/5,1),pos,basis],-1)
        extra=np.zeros((k,8),np.float32);seen_count+=o.visible
        lane=np.argmin(abs(o.values[:,1,None]-np.array(road.lane_centers)),axis=1)
        length=(road.s_max-road.s_min)/road.bins;bins=np.floor((o.values[:,0]-road.s_min)/length).astype(int)
        inside=o.visible&(bins>=0)&(bins<road.bins)&(abs(o.values[:,1]-np.array(road.lane_centers)[lane])<=road.lane_width/2)
        for cell in range(k):
            ix=np.flatnonzero(inside&(lane*road.bins+bins==cell))
            if not len(ix):continue
            x=o.values[ix]
            extra[cell]=[x[:,2].mean()/30,x[:,2].var()/900,x[:,4].mean()/5,x[:,3].mean()/5,x[:,6].mean()/10,((x[:,0]-road.tokens()[cell,0])/length).mean(),seen_count[ix].mean()/25,(x[:,4]<-.5).mean()]
        aggregate.append(agg);micro.append(extra)
    return dict(aggregate=np.array(aggregate,np.float32),micro=np.array(micro,np.float32),reference=latest.copy(),positions=pos.astype(np.float32))

def adjacency(road):
    x=road.tokens();length=(road.s_max-road.s_min)/road.bins
    a=((abs(x[:,None,0]-x[None,:,0])<=length*1.01)&(abs(x[:,None,1]-x[None,:,1])<=road.lane_width*1.01)).astype(np.float32)
    np.fill_diagonal(a,0);return a/np.maximum(a.sum(-1,keepdims=True),1)

def prepare(root,output):
    root,output=Path(root),Path(output);output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((root/'data_manifest.json').read_text())
    assert manifest['source']=='REAL_I24' and manifest['shared_source_track_ids']==0
    config=dict(data_manifest_sha256=sha(root/'data_manifest.json'),feature_source_sha256=sha(Path(__file__)),splits=['train','validation'],regimes=['dense','sparse50','outage'],observation_seed='7301 + within-split chronological scene number',scope='Examined November21 development; no test partition opened')
    write_immutable(output/'config.json',config)
    if (output/'manifest.json').exists():
        result=json.loads((output/'manifest.json').read_text())
        for e in result['entries']:assert sha(output/e['path'])==e['sha256']
        return result
    entries=[]
    for split in config['splits']:
        for number,item in enumerate(i for i in manifest['scenes'] if i['split']==split):
            p=root/item['path'];assert sha(p)==manifest['artifacts'][item['path']]
            scene=load_scene(p)
            with np.load(p.with_suffix('.fields.npz')) as f:target=f['fields'][25:125].copy();support=f['support'][25:125].copy()
            for regime in config['regimes']:
                obs,_=observations(scene,25,regime,7301+number)
                arrays=features(obs,scene.road);arrays.update(target=target,support=support,adjacency=adjacency(scene.road))
                path=output/f'{split}-{number}-{regime}.npz'
                if path.exists():raise ValueError('Incomplete cache preserved; new namespace required')
                np.savez_compressed(path,**arrays)
                entries.append(dict(path=path.name,sha256=sha(path),split=split,number=number,regime=regime,scene=scene.scene_id,source_path=str(p),history_speed=float(np.mean(arrays['reference'][:,0])),date=manifest['source_dates'][0]))
    result=dict(config=config,entries=entries);write_immutable(output/'manifest.json',result);return result
