from pathlib import Path
import json,hashlib,gzip,base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from orchestra_wm.i24.data import write_json,sha


def make_figures(out,scenes,cfg):
    dest=out/'figures';dest.mkdir(exist_ok=True);paths=[]
    plt.rcParams.update({'figure.constrained_layout.use':True,'axes.spines.top':False,'axes.spines.right':False,'font.size':10})
    def save(name):
        p=dest/f'{name}.png';plt.savefig(p,dpi=130);plt.close();paths.append(str(p))
    label='SYNTHETIC FIXTURE — not I-24' if cfg['source_kind']=='SYNTHETIC_FIXTURE' else 'I-24 MOTION reconstructed trajectories'
    scene=scenes[0];fig,ax=plt.subplots(figsize=(9,4))
    for i in range(len(scene.track_ids)):
        v=scene.existence[:,i];ax.plot(scene.values[v,i,0],scene.values[v,i,1],lw=1)
    ax.set(xlabel='Travel-direction s (m), straightened roadway',ylabel='Lateral d (m)',title=label);save('trajectory_geometry')
    training=pd.read_csv(out/'training.csv');fig,ax=plt.subplots(figsize=(8,4))
    for name,f in training.groupby('variant'):v=f.groupby('step').validation_loss.mean();ax.plot(v.index,v,label=name)
    ax.legend(fontsize=8);ax.set(xlabel='Optimizer updates',ylabel='Validation objective',title=label);save('training')
    f=pd.read_csv(out/'evaluation.csv');dense=f[f.regime=='dense'];fig,ax=plt.subplots(figsize=(8,4))
    for name,g in dense.groupby('variant'):
        if name=='macro_only':continue
        v=g.groupby('horizon_seconds').fde_m.mean();ax.plot(v.index,v,marker='o',label=name)
    ax.legend(fontsize=7);ax.set(xlabel='Autonomous horizon (s)',ylabel='FDE (m)',title=label);save('rollout_errors')
    predfile=out/'rollouts'/f'full-{cfg["training_seeds"][0]}-0-dense.npz';z=dict(np.load(predfile));fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,a,title in [(axs[0],z['truth'],'Withheld fixture' if cfg['source_kind']=='SYNTHETIC_FIXTURE' else 'Withheld reality'),(axs[1],z['agents'][0],'Autonomous sample 0')]:
        for i in range(a.shape[1]):ax.plot(a[:,i,0],a[:,i,1],lw=1)
        ax.set(title=title,xlabel='s (m)',ylabel='d (m)')
    fig.suptitle(label);save('true_generated')
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,x,title in [(axs[0],z['truth_fields'][:,:,0],'Recorded/fixture speed'),(axs[1],z['fields'].mean(0)[:,:,0],'Generated mean speed')]:
        im=ax.imshow(x.T,origin='lower',aspect='auto',extent=[0,max(cfg['horizons_seconds']),0,x.shape[1]],vmin=0,vmax=30);fig.colorbar(im,ax=ax,label='m/s');ax.set(xlabel='Forecast time (s)',ylabel='Lane × spatial bin',title=title)
    fig.suptitle(label);save('time_space')
    partial=f[(f.variant=='full')&(f.horizon_seconds==10)].groupby('regime')[['fde_m','trajectory_coverage90']].mean();fig,axs=plt.subplots(1,2,figsize=(11,4));partial.fde_m.plot.bar(ax=axs[0]);partial.trajectory_coverage90.plot.bar(ax=axs[1]);axs[1].axhline(.9,color='red',ls='--');axs[0].set(ylabel='FDE (m)');axs[1].set(ylabel='90% interval coverage');save('sensing_calibration')
    dense[dense.horizon_seconds==10].groupby('variant')[['macro_coverage90','overlap_pair_fraction','road_violation_fraction']].mean().plot.bar(figsize=(10,4));plt.xticks(rotation=25);plt.title(label);save('calibration_physics')
    full=f[(f.variant=='full')&(f.regime.isin(['dense','reset_memory','graph_off','permutation']))];full.groupby(['regime','horizon_seconds']).macro_energy.mean().unstack().plot.bar(figsize=(9,4));plt.ylabel('Macro energy score');plt.title(label);save('interaction_memory')
    return paths


def make_demo(out,cfg):
    dest=out/'demo';dest.mkdir(exist_ok=True);entries=[]
    for index in range(2):
        for regime in ['dense','sparse50','sparse20','blind','outage']:
            p=out/'rollouts'/f'full-{cfg["training_seeds"][0]}-{index}-{regime}.npz'
            if not p.exists():continue
            meta=json.loads(p.with_suffix('.json').read_text());z=dict(np.load(p));entry={k:z[k].tolist() for k in ['agents','fields','valid','truth','target_mask','truth_fields','observed','observed_mask']};entry['metadata']=meta;entry['evaluation_array_path']=str(p);entries.append(entry)
    payload={'label':'SYNTHETIC FIXTURE — NOT AN I-24-TRAINED MODEL' if cfg['source_kind']=='SYNTHETIC_FIXTURE' else 'I-24 MOTION — autonomous learned forecasts','entries':entries}
    data=json.dumps(payload,separators=(',',':'),allow_nan=False);compressed=gzip.compress(data.encode(),mtime=0);(dest/'data.json.gz').write_bytes(compressed)
    if (dest/'data.json').exists():(dest/'data.json').unlink()
    template=Path('orchestra_wm/i24/demo_template.html').read_text();html=template.replace('__PAYLOAD_GZIP__',base64.b64encode(compressed).decode()).replace('__VISIBLE_LABEL__',payload['label']);(dest/'index.html').write_text(html)
    links=[{'evaluation_array_path':e['evaluation_array_path'],'rollout_sha256':e['metadata']['rollout_sha256'],'checkpoint_sha256':e['metadata']['checkpoint_sha256']} for e in entries]
    write_json(dest/'provenance.json',{'kind':cfg['source_kind'],'data_sha256':sha(dest/'data.json.gz'),'evaluation_links':links,'rule':'Arrays are copied without resimulation or interpolation from saved evaluation rollouts.'})
    return str(dest/'index.html')


def verify_demo(out):
    d=json.loads(gzip.decompress((out/'demo/data.json.gz').read_bytes()))
    for e in d['entries']:
        path=Path(e['evaluation_array_path']);assert sha(path)==e['metadata']['rollout_sha256']
        with np.load(path) as original:
            for key in ['agents','fields','truth','observed','observed_mask']:assert np.array_equal(np.asarray(e[key]),original[key]),key
    return len(d['entries'])


def make_gif(out,cfg):
    """Small numerical-rollout export; no interpolated/replayed replacement forecasts."""
    from PIL import Image,ImageDraw
    p=out/'rollouts'/f'full-{cfg["training_seeds"][0]}-0-dense.npz';z=dict(np.load(p));meta=json.loads(p.with_suffix('.json').read_text());road=meta['road'];frames=[];indices=list(range(0,len(z['truth']),5))
    for t in indices:
        im=Image.new('RGB',(1000,460),'#081321');draw=ImageDraw.Draw(im)
        draw.text((20,12),'SYNTHETIC FIXTURE - NOT AN I-24-TRAINED MODEL' if cfg['source_kind']=='SYNTHETIC_FIXTURE' else 'I-24 autonomous generated rollout',fill='#ffcf91')
        draw.text((20,34),f'Forecast {(t+1)*cfg["dt"]:.1f} s | checkpoint {meta["checkpoint_sha256"][:16]} | sample 0',fill='white')
        for col,(name,values,mask,color) in enumerate([('Withheld truth (revealed)',z['truth'][t],z['target_mask'][t],'#ffc56c'),('Learned imagination',z['agents'][0,t],z['valid'][0,t],'#64edb5')]):
            left=20+col*490;draw.text((left,62),name,fill='white')
            for lane in road['lane_centers']:
                y=90+(lane-3)/16*300;draw.line((left,y,left+460,y),fill='#34546a')
            for a,valid in zip(values,mask):
                if not valid:continue
                x=left+(a[0]-road['s_min'])/(road['s_max']-road['s_min'])*460;y=90+(a[1]-3)/16*300
                if left<=x<=left+460 and 85<=y<415:draw.rectangle((x-4,y-3,x+4,y+3),fill=color)
        draw.text((20,432),'Straightened roadway metres. Prediction violations remain in evaluation.csv; no real-data skill claim.',fill='#b1c5d7');frames.append(im)
    path=out/'demo/rollout.gif';frames[0].save(path,save_all=True,append_images=frames[1:],duration=120,loop=0)
    write_json(out/'demo/gif_provenance.json',{'evaluation_array_path':str(p),'rollout_sha256':sha(p),'sample':0,'time_indices':indices,'no_interpolation':True})
    return str(path)
