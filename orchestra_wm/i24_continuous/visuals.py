"""Real-source congestion plots and exact saved-checkpoint forecast demonstration."""
import json
from pathlib import Path

import ijson
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image,ImageDraw

from orchestra_wm.i24.data import I24Adapter,sha
from .source import write_immutable


def source_figures(root,output):
    root,output=Path(root),Path(output);directory=output/'figures';directory.mkdir(parents=True,exist_ok=True)
    meta=json.loads((root/'acquisition.json').read_text());audit=json.loads((root/'source_audit.json').read_text())
    adapter=I24Adapter(meta);start=meta['start_unix_s'];duration=meta['end_unix_s']-start
    exposure=np.zeros((round(duration*5),32));distance=np.zeros_like(exposure);points=[]
    length=(audit['selection']['x_max']-audit['selection']['x_min'])*.3048/8
    for entry in meta['files']:
        if sha(root/entry['path'])!=entry['sha256']:raise ValueError('Source figure input integrity mismatch')
        with (root/entry['path']).open('rb') as handle:
            for record in ijson.items(handle,'item',use_float=True):
                frame,_=adapter.record(record,meta['session'],entry['sha256'])
                if frame is None:continue
                t=frame.time.to_numpy();s=frame.s.to_numpy();v=frame.vs.to_numpy();y=frame.d.to_numpy()/.3048
                dt=np.r_[0,np.diff(t)];f=np.floor((t-start)*5).astype(int);bins=np.floor(s/length).astype(int);lane=np.floor((y-12)/12).astype(int)
                valid=frame.usable.to_numpy()&(f>=0)&(f<len(exposure))&(bins>=0)&(bins<8)&(lane>=0)&(lane<4)
                index=lane[valid]*8+bins[valid]
                np.add.at(exposure,(f[valid],index),dt[valid]);np.add.at(distance,(f[valid],index),v[valid]*dt[valid])
                chosen=np.flatnonzero(valid)[::25]
                if len(chosen):points.append(np.stack([(t[chosen]-start)/60,s[chosen],v[chosen]],1))
    points=np.concatenate(points)
    fig,ax=plt.subplots(figsize=(11,5));artist=ax.scatter(points[:,0],points[:,1],c=points[:,2],s=.3,cmap='turbo',vmin=0,vmax=30,rasterized=True)
    fig.colorbar(artist,ax=ax,label='Backward-derived speed (m/s)');ax.set(xlabel='Minutes after 07:00 CST, 2022-11-21',ylabel='Westbound travel coordinate (m)',title='Authentic I-24 regional trajectories — source reconstruction, not forecasts')
    fig.tight_layout();fig.savefig(directory/'real_traffic_time_space.png',dpi=180);plt.close(fig)
    cumulative=np.vstack([np.zeros((1,32)),np.cumsum(exposure,axis=0)]);travel=np.vstack([np.zeros((1,32)),np.cumsum(distance,axis=0)])
    end=np.arange(1,len(exposure)+1);begin=np.maximum(0,end-5);seconds=(end-begin)*.2
    occ=cumulative[end]-cumulative[begin];metres=travel[end]-travel[begin]
    speed=np.divide(metres,occ,out=np.full_like(occ,np.nan),where=occ>0)
    density=np.where(occ>0,occ/(length*seconds[:,None])*1000,np.nan)
    flow=np.where(occ>0,metres/(length*seconds[:,None])*3600,np.nan)
    fig,axes=plt.subplots(3,1,figsize=(11,8),sharex=True)
    for ax,field,label,vmax in zip(axes,[speed,density,flow],['Speed (m/s)','Density (veh/km/lane)','Flow (veh/h/lane)'],[30,150,3000]):
        values=field.reshape(-1,4,8)
        observed=np.isfinite(values);total=np.nansum(values,axis=1)
        denominator=observed.sum(1)
        average=np.divide(total,denominator,out=np.full_like(total,np.nan),where=denominator>0)
        artist=ax.imshow(average.T,origin='lower',aspect='auto',extent=[0,duration/60,0,length*8],vmin=0,vmax=vmax,cmap='turbo')
        fig.colorbar(artist,ax=ax,label=label);ax.set_ylabel('WB metres')
    axes[0].set_title('Native 25Hz source: trailing 1s vehicle-time/distance exposure; empty support UNKNOWN')
    axes[-1].set_xlabel('Minutes after 07:00 CST, 2022-11-21');fig.tight_layout();fig.savefig(directory/'real_congestion_fields.png',dpi=180);plt.close(fig)
    return [str(directory/'real_traffic_time_space.png'),str(directory/'real_congestion_fields.png')]


def forecast_demo(output):
    output=Path(output);manifest=json.loads((output/'evaluation_manifest.json').read_text())
    directory=output/'demo';directory.mkdir(exist_ok=True);selected=[]
    for entry in manifest['entries']:
        if entry['variant']!='full' or entry['scene_number']!=0:continue
        path=output/entry['path']
        if sha(path)!=entry['sha256']:raise ValueError('Saved evaluation forecast integrity mismatch')
        with np.load(path) as arrays:
            item=dict(regime=entry['regime'],checkpoint_sha256=entry['checkpoint_sha256'],rollout_sha256=entry['sha256'],
                      mean_agents=arrays['agents'].mean(0)[:,:,:2].tolist(),
                      truth=arrays['truth'][:,:,:2].tolist(),valid=arrays['valid'].tolist(),
                      predicted_valid=(arrays['predicted_valid'].mean(0)>=.5).tolist(),
                      history=arrays['history'][:,:,:2].tolist(),history_valid=arrays['history_valid'].tolist(),
                      mean_fields=arrays['fields'].mean(0).tolist(),target_fields=arrays['target_fields'].tolist(),field_support=arrays['field_support'].tolist())
        selected.append(item)
    if not selected:raise ValueError('No full-model saved evaluation arrays')
    payload=json.dumps(dict(entries=selected,regional_length_m=999.744,dt=.2,source='REAL_I24',
                           warning='Preliminary 100-update, single-day regional pilot. Masks are EMULATED. No confirmed convergence or wave skill.'),separators=(',',':'),allow_nan=False)
    (directory/'payload.json').write_text(payload+'\n')
    html='''<!doctype html><html lang="en"><meta charset="utf-8"><title>ORCHESTRA — authentic continuous I-24 pilot</title>
<style>body{font:16px system-ui;background:#101821;color:#dce8f2;margin:30px auto;max-width:1100px}canvas{background:#18232e;width:100%;border-radius:8px}button,select,input{font:inherit;margin:12px 8px 12px 0}p{line-height:1.5}.small{color:#a8bacb;font-size:13px}</style>
<h1>ORCHESTRA: continuous I-24 pilot</h1><p>Real westbound recording · November 21, 2022 · 07:08 CST held-out interval. Forecasts start after five seconds of observed history and run autonomously for 20 seconds.</p>
<p class="small">Preliminary 100-update, single-day regional pilot. Partial observations and outages are emulated. These results do not establish convergence, traffic-wave skill, or generalization.</p>
<select id="regime"></select><button id="play">Play</button><label><input type="checkbox" id="truth" checked>Reveal recorded truth</label><input type="range" id="time" min="0" max="99" value="0"><span id="label"></span>
<canvas id="road" width="1100" height="320"></canvas><p class="small">Blue: saved full-model ensemble mean. Orange: recorded fixed-cohort truth. Source tracks may use offline future imagery. Empty bins have unknown coverage. Roadway coordinates are straightened metres.</p>
<canvas id="fields" width="1100" height="300"></canvas><p id="hash" class="small"></p>
<p class="small">Source: Gloudemans et al. (2023), I-24 MOTION: An instrument for freeway traffic science, Transportation Research Part C 155, 104311.</p>
<script>const DATA=__PAYLOAD__;const r=document.getElementById('regime'),slider=document.getElementById('time');
DATA.entries.forEach((e,i)=>{let o=document.createElement('option');o.value=i;o.textContent=e.regime;r.appendChild(o)});
function draw(){const e=DATA.entries[+r.value],t=+slider.value,c=document.getElementById('road'),x=c.getContext('2d');x.clearRect(0,0,c.width,c.height);x.fillStyle='#a8bacb';x.font='14px system-ui';
for(let i=0;i<4;i++){let y=40+i*65;x.strokeStyle='#506271';x.beginPath();x.moveTo(40,y+30);x.lineTo(1060,y+30);x.stroke();x.fillText('Lane '+(i+1),5,y)}
function cars(a,col,mask){x.fillStyle=col;a.forEach((v,i)=>{if(mask[i]){let px=40+v[0]/DATA.regional_length_m*1020,py=30+(v[1]-3.6576)/3.6576*65;x.fillRect(px-3,py-2,7,5)}})}
if(document.getElementById('truth').checked)cars(e.truth[t],'#ffb454',e.valid[t]);cars(e.mean_agents[t],'#50b8ff',e.predicted_valid[t]);
document.getElementById('label').textContent='Forecast +'+((t+1)*.2).toFixed(1)+' s';document.getElementById('hash').textContent='Checkpoint SHA256: '+e.checkpoint_sha256;
const f=document.getElementById('fields'),q=f.getContext('2d');q.clearRect(0,0,f.width,f.height);const names=['Speed (m/s)','Density (veh/km/lane)','Flow (veh/h/lane)'],scales=[30,100,3000];
for(let j=0;j<3;j++){q.fillStyle='#dce8f2';q.fillText(names[j],12,j*95+15);for(let k=0;k<32;k++){let xx=130+k*29,yy=j*95+80;if(document.getElementById('truth').checked&&e.field_support[t][k]){q.fillStyle='#ffb454';q.fillRect(xx,yy-e.target_fields[t][k][j]/scales[j]*55,9,e.target_fields[t][k][j]/scales[j]*55)}q.fillStyle='#50b8ff';q.fillRect(xx+10,yy-e.mean_fields[t][k][j]/scales[j]*55,9,e.mean_fields[t][k][j]/scales[j]*55)}}}
r.onchange=draw;slider.oninput=draw;document.getElementById('truth').onchange=draw;let timer;document.getElementById('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;return}timer=setInterval(()=>{slider.value=(+slider.value+1)%100;draw()},200)};draw();</script></html>'''
    (directory/'index.html').write_text(html.replace('__PAYLOAD__',payload))
    dense=next(e for e in selected if e['regime']=='dense');frames=[]
    for t in range(0,100,2):
        image=Image.new('RGB',(1000,330),'#101821');draw=ImageDraw.Draw(image)
        draw.text((20,15),f'Authentic I-24 held-out pilot | forecast +{(t+1)*.2:.1f}s | orange truth, blue full-model mean',fill='white')
        for lane in range(4):draw.line((20,75+lane*60,980,75+lane*60),fill='#506271')
        for key,color in [('truth','#ffb454'),('mean_agents','#50b8ff')]:
            for i,v in enumerate(dense[key][t]):
                if dense['valid' if key=='truth' else 'predicted_valid'][t][i]:
                    x=20+v[0]/999.744*960;y=45+(v[1]-3.6576)/3.6576*60
                    draw.rectangle((x-3,y-2,x+3,y+2),fill=color)
        draw.text((20,300),'100 updates, seed 101, single day; preliminary. Fixed past cohort; emulated sensing.',fill='#a8bacb')
        frames.append(image)
    frames[0].save(directory/'real_checkpoint_forecast.gif',save_all=True,append_images=frames[1:],duration=200,loop=0)
    write_immutable(directory/'provenance.json',dict(source='REAL_I24',entries=[dict(regime=e['regime'],checkpoint_sha256=e['checkpoint_sha256'],rollout_sha256=e['rollout_sha256']) for e in selected],
                                                   payload_sha256=sha(directory/'payload.json'),html_sha256=sha(directory/'index.html'),
                                                   display='Exact ensemble means from saved evaluation arrays, not replay or fabricated predictions'))
    return str(directory/'index.html')
