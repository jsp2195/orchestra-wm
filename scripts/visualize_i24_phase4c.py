"""Build new figures from verified final-checkpoint arrays; raw payload stays private."""
import base64
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from orchestra_wm.i24.data import sha, load_scene
from orchestra_wm.i24_continuous.source import write_immutable

p=argparse.ArgumentParser()
p.add_argument('--training-output',default='outputs/i24_phase4b/matched_macro_seed101_u300')
p.add_argument('--evaluation-output',default='outputs/i24_phase4c/matched_macro_seed101_u300/development_test')
p.add_argument('--output',default='outputs/i24_phase4c/gallery')
args=p.parse_args()
training=Path(args.training_output);evaluation=Path(args.evaluation_output)
root=Path(args.output); root.mkdir(parents=True,exist_ok=True)
config=json.loads((training/'run_manifest.json').read_text())['config'];variants=config['variants']
joint='graph_off' in variants
motion_label='jointly supervised cohort' if joint else 'macro-trained motion diagnostic'
private=root/'private';private.mkdir(exist_ok=True)
data_root=Path('outputs/i24_phase4b/development_v1')
data=json.loads((data_root/'data_manifest.json').read_text())
manifest=json.loads((evaluation/'arrays_manifest.json').read_text())
summary=json.loads((evaluation/'summary.json').read_text())
assert sha(evaluation/'evaluation.csv')==summary['evaluation_sha256']
source_split='test' if summary['split']=='development_test' else 'validation'
items=[i for i in data['scenes'] if i['split']==source_split]
scene=load_scene(data_root/items[0]['path']);road=scene.road
entries=[]
for e in manifest:
    if e['scene_number']>=3 or e['variant'] not in variants:continue
    assert sha(evaluation/e['path'])==e['sha256']
    entry=dict(scene=e['scene_number'],regime=e['regime'],variant=e['variant'],
               checkpoint_sha256=e['checkpoint_sha256'],rollout_sha256=e['sha256'],source_path=e['path'])
    with np.load(evaluation/e['path']) as a:
        for k,original in [('agents','agents'),('valid','predicted_valid'),('fields','fields'),('truth','truth'),('truth_valid','valid'),('target_fields','target_fields'),('support','field_support'),('history','history'),('history_valid','history_valid')]:
            arr=a[original][...,:2] if k in ('agents','truth','history') else a[original]
            entry[k]=arr.tolist()
            assert np.array_equal(np.asarray(entry[k]),arr)
    entries.append(entry)
payload=dict(entries=entries,road=dict(s_min=road.s_min,s_max=road.s_max,lane_centers=road.lane_centers,lane_width=road.lane_width,bins=road.bins),dt=.2)
encoded=base64.b64encode(gzip.compress(json.dumps(payload,separators=(',',':')).encode())).decode()
html=Path('orchestra_wm/i24_phase4c/viewer.html').read_text().replace('__PAYLOAD__',encoded)
if joint:
    options=''.join(f'<option value="{v}">{v.replace("_"," ").title()}</option>' for v in variants if v!='full')
    html=html.replace('<option value="macro_only">Macro-only</option>',options)
    html=html.replace('Matched macro-information experiment:', 'Bounded joint-loss trial:')
    html=html.replace('300 updates per model, seed 101. The loss trains speed, density and flow forecasts. Full-model vehicle motion is a diagnostic and has no direct microscopic training loss.',
                      f'{config["updates"]} joint updates after the same 300-update macro warm start, seed 101. The loss jointly supervises historical-cohort trajectories and total retained-population fields. The graph-disabled control retains aggregate feedback.')
    html=html.replace('Macro-only has no vehicle predictions.','Individual future arrivals are not generated.')
    html=html.replace('macro-trained motion diagnostic','jointly supervised cohort')
if source_split=='validation':html=html.replace('Developmental window','Validation window')
(private/'demo.html').write_text(html)
D=next(e for e in entries if e['scene']==0 and e['regime']=='dense' and e['variant']=='full')
O=next(e for e in entries if e['scene']==0 and e['regime']=='outage' and e['variant']=='full')
agents=np.asarray(D['agents']);truth=np.asarray(D['truth']);valid=np.asarray(D['valid']);truth_valid=np.asarray(D['truth_valid'])
fields=np.asarray(D['fields']);target=np.asarray(D['target_fields']);support=np.asarray(D['support'])
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#f6f8fc'})
def save(name):
    plt.tight_layout();plt.savefig(root/(name+'.png'),dpi=140);plt.close()
fig,axs=plt.subplots(1,len(variants),figsize=(5.5*len(variants),4))
for ax,variant in zip(axs,variants):
    curve=pd.read_csv(training/f'{variant}_learning_curve.csv')
    ax.plot(curve.step,curve.train_macro_energy,alpha=.35,label='Training window')
    v=curve.dropna(subset=['validation_macro_energy'])
    ax.plot(v.step,v.validation_macro_energy,'o-',ms=3,label='All validation windows')
    ax.set_title(variant);ax.set_xlabel('Optimizer updates');ax.set_ylabel('Prior macro energy')
axs[0].legend();fig.suptitle('Macro energy component · fixed final checkpoints, no test selection');save('learning_curves')
if joint:
    fig,axs=plt.subplots(1,len(variants),figsize=(5.5*len(variants),4))
    for ax,variant in zip(axs,variants):
        curve=pd.read_csv(training/f'{variant}_learning_curve.csv');v=curve.dropna(subset=['validation_micro_energy'])
        ax.plot(curve.step,curve.train_micro_energy,alpha=.5,label='Train micro score');ax.plot(v.step,v.validation_micro_energy,'o-',label='Validation micro score')
        ax.set_title(variant);ax.set_xlabel('Additional joint updates');ax.set_ylabel('Normalized micro energy')
    axs[0].legend();fig.suptitle('Joint trial · micro component, separate from the matched macro experiment');save('joint_micro_learning_curves')
plt.figure(figsize=(11,4))
for j in range(len(scene.track_ids)):
    use=scene.existence[:,j];plt.plot(scene.time[use]-scene.time[0],scene.values[use,j,0],lw=.5,alpha=.6,color='#097b8b')
plt.xlabel('Physical seconds');plt.ylabel('Westbound s (m)');plt.title(f'First {summary["split"]} window · retained reconstructed trajectories');save('real_trajectories')
def lane_mean(array,k):
    x=np.where(support,array[:,:,k],np.nan).reshape(100,len(road.lane_centers),road.bins)
    count=np.isfinite(x).sum(1)
    return np.divide(np.nansum(x,axis=1),count,out=np.full((100,road.bins),np.nan),where=count>0)
for k,name,unit in [(0,'speed','m/s'),(1,'density','veh/km/lane'),(2,'flow','veh/h/lane')]:
    fig,axs=plt.subplots(1,2,figsize=(11,4),sharey=True)
    arrays=[lane_mean(target,k),lane_mean(fields.mean(0),k)]
    low=min(0,min(np.nanmin(a) for a in arrays));high=max(np.nanmax(a) for a in arrays)
    for ax,a,title in zip(axs,arrays,['Recorded fields','Autonomous ensemble mean']):
        im=ax.imshow(a.T,origin='lower',aspect='auto',extent=[.2,20,road.s_min,road.s_max],vmin=low,vmax=high,cmap='viridis');ax.set_title(title);ax.set_xlabel('Future seconds');fig.colorbar(im,ax=ax,label=unit)
    axs[0].set_ylabel('Westbound s (m)');fig.suptitle(f'{name.title()} · identical axes; unknown target bins masked');save(name+'_kymograph')
frame=pd.read_csv(evaluation/'evaluation.csv');means=frame.groupby(['variant','regime','horizon_seconds']).mean(numeric_only=True).reset_index();dense=means[means.regime=='dense']
fig,axs=plt.subplots(1,2,figsize=(11,4))
for variant,g in dense.groupby('variant'):
    axs[0].plot(g.horizon_seconds,g.macro_energy,'o-',label=variant)
    if variant in ('full','independent','graph_off','constant_velocity'):axs[1].plot(g.horizon_seconds,g.fde_m,'o-',label=variant)
axs[0].set_ylabel('Joint macro energy');axs[1].set_ylabel('Cohort FDE (m), motion diagnostic')
for ax in axs:ax.set_xlabel('Horizon (s)');ax.legend(fontsize=8)
fig.suptitle(f'Single-day {summary["split"]} · 23 windows, one contiguous block');save('baseline_horizons')
fig,axs=plt.subplots(1,3,figsize=(13,4))
for variant in variants:
    g=dense[dense.variant==variant]
    for ax,key in zip(axs,['macro_coverage90','macro_diversity','macro_negative_fraction']):ax.plot(g.horizon_seconds,g[key],'o-',label=variant)
for ax,title in zip(axs,['Marginal 90% empirical coverage','Normalized sample diversity','Negative generated fields']):ax.set_title(title);ax.set_xlabel('Horizon (s)')
axs[0].axhline(.9,ls='--',color='black');axs[0].set_ylim(0,1);axs[0].legend();save('calibration_and_validity')
fig,axs=plt.subplots(1,2,figsize=(11,4))
for (variant,regime),g in means[means.variant.isin(variants)].groupby(['variant','regime']):
    axs[0].plot(g.horizon_seconds,g.macro_energy,'o-',label=f'{variant} / {regime}');axs[1].plot(g.horizon_seconds,g.macro_coverage90,'o-')
axs[0].legend(fontsize=7);axs[0].set_ylabel('Macro energy');axs[1].set_ylabel('90% empirical coverage')
for ax in axs:ax.set_xlabel('Horizon (s)')
fig.suptitle('Emulated final-history outage · fixed checkpoints');save('missing_sensors')
fig,axs=plt.subplots(1,2,figsize=(11,4))
for variant,g in dense.groupby('variant'):
    axs[0].plot(g.horizon_seconds,g.flow_identity_mae_veh_h,'o-',label=variant)
g=dense[dense.variant=='full']
for key in ('lateral_departure_rate','reverse_speed_rate','longitudinal_exit_rate'):axs[1].plot(g.horizon_seconds,g[key],'o-',label=key)
axs[0].set_ylabel('Mean |q - 3.6 k v| (veh/h/lane)');axs[1].set_ylabel('Fraction of target-valid cohort sample times')
for ax in axs:ax.set_xlabel('Horizon (s)');ax.legend(fontsize=7)
fig.suptitle('Physical diagnostics · legal longitudinal exits shown separately');save('physical_consistency')
fig,axs=plt.subplots(1,2,figsize=(11,4))
g=dense[dense.variant=='full']
axs[0].plot(g.horizon_seconds,g.future_cohort_density_fraction,'o-');axs[0].set_ylabel('Observed-cohort / total target density')
axs[1].plot(g.horizon_seconds,g.total_decoder_vs_generated_cohort_rmse_scaled,'o-');axs[1].set_ylabel('Total-head vs generated-cohort scaled RMSE')
for ax in axs:ax.set_xlabel('Horizon (s)')
fig.suptitle('Different populations: decoder/cohort equality is not a valid conservation target');save('population_consistency')
fig,ax=plt.subplots(figsize=(12,4));ax.axis('off')
labels=[(.02,.68,'Past individual tracks\n5 seconds'),(.28,.68,'Agent GRU + graph\n32 dimensions'),(.55,.68,'Autonomous Gaussian prior\n4 latent dimensions'),(.80,.68,'Cohort motion\nnot directly supervised'),(.02,.15,'Past visible aggregate fields\nspeed / density / flow'),(.28,.15,'Field GRU\n32 dimensions'),(.55,.15,'Autonomous field decoder\nproper macro energy loss'),(.80,.15,'Withheld future fields\nscoring / training labels only')]
for x,y,label in labels:
    if joint:label=label.replace('not directly supervised','directly supervised').replace('proper macro energy loss','joint micro/macro scores')
    ax.text(x,y,label,transform=ax.transAxes,bbox=dict(boxstyle='round,pad=.6',fc='#deebf6',ec='#4a657f'),fontsize=8)
for x,y,xx,yy in [(.17,.68,.27,.68),(.46,.68,.54,.68),(.73,.68,.79,.68),(.19,.15,.27,.15),(.45,.15,.54,.15),(.65,.60,.65,.28),(.38,.59,.57,.27)]:ax.annotate('',xy=(xx,yy),xytext=(x,y),xycoords='axes fraction',arrowprops=dict(arrowstyle='->'))
ax.set_title('Joint trial · graph-off removes edge messages' if joint else 'Matched macro experiment · macro-only bypasses all agent modules');save('information_flow')

low=min(road.lane_centers)-road.lane_width/2;high=max(road.lane_centers)+road.lane_width/2
def panel(draw,box,xy,mask,title,color):
    x0,y0,x1,y1=box;draw.rectangle(box,fill='#14293e');draw.text((x0+5,y0+4),title,fill='white')
    for lane in road.lane_centers:
        yy=y0+24+(lane-low)/(high-low)*(y1-y0-30);draw.line((x0,yy,x1,yy),fill='#405d77')
    for pos,use in zip(xy,mask):
        if use and road.s_min<=pos[0]<road.s_max and low<=pos[1]<high:
            xx=x0+(pos[0]-road.s_min)/(road.s_max-road.s_min)*(x1-x0);yy=y0+24+(pos[1]-low)/(high-low)*(y1-y0-30)
            draw.rectangle((xx-2,yy-1,xx+2,yy+1),fill=color)
def gif(frames,name):frames[0].save(root/name,save_all=True,append_images=frames[1:],duration=180,loop=0)
frames=[]
for t in range(0,126,4):
    im=Image.new('RGB',(1000,220),'#f6f8fc');d=ImageDraw.Draw(im);d.text((12,10),f'REAL I-24 / retained reconstructed traffic / +{t*.2:.1f}s',fill='#19334d');panel(d,(10,40,990,200),scene.values[t,:,:2],scene.existence[t],'Westbound roadway coordinates / approximate lanes','#ffb35b');frames.append(im)
gif(frames,'01_real_highway.gif')
frames=[]
for t in range(0,100,5):
    im=Image.new('RGB',(1000,740),'#f6f8fc');d=ImageDraw.Draw(im);d.text((12,8),f'RECORDED + EIGHT AUTONOMOUS SAMPLES / +{(t+1)*.2:.1f}s / {motion_label}',fill='#19334d')
    for i in range(9):
        row,col=divmod(i,3);box=(10+330*col,35+230*row,330+330*col,250+230*row)
        panel(d,box,truth[t] if i==0 else agents[i-1,t],truth_valid[t] if i==0 else valid[i-1,t],'Recorded cohort' if i==0 else f'Prior sample {i}','#ffb35b' if i==0 else '#54dbea')
    frames.append(im)
gif(frames,'02_eight_futures.gif')
frames=[]
for t in range(0,100,5):
    fig,axs=plt.subplots(2,2,figsize=(8,5))
    for row,k in enumerate((0,1)):
        values=[lane_mean(target,k),lane_mean(fields.mean(0),k)];vmax=max(np.nanmax(a) for a in values)
        for col,a in enumerate(values):
            a=a.T.copy();a[:,t+1:]=np.nan
            axs[row,col].imshow(a,origin='lower',aspect='auto',extent=[0,20,road.s_min,road.s_max],vmin=0,vmax=vmax,cmap='viridis');axs[row,col].set_title(('Recorded' if col==0 else 'Generated mean')+(' speed' if k==0 else ' density'));axs[row,col].set_xlabel('Future seconds')
    fig.suptitle(f'Collective evolution +{(t+1)*.2:.1f}s / no verified propagation claim');fig.tight_layout();fig.canvas.draw();frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[...,:3].copy()));plt.close(fig)
gif(frames,'03_collective_evolution.gif')
frames=[]
for t in range(0,100,5):
    im=Image.new('RGB',(1000,430),'#f6f8fc');d=ImageDraw.Draw(im);d.text((12,8),f'EMULATED SENSOR OUTAGE / +{(t+1)*.2:.1f}s / eight generated samples overlaid',fill='#19334d')
    for j,e in enumerate((D,O)):
        box=(10,35+j*190,990,210+j*190)
        all_xy=np.asarray(e['agents'])[:,t].reshape(-1,2);all_mask=np.asarray(e['valid'])[:,t].reshape(-1)
        panel(d,box,all_xy,all_mask,e['regime']+' / '+motion_label,'#54dbea')
    frames.append(im)
gif(frames,'04_partial_observability.gif')
provenance=dict(evaluation_sha256=summary['evaluation_sha256'],checkpoints=summary['checkpoints'],
                selected_scenes=f'First three chronological {summary["split"]} windows, not performance-selected',
                exact_array_entries=len(entries),samples_each=8,private_html_sha256=sha(private/'demo.html'),
                source_sha256=sha(Path(__file__)),artifacts={p.name:sha(p) for p in root.iterdir() if p.suffix in ('.png','.gif')})
write_immutable(root/'provenance.json',provenance)
gallery='''# Phase 4C · checkpoint-backed evidence gallery

New seed-101, 300-update matched-macro checkpoints on authentic November 21 data.
These are preliminary single-day results. Vehicle forecasts are **macro-trained
motion diagnostics**, not a completed joint microscopic training experiment.
The historical Phase 4/4B galleries remain unchanged.

| Evidence | Preview |
|---|---|
| Real reconstructed highway traffic | ![Real highway](01_real_highway.gif) |
| Recorded cohort and eight autonomous futures | ![Eight futures](02_eight_futures.gif) |
| Collective field evolution | ![Fields](03_collective_evolution.gif) |
| Emulated sensor outage | ![Outage](04_partial_observability.gif) |

[Learning curves](learning_curves.png) · [Baseline comparisons](baseline_horizons.png) ·
[Calibration and validity](calibration_and_validity.png) · [Physical consistency](physical_consistency.png) · [Population consistency](population_consistency.png) ·
[Sensor outage](missing_sensors.png) · [Actual trajectories](real_trajectories.png) ·
[Speed](speed_kymograph.png) · [Density](density_kymograph.png) · [Flow](flow_kymograph.png) ·
[Information flow](information_flow.png)

The standalone interactive viewer is kept in the private recovery artifact because
it embeds selected authentic trajectory arrays. Its SHA256 and exact evaluated
array provenance are recorded in `provenance.json`; the source template is versioned.
The viewer supports three scenarios, both models, dense/outage history, eight prior
samples, recorded comparison, a physical-time scrubber and speed/density/flow bands.
No future truth initializes the forecasts. No congestion-wave propagation is claimed.
'''
if joint:
    gallery=gallery.replace('New seed-101, 300-update matched-macro checkpoints','Seed-101, 30-update joint checkpoints after an identical 300-update warm start')
    gallery=gallery.replace('These are preliminary single-day results. Vehicle forecasts are **macro-trained\nmotion diagnostics**, not a completed joint microscopic training experiment.',
                            'These are preliminary single-day joint-loss results, not a convergence or independent-date claim. Vehicle forecasts have direct cohort supervision; individual future arrivals are not generated.')
    gallery+='\n[Joint microscopic learning curves](joint_micro_learning_curves.png). Graph-off removes graph messages but retains aggregate feedback. Independent removes graph messages, shared microscopic innovations and macro feedback from vehicle dynamics; its separate head predicts total fields. All three share the same warm start and additional update budget.\n'
gallery+=f'\nDisplayed partition: **{summary["split"]}**.\n'
(root/'VISUAL_GALLERY.md').write_text(gallery)
print(json.dumps(provenance,indent=2))
