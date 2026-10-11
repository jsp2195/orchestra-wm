import base64,gzip,json
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
from orchestra_wm.i24.data import sha,load_scene

pilot=Path('outputs/i24_continuous/pilot_0700_wb');root=Path('outputs/i24_phase4b/gallery');root.mkdir(parents=True,exist_ok=True)
manifest=json.loads((pilot/'evaluation_manifest.json').read_text());data=json.loads((pilot/'data_manifest.json').read_text())
tests=[i for i in data['scenes'] if i['split']=='test'];scene=load_scene(pilot/tests[0]['path']);road=scene.road
entries=[]
for e in manifest['entries']:
 if e['variant']=='full' and e['regime'] in ('dense','outage'):
  assert sha(pilot/e['path'])==e['sha256']
  with np.load(pilot/e['path']) as a:
   entry=dict(scene=e['scene_number'],regime=e['regime'],checkpoint_sha256=e['checkpoint_sha256'],rollout_sha256=e['sha256'],source_path=e['path'],
    agents=a['agents'][...,:2].tolist(),valid=a['predicted_valid'].tolist(),fields=a['fields'].tolist(),truth=a['truth'][...,:2].tolist(),truth_valid=a['valid'].tolist(),
    target_fields=a['target_fields'].tolist(),support=a['field_support'].tolist(),history=a['history'][...,:2].tolist(),history_valid=a['history_valid'].tolist())
   # JSON roundtrip retains exact saved float32 values, all eight independent draws.
   assert np.array_equal(np.asarray(entry['agents']),a['agents'][...,:2])
   assert np.array_equal(np.asarray(entry['fields']),a['fields'])
   entries.append(entry)
payload=dict(entries=entries,road=dict(length=road.s_max,lane_centers=road.lane_centers,lane_width=road.lane_width),dt=.2,
 label='Historical 100-update pilot checkpoints; Phase4B audit visualizations. Not retrained or calibrated.',future_entrants='Fixed historical cohort; realized future entrants never supplied to model')
encoded=base64.b64encode(gzip.compress(json.dumps(payload,separators=(',',':')).encode())).decode()
(root/'payload.json.gz').write_bytes(gzip.compress(json.dumps(payload,separators=(',',':')).encode()))
D=next(e for e in entries if e['scene']==0 and e['regime']=='dense');O=next(e for e in entries if e['scene']==0 and e['regime']=='outage')
ag=np.asarray(D['agents']);truth=np.asarray(D['truth']);valid=np.asarray(D['valid']);tv=np.asarray(D['truth_valid']);fields=np.asarray(D['fields']);target=np.asarray(D['target_fields']);support=np.asarray(D['support'])
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#f6f8fc'})
def save(name):
 plt.tight_layout();plt.savefig(root/(name+'.png'),dpi=145);plt.close()
plt.figure(figsize=(11,5))
for j in range(len(scene.track_ids)):
 use=scene.existence[:,j];plt.plot(scene.time[use]-scene.time[0],scene.values[use,j,0],lw=.5,alpha=.55,color='#06768d')
plt.xlabel('Physical seconds');plt.ylabel('Westbound roadway s (m)');plt.title('Authentic reconstructed vehicle tracks · first historical test window');save('real_trajectories')
for component,name,units in [(0,'speed','m/s'),(1,'density','veh/km/lane'),(2,'flow','veh/h/lane')]:
 fig,axs=plt.subplots(1,2,figsize=(11,4),sharey=True)
 a=np.where(support,target[:,:,component],np.nan).reshape(100,4,8);a=np.nanmean(a,axis=1)
 b=np.nanmean(np.where(support,fields[:,:,:,component].mean(0),np.nan).reshape(100,4,8),axis=1)
 vmax=max(np.nanpercentile(a,98),np.nanpercentile(b,98))
 for ax,v,title in zip(axs,[a,b],['Recorded reconstructed fields','Autonomous ensemble mean']):
  im=ax.imshow(v.T,origin='lower',aspect='auto',extent=[.2,20,0,road.s_max],vmin=0,vmax=vmax,cmap='viridis');ax.set_title(title);ax.set_xlabel('Future seconds');fig.colorbar(im,ax=ax,label=units)
 axs[0].set_ylabel('Roadway s (m)');fig.suptitle(name.title()+' · lanes averaged; unknown target bins masked');save(name+'_kymograph')
eval=pd.read_csv(pilot/'evaluation.csv');audit=pd.read_csv('outputs/i24_phase4b/audit/supplementary_pilot_metrics.csv');curves=pd.read_csv(pilot/'training.csv')
fig,axs=plt.subplots(2,3,figsize=(12,6))
for ax,(v,g) in zip(axs.flat,curves.groupby('variant')):
 ax.plot(g.step,g.train_loss,label='training');ax.plot(g.step,g.validation_loss,label='validation');ax.set_title(v);ax.set_xlabel('Updates');ax.set_yscale('log')
axs[0,0].legend();fig.suptitle('Original pilot learning curves · variant objectives differ; convergence not established');save('learning_curves')
dense=eval[eval.regime=='dense'];means=dense.groupby(['variant','horizon_seconds']).mean(numeric_only=True).reset_index()
fig,axs=plt.subplots(1,2,figsize=(12,4))
for v,g in means.groupby('variant'):
 axs[0].plot(g.horizon_seconds,g.macro_energy,'o-',label=v);axs[1].plot(g.horizon_seconds,g.fde_m,'o-',label=v)
axs[0].set_ylabel('Joint macro energy (lower better)');axs[1].set_ylabel('Ensemble-mean FDE (m)')
for ax in axs:ax.set_xlabel('Autonomous horizon (s)')
axs[0].legend(fontsize=7,ncol=2);fig.suptitle('Historical held-out comparisons · three windows, one contiguous block');save('baseline_horizons')
fig,axs=plt.subplots(1,2,figsize=(11,4))
for v in ['full','macro_only','micro_only','independent']:
 g=means[means.variant==v];axs[0].plot(g.horizon_seconds,g.macro_coverage90,'o-',label=v)
 g=audit[(audit.regime=='dense')&(audit.variant==v)].groupby('horizon_seconds').mean(numeric_only=True);axs[1].plot(g.index,g.macro_diversity,'o-',label=v)
axs[0].axhline(.9,ls='--',color='black',label='nominal 90%');axs[0].set_ylim(0,1);axs[0].set_ylabel('Empirical marginal coverage');axs[0].legend(fontsize=7);axs[1].set_ylabel('Normalized pairwise sample diversity')
for ax in axs:ax.set_xlabel('Horizon (s)')
fig.suptitle('Nonzero stochastic diversity does not establish calibration');save('calibration')
fig,axs=plt.subplots(1,2,figsize=(11,4))
for v in ['full','micro_only','independent','constant_velocity']:
 g=audit[(audit.regime=='dense')&(audit.variant==v)].groupby('horizon_seconds').mean(numeric_only=True)
 axs[0].plot(g.index,g.decoder_vs_hard_generated_macro_rmse_scaled,'o-',label=v)
g=audit[(audit.regime=='dense')&(audit.variant=='full')].groupby('horizon_seconds').mean(numeric_only=True);axs[1].plot(g.index,g.future_cohort_density_fraction,'o-')
axs[0].legend(fontsize=8);axs[0].set_ylabel('Decoder / hard generated fields RMSE (scaled)');axs[1].set_ylabel('Known-cohort / all target density');axs[1].set_ylim(0,1)
for ax in axs:ax.set_xlabel('Horizon (s)')
fig.suptitle('Consistency mismatch and missing future population · post-hoc audit');save('micro_macro_consistency')
fig,axs=plt.subplots(1,2,figsize=(11,4))
for regime,g in eval[eval.variant=='full'].groupby('regime'):
 g=g.groupby('horizon_seconds').mean(numeric_only=True);axs[0].plot(g.index,g.macro_energy,'o-',label=regime);axs[1].plot(g.index,g.macro_coverage90,'o-')
axs[0].legend(fontsize=8);axs[0].set_ylabel('Joint macro energy');axs[1].set_ylabel('90% empirical coverage')
for ax in axs:ax.set_xlabel('Horizon (s)')
fig.suptitle('Emulated roadside missingness · same full checkpoint');save('missing_sensors')
fig,ax=plt.subplots(figsize=(12,4));ax.axis('off')
boxes=[(.03,.58,'Past detections\n5 s at 5 Hz'),(.27,.58,'Agent GRU + graph\n32 hidden dimensions'),(.53,.58,'Autonomous prior\n4 latent dimensions'),(.78,.58,'Generated vehicles\n+ predicted exits'),(.03,.08,'Visible historical fields\n1 s exposure'),(.27,.08,'Field GRU\n32 hidden dimensions'),(.53,.08,'Field recurrent decoder\nspeed / density / flow'),(.78,.08,'Scoring only:\nwithheld future truth')]
for x,y,t in boxes:ax.text(x,y,t,transform=ax.transAxes,bbox=dict(boxstyle='round,pad=.7',fc='#dcebf8',ec='#446688'),va='center',fontsize=9)
for x,y,xx,yy in [(.18,.58,.26,.58),(.46,.58,.52,.58),(.71,.58,.77,.58),(.19,.08,.26,.08),(.45,.08,.52,.08),(.62,.45,.62,.19),(.38,.45,.57,.2)]:ax.annotate('',xy=(xx,yy),xytext=(x,y),xycoords='axes fraction',arrowprops=dict(arrowstyle='->'))
ax.set_title('Historical architecture · autonomous prior recurrence; future labels enter losses/scoring only');save('information_flow')

def road_panel(draw,box,xy,mask,title,color):
 x0,y0,x1,y1=box;draw.rectangle(box,fill='#13263a');draw.text((x0+6,y0+4),title,fill='white')
 for lane in road.lane_centers:
  yy=y0+23+(lane-3.6576)/14.6304*(y1-y0-28);draw.line((x0,yy,x1,yy),fill='#425368')
 for pos,use in zip(xy,mask):
  if not use:continue
  xx=x0+np.clip(pos[0]/road.s_max,0,1)*(x1-x0);yy=y0+23+(pos[1]-3.6576)/14.6304*(y1-y0-28)
  if y0+20<=yy<y1:draw.rectangle((xx-2,yy-1,xx+2,yy+1),fill=color)
frames=[]
for t in range(0,126,3):
 im=Image.new('RGB',(1000,220),'#f6f8fc');d=ImageDraw.Draw(im);d.text((12,10),f'AUTHENTIC I-24 | roadway coordinates | recorded +{t*.2:.1f}s | reconstructed population',fill='#142d46')
 road_panel(d,(10,40,990,200),scene.values[t,:,:2],scene.existence[t],'Actual traffic · approximate documented lanes','#ffb252');frames.append(im)
frames[0].save(root/'01_real_highway.gif',save_all=True,append_images=frames[1:],duration=150,loop=0)
frames=[]
for t in range(0,100,4):
 im=Image.new('RGB',(1000,760),'#f6f8fc');d=ImageDraw.Draw(im);d.text((12,8),f'ACTUAL + EIGHT AUTONOMOUS SAMPLES | +{(t+1)*.2:.1f}s | historical 100-update checkpoint',fill='#142d46')
 for i in range(9):
  row,col=divmod(i,3);box=(10+330*col,35+240*row,330+330*col,265+240*row)
  road_panel(d,box,truth[t] if i==0 else ag[i-1,t],tv[t] if i==0 else valid[i-1,t],'Recorded cohort' if i==0 else f'Prior sample {i}','#ffb252' if i==0 else '#47d9ed')
 frames.append(im)
frames[0].save(root/'02_eight_generated_futures.gif',save_all=True,append_images=frames[1:],duration=160,loop=0)
frames=[]
for t in range(0,100,4):
 fig,axs=plt.subplots(2,2,figsize=(8,5))
 for row,comp in enumerate([0,1]):
  for col,array in enumerate([target,fields.mean(0)]):
   v=np.nanmean(np.where(support,array[:,:,comp],np.nan).reshape(100,4,8),axis=1).T.copy();v[:,t+1:]=np.nan
   axs[row,col].imshow(v,origin='lower',aspect='auto',extent=[0,20,0,road.s_max],vmin=0,vmax=10 if comp==0 else 60,cmap='viridis');axs[row,col].set_title(('Recorded' if col==0 else 'Generated mean')+(' speed' if comp==0 else ' density'));axs[row,col].set_xlabel('Future seconds')
 fig.suptitle(f'Collective evolution +{(t+1)*.2:.1f}s · no verified wave-propagation claim');fig.tight_layout();fig.canvas.draw();frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[...,:3].copy()));plt.close(fig)
frames[0].save(root/'03_collective_evolution.gif',save_all=True,append_images=frames[1:],duration=160,loop=0)
frames=[]
for t in range(0,100,4):
 im=Image.new('RGB',(1000,420),'#f6f8fc');d=ImageDraw.Draw(im);d.text((12,8),f'EMULATED FINAL-HISTORY OUTAGE | +{(t+1)*.2:.1f}s | all 8 samples overlaid',fill='#142d46')
 for i,e in enumerate([D,O]):
  box=(10,35+i*190,990,210+i*190)
  road_panel(d,box,np.asarray(e['agents'])[0,t],np.asarray(e['valid'])[0,t],e['regime']+' · blue futures / orange recorded','#47d9ed')
  for sample in range(1,8):
   for pos,use in zip(np.asarray(e['agents'])[sample,t],np.asarray(e['valid'])[sample,t]):
    xx=10+np.clip(pos[0]/road.s_max,0,1)*980;yy=box[1]+23+(pos[1]-3.6576)/14.6304*(box[3]-box[1]-28)
    if use and box[1]+20<=yy<box[3]:d.point((xx,yy),fill='#47d9ed')
  for pos,use in zip(np.asarray(e['truth'])[t],np.asarray(e['truth_valid'])[t]):
   xx=10+np.clip(pos[0]/road.s_max,0,1)*980;yy=box[1]+23+(pos[1]-3.6576)/14.6304*(box[3]-box[1]-28)
   if use and box[1]+20<=yy<box[3]:d.rectangle((xx-2,yy-1,xx+2,yy+1),fill='#ffb252')
 frames.append(im)
frames[0].save(root/'04_partial_observability.gif',save_all=True,append_images=frames[1:],duration=160,loop=0)
html=Path('orchestra_wm/i24_phase4b/viewer.html').read_text().replace('__PAYLOAD__',encoded)
(root/'index.html').write_text(html)
(root/'provenance.json').write_text(json.dumps(dict(source='PRESERVED_PILOT_CHECKPOINTS',entries=[{k:v for k,v in e.items() if k in ('scene','regime','checkpoint_sha256','rollout_sha256','source_path')} for e in entries],array_verification='Exact all-eight sample positions and fields checked against saved evaluated NPZs before encoding; no model rerun',payload_sha256=sha(root/'payload.json.gz'),html_sha256=sha(root/'index.html')),indent=2)+'\n')
print('Gallery built',sum(p.stat().st_size for p in root.iterdir()),flush=True)
