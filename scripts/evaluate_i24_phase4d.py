"""Validation-only scored arrays; independent-test opening is not implemented."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from orchestra_wm.i24.data import sha
from orchestra_wm.i24_continuous.source import write_immutable
from orchestra_wm.i24_continuous.evaluation import energy_score
from orchestra_wm.i24_phase4b.metrics import marginal_crps,sample_diversity
from orchestra_wm.i24_phase4d.training import load_cache
from orchestra_wm.i24_phase4d.model import ResidualFields
from orchestra_wm.i24_phase4d.baselines import DiffusionPersistence

p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True);p.add_argument('--output',required=True);p.add_argument('--samples',type=int,default=8);p.add_argument('--ablations',action='store_true');a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
_,items=load_cache('outputs/i24_phase4d/cache_v1');items=[(e,b) for e,b in items if e['split']=='validation']
models=[];checkpoints={}
for directory in a.runs:
 root=Path(directory);done=json.loads((root/'completed.json').read_text());path=root/'checkpoints/best.pt';assert sha(path)==done['best_checkpoint_sha256']
 saved=torch.load(path,weights_only=False);run=json.loads((root/'run_manifest.json').read_text());assert saved['fingerprint']==run['fingerprint']
 assert hashlib.sha256(json.dumps(dict(config=run['config'],sources=run['sources']),sort_keys=True).encode()).hexdigest()==run['fingerprint']
 for f,h in run['sources'].items():assert sha(Path(f))==h
 for variant in saved['models']:
  m=ResidualFields(saved['config']['dim']);m.load_state_dict(saved['models'][variant]);m.eval()
  label=f'residual_{variant}' if saved['config']['dim']==32 else f'capacity{saved["config"]["dim"]}_{variant}'
  models.append((label,saved['config']['seed'],variant,m,None,sha(path)))
  if a.ablations and variant=='full' and saved['config']['seed']==101:
   for ab in ['micro_removed','graph_off','memoryless','latent_off','deterministic']:models.append((f'full_{ab}',101,'full',m,ab,sha(path)))
 checkpoints[str(path)]=sha(path)
diffroot=Path('outputs/i24_phase4d/diffusion_baseline');diff=DiffusionPersistence();s=torch.load(diffroot/'checkpoints/last.pt',weights_only=False);diff.load_state_dict(s['model']);diff.eval()
checkpoints[str(diffroot/'checkpoints/last.pt')]=sha(diffroot/'checkpoints/last.pt')
protocol=dict(split='validation',samples=a.samples,checkpoints=checkpoints,source_sha256=sha(Path(__file__)),scope='Developmental November21 validation only; no test outcomes accessed',selection='Frozen common paired-validation checkpoint; same validation used for selection, not an unbiased test estimate')
write_immutable(out/'protocol.json',protocol)
if (out/'summary.json').exists():raise SystemExit('Completed evaluation preserved')
directory=out/'rollouts';directory.mkdir(exist_ok=True);rows=[];arrays=[]
def score(x,b,meta):
 target=b['target'].numpy();support=b['support'].numpy()
 for seconds in [5,10,20]:
  t=seconds*5-1;use=support[t];z=x[:,t,use];y=target[t,use];scale=np.array([30,100,3000]);lo,hi=np.quantile(z,[.05,.95],axis=0)
  r=dict(**meta,horizon_seconds=seconds,bins=int(use.sum()),macro_energy=energy_score(z/scale,y/scale),macro_coverage90=float(((y>=lo)&(y<=hi)).mean()),macro_diversity=sample_diversity(z/scale),negative_fraction=float((z<0).mean()),flow_identity_mae=float(abs(z[...,2]-3.6*z[...,0]*z[...,1]).mean()))
  for k,name in enumerate(['speed','density','flow']):
   r[name+'_mae']=float(abs(z.mean(0)-y)[:,k].mean());r[name+'_bias']=float((z.mean(0)-y)[:,k].mean());r[name+'_crps']=float(marginal_crps(z,y)[:,k].mean());r[name+'_coverage90']=float(((y>=lo)&(y<=hi))[:,k].mean());r[name+'_width90']=float((hi-lo)[:,k].mean())
  rows.append(r)
with torch.no_grad():
 for e,b in items:
  ref=b['reference'].numpy();physical=ref.copy();physical[:,:2]=np.maximum(physical[:,:2],.05);physical[:,2]=3.6*physical[:,0]*physical[:,1]
  forecasts=[('aggregate_persistence',0,None,np.repeat(ref[None,None],100,axis=1)),('physical_persistence',0,None,np.repeat(physical[None,None],100,axis=1)),('diffusion_persistence',101,checkpoints[str(diffroot/'checkpoints/last.pt')],diff(b,a.samples,7103+e['number']).numpy())]
  for name,seed,variant,model,ablation,checkpoint in models:
   if ablation and e['regime']!='dense':continue
   forecasts.append((name,seed,checkpoint,model(b,variant,a.samples,7103+e['number'],ablation=ablation).numpy()))
  for name,seed,checkpoint,x in forecasts:
   assert np.isfinite(x).all()
   identity=dict(variant=name,seed=seed,scene=e['scene'],scene_number=e['number'],regime=e['regime'],history_speed=e['history_speed'],speed_regime='congested_<10' if e['history_speed']<10 else ('intermediate_10_20' if e['history_speed']<20 else 'free_>=20'))
   score(x,b,identity)
   if not name.startswith('full_') or e['number']<3:
    path=directory/f"{e['number']}-{e['regime']}-{name}-{seed}.npz"
    if path.exists():
     with np.load(path) as old:assert np.array_equal(old['fields'],x)
    else:np.savez_compressed(path,fields=x,target=b['target'].numpy(),support=b['support'].numpy(),reference=ref)
    arrays.append(dict(**identity,path=str(path.relative_to(out)),sha256=sha(path),checkpoint_sha256=checkpoint))
  if e['regime']=='outage':print('Evaluated validation scene',e['number']+1,flush=True)
frame=pd.DataFrame(rows);frame.to_csv(out/'evaluation.csv',index=False)
frame.groupby(['variant','seed','regime','horizon_seconds']).mean(numeric_only=True).to_csv(out/'summary_table.csv')
frame.groupby(['variant','seed','regime','horizon_seconds','speed_regime']).agg(windows=('macro_energy','size'),macro_energy=('macro_energy','mean'),speed_mae=('speed_mae','mean'),density_mae=('density_mae','mean'),flow_mae=('flow_mae','mean'),coverage90=('macro_coverage90','mean')).to_csv(out/'regime_table.csv')
primary=[]
for seed in sorted({s for n,s,v,m,ab,c in models}):
 d=frame.query("seed==@seed and regime=='dense' and horizon_seconds==10 and variant in ['residual_full','residual_aggregate']").pivot(index='scene',columns='variant',values='macro_energy')
 primary.append(dict(seed=seed,aggregate_minus_full=float((d.residual_aggregate-d.residual_full).mean()),windows=len(d),independent_days=1,inference='Validation-selected descriptive result only; not independent information-value evidence'))
write_immutable(out/'arrays_manifest.json',arrays);write_immutable(out/'summary.json',dict(primary=primary,evaluation_sha256=sha(out/'evaluation.csv'),independent_test_evaluated=False));print(json.dumps(primary,indent=2))
