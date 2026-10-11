"""Frozen Phase4E validation or gated independent-date scoring."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from orchestra_wm.i24.data import sha
from orchestra_wm.i24_continuous.source import write_immutable
from orchestra_wm.i24_continuous.evaluation import energy_score
from orchestra_wm.i24_phase4b.metrics import marginal_crps,sample_diversity
from orchestra_wm.i24_phase4e.cache import load_cache
from orchestra_wm.i24_phase4d.model import ResidualFields
from orchestra_wm.i24_phase4d.baselines import DiffusionPersistence

p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True);p.add_argument('--output',required=True);p.add_argument('--samples',type=int,default=8);p.add_argument('--ablations',action='store_true');p.add_argument('--cache',default='outputs/i24_phase4e/cache_v1');p.add_argument('--split',choices=['validation','test'],default='validation');p.add_argument('--freeze');a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
if a.split=='test':
 if not a.freeze:raise ValueError('Independent evaluation requires a frozen validation promotion receipt')
 if a.ablations:raise ValueError('Unregistered independent-test ablations forbidden')
 frozen=json.loads(Path(a.freeze).read_text())
 if frozen['status']!='VALIDATION_PASSED_TEST_PROTOCOL_FROZEN' or a.samples!=frozen['samples'] or sha(Path(a.cache)/'manifest.json')!=frozen['test_cache_sha256']:raise ValueError('Independent protocol mismatch')
 for path,digest in frozen['artifacts'].items():
  if sha(path)!=digest:raise ValueError('Frozen evaluation artifact changed')
 if set(a.runs)!=set(frozen['runs']):raise ValueError('Frozen selected runs differ')
_,items=load_cache(a.cache,allowed_splits=('test',) if a.split=='test' else ('train','validation'));items=[(e,b) for e,b in items if e['split']==a.split]
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
diffroot=Path('outputs/i24_phase4e/diffusion_baseline');diff=DiffusionPersistence();s=torch.load(diffroot/'checkpoints/last.pt',weights_only=False);diff.load_state_dict(s['model']);diff.eval()
checkpoints[str(diffroot/'checkpoints/last.pt')]=sha(diffroot/'checkpoints/last.pt')
protocol=dict(split=a.split,cache_manifest_sha256=sha(Path(a.cache)/'manifest.json'),freeze_sha256=sha(a.freeze) if a.freeze else None,samples=a.samples,checkpoints=checkpoints,source_sha256=sha(Path(__file__)),scope='Independent November28/29' if a.split=='test' else 'November25 model-selection validation',selection='Common paired checkpoint chosen on November25 only')
write_immutable(out/'protocol.json',protocol)
if (out/'summary.json').exists():raise SystemExit('Completed evaluation preserved')
directory=out/'rollouts';directory.mkdir(exist_ok=True);rows=[];arrays=[]
def score(x,b,meta):
 target=b['target'].numpy();support=b['support'].numpy()
 for seconds in [5,10,20]:
  t=seconds*5-1;use=support[t];z=x[:,t,use];y=target[t,use];scale=np.array([30,100,3000]);lo,hi=np.quantile(z,[.05,.95],axis=0)
  r=dict(**meta,horizon_seconds=seconds,bins=int(use.sum()),macro_energy=energy_score(z/scale,y/scale),macro_coverage90=float(((y>=lo)&(y<=hi)).mean()),macro_diversity=sample_diversity(z/scale),negative_fraction=float((z<0).mean()),flow_identity_mae=float(abs(z[...,2]-3.6*z[...,0]*z[...,1]).mean()))
  r['speed_above55_fraction']=float((z[...,0]>55).mean());r['density_above200_fraction']=float((z[...,1]>200).mean())
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
   known=ref[:,1]>0;hs=float(ref[known,0].mean()) if known.any() else 0.;hd=float(ref[known,1].mean()) if known.any() else 0.
   identity=dict(date=e['date'],block=e['block'],variant=name,seed=seed,scene=e['scene'],scene_number=e['number'],regime=e['regime'],history_speed=hs,history_density=hd,history_known_bin_fraction=float(known.mean()),speed_regime='congested_<10' if hs<10 else ('intermediate_10_20' if hs<20 else 'free_>=20'),density_regime='low_<20' if hd<20 else ('moderate_20_40' if hd<40 else 'high_>=40'))
   score(x,b,identity)
   if not name.startswith('full_') or e['number']<3:
    path=directory/f"{e['number']}-{e['regime']}-{name}-{seed}.npz"
    if path.exists():
     with np.load(path) as old:assert np.array_equal(old['fields'],x)
    else:np.savez_compressed(path,fields=x,target=b['target'].numpy(),support=b['support'].numpy(),reference=ref)
    arrays.append(dict(**identity,path=str(path.relative_to(out)),sha256=sha(path),checkpoint_sha256=checkpoint))
  if e['regime']=='outage':print('Evaluated '+a.split+' scene',e['number']+1,flush=True)
frame=pd.DataFrame(rows);frame.to_csv(out/'evaluation.csv',index=False)
frame.groupby(['date','variant','seed','regime','horizon_seconds']).mean(numeric_only=True).to_csv(out/'per_day_table.csv')
frame.groupby(['variant','seed','regime','horizon_seconds']).mean(numeric_only=True).to_csv(out/'summary_table.csv')
frame.groupby(['variant','seed','regime','horizon_seconds','speed_regime']).agg(windows=('macro_energy','size'),macro_energy=('macro_energy','mean'),speed_mae=('speed_mae','mean'),density_mae=('density_mae','mean'),flow_mae=('flow_mae','mean'),coverage90=('macro_coverage90','mean')).to_csv(out/'regime_table.csv')
frame.groupby(['variant','seed','regime','horizon_seconds','density_regime']).agg(windows=('macro_energy','size'),macro_energy=('macro_energy','mean'),coverage90=('macro_coverage90','mean')).to_csv(out/'density_regime_table.csv')
primary=[]
for seed in sorted({s for n,s,v,m,ab,c in models}):
 d=frame.query("seed==@seed and regime=='dense' and horizon_seconds==10 and variant in ['residual_full','residual_aggregate']").pivot(index='scene',columns='variant',values='macro_energy')
 primary.append(dict(seed=seed,aggregate_minus_full=float((d.residual_aggregate-d.residual_full).mean()),windows=len(d),independent_days=len({e['date'] for e,b in items}),inference='Two test dates below five-day confirmation gate' if a.split=='test' else 'Validation-selected descriptive result; not independent evidence'))
write_immutable(out/'arrays_manifest.json',arrays);write_immutable(out/'summary.json',dict(primary=primary,evaluation_sha256=sha(out/'evaluation.csv'),independent_test_evaluated=a.split=='test'));print(json.dumps(primary,indent=2))
