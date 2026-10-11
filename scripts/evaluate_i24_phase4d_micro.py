"""Supported historical-cohort validation only; no total-population macro scoring."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from orchestra_wm.i24.data import load_scene,sha
from orchestra_wm.i24_continuous.data import observations
from orchestra_wm.i24_continuous.source import write_immutable
from orchestra_wm.i24_continuous.evaluation import energy_score
from orchestra_wm.i24_phase4b.metrics import boundary_rates,marginal_crps
from orchestra_wm.i24_phase4c.evaluation import baseline
from orchestra_wm.i24_phase4d.micro import prepare,KinematicResidual
p=argparse.ArgumentParser();p.add_argument('--run',default='outputs/i24_phase4d/micro_seed101');a=p.parse_args();root=Path(a.run);out=root/'validation';out.mkdir(parents=True,exist_ok=True);private=out/'examples';private.mkdir(exist_ok=True)
torch.set_num_threads(2);cfg=json.loads((root/'config.json').read_text());done=json.loads((root/'completed.json').read_text());path=root/'checkpoints/best.pt';assert sha(path)==done['checkpoint_sha256'];assert sha(Path('orchestra_wm/i24_phase4d/micro.py'))==cfg['model_sha256']
s=torch.load(path,weights_only=False);models={v:KinematicResidual() for v in cfg['variants']}
for v,m in models.items():m.load_state_dict(s['models'][v]);m.eval()
write_immutable(out/'protocol.json',dict(samples=8,selected_update=done['selected_update'],checkpoint_sha256=sha(path),source_sha256=sha(Path(__file__)),scope='Same validation used for selection; not independent test'))
if (out/'summary.json').exists():raise SystemExit('Completed micro evaluation preserved')
data=Path('outputs/i24_phase4b/development_v1');manifest=json.loads((data/'data_manifest.json').read_text());rows=[];examples=[]
with torch.no_grad():
 for number,item in enumerate(i for i in manifest['scenes'] if i['split']=='validation'):
  scene=load_scene(data/item['path'])
  for regime in ['dense','outage']:
   b=prepare(scene,regime,7301+number);obs,_=observations(scene,25,regime,7301+number);truth=b['target'].numpy();valid=b['support'].numpy()
   forecasts={'constant_velocity':baseline(obs,scene,'constant_velocity')['agents']}
   forecasts.update({v:m(b,v,8,7103+number).numpy() for v,m in models.items()})
   for v,x in forecasts.items():
    assert np.isfinite(x).all();mean=x.mean(0)
    for horizon in [5,10,20]:
     steps=5*horizon;use=valid[:steps];last=valid[steps-1];row=dict(variant=v,scene=scene.scene_id,regime=regime,horizon_seconds=horizon,valid_samples=int(use.sum()),valid_endpoint=int(last.sum()),ade_m=float(np.linalg.norm(mean[:steps,:,:2]-truth[:steps,:,:2],axis=-1)[use].mean()) if use.any() else None)
     row['fde_m']=float(np.linalg.norm(mean[steps-1,last,:2]-truth[steps-1,last,:2],axis=-1).mean()) if last.any() else None
     if use.any():row['acceleration_above8_fraction']=float((np.linalg.norm(x[:,:steps,:,4:6],axis=-1)[:,use]>8).mean())
     if last.any():
      z=x[:,steps-1,last,:2];y=truth[steps-1,last,:2];lo,hi=np.quantile(z,[.05,.95],axis=0)
      row.update(xy_coverage90=float(((y>=lo)&(y<=hi)).mean()),position_crps_m=float(marginal_crps(z,y).mean()),micro_energy_scaled=energy_score(z/np.array([100,3.6576]),y/np.array([100,3.6576])),mean_sample_fde=float(np.linalg.norm(z-y,axis=-1).mean()))
      dimensions=x[:,steps-1,last,6:8];pairs=np.triu_indices(int(last.sum()),1)
      if len(pairs[0]):
       overlap=(abs(z[:,:,None]-z[:,None,:])<(dimensions[:,:,None]+dimensions[:,None,:])/2).all(-1)
       row['endpoint_overlap_pair_fraction']=float(overlap[:,pairs[0],pairs[1]].mean())
     row.update(boundary_rates(x[:,:steps],use,scene.road));rows.append(row)
    if number<3:
     f=private/f'{number}-{regime}-{v}.npz';np.savez_compressed(f,agents=x,truth=truth,support=valid);examples.append(dict(path=str(f.relative_to(out)),sha256=sha(f),checkpoint_sha256=sha(path) if v!='constant_velocity' else None))
frame=pd.DataFrame(rows);frame.to_csv(out/'evaluation.csv',index=False);frame.groupby(['variant','regime','horizon_seconds']).mean(numeric_only=True).to_csv(out/'summary_table.csv');write_immutable(out/'summary.json',dict(evaluation_sha256=sha(out/'evaluation.csv'),examples=examples,independent_test_evaluated=False));print(frame.query("regime=='dense'").groupby(['variant','horizon_seconds'])[['ade_m','fde_m','xy_coverage90']].mean().to_string())
