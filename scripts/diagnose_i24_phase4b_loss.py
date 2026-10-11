"""No optimizer steps: decompose the preserved pilot's loss on one training scene."""
import json
from pathlib import Path
import numpy as np,torch
from orchestra_wm.i24.data import load_scene
from orchestra_wm.i24.model import observe_history
from orchestra_wm.i24_continuous.model import ContinuousWorldModel
from orchestra_wm.i24_continuous.data import observations

root=Path('outputs/i24_continuous/pilot_0700_wb');manifest=json.loads((root/'data_manifest.json').read_text());torch.set_num_threads(2)
item=next(s for s in manifest['scenes'] if s['split']=='train');path=root/item['path'];scene=load_scene(path)
with np.load(path.with_suffix('.fields.npz')) as a:fields=a['fields'].copy();support=a['support'].copy()
obs,cohort=observations(scene,25,'dense',991);target=torch.tensor(scene.values[25:75,cohort])[None];masks=torch.tensor(scene.existence[25:75,cohort])[None];tf=torch.tensor(fields[25:75])[None];fs=torch.tensor(support[25:75])[None]
results=[]
for variant in ('full','macro_only','micro_only','independent','no_crossscale','deterministic'):
 model=ContinuousWorldModel(32,4,variant);model.load_state_dict(torch.load(root/'checkpoints'/f'{variant}-101.pt',weights_only=False,map_location='cpu')['model']);model.eval();components=[]
 with torch.no_grad():
  state=observe_history(model,obs,scene.road);generator=torch.Generator().manual_seed(991)
  for step in range(50):
   previous=state;state,p=model.imagine_step(state,generator=generator);mask=masks[:,step,:,None];denom=mask.sum().clamp_min(1);fmask=fs[:,step,:,None];fd=(fmask.sum()*3).clamp_min(1)
   micro=((((p['agents'][:,:,:4]-target[:,step,:,:4])/model.scale[:4])**2)*mask).sum()/denom
   macro=((((p['fields']-tf[:,step])/model.field_scale)**2)*fmask).sum()/fd
   consistency=((((p['fields']-p['derived_fields'])/model.field_scale)**2)*fmask).sum()/fd
   _,posterior=model._transition(previous,generator=generator,posterior_target=torch.where(mask,target[:,step],previous.agents))
   acceleration=(target[:,step,:,2:4]-previous.agents[:,:,2:4])/model.dt
   nll=((.5*((acceleration-posterior['acc_mean'])/posterior['acc_std']).square()+posterior['acc_std'].log())*mask).sum()/denom
   fnll=((.5*((tf[:,step]-posterior['field_mean'])/posterior['field_std']).square()+posterior['field_std'].log())*fmask).sum()/fd
   actual_previous=torch.tensor(scene.values[24+step,cohort,2:4])[None]
   actual_acceleration=(target[:,step,:,2:4]-actual_previous)/model.dt
   acceleration_discrepancy=((acceleration-actual_acceleration).abs()*mask).sum()/(2*denom)
   components.append([float(v) for v in (micro,macro,.05*consistency,.001*nll,.0001*fnll,.001*posterior['kl'],acceleration_discrepancy)])
 means=np.mean(components,axis=0);keys=['micro_mse','macro_mse','weighted_consistency','weighted_acceleration_nll','weighted_field_nll','weighted_kl','pseudo_acceleration_vs_true_acceleration_mae']
 results.append(dict(variant=variant,scene=item['path'],scope='Dense first training scene; no optimizer update; raw terms shown even where variant disables them',**dict(zip(keys,means.tolist()))))
Path('outputs/i24_phase4b/audit/loss_decomposition.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
