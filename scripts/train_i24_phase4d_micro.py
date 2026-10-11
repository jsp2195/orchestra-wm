"""Bounded paired kinematic-residual trial; no future populations or macro loss."""
import json,time,hashlib,argparse,resource
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from orchestra_wm.i24.data import load_scene,sha
from orchestra_wm.i24.training import atomic_checkpoint
from orchestra_wm.i24_continuous.source import write_immutable
from orchestra_wm.i24_phase4b.training import verify_preservation
from orchestra_wm.i24_phase4d.micro import prepare,KinematicResidual,micro_energy
p=argparse.ArgumentParser();p.add_argument('--output',default='outputs/i24_phase4d/micro_seed101');p.add_argument('--wall-seconds',type=int,default=900);a=p.parse_args()
verify_preservation('work/phase4b/private_backup_receipt.json');torch.set_num_threads(2)
root=Path('outputs/i24_phase4b/development_v1');out=Path(a.output);out.mkdir(parents=True,exist_ok=True);cp=out/'checkpoints';cp.mkdir(exist_ok=True)
cfg=dict(seed=101,max_updates=500,min_updates=150,validation_every=25,patience=4,min_delta=.0001,lr=.001,samples=4,dim=24,tau_seconds=2,variants=['graph','independent'],source_sha256=sha(Path(__file__)),model_sha256=sha(Path('orchestra_wm/i24_phase4d/micro.py')),data_sha256=sha(root/'data_manifest.json'),selection='Common minimum paired validation joint micro energy at5/10/20s; separate from macro experiment; no test access')
fingerprint=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest();write_immutable(out/'config.json',cfg)
if (out/'completed.json').exists():raise SystemExit('Completed micro trial preserved')
manifest=json.loads((root/'data_manifest.json').read_text());data={s:[] for s in ['train','validation']}
for split in data:
 for number,item in enumerate(x for x in manifest['scenes'] if x['split']==split):
  path=root/item['path'];assert sha(path)==manifest['artifacts'][item['path']];scene=load_scene(path)
  data[split].append({regime:prepare(scene,regime,7301+number) for regime in ['dense','outage']})
torch.manual_seed(101);models={v:KinematicResidual() for v in cfg['variants']};models['independent'].load_state_dict(models['graph'].state_dict());optim={v:torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001) for v,m in models.items()};rng=np.random.default_rng(101)
rows=[];best=float('inf');best_step=0;stale=0;step0=0;elapsed=0
if (cp/'last.pt').exists():
 s=torch.load(cp/'last.pt',weights_only=False);assert s['fingerprint']==fingerprint
 for v,m in models.items():m.load_state_dict(s['models'][v]);optim[v].load_state_dict(s['optimizers'][v])
 rng.bit_generator.state=s['numpy_rng'];torch.set_rng_state(s['torch_rng']);rows=s['rows'];step0=s['step'];best=s['best'];best_step=s['best_step'];stale=s['stale'];elapsed=s['seconds']
start=time.monotonic();stop='MAX_UPDATES'
for step in range(step0+1,501):
 if time.monotonic()-start>a.wall_seconds:raise TimeoutError('Resume identical command; atomic state retained')
 b=data['train'][int(rng.integers(len(data['train'])))]['dense' if rng.random()<.75 else 'outage'];noise=int(rng.integers(2**30));row=dict(step=step)
 for v,m in models.items():
  optim[v].zero_grad();x=m(b,v,4,noise);loss=micro_energy(x,b['target'],b['support']);assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True);optim[v].step();row[v+'_train']=float(loss.detach());row[v+'_gradient']=float(norm)
 if step==1 or step%25==0:
  with torch.no_grad():
   for v,m in models.items():row[v+'_validation']=float(np.mean([float(micro_energy(m(b['dense'],v,8,8001+i)[:,[24,49,99]],b['dense']['target'][[24,49,99]],b['dense']['support'][[24,49,99]])) for i,b in enumerate(data['validation'])]))
  shared=np.mean([row[v+'_validation'] for v in models]);row['shared_validation']=float(shared)
  if shared<best-.0001:best=float(shared);best_step=step;stale=0
  else:stale+=1
  if stale==2:
   for o in optim.values():
    for g in o.param_groups:g['lr']*=.5
  print(json.dumps(row),flush=True)
 rows.append(row);state=dict(models={v:m.state_dict() for v,m in models.items()},optimizers={v:o.state_dict() for v,o in optim.items()},numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),rows=rows,step=step,best=best,best_step=best_step,stale=stale,seconds=elapsed+time.monotonic()-start,fingerprint=fingerprint,config=cfg)
 atomic_checkpoint(cp/'last.pt',state)
 if best_step==step:atomic_checkpoint(cp/'best.pt',state)
 if step>=150 and stale>=4:stop='VALIDATION_NO_IMPROVEMENT';break
pd.DataFrame(rows).to_csv(out/'learning_curve.csv',index=False);recent=pd.DataFrame(rows).dropna(subset=['shared_validation']).tail(4).shared_validation
write_immutable(out/'completed.json',dict(updates_per_model=step,selected_update=best_step,stop=stop,best_validation=best,validation_recent_range=float(recent.max()-recent.min()),wall_seconds=elapsed+time.monotonic()-start,parameters_each=sum(p.numel() for p in models['graph'].parameters()),peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,checkpoint_sha256=sha(cp/'best.pt'),scope='Single-day validation; historical-cohort kinematic baseline, no individual arrivals or total-population equality'))
print((out/'completed.json').read_text())
