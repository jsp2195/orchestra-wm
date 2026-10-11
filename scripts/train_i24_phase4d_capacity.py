"""Replay the completed seed101 schedule with only aggregate hidden size changed."""
import json,time,hashlib,argparse,resource
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from orchestra_wm.i24.data import sha
from orchestra_wm.i24.training import atomic_checkpoint
from orchestra_wm.i24_continuous.source import write_immutable
from orchestra_wm.i24_phase4d.training import load_cache,collate,score_validation
from orchestra_wm.i24_phase4d.model import ResidualFields,energy
p=argparse.ArgumentParser();p.add_argument('--parent',default='outputs/i24_phase4d/residual_seed101');p.add_argument('--output',default='outputs/i24_phase4d/capacity16_seed101');a=p.parse_args()
parent=Path(a.parent);done=json.loads((parent/'completed.json').read_text());original=json.loads((parent/'run_manifest.json').read_text());out=Path(a.output);out.mkdir(parents=True,exist_ok=True);cp=out/'checkpoints';cp.mkdir(exist_ok=True)
cfg=dict(original['config']);cfg.update(dim=16,max_updates=done['updates_per_model'],selected_update=done['selected_update'],variants=['aggregate'],selection='Replay parent selection index and LR schedule; capacity-only control',parent_fingerprint=original['fingerprint'])
sources={**original['sources'],str(Path(__file__)):sha(Path(__file__))};fingerprint=hashlib.sha256(json.dumps(dict(config=cfg,sources=sources),sort_keys=True).encode()).hexdigest();write_immutable(out/'run_manifest.json',dict(config=cfg,sources=sources,fingerprint=fingerprint))
if (out/'completed.json').exists():raise SystemExit('Completed capacity control preserved')
torch.set_num_threads(2);_,items=load_cache('outputs/i24_phase4d/cache_v1');groups={(e['number'],e['regime']):b for e,b in items if e['split']=='train'};n=len({e['number'] for e,b in items if e['split']=='train'});schedule=pd.read_csv(parent/'learning_curve.csv').set_index('step')
torch.manual_seed(101);m=ResidualFields(16);o=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);rng=np.random.default_rng(101);rows=[];start=time.monotonic();begin=0;elapsed=0
if (cp/'last.pt').exists():
 saved=torch.load(cp/'last.pt',weights_only=False);assert saved['fingerprint']==fingerprint;m.load_state_dict(saved['models']['aggregate']);o.load_state_dict(saved['optimizers']['aggregate']);rng.bit_generator.state=saved['numpy_rng'];rows=saved['rows'];begin=saved['step'];elapsed=saved['seconds']
for step in range(begin+1,cfg['max_updates']+1):
 regime=['dense','dense','sparse50','outage'][int(rng.integers(4))];b=collate([groups[(int(i),regime)] for i in rng.integers(n,size=4)]);noise=int(rng.integers(2**30));assert regime==schedule.loc[step,'regime']
 o.zero_grad();x=m(b,'aggregate',4,noise);loss=energy(x,b['target'],b['support']);assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True);o.step();row=dict(step=step,train=float(loss.detach()),gradient=float(norm))
 if step==1 or step%50==0:
  row['validation']=score_validation({'aggregate':m},items)['aggregate'];print(json.dumps(row),flush=True)
  for g in o.param_groups:g['lr']=float(schedule.loc[step,'lr'])
 rows.append(row);state=dict(models={'aggregate':m.state_dict()},optimizers={'aggregate':o.state_dict()},numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),rows=rows,step=step,seconds=elapsed+time.monotonic()-start,fingerprint=fingerprint,config=cfg)
 atomic_checkpoint(cp/'last.pt',state)
 if step==cfg['selected_update']:atomic_checkpoint(cp/'best.pt',state)
pd.DataFrame(rows).to_csv(out/'learning_curve.csv',index=False);write_immutable(out/'completed.json',dict(seed=101,updates_per_model=cfg['max_updates'],selected_update=cfg['selected_update'],best_checkpoint_sha256=sha(cp/'best.pt'),parameters=sum(p.numel() for p in m.parameters()),wall_seconds=elapsed+time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Capacity-only aggregate control; parent training/selection schedule replayed'))
