"""Fit a strong stochastic persistence control without validation selection."""
import argparse,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from orchestra_wm.i24.data import sha
from orchestra_wm.i24.training import atomic_checkpoint
from orchestra_wm.i24_continuous.source import write_immutable
from orchestra_wm.i24_phase4e.training import load_cache,collate
from orchestra_wm.i24_phase4d.baselines import DiffusionPersistence
from orchestra_wm.i24_phase4d.model import energy
p=argparse.ArgumentParser();p.add_argument('--output',default='outputs/i24_phase4e/diffusion_baseline');a=p.parse_args()
root=Path(a.output);root.mkdir(parents=True,exist_ok=True);cp=root/'checkpoints';cp.mkdir(exist_ok=True)
cfg=dict(seed=101,updates=500,learning_rate=.01,samples=4,batch_size=4,selection='Fixed final 500, train-only; no validation fit',source_sha256=sha(Path(__file__)),model_sha256=sha(Path('orchestra_wm/i24_phase4d/baselines.py')),cache_sha256=sha(Path('outputs/i24_phase4e/cache_v1/manifest.json')))
write_immutable(root/'config.json',cfg)
if (root/'completed.json').exists():raise SystemExit('Already completed; preserved')
torch.set_num_threads(2);_,items=load_cache('outputs/i24_phase4e/cache_v1');groups={(e['number'],e['regime']):x for e,x in items if e['split']=='train'};n=len({e['number'] for e,x in items if e['split']=='train'})
m=DiffusionPersistence();o=torch.optim.Adam(m.parameters(),lr=.01);rng=np.random.default_rng(101);start=time.monotonic();rows=[];begin=0
if (cp/'last.pt').exists():
 s=torch.load(cp/'last.pt',weights_only=False);assert s['config']==cfg;m.load_state_dict(s['model']);o.load_state_dict(s['optimizer']);rng.bit_generator.state=s['rng'];rows=s['rows'];begin=s['step']
for step in range(begin+1,501):
 regime=['dense','dense','sparse50','outage'][int(rng.integers(4))];b=collate([groups[(int(i),regime)] for i in rng.integers(n,size=4)]);noise=int(rng.integers(2**30))
 o.zero_grad();x=m(b,4,noise);loss=energy(x,b['target'],b['support']);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True);o.step();rows.append(dict(step=step,loss=float(loss.detach())))
 atomic_checkpoint(cp/'last.pt',dict(model=m.state_dict(),optimizer=o.state_dict(),rng=rng.bit_generator.state,step=step,config=cfg,rows=rows))
pd.DataFrame(rows).to_csv(root/'learning_curve.csv',index=False)
write_immutable(root/'completed.json',dict(updates=500,sigma=m.log_sigma.detach().exp().tolist(),wall_seconds=time.monotonic()-start,sha256=sha(cp/'last.pt'),parameters=2))
print((root/'completed.json').read_text())
