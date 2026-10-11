"""Bounded numerical pilot on authenticated train/validation caches, no test inputs."""
import argparse,json,time,resource
from pathlib import Path
import numpy as np
import torch
from orchestra_wm.i24.data import sha
from orchestra_wm.i24_phase4d.model import ResidualFields,energy
from orchestra_wm.i24_phase4e.cache import load_cache
from orchestra_wm.i24_phase4e.training import collate
from orchestra_wm.i24_continuous.source import write_immutable
p=argparse.ArgumentParser();p.add_argument('--cache',default='outputs/i24_phase4e/cache_v1');p.add_argument('--output',default='outputs/i24_phase4e/pilot.json');a=p.parse_args()
torch.set_num_threads(2);_,items=load_cache(a.cache);train=[(e,x) for e,x in items if e['split']=='train' and e['regime']=='dense']
assert {e['date'] for e,x in train}=={'2022-11-21','2022-11-22','2022-11-23'}
batch=collate([train[i][1] for i in np.linspace(0,len(train)-1,4,dtype=int)])
torch.manual_seed(101);models={v:ResidualFields(32) for v in ['full','aggregate']};models['aggregate'].load_state_dict(models['full'].state_dict())
rows=[];start=time.monotonic()
for variant,m in models.items():
    opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)
    for step in range(4):
        t=time.monotonic();opt.zero_grad();x=m(batch,variant,4,101+step);loss=energy(x,batch['target'],batch['support'])
        assert torch.isfinite(loss) and torch.isfinite(x).all();loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True)
        active=sum(int((p.grad.abs()>0).sum()) for p in m.parameters() if p.grad is not None)
        opt.step();rows.append(dict(variant=variant,step=step+1,loss=float(loss.detach()),gradient_norm=float(norm),active_parameters=active,seconds=time.monotonic()-t,negative_fraction=float((x<0).float().mean()),flow_identity_mae=float((x[...,2]-3.6*x[...,0]*x[...,1]).abs().mean().detach())))
        assert rows[-1]['negative_fraction']==0
result=dict(status='FINITE_AUTHENTIC_NUMERICAL_PILOT',updates=8,models_discarded=True,primary_training_starts_fresh=True,
            cache_sha256=sha(Path(a.cache)/'manifest.json'),rows=rows,wall_seconds=time.monotonic()-start,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
write_immutable(a.output,result);print(json.dumps(result,indent=2))
