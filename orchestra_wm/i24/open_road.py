"""Explicit open-boundary assumption, separate from learned within-road dynamics."""
import numpy as np
import torch
from dataclasses import replace
from orchestra_wm.i24.model import observe_history

@torch.no_grad()
def generate_open_road(model,observations,road,steps,seed=0,max_births=32):
    """Past-observed entry-rate Poisson inflow; not learned or known future entrants."""
    rng=np.random.default_rng(seed);state=observe_history(model,observations,road)
    visible=np.stack([o.visible for o in observations]);births=(~visible[:-1]&visible[1:]).sum()
    rate=float(births/max((len(observations)-1)*model.dt,model.dt))
    pool=np.concatenate([o.values[o.visible] for o in observations]);speed=float(np.median(pool[:,2]));length=float(np.median(pool[:,6]));width=float(np.median(pool[:,7]))
    generator=torch.Generator().manual_seed(seed);frames=[];born=0
    for step in range(steps):
        count=min(int(rng.poisson(rate*model.dt)),max_births-born)
        if count:
            ids=tuple(f'generated-entry-{born+i}' for i in range(count));new=model.initial_state({'road':road,'track_ids':ids});new.agents[:,:,0]=road.s_min+.1;new.agents[:,:,1]=torch.tensor(rng.choice(road.lane_centers,count));new.agents[:,:,2]=speed;new.agents[:,:,6]=length;new.agents[:,:,7]=width;new.seen[:]=True
            state=replace(state,**{k:torch.cat([getattr(state,k),getattr(new,k)],1) for k in ['agents','memory','valid','seen','age']},track_ids=state.track_ids+ids);born+=count
        state,p=model.imagine_step(state,generator=generator);frames.append({'agents':p['agents'].cpu().numpy(),'valid':p['valid'].cpu().numpy(),'track_ids':state.track_ids})
    return frames,{'mode':'open_road','inflow':'past-observed appearance rate Poisson assumption; appearance can be occlusion reacquisition','rate_per_second':rate,'generated_births':born,'max_births':max_births,'future_truth_used':False,'validated_real_inflow':False}
