import numpy as np
import pandas as pd
from orchestra_wm.evaluation.common import context
from orchestra_wm.planning.oracle import true_rollout


def named_plans(env,horizon):
    mask=np.array([v.connected for v in env.vehicles])
    plans={name:np.zeros((horizon,env.n),int) for name in
           ['maintain','one_yield','one_proceed','staggered','aggressive','conservative']}
    controlled=np.flatnonzero(mask)
    if len(controlled):
        plans['one_yield'][:,controlled[0]]=-1;plans['one_proceed'][:,controlled[0]]=1
    plans['aggressive'][:,mask]=1;plans['conservative'][:,mask]=-1
    for t in range(horizon):
        for j,i in enumerate(controlled): plans['staggered'][t,i]=1 if (t//4+j)%2 else -1
    return plans


def action_audit(cfg,path):
    rows=[]
    horizons=[1,5,10,20]
    for seed in cfg['seeds']:
        for family in cfg['scenarios']:
            env,_=context(cfg,family,seed+8000)
            plans=named_plans(env,max(horizons))
            reference,refc=true_rollout(env,plans['maintain'])
            for name,plan in plans.items():
                states,components=true_rollout(env,plan)
                for h in horizons:
                    delta=components[:h]-refc[:h]
                    rows.append({'seed':seed,'scenario':family,'plan':name,'horizon':h,
                        'position_divergence':float(np.linalg.norm((states[h-1,:,:2]-reference[h-1,:,:2])*100,axis=-1).mean()),
                        'speed_divergence':float(np.abs(states[h-1,:,2]-reference[h-1,:,2]).mean()*15),
                        'queue_divergence':float(np.abs(delta[:,1]).mean()),'delay_divergence':float(np.abs(delta[:,0]).mean()),
                        'conflict_divergence':float(np.abs(delta[:,4]).mean()),'collision_divergence':float(np.abs(delta[:,3]).mean()),
                        'travel_time_proxy_divergence':float(np.abs(delta[:,0]).sum()*env.dt)})
    frame=pd.DataFrame(rows);frame.to_csv(path,index=False)
    signal=frame[(frame.horizon==10)&(frame.plan=='aggressive')].position_divergence.mean()
    if signal<.25: raise RuntimeError(f'Insufficient true action excitation ({signal:.3f}m); do not train')
    return frame
