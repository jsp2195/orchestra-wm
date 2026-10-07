import numpy as np
import pandas as pd
import torch
from orchestra_wm.envs.traffic_env import TrafficEnv
from orchestra_wm.sim.observations import sense
from orchestra_wm.evaluation.common import infer,errors
from orchestra_wm.planning.oracle import true_rollout

@torch.no_grad()
def evaluate_interactions(cfg,models,out):
    rows=[];contrasts=[]
    for seed in cfg['seeds']:
        env=TrafficEnv({**cfg,'connected':2},'intersection',seed+30000)
        # Perpendicular leading AVs with near-synchronous arrivals.
        for i,lane in enumerate([0,2]):
            v=env.vehicles[i];v.lane=lane;v.route=(lane,);v.s=57;v.speed=5;v.desired_speed=10
        obs=sense(env);frames=[(obs,None,env.state_array())]
        inferred={k:infer(v,frames) for k,v in models.items() if k in ['orchestra','independent']}
        values={name:[] for name in ['true','orchestra','independent']}
        for a,b in [(1,1),(1,-1),(-1,1),(-1,-1)]:
            plan=np.zeros((12,env.n),int);plan[:,0]=a;plan[:,1]=b
            truth,components=true_rollout(env,plan);truecost=float((components@cfg['objective_weights']).sum())
            values['true'].append(truecost)
            for name,state in inferred.items():
                prediction=models[name].imagine(state,torch.tensor(plan,device=state.memory.device)[None])
                predicted=prediction['agents'][0].cpu().numpy();cost=float((prediction['components'][0].cpu().numpy()@cfg['objective_weights']).sum())
                values[name].append(cost)
                rows.append({'seed':seed,'model':name,'plan':f'{a:+d},{b:+d}','true_cost':truecost,'predicted_cost':cost,
                  'true_conflicts':float(components[:,4].sum()),'predicted_conflicts':float(prediction['components'][0,:,4].sum().cpu()),
                  **errors(predicted,truth)})
        contrast=lambda v:v[0]-v[1]-v[2]+v[3]
        for name in ['orchestra','independent']:
            contrasts.append({'seed':seed,'model':name,'true_interaction':contrast(values['true']),
                 'predicted_interaction':contrast(values[name]),'interaction_error':abs(contrast(values[name])-contrast(values['true']))})
    pd.DataFrame(rows).to_csv(out/'interactions.csv',index=False)
    pd.DataFrame(contrasts).to_csv(out/'interaction_contrasts.csv',index=False)
    return pd.DataFrame(rows)
