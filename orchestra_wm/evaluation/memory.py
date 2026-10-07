import numpy as np
import pandas as pd
import torch
from orchestra_wm.evaluation.common import observe,errors
from orchestra_wm.envs.traffic_env import TrafficEnv

@torch.no_grad()
def evaluate_memory(cfg,models,out):
    rows=[];examples={}
    for seed in cfg['seeds']:
        local={**cfg,'sensor_regime':'full','miss_probability':0,'detection_noise':0}
        env=TrafficEnv(local,'intersection',20000+seed);obs,_=env.reset(seed=20000+seed)
        target=cfg['connected'] # background vehicle, observed before forced camera loss
        states={name:m.initial_state(1,env.n,len(env.road.lanes)) for name,m in models.items() if name in ['orchestra','memoryless']}
        previous=None
        for t in range(13):
            masked={k:v.copy() for k,v in obs.items()}
            if 4<=t<12:
                masked['mask'][target]=False;masked['agents'][target,:6]=0;masked['agents'][target,7]=0;masked['agents'][target,9:]=0
            for name in states: states[name]=observe(models[name],states[name],masked,previous)
            if 4<=t<12:
                action=np.ones(env.n,int);truthenv=env.clone();truthenv.step(action);truth=truthenv.state_array()[target:target+1]
                for name in ['orchestra','memoryless','reset','current_observation']:
                    if name=='current_observation':
                        predicted=masked['agents'][target:target+1,:6].copy();predicted[:,:2]+=predicted[:,3:5]*predicted[:,2:3]*(15*cfg['dt']/100);conflict=np.nan
                    else:
                        m=models['memoryless'] if name=='memoryless' else models['orchestra']
                        s=states[name] if name in states else observe(m,m.initial_state(1,env.n,len(env.road.lanes)),masked,None)
                        p=m.imagine(s,torch.tensor(action,device=s.memory.device)[None,None])
                        predicted=p['agents'][0,0,target:target+1].cpu().numpy()
                        conflict=float(abs(p['components'][0,0,4].cpu()-truthenv.last_components[4]))
                    rows.append({'seed':seed,'model':name,'occlusion_steps':t-3,'reappearance':t==11,
                         'conflict_error':conflict,**errors(predicted,truth)})
            previous=np.ones(env.n,int);obs,*_=env.step(previous)
    frame=pd.DataFrame(rows);frame.to_csv(out/'memory.csv',index=False);return frame
