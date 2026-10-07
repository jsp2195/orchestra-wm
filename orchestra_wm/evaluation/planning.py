import time
import numpy as np
import pandas as pd
from orchestra_wm.envs.traffic_env import TrafficEnv
from orchestra_wm.data.collection import policy
from orchestra_wm.evaluation.common import observe
from orchestra_wm.planning.orchestrator import Orchestrator
from orchestra_wm.planning.oracle import oracle_plan


def run_controller(cfg,family,seed,name,model=None):
    env=TrafficEnv(cfg,family,seed+40000);obs,_=env.reset(seed=seed+40000)
    rng=np.random.default_rng(seed);components=[];latencies=[];collisions=conflicts=0;selected_actions=[]
    if model:
        state=model.initial_state(1,env.n,len(env.road.lanes));planner=Orchestrator(model,cfg,seed)
    previous=None
    for t in range(cfg['planning_steps']):
        started=time.perf_counter()
        if model:
            if cfg.get('withhold_lanes'): obs['lanes'][:,:4]=0
            state=observe(model,state,obs,previous,env.state_array() if model.variant=='privileged' else None)
            plan,_,_=planner.plan(state);action=plan[0]
        elif name=='oracle': action=oracle_plan(env,cfg,rng)[0][0]
        else: action=policy(env,'noop' if name in ['persistence','constant_velocity'] else name,rng,previous)
        latencies.append(time.perf_counter()-started)
        selected_actions.extend(np.asarray(action)[obs['control_mask']].tolist())
        obs,_,done,_,info=env.step(action);previous=action
        components.append(info['components']);collisions+=info['collisions'];conflicts+=info['conflicts']
        if done: break
    c=np.array(components)
    return {'seed':seed,'scenario':family,'controller':name,'objective':float((c@cfg['objective_weights']).sum()),
         'delay':float(c[:,0].mean()),'queue':float(c[:,1].mean()),'progress':float(c[:,2].sum()),
         'collisions':collisions,'conflicts':conflicts,'collision_rate':collisions/len(c),'conflict_rate':conflicts/len(c),
         'proceed_fraction':float(np.mean(np.array(selected_actions)==1)),
         'yield_fraction':float(np.mean(np.array(selected_actions)==-1)),
         'completed':env.completed,'completion_fraction':env.completed/env.n,
         'success':int(env.completed==env.n and collisions==0),'steps':len(c),'latency_ms':1000*float(np.mean(latencies))}


def evaluate_planning(cfg,models,out):
    rows=[]
    # Oracle first, so feasibility is established independently of learned prediction.
    controllers=['oracle','random','noop','independent','heuristic','orchestra','privileged','independent_model','no_actions','persistence','constant_velocity']
    for name in controllers:
        key='independent' if name=='independent_model' else name
        model=models.get(key) if name in ['orchestra','privileged','independent_model','no_actions'] else None
        for seed in cfg['seeds']:
            for family in cfg['scenarios']:
                rows.append(run_controller(cfg,family,seed,name,model))
        print('Planning evaluated:',name,flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(out/'planning.csv',index=False)
    return frame
