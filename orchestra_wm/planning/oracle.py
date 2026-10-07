"""Diagnostic simulator planner. Kept outside the learned planner module."""
import numpy as np
from orchestra_wm.planning.categorical_cem import categorical_cem

def true_rollout(env,plan):
    world=env.clone();states=[];components=[]
    for action in plan:
        _,_,_,_,info=world.step(action)
        states.append(info['state']);components.append(info['components'])
    return np.array(states),np.array(components)

def oracle_plan(env,cfg,rng):
    weights=np.array(cfg['objective_weights'])
    return categorical_cem(lambda plans:np.array([(true_rollout(env,p)[1]@weights).sum() for p in plans]),
        cfg['plan_horizon'],env.n,cfg['candidates'],cfg['cem_iterations'],rng,
        np.array([v.connected for v in env.vehicles]))
