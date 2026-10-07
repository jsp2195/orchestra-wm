"""Categorical CEM independent of the simulator and model implementation."""
import numpy as np

def categorical_cem(score,horizon,num_agents,candidates=32,iterations=3,rng=None,control_mask=None):
    rng=rng or np.random.default_rng(0)
    probs=np.full((horizon,num_agents,3),1/3)
    best=None;best_cost=np.inf
    history=[]
    for iteration in range(iterations):
        u=rng.random((candidates,horizon,num_agents,1))
        plans=(u>np.cumsum(probs,axis=-1)[None]).sum(-1)-1
        plans[0]=0
        if candidates>2: plans[1]=1;plans[2]=-1
        if best is not None: plans[-1]=best
        if control_mask is not None: plans[:,:,~np.asarray(control_mask,bool)]=0
        costs=np.asarray(score(plans))
        if not np.isfinite(costs).all(): raise ValueError('Nonfinite planning costs')
        elites=plans[np.argsort(costs)[:max(2,candidates//4)]]
        empirical=np.stack([(elites==a).mean(0) for a in [-1,0,1]],-1)
        probs=.2/3+.8*empirical
        i=int(np.argmin(costs))
        if costs[i]<best_cost: best,best_cost=plans[i].copy(),float(costs[i])
        history.append(best_cost)
    return best,best_cost,history
