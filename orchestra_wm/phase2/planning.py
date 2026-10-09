"""Balanced Phase-2 CEM. Learned scores have no simulator capability."""
import numpy as np
import torch
from orchestra_wm.phase2.scenarios import PAIRS


def initial_candidates(horizon,num_agents):
    plans=np.zeros((18,horizon,num_agents),dtype=np.int64)
    for k,(a,b) in enumerate(PAIRS):
        plans[k,:,0]=a;plans[k,:,1]=b
        plans[k+9,:,0]=a;plans[k+9,:,1]=b
        plans[k+9,horizon//2:,:2]*=-1
    # Replace the duplicate all-maintain sequence with a staggered exchange.
    plans[13,:horizon//2,:2]=[0,0];plans[13,horizon//2:,:2]=[1,-1]
    return plans


def balanced_cem(score,horizon,num_agents,iterations,rng,control):
    plans=initial_candidates(horizon,num_agents);best=None;best_cost=np.inf;history=[]
    for iteration in range(iterations):
        plans[:,:,~np.asarray(control,bool)]=0
        costs=np.asarray(score(plans));assert np.isfinite(costs).all()
        index=int(np.argmin(costs))
        if costs[index]<best_cost:best=plans[index].copy();best_cost=float(costs[index])
        history.append(best_cost)
        elite=plans[np.argsort(costs)[:4]]
        probs=.2/3+.8*np.stack([(elite==a).mean(0) for a in [-1,0,1]],-1)
        plans=(rng.random((18,horizon,num_agents,1))>np.cumsum(probs,axis=-1)[None]).sum(-1)-1
        plans[-1]=best
    return best,best_cost,history

class JointPlanner:
    def __init__(self,model,cfg,seed):self.model,self.cfg,self.rng=model,cfg,np.random.default_rng(seed)
    @torch.no_grad()
    def score(self,state,plans):
        pred=self.model.imagine(state.repeat(len(plans)),torch.as_tensor(plans,device=state.memory.device))
        weights=torch.as_tensor(self.cfg['objective_weights'],device=state.memory.device)
        return (pred['components']@weights).sum(1).cpu().numpy()
    def plan(self,state):
        assert self.cfg['candidates']==18
        return balanced_cem(lambda p:self.score(state,p),self.cfg['plan_horizon'],state.agents.shape[1],
                            self.cfg['cem_iterations'],self.rng,state.control[0].cpu().numpy()>.5)
