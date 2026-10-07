import numpy as np
import torch
from orchestra_wm.planning.categorical_cem import categorical_cem

class Orchestrator:
    """Capability boundary: planner holds model + config, never an environment."""
    def __init__(self,model,cfg,seed=0):
        self.model,self.cfg=model,cfg
        self.rng=np.random.default_rng(seed)
    @torch.no_grad()
    def score(self,state,plans):
        device=state.memory.device
        prediction=self.model.imagine(state.repeat(len(plans)),torch.as_tensor(plans,device=device))
        weights=torch.tensor(self.cfg['objective_weights'],device=device)
        return (prediction['components']@weights).sum(1).cpu().numpy()
    def plan(self,state):
        return categorical_cem(lambda p:self.score(state,p),self.cfg['plan_horizon'],state.agents.shape[1],
            self.cfg['candidates'],self.cfg['cem_iterations'],self.rng,state.control[0].cpu().numpy()>.5)
