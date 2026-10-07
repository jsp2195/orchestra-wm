import numpy as np
import torch

class TrajectoryDataset:
    """Episode-level training split. Evaluation uses entirely new simulator seeds."""
    def __init__(self,path):
        self.data=dict(np.load(path,allow_pickle=False))
        self.n=len(self.data['obs'])
    def sample(self,batch,context,horizon,rng,device,validation=False):
        split=max(1,int(self.n*.875))
        episodes=rng.integers(split,self.n,size=batch) if validation else rng.integers(0,split,size=batch)
        starts=rng.integers(0,self.data['actions'].shape[1]-context-horizon+1,size=batch)
        result={}
        for key in ['obs','mask','lanes','control','states','lane_targets','actions','components']:
            length=context+horizon if key not in ['actions','components'] else context+horizon-1
            values=np.stack([self.data[key][e,s:s+length] for e,s in zip(episodes,starts)])
            result[key]=torch.as_tensor(values,device=device)
        return result
