from dataclasses import replace
import torch
from torch import nn
from orchestra_wm.models.world_model import WorldModel

class PairwiseWorldModel(WorldModel):
    """Optional additive message channel, using only inferred relative geometry/actions."""
    def __init__(self,dim=128,layers=2,dt=.5):
        super().__init__(dim,layers,dt,'orchestra')
        self.pair_gate=nn.Sequential(nn.Linear(6,32),nn.GELU(),nn.Linear(32,1),nn.Sigmoid())
        self.sender=nn.Linear(2,dim)
    def imagine_step(self,state,joint_action):
        agents=state.agents;delta=agents[:,None,:,:2]-agents[:,:,None,:2]
        relative_speed=(agents[:,None,:,2]-agents[:,:,None,2]).unsqueeze(-1)
        heading=(agents[:,None,:,3:5]*agents[:,:,None,3:5]).sum(-1,keepdim=True)
        sender_action=(joint_action*state.control)[:,None,:,None].expand(-1,agents.shape[1],-1,-1)
        control=state.control[:,None,:,None].expand_as(sender_action)
        geometry=torch.cat([delta,relative_speed,heading,sender_action,control],-1)
        distance=torch.linalg.vector_norm(delta,dim=-1)
        gate=self.pair_gate(geometry).squeeze(-1)*torch.exp(-distance/.25)
        mask=state.valid[:,None,:]&state.valid[:,:,None]&~torch.eye(agents.shape[1],device=agents.device,dtype=torch.bool)[None]
        gate=gate*mask
        values=self.sender(torch.stack([joint_action*state.control,agents[:,:,2]],-1).float())
        message=torch.einsum('bij,bjd->bid',gate,values)/mask.sum(-1,keepdim=True).clamp_min(1)
        return super().imagine_step(replace(state,memory=state.memory+.1*message),joint_action)

def create_model(cfg,name):
    if name=='D_pairwise':return PairwiseWorldModel(cfg['model_dim'],cfg['layers'],cfg['dt'])
    variant={'independent':'independent','privileged':'privileged'}.get(name,'orchestra')
    return WorldModel(cfg['model_dim'],cfg['layers'],cfg['dt'],variant)

def load_phase2(path):
    data=torch.load(path,map_location='cpu',weights_only=False)
    m=create_model(data['config'],data['name']);m.load_state_dict(data['model']);m.eval();return m
