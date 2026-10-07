"""Learned state is the only input to imagination; no simulator dependency."""
from dataclasses import dataclass
import torch
from torch import nn

@dataclass
class WorldState:
    memory: torch.Tensor
    agents: torch.Tensor
    lanes: torch.Tensor
    control: torch.Tensor
    valid: torch.Tensor

    def repeat(self,n):
        return WorldState(*(x.repeat_interleave(n,dim=0) for x in (self.memory,self.agents,self.lanes,self.control,self.valid)))

class WorldModel(nn.Module):
    def __init__(self,dim=128,layers=2,dt=.5,variant='orchestra'):
        super().__init__()
        self.dim,self.dt,self.variant=dim,dt,variant
        self.agent_token=nn.Linear(12,dim)
        self.lane_token=nn.Linear(8,dim)
        self.action_token=nn.Sequential(nn.Linear(4,dim),nn.GELU(),nn.Linear(dim,dim))
        self.encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(dim,4,dim*4,dropout=0,batch_first=True,norm_first=True),layers,enable_nested_tensor=False)
        self.dynamics=nn.TransformerEncoder(nn.TransformerEncoderLayer(dim,4,dim*4,dropout=0,batch_first=True,norm_first=True),layers,enable_nested_tensor=False)
        self.memory_cell=nn.GRUCell(dim,dim)
        self.decoder=nn.Sequential(nn.Linear(dim,dim),nn.GELU(),nn.Linear(dim,6))
        self.cost_head=nn.Sequential(nn.Linear(dim,dim),nn.GELU(),nn.Linear(dim,6))
        self.lane_head=nn.Linear(dim,2)
        self.independent=nn.Sequential(nn.Linear(dim,dim*2),nn.GELU(),nn.Linear(dim*2,dim))
        nn.init.zeros_(self.decoder[-1].weight);nn.init.zeros_(self.decoder[-1].bias)

    def initial_state(self,batch_size=1,num_agents=10,num_lanes=4,device=None):
        device=device or next(self.parameters()).device
        z=lambda *shape:torch.zeros(*shape,device=device)
        return WorldState(z(batch_size,num_agents,self.dim),z(batch_size,num_agents,6),
                          z(batch_size,num_lanes,self.dim),z(batch_size,num_agents),
                          torch.ones(batch_size,num_agents,dtype=torch.bool,device=device))

    def observe(self,state,observation_tokens,observation_mask,previous_action):
        obs=observation_tokens['agents'];lanes=observation_tokens['lanes']
        valid=observation_tokens.get('valid',state.valid).bool()
        mask=observation_mask.bool() & valid
        lane_valid=lanes.abs().sum(-1)>0
        if self.variant=='memoryless':
            state=self.initial_state(obs.shape[0],obs.shape[1],lanes.shape[1],obs.device)
        if previous_action is not None:
            state,_=self.imagine_step(state,previous_action)
        agent=self.agent_token(obs)
        lane=self.lane_token(lanes)
        if self.variant=='independent':
            encoded=self.independent(agent)
        else:
            all_tokens=torch.cat([agent,lane],1)
            padding=torch.cat([~mask,~lane_valid],1)
            encoded=self.encoder(all_tokens,src_key_padding_mask=padding)[:,:obs.shape[1]]
        memory=self.memory_cell(encoded.reshape(-1,self.dim),state.memory.reshape(-1,self.dim)).reshape_as(state.memory)
        memory=torch.where(mask[...,None],memory,state.memory)
        agents=torch.where(mask[...,None],obs[:,:,:6],state.agents)
        # Connectivity is observable metadata even when camera detections are missing.
        return WorldState(memory*valid[...,None],agents*valid[...,None],lane*lane_valid[...,None],obs[:,:,6]*valid,valid)

    def imagine_step(self,state,joint_action):
        action=joint_action.to(state.memory.dtype)*state.control
        if self.variant=='no_actions': action=torch.zeros_like(action)
        feature=torch.stack([action,state.control,state.agents[:,:,2],state.agents[:,:,5]],-1)
        tokens=state.memory+self.action_token(feature)
        if self.variant=='independent':
            hidden=self.independent(tokens)
            lanes=state.lanes
        else:
            padding=torch.cat([~state.valid,state.lanes.abs().sum(-1)==0],1)
            # At least one lane or entity token must exist in a valid scene.
            both=self.dynamics(torch.cat([tokens,state.lanes],1),src_key_padding_mask=padding)
            hidden=both[:,:tokens.shape[1]];lanes=both[:,tokens.shape[1]:]
            lanes=lanes*(state.lanes.abs().sum(-1)>0)[...,None]
        memory=self.memory_cell(hidden.reshape(-1,self.dim),state.memory.reshape(-1,self.dim)).reshape_as(hidden)
        raw=self.decoder(memory)
        prior=state.agents
        # A kinematic residual prior stabilizes training; all action response is learned.
        xy=prior[:,:,:2]+prior[:,:,3:5]*prior[:,:,2:3]*(15*self.dt/100)
        predicted=torch.cat([xy+torch.tanh(raw[:,:,:2])*.06,
             (prior[:,:,2:3]+torch.tanh(raw[:,:,2:3])*.2).clamp(0,1.3),
             (prior[:,:,3:5]+torch.tanh(raw[:,:,3:5])*.15).clamp(-1,1),
             (prior[:,:,5:6]+torch.tanh(raw[:,:,5:6])*.2).clamp(0,1)],-1)
        next_state=WorldState(memory,predicted,lanes,state.control,state.valid)
        weights=((prior[:,:,5:6]>.1)&state.valid[...,None]).to(memory.dtype)
        pooled=(memory*weights).sum(1)/weights.sum(1).clamp_min(1)
        components=torch.nn.functional.softplus(self.cost_head(pooled))
        return next_state,{'agents':predicted,'lanes':torch.nn.functional.softplus(self.lane_head(lanes)),
                           'components':components}

    def imagine(self,state,joint_action_sequence):
        predictions=[]
        for action in joint_action_sequence.unbind(1):
            state,prediction=self.imagine_step(state,action)
            predictions.append(prediction)
        return {k:torch.stack([p[k] for p in predictions],1) for k in predictions[0]}
