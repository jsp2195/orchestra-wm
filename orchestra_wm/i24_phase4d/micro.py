"""Separate historical-cohort kinematic residual baseline, not a population census."""
import numpy as np
import torch
from torch import nn
from orchestra_wm.i24_continuous.data import observations

def prepare(scene,regime,seed):
    obs,cohort=observations(scene,25,regime,seed);hist=np.stack([o.values for o in obs]);visible=np.stack([o.visible for o in obs]);initial=hist[-1].copy()
    for j in range(len(cohort)):
        last=np.flatnonzero(visible[:,j])[-1];initial[j]=hist[last,j];initial[j,:2]+=initial[j,2:4]*(24-last)*.2
    lanes=np.array(scene.road.lane_centers);lane=np.argmin(abs(hist[...,1,None]-lanes),axis=-1)
    x=np.stack([hist[...,2]/30,hist[...,3]/5,hist[...,4]/5,hist[...,5]/5,hist[...,6]/10,hist[...,7]/3,(hist[...,1]-lanes[lane])/scene.road.lane_width,visible],-1).astype(np.float32);x[~visible]=0
    delta=abs(initial[:,None,:2]-initial[None,:,:2]);adj=((delta[...,0]<100)&(delta[...,1]<scene.road.lane_width*1.1)).astype(np.float32);np.fill_diagonal(adj,0);adj/=np.maximum(adj.sum(-1,keepdims=True),1)
    return {k:torch.from_numpy(v) for k,v in dict(history=x,visible=visible,initial=initial,adjacency=adj,target=scene.values[25:125,cohort],support=scene.existence[25:125,cohort]).items()}

class KinematicResidual(nn.Module):
    def __init__(self,dim=24):
        super().__init__();self.dim=dim;self.encoder=nn.Linear(8,dim);self.memory=nn.GRUCell(dim,dim);self.message=nn.Linear(dim,dim);self.head=nn.Sequential(nn.Linear(2*dim,dim),nn.Tanh(),nn.Linear(dim,4))
        nn.init.normal_(self.head[-1].weight,std=.001);nn.init.zeros_(self.head[-1].bias)
        with torch.no_grad():self.head[-1].bias[2:]=-2
    def forward(self,batch,variant='graph',samples=8,seed=0):
        hist,visible,initial=batch['history'],batch['visible'],batch['initial'];n=len(initial);h=hist.new_zeros((n,self.dim))
        for t in range(25):
            nxt=self.memory(torch.tanh(self.encoder(hist[t])),h);h=torch.where(visible[t,:,None],nxt,h)
        if variant not in ('graph','independent'):raise ValueError(variant)
        a=batch['adjacency'] if variant=='graph' else torch.eye(n,device=hist.device)
        raw=self.head(torch.cat([h,torch.tanh(self.message(a@h))],-1));mu=torch.tanh(raw[:,:2])*hist.new_tensor([3.,.5]);sigma=.005+.2*torch.nn.functional.softplus(raw[:,2:])
        g=torch.Generator(device=hist.device).manual_seed(seed);acc=mu[None]+sigma[None]*torch.randn((samples,n,2),generator=g,device=hist.device)
        t=torch.arange(1,101,device=hist.device,dtype=hist.dtype)*.2;t=t[None,:,None,None];tau=2.
        decay=torch.exp(-t/tau);velocity=initial[None,None,:,2:4]+acc[:,None]*tau*(1-decay)
        position=initial[None,None,:,:2]+initial[None,None,:,2:4]*t+acc[:,None]*tau*(t-tau*(1-decay))
        acceleration=acc[:,None]*decay;dims=initial[None,None,:,6:8].expand(samples,100,-1,-1)
        return torch.cat([position,velocity,acceleration,dims],-1)

def micro_energy(samples,target,support):
    scale=samples.new_tensor([100.,3.6576,30.,5.]);x=samples[...,:4]/scale;y=target[...,:4]/scale;mask=support[...,None];den=(support.sum(-1)*4).clamp_min(1)
    first=torch.sqrt(((x-y).square()*mask).sum((-1,-2))/den+1e-12).mean(0);pair=[]
    for i in range(len(x)):
        for j in range(i):pair.append(torch.sqrt(((x[i]-x[j]).square()*mask).sum((-1,-2))/den+1e-12))
    value=first-.5*torch.stack(pair).mean(0)
    return value[support.any(-1)].mean()
