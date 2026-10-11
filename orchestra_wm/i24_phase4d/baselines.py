"""A two-parameter probabilistic persistence control; training-only fitting."""
import torch
from torch import nn

class DiffusionPersistence(nn.Module):
    def __init__(self):
        super().__init__();self.log_sigma=nn.Parameter(torch.full((2,),-3.5))
    def forward(self,batch,samples=8,seed=0):
        ref=batch['reference'][...,:2].clamp_min(.05)
        single=ref.ndim==2
        if single:ref=ref[None]
        sigma=self.log_sigma.exp();g=torch.Generator().manual_seed(seed)
        increments=torch.randn((samples,len(ref),100,ref.shape[1],2),generator=g)*sigma*(.2**.5)-.1*sigma.square()
        sd=ref[None,:,None]*torch.exp(increments.cumsum(2))
        x=torch.cat([sd,3.6*sd[...,:1]*sd[...,1:2]],-1)
        return x[:,0] if single else x
