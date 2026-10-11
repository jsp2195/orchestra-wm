"""Capacity-matched stochastic residual field dynamics on a fixed roadway graph.

Both controls execute every layer on the same 32 tokens. Full adds eight causal
vehicle-history statistics to an aggregate-only feature basis. Aggregate-only
executes the identical feature path without those additions; no dormant branch.
Individual vehicle generation is a separate observed-cohort experiment.
"""
import torch
from torch import nn

class ResidualFields(nn.Module):
    def __init__(self,dim=32):
        super().__init__();self.dim=dim
        self.encoder=nn.Linear(15,dim);self.message=nn.Linear(dim,dim)
        self.history=nn.GRUCell(2*dim,dim);self.feedback=nn.Linear(9,dim)
        self.transition=nn.GRUCell(2*dim,dim);self.head=nn.Linear(dim,4)
        self.loading=nn.Linear(dim,8)
        self.register_buffer('scales',torch.tensor([30.,100.,3000.]))
        nn.init.normal_(self.head.weight,std=.001);nn.init.zeros_(self.head.bias)
        with torch.no_grad():self.head.bias[2:]=-2.
        nn.init.normal_(self.loading.weight,std=.001);nn.init.zeros_(self.loading.bias)

    def forward(self,batch,variant='full',samples=4,seed=0,horizon=100,ablation=None):
        if variant not in ('full','aggregate'):raise ValueError(variant)
        x=batch['aggregate'];single=x.ndim==3
        if single:batch={k:v[None] for k,v in batch.items()};x=batch['aggregate']
        b,t,k,_=x.shape;x=x.clone()
        if variant=='full':x[...,7:]+=batch['micro']
        if ablation=='micro_removed' and variant=='full':x[...,7:]-=batch['micro']
        a=batch['adjacency']
        if ablation=='graph_off':a=torch.eye(k,device=x.device)[None].expand(b,-1,-1)
        h=x.new_zeros((b,k,self.dim))
        def advance(cell,own,h,adj):
            msg=torch.tanh(self.message(adj@h))
            return cell(torch.cat([own,msg],-1).reshape(-1,2*self.dim),h.reshape(-1,self.dim)).reshape_as(h)
        for i in range(t):
            if ablation=='memoryless':h=torch.zeros_like(h)
            h=advance(self.history,torch.tanh(self.encoder(x[:,i])),h,a)
        h=h[None].expand(samples,-1,-1,-1).reshape(samples*b,k,self.dim)
        a=a[None].expand(samples,-1,-1,-1).reshape(samples*b,k,k)
        reference=batch['reference'][...,:2].clamp_min(.05)
        reference=reference[None].expand(samples,-1,-1,-1).reshape(samples*b,k,2)
        pos=batch['positions'][None].expand(samples,-1,-1,-1).reshape(samples*b,k,2)
        initial=torch.cat([reference,3.6*reference[...,:1]*reference[...,1:2]],-1)
        current=initial;logratio=torch.zeros_like(reference);result=[]
        generator=torch.Generator(device=x.device).manual_seed(seed)
        for step in range(horizon):
            clock=x.new_full((samples*b,k,1),(step+1)/100)
            own=torch.tanh(self.feedback(torch.cat([current/self.scales,initial/self.scales,pos,clock],-1)))
            h=advance(self.transition,own,h,a)
            raw=self.head(h);drift=.15*torch.tanh(raw[...,:2]);sigma=.002+.25*torch.sigmoid(raw[...,2:])
            load=.10*torch.tanh(self.loading(h).reshape(samples*b,k,2,4))
            local=torch.randn(sigma.shape,generator=generator,device=x.device)
            latent=torch.randn((samples*b,1,1,4),generator=generator,device=x.device)
            innovation=sigma*local+(load*latent).sum(-1)
            variance=sigma.square()+load.square().sum(-1)
            if ablation=='latent_off':
                innovation=sigma*local;variance=sigma.square()
            if ablation=='deterministic':innovation=torch.zeros_like(innovation);variance=torch.zeros_like(variance)
            logratio=logratio+.2*drift-.1*variance+(0.2**.5)*innovation
            speed_density=reference*torch.exp(logratio)
            current=torch.cat([speed_density,3.6*speed_density[...,:1]*speed_density[...,1:2]],-1)
            result.append(current.reshape(samples,b,k,3))
        output=torch.stack(result,2)
        return output[:,0] if single else output

def energy(samples,target,support):
    """Unbiased scene/time joint energy, same physical scales as preregistration."""
    scales=samples.new_tensor([30.,100.,3000.]);mask=support[...,None]
    x=samples/scales;y=target/scales
    denominator=(support.sum(-1)*3).clamp_min(1)
    first=torch.sqrt(((x-y[None]).square()*mask).sum((-1,-2))/denominator+1e-12).mean(0)
    n=len(x);pair=[]
    for i in range(n):
        for j in range(i):pair.append(torch.sqrt(((x[i]-x[j]).square()*mask).sum((-1,-2))/denominator+1e-12))
    score=first-(.5*torch.stack(pair).mean(0) if pair else 0)
    return score[support.any(-1)].mean()
