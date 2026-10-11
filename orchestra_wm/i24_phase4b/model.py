"""Isolated corrections demonstrated by the audit, with unchanged layer sizes.

Not a newly trained model. Original checkpoints and historical implementation
remain untouched. Macro-only inference uses no microscopic state or roster-sized
random draw, and its field transition uses the learned autonomous prior.
"""
from dataclasses import replace
import torch
from torch.nn import functional as F
from orchestra_wm.i24_continuous.model import ContinuousWorldModel, temporal_fields


def bounded_temporal_fields(history,road,dt=.2):
    low=min(road.lane_centers)-road.lane_width/2
    high=max(road.lane_centers)+road.lane_width/2
    masked=tuple((a,v&(a[:,:,1]>=low)&(a[:,:,1]<high)) for a,v in history)
    return temporal_fields(masked,road,dt)


class AuditedWorldModel(ContinuousWorldModel):
    def _transition(self,state,generator=None,posterior_target=None):
        if posterior_target is not None:
            raise ValueError('Phase4B prior-only objective: future-conditioned transition forbidden')
        if self.variant!='macro_only':
            result,p=super()._transition(state,generator,None)
            derived=bounded_temporal_fields(result.exposure_history,state.road,self.dt)
            p['derived_fields']=derived
            if self.variant in ('independent','micro_only'):
                p['fields']=derived;p['field_mean']=derived;result=replace(result,fields=derived)
            return result,p
        context=state.field_memory.mean(1)
        mean,logstd=self.prior(context).chunk(2,-1)
        z=mean+logstd.clamp(-4,1).exp()*torch.randn(mean.shape,generator=generator,device=mean.device)
        centers=torch.as_tensor(state.road.tokens(),device=state.fields.device)/self.scale[:2]
        own=self.field_encoder(torch.cat([state.fields/self.field_scale,torch.ones_like(state.fields[:,:,:1]),centers[None].expand(len(state.fields),-1,-1)],-1))
        inp=torch.cat([state.field_memory+own,torch.zeros_like(state.field_memory),z[:,None].expand(-1,state.field_memory.shape[1],-1)],-1)
        memory=self.field_transition(inp.reshape(-1,inp.shape[-1]),state.field_memory.reshape(-1,self.dim)).reshape_as(state.field_memory)
        raw=self.field_decoder(memory);field_mean=F.softplus(raw[:,:,:3])*self.field_scale*.25
        std=(F.softplus(raw[:,:,3:])*.03+.002)*self.field_scale
        fields=field_mean+std*torch.randn(std.shape,generator=generator,device=std.device)
        # Microscopic outputs are explicitly inapplicable to this variant.
        result=replace(state,fields=fields,field_memory=memory,valid=torch.zeros_like(state.valid))
        return result,dict(agents=state.agents,fields=fields,field_mean=field_mean,field_std=std,
                           derived_fields=torch.zeros_like(fields),valid=result.valid,kl=fields.new_zeros(()))


def matched_macro_energy(samples,target,support,scales):
    """Same proper macro objective for full and macro-only, no micro penalty.

    Distinct from a joint micro/macro experiment: use as an information-value
    control, not as evidence of well-trained microscopic motion. Two or more
    independently sampled prior rollouts are required. Target support only scores
    predictions; it never enters the inference state.
    """
    if samples.shape[0]<2:raise ValueError('At least two prior samples required')
    if not support.any():raise ValueError('No supported target fields')
    x=(samples[:,support]/scales).flatten(1);y=(target[support]/scales).flatten()
    n=len(x);norm=y.numel()**.5
    return torch.linalg.vector_norm(x-y,dim=-1).mean()/norm-.5*torch.cdist(x,x).sum()/(n*(n-1)*norm)
