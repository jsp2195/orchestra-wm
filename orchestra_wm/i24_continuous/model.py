"""Preserved passive model with causal temporal exposure consistency fields."""
from dataclasses import dataclass,replace
import torch
from orchestra_wm.i24.model import Belief,RoadsideWorldModel


@dataclass
class ContinuousBelief(Belief):
    exposure_history: tuple=()

    def repeat(self,n):
        state=super().repeat(n)
        return replace(state,exposure_history=tuple((a.repeat_interleave(n,0),v.repeat_interleave(n,0)) for a,v in self.exposure_history))


def temporal_fields(history,road,dt=.2):
    """Soft spatial assignment, trailing 1s vehicle-time/distance exposure."""
    a=history[-1][0];centres=torch.as_tensor(road.tokens(),device=a.device,dtype=a.dtype)
    bandwidth=a.new_tensor([(road.s_max-road.s_min)/road.bins*.5,road.lane_width*.35])
    length=(road.s_max-road.s_min)/road.bins
    exposure=a.new_zeros((len(a),road.field_count));metres=torch.zeros_like(exposure)
    for agents,valid in history:
        delta=agents[:,:,:2,None]-centres.T[None,None]
        weights=torch.exp(-.5*((delta/bandwidth[None,None,:,None])**2).sum(2))*valid[:,:,None]
        inside=(agents[:,:,0]>=road.s_min)&(agents[:,:,0]<road.s_max)
        weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-6)*inside[:,:,None]
        exposure+=weights.sum(1)*dt
        metres+=(weights*agents[:,:,2,None]).sum(1)*dt
    span=len(history)*dt
    return torch.stack([metres/exposure.clamp_min(1e-6),exposure/(length*span)*1000,metres/(length*span)*3600],-1)


class ContinuousWorldModel(RoadsideWorldModel):
    def initial_state(self,scene_metadata):
        base=super().initial_state(scene_metadata)
        return ContinuousBelief(**base.__dict__)

    def observe(self,state,observation,observation_mask,dt):
        result=super().observe(state,observation,observation_mask,dt)
        history=(state.exposure_history+((observation['values'],observation_mask.bool()&state.valid),))[-round(1/self.dt):]
        return replace(result,exposure_history=history)

    def _transition(self,state,generator=None,posterior_target=None):
        result,prediction=super()._transition(state,generator,posterior_target)
        history=(state.exposure_history+((result.agents,result.valid),))[-round(1/self.dt):]
        derived=temporal_fields(history,state.road,self.dt)
        prediction['derived_fields']=derived
        if self.variant in ('independent','micro_only'):
            prediction['fields']=derived;prediction['field_mean']=derived
            result=replace(result,fields=derived)
        return replace(result,exposure_history=history),prediction
