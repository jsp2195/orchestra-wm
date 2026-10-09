"""Compact graph variational state-space model. No action or simulator pathway."""
from dataclasses import dataclass,replace
import torch
from torch import nn
import torch.nn.functional as F
from orchestra_wm.i24.schema import RoadMap,KnownFutureContext

SCALE=torch.tensor([800.,20.,30.,5.,8.,5.,8.,3.])
FIELD_SCALE=torch.tensor([30.,100.,3000.])

@dataclass
class Belief:
    agents: torch.Tensor
    memory: torch.Tensor
    fields: torch.Tensor
    field_memory: torch.Tensor
    valid: torch.Tensor
    seen: torch.Tensor
    age: torch.Tensor
    road: RoadMap
    track_ids: tuple[str,...]
    def repeat(self,n):
        return replace(self,**{k:getattr(self,k).repeat_interleave(n,dim=0) for k in ['agents','memory','fields','field_memory','valid','seen','age']})


def soft_fields(agents,valid,road):
    centers=torch.as_tensor(road.tokens(),device=agents.device,dtype=agents.dtype)
    delta=agents[:,:,:2,None]-centers.T[None,None]
    bandwidth=torch.tensor([(road.s_max-road.s_min)/road.bins*.5,road.lane_width*.35],device=agents.device)
    weights=torch.exp(-.5*((delta/bandwidth[None,None,:,None])**2).sum(2))*valid[:,:,None]
    # Normalize each vehicle's interior contribution; smooth binning enables consistency gradients.
    inside=(agents[:,:,0]>=road.s_min)&(agents[:,:,0]<road.s_max)
    weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-6)*inside[:,:,None]
    count=weights.sum(1);speed=(weights*agents[:,:,2,None]).sum(1)/count.clamp_min(1e-5)
    density=count/((road.s_max-road.s_min)/road.bins)*1000
    return torch.stack([speed,density,density*speed*3.6],-1)

class RoadsideWorldModel(nn.Module):
    def __init__(self,dim=64,latent=8,variant='full',dt=.2):
        super().__init__();self.dim,self.latent,self.variant,self.dt=dim,latent,variant,dt
        self.register_buffer('scale',SCALE.clone());self.register_buffer('field_scale',FIELD_SCALE.clone())
        self.agent_encoder=nn.Sequential(nn.Linear(11,dim),nn.SiLU(),nn.Linear(dim,dim))
        self.field_encoder=nn.Sequential(nn.Linear(6,dim),nn.SiLU(),nn.Linear(dim,dim))
        self.agent_gru=nn.GRUCell(dim,dim);self.field_gru=nn.GRUCell(dim,dim)
        self.edge=nn.Sequential(nn.Linear(6,dim),nn.SiLU(),nn.Linear(dim,dim))
        self.agent_transition=nn.GRUCell(dim*2+latent,dim)
        self.field_transition=nn.GRUCell(dim*2+latent,dim)
        self.prior=nn.Linear(dim,latent*2);self.posterior=nn.Sequential(nn.Linear(dim+4,dim),nn.SiLU(),nn.Linear(dim,latent*2))
        self.acceleration=nn.Linear(dim,4);self.field_decoder=nn.Linear(dim,6)
    def initial_state(self,scene_metadata):
        road=scene_metadata['road'];ids=tuple(scene_metadata['track_ids']);b=scene_metadata.get('batch',1);n=len(ids);device=self.scale.device
        z=lambda *shape:torch.zeros(*shape,device=device)
        a=z(b,n,8);a[:,:,6:]=torch.tensor([4.8,1.9],device=device)
        return Belief(a,z(b,n,self.dim),z(b,road.field_count,3),z(b,road.field_count,self.dim),torch.ones(b,n,device=device,dtype=torch.bool),torch.zeros(b,n,device=device,dtype=torch.bool),z(b,n),road,ids)
    def observe(self,state,observation,observation_mask,dt):
        if not isinstance(observation,dict) or set(observation)-{'values','confidence','fields','field_support'}:raise ValueError('Observation accepts past measured inputs only')
        x=observation['values'];mask=observation_mask.bool()&state.valid;confidence=observation['confidence'];age=state.age+dt
        pred=state.agents.clone();pred[:,:,:2]=pred[:,:,:2]+pred[:,:,2:4]*dt
        agents=torch.where(mask[:,:,None],x,pred)
        age=torch.where(mask,torch.zeros_like(age),age)
        inp=torch.cat([agents/self.scale,mask[:,:,None],confidence[:,:,None],age[:,:,None]/5],-1)
        encoded=self.agent_encoder(inp);memory=self.agent_gru(encoded.reshape(-1,self.dim),state.memory.reshape(-1,self.dim)).reshape_as(state.memory)
        # Invisible established entities keep inferred motion and recurrent belief; padded slots remain inert.
        memory=memory*state.valid[:,:,None];seen=state.seen|mask
        f=observation['fields'];support=observation['field_support'];centers=torch.as_tensor(state.road.tokens(),device=x.device)/self.scale[:2]
        fi=torch.cat([f/self.field_scale,support[:,:,None],centers[None].expand(len(x),-1,-1)],-1)
        fm=self.field_gru(self.field_encoder(fi).reshape(-1,self.dim),state.field_memory.reshape(-1,self.dim)).reshape_as(state.field_memory)
        return replace(state,agents=agents,memory=memory,fields=f,field_memory=fm,seen=seen,age=age)
    def _geometry(self,state):
        a=state.agents;delta=a[:,None,:,:2]-a[:,:,None,:2];dv=a[:,None,:,2:4]-a[:,:,None,2:4]
        dist=torch.linalg.vector_norm(delta,dim=-1);lane=(delta[...,1].abs()<state.road.lane_width*.5)
        edge=torch.cat([delta/torch.tensor([60.,4.],device=a.device),dv/10,lane[...,None],(delta[...,0]>0)[...,None]],-1)
        valid=state.valid&state.seen;mask=valid[:,None,:]&valid[:,:,None]&~torch.eye(a.shape[1],dtype=torch.bool,device=a.device)[None]
        weight=torch.exp(-dist/40)*mask*(dist<100)
        message=(self.edge(edge)*weight[:,:,:,None]).sum(2)/weight.sum(2).clamp_min(1)[:,:,None]
        return message
    def _transition(self,state,generator=None,posterior_target=None):
        valid=state.valid&state.seen;denom=valid.sum(1,keepdim=True).clamp_min(1)
        micro=(state.memory*valid[:,:,None]).sum(1)/denom;macro=state.field_memory.mean(1)
        context=macro if self.variant=='macro_only' else micro if self.variant in ['independent','micro_only','no_crossscale'] else (micro+macro)/2
        prior_mean,prior_logstd=self.prior(context).chunk(2,-1);prior_logstd=prior_logstd.clamp(-4,1)
        mean,logstd=prior_mean,prior_logstd;kl=mean.new_zeros(())
        if posterior_target is not None and self.variant!='deterministic':
            innovation=((posterior_target[:,:,:4]-state.agents[:,:,:4])*valid[:,:,None]).sum(1)/denom/self.scale[:4]
            mean,logstd=self.posterior(torch.cat([context,innovation],-1)).chunk(2,-1);logstd=logstd.clamp(-4,1)
            kl=(prior_logstd-logstd+(logstd.exp().square()+(mean-prior_mean).square())/(2*prior_logstd.exp().square())-.5).mean()
        noise=torch.randn(mean.shape,device=mean.device,generator=generator)
        z=mean+logstd.exp()*noise
        if self.variant=='deterministic':z=torch.zeros_like(z)
        message=self._geometry(state) if self.variant not in ['independent','macro_only','graph_off'] else torch.zeros_like(state.memory)
        if self.variant in ['full','deterministic','graph_off']:message=message+macro[:,None]
        za=z[:,None].expand(-1,state.agents.shape[1],-1)
        if self.variant=='independent':
            # No scene context/shared innovation in independent per-agent generation.
            za=torch.randn(za.shape,device=za.device,generator=generator)
        own=self.agent_encoder(torch.cat([state.agents/self.scale,valid[:,:,None],valid[:,:,None],state.age[:,:,None]/5],-1))
        inp=torch.cat([state.memory+own,message,za],-1)
        memory=self.agent_transition(inp.reshape(-1,inp.shape[-1]),state.memory.reshape(-1,self.dim)).reshape_as(state.memory)
        out=self.acceleration(memory);acc_mean=out[:,:,:2];acc_std=F.softplus(out[:,:,2:])*.3+.02
        acceleration=acc_mean
        if self.variant!='deterministic':acceleration=acceleration+acc_std*torch.randn(acc_mean.shape,device=acc_mean.device,generator=generator)
        agents=state.agents.clone();agents[:,:,:2]=agents[:,:,:2]+agents[:,:,2:4]*self.dt+.5*acceleration*self.dt**2;agents[:,:,2:4]=agents[:,:,2:4]+acceleration*self.dt;agents[:,:,4:6]=acceleration
        if self.variant=='macro_only':agents=state.agents.clone();agents[:,:,:2]+=agents[:,:,2:4]*self.dt
        feedback=micro if self.variant in ['full','deterministic','graph_off'] else torch.zeros_like(micro)
        field_z=z if self.variant not in ['no_crossscale','macro_only'] else torch.randn(z.shape,device=z.device,generator=generator)
        centers=torch.as_tensor(state.road.tokens(),device=state.fields.device)/self.scale[:2]
        field_own=self.field_encoder(torch.cat([state.fields/self.field_scale,torch.ones_like(state.fields[:,:,:1]),centers[None].expand(len(state.fields),-1,-1)],-1))
        fi=torch.cat([state.field_memory+field_own,feedback[:,None].expand_as(state.field_memory),field_z[:,None].expand(-1,state.field_memory.shape[1],-1)],-1)
        fm=self.field_transition(fi.reshape(-1,fi.shape[-1]),state.field_memory.reshape(-1,self.dim)).reshape_as(state.field_memory)
        raw=self.field_decoder(fm);field_mean=F.softplus(raw[:,:,:3])*self.field_scale*.25;field_std=(F.softplus(raw[:,:,3:])*.03+.002)*self.field_scale
        fields=field_mean
        if self.variant!='deterministic':fields=fields+field_std*torch.randn(fields.shape,device=fields.device,generator=generator)
        derived=soft_fields(agents,valid,state.road)
        if self.variant in ['independent','micro_only']:fields=derived;field_mean=derived
        # Boundary exits are predicted, not copied from withheld validity labels.
        alive=valid&(agents[:,:,0]>=state.road.s_min)&(agents[:,:,0]<state.road.s_max)
        next_state=replace(state,agents=agents,memory=memory*alive[:,:,None],fields=fields,field_memory=fm,valid=alive)
        return next_state,{'agents':agents,'fields':fields,'derived_fields':derived,'valid':alive,'acc_mean':acc_mean,'acc_std':acc_std,'field_mean':field_mean,'field_std':field_std,'kl':kl}
    def imagine_step(self,state,known_future_context=None,*,generator=None):
        if known_future_context is not None:
            if not isinstance(known_future_context,KnownFutureContext) or known_future_context.road!=state.road:raise ValueError('Only the same fixed map is known future context')
        return self._transition(state,generator)
    def imagine(self,state,horizon_steps,num_samples=8,rng_seed=0):
        generator=torch.Generator(device=self.scale.device).manual_seed(rng_seed);state=state.repeat(num_samples);rows=[]
        for _ in range(horizon_steps):state,p=self.imagine_step(state,generator=generator);rows.append(p)
        return {key:torch.stack([r[key] for r in rows],1) for key in ['agents','fields','derived_fields','valid']}
    def condition_on_actions(self,action):
        raise RuntimeError('Passive I-24 data does not identify action responses; interface disabled')


def observe_history(model,observations,road,reset=False,permutation=False):
    state=model.initial_state({'road':road,'track_ids':observations[0].track_ids});device=model.scale.device
    permutation_index=torch.arange(len(state.track_ids)-1,-1,-1,device=device)
    for obs in observations:
        if reset:state=model.initial_state({'road':road,'track_ids':obs.track_ids})
        values=torch.as_tensor(obs.values,dtype=torch.float32,device=device)[None]
        if permutation:
            values=values.clone();values[:,:,2:6]=values[:,permutation_index,2:6]
        batch={'values':values,'confidence':torch.as_tensor(obs.confidence,dtype=torch.float32,device=device)[None],'fields':torch.as_tensor(obs.fields,device=device)[None],'field_support':torch.as_tensor(obs.field_support,device=device)[None]}
        state=model.observe(state,batch,torch.as_tensor(obs.visible,device=device)[None],model.dt)
    return state
