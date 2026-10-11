"""Population-aware joint objective, distinct from the matched macro experiment.

The microscopic state is a historically observed identity cohort. The learned
macro decoder forecasts the entire retained reconstructed population. They are
different targets: no equality/conservation loss between these outputs is valid.
Future entrants are supervised aggregate residuals, never inference identities.
This formulation does not yet generate individual arrivals.
"""
import numpy as np
import torch
from dataclasses import replace
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from orchestra_wm.i24.model import observe_history
from orchestra_wm.i24_continuous.data import observations, exposure_fields
from orchestra_wm.i24_phase4b.model import AuditedWorldModel, matched_macro_energy

MICRO_SCALES = (100., 3.6576, 30., 5.) # longitudinal/lateral position and velocity


class JointWorldModel(AuditedWorldModel):
    """Same graph/GRU layers, with a physically constrained *total-field* head.

    Positive speed/density samples imply q=3.6kv in the existing physical units.
    This constitutive identity is valid within one population/exposure target;
    it does not equate the observed cohort and the total population. Microscopic
    trajectories remain unclipped and their violations must still be measured.
    Two old decoder rows (independent flow mean/std) are intentionally inactive.
    """
    def _transition(self,state,generator=None,posterior_target=None):
        if self.variant not in ('full','independent','graph_off'):
            raise ValueError('Joint trial supports full, independent and graph-off')
        result,prediction=super()._transition(state,generator,posterior_target)
        raw=self.field_decoder(result.field_memory)
        means=F.softplus(raw[...,:2])*self.field_scale[:2]*.25
        sigma=F.softplus(raw[...,3:5])*.10+.02
        noise=torch.randn(means.shape,device=means.device,generator=generator)
        sampled=means*torch.exp(sigma*noise-.5*sigma.square())
        fields=torch.cat([sampled,3.6*sampled[...,:1]*sampled[...,1:2]],-1)
        field_mean=torch.cat([means,3.6*means[...,:1]*means[...,1:2]],-1)
        prediction['fields']=fields;prediction['field_mean']=field_mean
        prediction['field_std']=torch.cat([means*torch.sqrt(torch.expm1(sigma.square())),
                                          field_mean[...,2:3]*torch.sqrt(torch.expm1(sigma.square().sum(-1,keepdim=True)))],-1)
        return replace(result,fields=fields),prediction


def population_targets(scene, cohort):
    """Offline scoring/training labels only; additive vehicle-time/distance.

    Speed is an intensive quantity and must never be subtracted to define the
    residual. Density and flow are additive under identical exposure/binning.
    Residual means absent from the observed identity cohort, not verified inflow:
    it also includes missed historical vehicles and reconstruction fragmentation.
    """
    keep = np.zeros(len(scene.track_ids), bool); keep[cohort] = True
    total, support = exposure_fields(scene.values, scene.existence, scene.road)
    known, known_support = exposure_fields(scene.values[:,keep], scene.existence[:,keep], scene.road)
    remainder, residual_support = exposure_fields(scene.values[:,~keep], scene.existence[:,~keep], scene.road)
    if not np.allclose(total[...,1:], known[...,1:]+remainder[...,1:], atol=2e-3, rtol=2e-5):
        raise ValueError('Population exposure partition is not additive')
    return dict(total=total, support=support, cohort=known, cohort_support=known_support,
                residual=remainder, residual_support=residual_support)


def joint_objective(model, scene, fields, support, regime, seed, horizon=100,
                    checkpointed=True, return_diagnostics=False):
    """Equal-weight normalized prior scores; no pseudo-acceleration likelihood.

    Macroscopic targets retain the complete *available reconstructed* population;
    microscopic targets retain only identities observed before cutoff. Targets
    and their validity masks are consumed after the autonomous transition only.
    Empty/missing micro targets cannot be treated as zero-position observations.
    Fixed physical scales and weights are frozen before follow-on training.
    """
    obs, cohort = observations(scene,25,regime,seed)
    state = observe_history(model,obs,scene.road).repeat(2)
    target_macro = torch.as_tensor(fields[25:25+horizon],dtype=torch.float32)
    macro_support = torch.as_tensor(support[25:25+horizon],dtype=torch.bool)
    target_micro = torch.as_tensor(scene.values[25:25+horizon,cohort,:4],dtype=torch.float32)
    micro_support = torch.as_tensor(scene.existence[25:25+horizon,cohort],dtype=torch.bool)
    scales = model.scale.new_tensor(MICRO_SCALES)
    macro_losses, micro_losses = [], []
    for t in range(horizon):
        def transition(current,step_seed=seed+104729*t):
            generator = torch.Generator(device=model.scale.device).manual_seed(step_seed)
            nxt, p = model.imagine_step(current,generator=generator)
            return nxt, p['agents'][...,:4], p['fields']
        if checkpointed and torch.is_grad_enabled():
            state, agents, macro = checkpoint(transition,state,use_reentrant=False,preserve_rng_state=False)
        else:
            state, agents, macro = transition(state)
        if macro_support[t].any():
            macro_losses.append(matched_macro_energy(macro,target_macro[t],macro_support[t],model.field_scale))
        if micro_support[t].any():
            micro_losses.append(matched_macro_energy(agents,target_micro[t],micro_support[t],scales))
    if not macro_losses or not micro_losses:
        raise ValueError('Joint supervision requires supported macro and observed-cohort micro targets')
    macro_loss, micro_loss = torch.stack(macro_losses).mean(), torch.stack(micro_losses).mean()
    loss = .5*macro_loss+.5*micro_loss
    if return_diagnostics:
        return loss, dict(macro_energy=float(macro_loss.detach()), micro_energy=float(micro_loss.detach()),
                          cohort_size=len(cohort), target_micro_count=int(micro_support.sum()))
    return loss
