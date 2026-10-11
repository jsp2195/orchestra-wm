import numpy as np
import torch
from orchestra_wm.i24.data import SyntheticFixtureAdapter
from orchestra_wm.i24_continuous.data import observations, exposure_fields
from orchestra_wm.i24_phase4b.model import AuditedWorldModel
from orchestra_wm.i24_phase4c.joint import joint_objective, population_targets


def test_residual_population_uses_extensive_quantities_not_speed_subtraction():
    scene = SyntheticFixtureAdapter().generate(count=1)[0]
    cohort = np.arange(0,len(scene.track_ids),2)
    result = population_targets(scene,cohort)
    assert np.allclose(result['total'][...,1:],result['cohort'][...,1:]+result['residual'][...,1:],atol=.002)
    assert result['residual'][...,1].sum()>0
    assert np.allclose(result['total'][...,2],3.6*result['total'][...,0]*result['total'][...,1],atol=.002)


def test_joint_scores_train_motion_and_macro_without_future_conditioning():
    torch.set_num_threads(2); torch.manual_seed(101)
    scene = SyntheticFixtureAdapter().generate(count=1)[0]
    fields,support = exposure_fields(scene.values,scene.existence,scene.road)
    model = AuditedWorldModel(32,4,'full',.2)
    loss, diagnostics = joint_objective(model,scene,fields,support,'dense',101,horizon=3,return_diagnostics=True)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.acceleration.weight.grad.abs().sum()>0
    assert model.field_decoder.weight.grad.abs().sum()>0
    assert model.posterior[0].weight.grad is None
    assert diagnostics['cohort_size']==len(observations(scene,25,'dense',101)[1])
