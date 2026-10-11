import numpy as np
import torch
from orchestra_wm.i24.data import SyntheticFixtureAdapter
from orchestra_wm.i24_continuous.data import observations, exposure_fields
from orchestra_wm.i24_phase4b.model import AuditedWorldModel
from orchestra_wm.i24_phase4c.joint import JointWorldModel, joint_objective, population_targets


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
    model = JointWorldModel(32,4,'full',.2)
    loss, diagnostics = joint_objective(model,scene,fields,support,'dense',101,horizon=3,return_diagnostics=True)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.acceleration.weight.grad.abs().sum()>0
    assert model.field_decoder.weight.grad.abs().sum()>0
    assert model.posterior[0].weight.grad is None
    assert diagnostics['cohort_size']==len(observations(scene,25,'dense',101)[1])


def test_total_field_identity_does_not_require_cohort_population_equality():
    from orchestra_wm.i24.model import observe_history
    scene=SyntheticFixtureAdapter().generate(count=1)[0]
    model=JointWorldModel(32,4,'full',.2);obs,_=observations(scene,25,'dense',101)
    state=observe_history(model,obs,scene.road)
    with torch.no_grad():forecast=model.imagine(state,3,8,101)
    fields=forecast['fields']
    assert (fields>=0).all()
    assert torch.allclose(fields[...,2],3.6*fields[...,0]*fields[...,1])
    assert not torch.allclose(fields,forecast['derived_fields'])


def test_independent_micro_forecast_ignores_neighbor_and_macro_state():
    from dataclasses import replace
    from orchestra_wm.i24.schema import RoadMap
    torch.manual_seed(101)
    model=JointWorldModel(32,4,'independent',.2)
    state=model.initial_state(dict(road=RoadMap(0,100,(5.,9.),4.,2,1.),track_ids=('a','b')))
    state.agents[:,:,0]=50;state.agents[:,:,1]=5;state.agents[:,:,2]=10
    state.seen[:]=True
    other=replace(state,agents=state.agents.clone(),memory=state.memory.clone(),
                  fields=torch.randn_like(state.fields)*10,field_memory=torch.randn_like(state.field_memory))
    other.agents[:,1,0]=70;other.agents[:,1,2]=20;other.memory[:,1]=5
    with torch.no_grad():
        a=model.imagine(state,3,8,101);b=model.imagine(other,3,8,101)
    assert torch.equal(a['agents'][:,:,0],b['agents'][:,:,0])
    assert not torch.equal(a['fields'],b['fields'])
