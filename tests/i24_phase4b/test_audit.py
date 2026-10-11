from dataclasses import replace
import numpy as np
import torch
from orchestra_wm.i24.schema import RoadMap
from orchestra_wm.i24_phase4b.metrics import marginal_crps,boundary_rates
from orchestra_wm.i24_phase4b.model import AuditedWorldModel,bounded_temporal_fields,matched_macro_energy
from orchestra_wm.i24_continuous.model import temporal_fields


def road():return RoadMap(0,100,(5.,9.),4.,2,1.)


def test_lateral_departure_is_not_density():
    a=torch.zeros(1,1,8);a[:,:,0]=50;a[:,:,1]=12;a[:,:,2]=10
    valid=torch.ones(1,1,dtype=torch.bool)
    assert temporal_fields(((a,valid),),road())[:,:,1].sum()>0 # historical defect
    assert bounded_temporal_fields(((a,valid),),road()).abs().sum()==0


def test_macro_only_ignores_roster_and_micro_inputs_and_uses_prior():
    torch.manual_seed(12);model=AuditedWorldModel(32,4,'macro_only')
    a=model.initial_state(dict(road=road(),track_ids=('a',)))
    b=model.initial_state(dict(road=road(),track_ids=('a','future-unavailable','c')))
    b=replace(b,agents=torch.randn_like(b.agents)*100,memory=torch.randn_like(b.memory),seen=torch.ones_like(b.seen))
    def forecast(s):return model.imagine_step(s,generator=torch.Generator().manual_seed(42))[1]['fields']
    first=forecast(a)
    assert torch.equal(first,forecast(b))
    with torch.no_grad():model.prior.bias[:4]+=10
    assert not torch.equal(first,forecast(a))


def test_crps_and_joint_score_dimensions():
    samples=np.array([[0.,1.],[2.,3.]])
    assert np.allclose(marginal_crps(samples,np.array([1.,2.])),0.)
    x=torch.tensor(samples)[:,None,:];target=torch.tensor([[1.,2.]])
    assert abs(float(matched_macro_energy(x,target,torch.tensor([True]),torch.ones(2))))<1e-10


def test_legal_exit_separated_from_lateral_violation():
    a=np.zeros((1,1,1,8));a[...,0]=101;a[...,1]=5;a[...,2]=10
    rates=boundary_rates(a,np.ones((1,1),bool),road())
    assert rates['longitudinal_exit_rate']==1 and rates['lateral_departure_rate']==0


def test_prior_transition_rejects_future_target():
    import pytest
    m=AuditedWorldModel(32,4,'full');s=m.initial_state(dict(road=road(),track_ids=('a',)))
    with pytest.raises(ValueError):m._transition(s,posterior_target=s.agents)


def test_future_only_identity_and_values_never_enter_observations():
    from orchestra_wm.i24.data import SyntheticFixtureAdapter
    from orchestra_wm.i24_continuous.data import observations
    scene=SyntheticFixtureAdapter().generate(count=1)[0] # explicitly engineering fixture
    first,cohort=observations(scene,25)
    changed=replace(scene,values=scene.values.copy(),detection=scene.detection.copy(),existence=scene.existence.copy())
    changed.values[25:]=1e6;changed.detection[25:]=~changed.detection[25:]
    changed.existence[25:]=~changed.existence[25:]
    second,other=observations(changed,25)
    assert np.array_equal(cohort,other)
    for a,b in zip(first,second):
        assert a.track_ids==b.track_ids
        assert np.array_equal(a.values,b.values) and np.array_equal(a.fields,b.fields)


def test_prior_objective_backpropagates_for_both_information_paths():
    from orchestra_wm.i24.data import SyntheticFixtureAdapter
    from orchestra_wm.i24_continuous.data import exposure_fields
    from orchestra_wm.i24_phase4b.training import objective
    torch.set_num_threads(2)
    s=SyntheticFixtureAdapter().generate(count=1)[0]
    fields,support=exposure_fields(s.values,s.existence,s.road)
    for variant in ('full','macro_only'):
        torch.manual_seed(17);m=AuditedWorldModel(32,4,variant)
        loss=objective(m,s,fields,support,'dense',991,horizon=3)
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(m.prior.weight.grad).all()
        assert m.prior.weight.grad.abs().sum()>0
        assert m.field_decoder.weight.grad.abs().sum()>0
        if variant=='macro_only':assert m.agent_transition.weight_hh.grad is None


def test_read_only_or_unverified_backup_cannot_start_training(tmp_path):
    import pytest,json
    from orchestra_wm.i24_phase4b.training import verify_preservation
    p=tmp_path/'receipt.json';p.write_text(json.dumps({'private':True,'sha256':'unverified'}))
    with pytest.raises(ValueError):verify_preservation(p)


def test_activation_checkpoint_recomputes_identical_random_samples_and_gradients():
    from orchestra_wm.i24.data import SyntheticFixtureAdapter
    from orchestra_wm.i24_continuous.data import exposure_fields
    from orchestra_wm.i24_phase4b.training import objective
    s=SyntheticFixtureAdapter().generate(count=1)[0]
    fields,support=exposure_fields(s.values,s.existence,s.road)
    gradients=[];losses=[]
    for checkpointed in (False,True):
        torch.manual_seed(17);m=AuditedWorldModel(32,4,'full')
        loss=objective(m,s,fields,support,'dense',991,horizon=3,checkpointed=checkpointed);loss.backward()
        losses.append(loss.detach());gradients.append(m.prior.weight.grad.clone())
    assert torch.equal(losses[0],losses[1]) and torch.allclose(gradients[0],gradients[1],atol=1e-7,rtol=1e-6)
