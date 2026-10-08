import numpy as np
import torch
from orchestra_wm.phase2.scenarios import interaction_residual,make_case,factorial_plans
from orchestra_wm.utils.config import load_config

def test_additive_projection_removes_marginals():
    additive=np.arange(3)[:,None]+2*np.arange(3)[None,:]
    np.testing.assert_allclose(interaction_residual(additive.ravel()),0,atol=1e-10)
    additive[2,2]+=3
    assert np.linalg.norm(interaction_residual(additive.ravel()))>0

def test_factorial_siblings_share_initial_state_and_cover_grid():
    c=load_config('configs/phase2.yaml');e,frames=make_case(c,0)
    plans=factorial_plans(e,5);before=e.state_array().copy()
    assert len(set(map(tuple,plans[:,0,:2])))==9
    for p in plans:
        clone=e.clone();clone.step(p[0]);np.testing.assert_array_equal(e.state_array(),before)
    assert frames[-1][0]['control_mask'].sum()==2

def test_pairwise_model_preserves_autonomous_api():
    from orchestra_wm.phase2.model import PairwiseWorldModel
    torch.set_num_threads(2);m=PairwiseWorldModel(32,1).eval();s=m.initial_state(1,4,2);s.control[:]=1
    p=m.imagine(s,torch.zeros(1,3,4))
    assert p['agents'].shape==(1,3,4,6) and torch.isfinite(p['agents']).all()

def test_reproduction_is_keyed_not_row_order():
    import pandas as pd
    from orchestra_wm.phase2.audits import compare_reproduction
    a=pd.DataFrame({'seed':[1,1],'model':['independent','orchestra'],'value':[2.,3.]})
    assert compare_reproduction(a,a.iloc[::-1],['seed','model'])==0

def test_balanced_candidate_pool_and_model_only_planning(monkeypatch):
    from orchestra_wm.phase2.planning import initial_candidates,JointPlanner
    from orchestra_wm.models.world_model import WorldModel
    from orchestra_wm.envs.traffic_env import TrafficEnv
    p=initial_candidates(10,10)
    assert len(np.unique(p.reshape(18,-1),axis=0))==18
    assert len(np.unique(p[:9,0,:2],axis=0))==9
    def forbidden(*a,**kw):raise AssertionError('Simulator queried inside learned planner')
    monkeypatch.setattr(TrafficEnv,'step',forbidden)
    m=WorldModel(32,1).eval();s=m.initial_state(1,4,2);s.control[:,:2]=1
    c=load_config('configs/phase2.yaml')
    plan,_,_=JointPlanner(m,c,5).plan(s)
    assert np.isin(plan,[-1,0,1]).all() and np.all(plan[:,2:]==0)

def test_phase2_ordinary_loss_matches_phase1():
    from orchestra_wm.models.world_model import WorldModel
    from orchestra_wm.models.training import rollout_loss
    from orchestra_wm.phase2.training import loss_and_predictions
    from orchestra_wm.data.dataset import TrajectoryDataset
    from pathlib import Path
    import pytest
    if not Path('outputs/smoke/data.npz').exists():pytest.skip('Phase-1 local dataset not restored')
    c=load_config('configs/phase2.yaml');c={**c,'train_horizon':3}
    b=TrajectoryDataset('outputs/smoke/data.npz').sample(2,4,3,np.random.default_rng(7),'cpu')
    m=WorldModel(32,1).eval()
    torch.testing.assert_close(loss_and_predictions(m,b,c)[0],rollout_loss(m,b,c,'cpu'))

def test_pairing_shuffle_preserves_marginals_and_changes_pairings():
    from orchestra_wm.phase2.evaluation import pairing_shuffle
    from orchestra_wm.phase2.planning import initial_candidates
    plans=initial_candidates(10,10)[:9];shuffled=pairing_shuffle(plans,123)
    np.testing.assert_array_equal(plans[:,:,0],shuffled[:,:,0])
    for action in [-1,0,1]:assert (plans[:,:,1]==action).sum()==(shuffled[:,:,1]==action).sum()
    assert not np.array_equal(plans,shuffled)

def test_confidence_interval_uses_independent_training_seeds():
    from orchestra_wm.phase2.statistics import seed_interval,hierarchical_interval
    x=seed_interval([1,2,3]);assert x['n_seeds']==3 and x['ci_low']<0 and x['ci_high']>4
    zero=seed_interval([0,0,0]);assert zero['ci_low']==zero['ci_high']==0
    b=hierarchical_interval(np.zeros((3,16)),replicates=30)
    assert b['bootstrap_low']==b['bootstrap_high']==0

def test_dataset_is_family_disjoint_and_sibling_contexts_identical():
    from pathlib import Path
    import pytest
    path=Path('outputs/phase2/dataset_v2_factorial/data.npz')
    if not path.exists():pytest.skip('Factorial dataset not yet generated')
    d=np.load(path);train=set(d['family_ids'][d['splits']=='train']);validation=set(d['family_ids'][d['splits']=='validation'])
    assert train.isdisjoint(validation)
    for fid in np.unique(d['family_ids']):
        indices=np.flatnonzero(d['family_ids']==fid);assert len(indices)==9
        for i in indices[1:]:np.testing.assert_array_equal(d['obs'][indices[0],:4],d['obs'][i,:4])


def test_balanced_cem_retains_best_and_masks_background():
    from orchestra_wm.phase2.planning import balanced_cem
    seen=[]
    def score(plans):
        seen.append(plans.copy())
        return (plans[:,:,:2] != np.array([1,-1])).sum((1,2)).astype(float)
    plan,cost,history=balanced_cem(score,10,5,2,np.random.default_rng(10),np.array([1,1,0,0,0],bool))
    assert cost==0 and np.all(plan[:,:2]==[1,-1])
    assert all(np.all(x[:,:,2:]==0) for x in seen)
    assert np.all(np.diff(history)<=0)


def test_independent_cost_has_zero_factorial_interaction():
    import torch
    from orchestra_wm.phase2.model import create_model
    from orchestra_wm.phase2.planning import initial_candidates
    cfg={'model_dim':32,'layers':1,'dt':.5}
    model=create_model(cfg,'independent').eval()
    state=model.initial_state(1,4,4)
    state.control[:,:2]=1
    with torch.no_grad():
        costs=model.imagine(state.repeat(9),torch.as_tensor(initial_candidates(3,4)[:9]))['components'].sum((1,2)).numpy()
    assert np.max(np.abs(interaction_residual(costs)))<1e-5
