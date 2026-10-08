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
