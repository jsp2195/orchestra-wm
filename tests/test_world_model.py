import inspect
import numpy as np
import torch
from orchestra_wm.models.world_model import WorldModel
from orchestra_wm.planning.orchestrator import Orchestrator
from orchestra_wm.planning.categorical_cem import categorical_cem

def test_imagination_is_autonomous_and_action_conditioned():
    torch.set_num_threads(2);torch.manual_seed(1)
    m=WorldModel(32,1).eval()
    # Nonzero learned decoder emulates a trained model for API contract testing.
    torch.nn.init.normal_(m.decoder[-1].weight,std=.01)
    s=m.initial_state(1,4,2);s.control[:]=1;s.agents[:,:,5]=1
    a=torch.ones(1,3,4)
    first=m.imagine(s,a)['agents'];second=m.imagine(s,-a)['agents']
    assert first.shape==(1,3,4,6)
    assert not torch.allclose(first,second)
    assert set(inspect.signature(m.imagine_step).parameters)=={'state','joint_action'}
    torch.testing.assert_close(first,m.imagine(s,a)['agents'])
    assert torch.equal(s.memory,torch.zeros_like(s.memory))

def test_planner_cannot_query_simulator(monkeypatch):
    from orchestra_wm.envs.traffic_env import TrafficEnv
    def forbidden(*a,**kw): raise AssertionError('simulator leak')
    monkeypatch.setattr(TrafficEnv,'step',forbidden)
    m=WorldModel(32,1).eval();s=m.initial_state(1,4,2);s.control[:]=1
    p=Orchestrator(m,{'objective_weights':[1,1,-1,10,2,.1],'plan_horizon':2,'candidates':4,'cem_iterations':2})
    assert p.plan(s)[0].shape==(2,4)

def test_cem_improves_known_objective_and_masks():
    plan,cost,history=categorical_cem(lambda p:((p-1)**2).sum((1,2)),4,3,32,3,np.random.default_rng(3),[True,True,False])
    assert np.all(plan[:,2]==0)
    assert cost==4
    assert all(a>=b for a,b in zip(history,history[1:]))

def test_hidden_agent_memory_is_retained():
    m=WorldModel(32,1).eval();s=m.initial_state(1,4,2)
    obs=torch.zeros(1,4,12);obs[:,:,5]=1;obs[:,:,0]=.3
    s=m.observe(s,{'agents':obs,'lanes':torch.zeros(1,2,8)},torch.ones(1,4,dtype=torch.bool),None)
    h=m.observe(s,{'agents':torch.zeros_like(obs),'lanes':torch.zeros(1,2,8)},torch.zeros(1,4,dtype=torch.bool),None)
    torch.testing.assert_close(s.agents,h.agents)

def test_future_observations_cannot_enter_imagination(monkeypatch):
    torch.set_num_threads(2)
    m=WorldModel(32,1).eval();s=m.initial_state(1,5,2)
    def forbidden(*a,**kw): raise AssertionError('observe invoked during imagination')
    monkeypatch.setattr(m,'observe',forbidden)
    result=m.imagine(s,torch.zeros(1,5,5))
    assert torch.isfinite(result['agents']).all()
    import pytest
    with pytest.raises(TypeError): m.imagine(s,torch.zeros(1,5,5),future_observations=torch.zeros(1))

def test_padding_does_not_change_real_agent_predictions():
    torch.manual_seed(7);m=WorldModel(32,1).eval()
    torch.nn.init.normal_(m.decoder[-1].weight,std=.02)
    obs=torch.randn(1,3,12);obs[:,:,5:8]=1;lane=torch.randn(1,2,8)
    s=m.observe(m.initial_state(1,3,2),{'agents':obs,'lanes':lane},torch.ones(1,3,dtype=torch.bool),None)
    padded=torch.cat([obs,torch.randn(1,2,12)*100],1)
    valid=torch.tensor([[1,1,1,0,0]],dtype=torch.bool)
    p=m.observe(m.initial_state(1,5,2),{'agents':padded,'lanes':lane,'valid':valid},valid,None)
    a=m.imagine(s,torch.ones(1,3,3))['agents'];b=m.imagine(p,torch.ones(1,3,5))['agents'][:,:,:3]
    torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-5)

def test_checkpoint_roundtrip(tmp_path):
    from orchestra_wm.models.training import load_model
    m=WorldModel(32,1).eval();path=tmp_path/'model.pt'
    torch.save({'model':m.state_dict(),'config':{'model_dim':32,'layers':1,'dt':.5},'variant':'orchestra'},path)
    other=load_model(path);s=m.initial_state(1,3,2);actions=torch.ones(1,2,3)
    torch.testing.assert_close(m.imagine(s,actions)['agents'],other.imagine(s,actions)['agents'])

def test_oracle_is_non_mutating_and_beats_included_noop():
    from orchestra_wm.envs.traffic_env import TrafficEnv
    from orchestra_wm.planning.oracle import oracle_plan,true_rollout
    e=TrafficEnv(seed=5);before=e.state_array().copy()
    cfg={'objective_weights':[1,.5,-.8,12,2,.04],'plan_horizon':5,'candidates':8,'cem_iterations':2}
    plan,cost,_=oracle_plan(e,cfg,np.random.default_rng(7))
    noop=(true_rollout(e,np.zeros((5,e.n),int))[1]@np.array(cfg['objective_weights'])).sum()
    assert cost<=noop+1e-6
    assert np.isin(plan,[-1,0,1]).all()
    np.testing.assert_array_equal(before,e.state_array())

def test_independent_dynamics_do_not_mix_entities_and_no_action_is_invariant():
    torch.manual_seed(3)
    for variant in ['independent','no_actions']:
        m=WorldModel(32,1,variant=variant).eval()
        torch.nn.init.normal_(m.decoder[-1].weight,std=.01)
        s=m.initial_state(1,3,2);s.control[:]=1;s.agents[:,:,5]=1
        first=m.imagine(s,torch.ones(1,3,3))['agents']
        if variant=='no_actions':
            torch.testing.assert_close(first,m.imagine(s,-torch.ones(1,3,3))['agents'])
        else:
            s.memory[:,1:]+=20
            other=m.imagine(s,torch.ones(1,3,3))['agents']
            torch.testing.assert_close(first[:,:,0],other[:,:,0])
