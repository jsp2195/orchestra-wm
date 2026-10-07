import numpy as np
import pytest
from orchestra_wm.envs.traffic_env import TrafficEnv

@pytest.mark.parametrize('family',['intersection','merge','corridor','grid'])
def test_reproducible_and_bounded(family):
    a,b=TrafficEnv(family=family,seed=9),TrafficEnv(family=family,seed=9)
    for _ in range(50):
        oa,ra,*_=a.step(np.ones(a.n,int)); ob,rb,*_=b.step(np.ones(b.n,int))
        np.testing.assert_array_equal(oa['agents'],ob['agents']); assert ra==rb
        assert all(0<=v.speed<=18 and abs(v.acceleration)<=4 for v in a.vehicles)

def test_clone_action_effect_and_background_mask():
    a=TrafficEnv(seed=3); b=a.clone()
    for _ in range(8): a.step(np.ones(a.n,int));b.step(-np.ones(b.n,int))
    assert np.linalg.norm(a.state_array()-b.state_array())>.1
    a=TrafficEnv({'connected':0},seed=3); b=a.clone()
    a.step(np.ones(a.n,int)); b.step(-np.ones(b.n,int))
    np.testing.assert_array_equal(a.state_array(),b.state_array())

def test_sensing_hides_state_and_supports_variable_counts():
    for n in [4,8,16,32,40]:
        e=TrafficEnv({'vehicles':n,'connected':2,'sensor_regime':'probe'})
        o,_=e.reset(seed=2)
        assert o['mask'].sum()==2
        assert np.all(o['agents'][~o['mask'],:6]==0)
        assert e.observation_space.contains(o)

def test_departures_and_respawn():
    e=TrafficEnv({'respawn':True})
    for v in e.vehicles: v.s=e.road.lanes[v.lane].length-0.01
    e.step(np.ones(e.n,int))
    assert e.completed>0
    assert any(v.generation>0 for v in e.vehicles)

def test_collision_detection():
    e=TrafficEnv()
    e.vehicles[0].lane=e.vehicles[1].lane=0
    e.vehicles[0].s=e.vehicles[1].s=60
    _,_,_,_,info=e.step(np.zeros(e.n,int))
    assert info['collisions']>=1

def test_ids_routes_jerk_and_outage():
    e=TrafficEnv({'sensor_regime':'dropout','connected':2,'miss_probability':0},seed=19)
    obs,_=e.reset(seed=19);ids=obs['ids'].copy()
    previous=np.array([v.acceleration for v in e.vehicles])
    for t in range(16):
        progress=np.array([v.s for v in e.vehicles])
        obs,*_=e.step(np.full(e.n,1 if t<5 else -1))
        np.testing.assert_array_equal(ids,obs['ids'])
        assert all(v.route==(v.lane,) for v in e.vehicles)
        assert np.all(np.array([v.s for v in e.vehicles])>=progress)
        acceleration=np.array([v.acceleration for v in e.vehicles])
        assert np.max(np.abs(acceleration-previous))<=3*e.dt+1e-6
        previous=acceleration
        if 8<=e.t<14: assert not obs['mask'][2:].any()
    assert e.vehicles[0].speed<6

def test_reset_replays_same_sensor_noise():
    e=TrafficEnv(seed=13);a,_=e.reset(seed=13)
    e.step(np.ones(e.n,int));b,_=e.reset(seed=13)
    for key in a: np.testing.assert_array_equal(a[key],b[key])

def test_crowded_initialization_has_no_stacked_same_lane_vehicles():
    for n in [16,24,32,40]:
        e=TrafficEnv({'vehicles':n},seed=7)
        for lane in e.road.lanes:
            positions=sorted(v.s for v in e.vehicles if v.lane==lane.id)
            assert np.min(np.diff(positions))>3
