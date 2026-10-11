import numpy as np
from orchestra_wm.i24.schema import RoadMap
from orchestra_wm.i24_phase4c.evaluation import score_arrays


def arrays():
    a = np.zeros((8,100,2,8))
    a[...,0] = 50; a[...,1] = 5
    return dict(agents=a, fields=np.ones((8,100,4,3)),
                truth=a[0].copy(), valid=np.ones((100,2),bool),
                target_fields=np.ones((100,4,3)), field_support=np.ones((100,4),bool))


def test_physical_horizons_and_scoring_support():
    a = arrays()
    a['agents'][:,49,0,0] += 10
    a['fields'][:,49,0,0] += 30
    a['field_support'][:,3] = False
    a['fields'][:,:,3] = 1e8 # Unsupported values must not affect score.
    rows = score_arrays(a,RoadMap(0,100,(5.,9.),4.,2,1.),'full')
    assert rows[0]['fde_m'] == 0 and rows[2]['fde_m'] == 0
    assert rows[1]['fde_m'] == 5
    assert np.isclose(rows[1]['macro_energy'],1/3)
    assert rows[0]['macro_energy'] == 0 and rows[2]['macro_energy'] == 0
    assert rows[1]['valid_micro_samples'] == 100
    assert rows[1]['valid_micro_endpoint'] == 2
    assert rows[1]['valid_lane_bins'] == 3


def test_macro_only_has_no_micro_metrics_and_target_missingness_is_respected():
    a = arrays(); a['valid'][:,1] = False
    a['truth'][:,1,:2] = 1e8
    road = RoadMap(0,100,(5.,9.),4.,2,1.)
    assert 'fde_m' not in score_arrays(a,road,'macro_only')[0]
    assert score_arrays(a,road,'full')[0]['fde_m'] == 0


def test_interval_coverage_and_flow_identity_are_physical():
    a = arrays(); a['target_fields'][...,0] = 2
    r = score_arrays(a,RoadMap(0,100,(5.,9.),4.,2,1.),'full')[0]
    assert r['speed_coverage90'] == 0 and r['density_coverage90'] == 1
    assert r['macro_diversity'] == 0 and r['speed_width90'] == 0
    assert np.isclose(r['flow_identity_mae_veh_h'],2.6)


def test_aggregate_persistence_does_not_treat_sensor_outage_as_empty_road():
    from orchestra_wm.i24.data import SyntheticFixtureAdapter
    from orchestra_wm.i24_continuous.data import observations
    from orchestra_wm.i24_phase4c.evaluation import baseline
    scene=SyntheticFixtureAdapter().generate(count=1)[0]
    obs,_=observations(scene,25,'outage',101)
    assert not obs[-1].field_support.any()
    result=baseline(obs,scene,'aggregate_persistence')
    assert result['fields'][0,0,:,1].sum()>0
    assert np.array_equal(result['fields'][0,0],result['fields'][0,-1])
