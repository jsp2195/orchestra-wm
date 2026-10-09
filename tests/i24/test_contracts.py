import copy,json
from dataclasses import replace
import numpy as np
import pytest
import torch
from orchestra_wm.i24.schema import RoadMap,KnownFutureContext,PassiveAction,CavnueAdapter
from orchestra_wm.i24.data import I24Adapter,SyntheticFixtureAdapter,observations,fields_numpy,split_scenes,save_scene,load_scene,collate_observations
from orchestra_wm.i24.model import RoadsideWorldModel,observe_history,soft_fields
from orchestra_wm.i24.training import atomic_checkpoint,load_checkpoint
from orchestra_wm.i24.evaluation import energy_score,cluster_interval

@pytest.fixture
def scene():return SyntheticFixtureAdapter().generate(1,duration=30)[0]

@pytest.fixture
def acquisition():return {'dataset':'I-24 MOTION','release':'I24MOTION_PUBLIC_v1.0','format':'json-array','direction':-1,'terms_accepted':True,'source_url':'https://i24motion.org/data','x_origin_ft':316800,'files':[{'path':'schema-test-only.json','session':'unit-test'}]}


def test_verified_v1_schema_units_direction_and_backward_derivatives(acquisition):
    r={'_id':{'$oid':'test-only'},'timestamp':[0,.1,.2,.3], 'x_position':[316800,316797,316794,316791],'y_position':[18]*4,'length':15,'width':6,'direction':-1}
    f,q=I24Adapter(acquisition).record(r,'test','fixturehash');assert np.isclose(f.s.iloc[0],15*.3048/2);assert np.isclose(f.vs.iloc[-1],9.144);assert np.isclose(f.d.iloc[0],18*.3048)
    changed=copy.deepcopy(r);changed['x_position'][-1]-=10;other,_=I24Adapter(acquisition).record(changed,'test','fixturehash');assert np.array_equal(f.vs.iloc[:-1],other.vs.iloc[:-1])
    assert not f.usable.iloc[0] and q['samples']==4
    with pytest.raises(ValueError):I24Adapter({**acquisition,'dataset':'single-vehicle CAN GPS'})
    with pytest.raises(ValueError):I24Adapter({**acquisition,'release':'unverified-v2'})
    with pytest.raises(ValueError):I24Adapter(acquisition).record({'timestamp':[0,1],'speed':[10,11]},'x','x')


def test_qc_time_gaps_and_invalid_samples(acquisition):
    r={'_id':'test','timestamp':[0,.1,.1], 'x_position':[316800]*3,'y_position':[18]*3,'length':15,'width':6,'direction':-1}
    f,q=I24Adapter(acquisition).record(r,'test','hash');assert f is None and q['invalid_time_order_track']==1


def test_columnar_scene_roundtrip(scene,tmp_path):
    p=tmp_path/'scene.parquet';save_scene(scene,p);loaded=load_scene(p);assert np.array_equal(scene.values,loaded.values);assert loaded.provenance['kind']=='SYNTHETIC_FIXTURE'
    with pytest.raises(ValueError):CavnueAdapter().validate(scene)


def test_future_roster_and_fields_do_not_leak(scene):
    obs,ids=observations(scene,25,'sparse50',19);changed=copy.deepcopy(scene);changed.values[25:]=100000;changed.existence[25:]=True
    other,otherids=observations(changed,25,'sparse50',19);assert np.array_equal(ids,otherids)
    for a,b in zip(obs,other):
        assert np.array_equal(a.values,b.values) and np.array_equal(a.fields,b.fields)
        fields,support=fields_numpy(a.values,a.visible,scene.road);assert np.array_equal(fields,a.fields)
    assert len(ids)<len(scene.track_ids) and scene.track_ids[-1] not in obs[0].track_ids
    dense,_=observations(scene,25);assert not np.array_equal(dense[-1].fields,obs[-1].fields)


def test_variable_agents_padding_invisibility_and_births(scene):
    scenes=SyntheticFixtureAdapter().generate(4);items=[observations(s,25)[0][-1] for s in scenes];b=collate_observations(items)
    assert b['padding'].any() and not (b['visible']&b['padding']).any()
    assert not scene.existence[0,-1] and scene.existence[40,-1]
    obs,_=observations(scene,25,'outage');assert not obs[-1].visible.any() and len(obs[-1].track_ids)>0


def test_split_integrity_and_overlap_guard():
    scenes=SyntheticFixtureAdapter().generate(10);s=split_scenes(scenes);assert all(s.values())
    bad=copy.deepcopy(scenes);bad[-1].session_id=bad[0].session_id
    with pytest.raises(ValueError,match='Shared raw time'):split_scenes(bad)


def test_strict_imagination_seed_variance_and_disabled_actions(scene):
    torch.set_num_threads(2);torch.manual_seed(2);m=RoadsideWorldModel(24,4);obs,_=observations(scene,25);s=observe_history(m,obs,scene.road)
    with torch.no_grad():a=m.imagine(s,5,3,42);b=m.imagine(s,5,3,42);c=m.imagine(s,5,3,43)
    assert torch.equal(a['agents'],b['agents']) and not torch.equal(a['agents'],c['agents'])
    assert a['agents'].shape==(3,5,len(obs[0].track_ids),8) and a['fields'].shape==(3,5,scene.road.field_count,3)
    with pytest.raises(ValueError):m.imagine_step(s,{'future_tracks':scene.values[25:]})
    with pytest.raises(RuntimeError):m.condition_on_actions(PassiveAction(np.zeros(3)))
    with pytest.raises(ValueError):m.observe(s,{'future':scene.values},torch.ones_like(s.valid),.2)
    changed=copy.deepcopy(scene);changed.values[25:]=np.random.default_rng(1).normal(size=changed.values[25:].shape)
    altered,_=observations(changed,25);state=observe_history(m,altered,scene.road)
    with torch.no_grad():d=m.imagine(state,5,3,42)
    assert torch.equal(a['agents'],d['agents'])


def test_memory_and_independent_cross_agent_exclusion(scene):
    torch.manual_seed(9);m=RoadsideWorldModel(24,4,'independent');obs,_=observations(scene,25);s=observe_history(m,obs,scene.road);other=replace(s,agents=s.agents.clone(),memory=s.memory.clone());other.agents[:,1,0]+=100;other.memory[:,1]+=5
    with torch.no_grad():a=m.imagine(s,4,2,41);b=m.imagine(other,4,2,41)
    assert torch.equal(a['agents'][:,:,0],b['agents'][:,:,0])
    full=RoadsideWorldModel(24,4);hidden,_=observations(scene,25,'outage');intact=observe_history(full,hidden,scene.road);reset=observe_history(full,hidden,scene.road,reset=True)
    assert not torch.equal(intact.memory,reset.memory)


def test_macro_consistency_mass_and_energy_score(scene):
    a=torch.tensor(scene.values[0:1]);valid=torch.tensor(scene.existence[0:1]);f=soft_fields(a,valid,scene.road)
    assert torch.isclose(f[:,:,1].sum()*(scene.road.s_max-scene.road.s_min)/scene.road.bins/1000,valid.sum().float(),atol=1e-4)
    assert energy_score(np.zeros((4,3)),np.zeros(3))==0
    assert energy_score(np.ones((4,3)),np.zeros(3))>0
    assert cluster_interval([1,2],['same','same'])['ci_low'] is None


def test_atomic_checkpoint_restores_predictions(scene,tmp_path):
    m=RoadsideWorldModel(24,4);c={'model_dim':24,'latent_dim':4,'dt':.2};p=tmp_path/'test.pt';atomic_checkpoint(p,{'model':m.state_dict(),'config':c,'variant':'full'});loaded,_=load_checkpoint(p)
    obs,_=observations(scene,25)
    with torch.no_grad():a=m.imagine(observe_history(m,obs,scene.road),3,2,71);b=loaded.imagine(observe_history(loaded,obs,scene.road),3,2,71)
    assert torch.equal(a['agents'],b['agents']) and not p.with_suffix('.tmp').exists()


def test_open_road_never_uses_future(scene):
    from orchestra_wm.i24.open_road import generate_open_road
    m=RoadsideWorldModel(16,3);obs,_=observations(scene,45);a,meta=generate_open_road(m,obs,scene.road,10,seed=9);b,_=generate_open_road(m,obs,scene.road,10,seed=9)
    assert not meta['future_truth_used'] and np.array_equal(a[-1]['agents'],b[-1]['agents'])


def test_generated_demo_and_complete_smoke_artifacts():
    from pathlib import Path
    from orchestra_wm.i24.reporting import verify_demo
    out=Path('outputs/i24')
    if not (out/'validation.json').exists() or not (out/'rollouts').exists():pytest.skip('Full smoke integration is executed by its CLI before final validation')
    v=json.loads((out/'validation.json').read_text());assert v['pass'];assert verify_demo(out)>0
    m=json.loads((out/'metrics.json').read_text());assert m['source_kind']=='SYNTHETIC_FIXTURE' and m['real_data_status']=='BLOCKED'
    assert 'SYNTHETIC FIXTURE' in (out/'demo/index.html').read_text()


def test_streamed_authentic_schema_example_and_provenance(acquisition,tmp_path):
    # Fabricated numbers in the documented schema are a UNIT TEST, never real source evidence.
    r={'_id':'schema-unit-test','timestamp':[0,.1,.2,.3], 'x_position':[316800,316797,316794,316791],'y_position':[18]*4,'length':15,'width':6,'direction':-1}
    (tmp_path/'schema-test-only.json').write_text(json.dumps([r]))
    m=I24Adapter(acquisition).convert(tmp_path,tmp_path/'converted',100)
    assert m['rows']==4 and m['tracks']==1 and len(m['sources'][0]['sha256'])==64
    assert (tmp_path/'converted/part-00000.parquet').exists()


def test_no_silent_real_or_research_fixture_fallback(tmp_path):
    import yaml
    from orchestra_wm.i24.pipeline import run
    # Existing REAL config must fail before constructing any fixture or model.
    with pytest.raises(ValueError,match='REAL data gate blocked'):run('configs/i24_pilot.yaml')
    with pytest.raises(ValueError,match='Research gated'):run('configs/i24_research.yaml')


def test_idm_calibration_gate_and_following_physics(scene):
    from orchestra_wm.i24.baselines import calibrate_idm,idm_acceleration
    params,status=calibrate_idm([scene]);assert params is None and status['status']=='NOT_APPLICABLE'
    p=[30,1.4,1.2,2.,2.]
    assert idm_acceleration(20,5,8,p)<idm_acceleration(20,0,100,p)


def test_immutable_conversion_resume(acquisition,tmp_path):
    r={'_id':'schema-resume','timestamp':[0,.1,.2,.3], 'x_position':[316800,316797,316794,316791],'y_position':[18]*4,'length':15,'width':6,'direction':-1}
    source=tmp_path/'schema-test-only.json';source.write_text(json.dumps([r]));adapter=I24Adapter(acquisition)
    first=adapter.convert(tmp_path,tmp_path/'converted',100);second=adapter.convert(tmp_path,tmp_path/'converted',100);assert first==second
    source.write_text(json.dumps([r,r]))
    with pytest.raises(ValueError,match='Source content changed'):adapter.convert(tmp_path,tmp_path/'converted',100)
