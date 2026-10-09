"""Actual archived Harvard records, when mounted; skips are explicit on clean CI."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from orchestra_wm.i24_msd.adapter import I24MSDAdapter, tfrecords
from orchestra_wm.i24_msd.protos.scenario_pb2 import Scenario
from orchestra_wm.i24_msd.experiment import MSDWorldModel, split_scenes, load_model
from orchestra_wm.i24.data import observations, collate_observations
from orchestra_wm.i24.model import observe_history


@pytest.fixture(scope='module')
def real_data():
    root=Path('data/i24_msd/raw/pilot_members')
    if not root.exists():pytest.skip('Authentic Harvard files not mounted; run I24-MSD acquisition/pilot')
    manifest=json.loads(Path('outputs/i24_msd/DOWNLOAD_MANIFEST.json').read_text())
    adapter=I24MSDAdapter(manifest)
    paths=sorted(root.rglob('*tfrecord*'))
    scenes=adapter.scenes(paths)
    return adapter,paths,scenes


def test_real_multivehicle_schema_timing_geometry_and_units(real_data):
    adapter,paths,scenes=real_data
    assert len(scenes)==182 and max(len(s.track_ids) for s in scenes)>=8
    byid={s.scene_id:s for s in scenes}
    tested=0
    for path in paths:
        for raw in tfrecords(path,max_records=10000):
            proto=Scenario.FromString(raw)
            if proto.scenario_id not in byid:continue
            scene=byid[proto.scenario_id]
            assert proto.current_time_index==10 and len(proto.timestamps_seconds)==91
            assert np.isclose(scene.time[5],1,atol=1e-5) and np.isclose(scene.time[-1],9,atol=1e-5)
            assert len(scene.provenance['source_map'])==len(proto.map_features)>0
            for j,track in enumerate(proto.tracks):
                for i,native_i in enumerate(range(0,91,2)):
                    state=track.states[native_i]
                    assert scene.existence[i,j]==state.valid
                    if state.valid:
                        x=float(scene.values[i,j,0])*scene.provenance['travel_sign']+scene.provenance['x_origin_m']
                        assert np.isclose(x,state.center_x,atol=1e-3,rtol=0)
                        assert np.isclose(scene.values[i,j,2],state.velocity_x*scene.provenance['travel_sign'])
                        assert scene.values[i,j,6]==state.length
            tested+=1
    assert tested==182


def test_adapter_future_changes_cannot_change_observed_inputs(real_data):
    adapter,paths,_=real_data
    for raw in tfrecords(paths[1],max_records=150):
        scene=adapter.record(raw,str(paths[1]))
        if scene is not None:break
    proto=Scenario.FromString(raw)
    for track in proto.tracks:
        for state in track.states[11:]:
            state.center_x+=500;state.velocity_x+=10
    changed=adapter.record(proto.SerializeToString(),str(paths[1]))
    assert scene.provenance['x_origin_m']==changed.provenance['x_origin_m']
    a,ids=observations(scene,6);b,ids2=observations(changed,6)
    assert np.array_equal(ids,ids2) and scene.road==changed.road
    for x,y in zip(a,b):assert np.array_equal(x.values,y.values) and np.array_equal(x.visible,y.visible)


def test_real_splits_group_ids_and_masks(real_data):
    _,_,scenes=real_data;splits=split_scenes(scenes)
    assert [len(splits[x]) for x in ('train','validation','test')]==[57,19,19]
    sets=[{track for scene in splits[k] for track in scene.track_ids} for k in splits]
    for i in range(3):
        for j in range(i):assert not sets[i]&sets[j]
    items=[observations(s,6)[0][-1] for s in scenes[:20]]
    batch=collate_observations(items)
    assert batch['padding'].any() and not (batch['visible']&batch['padding']).any()
    assert any((~s.existence).any() for s in scenes)
    duplicate=copy.deepcopy(scenes);duplicate.append(copy.deepcopy(scenes[0]))
    with pytest.raises(Exception,match='Duplicate scenario'):split_scenes(duplicate)


def test_real_checkpoint_autonomy_reproducibility_and_graph_exclusion(real_data):
    _,_,scenes=real_data
    path=Path('outputs/i24_msd/checkpoints/full-101.pt')
    if not path.exists():pytest.skip('Real pilot checkpoint not mounted')
    model,checkpoint=load_model(path);torch.set_num_threads(2)
    assert checkpoint['source_kind']=='REAL_I24_MSD' and checkpoint['step']==500
    scene=max(scenes,key=lambda s:len(s.track_ids));obs,_=observations(scene,6)
    with torch.no_grad():
        state=observe_history(model,obs,scene.road)
        a=model.imagine(state,10,3,42);b=model.imagine(state,10,3,42);c=model.imagine(state,10,3,43)
    assert torch.equal(a['agents'],b['agents']) and not torch.equal(a['agents'],c['agents'])
    with pytest.raises(ValueError):model.imagine_step(state,{'future':scene.values[6:]})
    altered=copy.deepcopy(scene);altered.values[6:]=99999;other,_=observations(altered,6)
    with torch.no_grad():d=model.imagine(observe_history(model,other,scene.road),10,3,42)
    assert torch.equal(a['agents'],d['agents'])
    assert not state.field_memory.any()  # unknown census cannot condition the pilot
    independent=MSDWorldModel(24,4,'independent');state=observe_history(independent,obs,scene.road)
    changed=copy.copy(state);changed.agents=state.agents.clone();changed.memory=state.memory.clone()
    changed.agents[:,1,0]+=100;changed.memory[:,1]+=5
    with torch.no_grad():
        a=independent.imagine(state,3,2,42);b=independent.imagine(changed,3,2,42)
    assert torch.equal(a['agents'][:,:,0],b['agents'][:,:,0])


def test_real_demo_exact_arrays_and_checkpoint_regeneration(real_data):
    from orchestra_wm.i24_msd.showcase import verify_demo
    out=Path('outputs/i24_msd')
    if not (out/'demo/index.html').exists():pytest.skip('Real pilot/demo not run')
    assert verify_demo(out)>0
    assert 'I24-MSD REAL 9-SECOND SCENARIOS' in (out/'demo/index.html').read_text()
    _,_,scenes=real_data;scene=next(s for s in scenes if (out/'rollouts'/f'{s.scene_id}-dense.npz').exists())
    loaded=np.load(out/'rollouts'/f'{scene.scene_id}-dense.npz');model,_=load_model(out/'checkpoints/full-101.pt')
    obs,_=observations(scene,6,'dense',7301)
    with torch.no_grad():pred=model.imagine(observe_history(model,obs,scene.road),30,16,9137)['agents'].numpy()
    assert np.array_equal(pred,loaded['prediction'])
    assert not np.array_equal(pred[0,:,:,:2][loaded['valid']],loaded['truth'][:,:,:2][loaded['valid']])


def test_prior_campaign_preservation():
    from orchestra_wm.i24_msd.pipeline import preserve
    assert preserve()>268
