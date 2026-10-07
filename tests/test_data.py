import numpy as np
from orchestra_wm.data.collection import collect
from orchestra_wm.utils.config import load_config

def test_every_scenario_gets_every_behavior(tmp_path):
    c=load_config('configs/smoke.yaml');c.update(episodes=12,episode_steps=3)
    path=tmp_path/'data.npz';collect(c,path);d=np.load(path)
    for family in c['scenarios']:
        assert len(set(d['policies'][d['families']==family]))==6
    assert 'rewards' in d and 'ids' in d
    assert np.all(d['ids'][:,1:]==d['ids'][:,:-1])
