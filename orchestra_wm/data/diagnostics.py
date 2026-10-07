import numpy as np
from orchestra_wm.evaluation.common import write_json

def diagnose(path,out):
    d=np.load(path)
    a=d['actions'];control=d['control'][:,:-1]
    selected=a[control]
    result={'episodes':len(a),'transitions':int(np.prod(a.shape[:2])),
        'scenario_counts':dict(zip(*np.unique(d['families'],return_counts=True))),
        'action_fraction':{str(v):float((selected==v).mean()) for v in [-1,0,1]},
        'connected_count_mean':float(control.sum(-1).mean()),
        'missing_detection_fraction':float(1-d['mask'].mean()),
        'conflict_step_fraction':float((d['components'][...,4]>0).mean()),
        'collision_step_fraction':float((d['components'][...,3]>0).mean()),
        'interaction_definition':'at least one pair closer than 7 synthetic metres',
        'seed_range':[int(d['episode_seeds'].min()),int(d['episode_seeds'].max())],
        'policy_scenario_counts':{f'{f}/{p}':int(((d['families']==f)&(d['policies']==p)).sum()) for f in np.unique(d['families']) for p in np.unique(d['policies'])}}
    assert np.isfinite(d['obs']).all()
    assert d['states'].shape[:-1]==d['obs'].shape[:-1]
    assert np.all(d['obs'][~d['mask'],:6]==0)
    assert np.all(d['ids'][:,1:]==d['ids'][:,:-1])
    assert result['action_fraction']['0']<.6
    write_json(out,result)
    return result
