import numpy as np
from orchestra_wm.envs.traffic_env import TrafficEnv

def policy(env,kind,rng,previous=None):
    n=env.n
    if kind=='random': return rng.integers(-1,2,n)
    if kind=='aggressive': return np.ones(n,int)
    if kind=='conservative': return -np.ones(n,int)
    if kind=='correlated':
        return previous if previous is not None and env.t%6 else rng.integers(-1,2,n)
    if kind=='independent':
        # Each AV sees only same-lane traffic within a local forward range.
        return np.array([-1 if any(w.active and w.lane==v.lane and 0<w.s-v.s<12 for w in env.vehicles) else 1 for v in env.vehicles])
    if kind=='noop': return np.zeros(n,int)
    # Central heuristic alternates conflicting streams in fixed time slots.
    return np.array([1 if v.lane==int(env.t//6)%len(env.road.lanes) else -1 for v in env.vehicles])

def collect(cfg,path):
    keys=['obs','mask','lanes','control','states','lane_targets','actions','components','ids','rewards']
    storage={k:[] for k in keys}
    families=[];policies=[]
    max_lanes=max(len(TrafficEnv(cfg,f).road.lanes) for f in cfg['scenarios'])
    for ep in range(cfg['episodes']):
        family=cfg['scenarios'][ep%len(cfg['scenarios'])]
        env=TrafficEnv(cfg,family,cfg['seed']+ep)
        obs,_=env.reset(seed=cfg['seed']+ep)
        records={k:[] for k in keys}
        rng=np.random.default_rng(cfg['seed']+ep+123)
        kind=['random','correlated','heuristic','aggressive','conservative','independent'][(ep//len(cfg['scenarios']))%6]
        previous=None
        for t in range(cfg['episode_steps']+1):
            records['obs'].append(obs['agents']); records['mask'].append(obs['mask'])
            records['control'].append(obs['control_mask']);records['ids'].append(obs['ids'])
            records['lanes'].append(np.pad(obs['lanes'],((0,max_lanes-len(obs['lanes'])),(0,0))))
            records['states'].append(env.state_array())
            records['lane_targets'].append(np.pad(env.lane_targets(),((0,max_lanes-len(obs['lanes'])),(0,0))))
            if t==cfg['episode_steps']: break
            action=policy(env,kind,rng,previous)
            obs,reward,_,_,info=env.step(action)
            records['rewards'].append(reward)
            records['actions'].append(action);records['components'].append(info['components'])
            previous=action
        for k in keys: storage[k].append(np.array(records[k]))
        families.append(family);policies.append(kind)
    np.savez_compressed(path,**{k:np.array(v) for k,v in storage.items()},families=np.array(families),policies=np.array(policies),episode_seeds=np.arange(cfg['episodes'])+cfg['seed'])
    return storage
