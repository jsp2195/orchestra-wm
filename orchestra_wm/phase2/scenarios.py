import itertools
import numpy as np
from orchestra_wm.envs.traffic_env import TrafficEnv
from orchestra_wm.sim.observations import sense

PAIRS=list(itertools.product([-1,0,1],repeat=2))
LABELS=['YY','YM','YP','MY','MM','MP','PY','PM','PP']

def interaction_residual(values):
    values=np.asarray(values)
    grid=values.reshape(3,3,*values.shape[1:])
    return grid-grid.mean(0,keepdims=True)-grid.mean(1,keepdims=True)+grid.mean((0,1),keepdims=True)

def make_case(cfg,index,split='train',family=None):
    offsets={'audit':110000,'train':210000,'validation':310000,'test':410000,'memory':510000}
    seed=offsets[split]+index
    family=family or ['intersection','merge'][index%2]
    env=TrafficEnv({**cfg,'connected':2},family,seed)
    rng=np.random.default_rng(seed)
    distance=float(rng.uniform(14,28));asymmetry=[-8,-4,0,4,8][(index//2)%5]
    for i in range(2):
        v=env.vehicles[i];v.lane=([0,2] if family=='intersection' else [0,1])[i];v.route=(v.lane,)
        lane=env.road.lanes[v.lane]
        conflict_s=(77 if i==0 else 73) if family=='intersection' else np.linalg.norm(np.diff(lane.centerline[:3],axis=0),axis=1).sum()
        d=distance+(asymmetry if i else 0)
        v.s=float(conflict_s-d);v.speed=float(rng.uniform(4.5,7));v.desired_speed=float(rng.uniform(8.5,11.5))
        v.acceleration=0
    # Background queues remain physically behind any leading same-lane AV.
    for lane in env.road.lanes:
        leaders=[v.s for v in env.vehicles[:2] if v.lane==lane.id]
        start=min(leaders)-14 if leaders else 22
        for rank,v in enumerate([v for v in env.vehicles[2:] if v.lane==lane.id]):
            v.s=max(0,start-rank*10);v.speed=3
    env.lane_history=[];obs=sense(env);frames=[];previous=None
    for t in range(cfg['context']):
        frames.append((obs,previous,env.state_array()))
        if t<cfg['context']-1:
            previous=np.zeros(env.n,int);obs,*_=env.step(previous)
    return env,frames

def factorial_plans(env,horizon):
    plans=np.zeros((9,horizon,env.n),int)
    for k,(a,b) in enumerate(PAIRS):plans[k,:,0]=a;plans[k,:,1]=b
    return plans

def diverse_plans(env,horizon):
    first=factorial_plans(env,horizon)
    second=first.copy()
    second[:,horizon//2:,:2]=-second[:,horizon//2:,:2]
    return np.concatenate([first,second])

def pair_relevant(state,i,j,family,lane_i,lane_j,distance=30):
    if state[i,5]<.5 or state[j,5]<.5 or lane_i==lane_j:return False
    if family=='intersection' and abs(np.dot(state[i,3:5],state[j,3:5]))>.5:return False
    node=np.array([5.,0.]) if family=='merge' else np.zeros(2)
    for k in [i,j]:
        delta=node-state[k,:2]*100
        if np.linalg.norm(delta)>distance or np.dot(delta,state[k,3:5])<-3:return False
    return True
