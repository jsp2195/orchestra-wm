import numpy as np

OBS_DIM = 12
STATE_DIM = 6
LANE_DIM = 8

def sense(env):
    state = env.state_array()
    n = len(state)
    visible = env.sensor_rng.random(n) > env.cfg.get('miss_probability', .18)
    xy = state[:,:2]*100
    # A fixed camera blind spot and a repeatable outage force useful memory.
    visible &= ~((xy[:,0]>-16)&(xy[:,0]<-3)&(xy[:,1]<5))
    regime = env.cfg.get('sensor_regime','mixed')
    if regime == 'full': visible[:] = True
    if regime == 'sparse': visible &= xy[:,0] < 5
    if regime == 'probe': visible[:] = False
    if regime == 'dropout' or regime == 'mixed':
        if 8 <= env.t % 24 < 14: visible[:] = False
    connected = np.array([v.connected and v.active for v in env.vehicles])
    visible |= connected
    visible &= state[:,5] > .5
    env.observation_age = np.where(visible,0,env.observation_age+env.dt)
    obs = np.zeros((n,OBS_DIM),np.float32)
    obs[:,:6] = state
    noise = env.sensor_rng.normal(0,env.cfg.get('detection_noise',.6)/100,(n,2))
    obs[:,:2] += noise * (~connected[:,None])
    obs[:,6] = connected
    obs[:,7] = visible
    obs[:,8] = env.observation_age/10
    obs[:,9] = np.array([v.lane for v in env.vehicles])/8
    obs[:,10] = np.array([v.s/env.road.lanes[v.lane].length for v in env.vehicles])
    obs[:,11] = 1.0  # track-confidence; zeroed for missing detections
    obs[~visible,:6] = 0
    obs[~visible,9:] = 0
    if env.cfg.get('track_loss',False):
        # Track loss removes detections; stable slot identity resumes on reacquisition.
        lost = (env.t % 17 >= 10) & ~connected
        visible[lost] = False
        obs[lost,:6] = 0
        obs[lost,7] = 0
        obs[lost,9:] = 0
    lanes = np.zeros((len(env.road.lanes),LANE_DIM),np.float32)
    for lane in env.road.lanes:
        vs = [v for v in env.vehicles if v.active and v.lane == lane.id]
        lanes[lane.id] = [len(vs)/10, np.mean([v.speed for v in vs])/15 if vs else 0,
                          sum(v.speed<2 for v in vs)/10, 1,
                          * (lane.centerline[0]/100), *(lane.centerline[-1]/100)]
    # Delayed loop measurements; cameras and lane loops are distinct modalities.
    current = lanes.copy()
    if env.lane_history and env.cfg.get('lane_delay',2):
        lanes[:,:3] = env.lane_history[max(0,len(env.lane_history)-env.cfg.get('lane_delay',2))][:,:3]
    if regime == 'probe': lanes[:,:4] = 0
    env.lane_history.append(current)
    env.lane_history = env.lane_history[-4:]
    return {'agents':obs, 'mask':visible.astype(bool), 'lanes':lanes,
            'control_mask':connected, 'ids':np.array([v.id + v.generation*1000 for v in env.vehicles],np.int64)}
