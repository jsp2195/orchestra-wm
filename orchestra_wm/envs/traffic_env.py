"""Self-contained synthetic Gymnasium environment with cloneable latent state."""
import copy
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from orchestra_wm.sim.road_graph import make_road
from orchestra_wm.sim.vehicle import Vehicle
from orchestra_wm.sim.drivers import desired_acceleration
from orchestra_wm.sim.dynamics import integrate
from orchestra_wm.sim.observations import sense

class TrafficEnv(gym.Env):
    metadata = {'render_modes':['rgb_array']}
    def __init__(self,cfg=None,family='intersection',seed=0):
        self.cfg = dict(cfg or {})
        self.family = family
        self.dt = self.cfg.get('dt',.5)
        self.n = self.cfg.get('vehicles',10)
        self.action_space = spaces.MultiDiscrete(np.full(self.n,3),start=np.full(self.n,-1))
        self.road = make_road(family,self.cfg.get('geometry_scale',1))
        self.observation_space = spaces.Dict({
            'agents':spaces.Box(-np.inf,np.inf,(self.n,12),np.float32),
            'mask':spaces.MultiBinary(self.n),
            'lanes':spaces.Box(-np.inf,np.inf,(len(self.road.lanes),8),np.float32),
            'control_mask':spaces.MultiBinary(self.n),
            'ids':spaces.Box(0,np.iinfo(np.int64).max,(self.n,),np.int64)})
        self.reset(seed=seed)

    def reset(self,seed=None,options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        self.sensor_rng = np.random.default_rng(None if seed is None else seed+90000)
        self.t = 0
        self.completed = 0
        self.vehicles=[]
        for i in range(self.n):
            lane = i % len(self.road.lanes)
            rank = i // len(self.road.lanes)
            maxrank=(self.n-1)//len(self.road.lanes)
            spacing=min(13,45/max(1,maxrank))
            jitter=3 if spacing==13 else min(.5,spacing*.1)
            s = max(0, 45 - rank*spacing + self.rng.uniform(-jitter,jitter))
            self.vehicles.append(Vehicle(i,lane,s,float(self.rng.uniform(3,7)),i<self.cfg.get('connected',4),
                float(self.rng.uniform(8,12)*self.cfg.get('behavior_scale',1)),float(self.rng.uniform(.7,1.3)),
                float(self.rng.uniform(.9,1.7)),route=(lane,)))
        self.observation_age=np.zeros(self.n)
        self.lane_history=[]
        self.last_components=np.zeros(6,np.float32)
        return sense(self),{}

    def clone(self):
        return copy.deepcopy(self)

    def state_array(self):
        result=np.zeros((self.n,6),np.float32)
        for i,v in enumerate(self.vehicles):
            p,d=self.road.lanes[v.lane].pose(v.s)
            result[i]=[* (p/100),v.speed/15,*d,float(v.active)]
        return result

    def lane_targets(self):
        return np.array([[sum(v.active and v.lane==l.id for v in self.vehicles)/10,
                          sum(v.active and v.lane==l.id and v.speed<2 for v in self.vehicles)/10]
                         for l in self.road.lanes],np.float32)

    def step(self,action):
        action=np.asarray(action)
        if action.shape!=(self.n,) or not np.isin(action,[-1,0,1]).all():
            raise ValueError('Expected one categorical advisory (-1,0,1) per vehicle')
        before=self.state_array()
        acc=[]
        for i,v in enumerate(self.vehicles):
            if not v.active: acc.append(0); continue
            ahead=[w for w in self.vehicles if w.active and w.lane==v.lane and w.s>v.s]
            lead=min(ahead,key=lambda w:w.s) if ahead else None
            gap=lead.s-v.s-4 if lead else 1000
            p=before[i,:2]*100
            yielding=False
            for j,w in enumerate(self.vehicles):
                if j==i or not w.active or w.lane==v.lane: continue
                q=before[j,:2]*100
                for node in self.road.intersections:
                    if np.linalg.norm(p-node)<22 and np.linalg.norm(q-node)<18 and (w.id<v.id or w.connected):
                        yielding=True
            a=desired_acceleration(v,gap,lead.speed if lead else v.speed,action[i],yielding)
            if self.cfg.get('stopped_vehicle') and i==self.n-1: a=-4
            if self.cfg.get('blocked_lane') and v.lane==0 and v.s>45: a=-4
            acc.append(a)
        progress=0
        for v,a in zip(self.vehicles,acc):
            if not v.active: continue
            old=v.s
            integrate(v,a,self.dt)
            progress+=v.s-old
            if v.s>=self.road.lanes[v.lane].length:
                v.active=False
                self.completed+=1
            if not v.active and self.cfg.get('respawn',False):
                others=[w for w in self.vehicles if w.active and w.lane==v.lane]
                if all(w.s>10 for w in others):
                    v.s=0; v.speed=3; v.active=True; v.generation+=1
        after=self.state_array()
        collisions=conflicts=0
        for i in range(self.n):
            for j in range(i):
                if after[i,5]*after[j,5]<.5: continue
                dist=np.linalg.norm((after[i,:2]-after[j,:2])*100)
                collisions+=int(dist<3)
                conflicts+=int(dist<7)
        active=[v for v in self.vehicles if v.active]
        delay=sum(max(0,1-v.speed/v.desired_speed) for v in active)/self.n
        self.last_components=np.array([delay,sum(v.speed<2 for v in active)/self.n,
            progress/(self.n*15*self.dt),collisions/self.n,conflicts/self.n,
            sum(v.acceleration**2 for v in active)/(self.n*16)],np.float32)
        self.t+=1
        reward=-float(self.last_components@np.array(self.cfg.get('objective_weights',[1,.5,-.8,12,2,.04])))
        info={'components':self.last_components.copy(),'completed':self.completed,
              'collisions':collisions,'conflicts':conflicts,'state':after,'lane_targets':self.lane_targets()}
        return sense(self),reward,not any(v.active for v in self.vehicles),self.t>=self.cfg.get('episode_steps',48),info
