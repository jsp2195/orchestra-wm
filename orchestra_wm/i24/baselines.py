"""Conventional IDM comparator, calibrated on training observations only when eligible."""
import numpy as np
from scipy.optimize import least_squares
from orchestra_wm.i24.evaluation import kinematic_forecast
from orchestra_wm.i24.data import fields_numpy


def idm_acceleration(speed,relative_speed,gap,parameters):
    desired,timegap,accel,braking,min_gap=parameters
    star=min_gap+np.maximum(0,speed*timegap+speed*relative_speed/(2*np.sqrt(accel*braking)))
    return accel*(1-(np.maximum(speed,0)/desired)**4-(star/np.maximum(gap,.5))**2)


def leader_pairs(values,valid,lane_width):
    rows=[]
    for i in np.flatnonzero(valid):
        delta=values[:,0]-values[i,0];eligible=valid&(delta>0)&(np.abs(values[:,1]-values[i,1])<lane_width*.4)
        if eligible.any():
            candidates=np.flatnonzero(eligible);j=candidates[np.argmin(delta[candidates])];gap=delta[j]-(values[i,6]+values[j,6])/2
            if gap>1:rows.append((i,j,gap))
    return rows


def calibrate_idm(training_scenes):
    if not training_scenes or any(s.provenance['kind']!='REAL_I24' for s in training_scenes):return None,{'status':'NOT_APPLICABLE','reason':'Requires authentic training-set leader/follower spans; no real records available.'}
    x=[];y=[]
    for scene in training_scenes:
        for t in range(2,len(scene.time),5):
            a=scene.values[t]
            for i,j,gap in leader_pairs(a,scene.existence[t],scene.road.lane_width):
                if abs(a[i,3])<.2 and abs(a[j,3])<.2 and 2<gap<100 and abs(a[i,4])<8:
                    x.append([a[i,2],a[i,2]-a[j,2],gap]);y.append(a[i,4])
                    if len(x)>=10000:break
            if len(x)>=10000:break
        if len(x)>=10000:break
    if len(x)<100:return None,{'status':'NOT_APPLICABLE','reason':'Fewer than 100 eligible training following samples','eligible_samples':len(x)}
    x=np.array(x);y=np.array(y);fit=least_squares(lambda p:idm_acceleration(x[:,0],x[:,1],x[:,2],p)-y,[30,1.4,1.2,2.,2.],bounds=([8,.4,.2,.3,.5],[45,3,4,6,6]),loss='soft_l1',max_nfev=200)
    return fit.x,{'status':'CALIBRATED','parameters':fit.x.tolist(),'eligible_samples':len(x),'training_acceleration_rmse':float(np.sqrt(np.mean(fit.fun**2))),'success':bool(fit.success),'warning':'Conditional curve calibration only; no causal or driver-identity inference'}


def idm_forecast(obs,road,steps,dt,parameters):
    initial=kinematic_forecast(obs,road,1,0,'constant_position')['agents'][0,0];x=initial.copy();rows=[];masks=[]
    for _ in range(steps):
        valid=(x[:,0]>=road.s_min)&(x[:,0]<road.s_max);acc=idm_acceleration(x[:,2],np.zeros(len(x)),np.full(len(x),1e6),parameters)
        for i,j,gap in leader_pairs(x,valid,road.lane_width):acc[i]=idm_acceleration(x[i,2],x[i,2]-x[j,2],gap,parameters)
        # Conventional IDM numerical stabilization is explicit, not applied to learned forecasts.
        acc=np.clip(acc,-8,4);x[:,0]+=x[:,2]*dt+.5*acc*dt**2;x[:,2]=np.maximum(0,x[:,2]+acc*dt);x[:,4]=acc;rows.append(x.copy());masks.append(valid)
    a=np.array(rows);valid=np.array(masks);fields,_=fields_numpy(a,valid,road)
    return {'agents':a[None],'fields':fields[None],'derived_fields':fields[None],'valid':valid[None]}
