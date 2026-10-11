"""Additional audit metrics; never rewrite historical evaluation tables."""
import numpy as np


def marginal_crps(samples, target):
    """Unbiased ensemble estimator of marginal CRPS, one value per target."""
    samples=np.asarray(samples,float);target=np.asarray(target,float)
    first=np.abs(samples-target).mean(0)
    n=len(samples)
    if n<2:return first
    return first-.5*np.abs(samples[:,None]-samples[None]).sum((0,1))/(n*(n-1))


def sample_diversity(samples):
    samples=np.asarray(samples,float).reshape(len(samples),-1)
    n=len(samples)
    if n<2 or not samples.shape[1]:return 0.
    return float(np.linalg.norm(samples[:,None]-samples[None],axis=-1).sum()/(n*(n-1)*np.sqrt(samples.shape[1])))


def boundary_rates(agents, valid, road):
    """Longitudinal exits are separate from lateral road departures."""
    use=np.broadcast_to(valid,agents.shape[:-1])
    if not use.any():return dict(longitudinal_exit_rate=None,lateral_departure_rate=None,reverse_speed_rate=None)
    low=min(road.lane_centers)-road.lane_width/2;high=max(road.lane_centers)+road.lane_width/2
    return dict(longitudinal_exit_rate=float(((agents[...,0]<road.s_min)|(agents[...,0]>=road.s_max))[use].mean()),
                lateral_departure_rate=float(((agents[...,1]<low)|(agents[...,1]>=high))[use].mean()),
                reverse_speed_rate=float((agents[...,2]<0)[use].mean()))
