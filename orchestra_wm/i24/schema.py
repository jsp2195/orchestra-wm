from dataclasses import dataclass, field
from typing import Literal
import numpy as np

SCHEMA_VERSION='roadside-scene/1.0'
FEATURES=('s_m','d_m','vs_mps','vd_mps','as_mps2','ad_mps2','length_m','width_m')

@dataclass(frozen=True)
class RoadMap:
    s_min: float
    s_max: float
    lane_centers: tuple[float,...]
    lane_width: float
    bins: int=8
    confidence: float=.5
    coordinate_frame: str='straightened curvilinear roadway metres; not geographic XY'
    def validate(self):
        assert self.s_max>self.s_min and self.bins>0 and len(self.lane_centers)>0
        assert self.lane_width>0 and 0<=self.confidence<=1
    @property
    def field_count(self):return self.bins*len(self.lane_centers)
    def tokens(self):
        x=np.linspace(self.s_min,self.s_max,self.bins+1);s=(x[1:]+x[:-1])/2
        return np.array([[v,d] for d in self.lane_centers for v in s],np.float32)

@dataclass
class RoadsideScene:
    scene_id: str
    session_id: str
    provenance: dict
    time: np.ndarray
    track_ids: tuple[str,...]
    values: np.ndarray  # [T,N,8], SI units
    existence: np.ndarray
    detection: np.ndarray
    confidence: np.ndarray
    road: RoadMap
    vehicle_class: np.ndarray
    source_ids: tuple[str,...]|None=None
    schema_version: str=SCHEMA_VERSION
    def validate(self):
        self.road.validate();t,n=len(self.time),len(self.track_ids)
        assert self.schema_version==SCHEMA_VERSION and t>1 and n>0
        assert len(set(self.track_ids))==n and np.all(np.diff(self.time)>0)
        assert self.values.shape==(t,n,8) and np.isfinite(self.values).all()
        assert self.existence.shape==self.detection.shape==self.confidence.shape==(t,n)
        assert not np.any(self.detection & ~self.existence)
        assert np.all((self.confidence>=0)&(self.confidence<=1))
        assert self.provenance['kind'] in ['REAL_I24','SYNTHETIC_FIXTURE','PARTNER_AUTHORIZED']
        assert self.provenance.get('source_hash') and self.provenance.get('kinematics')
        assert np.all(self.values[self.existence,6:8]>0)
        return self

@dataclass(frozen=True)
class Observation:
    values: np.ndarray
    visible: np.ndarray
    track_ids: tuple[str,...]
    confidence: np.ndarray
    fields: np.ndarray
    field_support: np.ndarray
    sensing: str='EMULATED roadside sensing'

@dataclass(frozen=True)
class PassiveAction:
    commands: np.ndarray

@dataclass(frozen=True)
class KnownFutureContext:
    road: RoadMap  # fixed, known map only; no realized inflows/incidents

class CavnueAdapter:
    """Partner schema validator only: no assumed Cavnue product/protocol compatibility."""
    def convert(self,*args,**kwargs):
        raise NotImplementedError('Requires authorized partner schema and calibrated track stream')
    def validate(self,scene):
        scene.validate()
        if scene.provenance['kind']!='PARTNER_AUTHORIZED':raise ValueError('No fabricated partner provenance')
        for key in ['permission_reference','clock_reference','coordinate_calibration','coverage_description']:
            if not scene.provenance.get(key):raise ValueError('Missing partner requirement: '+key)
        return scene
