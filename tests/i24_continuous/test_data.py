import numpy as np
from orchestra_wm.i24.schema import RoadMap,RoadsideScene
from orchestra_wm.i24_continuous.data import exposure_fields,observations


def test_exposure_uses_past_vehicle_time_and_distance_and_unknown_bins():
    road=RoadMap(0,100,(5.,),4.,2,1.)
    values=np.zeros((10,1,8),np.float32);values[:,:,0]=10;values[:,:,1]=5;values[:,:,2]=10
    visible=np.ones((10,1),bool);visible[5:]=False
    fields,support=exposure_fields(values,visible,road)
    assert np.allclose(fields[4,0],[10,20,720])
    assert np.allclose(fields[5,0],[10,16,576])
    assert support[8,0] and not support[9,0]
    assert not support[:,1].any()
    future=values.copy();future[6:,:,2]=1000
    changed,_=exposure_fields(future,visible,road)
    assert np.array_equal(changed[:6],fields[:6])


def test_observations_never_expose_future_roster():
    road=RoadMap(0,100,(5.,),4.,2,1.)
    values=np.zeros((30,2,8),np.float32);values[:,:,0]=10;values[:,:,1]=5;values[:,:,2]=10;values[:,:,6:]=[5,2]
    visible=np.ones((30,2),bool);visible[:25,1]=False
    scene=RoadsideScene('unit','UNIT_FIXTURE',dict(kind='SYNTHETIC_FIXTURE',source_hash='unit',kinematics='unit'),
                       np.arange(30)*.2,('past','future'),values,visible,visible,visible.astype(float),road,np.zeros(2,int)).validate()
    obs,cohort=observations(scene,25)
    assert cohort.tolist()==[0] and obs[0].track_ids==('past',)
    assert obs[0].values.shape==(1,8)
