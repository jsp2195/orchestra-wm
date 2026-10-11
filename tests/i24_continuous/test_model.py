import numpy as np
import torch
from orchestra_wm.i24.schema import RoadMap
from orchestra_wm.i24_continuous.model import ContinuousWorldModel,temporal_fields


def test_temporal_exposure_retains_past_occupancy_and_batch_repeat():
    road=RoadMap(0,100,(5.,),4.,1,1.)
    values=torch.zeros(1,1,8);values[:,:,0]=50;values[:,:,1]=5;values[:,:,2]=10
    visible=torch.ones(1,1,dtype=torch.bool);missing=torch.zeros_like(visible)
    history=((values,visible),)*4+((values,missing),)
    assert torch.allclose(temporal_fields(history,road),torch.tensor([[[10.,8.,288.]]]))
    model=ContinuousWorldModel(8,2,'full',.2)
    state=model.initial_state({'road':road,'track_ids':('unit',)})
    batch=dict(values=values,confidence=visible.float(),fields=torch.tensor([[[10.,10.,360.]]]),field_support=visible)
    state=model.observe(state,batch,visible,.2)
    repeated=state.repeat(3)
    assert repeated.exposure_history[0][0].shape==(3,1,8)
    result,p=model.imagine_step(repeated,generator=torch.Generator().manual_seed(1))
    assert p['derived_fields'].shape==(3,1,3) and torch.isfinite(p['derived_fields']).all()
    assert len(result.exposure_history)==2


def test_autonomous_training_objective_backpropagates_without_future_inputs():
    from orchestra_wm.i24.schema import RoadsideScene
    from orchestra_wm.i24_continuous.data import exposure_fields
    from orchestra_wm.i24_continuous.training import objective
    torch.set_num_threads(2)
    road=RoadMap(0,1000,(5.,),4.,2,1.)
    t=np.arange(126)*.2;values=np.zeros((126,2,8),np.float32)
    values[:,:,0]=t[:,None]*10+np.array([20,40]);values[:,:,1]=5;values[:,:,2]=10;values[:,:,6:]=[5,2]
    visible=np.ones((126,2),bool)
    scene=RoadsideScene('unit','UNIT_FIXTURE',dict(kind='SYNTHETIC_FIXTURE',source_hash='unit',kinematics='unit'),
                       t,('a','b'),values,visible,visible,visible.astype(float),road,np.zeros(2,int)).validate()
    fields,support=exposure_fields(values,visible,road)
    model=ContinuousWorldModel(8,2,'full',.2)
    value=objective(model,scene,fields,support,np.random.default_rng(101),True)
    assert torch.isfinite(value)
    value.backward()
    assert torch.isfinite(model.field_decoder.weight.grad).all()
    assert model.acceleration.weight.grad.abs().sum()>0
