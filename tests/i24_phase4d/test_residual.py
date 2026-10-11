import numpy as np
import torch
from orchestra_wm.i24_phase4d.model import ResidualFields,energy
from orchestra_wm.i24_phase4d.data import features,adjacency
from orchestra_wm.i24.schema import RoadMap,Observation
from orchestra_wm.i24_continuous.evaluation import energy_score

def batch():
    torch.manual_seed(12);k=8
    return dict(aggregate=torch.randn(2,25,k,15),micro=torch.randn(2,25,k,8),reference=torch.rand(2,k,3)*20+1,positions=torch.rand(2,k,2),adjacency=torch.ones(2,k,k)/k)

def test_information_restriction_and_identical_path():
    torch.set_num_threads(2);m=ResidualFields();b=batch();b['micro'].zero_()
    a=m(b,'aggregate',2,13,horizon=5);f=m(b,'full',2,13,horizon=5);assert torch.equal(a,f)
    b['micro'].normal_();assert torch.equal(a,m(b,'aggregate',2,13,horizon=5))
    assert not torch.equal(a,m(b,'full',2,13,horizon=5))

def test_reference_and_physical_identity():
    m=ResidualFields();b=batch()
    with torch.no_grad():m.head.weight[:2].zero_();m.head.bias[:2].zero_()
    x=m(b,'full',3,0,horizon=10,ablation='deterministic')
    assert torch.allclose(x[...,0],b['reference'][None,:,None,:,0].expand_as(x[...,0]))
    assert (x>0).all() and torch.equal(x[...,2],3.6*x[...,0]*x[...,1])

def test_energy_is_unbiased_and_masked():
    rng=np.random.default_rng(51);x=rng.normal(size=(4,2,3,3)).astype('float32');y=rng.normal(size=(2,3,3)).astype('float32');mask=np.array([[1,1,0],[0,1,1]],bool)
    expected=np.mean([energy_score(x[:,i,mask[i]]/np.array([30,100,3000]),y[i,mask[i]]/np.array([30,100,3000])) for i in range(2)])
    assert abs(float(energy(torch.tensor(x),torch.tensor(y),torch.tensor(mask)))-expected)<1e-6
    y[~mask]=1e8;assert abs(float(energy(torch.tensor(x),torch.tensor(y),torch.tensor(mask)))-expected)<1e-6

def test_outage_reference_uses_supported_past_only():
    road=RoadMap(0,100,(2.,6.),4.,4);v=np.array([[10,2,8,0,0,0,5,2]],np.float32);mask=np.array([True]);field=np.ones((8,3),np.float32);field[:,0]=12
    first=Observation(v,mask,('a',),mask.astype(float),field,np.ones(8,bool))
    hidden=Observation(np.zeros_like(v),~mask,('a',),np.zeros(1),np.zeros_like(field),np.zeros(8,bool))
    f=features([first]+[hidden]*24,road)
    assert np.array_equal(f['reference'],field)
    assert np.isfinite(f['aggregate']).all() and np.isfinite(adjacency(road)).all()

def test_future_labels_cannot_change_inference():
    m=ResidualFields();b=batch();x=m(b,'full',2,5,horizon=4)
    b['target']=torch.full((2,100,8,3),float('nan'));b['support']=torch.zeros(2,100,8,dtype=torch.bool)
    assert torch.equal(x,m(b,'full',2,5,horizon=4))

def test_micro_independence_and_future_exclusion():
    from orchestra_wm.i24_phase4d.micro import KinematicResidual
    torch.manual_seed(7);m=KinematicResidual();b=dict(history=torch.randn(25,3,8),visible=torch.ones(25,3,dtype=torch.bool),initial=torch.randn(3,8),adjacency=(torch.ones(3,3)-torch.eye(3))/2)
    first=m(b,'independent',3,91)[:,:,0].clone();graph=m(b,'graph',3,91)[:,:,0].clone()
    b['history'][:,1:]*=10;b['initial'][1:]*=20;b['target']=torch.full((100,3,8),float('nan'));b['support']=torch.zeros(100,3,dtype=torch.bool)
    assert torch.equal(first,m(b,'independent',3,91)[:,:,0])
    assert not torch.equal(graph,m(b,'graph',3,91)[:,:,0])
