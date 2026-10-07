import torch

def trivial_forecast(agents,horizon,dt=.5,constant_velocity=False):
    values=[]
    for step in range(1,horizon+1):
        value=agents.clone()
        if constant_velocity:
            value[:,:,:2]+=agents[:,:,3:5]*agents[:,:,2:3]*(15*dt*step/100)
        values.append(value)
    return torch.stack(values,1)
