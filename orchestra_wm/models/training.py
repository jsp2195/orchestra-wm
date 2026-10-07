from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from orchestra_wm.models.world_model import WorldModel
from orchestra_wm.data.dataset import TrajectoryDataset
from orchestra_wm.utils.seed import device_for,seed_everything
from orchestra_wm.evaluation.common import write_json

VARIANTS=['independent','no_actions','privileged','memoryless','orchestra']

def rollout_loss(model,b,cfg,device):
    state=model.initial_state(len(b['obs']),b['obs'].shape[2],b['lanes'].shape[2],device)
    for t in range(cfg['context']):
        obs=b['obs'][:,t].float();mask=b['mask'][:,t]
        if model.variant=='privileged':
            obs=obs.clone();obs[:,:,:6]=b['states'][:,t];mask=torch.ones_like(mask)
        state=model.observe(state,{'agents':obs,'lanes':b['lanes'][:,t].float()},mask,b['actions'][:,t-1] if t else None)
    loss=0;start=cfg['context']-1
    for h in range(cfg['train_horizon']):
        state,pred=model.imagine_step(state,b['actions'][:,start+h])
        truth=b['states'][:,start+h+1];active=truth[:,:,5:6]
        state_loss=(((pred['agents']-truth)**2)*torch.tensor([12,12,2,.2,.2,.2],device=device)*active).sum()/active.sum().clamp_min(1)
        lane_loss=((pred['lanes']-b['lane_targets'][:,start+h+1])**2).mean()
        cost_loss=((pred['components']-b['components'][:,start+h])**2*torch.tensor([1,1,1,8,3,.2],device=device)).mean()
        # Future observations ONLY supply a detached consistency target.
        with torch.no_grad(): target=model.agent_token(b['obs'][:,start+h+1].float())
        consistency=((state.memory-target)**2*b['mask'][:,start+h+1,:,None]).mean()
        loss+=state_loss+.5*lane_loss+cost_loss+.001*consistency
    return loss/cfg['train_horizon']

def train(cfg,data_path,out,variants=VARIANTS):
    out=Path(out);(out/'checkpoints').mkdir(exist_ok=True,parents=True)
    device=device_for(cfg.get('device','auto'));data=TrajectoryDataset(data_path)
    rows=[];metadata={}
    for variant in variants:
        seed_everything(cfg['seed'],cfg['threads']);started=time.perf_counter()
        model=WorldModel(cfg['model_dim'],cfg['layers'],cfg['dt'],variant).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=1e-4)
        rng=np.random.default_rng(cfg['seed']);writer=SummaryWriter(str(out/'tensorboard'/variant))
        for epoch in range(cfg['epochs']):
            model.train();total=0
            for step in range(cfg['train_batches']):
                b=data.sample(cfg['batch_size'],cfg['context'],cfg['train_horizon'],rng,device)
                loss=rollout_loss(model,b,cfg,device)
                optimizer.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();total+=float(loss.detach())
            model.eval();validation=[];vrng=np.random.default_rng(12345)
            with torch.no_grad():
                for _ in range(4):
                    b=data.sample(cfg['batch_size'],cfg['context'],cfg['train_horizon'],vrng,device,validation=True)
                    validation.append(float(rollout_loss(model,b,cfg,device)))
            avg=total/cfg['train_batches'];val=float(np.mean(validation))
            rows.append({'variant':variant,'epoch':epoch+1,'loss':avg,'validation_loss':val,'wall_seconds':time.perf_counter()-started})
            writer.add_scalar('train/loss',avg,epoch+1);writer.add_scalar('validation/loss',val,epoch+1)
            print(f'{variant}: epoch {epoch+1}/{cfg["epochs"]} train={avg:.5f} validation={val:.5f}',flush=True)
        writer.close()
        parameters=sum(p.numel() for p in model.parameters())
        torch.save({'model':model.state_dict(),'config':cfg,'variant':variant,'parameters':parameters},out/'checkpoints'/f'{variant}.pt')
        metadata[variant]={'parameters':parameters,'steps':cfg['epochs']*cfg['train_batches'],'seconds':time.perf_counter()-started,
            'final_train_loss':avg,'final_validation_loss':val,'training_seed':cfg['seed'],'device':str(device)}
        pd.DataFrame(rows).to_csv(out/'training.csv',index=False);write_json(out/'training_metadata.json',metadata)
    return metadata

def load_model(path,device='cpu'):
    saved=torch.load(path,map_location=device,weights_only=False)
    c=saved['config'];model=WorldModel(c['model_dim'],c['layers'],c['dt'],saved['variant']).to(device)
    model.load_state_dict(saved['model']);model.eval()
    return model
