from pathlib import Path
import time
import hashlib
import numpy as np
import pandas as pd
import torch
from orchestra_wm.phase2.data import SiblingDataset
from orchestra_wm.data.dataset import TrajectoryDataset
from orchestra_wm.phase2.model import create_model
from orchestra_wm.evaluation.common import write_json
from orchestra_wm.utils.seed import seed_everything,device_for


def loss_and_predictions(model,b,cfg):
    device=b['obs'].device;state=model.initial_state(len(b['obs']),b['obs'].shape[2],b['lanes'].shape[2],device)
    for t in range(cfg['context']):
        obs=b['obs'][:,t].float();mask=b['mask'][:,t]
        if model.variant=='privileged':obs=obs.clone();obs[:,:,:6]=b['states'][:,t];mask=torch.ones_like(mask)
        state=model.observe(state,{'agents':obs,'lanes':b['lanes'][:,t].float()},mask,b['actions'][:,t-1] if t else None)
    start=cfg['context']-1;loss=0;predictions=[];costs=[]
    for h in range(cfg['train_horizon']):
        state,p=model.imagine_step(state,b['actions'][:,start+h]);truth=b['states'][:,start+h+1];active=truth[:,:,5:6]
        sl=(((p['agents']-truth)**2)*torch.tensor([12,12,2,.2,.2,.2],device=device)*active).sum()/active.sum().clamp_min(1)
        ll=((p['lanes']-b['lane_targets'][:,start+h+1])**2).mean()
        cl=((p['components']-b['components'][:,start+h])**2*torch.tensor([1,1,1,8,3,.2],device=device)).mean()
        with torch.no_grad():target=model.agent_token(b['obs'][:,start+h+1].float())
        consistency=((state.memory-target)**2*b['mask'][:,start+h+1,:,None]).mean()
        loss+=sl+.5*ll+cl+.001*consistency
        predictions.append(p['agents']);costs.append(p['components'])
    return loss/cfg['train_horizon'],torch.stack(predictions,1),torch.stack(costs,1)


def counterfactual_loss(pred,cost,b,cfg):
    h=cfg['train_horizon'];t=cfg['context'];g=len(pred)//9
    # Subtract all-sibling means: equivalent to mean pairwise difference loss up to a fixed factor.
    p=pred[:,:,:2,:3].reshape(g,9,h,2,3);truth=b['states'][:,t:t+h,:2,:3].reshape_as(p)
    delta=(p-p.mean(1,keepdim=True))-(truth-truth.mean(1,keepdim=True))
    state_loss=(delta.square()*torch.tensor([12,12,2],device=p.device)).mean()
    cp=cost.reshape(g,9,h,6);ct=b['components'][:,t-1:t+h-1].reshape_as(cp)
    difference=(cp-cp.mean(1,keepdim=True))-(ct-ct.mean(1,keepdim=True))
    return state_loss+(difference.square()*torch.tensor([1,1,1,8,3,.2],device=p.device)).mean()


def train_phase2(cfg,out,explicit=False):
    natural=TrajectoryDataset('outputs/smoke/data.npz');siblings=SiblingDataset(out/'dataset_v2_factorial/data.npz')
    names=['A_original','B_factorial','C_difference','independent','privileged']+(['D_pairwise'] if explicit else [])
    signature=hashlib.sha256(b''.join(Path(p).read_bytes() for p in ['orchestra_wm/phase2/training.py','orchestra_wm/phase2/model.py','orchestra_wm/models/world_model.py'])+(out/'dataset_v2_factorial/manifest.json').read_bytes()).hexdigest()
    rows=[];metadata=[];device=device_for(cfg['device']);checkpoint_dir=out/'checkpoints';checkpoint_dir.mkdir(exist_ok=True)
    for seed in cfg['training_seeds']:
        for name in names:
            checkpoint=checkpoint_dir/f'{name}_{seed}.pt'
            if checkpoint.exists():
                stored=torch.load(checkpoint,map_location='cpu',weights_only=False)
                if stored['config']!=cfg or stored.get('training_source_sha256')!=signature:raise RuntimeError('Checkpoint config differs; use a new Phase-2 output directory')
                rows.extend(stored['training_rows']);metadata.append(stored['metadata']);print('Reuse trained',name,seed,flush=True);continue
            seed_everything(seed,cfg['threads']);started=time.perf_counter();rng=np.random.default_rng(seed)
            model=create_model(cfg,name).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=1e-4)
            losses=[];cfvalues=[];modelrows=[]
            for step in range(1,cfg['training_steps']+1):
                # Identical natural/factorial schedules and RNG draws across B/C/D and controls.
                use_natural=(step%4==0) or name=='A_original'
                if use_natural:b=natural.sample(cfg['batch_size'],cfg['context'],cfg['train_horizon'],rng,device)
                else:b=siblings.sample(cfg['batch_size'],rng,device)
                model.train();loss,p,c=loss_and_predictions(model,b,cfg);cf=loss.new_zeros(())
                if not use_natural and name in ['C_difference','D_pairwise']:
                    cf=counterfactual_loss(p,c,b,cfg);loss=loss+cfg['cf_weight']*cf
                optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
                losses.append(float(loss.detach()));cfvalues.append(float(cf.detach()))
                if step%cfg['validation_interval']==0:
                    model.eval();v=[];vrng=np.random.default_rng(991)
                    with torch.no_grad():
                        for _ in range(3):v.append(float(loss_and_predictions(model,siblings.sample(cfg['batch_size'],vrng,device,True),cfg)[0]))
                        ordinary=float(loss_and_predictions(model,natural.sample(cfg['batch_size'],cfg['context'],cfg['train_horizon'],vrng,device,True),cfg)[0])
                    row={'seed':seed,'model':name,'step':step,'train_loss':float(np.mean(losses)),
                         'cf_loss':float(np.mean(cfvalues)),'factorial_validation_loss':float(np.mean(v)),'natural_validation_loss':ordinary,'seconds':time.perf_counter()-started}
                    rows.append(row);modelrows.append(row);losses=[];cfvalues=[]
                    print(f'{name} seed {seed} step {step}: validation {row["factorial_validation_loss"]:.4f}',flush=True)
            meta={'seed':seed,'model':name,'parameters':sum(p.numel() for p in model.parameters()),'training_steps':cfg['training_steps'],
                  'seconds':time.perf_counter()-started,'device':str(device),'final_factorial_validation_loss':row['factorial_validation_loss']}
            metadata.append(meta)
            torch.save({'model':model.cpu().state_dict(),'training_source_sha256':signature,'config':cfg,'name':name,'seed':seed,'training_rows':modelrows,'metadata':meta},checkpoint)
            pd.DataFrame(rows).to_csv(out/'training.csv',index=False);write_json(out/'training_metadata.json',metadata)
    pd.DataFrame(rows).to_csv(out/'training.csv',index=False);write_json(out/'training_metadata.json',metadata)
    return names
