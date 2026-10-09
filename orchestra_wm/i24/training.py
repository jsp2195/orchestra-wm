from pathlib import Path
import time,os,json,hashlib
import numpy as np
import torch
from torch.nn import functional as F
from orchestra_wm.i24.data import observations,fields_numpy,write_json,sha
from orchestra_wm.i24.model import RoadsideWorldModel,observe_history


def atomic_checkpoint(path,payload):
    path=Path(path);temp=path.with_suffix('.tmp');torch.save(payload,temp);os.replace(temp,path)

def load_checkpoint(path):
    payload=torch.load(path,map_location='cpu',weights_only=False);c=payload['config'];m=RoadsideWorldModel(c['model_dim'],c['latent_dim'],payload['variant'],c['dt']);m.load_state_dict(payload['model']);m.eval();return m,payload


def loss(model,scene,cfg,rng,validation=False):
    context=round(cfg['context_seconds']/cfg['dt']);horizon=cfg['train_horizon_steps']
    regime='dense' if validation else ['dense','dense','sparse50','outage'][int(rng.integers(4))]
    obs,cohort=observations(scene,context,regime,int(rng.integers(100000)));state=observe_history(model,obs,scene.road)
    targets=torch.tensor(scene.values[context:context+horizon,cohort],dtype=torch.float32)[None];valid=torch.tensor(scene.existence[context:context+horizon,cohort])[None]
    field,_=fields_numpy(scene.values[context:context+horizon],scene.existence[context:context+horizon],scene.road);field=torch.tensor(field)[None]
    g=torch.Generator().manual_seed(int(rng.integers(2**30)));total=state.agents.new_zeros(());components=[]
    for t in range(horizon):
        previous=state
        # Prior transition is the only recurrent state used in subsequent rollout steps.
        state,p=model.imagine_step(previous,generator=g)
        mask=valid[:,t,:,None];denom=mask.sum().clamp_min(1)
        micro=(((p['agents'][:,:,:4]-targets[:,t,:,:4])/model.scale[:4])**2*mask).sum()/denom
        macro=F.mse_loss(p['fields']/model.field_scale,field[:,t]/model.field_scale)
        consistency=F.mse_loss(p['fields']/model.field_scale,p['derived_fields']/model.field_scale)
        # Training-only posterior sees target innovation for an ELBO reconstruction term.
        _,q=model._transition(previous,generator=g,posterior_target=targets[:,t])
        observed_acc=(targets[:,t,:,2:4]-previous.agents[:,:,2:4])/model.dt
        nll=((.5*((observed_acc-q['acc_mean'])/q['acc_std']).square()+q['acc_std'].log())*mask).sum()/denom
        field_nll=(.5*((field[:,t]-q['field_mean'])/q['field_std']).square()+q['field_std'].log()).mean()
        stochastic=(.001*nll+.0001*field_nll+.001*q['kl']) if model.variant!='deterministic' else total.new_zeros(())
        if model.variant=='macro_only':micro=micro*0;stochastic=.0001*field_nll+.001*q['kl']
        if model.variant in ['independent','micro_only']:macro=macro*0;consistency=consistency*0
        if model.variant=='no_crossscale':consistency=consistency*0
        value=micro+macro+.05*consistency+stochastic
        total+=value;components.append([float(micro.detach()),float(macro.detach()),float(q['kl'].detach())])
    return total/horizon,np.mean(components,axis=0)


def train_models(cfg,splits,out,fingerprint,started):
    directory=out/'checkpoints';directory.mkdir(exist_ok=True);records=[];metadata=[]
    for seed in cfg['training_seeds']:
        for variant in cfg['variants']:
            path=directory/f'{variant}-{seed}.pt';torch.manual_seed(seed);rng=np.random.default_rng(seed)
            model=RoadsideWorldModel(cfg['model_dim'],cfg['latent_dim'],variant,cfg['dt']);optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate']);rows=[];step0=0;elapsed=0
            if path.exists():
                saved=torch.load(path,map_location='cpu',weights_only=False)
                if saved['fingerprint']!=fingerprint:raise ValueError('Checkpoint fingerprint mismatch; use a new namespaced output')
                model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);step0=saved['step'];rows=saved['rows'];elapsed=saved['elapsed'];rng.bit_generator.state=saved['numpy_rng'];torch.set_rng_state(saved['torch_rng'])
            begin=time.perf_counter()
            for step in range(step0+1,cfg['training_steps']+1):
                if time.perf_counter()-started>cfg['max_wall_seconds']:raise TimeoutError('Campaign wall budget reached; rerun to resume atomic checkpoints')
                model.train();scene=splits['train'][int(rng.integers(len(splits['train'])))];optimizer.zero_grad();objective,parts=loss(model,scene,cfg,rng);assert torch.isfinite(objective)
                objective.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
                if step==1 or step%20==0 or step==cfg['training_steps']:
                    model.eval();vrng=np.random.default_rng(991)
                    with torch.no_grad():vl=float(np.mean([float(loss(model,s,cfg,vrng,True)[0]) for s in splits['validation'][:3]]))
                    row={'seed':seed,'variant':variant,'step':step,'train_loss':float(objective.detach()),'validation_loss':vl,'micro_loss':float(parts[0]),'macro_loss':float(parts[1]),'kl':float(parts[2])};rows.append(row)
                    atomic_checkpoint(path,{'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,'config':cfg,'variant':variant,'seed':seed,'fingerprint':fingerprint,'source_kind':cfg['source_kind'],'rows':rows,'numpy_rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),'elapsed':elapsed+time.perf_counter()-begin})
                    print(f'{variant} seed={seed} step={step} train={objective.item():.4f} validation={vl:.4f}',flush=True)
            saved=torch.load(path,map_location='cpu',weights_only=False);records.extend(rows)
            metadata.append({'variant':variant,'seed':seed,'parameters':sum(p.numel() for p in model.parameters()),'updates':saved['step'],'sampled_scene_passes':saved['step']/len(splits['train']),'seconds':saved['elapsed'],'checkpoint_sha256':sha(path),'source_kind':cfg['source_kind'],'final_validation_loss':rows[-1]['validation_loss']})
    import pandas as pd
    pd.DataFrame(records).to_csv(out/'training.csv',index=False);write_json(out/'training_metadata.json',metadata);return metadata
