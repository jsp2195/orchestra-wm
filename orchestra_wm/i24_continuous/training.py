"""Fixed-budget authentic continuous training; prior-only autonomous recurrence."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from orchestra_wm.i24.data import load_scene,sha
from orchestra_wm.i24.model import observe_history
from orchestra_wm.i24.training import atomic_checkpoint
from .data import observations
from .model import ContinuousWorldModel
from .source import write_immutable


def objective(model,scene,fields,support,rng,validation=False):
    regime='dense' if validation else ['dense','dense','sparse50','outage'][int(rng.integers(4))]
    obs,cohort=observations(scene,25,regime,int(rng.integers(100000)))
    state=observe_history(model,obs,scene.road)
    targets=torch.as_tensor(scene.values[25:75,cohort],dtype=torch.float32)[None]
    masks=torch.as_tensor(scene.existence[25:75,cohort])[None]
    target_fields=torch.as_tensor(fields[25:75])[None];field_support=torch.as_tensor(support[25:75])[None]
    generator=torch.Generator().manual_seed(int(rng.integers(2**30)))
    total=state.agents.new_zeros(())
    for step in range(50):
        previous=state
        state,p=model.imagine_step(previous,generator=generator)
        mask=masks[:,step,:,None];denom=mask.sum().clamp_min(1)
        fs=field_support[:,step,:,None];field_denom=(fs.sum()*3).clamp_min(1)
        micro=((((p['agents'][:,:,:4]-targets[:,step,:,:4])/model.scale[:4])**2)*mask).sum()/denom
        macro=((((p['fields']-target_fields[:,step])/model.field_scale)**2)*fs).sum()/field_denom
        consistency=((((p['fields']-p['derived_fields'])/model.field_scale)**2)*fs).sum()/field_denom
        posterior_target=torch.where(masks[:,step,:,None],targets[:,step],previous.agents)
        _,posterior=model._transition(previous,generator=generator,posterior_target=posterior_target)
        acc=(targets[:,step,:,2:4]-previous.agents[:,:,2:4])/model.dt
        nll=((.5*((acc-posterior['acc_mean'])/posterior['acc_std']).square()+posterior['acc_std'].log())*mask).sum()/denom
        fnll=((.5*((target_fields[:,step]-posterior['field_mean'])/posterior['field_std']).square()+posterior['field_std'].log())*fs).sum()/field_denom
        stochastic=.001*nll+.0001*fnll+.001*posterior['kl']
        if model.variant=='deterministic':stochastic=total.new_zeros(())
        if model.variant=='macro_only':micro=micro*0;stochastic=.0001*fnll+.001*posterior['kl']
        if model.variant in ('independent','micro_only'):macro=macro*0;consistency=consistency*0
        if model.variant=='no_crossscale':consistency=consistency*0
        total+=micro+macro+.05*consistency+stochastic
    return total/50


def train(data_output,output,updates=100,variants=('full','independent','deterministic','macro_only','micro_only','no_crossscale'),seed=101,wall_seconds=1800):
    data_output,output=Path(data_output),Path(output);output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    manifest=json.loads((data_output/'data_manifest.json').read_text())
    if manifest['source']!='REAL_I24' or manifest['shared_source_track_ids']!=0:raise ValueError('Authentic track-disjoint data required')
    for name,digest in manifest['artifacts'].items():
        if sha(data_output/name)!=digest:raise ValueError('Prepared data integrity mismatch')
    cfg=dict(source_kind='REAL_I24',updates=updates,variants=list(variants),seed=seed,model_dim=32,latent_dim=4,
             dt=.2,context_seconds=5,train_horizon_seconds=10,evaluation_horizons=[5,10,20],learning_rate=.001,
             device='cpu',threads=2,workers=0,checkpoint_every_updates=10,
             macro_definition=manifest['macro_definition'],wall_seconds=wall_seconds)
    sources=[Path(p) for p in ('orchestra_wm/i24/model.py','orchestra_wm/i24/schema.py','orchestra_wm/i24/data.py',
                               'orchestra_wm/i24_continuous/model.py','orchestra_wm/i24_continuous/data.py',
                               'orchestra_wm/i24_continuous/training.py')]
    sources=sorted(sources)
    fingerprint=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()+(data_output/'data_manifest.json').read_bytes()+b''.join(p.read_bytes() for p in sources)).hexdigest()
    write_immutable(output/'run_manifest.json',dict(config=cfg,fingerprint=fingerprint,source_sha256=manifest['source_sha256'],
                                                   data_manifest_sha256=sha(data_output/'data_manifest.json'),
                                                   implementation_sha256={str(p):sha(p) for p in sources},
                                                   scope='Single-day regional preliminary optimization; no convergence/generalization claim'))
    scenes={name:[] for name in ('train','validation','test')}
    for item in manifest['scenes']:
        path=data_output/item['path'];scene=load_scene(path)
        with np.load(path.with_suffix('.fields.npz')) as arrays:
            scenes[item['split']].append((scene,arrays['fields'].copy(),arrays['support'].copy()))
    if any(not v for v in scenes.values()):raise ValueError('Missing train/validation/test support')
    checkpoint_dir=output/'checkpoints';checkpoint_dir.mkdir(exist_ok=True);began=time.monotonic();metadata=[]
    for variant in variants:
        torch.manual_seed(seed);rng=np.random.default_rng(seed)
        model=ContinuousWorldModel(32,4,variant,.2);optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        path=checkpoint_dir/f'{variant}-{seed}.pt';step0=0;rows=[]
        if path.exists():
            saved=torch.load(path,map_location='cpu',weights_only=False)
            if saved['fingerprint']!=fingerprint:raise ValueError('Checkpoint source/config mismatch')
            model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
            step0=saved['step'];rows=saved['rows'];rng.bit_generator.state=saved['numpy_rng'];torch.set_rng_state(saved['torch_rng'])
        for step in range(step0+1,updates+1):
            if time.monotonic()-began>wall_seconds:raise TimeoutError('Wall budget reached; verified data and checkpoints preserved; rerun identical command')
            model.train();optimizer.zero_grad()
            scene,fields,support=scenes['train'][int(rng.integers(len(scenes['train'])))]
            loss=objective(model,scene,fields,support,rng)
            if not torch.isfinite(loss):raise ValueError('Nonfinite authentic training objective')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            if step==1 or step%10==0 or step==updates:
                model.eval();vrng=np.random.default_rng(991)
                with torch.no_grad():validation=float(np.mean([float(objective(model,*s,vrng,True)) for s in scenes['validation'][:3]]))
                if not np.isfinite(validation):raise ValueError('Nonfinite autonomous validation loss')
                rows.append(dict(seed=seed,variant=variant,step=step,train_loss=float(loss.detach()),validation_loss=validation))
                atomic_checkpoint(path,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,config=cfg,
                                           variant=variant,seed=seed,fingerprint=fingerprint,source_kind='REAL_I24',rows=rows,
                                           numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state()))
                print(f'{variant}: update {step}/{updates}, train={loss.item():.6f}, validation={validation:.6f}',flush=True)
        saved=torch.load(path,map_location='cpu',weights_only=False)
        metadata.append(dict(variant=variant,seed=seed,updates=saved['step'],checkpoint_sha256=sha(path),
                             final_validation_loss=saved['rows'][-1]['validation_loss'],parameters=sum(p.numel() for p in model.parameters())))
    pd.DataFrame([r for m in metadata for r in torch.load(checkpoint_dir/f'{m["variant"]}-{seed}.pt',map_location='cpu',weights_only=False)['rows']]).to_csv(output/'training.csv',index=False)
    write_immutable(output/'training_metadata.json',metadata)
    return metadata
