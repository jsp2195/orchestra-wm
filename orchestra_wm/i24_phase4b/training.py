"""Bounded, resumable matched-macro information-value control.

This prospective control is not a completed Phase4B experiment. It requires a
verified private preservation receipt before allocating training compute.
"""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.checkpoint import checkpoint as activation_checkpoint
from orchestra_wm.i24.data import sha,load_scene
from orchestra_wm.i24.model import observe_history
from orchestra_wm.i24.training import atomic_checkpoint
from orchestra_wm.i24_continuous.data import observations
from orchestra_wm.i24_continuous.source import write_immutable
from .model import AuditedWorldModel,matched_macro_energy


def verify_preservation(receipt_path):
    receipt=json.loads(Path(receipt_path).read_text())
    preservation=json.loads(Path('outputs/i24_phase4b/audit/preservation.json').read_text())
    expected=preservation['local_recovery_bundle']['sha256']
    if receipt.get('sha256')!=expected or receipt.get('readback_sha256')!=expected:
        raise ValueError('Original recovery bundle must have verified remote readback')
    if receipt.get('private') is not True or not receipt.get('remote_object_id'):
        raise ValueError('Approved private durable object required')
    if not receipt.get('expanded_dataset_object_id') or not receipt.get('expanded_dataset_readback_sha256'):
        raise ValueError('Expanded prepared dataset must also be preserved')
    return receipt


def objective(model,scene,fields,support,regime,seed,horizon=100,checkpointed=True):
    obs,cohort=observations(scene,25,regime,seed)
    state=observe_history(model,obs,scene.road)
    # Two independent prior samples provide an unbiased proper-score estimator.
    state=state.repeat(2)
    target=torch.as_tensor(fields[25:25+horizon],dtype=torch.float32)
    valid=torch.as_tensor(support[25:25+horizon],dtype=torch.bool)
    # Average equal-dimensional scores per time step: support may vary by time.
    losses=[]
    for t in range(horizon):
        # Re-create explicit RNG on recomputation; checkpointing must not advance
        # an external Generator twice and silently change the backward sample.
        def transition(current,step_seed=seed+104729*t):
            generator=torch.Generator(device=model.scale.device).manual_seed(step_seed)
            next_state,p=model.imagine_step(current,generator=generator)
            return next_state,p['fields']
        if checkpointed and torch.is_grad_enabled():
            state,prediction=activation_checkpoint(transition,state,use_reentrant=False,preserve_rng_state=False)
        else:
            state,prediction=transition(state)
        if valid[t].any():losses.append(matched_macro_energy(prediction,target[t],valid[t],model.field_scale))
    return torch.stack(losses).mean()


def train(data_root,output,receipt,updates=300,variants=('full','macro_only'),seed=101,wall_seconds=1800):
    receipt=verify_preservation(receipt)
    root=Path(data_root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((root/'data_manifest.json').read_text())
    quality=json.loads((root/'quality_and_recovery.json').read_text())
    if manifest.get('source')!='REAL_I24' or quality.get('status')!='EXPANDED_AUTHENTIC_DEVELOPMENT_QC_PASSED':
        raise ValueError('Verified authentic expanded data required; no fixture substitution')
    if receipt.get('expanded_dataset_readback_sha256')!=quality['local_recovery_bundle']['sha256']:
        raise ValueError('Expanded dataset remote readback hash mismatch')
    if receipt.get('expanded_data_manifest_sha256')!=sha(root/'data_manifest.json'):
        raise ValueError('Preserved expanded data manifest must match this training dataset')
    if manifest['scene_counts']['train']<=13:raise ValueError('Expanded authentic training data required')
    if manifest['shared_source_track_ids']!=0:raise ValueError('Split identity violation')
    for p,h in manifest['artifacts'].items():
        if sha(root/p)!=h:raise ValueError('Prepared data changed')
    cfg=dict(seed=seed,updates=updates,variants=list(variants),dim=32,latent=4,dt=.2,context_steps=25,
             horizon_steps=100,train_samples=2,lr=.001,threads=2,workers=0,validation_every=10,
             activation_checkpointing='per-transition; deterministically reseeded explicit generator',
             objective='Identical prior-only proper joint macro energy; no posterior/micro/consistency penalty',
             scope='Matched macro information control; does not establish trained microscopic fidelity',
             data_manifest_sha256=sha(root/'data_manifest.json'))
    sources={str(p):sha(p) for p in [Path(__file__),Path(__file__).with_name('model.py'),
                                    Path('orchestra_wm/i24/model.py'),Path('orchestra_wm/i24_continuous/model.py'),
                                    Path('orchestra_wm/i24_continuous/data.py'),Path('orchestra_wm/i24/data.py'),
                                    Path('orchestra_wm/i24/schema.py')]}
    fingerprint=hashlib.sha256(json.dumps(dict(config=cfg,sources=sources),sort_keys=True).encode()).hexdigest()
    write_immutable(output/'run_manifest.json',dict(config=cfg,sources=sources,fingerprint=fingerprint,receipt_sha256=sha(Path(receipt['receipt_path'])) if 'receipt_path' in receipt else None))
    if (output/'completed.json').exists():
        completed=json.loads((output/'completed.json').read_text())
        for row in completed:
            if sha(output/row['checkpoint'])!=row['sha256']:raise ValueError('Completed checkpoint changed')
        return completed
    torch.set_num_threads(2);started=time.monotonic();completed=[]
    items={s:[i for i in manifest['scenes'] if i['split']==s] for s in ('train','validation')}
    def load(item):
        p=root/item['path'];scene=load_scene(p)
        with np.load(p.with_suffix('.fields.npz')) as a:return scene,a['fields'].copy(),a['support'].copy()
    validation=[load(i) for i in items['validation']] # fixed entire validation block; never test outcomes
    for variant in variants:
        torch.manual_seed(seed);rng=np.random.default_rng(seed)
        model=AuditedWorldModel(32,4,variant,.2);optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        checkpoint=output/'checkpoints'/f'{variant}-{seed}.pt';checkpoint.parent.mkdir(exist_ok=True)
        step0=0;rows=[];elapsed_prior=0.
        if checkpoint.exists():
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            if saved['fingerprint']!=fingerprint:raise ValueError('Incompatible budget/source/config: use a new run namespace; never bypass fingerprint')
            model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
            rng.bit_generator.state=saved['numpy_rng'];torch.set_rng_state(saved['torch_rng'])
            step0=saved['step'];rows=saved['rows'];elapsed_prior=saved['training_wall_seconds']
        variant_start=time.monotonic()
        for step in range(step0+1,updates+1):
            if time.monotonic()-started>=wall_seconds:
                raise TimeoutError('Wall budget reached; atomic checkpoints retained. Rerun identical command; wall allowance is operational, not fingerprinted.')
            model.train();optimizer.zero_grad();item=items['train'][int(rng.integers(len(items['train'])))]
            regime=['dense','dense','sparse50','outage'][int(rng.integers(4))]
            loss=objective(model,*load(item),regime,int(rng.integers(2**30)))
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss; last checkpoint preserved')
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            row=dict(variant=variant,step=step,train_macro_energy=float(loss.detach()),gradient_norm=float(grad),validation_macro_energy=None)
            if step==1 or step%10==0 or step==updates:
                model.eval()
                with torch.no_grad():row['validation_macro_energy']=float(np.mean([float(objective(model,*s,'dense',991+i)) for i,s in enumerate(validation)]))
                if not np.isfinite(row['validation_macro_energy']):raise ValueError('Nonfinite validation score; last checkpoint preserved')
            rows.append(row)
            # Every update atomic; wall interruption loses no completed update.
            atomic_checkpoint(checkpoint,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,rows=rows,
                              fingerprint=fingerprint,numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),
                              config=cfg,variant=variant,training_wall_seconds=elapsed_prior+time.monotonic()-variant_start))
            if row['validation_macro_energy'] is not None:print(json.dumps(row),flush=True)
        completed.append(dict(variant=variant,updates=updates,checkpoint=str(checkpoint.relative_to(output)),sha256=sha(checkpoint),
                              parameters=sum(p.numel() for p in model.parameters()),training_wall_seconds=elapsed_prior+time.monotonic()-variant_start))
        pd.DataFrame(rows).to_csv(output/f'{variant}_learning_curve.csv',index=False)
    write_immutable(output/'completed.json',completed)
    return completed
