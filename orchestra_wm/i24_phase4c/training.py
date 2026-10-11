"""Small prospective joint-loss continuation; never modifies the macro control."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from orchestra_wm.i24.data import sha,load_scene
from orchestra_wm.i24.training import atomic_checkpoint
from orchestra_wm.i24_continuous.source import write_immutable
from orchestra_wm.i24_phase4b.model import AuditedWorldModel
from orchestra_wm.i24_phase4b.training import verify_preservation
from .joint import joint_objective, MICRO_SCALES


def train_joint(data_root,initial_root,output,receipt,updates=30,seed=101,wall_seconds=900):
    root,initial,output=map(Path,(data_root,initial_root,output))
    verified=verify_preservation(receipt)
    manifest=json.loads((root/'data_manifest.json').read_text())
    quality=json.loads((root/'quality_and_recovery.json').read_text())
    assert verified['expanded_data_manifest_sha256']==sha(root/'data_manifest.json')
    assert verified['expanded_dataset_readback_sha256']==quality['local_recovery_bundle']['sha256']
    assert quality['status']=='EXPANDED_AUTHENTIC_DEVELOPMENT_QC_PASSED'
    assert manifest['source']=='REAL_I24' and manifest['shared_source_track_ids']==0
    for p,h in manifest['artifacts'].items():assert sha(root/p)==h
    prior_run=json.loads((initial/'run_manifest.json').read_text())
    complete=json.loads((initial/'completed.json').read_text())
    init_row=next(r for r in complete if r['variant']=='full')
    init_path=initial/init_row['checkpoint'];assert sha(init_path)==init_row['sha256']
    saved=torch.load(init_path,map_location='cpu',weights_only=False)
    assert saved['step']==prior_run['config']['updates'] and saved['fingerprint']==prior_run['fingerprint']
    for p,h in prior_run['sources'].items():assert sha(Path(p))==h
    cfg=dict(seed=seed,updates=updates,variants=['full','graph_off'],dim=32,latent=4,dt=.2,context_steps=25,horizon_steps=100,
             lr=.001,threads=2,workers=0,train_samples=2,validation_every=10,
             data_manifest_sha256=sha(root/'data_manifest.json'),initial_checkpoint_sha256=sha(init_path),
             initial_updates=saved['step'],micro_scales=MICRO_SCALES,macro_scales=[30,100,3000],loss_weights=[.5,.5],
             initialization='Identical full macro-control weights; reset AdamW for the new joint objective; preserve both original optimizer states',
             objective='Equal normalized proper micro and macro prior scores; no posterior or cross-population conservation loss',
             scope='Jointly supervised historical-cohort motion and retained-population fields; no individual future arrivals generated')
    sources=dict(prior_run['sources'])
    for p in [Path(__file__),Path(__file__).with_name('joint.py')]:sources[str(p)]=sha(p)
    fingerprint=hashlib.sha256(json.dumps(dict(config=cfg,sources=sources),sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True,exist_ok=True)
    write_immutable(output/'run_manifest.json',dict(config=cfg,sources=sources,fingerprint=fingerprint,preservation_receipt_sha256=sha(Path(receipt))))
    if (output/'completed.json').exists():
        completed=json.loads((output/'completed.json').read_text())
        for r in completed:assert sha(output/r['checkpoint'])==r['sha256']
        return completed
    torch.set_num_threads(2);started=time.monotonic();completed=[]
    items={s:[i for i in manifest['scenes'] if i['split']==s] for s in ['train','validation']}
    def load(item):
        p=root/item['path'];scene=load_scene(p)
        with np.load(p.with_suffix('.fields.npz')) as a:return scene,a['fields'].copy(),a['support'].copy()
    validation=[load(i) for i in items['validation']]
    for variant in cfg['variants']:
        torch.manual_seed(seed);rng=np.random.default_rng(seed)
        model=AuditedWorldModel(32,4,variant,.2);model.load_state_dict(saved['model'])
        optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'])
        path=output/'checkpoints'/f'{variant}-{seed}.pt';path.parent.mkdir(exist_ok=True)
        step0=0;rows=[];prior_seconds=0.
        if path.exists():
            state=torch.load(path,map_location='cpu',weights_only=False)
            assert state['fingerprint']==fingerprint
            model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
            rng.bit_generator.state=state['numpy_rng'];torch.set_rng_state(state['torch_rng'])
            step0=state['step'];rows=state['rows'];prior_seconds=state['training_wall_seconds']
        variant_start=time.monotonic()
        for step in range(step0+1,updates+1):
            if time.monotonic()-started>=wall_seconds:raise TimeoutError('Joint wall allowance reached; resume identical configuration')
            model.train();optimizer.zero_grad(set_to_none=True)
            item=items['train'][int(rng.integers(len(items['train'])))];regime=['dense','dense','sparse50','outage'][int(rng.integers(4))]
            loss,diagnostics=joint_objective(model,*load(item),regime,int(rng.integers(2**30)),return_diagnostics=True)
            if not torch.isfinite(loss):raise ValueError('Nonfinite joint loss; previous checkpoint preserved')
            loss.backward();gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            row=dict(variant=variant,step=step,train_joint_energy=float(loss.detach()),gradient_norm=float(gradient),
                     train_macro_energy=diagnostics['macro_energy'],train_micro_energy=diagnostics['micro_energy'],validation_joint_energy=None,
                     validation_macro_energy=None,validation_micro_energy=None)
            if step==1 or step%10==0 or step==updates:
                model.eval()
                with torch.no_grad():
                    values=[joint_objective(model,*s,'dense',991+i,return_diagnostics=True) for i,s in enumerate(validation)]
                row['validation_joint_energy']=float(np.mean([float(v[0]) for v in values]))
                row['validation_macro_energy']=float(np.mean([v[1]['macro_energy'] for v in values]))
                row['validation_micro_energy']=float(np.mean([v[1]['micro_energy'] for v in values]))
                if not np.isfinite(row['validation_joint_energy']):raise ValueError('Nonfinite joint validation')
            rows.append(row)
            atomic_checkpoint(path,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,rows=rows,fingerprint=fingerprint,
                                        numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),config=cfg,variant=variant,
                                        training_wall_seconds=prior_seconds+time.monotonic()-variant_start))
            if row['validation_joint_energy'] is not None:print(json.dumps(row),flush=True)
        completed.append(dict(variant=variant,updates=updates,initial_updates=cfg['initial_updates'],checkpoint=str(path.relative_to(output)),sha256=sha(path),
                              parameters=sum(p.numel() for p in model.parameters()),training_wall_seconds=prior_seconds+time.monotonic()-variant_start))
        pd.DataFrame(rows).to_csv(output/f'{variant}_learning_curve.csv',index=False)
    write_immutable(output/'completed.json',completed)
    return completed
