"""Predeclared training-gradient and validation-only sensitivity diagnostics."""
import json
from pathlib import Path
import numpy as np
import torch
from orchestra_wm.i24.data import sha, load_scene
from orchestra_wm.i24.model import observe_history
from orchestra_wm.i24_continuous.data import observations
from orchestra_wm.i24_continuous.evaluation import energy_score
from orchestra_wm.i24_continuous.source import write_immutable
from orchestra_wm.i24_phase4b.model import AuditedWorldModel
from orchestra_wm.i24_phase4b.training import objective


def diagnose(data_root, training_root, output):
    root,training,output = map(Path,(data_root,training_root,output))
    torch.set_num_threads(2)
    manifest=json.loads((root/'data_manifest.json').read_text())
    run=json.loads((training/'run_manifest.json').read_text());cfg=run['config']
    complete=json.loads((training/'completed.json').read_text())
    signature=dict(checkpoints={r['variant']:r['sha256'] for r in complete},source_sha256=sha(Path(__file__)))
    output.mkdir(parents=True,exist_ok=True);write_immutable(output/'protocol.json',signature)
    if (output/'diagnostics.json').exists():return json.loads((output/'diagnostics.json').read_text())
    train_item=next(i for i in manifest['scenes'] if i['split']=='train')
    scene=load_scene(root/train_item['path'])
    with np.load((root/train_item['path']).with_suffix('.fields.npz')) as a:fields,support=a['fields'].copy(),a['support'].copy()
    results=[]
    for variant in cfg['variants']:
        path=training/'checkpoints'/f'{variant}-{cfg["seed"]}.pt'
        assert sha(path)==signature['checkpoints'][variant]
        payload=torch.load(path,map_location='cpu',weights_only=False)
        torch.manual_seed(cfg['seed']);model=AuditedWorldModel(cfg['dim'],cfg['latent'],variant,cfg['dt'])
        initial={k:v.clone() for k,v in model.state_dict().items()};model.load_state_dict(payload['model'])
        loss=objective(model,scene,fields,support,'dense',101,horizon=3)
        loss.backward()
        modules={}
        for name,p in model.named_parameters():
            prefix=name.split('.')[0]
            row=modules.setdefault(prefix,dict(allocated_parameters=0,gradient_connected_parameters=0,gradient_squared_norm=0.,weight_change_squared_norm=0.))
            row['allocated_parameters']+=p.numel()
            if p.grad is not None:
                row['gradient_connected_parameters']+=p.numel();row['gradient_squared_norm']+=float(p.grad.square().sum())
            row['weight_change_squared_norm']+=float((p.detach()-initial[name]).square().sum())
        model.zero_grad(set_to_none=True);model.eval();sensitivity=[]
        # First three validation windows, fixed by chronology, never performance.
        for number,item in enumerate([i for i in manifest['scenes'] if i['split']=='validation'][:3]):
            s=load_scene(root/item['path']);obs,_=observations(s,25,'dense',7301+number)
            with np.load((root/item['path']).with_suffix('.fields.npz')) as a:y=a['fields'][74].copy();use=a['support'][74].copy()
            with torch.no_grad():
                state=observe_history(model,obs,s.road)
                base=model.imagine(state,50,8,7103+number)['fields'].numpy()[:,49]
                # Remove only the learned prior mean; retain learned variance,
                # decoder noise and paired RNG. This is a diagnostic shift.
                hook=model.prior.register_forward_hook(lambda m,args,out:torch.cat([torch.zeros_like(out[:,:cfg['latent']]),out[:,cfg['latent']:]],-1))
                try:perturbed=model.imagine(state,50,8,7103+number)['fields'].numpy()[:,49]
                finally:hook.remove()
                reset=model.imagine(observe_history(model,obs,s.road,reset=True),50,8,7103+number)['fields'].numpy()[:,49]
                row=dict(scene=s.scene_id,prior_mean_removal_scaled_rmse=float(np.sqrt(np.mean(((base-perturbed)/[30,100,3000])**2))),
                         prior_mean_removal_energy_change=energy_score(perturbed[:,use]/[30,100,3000],y[use]/[30,100,3000])-energy_score(base[:,use]/[30,100,3000],y[use]/[30,100,3000]),
                         memory_reset_scaled_rmse=float(np.sqrt(np.mean(((base-reset)/[30,100,3000])**2))))
                if variant=='full':
                    model.variant='graph_off'
                    try:no_graph=model.imagine(state,50,8,7103+number)['fields'].numpy()[:,49]
                    finally:model.variant=variant
                    row['graph_removal_scaled_rmse']=float(np.sqrt(np.mean(((base-no_graph)/[30,100,3000])**2)))
                    row['graph_removal_energy_change']=energy_score(no_graph[:,use]/[30,100,3000],y[use]/[30,100,3000])-energy_score(base[:,use]/[30,100,3000],y[use]/[30,100,3000])
            sensitivity.append(row)
            print(f'Diagnostic {variant} validation {number+1}/3',flush=True)
        results.append(dict(variant=variant,modules=modules,sensitivity=sensitivity,
                            qualification='Three-step training gradient connectivity is a local diagnostic, not functional capacity equivalence. Ablations are distribution shifts, not causal interventions or retrained controls.'))
    write_immutable(output/'diagnostics.json',results)
    return results
