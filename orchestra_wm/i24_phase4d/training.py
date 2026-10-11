"""Paired, capacity-matched optimization with joint validation stopping."""
import hashlib,json,time,resource
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from orchestra_wm.i24.data import sha
from orchestra_wm.i24.training import atomic_checkpoint
from orchestra_wm.i24_continuous.source import write_immutable
from orchestra_wm.i24_phase4b.training import verify_preservation
from .model import ResidualFields,energy

def load_cache(root):
    root=Path(root);m=json.loads((root/'manifest.json').read_text());items=[]
    assert m['config']['feature_source_sha256']==sha(Path(__file__).with_name('data.py'))
    original=Path('outputs/i24_phase4b/development_v1');source=json.loads((original/'data_manifest.json').read_text())
    assert sha(original/'data_manifest.json')==m['config']['data_manifest_sha256']
    for scene in source['scenes']:
        if scene['split'] not in ('train','validation'):continue
        for p in [original/scene['path'],(original/scene['path']).with_suffix('.fields.npz')]:
            assert sha(p)==source['artifacts'][str(p.relative_to(original))]
    for e in m['entries']:
        assert e['split'] in ('train','validation')
        p=root/e['path'];assert sha(p)==e['sha256']
        with np.load(p) as f:a={k:torch.from_numpy(f[k].copy()) for k in f.files}
        items.append((e,a))
    return m,items

def collate(items):return {k:torch.stack([x[k] for x in items]) for k in items[0]}

def score_validation(models,items,samples=8,ablation=None):
    values={name:[] for name in models}
    validation=[a for e,a in items if e['split']=='validation' and e['regime']=='dense']
    with torch.no_grad():
        for begin in range(0,len(validation),4):
            batch=collate(validation[begin:begin+4])
            for name,model in models.items():
                model.eval();x=model(batch,name,samples,8001+begin,ablation=ablation)
                for j in range(len(batch['target'])):
                    values[name].append(float(energy(x[:,j,[24,49,99]],batch['target'][j,[24,49,99]],batch['support'][j,[24,49,99]])))
    return {name:float(np.mean(v)) for name,v in values.items()}

def train(cache,output,seed=101,max_updates=800,wall_seconds=1800):
    verify_preservation('work/phase4b/private_backup_receipt.json')
    output=Path(output);output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    _,items=load_cache(cache)
    cfg=dict(seed=seed,max_updates=max_updates,min_updates=250,validation_every=50,patience=5,min_delta=.0001,
             lr=.001,dim=32,batch_size=4,samples=4,variants=['full','aggregate'],dt=.2,history_steps=25,forecast_steps=100,
             cache_manifest_sha256=sha(Path(cache)/'manifest.json'),objective='Mean per-time normalized joint macro energy, all 100 autonomous steps',
             selection='One common checkpoint minimizing mean full/aggregate validation energy averaged over 5/10/20s; same update budget; no test',
             stopping='At >=250 updates, five validation checks without shared improvement >=0.0001; max800; stopping alone does not prove convergence',
             active_capacity='Identical layers and token count; full only adds past microscopic statistics to aggregate feature basis')
    sources={str(p):sha(p) for p in [Path(__file__),Path(__file__).with_name('model.py'),Path(__file__).with_name('data.py'),Path('orchestra_wm/i24_continuous/data.py'),Path('orchestra_wm/i24/data.py'),Path('orchestra_wm/i24_phase4b/training.py')]}
    fingerprint=hashlib.sha256(json.dumps(dict(config=cfg,sources=sources),sort_keys=True).encode()).hexdigest()
    write_immutable(output/'run_manifest.json',dict(config=cfg,sources=sources,fingerprint=fingerprint))
    if (output/'completed.json').exists():
        done=json.loads((output/'completed.json').read_text());assert sha(output/'checkpoints/best.pt')==done['best_checkpoint_sha256'];return done
    torch.manual_seed(seed);models={v:ResidualFields(cfg['dim']) for v in cfg['variants']}
    models['aggregate'].load_state_dict(models['full'].state_dict())
    optimizers={v:torch.optim.AdamW(m.parameters(),lr=cfg['lr'],weight_decay=.0001) for v,m in models.items()}
    rng=np.random.default_rng(seed);step0=0;rows=[];best=float('inf');stale=0;elapsed=0;best_step=0
    masks={v:{n:torch.zeros_like(p,dtype=torch.bool) for n,p in m.named_parameters()} for v,m in models.items()}
    cp=output/'checkpoints';cp.mkdir(exist_ok=True);last=cp/'last.pt'
    if last.exists():
        saved=torch.load(last,weights_only=False);assert saved['fingerprint']==fingerprint
        for v,m in models.items():m.load_state_dict(saved['models'][v]);optimizers[v].load_state_dict(saved['optimizers'][v])
        rng.bit_generator.state=saved['numpy_rng'];torch.set_rng_state(saved['torch_rng']);step0=saved['step'];rows=saved['rows'];best=saved['best'];best_step=saved['best_step'];stale=saved['stale'];elapsed=saved['seconds'];masks=saved['gradient_masks']
    groups={(e['number'],e['regime']):a for e,a in items if e['split']=='train'};ntrain=len({e['number'] for e,a in items if e['split']=='train'})
    start=time.monotonic();stop='MAX_UPDATES';capacity={}
    def state(step):return dict(models={v:m.state_dict() for v,m in models.items()},optimizers={v:o.state_dict() for v,o in optimizers.items()},numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),step=step,rows=rows,best=best,best_step=best_step,stale=stale,seconds=elapsed+time.monotonic()-start,gradient_masks=masks,fingerprint=fingerprint,config=cfg)
    for step in range(step0+1,max_updates+1):
        if time.monotonic()-start>=wall_seconds:raise TimeoutError('Paired checkpoint retained; resume identical command/configuration')
        regime=['dense','dense','sparse50','outage'][int(rng.integers(4))]
        batch=collate([groups[(int(i),regime)] for i in rng.integers(ntrain,size=cfg['batch_size'])]);noise=int(rng.integers(2**30));row=dict(step=step,regime=regime)
        for name,model in models.items():
            model.train();o=optimizers[name];o.zero_grad(set_to_none=True)
            prediction=model(batch,name,cfg['samples'],noise);loss=energy(prediction,batch['target'],batch['support'])
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss; prior checkpoint retained')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            for n,p in model.named_parameters():
                if p.grad is not None:masks[name][n]|=p.grad.detach().abs()>0
            o.step();row[name+'_train']=float(loss.detach());row[name+'_gradient']=float(norm)
        if step==1 or step%cfg['validation_every']==0:
            scores=score_validation(models,items);shared=float(np.mean(list(scores.values())))
            row.update({n+'_validation':v for n,v in scores.items()});row['shared_validation']=shared
            if shared<best-cfg['min_delta']:best=shared;best_step=step;stale=0
            else:stale+=1
            if stale in (2,4):
                for o in optimizers.values():
                    for g in o.param_groups:g['lr']*=.5
            row['lr']=optimizers['full'].param_groups[0]['lr'];print(json.dumps(row),flush=True)
        rows.append(row);atomic_checkpoint(last,state(step))
        if best_step==step:atomic_checkpoint(cp/'best.pt',state(step))
        if step>=cfg['min_updates'] and stale>=cfg['patience']:stop='VALIDATION_NO_IMPROVEMENT';break
    for name,model in models.items():capacity[name]=dict(allocated=sum(p.numel() for p in model.parameters()),observed_nonzero_gradient_scalars=sum(int(m.sum()) for m in masks[name].values()),by_parameter={n:int(m.sum()) for n,m in masks[name].items()})
    frame=pd.DataFrame(rows);frame.to_csv(output/'learning_curve.csv',index=False)
    recent=frame.dropna(subset=['shared_validation']).tail(4).shared_validation
    stable=len(recent)==4 and float(recent.max()-recent.min())<=max(.002,.02*float(recent.mean()))
    done=dict(seed=seed,updates_per_model=step,selected_update=best_step,stop=stop,validation_recent_range=float(recent.max()-recent.min()),validation_stabilized_heuristic=bool(stable),best_shared_validation=best,wall_seconds=elapsed+time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,capacity=capacity,best_checkpoint_sha256=sha(cp/'best.pt'),last_checkpoint_sha256=sha(last),fingerprint=fingerprint,scope='Training/validation November21 only; independent testing unopened')
    write_immutable(output/'completed.json',done);return done
