from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from orchestra_wm.envs.traffic_env import TrafficEnv
from orchestra_wm.data.collection import policy


def write_json(path,value):
    def clean(x):
        if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)): return [clean(v) for v in x]
        if isinstance(x,np.ndarray): return clean(x.tolist())
        if isinstance(x,(np.floating,float)): return float(x) if np.isfinite(x) else None
        if isinstance(x,np.integer): return int(x)
        return x
    Path(path).write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n')

def correlations(a,b):
    a,b=np.asarray(a),np.asarray(b)
    if len(a)<2 or np.std(a)<1e-10 or np.std(b)<1e-10: return {'pearson':float('nan'),'spearman':float('nan')}
    return {'pearson':float(pearsonr(a,b).statistic),'spearman':float(spearmanr(a,b).statistic)}

@torch.no_grad()
def observe(model,state,obs,previous=None,privileged=None):
    device=next(model.parameters()).device
    agents=obs['agents'].copy();mask=obs['mask'].copy()
    if model.variant=='privileged':
        if privileged is None: raise ValueError('Privileged diagnostic requires an explicit state')
        agents[:,:6]=privileged;mask[:]=True
    tokens={'agents':torch.tensor(agents,device=device)[None],
            'lanes':torch.tensor(obs['lanes'],device=device)[None]}
    action=None if previous is None else torch.tensor(previous,device=device)[None]
    return model.observe(state,tokens,torch.tensor(mask,device=device)[None],action)

def context(cfg,family,seed,steps=None):
    env=TrafficEnv(cfg,family,seed);obs,_=env.reset(seed=seed)
    frames=[];previous=None;rng=np.random.default_rng(seed+44)
    for _ in range(steps or cfg['context']):
        frames.append((obs,previous,env.state_array()))
        if len(frames)<(steps or cfg['context']):
            previous=policy(env,'correlated',rng,previous);obs,*_=env.step(previous)
    return env,frames

@torch.no_grad()
def infer(model,frames):
    obs=frames[0][0]
    state=model.initial_state(1,len(obs['agents']),len(obs['lanes']))
    for obs,prev,truth in frames:
        state=observe(model,state,obs,prev,truth if model.variant=='privileged' else None)
    return state


def paired_cases(cfg):
    # Evaluation seeds never overlap the training episodes; three independent scenario seeds.
    for seed in cfg['seeds']:
        for family in cfg['scenarios']:
            yield seed,family,context(cfg,family,10000+seed)


def errors(pred,truth):
    mask=truth[...,5]>.5
    if not mask.any(): mask=np.ones_like(mask,dtype=bool)
    return {'position_error':float(np.linalg.norm((pred[...,:2]-truth[...,:2])*100,axis=-1)[mask].mean()),
            'speed_error':float((np.abs(pred[...,2]-truth[...,2])*15)[mask].mean()),
            'world_error':float(((pred-truth)**2)[mask].mean())}


def aggregate_by_seed(frame,group,value):
    per=frame.groupby(group+['seed'])[value].mean().reset_index()
    result=per.groupby(group)[value].agg(['mean','std','count']).reset_index()
    result['sem']=result['std']/np.sqrt(result['count'])
    return result
