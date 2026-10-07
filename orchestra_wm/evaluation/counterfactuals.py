import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist,squareform
from orchestra_wm.evaluation.common import paired_cases,infer,correlations
from orchestra_wm.evaluation.action_sensitivity import named_plans
from orchestra_wm.planning.oracle import true_rollout
from orchestra_wm.models.baselines import trivial_forecast


def candidates(env,cfg,seed):
    rng=np.random.default_rng(seed)
    h=cfg['plan_horizon'];k=cfg['candidates']
    plans=np.repeat(rng.integers(-1,2,(k,int(np.ceil(h/3)),env.n)),3,axis=1)[:,:h]
    plans[:min(k,6)]=np.array(list(named_plans(env,h).values()))[:min(k,6)]
    plans[:,:,~np.array([v.connected for v in env.vehicles])]=0
    return plans


def rank_metrics(predicted,true):
    n=len(true);best=int(np.argmin(predicted));order=np.argsort(true)
    metrics=correlations(predicted,true)
    metrics.update({'selected_true_rank':int(np.flatnonzero(order==best)[0])+1,
        'regret':float(true[best]-np.min(true)),
        'rank_accuracy':float(np.mean([np.sign(predicted[i]-predicted[j])==np.sign(true[i]-true[j]) for i in range(n) for j in range(i)]))})
    for pct in [5,10]:
        k=max(1,int(np.ceil(n*pct/100)))
        metrics[f'top_{pct}_overlap']=len(set(np.argsort(predicted)[:k])&set(order[:k]))/k
    return metrics


def proxy_components(states,dt):
    # Observable-only objective surrogate for trivial baselines (no action dependence).
    speed=states[...,2]*15;active=states[...,5]>.5
    components=np.zeros((*states.shape[:2],6))
    components[...,0]=(np.maximum(0,1-speed/10)*active).mean(-1)
    components[...,1]=((speed<2)&active).mean(-1)
    components[...,2]=(speed/15*active).mean(-1)
    for i in range(states.shape[2]):
        for j in range(i):
            distance=np.linalg.norm((states[:,:,i,:2]-states[:,:,j,:2])*100,axis=-1)
            valid=active[:,:,i]&active[:,:,j]
            components[...,3]+=(distance<3)*valid/states.shape[2]
            components[...,4]+=(distance<7)*valid/states.shape[2]
    return components

@torch.no_grad()
def evaluate_counterfactuals(cfg,models,out):
    rows=[];rankrows=[];first=True
    for seed,family,(env,frames) in paired_cases(cfg):
        plans=candidates(env,cfg,seed+9);truths=[];truecost=[]
        for plan in plans:
            states,components=true_rollout(env,plan);truths.append(states)
            truecost.append(float((components@np.array(cfg['objective_weights'])).sum()))
        truths=np.array(truths);truecost=np.array(truecost)
        true_distance=pdist(truths[:,-1,:,:2].reshape(len(plans),-1)*100)
        for name,model in {**models,'persistence':None,'constant_velocity':None,'oracle':None}.items():
            if name=='oracle': predictions=truths;cost=truecost
            elif model is None:
                base=torch.tensor(frames[-1][0]['agents'][:,:6])[None]
                predictions=trivial_forecast(base,cfg['plan_horizon'],cfg['dt'],name=='constant_velocity').numpy().repeat(len(plans),axis=0)
                cost=(proxy_components(predictions,cfg['dt'])@np.array(cfg['objective_weights'])).sum(1)
            else:
                state=infer(model,frames)
                pred=model.imagine(state.repeat(len(plans)),torch.tensor(plans,device=state.memory.device))
                predictions=pred['agents'].cpu().numpy()
                cost=(pred['components'].cpu().numpy()@np.array(cfg['objective_weights'])).sum(1)
            distance=pdist(predictions[:,-1,:,:2].reshape(len(plans),-1)*100)
            corr=correlations(distance,true_distance)
            rankrows.append({'seed':seed,'scenario':family,'model':name,**rank_metrics(cost,truecost),
                'outcome_pearson':corr['pearson'],'outcome_spearman':corr['spearman']})
            for i in range(len(plans)):
                rows.append({'seed':seed,'scenario':family,'model':name,'plan':i,'predicted_cost':cost[i],'true_cost':truecost[i]})
            if first and name=='orchestra':
                np.savez_compressed(out/'counterfactual_example.npz',predicted=predictions,truth=truths,plans=plans,
                     predicted_cost=cost,true_cost=truecost,model_distances=squareform(distance),true_distances=squareform(true_distance))
        first=False
    pd.DataFrame(rows).to_csv(out/'plan_costs.csv',index=False)
    pd.DataFrame(rankrows).to_csv(out/'rank_fidelity.csv',index=False)
    return pd.DataFrame(rankrows)
