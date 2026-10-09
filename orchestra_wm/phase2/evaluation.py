from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist
from scipy.stats import rankdata
from orchestra_wm.phase2.scenarios import make_case,interaction_residual,pair_relevant,LABELS
from orchestra_wm.phase2.planning import initial_candidates,JointPlanner,balanced_cem
from orchestra_wm.phase2.model import load_phase2
from orchestra_wm.evaluation.common import infer,observe,correlations,write_json
from orchestra_wm.planning.oracle import true_rollout
from orchestra_wm.data.collection import policy


def pairing_shuffle(plans,seed):
    rng=np.random.default_rng(seed);shuffled=plans.copy();permutation=rng.permutation(len(plans))
    shuffled[:,:,1]=plans[permutation,:,1]
    return shuffled


def outcome_error(pred,truth):
    pair=np.linalg.norm((pred[-1,:2,:2]-truth[-1,:2,:2])*100,axis=-1).mean()
    all_agents=np.linalg.norm((pred[-1,:,:2]-truth[-1,:,:2])*100,axis=-1).mean()
    return float(pair),float(all_agents)


def rank_summary(pred,truth):
    corr=correlations(pred,truth);k=2;index=int(np.argmin(pred))
    selected_rank=float(rankdata(truth,method='average')[index])
    return {**corr,'top10_overlap':len(set(np.argsort(pred)[:k])&set(np.argsort(truth)[:k]))/k,
       'regret':float(truth[index]-np.min(truth)),'selected_percentile':100*(selected_rank-1)/(len(truth)-1),
       'rank_agreement':float(np.mean([np.sign(pred[i]-pred[j])==np.sign(truth[i]-truth[j]) for i in range(len(truth)) for j in range(i)]))}


def cache_truth(cfg,out):
    path=out/'evaluation_truth.npz'
    if path.exists():return dict(np.load(path))
    states=[];components=[]
    for index in range(cfg['evaluation_families']):
        env,frames=make_case(cfg,index,'test')
        assert pair_relevant(env.state_array(),0,1,env.family,env.vehicles[0].lane,env.vehicles[1].lane),index
        plans=initial_candidates(cfg['plan_horizon'],env.n);ss=[];cc=[]
        for plan in plans:
            s,c=true_rollout(env,plan);ss.append(s);cc.append(c)
        states.append(ss);components.append(cc)
    np.savez_compressed(path,states=np.array(states),components=np.array(components))
    return dict(np.load(path))

@torch.no_grad()
def evaluate_models(cfg,out,names):
    truth=cache_truth(cfg,out);rows=[];ranks=[];shuffles=[];costrows=[]
    dest=out/'evaluation';dest.mkdir(exist_ok=True)
    for seed in cfg['training_seeds']:
        for name in names:
            files=[dest/f'{name}_{seed}_{kind}.csv' for kind in ['prediction','rank','shuffle','costs']]
            if all(p.exists() for p in files):
                for target,path in zip([rows,ranks,shuffles,costrows],files):target.extend(pd.read_csv(path).to_dict('records'))
                continue
            model=load_phase2(out/'checkpoints'/f'{name}_{seed}.pt');rr=[];rankrows=[];sr=[];cr=[]
            for index in range(cfg['evaluation_families']):
                env,frames=make_case(cfg,index,'test');state=infer(model,frames)
                plans=initial_candidates(cfg['plan_horizon'],env.n)
                p=model.imagine(state.repeat(18),torch.as_tensor(plans));pred=p['agents'].numpy();pc=p['components'].numpy()
                ts=truth['states'][index];tc=truth['components'][index]
                predicted_cost=(pc@cfg['objective_weights']).sum(1);true_cost=(tc@cfg['objective_weights']).sum(1)
                rank=rank_summary(predicted_cost,true_cost)
                rankrows.append({'seed':seed,'model':name,'family_id':index,'scenario':env.family,**rank})
                cp=correlations(predicted_cost[:9],true_cost[:9])
                geometry=correlations(pdist(pred[:9,-1,:2,:2].reshape(9,-1)),pdist(ts[:9,-1,:2,:2].reshape(9,-1)))
                residual_error=float(np.sqrt(np.mean((interaction_residual(predicted_cost[:9])-interaction_residual(true_cost[:9]))**2)))
                pair_errors=[];all_errors=[]
                for k in range(9):
                    pair,all_agent=outcome_error(pred[k],ts[k]);pair_errors.append(pair);all_errors.append(all_agent)
                baseline=float(np.mean(pair_errors))
                rr.append({'seed':seed,'model':name,'family_id':index,'scenario':env.family,
                    'pair_position_error':baseline,'all_position_error':float(np.mean(all_errors)),
                    'pair_speed_error':float(np.abs(pred[:9,-1,:2,2]-ts[:9,-1,:2,2]).mean()*15),
                    'conflict_proxy_error':float(np.abs(pc[:9,:,4]-tc[:9,:,4]).mean()),
                    'objective_error':float(np.abs(predicted_cost[:9]-true_cost[:9]).mean()),'interaction_residual_error':residual_error,
                    'joint_cf_pearson':cp['pearson'],'joint_cf_spearman':cp['spearman'],
                    'outcome_distance_pearson':geometry['pearson'],'outcome_distance_spearman':geometry['spearman'],
                    'true_interaction_rms':float(np.sqrt((interaction_residual(true_cost[:9])**2).mean()))})
                for mode,actions in [('correct',plans[:9]),('joint_pairing',pairing_shuffle(plans[:9],90000+index)),
                                     ('ordinary',plans[np.random.default_rng(91000+index).permutation(9)])]:
                    forecast=pred[:9] if mode=='correct' else model.imagine(state.repeat(9),torch.as_tensor(actions))['agents'].numpy()
                    error=float(np.mean([outcome_error(forecast[k],ts[k])[0] for k in range(9)]))
                    sr.append({'seed':seed,'model':name,'family_id':index,'scenario':env.family,'condition':mode,
                          'pair_position_error':error,'gap':error-baseline,'relative_degradation':error/max(baseline,1e-12)-1})
                for k in range(18):
                    cr.append({'seed':seed,'model':name,'family_id':index,'scenario':env.family,'plan':k,
                               'predicted_cost':float(predicted_cost[k]),'true_cost':float(true_cost[k])})
            for records,path,target in zip([rr,rankrows,sr,cr],files,[rows,ranks,shuffles,costrows]):
                pd.DataFrame(records).to_csv(path,index=False);target.extend(records)
            print('Evaluated predictions:',name,seed,flush=True)
    for records,filename in [(rows,'interaction_prediction'),(ranks,'interaction_rank'),(shuffles,'joint_shuffle'),(costrows,'factorial_plan_costs')]:
        pd.DataFrame(records).to_csv(out/f'{filename}.csv',index=False)


def run_episode(cfg,index,name,model=None,hidden=False,reset_memory=False,split='test'):
    env,frames=make_case(cfg,index,split);rng=np.random.default_rng(70000+index)
    if model:
        state=infer(model,frames);planner=JointPlanner(model,cfg,70000+index)
    obs=frames[-1][0];previous=None;components=[];actions=[];collision=conflict=0
    for step in range(cfg['planning_steps']):
        current={k:v.copy() for k,v in obs.items()}
        if hidden and step<6:
            # Synthetic loss of all direct tracks/telemetry for AV B, after a visible history.
            current['agents'][1,:6]=0;current['agents'][1,7:]=0;current['mask'][1]=False
        if model:
            if reset_memory:state=model.initial_state(1,env.n,len(env.road.lanes))
            if step or hidden or reset_memory:
                state=observe(model,state,current,None if step==0 or reset_memory else previous,
                              env.state_array() if model.variant=='privileged' else None)
            action=planner.plan(state)[0][0]
        elif name=='oracle':
            weights=np.asarray(cfg['objective_weights'])
            score=lambda plans:np.array([(true_rollout(env,p)[1]@weights).sum() for p in plans])
            action=balanced_cem(score,cfg['plan_horizon'],env.n,cfg['cem_iterations'],rng,current['control_mask'])[0][0]
        else:action=policy(env,name,rng)
        actions.append({'step':step,'a':int(action[0]),'b':int(action[1]),'hidden':hidden and step<6})
        obs,_,_,_,info=env.step(action);previous=action;components.append(info['components'])
        collision+=info['collisions'];conflict+=info['conflicts']
    values=np.array(components)
    return {'family_id':index,'scenario':env.family,'objective':float((values@cfg['objective_weights']).sum()),
           'delay':float(values[:,0].sum()),'progress':float(values[:,2].sum()),'queue':float(values[:,1].sum()),
           'conflicts':conflict,'collisions':collision,'completed':env.completed},actions


def evaluate_mpc(cfg,out,names):
    destination=out/'mpc';destination.mkdir(exist_ok=True);allrows=[];allactions=[]
    schedule=[(0,n) for n in ['oracle','heuristic','noop']]+[(s,n) for s in cfg['training_seeds'] for n in names]
    for seed,name in schedule:
        path=destination/f'{name}_{seed}.csv';apath=destination/f'{name}_{seed}_actions.csv'
        if path.exists() and apath.exists():
            allrows.extend(pd.read_csv(path).to_dict('records'));allactions.extend(pd.read_csv(apath).to_dict('records'));continue
        model=load_phase2(out/'checkpoints'/f'{name}_{seed}.pt') if seed else None
        rows=[];acts=[]
        for index in range(cfg['evaluation_families']):
            row,actions=run_episode(cfg,index,name,model)
            rows.append({'seed':seed,'model':name,**row})
            acts.extend({'seed':seed,'model':name,'family_id':index,'scenario':row['scenario'],**a} for a in actions)
        pd.DataFrame(rows).to_csv(path,index=False);pd.DataFrame(acts).to_csv(apath,index=False)
        allrows.extend(rows);allactions.extend(acts)
        print('Evaluated MPC:',name,seed,flush=True)
    frame=pd.DataFrame(allrows);oracle=frame[frame.model=='oracle'].set_index('family_id').objective
    frame['regret_to_oracle']=frame.objective-frame.family_id.map(oracle)
    frame.to_csv(out/'mpc_results.csv',index=False);pd.DataFrame(allactions).to_csv(out/'mpc_actions.csv',index=False)


def evaluate_memory(cfg,out):
    rows=[];acts=[]
    for seed in cfg['training_seeds']:
        model=load_phase2(out/'checkpoints'/f'C_difference_{seed}.pt')
        for index in range(4):
            for reset in [False,True]:
                row,actions=run_episode(cfg,index,'C_difference',model,hidden=True,reset_memory=reset,split='memory')
                rows.append({'seed':seed,'memory':'reset' if reset else 'intact',**row})
                acts.extend({'seed':seed,'memory':'reset' if reset else 'intact','family_id':index,**a} for a in actions)
    pd.DataFrame(rows).to_csv(out/'memory_coordination.csv',index=False)
    pd.DataFrame(acts).to_csv(out/'memory_actions.csv',index=False)
