import numpy as np
import pandas as pd
from scipy.stats import t
from orchestra_wm.evaluation.common import write_json


def seed_interval(values):
    x=np.asarray(values,float);n=len(x);assert n>=2 and np.isfinite(x).all()
    sd=float(x.std(ddof=1));se=sd/np.sqrt(n);margin=float(t.ppf(.975,n-1)*se)
    return {'mean':float(x.mean()),'sd':sd,'se':se,'ci_low':float(x.mean()-margin),'ci_high':float(x.mean()+margin),'n_seeds':n}


def hierarchical_interval(matrix,seed=1701,replicates=5000):
    x=np.asarray(matrix,float);rng=np.random.default_rng(seed);samples=[]
    for _ in range(replicates):
        chosen=rng.integers(len(x),size=len(x));means=[]
        for row in chosen:means.append(x[row,rng.integers(x.shape[1],size=x.shape[1])].mean())
        samples.append(np.mean(means))
    return {'bootstrap_low':float(np.quantile(samples,.025)),'bootstrap_high':float(np.quantile(samples,.975)),'bootstrap_replicates':replicates}


def paired_gain(frame,value,baseline,model,reverse=False):
    p=frame[frame.model.isin([baseline,model])].pivot(index=['seed','family_id','scenario'],columns='model',values=value).reset_index()
    p['gain']=p[baseline]-p[model]
    if reverse:p['gain']=-p['gain']
    return p


def compute_statistics(cfg,out):
    prediction=pd.read_csv(out/'interaction_prediction.csv');mpc=pd.read_csv(out/'mpc_results.csv')
    rank=pd.read_csv(out/'interaction_rank.csv');shuffle=pd.read_csv(out/'joint_shuffle.csv');actions=pd.read_csv(out/'mpc_actions.csv')
    names=list(prediction.model.unique());gainrows=[]
    for name in names:
        if name=='independent':continue
        paired=paired_gain(mpc,'objective','independent',name)
        for task in ['all','intersection','merge']:
            p=paired if task=='all' else paired[paired.scenario==task]
            seedmeans=p.groupby('seed').gain.mean();matrix=p.pivot(index='seed',columns='family_id',values='gain').to_numpy()
            gainrows.append({'model':name,'task':task,**seed_interval(seedmeans),**hierarchical_interval(matrix)})
    gains=pd.DataFrame(gainrows);gains.to_csv(out/'coordination_gain.csv',index=False)
    errors=paired_gain(prediction,'pair_position_error','independent','C_difference');int_c=seed_interval(errors.groupby('seed').gain.mean())
    c_pred=prediction[prediction.model=='C_difference'];c_rank=rank[rank.model=='C_difference']
    c_shuffle=shuffle[(shuffle.model=='C_difference')&(shuffle.condition=='joint_pairing')]
    # Ratio of per-seed mean errors, not a mean of per-episode ratios.
    correct=shuffle[(shuffle.model=='C_difference')&(shuffle.condition=='correct')].groupby('seed').pair_position_error.mean()
    shuffled=c_shuffle.groupby('seed').pair_position_error.mean()
    int_e=seed_interval((shuffled/correct-1).to_numpy())
    c_actions=actions[actions.model=='C_difference'];commands=c_actions[['a','b']].to_numpy().ravel()
    freqs={str(a):float((commands==a).mean()) for a in [-1,0,1]};strategies=len(c_actions[['a','b']].drop_duplicates())
    strength=pd.read_csv(out/'simulator_interaction_strength.csv')
    families=pd.read_csv(out/'dataset_v2_factorial/families.csv');train=families[families.split=='train']
    int_h=gains[(gains.model=='C_difference')&(gains.task=='all')].iloc[0].to_dict()
    gates={
      'INT-A':{'pass':bool(strength.interaction_rms.mean()>.05 and (strength.interaction_rms>.05).mean()>=1/3),
               'interaction_rms':float(strength.interaction_rms.mean()),'fraction_above_005':float((strength.interaction_rms>.05).mean())},
      'INT-B':{'pass':bool(train.groupby('family_id').joint_action.nunique().eq(9).all()),'training_families':int(train.family_id.nunique()),'training_siblings':len(train)},
      'INT-C':{'pass':int_c['ci_low']>0,**int_c},
      'INT-D':{'pass':bool(c_pred.joint_cf_pearson.mean()>0 and c_pred.outcome_distance_pearson.mean()>0),
               'objective_pearson':float(c_pred.joint_cf_pearson.mean()),'outcome_distance_pearson':float(c_pred.outcome_distance_pearson.mean())},
      'INT-E':{'pass':int_e['mean']>=.05,**int_e},
      'INT-F':{'pass':bool(c_rank.spearman.mean()>.5 and c_rank.top10_overlap.mean()>2/18),
               'spearman':float(c_rank.spearman.mean()),'top10_overlap':float(c_rank.top10_overlap.mean()),'regret':float(c_rank.regret.mean()),'selected_percentile':float(c_rank.selected_percentile.mean())},
      'INT-G':{'pass':max(freqs.values())<=.95 and strategies>=2,'command_frequencies':freqs,'joint_strategies':strategies},
      'INT-H':{'pass':int_h['ci_low']>cfg['coordination_min_effect'],**int_h}}
    write_json(out/'gates.json',gates)
    summary=[]
    for name in names:
        r=prediction[prediction.model==name];s=shuffle[(shuffle.model==name)&(shuffle.condition=='joint_pairing')];k=rank[rank.model==name]
        objective=float(mpc[mpc.model==name].objective.mean())
        summary.append({'model':name,'interaction_error':float(r.pair_position_error.mean()),'joint_cf_corr':float(r.joint_cf_pearson.mean()),
               'joint_shuffle_gap':float(s.gap.mean()),'plan_rank_spearman':float(k.spearman.mean()),'mpc_objective':objective,
               'coordination_gain':float(mpc[mpc.model=='independent'].objective.mean()-objective)})
    # Oracle forecasts and ranks are the executed simulator reference, not a learned-model claim.
    oracle=float(mpc[mpc.model=='oracle'].objective.mean())
    summary.append({'model':'oracle','interaction_error':0.,'joint_cf_corr':1.,'joint_shuffle_gap':None,'plan_rank_spearman':1.,'mpc_objective':oracle,
         'coordination_gain':float(mpc[mpc.model=='independent'].objective.mean()-oracle)})
    pd.DataFrame(summary).to_csv(out/'definitive_comparison.csv',index=False)
    rows=[]
    for name in names:
        for metric in ['pair_position_error','joint_cf_pearson','interaction_residual_error']:
            per=prediction[prediction.model==name].groupby('seed')[metric].mean()
            rows.append({'model':name,'metric':metric,**seed_interval(per)})
        per=mpc[mpc.model==name].groupby('seed').objective.mean()
        rows.append({'model':name,'metric':'mpc_objective',**seed_interval(per)})
    pd.DataFrame(rows).to_csv(out/'seed_summary.csv',index=False)
    return gates


def supplemental_statistics(out):
    """Secondary diagnostics, never substituted for the preregistered gates."""
    from orchestra_wm.phase2.scenarios import interaction_residual
    from orchestra_wm.evaluation.common import correlations
    costs=pd.read_csv(out/'factorial_plan_costs.csv');rows=[]
    for keys,frame in costs[costs.plan<9].groupby(['seed','model','family_id','scenario']):
        frame=frame.sort_values('plan');truth=interaction_residual(frame.true_cost.to_numpy()).ravel();pred=interaction_residual(frame.predicted_cost.to_numpy()).ravel()
        corr=correlations(pred,truth)
        rows.append(dict(zip(['seed','model','family_id','scenario'],keys),true_residual_rms=float(np.sqrt(np.mean(truth**2))),predicted_residual_rms=float(np.sqrt(np.mean(pred**2))),residual_pearson=corr['pearson'],residual_spearman=corr['spearman'],factorial_rank_agreement=float(np.mean([np.sign(frame.predicted_cost.iloc[i]-frame.predicted_cost.iloc[j])==np.sign(frame.true_cost.iloc[i]-frame.true_cost.iloc[j]) for i in range(9) for j in range(i)]))))
    pd.DataFrame(rows).to_csv(out/'factorial_residual_agreement.csv',index=False)
    actions=pd.read_csv(out/'mpc_actions.csv')
    joint=actions.groupby(['seed','model','scenario','a','b']).size().rename('count').reset_index()
    joint['frequency']=joint['count']/joint.groupby(['seed','model','scenario'])['count'].transform('sum');joint.to_csv(out/'joint_strategy_frequencies.csv',index=False)
    long=pd.concat([actions[['seed','model','scenario','a']].rename(columns={'a':'command'}),actions[['seed','model','scenario','b']].rename(columns={'b':'command'})])
    freq=long.groupby(['seed','model','scenario','command']).size().rename('count').reset_index();freq['frequency']=freq['count']/freq.groupby(['seed','model','scenario'])['count'].transform('sum');freq.to_csv(out/'command_frequencies.csv',index=False)
    mpc=pd.read_csv(out/'mpc_results.csv');mpc.groupby(['seed','model','scenario'])[['objective','delay','progress','conflicts','collisions','completed','regret_to_oracle']].mean().to_csv(out/'task_results.csv')
    memory=pd.read_csv(out/'memory_coordination.csv');memory.groupby(['seed','memory','scenario'])[['objective','conflicts','collisions']].mean().to_csv(out/'memory_summary.csv')
