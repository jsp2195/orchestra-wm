from pathlib import Path
import hashlib
import json
import itertools
import numpy as np
import pandas as pd
import torch
from orchestra_wm.utils.config import load_config
from orchestra_wm.utils.seed import seed_everything
from orchestra_wm.models.training import load_model
from orchestra_wm.evaluation.interactions import evaluate_interactions
from orchestra_wm.evaluation.planning import evaluate_planning
from orchestra_wm.evaluation.common import observe,infer,write_json
from orchestra_wm.envs.traffic_env import TrafficEnv
from orchestra_wm.planning.orchestrator import Orchestrator
from orchestra_wm.planning.oracle import true_rollout,oracle_plan
from orchestra_wm.phase2.planning import balanced_cem
from orchestra_wm.phase2.scenarios import make_case,factorial_plans,interaction_residual,pair_relevant,LABELS

def preserve_manifest(out):
    if (out/'phase1_preservation.json').exists():
        verify_preservation(out)
        return
    files=list(Path('outputs/smoke').rglob('*'))+[Path('README.md'),Path('docs/CLAIM_LEDGER.md')]
    write_json(out/'phase1_preservation.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()})

def verify_preservation(out,allow_appended_docs=True):
    manifest=json.loads((out/'phase1_preservation.json').read_text())
    for name,expected in manifest.items():
        data=Path(name).read_bytes()
        if allow_appended_docs and name in ['README.md','docs/CLAIM_LEDGER.md']:
            data=data.split(b'\n<!-- PHASE2_RESULTS -->')[0]
        assert hashlib.sha256(data).hexdigest()==expected, f'Phase 1 changed: {name}'
    return len(manifest)

def compare_reproduction(original,new,keys):
    original=original.sort_values(keys).reset_index(drop=True)
    new=new.sort_values(keys).reset_index(drop=True)
    assert original[keys].equals(new[keys])
    cols=[c for c in original.select_dtypes(include='number') if c!='latency_ms']
    return float(np.nanmax(np.abs(original[cols].to_numpy()-new[cols].to_numpy())))

def reproduce(out):
    cfg=load_config('configs/smoke.yaml');seed_everything(cfg['seed'],cfg['threads'])
    target=out/'phase1_reproduction';target.mkdir(exist_ok=True)
    models={k:load_model(Path('outputs/smoke/checkpoints')/f'{k}.pt') for k in ['orchestra','independent','privileged','no_actions','memoryless']}
    if not all((target/f'{name}.csv').exists() for name in ['interactions','interaction_contrasts','planning']):
        evaluate_interactions(cfg,models,target);evaluate_planning(cfg,models,target)
    comparisons={}
    for name in ['interactions','interaction_contrasts','planning']:
        original=pd.read_csv(f'outputs/smoke/{name}.csv');new=pd.read_csv(target/f'{name}.csv')
        keys={'interactions':['seed','model','plan'],'interaction_contrasts':['seed','model'],'planning':['seed','scenario','controller']}[name]
        error=compare_reproduction(original,new,keys)
        assert error<1e-5,(name,error)
        comparisons[name]={'maximum_absolute_difference':error,'rows':len(new)}
    write_json(out/'phase1_reproduction.json',comparisons)
    return models

@torch.no_grad()
def action_choices(models,out):
    cfg=load_config('configs/smoke.yaml');rows=[]
    for seed in cfg['seeds']:
        for family in cfg['scenarios']:
            e=TrafficEnv(cfg,family,seed+40000);obs,_=e.reset(seed=seed+40000)
            m=models['orchestra'];state=m.initial_state(1,e.n,len(e.road.lanes));prev=None;p=Orchestrator(m,cfg,seed)
            for t in range(cfg['planning_steps']):
                state=observe(m,state,obs,prev);action=p.plan(state)[0][0];s=e.state_array()
                for i in np.flatnonzero(obs['control_mask']):
                    relevant=[j for j in np.flatnonzero(obs['control_mask']) if i!=j and pair_relevant(s,i,j,family,e.vehicles[i].lane,e.vehicles[j].lane)]
                    node=np.array([5.,0.]) if family=='merge' else np.zeros(2)
                    distance=float(np.linalg.norm(s[i,:2]*100-node))
                    occupancy=sum(not v.connected and v.active and np.linalg.norm(s[k,:2]*100-node)<25 for k,v in enumerate(e.vehicles))
                    rows.append({'seed':seed,'scenario':family,'step':t,'agent':i,'action':int(action[i]),
                      'pair':f'{min(i,relevant[0])}-{max(i,relevant[0])}' if relevant else 'none','conflict_present':bool(relevant),'distance':distance,'distance_bin':'0-10' if distance<10 else '10-20' if distance<20 else '20-30' if distance<30 else '30+',
                      'opposing_action':int(action[relevant[0]]) if relevant else 9,'background_occupancy':occupancy})
                obs,*_=e.step(action);prev=action
    frame=pd.DataFrame(rows);frame.to_csv(out/'phase1_action_choices_raw.csv',index=False)
    stats=[]
    for dimension in ['agent','pair','conflict_present','distance_bin','opposing_action','background_occupancy']:
        for (family,value),g in frame.groupby(['scenario',dimension]):
            stats.append({'scenario':family,'stratum':dimension,'value':value,'count':len(g),**{f'P_{name}':float((g.action==a).mean()) for a,name in [(-1,'yield'),(0,'maintain'),(1,'proceed')]}})
    pd.DataFrame(stats).to_csv(out/'phase1_action_choice_distribution.csv',index=False)
    return frame

def coverage(out):
    d=np.load('outputs/smoke/data.npz');counts={(f,label):0 for f in ['intersection','merge'] for label in LABELS};relevant=0
    lookup={(-1,-1):'YY',(-1,0):'YM',(-1,1):'YP',(0,-1):'MY',(0,0):'MM',(0,1):'MP',(1,-1):'PY',(1,0):'PM',(1,1):'PP'}
    for ep,family in enumerate(d['families']):
        lanes=4 if family=='intersection' else 2
        for t,s in enumerate(d['states'][ep,:-1]):
            for i,j in itertools.combinations(np.flatnonzero(d['control'][ep,t]),2):
                if pair_relevant(s,i,j,family,i%lanes,j%lanes):
                    label=lookup[tuple(d['actions'][ep,t,[i,j]])];counts[family,label]+=1;relevant+=1
    rows=[{'scenario':f,'joint_action':a,'count':count,'actual_conflict_relationship':True,'both_within_horizon':True,
       'same_state_counterfactual_siblings':0} for (f,a),count in counts.items()]
    frame=pd.DataFrame(rows);frame.to_csv(out/'factorial_action_coverage.csv',index=False);return frame

@torch.no_grad()
def factorial_audit(cfg,models,out):
    rows=[];strength=[];sensitivity=[];oracle=[]
    for index in range(24):
        env,frames=make_case(cfg,index,'audit');plans=factorial_plans(env,cfg['sibling_horizon']);values=[]
        for k,plan in enumerate(plans):
            states,c=true_rollout(env,plan);cost=float((c@cfg['objective_weights']).sum());values.append(cost)
            rows.append({'family_id':index,'scenario':env.family,'joint_action':LABELS[k],'a':int(plan[0,0]),'b':int(plan[0,1]),
                'objective':cost,'delay':float(c[:,0].sum()),'progress':float(c[:,2].sum()),'near_conflict':float(c[:,4].sum()),'collision':float(c[:,3].sum()),
                **{f'{xy}{i}':float(states[-1,i,j]*100) for i in [0,1] for j,xy in enumerate(['x','y'])}})
        residual=interaction_residual(values)
        strength.append({'family_id':index,'scenario':env.family,'interaction_rms':float(np.sqrt((residual**2).mean())),
           'best_joint_action':LABELS[int(np.argmin(values))],'PP_minus_best':values[8]-min(values)})
        for name,m in models.items():
            if name not in ['orchestra','independent']:continue
            for relation in ['interacting','distant']:
                observations=[({k:v.copy() for k,v in o.items()},None if a is None else a.copy(),s.copy()) for o,a,s in frames]
                if relation=='distant':
                    # Move the target's observable position back along its own direction; retain all other tokens.
                    for o,_,_ in observations:o['agents'][1,:2]-=o['agents'][1,3:5]*.6;o['agents'][1,10]=0
                state=infer(m,observations);a=np.zeros((2,6,env.n),int);a[0,:,0]=-1;a[1,:,0]=1
                pred=m.imagine(state.repeat(2),torch.tensor(a))['agents'].numpy()
                effect=np.linalg.norm((pred[1,-1,1,:2]-pred[0,-1,1,:2])*100)/2
                sensitivity.append({'family_id':index,'scenario':env.family,'model':name,'relation':relation,'cross_effect_m_per_action':float(effect)})
        if index<8:
            old=load_config('configs/smoke.yaml')
            for label,cfgp in [('phase1_horizon',old),('longer_horizon',cfg),('balanced_longer_horizon',cfg)]:
                if label=='balanced_longer_horizon':
                    score=lambda plans:np.array([(true_rollout(env,p)[1]@cfgp['objective_weights']).sum() for p in plans])
                    plan,cost,_=balanced_cem(score,cfgp['plan_horizon'],env.n,cfgp['cem_iterations'],np.random.default_rng(100+index),frames[-1][0]['control_mask'])
                else:plan,cost,_=oracle_plan(env,cfgp,np.random.default_rng(100+index))
                oracle.append({'family_id':index,'scenario':env.family,'planner':label,'a':int(plan[0,0]),'b':int(plan[0,1]),'objective':cost})
    pd.DataFrame(rows).to_csv(out/'simulator_factorial_interactions.csv',index=False)
    st=pd.DataFrame(strength);st.to_csv(out/'simulator_interaction_strength.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(out/'cross_agent_action_sensitivity.csv',index=False)
    pd.DataFrame(oracle).to_csv(out/'oracle_horizon_audit.csv',index=False)
    if st.interaction_rms.mean()<=.05 or (st.interaction_rms>.05).mean()<1/3:
        raise RuntimeError('Factorial scenarios have insufficient nonadditivity; revise scenario distribution before training.')
    return st
