import time
import numpy as np
import pandas as pd
import torch
from orchestra_wm.evaluation.common import context,infer,errors
from orchestra_wm.evaluation.counterfactuals import candidates,rank_metrics
from orchestra_wm.planning.orchestrator import Orchestrator
from orchestra_wm.planning.oracle import true_rollout
from orchestra_wm.evaluation.planning import run_controller

@torch.no_grad()
def evaluate_conditions(cfg,model,out):
    rows=[]
    conditions={
      'ID':({},'intersection'),
      'density_OOD':({'vehicles':16},'intersection'),
      'behavior_OOD':({'behavior_scale':1.4},'intersection'),
      'topology_OOD':({},'corridor'),
      'grid_OOD':({},'grid'),
      'geometry_OOD':({'geometry_scale':1.3},'intersection'),
      'combined_OOD':({'vehicles':20,'behavior_scale':1.4,'sensor_regime':'dropout'},'grid'),
      'stopped_vehicle':({'stopped_vehicle':True},'intersection'),
      'blocked_lane':({'blocked_lane':True},'intersection'),
      'full_sensing':({'sensor_regime':'full'},'intersection'),
      'sparse_sensing':({'sensor_regime':'sparse'},'intersection'),
      'probe_only':({'sensor_regime':'probe'},'intersection'),
      'temporary_outage':({'sensor_regime':'dropout','miss_probability':.4},'intersection'),
      'without_lane_sensors':({'withhold_lanes':True},'intersection')}
    for name,(updates,family) in conditions.items():
        c={**cfg,**updates,'planning_steps':8}
        for seed in cfg['seeds']:
            env,frames=context(c,family,50000+seed,steps=12)
            if updates.get('withhold_lanes'):
                # Keep public geometry; withhold the dynamic loop measurements only.
                for obs,_,_ in frames: obs['lanes'][:,:4]=0
            state=infer(model,frames);plans=candidates(env,c,seed)
            p=model.imagine(state.repeat(len(plans)),torch.tensor(plans,device=state.memory.device))
            costs=(p['components'].cpu().numpy()@cfg['objective_weights']).sum(1)
            truthcost=[];truth=None
            for i,plan in enumerate(plans):
                states,comps=true_rollout(env,plan);truthcost.append((comps@cfg['objective_weights']).sum())
                if i==0: truth=states
            prediction=p['agents'][0].cpu().numpy();err=errors(prediction,truth)
            hidden=~frames[-1][0]['mask']
            hidden_error=errors(prediction[:,hidden],truth[:,hidden])['position_error'] if hidden.any() else np.nan
            rank=rank_metrics(costs,np.array(truthcost))
            planned=run_controller(c,family,seed,'orchestra',model)
            noop=run_controller(c,family,seed,'noop')
            rows.append({'condition':name,'seed':seed,'scenario':family,**err,'hidden_position_error':hidden_error,
                'plan_spearman':rank['spearman'],'plan_regret':rank['regret'],
                'planning_objective':planned['objective'],'noop_objective':noop['objective']})
    result=pd.DataFrame(rows);result.to_csv(out/'generalization.csv',index=False)
    return result

@torch.no_grad()
def evaluate_scaling(cfg,model,out):
    rows=[]
    for n in [4,8,16,24,32]:
        for seed in cfg['seeds']:
            c={**cfg,'vehicles':n,'connected':min(4,n),'planning_steps':4}
            env,frames=context(c,'intersection',seed+60000)
            s=infer(model,frames);plans=candidates(env,c,seed)
            started=time.perf_counter()
            p=model.imagine(s,torch.tensor(plans[0],device=s.memory.device)[None])
            inference=(time.perf_counter()-started)*1000
            truth,_=true_rollout(env,plans[0])
            started=time.perf_counter();best,_,_=Orchestrator(model,c,seed).plan(s)
            latency=(time.perf_counter()-started)*1000
            _,components=true_rollout(env,best)
            rows.append({'seed':seed,'agents':n,'inference_ms':inference,'planner_ms':latency,
                 'selected_true_cost':float((components@cfg['objective_weights']).sum()),
                 **errors(p['agents'][0].cpu().numpy(),truth)})
    frame=pd.DataFrame(rows);frame.to_csv(out/'scaling.csv',index=False);return frame
