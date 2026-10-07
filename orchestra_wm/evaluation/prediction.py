import numpy as np
import pandas as pd
import torch
from orchestra_wm.evaluation.common import paired_cases,infer,errors,aggregate_by_seed
from orchestra_wm.models.baselines import trivial_forecast
from orchestra_wm.planning.oracle import true_rollout

@torch.no_grad()
def evaluate_prediction(cfg,models,out):
    rows=[];shuffle=[]
    for seed,family,(env,frames) in paired_cases(cfg):
        horizon=max(cfg['horizons']);rng=np.random.default_rng(seed+33)
        blocks=rng.integers(-1,2,(int(np.ceil(horizon/4)),env.n))
        actions=np.repeat(blocks,4,axis=0)[:horizon]
        actions[:,~frames[-1][0]['control_mask']]=0
        true,components=true_rollout(env,actions)
        # Full lane target rollouts are computed independently for reporting.
        world=env.clone();lane_truth=[]
        for action in actions:
            world.step(action);lane_truth.append(world.lane_targets())
        lane_truth=np.array(lane_truth)
        for name,model in {**models,'persistence':None,'constant_velocity':None}.items():
            if model is None:
                base=torch.tensor(frames[-1][0]['agents'][:,:6])[None]
                pred=trivial_forecast(base,horizon,cfg['dt'],name=='constant_velocity')[0].numpy()
                lane_pred=np.repeat(frames[-1][0]['lanes'][:,[0,2]][None],horizon,axis=0)
            else:
                state=infer(model,frames);p=model.imagine(state,torch.tensor(actions,device=state.memory.device)[None])
                pred=p['agents'][0].cpu().numpy();lane_pred=p['lanes'][0].cpu().numpy()
            for h in cfg['horizons']:
                result=errors(pred[h-1],true[h-1])
                result['lane_error']=float(np.abs(lane_pred[h-1,:,0]-lane_truth[h-1,:,0]).mean())
                result['queue_error']=float(np.abs(lane_pred[h-1,:,1]-lane_truth[h-1,:,1]).mean())
                result['collision_proxy_error']=float(abs(p['components'][0,h-1,3].cpu()-components[h-1,3])) if model else np.nan
                rows.append({'seed':seed,'scenario':family,'model':name,'horizon':h,**result})
            shuffled=actions.copy();controlled=np.flatnonzero(frames[-1][0]['control_mask'])
            values=shuffled[:,controlled].ravel();rng.shuffle(values);shuffled[:,controlled]=values.reshape(horizon,-1)
            for mode,sequence in [('correct',actions),('shuffled',shuffled),('noop',np.zeros_like(actions)),('opposite',-actions)]:
                if model:
                    forecast=model.imagine(state,torch.tensor(sequence,device=state.memory.device)[None])['agents'][0].cpu().numpy()
                else: forecast=pred
                for h in cfg['horizons']:
                    shuffle.append({'seed':seed,'scenario':family,'model':name,'condition':mode,'horizon':h,**errors(forecast[h-1],true[h-1])})
    frame=pd.DataFrame(rows);frame.to_csv(out/'prediction.csv',index=False)
    aggregate_by_seed(frame,['model','horizon'],'position_error').to_csv(out/'prediction_aggregate.csv',index=False)
    pd.DataFrame(shuffle).to_csv(out/'action_shuffle.csv',index=False)
    return frame,pd.DataFrame(shuffle)
