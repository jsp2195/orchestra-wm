from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import torch
from orchestra_wm.phase2.scenarios import make_case,factorial_plans,LABELS,pair_relevant
from orchestra_wm.evaluation.common import write_json


def collect_factorial(cfg,out):
    keys=['obs','mask','lanes','control','states','lane_targets','actions','components','ids']
    data={k:[] for k in keys};metadata=[];flags=[]
    for split,count in [('train',cfg['train_families']),('validation',cfg['validation_families'])]:
        for family_id in range(count):
            env,frames=make_case(cfg,family_id,split)
            plans=factorial_plans(env,cfg['sibling_horizon'])
            snapshot_hash=hashlib.sha256(env.state_array().tobytes()).hexdigest()
            for k,plan in enumerate(plans):
                e=env.clone();r={key:[] for key in keys};heavy=[]
                def record(o,s,lt):
                    r['obs'].append(o['agents']);r['mask'].append(o['mask']);r['control'].append(o['control_mask']);r['ids'].append(o['ids'])
                    r['states'].append(s);r['lanes'].append(np.pad(o['lanes'],((0,4-len(o['lanes'])),(0,0))))
                    r['lane_targets'].append(np.pad(lt,((0,4-len(lt)),(0,0))))
                    heavy.append(pair_relevant(s,0,1,e.family,e.vehicles[0].lane,e.vehicles[1].lane))
                for t,(o,a,s) in enumerate(frames):
                    record(o,s,o['lanes'][:,[0,2]])
                    if t<len(frames)-1:r['actions'].append(frames[t+1][1]);r['components'].append(np.zeros(6))
                for action in plan:
                    obs,_,_,_,info=e.step(action);r['actions'].append(action);r['components'].append(info['components'])
                    record(obs,e.state_array(),e.lane_targets())
                for key in keys:data[key].append(np.asarray(r[key]))
                flags.append(heavy)
                metadata.append({'sample':len(metadata),'split':split,'family_id':f'{split}_{family_id}',
                    'scenario':e.family,'joint_action':LABELS[k],'initial_state_sha256':snapshot_hash,'initial_relevant':heavy[cfg['context']-1]})
    path=out/'dataset_v2_factorial';path.mkdir(exist_ok=True)
    np.savez_compressed(path/'data.npz',**{k:np.array(v) for k,v in data.items()},interaction_heavy=np.array(flags),
                        family_ids=np.array([m['family_id'] for m in metadata]),splits=np.array([m['split'] for m in metadata]))
    frame=pd.DataFrame(metadata);frame.to_csv(path/'families.csv',index=False)
    coverage=frame.groupby(['split','scenario','joint_action']).size().rename('count').reset_index();coverage.to_csv(out/'factorial_v2_coverage.csv',index=False)
    assert all(len(g)==9 and g.initial_state_sha256.nunique()==1 for _,g in frame.groupby('family_id'))
    assert set(frame[frame.split=='train'].family_id).isdisjoint(frame[frame.split=='validation'].family_id)
    write_json(path/'manifest.json',{'episodes':len(frame),'families':frame.family_id.nunique(),
        'transitions':len(frame)*cfg['sibling_horizon'],'split_by':'entire counterfactual family',
        'balanced_action_counts':coverage.to_dict('records'),'initial_relevant_fraction':float(frame.initial_relevant.mean()),
        'hidden_metadata_not_inputs':['interaction_heavy','family_ids','initial_state_sha256']})

class SiblingDataset:
    def __init__(self,path):
        self.data=dict(np.load(path));self.groups={split:[] for split in ['train','validation']}
        for split in self.groups:
            for fid in np.unique(self.data['family_ids'][self.data['splits']==split]):
                indices=np.flatnonzero(self.data['family_ids']==fid);assert len(indices)==9
                self.groups[split].append(indices)
    def sample(self,batch,rng,device,validation=False):
        groups=self.groups['validation' if validation else 'train']
        indices=np.concatenate([groups[rng.integers(len(groups))] for _ in range(batch//9)])
        keys=['obs','mask','lanes','control','states','lane_targets','actions','components']
        return {k:torch.as_tensor(self.data[k][indices],device=device) for k in keys}
