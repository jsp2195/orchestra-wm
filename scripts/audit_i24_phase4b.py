"""Measured post-hoc audit of preserved pilot arrays, never new training results."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from orchestra_wm.i24.data import sha,load_scene
from orchestra_wm.i24_continuous.data import exposure_fields
from orchestra_wm.i24_phase4b.metrics import marginal_crps,sample_diversity,boundary_rates

pilot=Path('outputs/i24_continuous/pilot_0700_wb');output=Path('outputs/i24_phase4b/audit')
manifest=json.loads((pilot/'evaluation_manifest.json').read_text())
data=json.loads((pilot/'data_manifest.json').read_text())
road=load_scene(pilot/data['scenes'][0]['path']).road
rows=[]
for entry in manifest['entries']:
    path=pilot/entry['path'];assert sha(path)==entry['sha256']
    with np.load(path) as a:
        agents=a['agents'];fields=a['fields'];truth=a['truth'];valid=a['valid'];target=a['target_fields'];support=a['field_support']
        for seconds in (5,10,20):
            step=round(seconds/.2);fs=support[step-1];x=fields[:,step-1,fs];y=target[step-1,fs]
            lo,hi=np.quantile(x,[.05,.95],axis=0)
            crps=marginal_crps(x,y).mean(0)
            rates=boundary_rates(agents[:,:step],valid[:step],road)
            mask=valid[:step];error=np.linalg.norm(agents[:,:step,:,:2]-truth[None,:step,:,:2],axis=-1)
            row=dict(scene=entry['scene_number'],variant=entry['variant'],regime=entry['regime'],horizon_seconds=seconds,
                     crps_speed=crps[0],crps_density=crps[1],crps_flow=crps[2],
                     macro_diversity=sample_diversity(x/np.array([30,100,3000])),
                     interval_width_speed=float((hi-lo)[:,0].mean()),interval_width_density=float((hi-lo)[:,1].mean()),
                     interval_width_flow=float((hi-lo)[:,2].mean()),
                     negative_macro_component_rate=float((x<0).mean()),
                     expected_sample_ade_m=float(error[:,mask].mean()) if mask.any() and entry['variant']!='macro_only' else None,
                     **{k:v if entry['variant']!='macro_only' else None for k,v in rates.items()})
            if entry['regime']=='dense' and entry['variant']!='macro_only':
                derived=[]
                for sample in range(len(agents)):
                    f,_=exposure_fields(np.concatenate([a['history'],agents[sample]]),np.concatenate([a['history_valid'],a['predicted_valid'][sample]]),road)
                    derived.append(f[24+step])
                derived=np.asarray(derived)
                row['decoder_vs_hard_generated_macro_rmse_scaled']=float(np.sqrt(np.mean(((fields[:,step-1]-derived)/np.array([30,100,3000]))**2)))
                cohort_fields,_=exposure_fields(np.concatenate([a['history'],truth]),np.concatenate([a['history_valid'],valid]),road)
                row['future_cohort_density_fraction']=float(cohort_fields[24+step,:,1].sum()/max(target[step-1,:,1].sum(),1e-9))
            rows.append(row)
    print('Audited',entry['scene_number'],entry['regime'],entry['variant'],flush=True)
frame=pd.DataFrame(rows);frame.to_csv(output/'supplementary_pilot_metrics.csv',index=False)
summary=frame[(frame.regime=='dense')&(frame.horizon_seconds==10)].groupby('variant').mean(numeric_only=True)
summary.to_csv(output/'supplementary_dense10.csv')
curves=pd.read_csv(pilot/'training.csv');convergence=[]
for variant,g in curves.groupby('variant'):
    last=g[g.step>=80]
    convergence.append(dict(variant=variant,final_validation=float(g.iloc[-1].validation_loss),
                            validation_change_80_to_100=float(last.iloc[-1].validation_loss-last.iloc[0].validation_loss),
                            interpretation='Three late measurements cannot establish convergence; objectives differ across variants'))
(output/'convergence_audit.json').write_text(json.dumps(convergence,indent=2)+'\n')
counts=data['counts']
result=dict(source='POST_HOC_AUDIT_OF_PRESERVED_PILOT_NOT_NEW_TRAINING',
            original_evaluation_sha256=sha(pilot/'evaluation.csv'),supplementary_sha256=sha(output/'supplementary_pilot_metrics.csv'),
            rows=len(rows),omitted_crossing_tracks=counts['crossing_or_guard_tracks_omitted'],
            omitted_samples=counts['crossing_or_guard_samples_omitted'],
            omitted_sample_fraction=counts['crossing_or_guard_samples_omitted']/sum(v for k,v in counts.items() if k.endswith('_samples') or k=='crossing_or_guard_samples_omitted'),
            conclusions=['No re-scoring replaces historical primary result','Single examined day; no independent-day confidence interval',
                         'Micro ADE/FDE in original table score ensemble mean; expected sample ADE is separate here',
                         'Exit and lateral departure rates separated; original combined rate is not solely a physical violation',
                         'CRPS uses off-diagonal unbiased ensemble estimator; small ensembles give noisy estimates'])
(output/'measured_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
