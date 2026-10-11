"""Training/validation-only diagnosis; never opens developmental-test arrays."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from orchestra_wm.i24.data import load_scene,sha
from orchestra_wm.i24_continuous.data import observations,exposure_fields

root=Path('outputs/i24_phase4b/development_v1');out=Path('outputs/i24_phase4d/diagnosis');out.mkdir(parents=True,exist_ok=True)
manifest=json.loads((root/'data_manifest.json').read_text());distribution=[]
for item in manifest['scenes']:
    if item['split'] not in ('train','validation'):continue
    with np.load((root/item['path']).with_suffix('.fields.npz')) as a:
        f,s=a['fields'],a['support'];v=f[s]
        distribution.append(dict(split=item['split'],scene=item['path'],mean_speed=float(v[:,0].mean()),p10_speed=float(np.quantile(v[:,0],.1)),p90_speed=float(np.quantile(v[:,0],.9)),mean_density=float(v[:,1].mean()),mean_flow=float(v[:,2].mean()),support_fraction=float(s.mean()),fraction_speed_over20=float((v[:,0]>20).mean()),fraction_speed_under10=float((v[:,0]<10).mean())))
pd.DataFrame(distribution).to_csv(out/'source_distribution.csv',index=False)
rows=[]
for label,where in [('macro_control','outputs/i24_phase4c/matched_macro_seed101_u300/validation'),('joint_trial','outputs/i24_phase4c/joint_seed101_u30/evaluation/validation')]:
    where=Path(where);entries=json.loads((where/'arrays_manifest.json').read_text())
    baselines={(e['scene'],e['regime']):e for e in entries if e['variant']=='aggregate_persistence'}
    for e in entries:
        if e['variant'] in ('aggregate_persistence','constant_velocity'):continue
        with np.load(where/e['path']) as a,np.load(where/baselines[e['scene'],e['regime']]['path']) as b:
            for horizon in (.2,1,5,10,20):
                t=round(horizon/.2)-1;use=a['field_support'][t];x=a['fields'][:,t,use].mean(0);y=a['target_fields'][t,use];ref=b['fields'][0,t,use]
                row=dict(experiment=label,variant=e['variant'],regime=e['regime'],scene=e['scene'],seconds=horizon,bins=int(use.sum()))
                for k,name in enumerate(['speed','density','flow']):
                    row.update({name+'_bias':float((x-y)[:,k].mean()),name+'_mean':float(x[:,k].mean()),name+'_target':float(y[:,k].mean()),name+'_persistence_mae':float(abs(ref-y)[:,k].mean()),name+'_mae':float(abs(x-y)[:,k].mean()),name+'_reference_departure':float(abs(x-ref)[:,k].mean()),name+'_scaled_mse':float(np.mean(((x-y)[:,k]/[30,100,3000][k])**2))})
                mask=a['valid'][:t+1];err=np.linalg.norm(a['agents'][:,:t+1,:,:2].mean(0)-a['truth'][:t+1,:,:2],axis=-1)
                row['micro_ade']=float(err[mask].mean()) if mask.any() else None
                rows.append(row)
frame=pd.DataFrame(rows);frame.to_csv(out/'validation_bias_and_drift.csv',index=False)
frame.groupby(['experiment','variant','regime','seconds']).mean(numeric_only=True).to_csv(out/'validation_summary.csv')
curves=[]
for directory in ['outputs/i24_phase4b/matched_macro_seed101_u300','outputs/i24_phase4c/joint_seed101_u30']:
    for p in Path(directory).glob('*_learning_curve.csv'):
        f=pd.read_csv(p);key='validation_macro_energy'
        curves.append(dict(experiment=directory,variant=p.stem,updates=int(f.step.max()),nonfinite_gradients=int((~np.isfinite(f.gradient_norm)).sum()),gradient_median=float(f.gradient_norm.median()),gradient_p95=float(f.gradient_norm.quantile(.95)),fraction_clipped=float((f.gradient_norm>1).mean()),validation_best=float(f[key].min()),validation_final=float(f[key].dropna().iloc[-1]),validation_best_step=int(f.loc[f[key].idxmin(),'step'])))
pd.DataFrame(curves).to_csv(out/'optimization.csv',index=False)
result=dict(scope='Only frozen train and validation windows; no new developmental or independent test reads',data_manifest_sha256=sha(root/'data_manifest.json'),source_sha256=sha(Path(__file__)),distribution=pd.DataFrame(distribution).groupby('split').mean(numeric_only=True).to_dict('index'),normalization='Inputs and scores use physical [30,100,3000]; absolute decoder softplus(raw)*0.25*scale has raw-zero speed 5.1986 m/s. This is an initialization reference, not a measured learned cap.',optimization=curves)
(out/'diagnosis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
