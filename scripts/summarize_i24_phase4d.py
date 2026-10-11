"""Small validation-only scientific figures; no new demo or public trajectory arrays."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from orchestra_wm.i24.data import sha

root=Path('outputs/i24_phase4d');out=root/'analysis';out.mkdir(exist_ok=True)
new=pd.read_csv(root/'validation/evaluation.csv')
old=pd.read_csv('outputs/i24_phase4c/matched_macro_seed101_u300/validation/evaluation.csv')
joint=pd.read_csv('outputs/i24_phase4c/joint_seed101_u30/evaluation/validation/evaluation.csv')
# Exact outcome/support comparison before using historical scores as a reference.
prior_root=Path('outputs/i24_phase4c/matched_macro_seed101_u300/validation')
old_entries=json.loads((prior_root/'arrays_manifest.json').read_text())
lookup={(e['scene'],e['regime']):e for e in old_entries if e['variant']=='aggregate_persistence'}
current_entries=json.loads((root/'validation/arrays_manifest.json').read_text())
checked=0
for e in current_entries:
    if e['variant']!='aggregate_persistence':continue
    if (e['scene'],e['regime']) not in lookup:continue
    previous=lookup[e['scene'],e['regime']]
    with np.load(root/'validation'/e['path']) as now, np.load(prior_root/previous['path']) as before:
        assert np.array_equal(now['support'],before['field_support'])
        assert np.array_equal(now['target'],before['target_fields'])
        assert np.allclose(now['fields'],before['fields'],rtol=0,atol=1e-5)
    checked+=1
keys=['scene','regime','horizon_seconds']
pair=new.query("variant=='aggregate_persistence'").merge(old.query("variant=='aggregate_persistence'"),on=keys,suffixes=('_new','_old'))
assert len(pair)==138  # Historical Phase4C ran dense/outage, not sparse50.
delta=float(abs(pair.macro_energy_new-pair.macro_energy_old).max());assert delta<1e-6
support_audit=dict(matched_scene_regimes=checked,matched_scene_horizons=len(pair),maximum_persistence_score_difference=delta,exact_target_and_support_match=True)
(out/'historical_comparability.json').write_text(json.dumps(support_audit,indent=2)+'\n')
drift=[]
for e in current_entries:
    if e['regime']!='dense' or e['variant'] not in ['residual_full','residual_aggregate']:continue
    with np.load(root/'validation'/e['path']) as a:
        for seconds in [.2,1,5,10,20]:
            t=round(seconds*5)-1;mask=a['support'][t];x=a['fields'][:,t,mask].mean(0);y=a['target'][t,mask];ref=a['reference'][mask]
            row=dict(variant=e['variant'],seed=e['seed'],scene=e['scene'],seconds=seconds)
            for k,name in enumerate(['speed','density','flow']):
                row[name+'_bias']=float((x-y)[:,k].mean());row[name+'_reference_departure']=float(abs(x-ref)[:,k].mean())
            drift.append(row)
pd.DataFrame(drift).to_csv(out/'residual_drift.csv',index=False)

historical=[]
for label,df in [('phase4c_macro',old),('phase4c_joint',joint)]:
    df=df.copy();df['variant']=label+'_'+df['variant'];df['seed']=101;historical.append(df)
cols=['variant','seed','regime','horizon_seconds','macro_energy','speed_mae','density_mae','flow_mae','speed_crps','density_crps','flow_crps']
combined=pd.concat([new[cols]]+[df[cols] for df in historical],ignore_index=True)
table=combined.groupby(cols[:4]).mean(numeric_only=True).reset_index();table.to_csv(out/'macro_comparisons.csv',index=False)
micro=pd.read_csv(root/'micro_seed101/validation/evaluation.csv')
micro_tables=[micro.assign(variant='residual_'+micro.variant)]
for label,df in [('phase4c_joint',joint)]:
    use=df[df.fde_m.notna()].copy();use['variant']=label+'_'+use.variant;micro_tables.append(use)
pd.concat(micro_tables).groupby(['variant','regime','horizon_seconds'])[['ade_m','fde_m']].mean().to_csv(out/'micro_comparisons.csv')

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130})
fig,ax=plt.subplots(1,3,figsize=(13,3.5))
for seed in [101,202]:
    curve=pd.read_csv(root/f'residual_seed{seed}/learning_curve.csv')
    for v in ['full','aggregate']:
        c=curve.dropna(subset=[v+'_validation']);ax[0].plot(c.step,c[v+'_validation'],label=f'{v}, {seed}',ls='-' if v=='full' else '--')
    for v in ['full','aggregate']:ax[1].plot(curve.step,curve[v+'_gradient'].rolling(25,min_periods=1).median(),label=f'{v}, {seed}',alpha=.75)
mc=pd.read_csv(root/'micro_seed101/learning_curve.csv')
for v in ['graph','independent']:
    c=mc.dropna(subset=[v+'_validation']);ax[2].plot(c.step,c[v+'_validation'],label=v)
for a,title in zip(ax,['Macro validation selection objective','Median gradient norm, 25 updates','Micro validation selection objective']):a.set(title=title,xlabel='Updates');a.legend(fontsize=8)
ax[0].set_ylabel('Mean 5/10/20 s energy');ax[1].axhline(1,c='gray',ls=':',label='clip threshold');ax[2].set_ylabel('Scaled micro energy')
fig.tight_layout();fig.savefig(out/'learning_curves.png');plt.close(fig)

fig,ax=plt.subplots(1,3,figsize=(13,3.5))
labels=['aggregate_persistence','diffusion_persistence','residual_full','residual_aggregate','capacity16_aggregate','phase4c_macro_full','phase4c_macro_macro_only']
for v in labels:
    f=table[(table.variant==v)&(table.regime=='dense')];f=f.groupby('horizon_seconds').mean(numeric_only=True)
    ax[0].plot(f.index,f.macro_energy,'o-',label=v.replace('residual_','').replace('phase4c_macro_','old '))
    if v in labels[:4]:ax[1].plot(f.index,f.speed_mae,'o-',label=v.replace('residual_',''))
for v in ['graph','independent','constant_velocity']:
    f=micro[(micro.variant==v)&(micro.regime=='dense')].groupby('horizon_seconds').mean(numeric_only=True);ax[2].plot(f.index,f.fde_m,'o-',label=v)
for a,title in zip(ax,['Macro energy: validation only','Mean speed error (m/s)','Cohort vehicle FDE (m)']):a.set(title=title,xlabel='Forecast seconds');a.legend(fontsize=7);a.set_xticks([5,10,20])
fig.tight_layout();fig.savefig(out/'baseline_comparisons.png');plt.close(fig)

cal=pd.read_csv(root/'calibration64/calibration.csv')
fig,ax=plt.subplots(1,3,figsize=(13,3.5))
for v in ['full','aggregate','diffusion']:
    f=cal[(cal.variant==v)&(cal.horizon_seconds==10)].groupby('component').mean(numeric_only=True).reindex(['speed','density','flow'])
    ax[0].plot(f.index,100*f.quantile_coverage90,'o-',label=v)
ax[0].axhline(90,color='gray',ls=':');ax[0].set(title='64-particle interval coverage, 10 s',ylabel='Coverage (%)');ax[0].legend()
for v in ['aggregate_persistence','diffusion_persistence','residual_full','residual_aggregate']:
    f=new[(new.variant==v)&(new.horizon_seconds==10)].groupby('regime').mean(numeric_only=True).reindex(['dense','sparse50','outage'])
    ax[1].plot(f.index,f.macro_energy,'o-',label=v.replace('residual_',''))
    bins=pd.read_csv(root/'precision_audit/bin_regime_summary.csv')
    f=bins[(bins.variant==v)&(bins.horizon_seconds==10)].groupby('bin_history_speed_regime').mean(numeric_only=True).reindex(['congested_<10','intermediate_10_20','free_>=20'])
    ax[2].plot(f.index,f.speed_mae,'o-',label=v.replace('residual_',''))
ax[1].set(title='Emulated observation loss, 10 s',ylabel='Macro energy');ax[1].legend(fontsize=7)
ax[2].set(title='Historical-bin speed strata, 10 s',ylabel='Speed MAE (m/s)');ax[2].tick_params(axis='x',labelsize=8)
fig.tight_layout();fig.savefig(out/'calibration_and_regimes.png');plt.close(fig)

# Fixed first chronological validation scene, not selected for favorable outcomes.
example=next(e for e in current_entries if e['scene_number']==0 and e['regime']=='dense' and e['variant']=='residual_full' and e['seed']==101)
with np.load(root/'validation'/example['path']) as a:
    fields=a['fields'];target=a['target'];support=a['support']
    fig,ax=plt.subplots(1,3,figsize=(12,3.6))
    for axis,k,name in zip(ax,range(3),['Speed (m/s)','Density (veh/km/lane)','Flow (veh/h/lane)']):
        denom=support.sum(-1).clip(1);y=(target[...,k]*support).sum(-1)/denom
        x=(fields[...,k]*support[None]).sum(-1)/denom;lo,hi=np.quantile(x,[.05,.95],axis=0);time=np.arange(1,101)*.2
        axis.plot(time,y,c='black',label='Recorded');axis.plot(time,x.mean(0),c='tab:blue',label='Generated mean');axis.fill_between(time,lo,hi,color='tab:blue',alpha=.2,label='8-sample 5–95%')
        axis.set(title=name,xlabel='Autonomous seconds')
    ax[0].legend(fontsize=8);fig.suptitle('First validation scene: spatial mean over scoring support (support is not a model input)',fontsize=10)
    fig.tight_layout();fig.savefig(out/'checkpoint_example.png');plt.close(fig)
(out/'figure_provenance.json').write_text(json.dumps(dict(source_sha256=sha(Path(__file__)),validation_csv_sha256=sha(root/'validation/evaluation.csv'),example=example,scope='Selected validation checkpoints; correlated scenes; no independent-date outcomes or public trajectory arrays'),indent=2)+'\n')
print(json.dumps(support_audit))
