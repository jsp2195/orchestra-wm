"""Post-selection validation diagnostics, never a model-selection score.

Check finite-ensemble point error and within-scene historical speed strata.
The primary eight-sample joint scores and all selected checkpoints stay fixed.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from orchestra_wm.i24.data import sha
from orchestra_wm.i24_phase4b.metrics import marginal_crps
from orchestra_wm.i24_phase4d.training import load_cache
from orchestra_wm.i24_phase4d.model import ResidualFields
from orchestra_wm.i24_continuous.source import write_immutable

root=Path('outputs/i24_phase4d');out=root/'precision_audit';out.mkdir(exist_ok=True)
run=root/'residual_seed101';cp=run/'checkpoints/best.pt';done=json.loads((run/'completed.json').read_text());assert sha(cp)==done['best_checkpoint_sha256']
s=torch.load(cp,weights_only=False);models={v:ResidualFields() for v in ['full','aggregate']}
for v,m in models.items():m.load_state_dict(s['models'][v]);m.eval()
torch.set_num_threads(2)
write_immutable(out/'protocol.json',dict(source_sha256=sha(Path(__file__)),checkpoint_sha256=sha(cp),samples=64,scope='Post-selection descriptive validation sensitivity only; primary eight-particle scores unchanged; no test access'))
if (out/'summary.json').exists():raise SystemExit('Completed precision audit preserved')
_,items=load_cache(root/'cache_v1');rows=[]
with torch.no_grad():
    for e,b in items:
        if e['split']!='validation' or e['regime']!='dense':continue
        for v,m in models.items():
            x=m(b,v,64,11003+e['number']).numpy()
            for h in [5,10,20]:
                t=5*h-1;use=b['support'][t].numpy();z=x[:,t,use];y=b['target'][t].numpy()[use]
                for k,name in enumerate(['speed','density','flow']):
                    rows.append(dict(variant=v,scene=e['scene'],horizon_seconds=h,component=name,mae64=float(abs(z.mean(0)-y)[:,k].mean()),bias64=float((z.mean(0)-y)[:,k].mean()),crps64=float(marginal_crps(z,y)[:,k].mean())))
frame=pd.DataFrame(rows);frame.to_csv(out/'point_precision.csv',index=False);frame.groupby(['variant','horizon_seconds','component']).mean(numeric_only=True).to_csv(out/'summary_table.csv')

rows=[];manifest=json.loads((root/'validation/arrays_manifest.json').read_text())
for e in manifest:
    if e['regime']!='dense' or e['variant'] not in ['residual_full','residual_aggregate','aggregate_persistence','diffusion_persistence']:continue
    with np.load(root/'validation'/e['path']) as a:
        reference=a['reference'];known=reference[:,1]>0
        history_v=reference[:,0];masks=[('congested_<10',known&(history_v<10)),('intermediate_10_20',known&(history_v>=10)&(history_v<20)),('free_>=20',known&(history_v>=20))]
        for h in [5,10,20]:
            t=h*5-1
            for regime,mask in masks:
                use=mask&a['support'][t]
                if not use.any():continue
                z=a['fields'][:,t,use];y=a['target'][t,use]
                row=dict(variant=e['variant'],seed=e['seed'],scene=e['scene'],horizon_seconds=h,bin_history_speed_regime=regime,bins=int(use.sum()))
                for k,name in enumerate(['speed','density','flow']):row[name+'_mae']=float(abs(z.mean(0)-y)[:,k].mean());row[name+'_crps']=float(marginal_crps(z,y)[:,k].mean())
                rows.append(row)
f=pd.DataFrame(rows);f.to_csv(out/'bin_regime_errors.csv',index=False)
f.groupby(['variant','seed','horizon_seconds','bin_history_speed_regime']).agg(windows=('scene','size'),scored_bins=('bins','sum'),speed_mae=('speed_mae','mean'),density_mae=('density_mae','mean'),flow_mae=('flow_mae','mean'),speed_crps=('speed_crps','mean'),density_crps=('density_crps','mean'),flow_crps=('flow_crps','mean')).to_csv(out/'bin_regime_summary.csv')
write_immutable(out/'summary.json',dict(point_precision_sha256=sha(out/'point_precision.csv'),bin_regime_errors_sha256=sha(out/'bin_regime_errors.csv'),note='Scene-equal means within nonempty historical-bin strata; bins and windows are correlated; descriptive only'))
print('Validation point precision and historical-bin regime diagnostics completed')
