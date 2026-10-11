"""Exploratory observed-field evidence on training blocks only; no wave labels."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from orchestra_wm.i24.data import sha
from orchestra_wm.i24_continuous.source import write_immutable

root=Path('outputs/i24_phase4b/development_v1')
out=Path('outputs/i24_phase4c/recorded_dynamics');out.mkdir(parents=True,exist_ok=True)
manifest=json.loads((root/'data_manifest.json').read_text());results=[]
fig,axs=plt.subplots(2,1,figsize=(12,7),sharex=True)
for ax,block in zip(axs,['0615_train','0715_train']):
    items=sorted((i for i in manifest['scenes'] if i['split']=='train' and i['block']==block),key=lambda i:i['start'])
    assert all(abs(b['start']-a['start']-25)<1e-6 for a,b in zip(items,items[1:]))
    values=[];masks=[]
    for item in items:
        path=(root/item['path']).with_suffix('.fields.npz')
        assert sha(path)==manifest['artifacts'][str(path.relative_to(root))]
        with np.load(path) as a:values.append(a['fields'][:125]);masks.append(a['support'][:125])
    fields=np.concatenate(values).reshape(-1,4,8,3);support=np.concatenate(masks).reshape(-1,4,8)
    density=np.where(support,fields[...,1],0).sum(1);flow=np.where(support,fields[...,2],0).sum(1)
    speed=np.divide(flow,3.6*density,out=np.full_like(flow,np.nan),where=(density>0)&(support.sum(1)>=2))
    # Centered detrending is an offline diagnostic, never a model input.
    series=pd.DataFrame(speed);smooth=series.rolling(100,min_periods=50,center=True).mean()
    residual=series-series.rolling(300,min_periods=150,center=True).mean()
    im=ax.imshow(smooth.to_numpy().T,origin='lower',aspect='auto',extent=[0,len(speed)*.2/60,0,999.744],vmin=0,vmax=35,cmap='viridis')
    ax.set_title(block+' · 20 s centered display smoothing; density-weighted supported lanes');ax.set_ylabel('Westbound s (m)');fig.colorbar(im,ax=ax,label='m/s')
    for b in range(7):
        x=residual[b].to_numpy();y=residual[b+1].to_numpy();scores=[]
        for lag in range(-150,151,5):
            u,v=(x[:lag],y[-lag:]) if lag<0 else (x[lag:],y[:-lag]) if lag>0 else (x,y)
            # Convention: downstream(t), upstream(t+lag); positive lag means upstream propagation.
            use=np.isfinite(u)&np.isfinite(v)
            corr=float(np.corrcoef(u[use],v[use])[0,1]) if use.sum()>300 and min(u[use].std(),v[use].std())>1e-6 else float('nan')
            scores.append((corr,lag*.2,int(use.sum())))
        finite=[s for s in scores if np.isfinite(s[0])]
        best=max(finite,key=lambda s:s[0]) if finite else (None,None,0)
        lag=best[1]
        results.append(dict(block=block,upstream_bin=b,downstream_bin=b+1,peak_correlation=best[0],upstream_delay_seconds=lag,
                            apparent_wave_speed_m_s=-(999.744/8)/lag if lag else None,
                            paired_frames=best[2],lag_at_search_boundary=abs(lag)==30 if lag is not None else None))
axs[-1].set_xlabel('Minutes within fixed training block');fig.suptitle('Actual reconstructed traffic fields · exploratory evidence, no verified wave-event annotation');fig.tight_layout();fig.savefig(out/'observed_training_kymographs.png',dpi=150);plt.close(fig)
pd.DataFrame(results).to_csv(out/'adjacent_bin_lag_diagnostics.csv',index=False)
write_immutable(out/'analysis.json',dict(data_manifest_sha256=sha(root/'data_manifest.json'),source_sha256=sha(Path(__file__)),
    scope='Only preassigned training blocks; no developmental-test outcomes accessed',
    method='Adjacent-bin Pearson peak across -30..30 s lags, 1 s spacing, after centered 60 s detrending; two or more supported lanes required',
    lag_convention='Positive delay means upstream follows downstream; apparent speed is -124.968/delay m/s',
    inference='Exploratory correlation with multiple-lag selection and temporal dependence; not an independently verified wave or onset detector',
    generated_wave_reproduction='UNTESTED; 20-second forecasts do not establish reproduction of minute-scale propagation',
    csv_sha256=sha(out/'adjacent_bin_lag_diagnostics.csv'),figure_sha256=sha(out/'observed_training_kymographs.png')))
print(pd.DataFrame(results).to_string(index=False))
