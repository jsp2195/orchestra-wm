"""64-particle validation calibration audit, separate from eight-sample primary scores."""
import json,argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from orchestra_wm.i24.data import sha
from orchestra_wm.i24_continuous.source import write_immutable
from orchestra_wm.i24_phase4e.cache import load_cache
from orchestra_wm.i24_phase4d.model import ResidualFields
from orchestra_wm.i24_phase4d.baselines import DiffusionPersistence
from orchestra_wm.i24_phase4b.metrics import marginal_crps
p=argparse.ArgumentParser();p.add_argument('--run',default='outputs/i24_phase4e/residual_seed101');p.add_argument('--output',default='outputs/i24_phase4e/calibration64');a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=True);root=Path(a.run);done=json.loads((root/'completed.json').read_text());cp=root/'checkpoints/best.pt';assert sha(cp)==done['best_checkpoint_sha256'];s=torch.load(cp,weights_only=False)
torch.set_num_threads(2);models={v:ResidualFields() for v in ['full','aggregate']}
for v,m in models.items():m.load_state_dict(s['models'][v]);m.eval()
diff=DiffusionPersistence();diff.load_state_dict(torch.load('outputs/i24_phase4e/diffusion_baseline/checkpoints/last.pt',weights_only=False)['model']);models['diffusion']=diff
write_immutable(out/'protocol.json',dict(samples=64,seed=s["config"]["seed"],checkpoint_sha256=sha(cp),source_sha256=sha(Path(__file__)),scope='Validation dense only; diagnostic, does not replace eight-sample comparison',rank_method='Randomized (number of samples below truth + uniform[0,1])/(S+1); histogram only, correlated observations, no iid p-values'))
if (out/'summary.json').exists():raise SystemExit('Calibration audit already complete')
_,items=load_cache('outputs/i24_phase4e/cache_v1');rows=[];histograms={};rng=np.random.default_rng(3121)
with torch.no_grad():
 for e,b in items:
  if e['split']!='validation' or e['regime']!='dense':continue
  for v,m in models.items():
   x=(m(b,64,11003+e['number']) if v=='diffusion' else m(b,v,64,11003+e['number'])).numpy()
   for horizon in [5,10,20]:
    t=5*horizon-1;mask=b['support'][t].numpy();z=x[:,t,mask];y=b['target'][t].numpy()[mask];lo,hi=np.quantile(z,[.05,.95],axis=0)
    ranks=(z<y).sum(0);ties=(z==y).sum(0);pit=(ranks+rng.uniform(size=ranks.shape)*(ties+1))/65
    for k,name in enumerate(['speed','density','flow']):
     rows.append(dict(variant=v,horizon_seconds=horizon,scene=e['scene'],component=name,quantile_coverage90=float(((y>=lo)&(y<=hi))[:,k].mean()),randomized_rank_central90=float(((pit>=.05)&(pit<=.95))[:,k].mean()),width90=float((hi-lo)[:,k].mean()),bins=int(mask.sum()),mae=float(abs(z.mean(0)-y)[:,k].mean()),bias=float((z.mean(0)-y)[:,k].mean()),crps=float(marginal_crps(z,y)[:,k].mean()),predictive_sd=float(z[:,:,k].std(0).mean())))
     key=f'{v}/{horizon}/{name}';histograms.setdefault(key,np.zeros(10,int));histograms[key]+=np.histogram(pit[:,k],np.linspace(0,1,11))[0]
frame=pd.DataFrame(rows);frame.to_csv(out/'calibration.csv',index=False);frame.groupby(['variant','horizon_seconds','component']).mean(numeric_only=True).to_csv(out/'summary_table.csv');write_immutable(out/'summary.json',dict(csv_sha256=sha(out/'calibration.csv'),rank_histograms={k:v.tolist() for k,v in histograms.items()},qualification='Finite-particle and dependent validation observations; not independent-date calibration evidence'));print('64-particle calibration audit complete')
