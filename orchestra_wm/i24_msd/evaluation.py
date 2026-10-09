"""Held-out autonomous real-scene evaluation; saved arrays back every demo sample."""
from pathlib import Path
import hashlib
import json
import math

import numpy as np
import pandas as pd
import torch

from orchestra_wm.i24.data import observations
from orchestra_wm.i24.model import observe_history
from .acquisition import atomic_json, hashes, AcquisitionError
from .experiment import load_model


def energy(samples, truth):
    if not truth.size: return None
    x = samples.reshape(len(samples), -1); y = truth.reshape(-1)
    return float((np.linalg.norm(x-y,axis=-1).mean()-.5*np.linalg.norm(x[:,None]-x[None,:],axis=-1).mean())/math.sqrt(y.size))


def overlap(a, mask):
    # Oriented SAT with headings derived from predicted velocity. Degenerate
    # stationary headings use atan2(0,0)=0 and are disclosed as an approximation.
    collisions, pairs = 0, 0
    for t in range(a.shape[1]):
        for i in range(a.shape[2]):
            for j in range(i):
                if not (mask[t,i] and mask[t,j]): continue
                first, second = a[:,t,i], a[:,t,j]
                delta = second[:,:2]-first[:,:2]
                h1=np.arctan2(first[:,3],first[:,2]);h2=np.arctan2(second[:,3],second[:,2])
                u1=np.stack([np.cos(h1),np.sin(h1)],-1);v1=np.stack([-np.sin(h1),np.cos(h1)],-1)
                u2=np.stack([np.cos(h2),np.sin(h2)],-1);v2=np.stack([-np.sin(h2),np.cos(h2)],-1)
                hit=np.ones(len(a),bool)
                for axis in (u1,v1,u2,v2):
                    radius1=(np.abs((axis*u1).sum(-1))*first[:,6]+np.abs((axis*v1).sum(-1))*first[:,7])/2
                    radius2=(np.abs((axis*u2).sum(-1))*second[:,6]+np.abs((axis*v2).sum(-1))*second[:,7])/2
                    hit &= np.abs((delta*axis).sum(-1)) < radius1+radius2
                collisions += int(hit.sum());pairs += len(a)
    return (float(collisions/pairs) if pairs else None), pairs


def metric(pred, truth, mask):
    if not mask.any(): return None
    mean = pred.mean(0)
    distances = np.linalg.norm(mean[:,:,:2]-truth[:,:,:2],axis=-1)
    final = mask[-1]
    speed = np.abs(np.linalg.norm(mean[:,:,2:4],axis=-1)-np.linalg.norm(truth[:,:,2:4],axis=-1))
    low, high = np.quantile(pred[:,:,:,:2],[.05,.95],axis=0)
    coverage = ((truth[:,:,:2]>=low)&(truth[:,:,:2]<=high))[mask]
    samples = pred[:,:,:,:2][:,mask]; target = truth[:,:,:2][mask]
    crps = np.abs(samples-target).mean()-.5*np.abs(samples[:,None]-samples[None,:]).mean()
    collisions,pairs=overlap(pred,mask)
    true_collisions,_=overlap(truth[None],mask)
    heading=np.arctan2(mean[:,:,3],mean[:,:,2])-np.arctan2(truth[:,:,3],truth[:,:,2])
    angle=np.abs(np.arctan2(np.sin(heading),np.cos(heading)))
    acceleration=np.linalg.norm(np.diff(pred[:,:,:,2:4],axis=1)/.2,axis=-1)
    amask=mask[1:]&mask[:-1]
    return {'ade_m':float(distances[mask].mean()),'fde_m':float(distances[-1,final].mean()) if final.any() else None,
            'speed_mae_mps':float(speed[mask].mean()),'velocity_heading_error_rad':float(angle[mask].mean()),
            'joint_energy_m':energy(pred[:,-1,final,:2],truth[-1,final,:2]) if final.any() else None,
            'trajectory_energy_m':energy(samples,target),'position_crps_m':float(crps),
            'coverage90':float(coverage.mean()),'interval_width90_m':float((high-low)[mask].mean()),
            'diversity_rms_m':float(np.sqrt(np.var(samples,axis=0).mean())),
            'overlap_fraction':collisions,'overlap_pair_samples':pairs,'truth_overlap_fraction':true_collisions,
            'acceleration_over_8_fraction':float((acceleration[:,amask]>8).mean()) if amask.any() else None,
            'valid_agent_steps':int(mask.sum()),'valid_final_agents':int(final.sum()),'road_boundary_violation':None}


def boundary_metrics(pred, mask, provenance):
    """Footprint outside the two supplied finite edge polylines; no extrapolation."""
    edges=[np.array(m['polyline_xyz_m'])[:,:2] for m in provenance['source_map'] if m['type']=='road_edge' and len(m['polyline_xyz_m'])>1]
    if len(edges)!=2:return {'road_boundary_violation':None,'road_boundary_coverage':0.,'road_boundary_samples':0}
    x=pred[:,:,:,0].astype(float)*provenance['travel_sign']+provenance['x_origin_m']
    y=pred[:,:,:,1];covered=np.broadcast_to(mask,x.shape).copy();bounds=[]
    for edge in edges:
        order=np.argsort(edge[:,0]);edge=edge[order];xs,indices=np.unique(edge[:,0],return_index=True);ys=edge[indices,1]
        covered &= (x>=xs[0])&(x<=xs[-1]);bounds.append(np.interp(x,xs,ys))
    low=np.minimum(*bounds);high=np.maximum(*bounds)
    covered &= high-low>2
    h=np.arctan2(pred[:,:,:,3],pred[:,:,:,2])
    radius=(np.abs(np.sin(h))*pred[:,:,:,6]+np.abs(np.cos(h))*pred[:,:,:,7])/2
    invalid=(y-radius<low)|(y+radius>high)
    return {'road_boundary_violation':float(invalid[covered].mean()) if covered.any() else None,
            'road_boundary_coverage':float(covered.sum()/max(1,mask.sum()*len(pred))),
            'road_boundary_samples':int(covered.sum())}


def baseline(obs, horizon, variant):
    n=len(obs[0].track_ids);last=np.zeros((n,8),np.float32);seen=np.zeros(n,bool);age=np.zeros(n)
    for item in obs:
        age+=.2;last[item.visible]=item.values[item.visible];age[item.visible]=0;seen|=item.visible
    if variant=='constant_velocity':last[:,:2]+=last[:,2:4]*age[:,None]
    rows=[]
    for t in range(horizon):
        nxt=last.copy()
        if variant=='constant_velocity':nxt[:,:2]+=last[:,2:4]*(t+1)*.2
        rows.append(nxt)
    return np.array(rows)[None]


def evaluate(cfg, scenes, out):
    torch.set_num_threads(cfg['threads']);out=Path(out);(out/'rollouts').mkdir(exist_ok=True)
    rows=[];skipped=[];saved=[]
    horizon=30
    variants=['persistence','constant_velocity']+cfg['variants']
    for variant in variants:
        model=None;ckhash=None
        if variant not in ('persistence','constant_velocity'):
            path=out/'checkpoints'/f'{variant}-{cfg["training_seeds"][0]}.pt';model,_=load_model(path);ckhash=hashes(path)['SHA-256']
        regimes=['dense','sparse50','outage','graph_off','reset_memory'] if variant=='full' else ['dense']
        for regime in regimes:
            for scene in scenes:
                sensing=regime if regime in ('sparse50','outage') else 'dense'
                try:obs,cohort=observations(scene,6,sensing,7301)
                except ValueError as e:
                    skipped.append({'variant':variant,'regime':regime,'scene':scene.scene_id,'reason':str(e)});continue
                if model is None:pred=baseline(obs,horizon,variant)
                else:
                    model.variant='graph_off' if regime=='graph_off' else variant
                    with torch.no_grad():
                        state=observe_history(model,obs,scene.road,reset=regime=='reset_memory')
                        pred=model.imagine(state,horizon,cfg['samples'],rng_seed=9137)['agents'].cpu().numpy()
                if not np.isfinite(pred).all():raise AcquisitionError('Nonfinite autonomous prediction')
                truth=scene.values[6:36,cohort];valid=scene.existence[6:36,cohort]
                for seconds in (1,2,4,6):
                    steps=round(seconds/.2);metrics=metric(pred[:,:steps],truth[:steps],valid[:steps])
                    if metrics is None:
                        skipped.append({'variant':variant,'regime':regime,'scene':scene.scene_id,'horizon':seconds,'reason':'No valid future target for historical cohort'});continue
                    metrics.update(boundary_metrics(pred[:,:steps],valid[:steps],scene.provenance))
                    metrics['truth_road_boundary_violation']=boundary_metrics(truth[None,:steps],valid[:steps],scene.provenance)['road_boundary_violation']
                    rows.append({'variant':variant,'regime':regime,'scene':scene.scene_id,'group':scene.provenance['overlap_group'],
                                 'seed':cfg['training_seeds'][0],'horizon_seconds':seconds,'observed_history_fraction':float(np.stack([o.visible for o in obs]).mean()),**metrics})
                if variant=='full':
                    path=out/'rollouts'/f'{scene.scene_id}-{regime}.npz'
                    np.savez_compressed(path,prediction=pred,truth=truth,valid=valid,observed=np.stack([o.values for o in obs]),
                        observed_mask=np.stack([o.visible for o in obs]),time=scene.time[6:36],history_time=scene.time[:6],
                        track_ids=np.array(obs[0].track_ids),source_heading=np.array(scene.provenance['source_heading_rad'])[6:36,cohort])
                    meta={'scene':scene.scene_id,'regime':regime,'source_kind':'REAL_I24_MSD','doi':scene.provenance['doi'],
                          'source_hash':scene.provenance['source_hash'],'checkpoint_sha256':ckhash,'array_sha256':hashes(path)['SHA-256'],
                          'path':str(path),'cutoff_seconds':float(scene.time[5]),'forecast_seconds':6,'source_duration_seconds':float(scene.time[-1]),
                          'source_map':scene.provenance['source_map'],'x_origin_m':scene.provenance['x_origin_m'],'travel_sign':scene.provenance['travel_sign'],
                          'units':'metres, seconds, metres/second','generation_seed':9137,'map_coverage':'finite provided polylines only; no macro census'}
                    atomic_json(path.with_suffix('.json'),meta);saved.append(meta)
    frame=pd.DataFrame(rows);frame.to_csv(out/'evaluation.csv',index=False)
    atomic_json(out/'skipped_evaluations.json',skipped)
    summary=frame.groupby(['variant','regime','horizon_seconds']).mean(numeric_only=True).reset_index()
    summary.to_csv(out/'baseline_table.csv',index=False)
    primary=frame[(frame.horizon_seconds==4)&(frame.regime=='dense')].pivot(index='scene',columns='variant',values='joint_energy_m')
    if 'full' in primary and 'independent' in primary:
        primary=primary.dropna(subset=['full','independent'])
        gaps=primary.independent-primary.full
    else:
        gaps=pd.Series(dtype=float)
    rng=np.random.default_rng(7103)
    groupmap={s.scene_id:s.provenance['overlap_group'] for s in scenes}
    grouped={g:[float(gaps[s]) for s in gaps.index if groupmap[s]==g] for g in set(groupmap[s] for s in gaps.index)}
    groupvalues=list(grouped.values());draws=[]
    if len(groupvalues)>1:
        for _ in range(2000):draws.append(np.mean([v for idx in rng.integers(len(groupvalues),size=len(groupvalues)) for v in groupvalues[idx]]))
    primary_result={'paired_energy_reduction_m':float(gaps.mean()) if len(gaps) else None,
        'ci95_provisional':np.quantile(draws,[.025,.975]).tolist() if draws else None,
        'eligible_test_scenarios':len(gaps),'track_disjoint_groups':len(grouped),'independent_recording_groups':1,
        'confirmatory_support':'UNTESTED: fewer than five independently identified recording groups',
        'interval_unit':'supplied-track connected components; not recording-day uncertainty'}
    atomic_json(out/'primary_result.json',primary_result)
    from .showcase import make_demo
    demo=make_demo(out,saved)
    return {'primary':primary_result,'summary':summary.replace({np.nan:None}).to_dict('records'),'saved_rollouts':len(saved),'demo':demo}


def report(cfg,out,trained,results,qc,runtime):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out);fig=out/'figures';fig.mkdir(exist_ok=True)
    frame=pd.read_csv(out/'evaluation.csv');train=pd.read_csv(out/'training.csv')
    quality=pd.read_csv(out/'data_qc.csv')
    figure,axes=plt.subplots(1,2,figsize=(9,3.5))
    axes[0].hist(quality.agents,bins=np.arange(.5,quality.agents.max()+1.5));axes[0].set_xlabel('Source agents per eligible scene');axes[0].set_ylabel('Scenes')
    axes[1].hist(quality.missing_fraction,bins=15);axes[1].set_xlabel('Source invalid/missing-state fraction')
    figure.suptitle('REAL I24-MSD — three inspected TFRecord shards');figure.tight_layout();figure.savefig(fig/'source_qc.png',dpi=150);plt.close(figure)
    for col,title in [('validation_loss','Autonomous validation loss'),('train_loss','Stochastic training loss')]:
        plt.figure(figsize=(7,4))
        for name,group in train.groupby('variant'):plt.plot(group.step,group[col],label=name)
        plt.yscale('symlog',linthresh=.001);plt.xlabel('Optimizer updates');plt.ylabel(col);plt.title('REAL I24-MSD — '+title);plt.legend();plt.tight_layout();plt.savefig(fig/(col+'.png'),dpi=150);plt.close()
    for col,title in [('fde_m','FDE (m)'),('joint_energy_m','Joint energy score (m)'),('coverage90','Marginal 90% coverage'),('overlap_fraction','Oriented overlap fraction')]:
        plt.figure(figsize=(7,4))
        for name,g in frame[frame.regime=='dense'].groupby('variant'):
            values=g.groupby('horizon_seconds')[col].mean();plt.plot(values.index,values.values,marker='o',label=name)
        if col=='coverage90':plt.axhline(.9,color='gray',linestyle='--')
        plt.xlabel('Future seconds after 1 s context');plt.ylabel(title);plt.title('REAL I24-MSD — held-out pilot');plt.legend(fontsize=8);plt.tight_layout();plt.savefig(fig/(col+'.png'),dpi=150);plt.close()
    ab=frame[(frame.variant=='full')&(frame.horizon_seconds==4)].groupby('regime').fde_m.mean()
    ab.plot.bar(figsize=(7,4));plt.ylabel('4 s FDE (m)');plt.title('REAL I24-MSD — same checkpoint ablations');plt.tight_layout();plt.savefig(fig/'ablations.png',dpi=150);plt.close()
    atomic_json(out/'metrics.json',{'source_kind':'REAL_I24_MSD','qc':qc,'training':trained,'results':results,'pipeline_seconds':runtime,
        'convergence':f"Fixed {cfg['training_steps']}-update run; inspect curves. No convergence or research-level generalization claim.",
        'macro_metrics':None,'long_horizon_waves':'UNTESTED','cavnue':'INTERFACE ONLY'})
    from .reporting import write_summary
    write_summary(out)
