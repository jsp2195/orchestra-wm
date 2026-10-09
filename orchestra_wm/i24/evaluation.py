from pathlib import Path
import numpy as np
import pandas as pd
import torch
from orchestra_wm.i24.data import observations,fields_numpy,sha,write_json
from orchestra_wm.i24.model import observe_history
from orchestra_wm.i24.training import load_checkpoint


def energy_score(samples,truth):
    s=np.asarray(samples).reshape(len(samples),-1);y=np.asarray(truth).reshape(-1)
    if not y.size:return 0.
    return float(np.linalg.norm(s-y,axis=1).mean()/np.sqrt(y.size)-.5*np.linalg.norm(s[:,None]-s[None,:],axis=-1).mean()/np.sqrt(y.size))

def crps(samples,truth):
    return float(np.abs(samples-truth).mean()-.5*np.abs(samples[:,None]-samples[None,:]).mean())

def cluster_interval(values,clusters,seed=7103):
    unique=sorted(set(clusters));means=np.array([np.mean(np.array(values)[np.array(clusters)==c]) for c in unique])
    if len(means)<2:return {'mean':float(means.mean()),'ci_low':None,'ci_high':None,'clusters':len(means),'status':'INSUFFICIENT_CLUSTERS'}
    rng=np.random.default_rng(seed);draws=np.array([rng.choice(means,len(means),replace=True).mean() for _ in range(2000)])
    return {'mean':float(means.mean()),'ci_low':float(np.quantile(draws,.025)),'ci_high':float(np.quantile(draws,.975)),'clusters':len(means),'status':'PRELIMINARY' if len(means)<5 else 'ESTIMATED'}


def forecast_metrics(pred,truth,mask,fields_true,road,dt,h):
    agents=pred['agents'][:,:h];target=truth[:h];valid=mask[:h];mean=agents.mean(0)
    error=np.linalg.norm(mean[:,:,:2]-target[:,:,:2],axis=-1)
    final=valid[-1];full=valid
    f=pred['fields'][:,h-1];ft=fields_true[h-1];scale=np.array([30,100,3000]);scaled=f/scale
    lo,hi=np.quantile(agents[:,:,:,0],[.05,.95],axis=0);flo,fhi=np.quantile(f,[.05,.95],axis=0)
    # All violations are reported on predicted existence support, never clipped away.
    physical=pred['valid'][:,:h];longitudinal=agents[:,:,:,0];lateral=agents[:,:,:,1]
    count=0;overlap=0;ordering=0
    for k in range(len(agents)):
        for t in range(h):
            active=np.flatnonzero(physical[k,t])
            for x,i in enumerate(active):
                for j in active[:x]:
                    count+=1;near=abs(lateral[k,t,i]-lateral[k,t,j])<(agents[k,t,i,7]+agents[k,t,j,7])/2
                    overlap+=int(near and abs(longitudinal[k,t,i]-longitudinal[k,t,j])<(agents[k,t,i,6]+agents[k,t,j,6])/2)
                    if t and near:ordering+=int(np.sign(longitudinal[k,t,i]-longitudinal[k,t,j])!=np.sign(longitudinal[k,t-1,i]-longitudinal[k,t-1,j]))
    dmin=min(road.lane_centers)-road.lane_width/2;dmax=max(road.lane_centers)+road.lane_width/2
    outcome={'horizon_seconds':h*dt,'ade_m':float(error[full].mean()) if full.any() else None,'fde_m':float(error[-1,final].mean()) if final.any() else None,'speed_error_mps':float(np.abs(mean[-1,final,2]-target[-1,final,2]).mean()) if final.any() else None,
       'valid_agent_times':int(full.sum()),'valid_final_agents':int(final.sum()),'micro_energy_m':energy_score(agents[:,-1,final,:2],target[-1,final,:2]) if final.any() else None,
       'macro_energy':energy_score(scaled,ft/scale),'macro_crps':crps(scaled,ft/scale),'macro_speed_mae_mps':float(np.abs(f.mean(0)[:,0]-ft[:,0]).mean()),'density_mae_veh_km_lane':float(np.abs(f.mean(0)[:,1]-ft[:,1]).mean()),'flow_mae_veh_h_lane':float(np.abs(f.mean(0)[:,2]-ft[:,2]).mean()),
       'trajectory_coverage90':float(((target[:,:,0]>=lo)&(target[:,:,0]<=hi))[full].mean()) if full.any() else None,'trajectory_interval_width_m':float((hi-lo)[full].mean()) if full.any() else None,'macro_coverage90':float(((ft>=flo)&(ft<=fhi)).mean()),'macro_interval_width_scaled':float(((fhi-flo)/scale).mean()),
       'diversity_m':float(agents[:,-1,:,:2].std(0).mean()),'minade_m':float(min(np.linalg.norm(a[:,:,:2]-target[:,:,:2],axis=-1)[full].mean() for a in agents)) if full.any() else None,'overlap_pair_fraction':overlap/max(count,1),'pair_denominator':count,'ordering_changes':ordering,
       'unreasonable_acceleration_fraction':float((np.abs(agents[:,:,:,4:6])>8).any(-1)[physical].mean()) if physical.any() else 0.,'road_violation_fraction':float(((lateral<dmin)|(lateral>dmax))[physical].mean()) if physical.any() else 0.,'negative_macro_fraction':float((f<0).mean()),'micro_macro_consistency_scaled':float(np.abs((pred['fields'][:,:h]-pred['derived_fields'][:,:h])/scale).mean())}
    return outcome


def kinematic_forecast(obs,road,steps,dt,kind):
    n=len(obs[0].track_ids);last=np.zeros((n,8),np.float32);last[:,6:]=[4.8,1.9]
    for o in obs:
        last[:,:2]+=last[:,2:4]*dt;last[o.visible]=o.values[o.visible]
    result=[]
    for _ in range(steps):
        if kind=='constant_velocity':last[:,:2]+=last[:,2:4]*dt
        result.append(last.copy())
    a=np.array(result);mask=(a[:,:,0]>=road.s_min)&(a[:,:,0]<road.s_max);f,_=fields_numpy(a,mask,road)
    return {'agents':a[None],'fields':f[None],'derived_fields':f[None],'valid':mask[None]}


def cheap_baselines(cfg,scenes,out,training_scenes=None):
    from orchestra_wm.i24.baselines import calibrate_idm,idm_forecast
    parameters,idm_status=calibrate_idm(training_scenes or [])
    rows=[];context=round(cfg['context_seconds']/cfg['dt']);steps=round(max(cfg['horizons_seconds'])/cfg['dt'])
    for scene in scenes:
        obs,cohort=observations(scene,context);truth=scene.values[context:context+steps,cohort];mask=scene.existence[context:context+steps,cohort];fields,_=fields_numpy(scene.values[context:context+steps],scene.existence[context:context+steps],scene.road)
        for name in ['constant_position','constant_velocity']+(['idm'] if parameters is not None else []):
            pred=idm_forecast(obs,scene.road,steps,cfg['dt'],parameters) if name=='idm' else kinematic_forecast(obs,scene.road,steps,cfg['dt'],name)
            for seconds in cfg['horizons_seconds']:rows.append({'scene':scene.scene_id,'cluster':scene.provenance.get('cluster',scene.session_id),'variant':name,'seed':0,'regime':'dense',**forecast_metrics(pred,truth,mask,fields,scene.road,cfg['dt'],round(seconds/cfg['dt']))})
    pd.DataFrame(rows).to_csv(out/'baseline_metrics.csv',index=False)
    write_json(out/'idm_status.json',idm_status)
    return rows


def evaluate(cfg,scenes,out,baseline_rows):
    rows=list(baseline_rows);skipped=[];context=round(cfg['context_seconds']/cfg['dt']);steps=round(max(cfg['horizons_seconds'])/cfg['dt']);cache=out/'rollouts';cache.mkdir(exist_ok=True);drift=[];strata=[]
    for seed in cfg['training_seeds']:
        for variant in cfg['variants']:
            checkpoint=out/'checkpoints'/f'{variant}-{seed}.pt';model,payload=load_checkpoint(checkpoint)
            modes=['dense','sparse50','sparse20','blind','outage','reset_memory','graph_off','permutation'] if variant=='full' else ['dense']
            for regime in modes:
                for index,scene in enumerate(scenes):
                    sensing=regime if regime in ['dense','sparse50','sparse20','blind','outage'] else 'dense'
                    try:obs,cohort=observations(scene,context,sensing,7301+index)
                    except ValueError as exc:
                        skipped.append({'scene':scene.scene_id,'variant':variant,'seed':seed,'regime':regime,'reason':str(exc)});continue
                    original=model.variant
                    if regime=='graph_off':model.variant='graph_off'
                    with torch.no_grad():
                        state=observe_history(model,obs,scene.road,reset=regime=='reset_memory',permutation=regime=='permutation');prediction=model.imagine(state,steps,cfg['samples'],seed+index)
                    model.variant=original;pred={k:v.cpu().numpy() for k,v in prediction.items()}
                    truth=scene.values[context:context+steps,cohort];mask=scene.existence[context:context+steps,cohort];fields,_=fields_numpy(scene.values[context:context+steps],scene.existence[context:context+steps],scene.road)
                    file=cache/f'{variant}-{seed}-{index}-{regime}.npz';np.savez_compressed(file,**pred,truth=truth,target_mask=mask,truth_fields=fields,observed=np.stack([o.values for o in obs]),observed_mask=np.stack([o.visible for o in obs]))
                    write_json(file.with_suffix('.json'),{'scene':scene.scene_id,'session':scene.session_id,'source':scene.provenance,'checkpoint_sha256':sha(checkpoint),'rollout_sha256':sha(file),'split':'test','dt':cfg['dt'],'context_seconds':cfg['context_seconds'],'horizon_seconds':max(cfg['horizons_seconds']),'rng_seed':seed+index,'regime':regime,'track_ids':obs[0].track_ids,'road':{'s_min':scene.road.s_min,'s_max':scene.road.s_max,'lane_centers':scene.road.lane_centers,'lane_width':scene.road.lane_width,'bins':scene.road.bins}})
                    for seconds in cfg['horizons_seconds']:
                        metrics=forecast_metrics(pred,truth,mask,fields,scene.road,cfg['dt'],round(seconds/cfg['dt']))
                        if variant=='macro_only':
                            # Macro-only kinematic placeholder is not an agent prediction result.
                            for key in ['ade_m','fde_m','speed_error_mps','micro_energy_m','minade_m','trajectory_coverage90','trajectory_interval_width_m','diversity_m','overlap_pair_fraction','ordering_changes','unreasonable_acceleration_fraction','road_violation_fraction','micro_macro_consistency_scaled']:metrics[key]=None
                        rows.append({'scene':scene.scene_id,'cluster':scene.provenance.get('cluster',scene.session_id),'variant':variant,'seed':seed,'regime':regime,'observed_history_fraction':float(sum(o.visible.sum() for o in obs)/max(scene.detection[:context].sum(),1)),'cohort_tracks':len(cohort),**metrics})
                    if regime=='dense':
                        mean=pred['agents'].mean(0);e=np.linalg.norm(mean[:,:,:2]-truth[:,:,:2],axis=-1)
                        for t in range(steps):drift.append({'variant':variant,'seed':seed,'scene':scene.scene_id,'seconds':(t+1)*cfg['dt'],'error_m':float(e[t,mask[t]].mean()) if mask[t].any() else 0.,'valid':int(mask[t].sum())})
                        changes=np.ptp(scene.values[:,cohort,1],axis=0)>scene.road.lane_width*.5
                        for label,subset in [('lane_change',changes),('lane_keep',~changes)]:
                            v=mask&subset[None];strata.append({'variant':variant,'seed':seed,'scene':scene.scene_id,'stratum':label,'valid_agent_times':int(v.sum()),'ade_m':float(e[v].mean()) if v.any() else None})
                        for klass in np.unique(scene.vehicle_class[cohort]):
                            v=mask&(scene.vehicle_class[cohort]==klass)[None];strata.append({'variant':variant,'seed':seed,'scene':scene.scene_id,'stratum':f'class_{klass}','valid_agent_times':int(v.sum()),'ade_m':float(e[v].mean()) if v.any() else None})
            print('Evaluated',variant,seed,flush=True)
    write_json(out/'skipped_evaluations.json',skipped)
    frame=pd.DataFrame(rows);frame.to_csv(out/'evaluation.csv',index=False);pd.DataFrame(drift).to_csv(out/'rollout_drift.csv',index=False);pd.DataFrame(strata).to_csv(out/'stratified_metrics.csv',index=False)
    paired=frame[(frame.regime=='dense')&(frame.horizon_seconds==10)&frame.variant.isin(['macro_only','full'])].pivot(index=['seed','scene','cluster'],columns='variant',values='macro_energy').reset_index();paired['gain']=paired.macro_only-paired.full;paired.to_csv(out/'paired_information.csv',index=False)
    per=paired.groupby(['scene','cluster']).gain.mean().reset_index();interval=cluster_interval(per.gain,per.cluster)
    interval.update({'scientific_status':'BLOCKED' if cfg['source_kind']=='SYNTHETIC_FIXTURE' else 'PRELIMINARY','source_kind':cfg['source_kind'],'training_seeds':cfg['training_seeds'],'metric':'10-second normalized macro energy-score reduction','note':'Fixture clusters are synthetic generation seeds, not independent real days. Fewer than five real clusters cannot support the primary scientific claim.'})
    write_json(out/'confidence_intervals.json',interval)
    return frame


def history_matching(scenes,cfg,out):
    context=round(cfg['context_seconds']/cfg['dt']);features=[];micro=[]
    for scene in scenes:
        obs,_=observations(scene,context);features.append(np.mean([o.fields for o in obs],axis=0).reshape(-1)/np.tile([30,100,3000],scene.road.field_count))
        from orchestra_wm.i24.baselines import leader_pairs
        last=obs[-1].values;edges=leader_pairs(last,obs[-1].visible,scene.road.lane_width);micro.append([float(np.mean([gap for i,j,gap in edges])) if edges else 0.,float(np.std([last[i,2]-last[j,2] for i,j,gap in edges])) if edges else 0.])
    rows=[]
    for i in range(len(scenes)):
        for j in range(i):
            distance=float(np.linalg.norm(features[i]-features[j])/np.sqrt(len(features[i])))
            if distance<=.1:rows.append({'scene_a':scenes[i].scene_id,'scene_b':scenes[j].scene_id,'history_macro_distance':distance,'headway_difference_m':abs(micro[i][0]-micro[j][0]),'relative_speed_std_difference_mps':abs(micro[i][1]-micro[j][1])})
    pd.DataFrame(rows,columns=['scene_a','scene_b','history_macro_distance','headway_difference_m','relative_speed_std_difference_mps']).to_csv(out/'matched_macro_histories.csv',index=False)
    if rows and (out/'paired_information.csv').exists():
        scores=pd.read_csv(out/'paired_information.csv').groupby('scene').gain.mean()
        matched=pd.DataFrame(rows);matched['mean_information_gain']=.5*(matched.scene_a.map(scores)+matched.scene_b.map(scores));matched.to_csv(out/'matched_history_scores.csv',index=False)
    return len(rows)
