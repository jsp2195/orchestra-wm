"""Matched authentic-data autonomous forecasts with fixed physical scales."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from orchestra_wm.i24.data import load_scene,sha
from orchestra_wm.i24.model import observe_history
from .data import observations,exposure_fields
from .model import ContinuousWorldModel
from .source import write_immutable


def energy_score(samples,target):
    samples=np.asarray(samples,dtype=float);target=np.asarray(target,dtype=float)
    if not target.size:return None
    samples=samples.reshape(len(samples),-1);target=target.reshape(-1)
    norm=np.sqrt(target.size)
    first=np.linalg.norm(samples-target,axis=1).mean()/norm
    if len(samples)<2:return float(first)
    pairs=np.linalg.norm(samples[:,None]-samples[None],axis=-1)
    return float(first-.5*pairs.sum()/(len(samples)*(len(samples)-1))/norm)


def evaluate(data_output,output,samples=8):
    data_output,output=Path(data_output),Path(output);torch.set_num_threads(2)
    manifest=json.loads((data_output/'data_manifest.json').read_text())
    run=json.loads((output/'run_manifest.json').read_text());cfg=run['config']
    evaluation_source_sha256=sha(Path(__file__))
    if not (output/'training_metadata.json').exists():raise ValueError('All fixed-budget models must complete before comparisons')
    for name,digest in manifest['artifacts'].items():
        if sha(data_output/name)!=digest:raise ValueError('Evaluation source integrity mismatch')
    directory=output/'rollouts';directory.mkdir(exist_ok=True);rows=[];saved_entries=[]
    checkpoint_hashes={}
    for variant in cfg['variants']:
        checkpoint=output/'checkpoints'/f'{variant}-{cfg["seed"]}.pt'
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if payload['fingerprint']!=run['fingerprint'] or payload['step']!=cfg['updates']:raise ValueError('Final checkpoint/source mismatch')
        checkpoint_hashes[variant]=sha(checkpoint)
    if (output/'evaluation_manifest.json').exists() and (output/'scientific_summary.json').exists():
        cached=json.loads((output/'evaluation_manifest.json').read_text())
        report=json.loads((output/'scientific_summary.json').read_text())
        if cached.get('evaluation_source_sha256')!=evaluation_source_sha256 or cached['samples']!=samples or cached['fingerprint']!=run['fingerprint'] or report['checkpoints']!=checkpoint_hashes:
            raise ValueError('Existing evaluation semantics/checkpoints differ; preserve results and use a separate output namespace')
        for entry in cached['entries']:
            if sha(output/entry['path'])!=entry['sha256']:raise ValueError('Cached evaluated forecast integrity mismatch')
        if sha(output/'evaluation.csv')!=report['evaluation_sha256']:raise ValueError('Cached evaluation table integrity mismatch')
        print('Verified and reused immutable final-checkpoint evaluation',flush=True)
        return report
    for scene_number,item in enumerate(s for s in manifest['scenes'] if s['split']=='test'):
        path=data_output/item['path'];scene=load_scene(path)
        with np.load(path.with_suffix('.fields.npz')) as array:target_fields=array['fields'].copy();field_support=array['support'].copy()
        for regime in ('dense','sparse50','sparse20','blind','outage'):
            obs,cohort=observations(scene,25,regime,7301+scene_number)
            truth=scene.values[25:125,cohort];valid=scene.existence[25:125,cohort]
            for variant in ['persistence','constant_velocity']+cfg['variants']:
                if variant in ('persistence','constant_velocity'):
                    initial=obs[-1].values.copy()
                    # Last available measured state for outage cases, causally advanced.
                    for agent in range(len(cohort)):
                        index=max(i for i,o in enumerate(obs) if o.visible[agent])
                        initial[agent]=obs[index].values[agent]
                        initial[agent,:2]+=initial[agent,2:4]*(24-index)*.2
                    prediction=np.repeat(initial[None],100,axis=0)
                    if variant=='constant_velocity':
                        prediction[:,:,:2]+=np.arange(1,101)[:,None,None]*.2*initial[None,:,2:4]
                        prediction[:,:,4:6]=0
                    else:prediction[:,:,2:6]=0
                    inside=(prediction[:,:,0]>=scene.road.s_min)&(prediction[:,:,0]<scene.road.s_max)
                    historical=np.stack([o.values for o in obs]);visibility=np.stack([o.visible for o in obs])
                    fields,_=exposure_fields(np.concatenate([historical,prediction]),np.concatenate([visibility,inside]),scene.road)
                    agents=prediction[None];macro=fields[25:125][None]
                    predicted_valid=inside[None]
                    checkpoint_hash='KINEMATIC_BASELINE'
                else:
                    checkpoint=output/'checkpoints'/f'{variant}-{cfg["seed"]}.pt'
                    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
                    model=ContinuousWorldModel(cfg['model_dim'],cfg['latent_dim'],variant,cfg['dt'])
                    model.load_state_dict(payload['model']);model.eval()
                    with torch.no_grad():
                        state=observe_history(model,obs,scene.road)
                        forecast=model.imagine(state,100,samples,7103+scene_number)
                    agents=forecast['agents'].numpy()[:,:,:,:8];macro=forecast['fields'].numpy()
                    predicted_valid=forecast['valid'].numpy()
                    checkpoint_hash=checkpoint_hashes[variant]
                if not np.isfinite(agents).all() or not np.isfinite(macro).all():raise ValueError('Nonfinite autonomous forecasts')
                saved=directory/f'{scene_number}-{regime}-{variant}.npz'
                np.savez_compressed(saved,agents=agents,fields=macro,predicted_valid=predicted_valid,truth=truth,valid=valid,
                                    target_fields=target_fields[25:125],field_support=field_support[25:125],
                                    cohort=cohort,history=scene.values[:25,cohort],history_valid=scene.existence[:25,cohort])
                saved_entries.append(dict(scene=scene.scene_id,scene_number=scene_number,regime=regime,variant=variant,
                                          path=str(saved.relative_to(output)),sha256=sha(saved),checkpoint_sha256=checkpoint_hash))
                for seconds in (5,10,20):
                    step=round(seconds/.2);fs=field_support[24+step]
                    scaled=macro[:,step-1,fs]/np.array([30,100,3000])
                    target=target_fields[24+step,fs]/np.array([30,100,3000])
                    score=energy_score(scaled,target)
                    mask=valid[:step];count=int(mask.sum());last=valid[step-1]
                    mean=agents.mean(0);error=np.linalg.norm(mean[:step,:,:2]-truth[:step,:,:2],axis=-1)
                    ade=float(error[mask].mean()) if count and variant!='macro_only' else None
                    fde=float(np.linalg.norm(mean[step-1,:,:2]-truth[step-1,:,:2],axis=-1)[last].mean()) if last.any() and variant!='macro_only' else None
                    field_mae=abs(macro[:,step-1,fs].mean(0)-target_fields[24+step,fs]).mean(0).tolist() if fs.any() else [None]*3
                    lo,hi=np.quantile(macro[:,step-1,fs],[.05,.95],axis=0)
                    coverage=float(((target_fields[24+step,fs]>=lo)&(target_fields[24+step,fs]<=hi)).mean()) if fs.any() else None
                    cohort_mask=np.broadcast_to(mask[None],agents[:,:step,:,0].shape)
                    acceleration=np.linalg.norm(agents[:,:step,:,4:6],axis=-1)
                    lateral_min=min(scene.road.lane_centers)-scene.road.lane_width/2
                    lateral_max=max(scene.road.lane_centers)+scene.road.lane_width/2
                    departures=(agents[:,:step,:,0]<scene.road.s_min)|(agents[:,:step,:,0]>=scene.road.s_max)|(agents[:,:step,:,1]<lateral_min)|(agents[:,:step,:,1]>=lateral_max)
                    acceleration_rate=float((acceleration[cohort_mask]>8).mean()) if cohort_mask.any() and variant!='macro_only' else None
                    departure_rate=float(departures[cohort_mask].mean()) if cohort_mask.any() and variant!='macro_only' else None
                    endpoint=agents[:,step-1,last]
                    pair_count=len(endpoint[0])*(len(endpoint[0])-1)//2
                    overlap=None
                    if pair_count and variant!='macro_only':
                        longitudinal=abs(endpoint[:,:,None,0]-endpoint[:,None,:,0])<(endpoint[:,:,None,6]+endpoint[:,None,:,6])/2
                        lateral=abs(endpoint[:,:,None,1]-endpoint[:,None,:,1])<(endpoint[:,:,None,7]+endpoint[:,None,:,7])/2
                        overlap=float((longitudinal&lateral)[:,np.triu_indices(len(endpoint[0]),1)[0],np.triu_indices(len(endpoint[0]),1)[1]].mean())
                    rows.append(dict(scene=scene.scene_id,regime=regime,variant=variant,horizon_seconds=seconds,
                                     macro_energy=score,macro_speed_mae_m_s=field_mae[0],macro_density_mae_veh_km=field_mae[1],macro_flow_mae_veh_h=field_mae[2],
                                     macro_coverage90=coverage,ade_m=ade,fde_m=fde,valid_micro_samples=count,
                                     valid_micro_endpoint=int(last.sum()),valid_lane_bins=int(fs.sum()),samples=len(agents),
                                     acceleration_over_8m_s2_rate=acceleration_rate,road_departure_rate=departure_rate,
                                     endpoint_overlap_pair_fraction=overlap,
                                     micro_status='NOT_APPLICABLE' if variant=='macro_only' else 'FIXED_PAST_COHORT',
                                     checkpoint_sha256=checkpoint_hash))
                print(f'Evaluated scene {scene_number+1}/3, {regime}, {variant}',flush=True)
    frame=pd.DataFrame(rows)
    csv_payload=frame.to_csv(index=False);csv_path=output/'evaluation.csv'
    if csv_path.exists() and csv_path.read_text()!=csv_payload:raise ValueError('Existing evaluation table differs; preserved')
    csv_path.write_text(csv_payload)
    numeric=['macro_energy','macro_speed_mae_m_s','macro_density_mae_veh_km','macro_flow_mae_veh_h','macro_coverage90','ade_m','fde_m']
    frame.groupby(['variant','regime','horizon_seconds'])[numeric].mean().reset_index().to_csv(output/'baseline_table.csv',index=False)
    dense=frame[(frame.regime=='dense')&(frame.horizon_seconds==10)].pivot(index='scene',columns='variant',values='macro_energy')
    paired=(dense.macro_only-dense.full).tolist()
    report=dict(source='REAL_I24',training_seed=cfg['seed'],updates_each=cfg['updates'],independent_recording_sessions=1,
                test_windows=3,primary='10s macro-only minus full energy score',paired_differences=paired,
                mean_difference=float(np.mean(paired)),ci95=None,
                inference='PRELIMINARY_DESCRIPTIVE_ONLY; one contiguous held-out block cannot support session-bootstrap confirmation',
                physical_scales=[30,100,3000],macro_scope=manifest['limitation'],
                sensing='EMULATED dense/sparse50/sparse20/blind/final-half-history outage',
                convergence='Not established by this fixed 100-update pilot',wave_onset='UNTESTED',cross_day_generalization='UNTESTED',
                checkpoints=checkpoint_hashes,evaluation_sha256=sha(output/'evaluation.csv'))
    write_immutable(output/'evaluation_manifest.json',dict(entries=saved_entries,samples=samples,fingerprint=run['fingerprint'],evaluation_source_sha256=evaluation_source_sha256))
    write_immutable(output/'scientific_summary.json',report)
    return report
