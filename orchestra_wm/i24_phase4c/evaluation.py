"""Final-checkpoint evaluation with immutable, independently reusable rollouts.

Validation must finish and its decision be frozen before developmental test opens.
The selected checkpoint is always the preregistered final update, never test-selected.
"""
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from orchestra_wm.i24.data import load_scene, sha
from orchestra_wm.i24.model import observe_history
from orchestra_wm.i24_continuous.data import observations, exposure_fields
from orchestra_wm.i24_continuous.evaluation import energy_score
from orchestra_wm.i24_continuous.source import write_immutable
from orchestra_wm.i24_phase4b.metrics import marginal_crps, sample_diversity, boundary_rates
from orchestra_wm.i24_phase4b.model import AuditedWorldModel
from .joint import JointWorldModel

SCALES = np.array([30., 100., 3000.])


def score_arrays(a, road, variant):
    """Equal scene weighting; support never enters generation, only scoring."""
    agents, fields = a['agents'], a['fields']
    truth, valid = a['truth'], a['valid']
    target, support = a['target_fields'], a['field_support']
    result = []
    micro_applicable = variant not in ('macro_only', 'aggregate_persistence')
    cohort_fields = None
    if 'history' in a:
        cohort_fields, _ = exposure_fields(np.concatenate([a['history'], truth]),
                                           np.concatenate([a['history_valid'], valid]), road)
        cohort_fields = cohort_fields[25:125]
    for seconds in (5, 10, 20):
        step = round(seconds / .2)
        use = support[step - 1]
        x, y = fields[:, step - 1, use], target[step - 1, use]
        row = dict(horizon_seconds=seconds, samples=len(fields), valid_lane_bins=int(use.sum()),
                   valid_micro_samples=int(valid[:step].sum()),
                   valid_micro_endpoint=int(valid[step - 1].sum()),
                   micro_status='FIXED_OBSERVED_COHORT' if micro_applicable else 'NOT_APPLICABLE')
        if use.any():
            lo, hi = np.quantile(x, [.05, .95], axis=0)
            row.update(macro_energy=energy_score(x / SCALES, y / SCALES),
                       macro_diversity=sample_diversity(x / SCALES),
                       macro_coverage90=float(((y >= lo) & (y <= hi)).mean()),
                       macro_negative_fraction=float((x < 0).mean()),
                       flow_identity_mae_veh_h=float(abs(x[..., 2] - 3.6*x[..., 0]*x[..., 1]).mean()))
            if cohort_fields is not None:
                denominator = float(y[:,1].sum())
                row['future_cohort_density_fraction'] = float(cohort_fields[step-1,use,1].sum()/denominator) if denominator>0 else None
            if 'derived_fields' in a and micro_applicable:
                row['total_decoder_vs_generated_cohort_rmse_scaled'] = float(np.sqrt(np.mean(((x-a['derived_fields'][:,step-1,use])/SCALES)**2)))
            for k, name in enumerate(('speed', 'density', 'flow')):
                row[f'{name}_mae'] = float(abs(x.mean(0)-y).mean(0)[k])
                row[f'{name}_crps'] = float(marginal_crps(x, y).mean(0)[k])
                row[f'{name}_coverage90'] = float(((y >= lo) & (y <= hi)).mean(0)[k])
                row[f'{name}_width90'] = float((hi-lo).mean(0)[k])
        if micro_applicable:
            mean = agents.mean(0)
            mask, last = valid[:step], valid[step-1]
            error = np.linalg.norm(mean[:step, :, :2]-truth[:step, :, :2], axis=-1)
            row['ade_m'] = float(error[mask].mean()) if mask.any() else None
            row['fde_m'] = float(np.linalg.norm(mean[step-1, :, :2]-truth[step-1, :, :2], axis=-1)[last].mean()) if last.any() else None
            sample_errors = np.linalg.norm(agents[:, :step, :, :2]-truth[None, :step, :, :2], axis=-1)
            row['sample_mean_ade_m'] = float(sample_errors[:, mask].mean()) if mask.any() else None
            if last.any():
                endpoint = agents[:, step-1, last, :2]
                row['micro_energy_scaled'] = energy_score(endpoint/np.array([800.,20.]), truth[step-1,last,:2]/np.array([800.,20.]))
                lo, hi = np.quantile(endpoint, [.05,.95], axis=0)
                row['micro_xy_coverage90'] = float(((truth[step-1,last,:2] >= lo) & (truth[step-1,last,:2] <= hi)).mean())
                dimensions = agents[:,step-1,last,6:8]
                pairs = np.triu_indices(int(last.sum()),1)
                row['endpoint_pairs_per_sample'] = len(pairs[0])
                if len(pairs[0]):
                    overlap = (abs(endpoint[:,:,None]-endpoint[:,None,:]) < (dimensions[:,:,None]+dimensions[:,None,:])/2).all(-1)
                    row['endpoint_overlap_pair_fraction'] = float(overlap[:,pairs[0],pairs[1]].mean())
                    diagonal = np.arange(endpoint.shape[1]);overlap[:,diagonal,diagonal] = False
                    row['endpoint_vehicle_overlap_fraction'] = float(overlap.any(-1).mean())
                    observed = truth[step-1,last]
                    observed_overlap = (abs(observed[:,None,:2]-observed[None,:,:2]) < (observed[:,None,6:8]+observed[None,:,6:8])/2).all(-1)
                    observed_overlap[diagonal,diagonal] = False
                    row['observed_endpoint_vehicle_overlap_fraction'] = float(observed_overlap.any(-1).mean())
            row.update(boundary_rates(agents[:, :step], mask, road))
            broad = np.broadcast_to(mask, agents[:, :step, :, 0].shape)
            row['acceleration_over_8m_s2_rate'] = float((np.linalg.norm(agents[:, :step, :, 4:6],axis=-1)[broad] > 8).mean()) if broad.any() else None
        result.append(row)
    return result


def baseline(obs, scene, variant):
    initial = obs[-1].values.copy()
    for j in range(len(initial)):
        i = max(i for i, o in enumerate(obs) if o.visible[j])
        initial[j] = obs[i].values[j]
        initial[j, :2] += initial[j, 2:4]*(24-i)*.2
    agents = np.repeat(initial[None], 100, axis=0)
    agents[:, :, :2] += np.arange(1, 101)[:, None, None]*.2*initial[None, :, 2:4]
    agents[:, :, 4:6] = 0
    low = min(scene.road.lane_centers)-scene.road.lane_width/2
    high = max(scene.road.lane_centers)+scene.road.lane_width/2
    valid = ((agents[:,:,0] >= scene.road.s_min) & (agents[:,:,0] < scene.road.s_max)
             & (agents[:,:,1] >= low) & (agents[:,:,1] < high))
    fields, _ = exposure_fields(np.concatenate([np.stack([o.values for o in obs]), agents]),
                               np.concatenate([np.stack([o.visible for o in obs]), valid]), scene.road)
    fields = fields[25:125]
    if variant == 'aggregate_persistence':
        # Unknown/outage bins are not observed zeros: retain each bin's latest
        # supported historical measurement. Never-seen bins use a zero fallback.
        latest = np.zeros_like(obs[-1].fields)
        for observation in obs:
            latest[observation.field_support] = observation.fields[observation.field_support]
        fields = np.repeat(latest[None], 100, axis=0)
    return dict(agents=agents[None], fields=fields[None], predicted_valid=valid[None])


def evaluate(data_root, training_root, output, split='validation', samples=8):
    root, training, output = map(Path, (data_root, training_root, output))
    torch.set_num_threads(2)
    run = json.loads((training/'run_manifest.json').read_text())
    completed = json.loads((training/'completed.json').read_text())
    cfg = run['config']
    expected_fingerprint = hashlib.sha256(json.dumps(dict(config=cfg,sources=run['sources']),sort_keys=True).encode()).hexdigest()
    if expected_fingerprint != run['fingerprint']:
        raise ValueError('Training manifest configuration/source fingerprint is inconsistent')
    checkpoints = {r['variant']: r['sha256'] for r in completed}
    manifest = json.loads((root/'data_manifest.json').read_text())
    assert sha(root/'data_manifest.json') == cfg['data_manifest_sha256']
    for name, digest in run['sources'].items():
        if sha(Path(name)) != digest:
            raise ValueError('Training source changed; refuse semantic drift')
    for name, digest in manifest['artifacts'].items():
        if sha(root/name) != digest:
            raise ValueError('Prepared data integrity mismatch')
    source_hashes = {str(p): sha(p) for p in (Path(__file__), Path('orchestra_wm/i24_phase4b/metrics.py'), Path('orchestra_wm/i24_continuous/evaluation.py'))}
    protocol = dict(training_fingerprint=run['fingerprint'], checkpoints=checkpoints,
                    split=split, samples=samples, regimes=['dense','outage'], sources=source_hashes,
                    selection=f'Final {cfg["updates"]}-update checkpoint; no developmental-test selection',
                    primary=('10-second macro_only minus full joint energy; [30,100,3000]' if 'macro_only' in cfg['variants']
                             else 'Exploratory graph-off control; not a macro-information comparison'),
                    uncertainty='Single recording day; no independent-session confidence interval')
    output.mkdir(parents=True, exist_ok=True)
    if split != 'validation':
        decision = json.loads((output.parent/'validation_decision.json').read_text())
        if decision['checkpoints'] != checkpoints or decision['decision'] != 'FIXED_FINAL_CHECKPOINTS_FROZEN':
            raise ValueError('Freeze validation decisions before developmental test')
        if sha(output.parent/'validation'/'summary.json') != decision['validation_summary_sha256']:
            raise ValueError('Validation changed after decision')
    write_immutable(output/'protocol.json', protocol)
    if (output/'summary.json').exists():
        summary = json.loads((output/'summary.json').read_text())
        assert sha(output/'evaluation.csv') == summary['evaluation_sha256']
        for e in json.loads((output/'arrays_manifest.json').read_text()):
            assert sha(output/e['path']) == e['sha256']
        return summary
    models = {}
    for variant in cfg['variants']:
        path = training/'checkpoints'/f'{variant}-{cfg["seed"]}.pt'
        assert sha(path) == checkpoints[variant]
        saved = torch.load(path, map_location='cpu', weights_only=False)
        assert saved['step'] == cfg['updates'] and saved['fingerprint'] == run['fingerprint']
        model_name = cfg.get('model_class','AuditedWorldModel')
        if model_name not in ('AuditedWorldModel','JointWorldModel'):
            raise ValueError('Unsupported checkpoint model class')
        model_type = JointWorldModel if model_name == 'JointWorldModel' else AuditedWorldModel
        model = model_type(cfg['dim'], cfg['latent'], variant, cfg['dt'])
        model.load_state_dict(saved['model']); model.eval(); models[variant] = model
    directory = output/'rollouts'; directory.mkdir(exist_ok=True)
    rows, entries = [], []
    source_split = 'test' if split == 'development_test' else split
    items = [i for i in manifest['scenes'] if i['split'] == source_split]
    if not items:
        raise ValueError('Requested split has no scenes')
    for number, item in enumerate(items):
        scene = load_scene(root/item['path'])
        with np.load((root/item['path']).with_suffix('.fields.npz')) as f:
            target, support = f['fields'].copy(), f['support'].copy()
        for regime in protocol['regimes']:
            obs, cohort = observations(scene, 25, regime, 7301+number)
            for variant in ['aggregate_persistence','constant_velocity']+cfg['variants']:
                path = directory/f'{number}-{regime}-{variant}.npz'
                receipt = path.with_suffix('.json')
                identity = dict(scene=scene.scene_id, scene_number=number, variant=variant, regime=regime,
                                path=str(path.relative_to(output)), checkpoint_sha256=checkpoints.get(variant,'KINEMATIC_BASELINE'))
                if receipt.exists():
                    record = json.loads(receipt.read_text())
                    assert all(record[k] == v for k,v in identity.items()) and sha(path) == record['sha256']
                    with np.load(path) as f:
                        arrays = {k:f[k].copy() for k in f.files}
                else:
                    if path.exists():
                        raise ValueError('Unverified partial artifact preserved; inspect before resuming')
                    if variant in models:
                        with torch.no_grad():
                            state = observe_history(models[variant], obs, scene.road)
                            forecast = models[variant].imagine(state, 100, samples, 7103+number)
                        arrays = {k: forecast[v].numpy() for k,v in [('agents','agents'),('fields','fields'),('predicted_valid','valid'),('derived_fields','derived_fields')]}
                    else:
                        arrays = baseline(obs, scene, variant)
                    arrays.update(truth=scene.values[25:125,cohort], valid=scene.existence[25:125,cohort],
                                  target_fields=target[25:125], field_support=support[25:125], cohort=cohort,
                                  history=np.stack([o.values for o in obs]), history_valid=np.stack([o.visible for o in obs]))
                    assert all(np.isfinite(arrays[k]).all() for k in ('agents','fields','truth','target_fields'))
                    temporary = path.with_suffix('.tmp.npz')
                    np.savez_compressed(temporary, **arrays); temporary.replace(path)
                    write_immutable(receipt, dict(**identity, sha256=sha(path)))
                entries.append(dict(**identity,sha256=sha(path)))
                for metrics in score_arrays(arrays, scene.road, variant):
                    rows.append(dict(**identity, **metrics))
            print(f'{split}: scene {number+1}/{len(items)}, {regime} complete', flush=True)
    frame = pd.DataFrame(rows)
    table = frame.to_csv(index=False)
    if (output/'evaluation.csv').exists() and (output/'evaluation.csv').read_text() != table:
        raise ValueError('Existing scoring results preserved')
    (output/'evaluation.csv').write_text(table)
    numeric = frame.select_dtypes(include='number').columns.difference(['scene_number','horizon_seconds'])
    means = frame.groupby(['variant','regime','horizon_seconds'])[numeric].mean().reset_index()
    means.to_csv(output/'baseline_table.csv',index=False)
    dense = frame[(frame.regime=='dense') & (frame.horizon_seconds==10)].pivot(index='scene',columns='variant',values='macro_energy')
    paired = dense.macro_only-dense.full if 'macro_only' in dense else None
    report = dict(split=split, scenes=len(items), independent_recording_days=1,
                  primary_mean_difference=float(paired.mean()) if paired is not None else None,
                  paired_differences=paired.tolist() if paired is not None else None, ci95=None,
                  inference='SINGLE_DAY_DEVELOPMENTAL_DESCRIPTIVE_ONLY', checkpoints=checkpoints,
                  updates=cfg['updates'], seed=cfg['seed'], evaluation_sha256=sha(output/'evaluation.csv'),
                  micro_scope=cfg['scope'],
                  population='Retained reconstructed population, not complete roadway census; fields and cohort have different populations',
                  calibration='Empirical 5th/95th ensemble quantiles from eight samples; finite-ensemble intervals are coarse')
    if 'graph_off' in dense:
        report['exploratory_graph_off_minus_full_macro_energy'] = float((dense.graph_off-dense.full).mean())
    if 'independent' in dense:
        report['exploratory_independent_minus_full_macro_energy'] = float((dense.independent-dense.full).mean())
    write_immutable(output/'arrays_manifest.json', entries)
    write_immutable(output/'summary.json', report)
    return report


def freeze_validation(root):
    root = Path(root)
    summary = json.loads((root/'validation'/'summary.json').read_text())
    write_immutable(root/'validation_decision.json', dict(
        decision='FIXED_FINAL_CHECKPOINTS_FROZEN', checkpoints=summary['checkpoints'],
        validation_summary_sha256=sha(root/'validation'/'summary.json'),
        choice='Use preregistered final update; no architecture, budget or checkpoint selected from developmental test',
        limitation='Finite validation forecasts do not establish calibration, convergence or physical validity'))
