"""Validation-only readiness and promotion. Test outcomes cannot influence these."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .storage import digest


def promotion(validation, runs, cache='outputs/i24_phase4e/cache_v1'):
    validation = Path(validation)
    summary = json.loads((validation/'summary.json').read_text())
    protocol = json.loads((validation/'protocol.json').read_text())
    if summary['independent_test_evaluated'] or protocol['split'] != 'validation' or protocol['samples'] != 8:
        raise ValueError('Promotion requires frozen primary validation only')
    if digest(validation/'evaluation.csv') != summary['evaluation_sha256']:
        raise ValueError('Validation outcomes changed')
    cache_path=Path(cache)/'manifest.json'
    if digest(cache_path)!=protocol['cache_manifest_sha256']:
        raise ValueError('Validation cache changed')
    manifest=json.loads(cache_path.read_text())
    expected_scenes={e['scene'] for e in manifest['entries'] if e['split']=='validation'}
    seeds = set()
    stable = True
    for run in map(Path,runs):
        done = json.loads((run/'completed.json').read_text())
        seeds.add(done['seed'])
        if digest(run/'checkpoints/best.pt') != done['best_checkpoint_sha256']:
            raise ValueError('Selected checkpoint changed')
        if not 500 <= done['updates_per_model'] <= 1600:
            raise ValueError('Frozen training update budget not completed')
        if any(done['capacity'][v][k] != 21100 for v in ['full','aggregate']
               for k in ['allocated','observed_nonzero_gradient_scalars']):
            raise ValueError('Matched active capacity requirement not met')
        stable &= done['validation_stabilized_heuristic'] is True
    if seeds != {101,202} or len(runs) != 2:
        raise ValueError('Both paired seeds required')
    frame = pd.read_csv(validation/'evaluation.csv')
    if set(frame.date) != {'2022-11-25'}:
        raise ValueError('Promotion must use Nov25 only')
    variants = [('aggregate_persistence',0),('physical_persistence',0),('diffusion_persistence',101)]
    variants += [(v,s) for v in ['residual_full','residual_aggregate'] for s in [101,202]]
    reference = None
    for v,s in variants:
        rows = frame[(frame.variant==v)&(frame.seed==s)]
        keys = list(zip(rows.scene,rows.regime,rows.horizon_seconds))
        if len(keys)!=len(set(keys)) or not keys or not np.isfinite(rows.macro_energy).all():
            raise ValueError('Incomplete, duplicate, or nonfinite validation scores')
        if set(rows.regime)!={'dense','sparse50','outage'} or set(rows.horizon_seconds)!={5,10,20}:
            raise ValueError('Frozen regimes or horizons missing')
        expected={(scene,regime,h) for scene in rows.scene.unique() for regime in ['dense','sparse50','outage'] for h in [5,10,20]}
        if set(rows.scene)!=expected_scenes or set(keys)!=expected or (reference is not None and set(keys)!=reference):
            raise ValueError('Models do not share all validation windows/horizons/regimes')
        reference=set(keys)
    scores=frame[frame.regime=='dense'].groupby(['variant','seed']).macro_energy.mean()
    baseline=float(scores.loc[('aggregate_persistence',0)])
    learned={f'{v}_{s}':float(scores.loc[(v,s)]) for v,s in variants if v.startswith('residual_')}
    passed=all(x<baseline for x in learned.values())
    status='VALIDATION_PASSED_READY_TO_FREEZE' if passed and stable else (
        'VALIDATION_FAILED_TEST_UNOPENED' if not passed else 'TRAINING_STABILITY_REVIEW_REQUIRED_TEST_UNOPENED')
    return dict(status=status,allowed=bool(passed and stable),promotion_passed=bool(passed),
                stability_heuristic_passed=bool(stable),stability_is_convergence_proof=False,
                aggregate_persistence=baseline,learned=learned,
                diffusion_persistence=float(scores.loc[('diffusion_persistence',101)]),
                test_outcomes_opened=False)
