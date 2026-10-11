"""Fail-closed November25 promotion and independent-test specification."""
import argparse,json,datetime
from pathlib import Path
import pandas as pd
from orchestra_wm.i24.data import sha
from orchestra_wm.i24_continuous.source import write_immutable
from orchestra_wm.i24_phase4e.cache import prepare,load_cache
p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True);p.add_argument('--validation',default='outputs/i24_phase4e/validation');p.add_argument('--output',default='outputs/i24_phase4e/independent_test_protocol.json');a=p.parse_args()
root=Path(a.validation);summary=json.loads((root/'summary.json').read_text())
if sha(root/'evaluation.csv')!=summary['evaluation_sha256']:raise ValueError('Validation outcomes changed')
protocol=json.loads((root/'protocol.json').read_text())
if protocol['split']!='validation' or protocol['samples']!=8:raise ValueError('Not primary validation evaluation')
frame=pd.read_csv(root/'evaluation.csv')
if set(frame.date)!= {'2022-11-25'}:raise ValueError('Validation must use only November25')
dense=frame[frame.regime=='dense'];scores=dense.groupby(['variant','seed']).macro_energy.mean()
baseline=float(scores.loc[('aggregate_persistence',0)])
required=[(v,s) for v in ['residual_full','residual_aggregate'] for s in [101,202]]
gate={f'{v}_{s}':float(scores.loc[(v,s)]) for v,s in required}
passed=all(value<baseline for value in gate.values())
status=dict(status='VALIDATION_PASSED' if passed else 'VALIDATION_FAILED_TEST_UNOPENED',objective='Scene-equal dense mean5/10/20 macro energy',aggregate_persistence=baseline,learned=gate,stochastic_persistence=float(scores.loc[('diffusion_persistence',101)]))
write_immutable(Path(a.output).with_name('validation_gate.json'),status)
if not passed:raise SystemExit('Promotion failed; independent forecast outcomes remain unopened')
test_cache=Path('outputs/i24_phase4e/test_cache_v1')
prepare('outputs/i24_phase4e/data_v1',test_cache,splits=('test',))
load_cache(test_cache,allowed_splits=('test',))
artifacts={str(root/f):sha(root/f) for f in ['evaluation.csv','summary.json','protocol.json']}
for directory in a.runs:
    run=Path(directory)
    done=json.loads((run/'completed.json').read_text())
    if any(done['capacity'][v]['allocated']!=21100 or done['capacity'][v]['observed_nonzero_gradient_scalars']!=21100 for v in ['full','aggregate']):
        raise ValueError('Matched active-capacity gate failed')
    if sha(run/'checkpoints/best.pt')!=done['best_checkpoint_sha256']:raise ValueError('Selected checkpoint changed')
    for f in ['checkpoints/best.pt','completed.json','run_manifest.json']:artifacts[str(run/f)]=sha(run/f)
    for f,h in json.loads((run/'run_manifest.json').read_text())['sources'].items():artifacts[f]=h
for path in ['scripts/evaluate_i24_phase4e.py','orchestra_wm/i24_phase4d/baselines.py','orchestra_wm/i24_phase4b/metrics.py','orchestra_wm/i24_continuous/evaluation.py','outputs/i24_phase4e/diffusion_baseline/checkpoints/last.pt']:
    artifacts[path]=sha(path)
result=dict(status='VALIDATION_PASSED_TEST_PROTOCOL_FROZEN',version=1,frozen_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    samples=8,runs=a.runs,artifacts=artifacts,test_cache_sha256=sha(test_cache/'manifest.json'),horizons_seconds=[5,10,20],
    primary='Aggregate-only minus full10s energy; positive favors micro; paired seed/day/window',seeds=[101,202],
    forecast_noise_seed='7103 + scene number',regimes=['dense','sparse50','outage'],scales=[30,100,3000],
    test_dates=['2022-11-28','2022-11-29'],five_independent_group_confirmation_gate_satisfied=False,
    uncertainty='Report both day effects and their range; windows/paired seeds are not independent recording days',
    diagnostic_protocol_sha256=sha('docs/I24_PHASE4E_PROTOCOL.md'))
write_immutable(a.output,result);print(json.dumps(status,indent=2))
