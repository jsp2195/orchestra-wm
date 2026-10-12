import json
from pathlib import Path
import pandas as pd
import pytest
from orchestra_wm.i24_phase4e_colab.gates import promotion
from orchestra_wm.i24_phase4e_colab.storage import digest


def setup(tmp):
    runs=[]
    for seed in [101,202]:
        root=tmp/str(seed);(root/'checkpoints').mkdir(parents=True)
        (root/'checkpoints/best.pt').write_bytes(b'fixture checkpoint')
        (root/'completed.json').write_text(json.dumps(dict(seed=seed,updates_per_model=600,
            best_checkpoint_sha256=digest(root/'checkpoints/best.pt'),validation_stabilized_heuristic=True,
            capacity={v:dict(allocated=21100,observed_nonzero_gradient_scalars=21100) for v in ['full','aggregate']})))
        runs.append(str(root))
    cache=tmp/'cache';cache.mkdir();(cache/'manifest.json').write_text(json.dumps(dict(entries=[dict(scene='s1',split='validation')])))
    val=tmp/'validation';val.mkdir()
    variants=[('aggregate_persistence',0),('physical_persistence',0),('diffusion_persistence',101)]
    variants += [(v,s) for v in ['residual_full','residual_aggregate'] for s in [101,202]]
    rows=[dict(date='2022-11-25',scene='s1',variant=v,seed=s,regime=r,horizon_seconds=h,macro_energy=.5 if v.startswith('residual') else 1.)
          for v,s in variants for r in ['dense','sparse50','outage'] for h in [5,10,20]]
    pd.DataFrame(rows).to_csv(val/'evaluation.csv',index=False)
    (val/'protocol.json').write_text(json.dumps(dict(split='validation',samples=8,cache_manifest_sha256=digest(cache/'manifest.json'))))
    seal(val)
    return val,runs,cache


def seal(val):
    (val/'summary.json').write_text(json.dumps(dict(independent_test_evaluated=False,evaluation_sha256=digest(val/'evaluation.csv'))))


def test_gate_pass_and_no_automatic_unstable_promotion(tmp_path):
    val,runs,cache=setup(tmp_path)
    assert promotion(val,runs,cache)['allowed']
    done=Path(runs[0])/'completed.json';d=json.loads(done.read_text());d['validation_stabilized_heuristic']=False;done.write_text(json.dumps(d))
    assert promotion(val,runs,cache)['status']=='TRAINING_STABILITY_REVIEW_REQUIRED_TEST_UNOPENED'


def test_each_model_must_beat_persistence_and_all_rows_required(tmp_path):
    val,runs,cache=setup(tmp_path)
    frame=pd.read_csv(val/'evaluation.csv');frame.loc[(frame.variant=='residual_aggregate')&(frame.seed==202),'macro_energy']=1.1
    frame.to_csv(val/'evaluation.csv',index=False);seal(val)
    assert promotion(val,runs,cache)['status']=='VALIDATION_FAILED_TEST_UNOPENED'
    frame.iloc[:-1].to_csv(val/'evaluation.csv',index=False);seal(val)
    with pytest.raises(ValueError,match='all validation'):promotion(val,runs,cache)


def test_test_outcomes_and_changed_selected_checkpoint_rejected(tmp_path):
    val,runs,cache=setup(tmp_path)
    frame=pd.read_csv(val/'evaluation.csv');frame['date']='2022-11-28';frame.to_csv(val/'evaluation.csv',index=False);seal(val)
    with pytest.raises(ValueError,match='Nov25'):promotion(val,runs,cache)
    (Path(runs[1])/'checkpoints/best.pt').write_bytes(b'different')
    with pytest.raises(ValueError,match='checkpoint changed'):promotion(val,runs,cache)
