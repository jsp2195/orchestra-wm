"""Run the frozen experiment through mounted files only; survives Colab restarts."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys
import uuid
from unittest.mock import patch

from orchestra_wm.i24_continuous.source import write_immutable
from orchestra_wm.i24_phase4e.cache import prepare as prepare_cache
from . import data, mounted, training_checkpoint
from .gates import promotion
from .recovery import recover_date
from .storage import atomic_json, digest, snapshot, verify_snapshot
from .training import train

BASE=Path('outputs/i24_phase4e')
DATA=BASE/'data_v4'
CACHE=BASE/'cache_v1'
RUNS=[str(BASE/f'residual_seed{s}') for s in [101,202]]


def workspace(code, private):
    root=private/'workspace';root.mkdir(parents=True,exist_ok=True)
    for name in ['orchestra_wm','scripts','configs','docs']:
        target=root/name
        if target.is_symlink():
            if target.resolve()!=code/name:
                target.unlink()  # Code link only; never experiment bytes.
                target.symlink_to(code/name,target_is_directory=True)
        elif target.exists():
            raise ValueError('Unexpected workspace code directory; preserve and inspect it')
        else:target.symlink_to(code/name,target_is_directory=True)
    acquisition=Path('outputs/i24_continuous/pilot_0700_wb/acquisition.json')
    target=root/acquisition;target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if digest(target)!=digest(code/acquisition):raise ValueError('Historical acquisition differs')
    else:target.write_bytes((code/acquisition).read_bytes())
    os.chdir(root)
    Path('work/phase4e').mkdir(parents=True,exist_ok=True)
    # All paths stored in manifests remain relative across different Colab runtimes.
    lock={str(p.relative_to(code)):digest(p) for p in sorted((code/'orchestra_wm/i24_phase4e_colab').glob('*.py'))}
    for p in [*sorted((code/'orchestra_wm/i24_phase4e').glob('*.py')),
              *sorted((code/'scripts').glob('*i24_phase4e*.py')),
              code/'docs/I24_PHASE4E_PROTOCOL.md',code/'configs/i24_phase4e_colab_sources.json']:
        lock[str(p.relative_to(code))]=digest(p)
    write_immutable('work/phase4e/colab_code_lock.json',lock)
    return root


def execute(script, args=()):
    old=sys.argv[:]
    try:
        sys.argv=[script,*map(str,args)]
        # Diffusion trainer imports this function; preserve every optimizer update
        # using the same mounted write/readback logic as the paired trainer.
        with patch('orchestra_wm.i24.training.atomic_checkpoint',training_checkpoint.atomic_checkpoint):
            try:runpy.run_path(script,run_name='__main__')
            except SystemExit as error:
                if error.code not in (None,0):raise
    finally:sys.argv=old


def stage(script,args,complete,roots,label,backups):
    complete=Path(complete)
    marker=Path('work/phase4e/stages')/(label+'.json')
    if marker.exists():
        old=json.loads(marker.read_text())
        verify_snapshot(old['backup'])
        if not complete.exists() or digest(complete)!=old['completion_sha256']:raise ValueError('Stage completion missing or changed')
        print('Verified completed stage:',label,flush=True)
        return Path(old['backup'])
    if not complete.exists():execute(script,args)
    if not complete.exists():raise ValueError('Stage did not complete: '+label)
    receipt=snapshot(roots,backups,label)
    atomic_json(marker,dict(backup=str(receipt),completion_sha256=digest(complete)))
    return receipt


def prepare_date(date,entries,sources,backups):
    root=DATA/date
    marker=Path('work/phase4e/stages')/(date+'.json')
    if marker.exists():
        verify_snapshot(json.loads(marker.read_text())['backup'])
        print('Verified completed date:',date,flush=True)
        return
    if date in ['2022-11-21','2022-11-22']:
        raise ValueError('Completed dates must be restored, never reingested')
    name=date[5:7]+'-'+date[8:]+'-'+date[:4]+'.zip'
    metadata=sources[name]
    path=mounted.source_path(entries,metadata)
    data.prepare(date,metadata,root,path)
    mounted.verify_source(root)
    qc=BASE/('source_qc_colab_'+date)
    receipt=stage('scripts/qc_i24_phase4e.py',['--data',DATA,'--dates',date,'--output',qc],
          qc/'audit.json',[root,qc],'source_'+date,backups)
    atomic_json(marker,dict(backup=str(receipt)))


def prepare_cache_restart(root):
    root=Path(root)
    if root.exists() and not (root/'manifest.json').exists():
        preserved=Path('work/phase4e')/('interrupted-cache-'+uuid.uuid4().hex)
        root.rename(preserved)
        print('Preserved interrupted feature cache; rebuilding deterministic features.',flush=True)


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--drive-root',required=True)
    p.add_argument('--private-root',required=True)
    p.add_argument('--source-root')
    p.add_argument('--stage',choices=['prepare','train-validation','independent-test','all'],default='all')
    a=p.parse_args(argv)
    code=Path(__file__).resolve().parents[2]
    private=Path(a.private_root).resolve();private.mkdir(parents=True,exist_ok=True)
    workspace(code,private)
    backups=private/'backups';training_checkpoint.BACKUP_ROOT=backups
    atomic_json('work/phase4e/colab_execution.json',dict(stage=a.stage,status='RUNNING',
        source_transport='COLAB_MOUNTED_DRIVE',training='NOT_YET_COMPLETED',independent_testing='UNOPENED'))
    roots=[Path(a.drive_root)]
    if a.source_root and not Path(a.source_root).is_relative_to(roots[0]):roots.append(Path(a.source_root))
    print('Finding original archives and completed-date backups in mounted Drive...',flush=True)
    entries=mounted.index_drive(roots,exclude=[private])
    source_entries=mounted.index_drive([Path(a.source_root)],exclude=[private]) if a.source_root else entries
    sources=json.loads((code/'configs/i24_phase4e_colab_sources.json').read_text())
    if a.stage!='independent-test':
        # Fail early if the next two required originals are not visible. Never
        # request OAuth secrets or download Nov21/22 originals to compensate.
        for date in ['2022-11-23','2022-11-25']:
            if not (DATA/date/'data_manifest.json').exists():
                mounted.source_path(source_entries,sources[date[5:7]+'-'+date[8:]+'-'+date[:4]+'.zip'])
        old=json.loads((code/BASE/'continuation_backup_status.json').read_text())
        nov21=next(x for x in old['new_backups'] if x['scope']=='COMPLETED_NOV21')
        nov22=json.loads((code/BASE/'nov22_backup_status.json').read_text())
        for date,spec in [('2022-11-21',nov21),('2022-11-22',nov22)]:
            recover_date(date,entries,spec,'work/phase4e/recovery')
            print('Reused completed date:',date,flush=True)
        for date in ['2022-11-23','2022-11-25']:prepare_date(date,source_entries,sources,backups)
        qc=BASE/'source_qc_colab_train_validation'
        stage('scripts/qc_i24_phase4e.py',['--data',DATA,'--dates',*list(data.ROLES)[:4],'--output',qc],
              qc/'audit.json',[qc],'train_validation_qc',backups)
        if not (CACHE/'manifest.json').exists():
            prepare_cache_restart(CACHE)
            prepare_cache(DATA,CACHE)
        receipt_path=Path('work/phase4e/data_backup_private.json')
        if receipt_path.exists():
            from .storage import verify_training_backup
            verify_training_backup(receipt_path,CACHE)
        else:
            receipt=snapshot([DATA/d for d in list(data.ROLES)[:4]]+[CACHE,qc],backups,'train_validation_data',CACHE)
            atomic_json(receipt_path,json.loads(receipt.read_text()))
        if a.stage=='prepare':
            atomic_json('work/phase4e/colab_execution.json',dict(status='DATA_PREPARATION_COMPLETE',training='NOT_RUN',independent_testing='UNOPENED'))
            print('Four dates and cache verified and backed up. Ready for training.',flush=True)
            return
        stage('scripts/pilot_i24_phase4e.py',[],BASE/'pilot.json',[BASE/'pilot.json'],'pilot',backups)
        pilot=json.loads((BASE/'pilot.json').read_text())
        if pilot['status']!='FINITE_AUTHENTIC_NUMERICAL_PILOT' or pilot['cache_sha256']!=digest(CACHE/'manifest.json'):
            raise ValueError('Authentic numerical pilot required')
        for seed,run in zip([101,202],RUNS):
            while True:
                try:train(CACHE,run,seed,1600,3300);break
                except TimeoutError:
                    print('Time slice saved; resuming same paired model/configuration.',flush=True)
        stage('scripts/fit_i24_phase4e_diffusion.py',[],BASE/'diffusion_baseline/completed.json',
              [BASE/'diffusion_baseline'],'diffusion',backups)
        stage('scripts/evaluate_i24_phase4e.py',['--runs',*RUNS,'--output',BASE/'validation'],
              BASE/'validation/summary.json',[BASE/'validation'],'validation',backups)
        for seed,run in zip([101,202],RUNS):
            output=BASE/f'calibration64_seed{seed}'
            stage('scripts/calibrate_i24_phase4e.py',['--run',run,'--output',output],
                  output/'summary.json',[output],f'calibration64_seed{seed}',backups)
    from .storage import verify_training_backup
    verify_training_backup('work/phase4e/data_backup_private.json',CACHE)
    decision=promotion(BASE/'validation',RUNS)
    write_immutable(BASE/'colab_readiness.json',decision)
    print(json.dumps(decision,indent=2),flush=True)
    if not decision['allowed'] or a.stage=='train-validation':
        atomic_json('work/phase4e/colab_execution.json',dict(status=decision['status'],training='COMPLETED',
                    independent_testing='UNOPENED',next_action='Review validation only; no automatic budget/architecture changes'))
        snapshot([BASE/'colab_readiness.json','work/phase4e/colab_execution.json'],backups,'readiness')
        return
    # No test archive is ingested, cached, or forecast before the validation decision.
    for date in ['2022-11-28','2022-11-29']:prepare_date(date,source_entries,sources,backups)
    qc=BASE/'source_qc_colab_all_dates'
    stage('scripts/qc_i24_phase4e.py',['--data',DATA,'--dates',*list(data.ROLES),'--output',qc],
          qc/'audit.json',[qc],'all_dates_source_qc',backups)
    frozen=BASE/'independent_test_protocol.json'
    if not frozen.exists():
        prepare_cache_restart(BASE/'test_cache_v1')
        execute('scripts/freeze_i24_phase4e.py',['--runs',*RUNS,'--data',DATA])
    spec=json.loads(frozen.read_text())
    if spec['status']!='VALIDATION_PASSED_TEST_PROTOCOL_FROZEN':raise ValueError('Test protocol not frozen')
    # Bind the mounted adapter/code lock and readiness decision before any test forecast.
    lock=BASE/'colab_test_lock.json'
    write_immutable(lock,dict(protocol_sha256=digest(frozen),code_lock_sha256=digest('work/phase4e/colab_code_lock.json'),
                              readiness_sha256=digest(BASE/'colab_readiness.json')))
    for path,h in spec['artifacts'].items():
        if digest(path)!=h:raise ValueError('Frozen artifact changed')
    test_cache=BASE/'test_cache_v1'
    test_receipt=snapshot([DATA/d for d in ['2022-11-28','2022-11-29']]+[test_cache,frozen,lock],
                          backups,'frozen_test_data',test_cache)
    verify_snapshot(test_receipt,[frozen,lock],test_cache)
    stage('scripts/evaluate_i24_phase4e.py',['--runs',*RUNS,'--cache',test_cache,'--split','test',
          '--freeze',frozen,'--output',BASE/'independent_test'],BASE/'independent_test/summary.json',
          [BASE/'independent_test'],'independent_test',backups)
    # Equal-day effects; never treat windows or paired seeds as independent dates.
    import pandas as pd
    frame=pd.read_csv(BASE/'independent_test/evaluation.csv')
    primary=frame[(frame.regime=='dense')&(frame.horizon_seconds==10)&frame.variant.isin(['residual_full','residual_aggregate'])]
    paired=primary.pivot(index=['date','seed','scene'],columns='variant',values='macro_energy')
    effects=(paired.residual_aggregate-paired.residual_full).groupby(['date','seed']).mean()
    effects.to_csv(BASE/'independent_test/paired_day_seed_effects.csv',header=['aggregate_minus_full'])
    days=effects.groupby('date').mean()
    atomic_json(BASE/'independent_test/equal_day_summary.json',dict(equal_day_effect=float(days.mean()),
                per_day={str(k):float(v) for k,v in days.items()},day_effect_range=[float(days.min()),float(days.max())],
                independent_days=len(days),confirmation_gate_met=False,
                limitation='Two independent dates, below five-group gate. No confirmed microscopic interaction benefit.'))
    atomic_json('work/phase4e/colab_execution.json',dict(status='FROZEN_EXPERIMENT_COMPLETED',training='COMPLETED',
                independent_testing='COMPLETED',confirmed_micro_advantage=False))
    snapshot([BASE/'independent_test','work/phase4e/colab_execution.json'],backups,'final_results')
    print('Completed frozen evaluation. Results and verified backups remain in your private Drive folder.',flush=True)


def run(argv=None):
    try:
        main(argv)
    except BaseException as error:
        if isinstance(error,SystemExit) and error.code in (None,0):raise
        # Best effort only if Drive is still mounted; never obscure the original error.
        if Path('work/phase4e/colab_code_lock.json').exists():
            try:
                atomic_json('work/phase4e/colab_execution.json',dict(status='INTERRUPTED_OR_BLOCKED',
                    error_type=type(error).__name__,resume='Rerun the same pinned notebook and private folder',
                    training='COMPLETED' if all((Path(r)/'completed.json').exists() for r in RUNS) else 'INCOMPLETE',
                    independent_testing='COMPLETED' if (BASE/'independent_test/summary.json').exists() else 'UNOPENED_OR_INCOMPLETE'))
            except Exception:
                pass
        raise
