import json
import hashlib
import pytest
from orchestra_wm.i24_continuous.access import inspect_local, AccessBlocked

BUDGET = {'min_free_bytes':0,'max_source_bytes':1000000,'max_samples':10000}


def source(tmp_path, payload=b'[]'):
    p = tmp_path/'unit.json'; p.write_bytes(payload)
    meta = dict(dataset='I-24 MOTION',release='I24MOTION_PUBLIC_v1.0',format='json-array',direction=-1,
                terms_accepted=True,corrected_timestamps=True,source_url='https://i24motion.org/access_data',
                x_origin_ft=1000,session='UNIT_FIXTURE_NOT_REAL',recording_date='2022-11-22',
                start_unix_s=0,end_unix_s=600,
                files=[dict(path='unit.json',bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())])
    (tmp_path/'acquisition.json').write_text(json.dumps(meta)); return meta


def test_missing(tmp_path):
    with pytest.raises(AccessBlocked,match='Missing'): inspect_local(tmp_path,BUDGET)


@pytest.mark.parametrize('payload',[b'<html>Login</html>',b'{"error":"unauthorized"}',b'[]'])
def test_not_real(tmp_path,payload):
    source(tmp_path,payload)
    with pytest.raises(AccessBlocked): inspect_local(tmp_path,BUDGET)


@pytest.mark.parametrize('change',[
    {'source_url':'https://i24motion.org.evil.example/data'},
    {'source_url':'https://i24motion.org/data?token=DO_NOT_LOG'},
    {'password':'DO_NOT_LOG'}, {'corrected_timestamps':False}])
def test_secure_metadata(tmp_path,change):
    meta=source(tmp_path);meta.update(change)
    (tmp_path/'acquisition.json').write_text(json.dumps(meta))
    with pytest.raises(AccessBlocked) as e: inspect_local(tmp_path,BUDGET)
    assert 'DO_NOT_LOG' not in str(e.value)


def test_integrity_and_budget(tmp_path):
    source(tmp_path)
    with pytest.raises(AccessBlocked,match='budget'): inspect_local(tmp_path,{**BUDGET,'max_source_bytes':1})
    (tmp_path/'unit.json').write_text('[ ]')
    with pytest.raises(AccessBlocked,match='checksum'): inspect_local(tmp_path,BUDGET)


def test_schema_qc_never_authenticates_fixture(tmp_path):
    records=[]
    for ident in ['a','b']:
        t=list(range(-2,601))
        # Four identical samples per second; valid 0.25 s cadence.
        t=[x/4 for x in range(-8,2401)]
        records.append(dict(_id=ident,timestamp=t,x_position=[1000-10*x for x in t],
                            y_position=[18]*len(t),length=15,width=6,direction=-1))
    source(tmp_path,json.dumps(records).encode())
    result=inspect_local(tmp_path,BUDGET)
    assert result['min_tracks_per_second']==2
    assert result['status']=='LOCAL_SCHEMA_QC_PASSED_PROVENANCE_REVIEW_REQUIRED'
    assert result['training']=='BLOCKED_PENDING_REAL_SOURCE_REVIEW'


def test_prior_research_preservation():
    from pathlib import Path
    manifest=json.loads(Path('outputs/i24_continuous/preservation.json').read_text())
    import subprocess
    for name, digest in manifest['files'].items():
        current=Path(name).read_bytes()
        if name in manifest['append_only']:
            original=subprocess.check_output(['git','show',manifest['base_commit']+':'+name])
            assert current.startswith(original)
            current=original
        assert hashlib.sha256(current).hexdigest()==digest,name


def test_cli_blocks_without_real_data(tmp_path):
    import subprocess
    report=tmp_path/'status.json'
    result=subprocess.run(['uv','run','python','scripts/run_i24_continuous_pipeline.py',
        '--config','configs/i24_continuous_smoke.yaml','--data-root',str(tmp_path),
        '--report',str(report)],capture_output=True,text=True)
    assert result.returncode==2
    status=json.loads(report.read_text())
    assert status['status']=='BLOCKED' and status['training']=='NOT_RUN'
