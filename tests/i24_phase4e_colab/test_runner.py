import json
from pathlib import Path
import pytest
from orchestra_wm.i24_phase4e_colab.runner import workspace, prepare_cache_restart, prepare_date


def test_fresh_workspace_and_code_lock_survive_runtime_recreation(tmp_path,monkeypatch):
    repo=Path.cwd()
    monkeypatch.chdir(tmp_path)
    root=workspace(repo,tmp_path/'drive/private')
    lock=json.loads(Path('work/phase4e/colab_code_lock.json').read_text())
    assert 'orchestra_wm/i24_phase4e_colab/runner.py' in lock
    assert (root/'scripts').resolve()==repo/'scripts'
    workspace(repo,tmp_path/'drive/private')
    assert json.loads(Path('work/phase4e/colab_code_lock.json').read_text())==lock
    cache=Path('outputs/i24_phase4e/cache_v1');cache.mkdir(parents=True)
    (cache/'broken.npz').write_bytes(b'partial')
    prepare_cache_restart(cache)
    assert not cache.exists()
    assert len(list(Path('work/phase4e').glob('interrupted-cache-*/broken.npz')))==1


def test_no_reingestion_of_completed_dates(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError,match='never reingested'):
        prepare_date('2022-11-22',[],{},tmp_path/'backup')
