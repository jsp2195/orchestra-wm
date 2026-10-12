import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from orchestra_wm.i24_phase4e_colab import data
from orchestra_wm.i24_phase4e import data as original
from orchestra_wm.i24.data import load_scene


def test_interrupted_native_resume_matches_frozen_converter(tmp_path,monkeypatch):
    meta=dict(name='11-23-2022.zip',size='123',md5Checksum='0'*32,version='1')
    cfg=data.configuration('2022-11-23',meta);start=cfg['blocks'][0]['start']
    t=start+2+np.arange(2950)*.04
    fixtures=[dict(_id=str(i),timestamp=t.tolist(),x_position=(320070-np.arange(len(t))*.8).tolist(),
                   y_position=[18+12*(i%3)]*len(t),length=15,width=6,direction=-1) for i in range(7)]
    fail=[True]
    def records(path,metadata,date,audit):
        for i,r in enumerate(fixtures):
            if fail[0] and i==5:raise RuntimeError('simulated disconnect')
            yield r
        audit.update(nested_zip_sha256='1'*64,nested_members_crc_verified=True,outer_selected_member_crc_verified=True)
    monkeypatch.setattr(data,'records',records)
    monkeypatch.setattr(data.shutil,'disk_usage',lambda _:SimpleNamespace(free=20_000_000_000))
    out=tmp_path/'resumed'
    with pytest.raises(RuntimeError):data.prepare('2022-11-23',meta,out,'unused',checkpoint_every=2)
    assert (out/'private/native_checkpoint.json.gz').exists()
    # A shard flushed after the last checkpoint must be preserved outside accepted data.
    orphan=out/'native'/cfg['blocks'][0]['name']/'part-99999.parquet';orphan.write_bytes(b'not committed')
    fail[0]=False
    resumed=data.prepare('2022-11-23',meta,out,'unused',checkpoint_every=2)
    monkeypatch.setattr(original,'remote_records',lambda m,d,a:records(None,m,d,a))
    expected=original.prepare('2022-11-23',meta,tmp_path/'reference')
    assert resumed['counts']==expected['counts']
    assert resumed['scenes']==expected['scenes']
    assert resumed['scene_counts']==expected['scene_counts']
    for item in expected['scenes']:
        x=load_scene(out/item['path']);y=load_scene(tmp_path/'reference'/item['path'])
        np.testing.assert_array_equal(x.values,y.values)
        np.testing.assert_array_equal(x.existence,y.existence)
        with np.load((out/item['path']).with_suffix('.fields.npz')) as a, np.load((tmp_path/'reference'/item['path']).with_suffix('.fields.npz')) as b:
            for name in a.files:np.testing.assert_array_equal(a[name],b[name])
    assert list((out/'private').rglob('part-99999.parquet'))
    assert not orphan.exists()
