import io
import json
from pathlib import Path
import tarfile
import pytest
from orchestra_wm.i24_phase4e_colab.storage import snapshot,verify_snapshot,extract_verified,digest
from orchestra_wm.i24_phase4e_colab.recovery import recover_date


def test_snapshot_actual_readback_and_no_overwrite(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path('data').mkdir();Path('data/fixture.txt').write_text('private fixture')
    receipt=snapshot(['data'],tmp_path/'drive/backups','fixture')
    r=verify_snapshot(receipt,['data/fixture.txt'])
    extract_verified(r['archive'],r['files'],tmp_path/'restored')
    assert (tmp_path/'restored/data/fixture.txt').read_text()=='private fixture'
    (tmp_path/'restored/data/fixture.txt').write_text('preserve this')
    with pytest.raises(ValueError):extract_verified(r['archive'],r['files'],tmp_path/'restored')
    assert (tmp_path/'restored/data/fixture.txt').read_text()=='preserve this'
    Path(r['archive']).write_bytes(b'corrupted remote')
    with pytest.raises(ValueError):verify_snapshot(receipt)


def test_traversal_and_symlink_members_rejected(tmp_path):
    for name,kind in [('../escape',tarfile.REGTYPE),('link',tarfile.SYMTYPE)]:
        archive=tmp_path/'bad.tar.gz'
        with tarfile.open(archive,'w:gz') as tar:
            info=tarfile.TarInfo(name);info.type=kind;info.linkname='/tmp/escape';info.size=0;tar.addfile(info,io.BytesIO())
        with pytest.raises(ValueError):extract_verified(archive,{name:dict(bytes=0,sha256='invalid')},tmp_path/'out')


def test_existing_completed_date_recovery_without_private_ids(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    base=Path('outputs/i24_phase4e/data_v4/2022-11-22');base.mkdir(parents=True)
    (base/'canonical').mkdir();(base/'canonical/fixture.npz').write_bytes(b'fixture')
    (base/'native_manifest.json').write_text('{}')
    (base/'data_manifest.json').write_text(json.dumps(dict(source_dates=['2022-11-22'],scene_counts={'train':92})))
    (base/'source_verification.json').write_text(json.dumps(dict(native_manifest_sha256=digest(base/'native_manifest.json'))))
    receipt=snapshot([base],tmp_path/'backup','fixture');r=json.loads(receipt.read_text())
    archive=Path(r['archive']);part=tmp_path/'completed.part000';part.write_bytes(archive.read_bytes())
    manifest=tmp_path/'TEST_RECOVERY_MANIFEST.json'
    manifest.write_text(json.dumps(dict(archive_sha256=r['archive_sha256'],archive_bytes=archive.stat().st_size,
        files=r['files'],parts=[dict(index=0,sha256=digest(part),remote_bytes=part.stat().st_size)])))
    spec=dict(manifest_name=manifest.name,manifest_sha256=digest(manifest),archive_sha256=r['archive_sha256'],archive_bytes=archive.stat().st_size)
    # Move all original fixture files aside: exercise restoration, not just the fast path.
    base.rename(base.with_name('preserved-original'))
    recover_date('2022-11-22',[manifest,part],spec,'staging')
    assert (base/'canonical/fixture.npz').read_bytes()==b'fixture'
    recover_date('2022-11-22',[manifest,part],spec,'staging')
    assert base.with_name('preserved-original').exists()


def test_frozen_reserve_fails_before_creating_payload(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from orchestra_wm.i24_phase4e_colab import storage
    monkeypatch.setattr(storage,'FREE_RESERVE_BYTES',10_000_000_000)
    monkeypatch.setattr(storage.shutil,'disk_usage',lambda p:SimpleNamespace(free=10_000_000_001))
    with pytest.raises(OSError,match='reserve'):storage.atomic_bytes(tmp_path/'new.bin',b'oversize')
    assert not list(tmp_path.iterdir())
