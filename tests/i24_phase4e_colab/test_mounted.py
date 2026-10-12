import hashlib
import io
import json
from pathlib import Path
import zipfile
import pytest
from orchestra_wm.i24_phase4e_colab import mounted


def archive(tmp_path):
    session=mounted.SESSIONS['2022-11-23'];inner=io.BytesIO()
    with zipfile.ZipFile(inner,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr(session+'.json',json.dumps([{'_id':'fixture','timestamp':[1,2,3]}]))
    path=tmp_path/'11-23-2022.zip'
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('folder/README.txt',session+' Data Use Agreement')
        z.writestr('folder/'+session+'.zip',inner.getvalue())
    metadata=dict(name=path.name,size=str(path.stat().st_size),md5Checksum=hashlib.md5(path.read_bytes()).hexdigest(),version='fixture')
    return path,metadata,inner.getvalue()


def test_original_mounted_archive_without_credentials(tmp_path,monkeypatch):
    import urllib.request
    monkeypatch.setattr(urllib.request.OpenerDirector,'open',lambda *a,**k:pytest.fail('Network requested'))
    path,meta,inner=archive(tmp_path);audit={}
    assert list(mounted.records(path,meta,'2022-11-23',audit))[0]['_id']=='fixture'
    assert audit['nested_zip_sha256']==hashlib.sha256(inner).hexdigest()
    assert audit['full_outer_archive_md5_verified'] and audit['outer_selected_member_crc_verified']
    assert audit['nested_members_crc_verified'] and audit['mounted_stat_stable']


def test_wrong_original_and_changed_source_rejected(tmp_path):
    path,meta,_=archive(tmp_path)
    with pytest.raises(ValueError,match='MD5'):
        list(mounted.records(path,{**meta,'md5Checksum':'0'*32},'2022-11-23',{}))
    iterator=mounted.records(path,meta,'2022-11-23',{})
    next(iterator)
    import os
    stat=path.stat();os.utime(path,ns=(stat.st_atime_ns,stat.st_mtime_ns+1000000000))
    with pytest.raises(ValueError,match='changed'):list(iterator)


def test_ambiguous_source_fails_without_date_substitution(tmp_path):
    a,meta,_=archive(tmp_path);b=tmp_path/'other'/a.name;b.parent.mkdir();b.write_bytes(a.read_bytes())
    with pytest.raises(ValueError,match='one original'):mounted.source_path([a,b],meta)


def test_drive_shortcuts_are_followed_without_cycles_or_duplicate_archives(tmp_path):
    mydrive=tmp_path/'MyDrive';mydrive.mkdir()
    shared=tmp_path/'shared';shared.mkdir()
    archive=shared/'11-23-2022.zip';archive.write_bytes(b'original')
    (mydrive/'Source shortcut').symlink_to(shared,target_is_directory=True)
    (shared/'cycle').symlink_to(mydrive,target_is_directory=True)
    private=mydrive/'private';private.mkdir();(private/'backup.part000').write_bytes(b'private output')
    found=mounted.index_drive([mydrive,shared],exclude=[private])
    assert [p.resolve() for p in found]==[archive]
