import importlib.util
import io
import json
from pathlib import Path
import urllib.error
import pytest


def module():
    spec=importlib.util.spec_from_file_location('private_upload',Path(__file__).parents[2]/'scripts/upload_i24_private_artifact.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


class Response(io.BytesIO):
    def __init__(self,body=b'',code=200,headers=None):
        super().__init__(body);self.code=self.status=code;self.headers=headers or {}


def test_proxy_compatible_puts_never_repeat_oauth_and_readback_is_exact(tmp_path,monkeypatch):
    m=module();monkeypatch.chdir(tmp_path);monkeypatch.setenv('GOOGLE_DRIVE_ACCESS_TOKEN','test-placeholder')
    Path('work').mkdir();path=Path('work/artifact.bin');path.write_bytes(b'verified bytes')
    calls=[]
    metadata=dict(id='test-file-id',size=str(path.stat().st_size),permissions=[dict(type='user',role='owner')])
    responses=[Response(headers={'Location':m.BASE+'/upload-test-session'}),Response(json.dumps(metadata).encode()),Response(json.dumps(metadata).encode()),Response(path.read_bytes())]
    class Opener:
        def open(self,request,timeout):calls.append(request);return responses.pop(0)
    monkeypatch.setattr(m.urllib.request,'build_opener',lambda *a:Opener())
    receipt=m.upload(path,'work/session.json','work/receipt.json')
    assert receipt['readback_sha256']==m.digest(path)
    assert [r.method for r in calls]==['POST','PUT','GET','GET']
    assert all(r.get_header('Authorization') is None for r in calls if r.method=='PUT')
    assert all(r.get_header('Authorization') for r in calls if r.method!='PUT')
    assert Path('work/session.json').stat().st_mode & 0o777 == 0o600


def test_http_authentication_failure_stops_without_deleting_local_bytes(tmp_path,monkeypatch,capsys):
    m=module();monkeypatch.chdir(tmp_path);monkeypatch.setenv('GOOGLE_DRIVE_ACCESS_TOKEN','test-placeholder')
    Path('work').mkdir();path=Path('work/artifact.bin');path.write_bytes(b'keep me')
    class Opener:
        def open(self,request,timeout):raise urllib.error.HTTPError('hidden-capability',401,'Unauthorized',{},None)
    monkeypatch.setattr(m.urllib.request,'build_opener',lambda *a:Opener())
    with pytest.raises(SystemExit,match='HTTP 401'):m.upload(path,'work/session.json','work/receipt.json')
    assert path.read_bytes()==b'keep me'
    assert not Path('work/receipt.json').exists()
    assert 'hidden-capability' not in capsys.readouterr().out


def test_resume_uses_server_acknowledgement_without_restarting_bytes(tmp_path,monkeypatch):
    m=module();monkeypatch.chdir(tmp_path);monkeypatch.setenv('GOOGLE_DRIVE_ACCESS_TOKEN','test-placeholder')
    Path('work').mkdir();path=Path('work/artifact.bin');path.write_bytes(b'verified bytes')
    m.private_json(Path('work/session.json'),dict(session=m.BASE+'/upload-test-session',sha256=m.digest(path),bytes=path.stat().st_size,acknowledged=0))
    metadata=dict(id='test-file-id',size=str(path.stat().st_size),permissions=[dict(type='user',role='owner')])
    responses=[Response(code=308,headers={'Range':'bytes=0-3'}),Response(json.dumps(metadata).encode()),Response(json.dumps(metadata).encode()),Response(path.read_bytes())]
    calls=[]
    class Opener:
        def open(self,request,timeout):calls.append(request);return responses.pop(0)
    monkeypatch.setattr(m.urllib.request,'build_opener',lambda *a:Opener())
    m.upload(path,'work/session.json','work/receipt.json')
    assert [r.method for r in calls]==['PUT','PUT','GET','GET']
    assert calls[1].data==path.read_bytes()[4:]
    assert calls[1].get_header('Content-range')=='bytes 4-13/14'
