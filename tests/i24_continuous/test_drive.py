import hashlib
import io
import json
import zipfile
from email.message import Message
import pytest
from orchestra_wm.i24_continuous import drive


def archive():
    b=io.BytesIO()
    with zipfile.ZipFile(b,'w') as z:z.writestr('original/track.json','[]')
    return b.getvalue()


def setup(payload):
    expected=('example-id','unit.zip',len(payload))
    meta={'id':expected[0],'name':expected[1],'size':str(len(payload)),
          'mimeType':'application/zip','md5Checksum':hashlib.md5(payload).hexdigest(),
          'version':'1','capabilities':{'canDownload':True}}
    return expected,drive.validate_metadata(meta,expected)


class Response(io.BytesIO):
    def __init__(self,payload,status=200,headers=None):
        super().__init__(payload);self.status=status;self.headers=headers or {}


def test_no_secret_no_request(monkeypatch,tmp_path):
    monkeypatch.delenv('GOOGLE_DRIVE_ACCESS_TOKEN',raising=False)
    monkeypatch.setattr(drive,'request',lambda *a:pytest.fail('Network called without credential'))
    with pytest.raises(drive.TransferBlocked,match='Missing'):drive.run(tmp_path)


def test_download_and_cache(monkeypatch,tmp_path):
    payload=archive();expected,meta=setup(payload)
    monkeypatch.setattr(drive,'request',lambda *a:Response(payload,headers={'Content-Length':str(len(payload))}))
    result=drive.transfer(expected,meta,tmp_path,'UNIT_SECRET',min_free=0)
    assert result['sha256']==hashlib.sha256(payload).hexdigest()
    assert result['members'][0]['name']=='original/track.json'
    monkeypatch.setattr(drive,'request',lambda *a:pytest.fail('Redownloaded verified archive'))
    assert drive.transfer(expected,meta,tmp_path,'UNIT_SECRET',min_free=0)==result
    assert 'UNIT_SECRET' not in (tmp_path/'unit.zip.transfer.json').read_text()


def test_supported_resume(monkeypatch,tmp_path):
    payload=archive();expected,meta=setup(payload);offset=20
    (tmp_path/'unit.zip.part').write_bytes(payload[:offset])
    (tmp_path/'unit.zip.transfer.json').write_text(json.dumps(meta))
    def request(url,token,start):
        assert start==offset
        return Response(payload[offset:],206,{'Content-Range':f'bytes {offset}-{len(payload)-1}/{len(payload)}'})
    monkeypatch.setattr(drive,'request',request)
    drive.transfer(expected,meta,tmp_path,'UNIT_SECRET',min_free=0)
    assert (tmp_path/'unit.zip').read_bytes()==payload


@pytest.mark.parametrize('kind',['html','hash','range','disk','changed'])
def test_reject_invalid_transfer(monkeypatch,tmp_path,kind):
    payload=archive();expected,meta=setup(payload)
    headers={};status=200
    if kind=='html':headers={'Content-Type':'text/html'}
    if kind=='hash':payload=b'x'*len(payload)
    if kind=='changed':
        (tmp_path/'unit.zip.transfer.json').write_text('{}')
    if kind=='range':
        (tmp_path/'unit.zip.part').write_bytes(payload[:20])
        (tmp_path/'unit.zip.transfer.json').write_text(json.dumps(meta))
    monkeypatch.setattr(drive,'request',lambda *a:Response(payload,status,headers))
    with pytest.raises(drive.TransferBlocked):
        drive.transfer(expected,meta,tmp_path,'UNIT_SECRET',min_free=10**20 if kind=='disk' else 0)
    assert not (tmp_path/'unit.zip').exists()


def test_metadata_permission_and_size():
    payload=archive();expected,meta=setup(payload)
    with pytest.raises(drive.TransferBlocked,match='permission'):drive.validate_metadata(meta,expected)
    meta['capabilities']={'canDownload':True};meta['size']='1'
    with pytest.raises(drive.TransferBlocked,match='size'):drive.validate_metadata(meta,expected)


def test_zip_traversal(tmp_path):
    p=tmp_path/'bad.zip'
    with zipfile.ZipFile(p,'w') as z:z.writestr('../escape','bad')
    with pytest.raises(drive.TransferBlocked,match='unsafe'):drive.inspect_zip(p)


def test_redirect_never_forwards_auth():
    with pytest.raises(drive.TransferBlocked,match='not forwarded'):
        drive.NoRedirect().redirect_request(None,None,302,'',{},'https://another.example/')
