"""Private resumable Drive backup, proxy-compatible, with full SHA256 readback.

Use only a secure environment binding. Never print capability URLs or responses
containing credentials. State and receipts must be under ignored work/.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import urllib.error
import urllib.parse
import urllib.request

BASE='https://www.googleapis.com'
CHUNK=4*1024*1024


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(CHUNK),b''):h.update(b)
    return h.hexdigest()


def private_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
    with os.fdopen(fd,'w') as f:json.dump(value,f,indent=2);f.write('\n')
    temporary.replace(path)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args):raise RuntimeError('Redirect blocked; remote capability not exposed')


def upload(path,state_path,receipt_path):
    path=Path(path);state_path=Path(state_path);receipt_path=Path(receipt_path)
    work=Path('work').resolve()
    if not state_path.resolve().is_relative_to(work) or not receipt_path.resolve().is_relative_to(work):
        raise ValueError('Private upload state and receipt must remain under ignored work/')
    token=os.environ.get('GOOGLE_DRIVE_ACCESS_TOKEN')
    if not token:raise SystemExit('Secure GOOGLE_DRIVE_ACCESS_TOKEN binding unavailable; files preserved')
    auth={'Authorization':'Bearer '+token};size=path.stat().st_size;expected=digest(path)
    opener=urllib.request.build_opener(NoRedirect()) # Retains environment HTTPS proxy and CA trust.
    def request(url,data=None,method='GET',headers=None,partial=False):
        try:return opener.open(urllib.request.Request(url,data=data,method=method,headers=headers or {}),timeout=120)
        except urllib.error.HTTPError as e:
            if partial and e.code==308:return e
            status=e.code;reasons=[]
            try:
                error=json.loads(e.read(16384)).get('error',{})
                reasons=[r.get('reason','') for r in error.get('errors',[])]
                reasons=[r for r in reasons if re.fullmatch(r'[A-Za-z0-9_]+',r)]
            except (ValueError,AttributeError,TypeError):pass
            e.close();reason=','.join(reasons) or 'no structured reason'
            raise SystemExit(f'HTTP {status} during {method} ({reason}); stopped, local bytes and resumable state preserved') from None
        except urllib.error.URLError:
            raise SystemExit('Network request failed; local bytes and resumable state preserved; no URL logged') from None
    if receipt_path.exists():
        receipt=json.loads(receipt_path.read_text())
        assert receipt['sha256']==expected and receipt['readback_sha256']==expected and receipt['bytes']==size
        print('Existing verified receipt matches the unchanged local artifact');return receipt
    state=json.loads(state_path.read_text()) if state_path.exists() else None
    if state:
        if state['sha256']!=expected or state['bytes']!=size:raise ValueError('Local artifact changed; existing remote session preserved')
    else:
        metadata=json.dumps(dict(name=path.name,description='Private ORCHESTRA Phase 4C recovery artifact')).encode()
        with request(BASE+'/upload/drive/v3/files?uploadType=resumable&fields=id,name,size,webViewLink',metadata,'POST',
                     {**auth,'Content-Type':'application/json','X-Upload-Content-Type':'application/octet-stream','X-Upload-Content-Length':str(size)}) as response:
            session=response.headers['Location']
        state=dict(session=session,sha256=expected,bytes=size,acknowledged=0)
        private_json(state_path,state)
    parsed=urllib.parse.urlsplit(state['session'])
    if parsed.scheme!='https' or parsed.hostname!='www.googleapis.com' or parsed.username or parsed.port not in (None,443):
        raise ValueError('Unapproved session host; capability not disclosed')
    info=state.get('completed_file')
    if info is None:
        # A zero-length status PUT determines the actual acknowledged offset.
        # Session URL is the capability: intentionally no Authorization header.
        with request(state['session'],b'','PUT',{'Content-Length':'0','Content-Range':f'bytes */{size}'},True) as response:
            if response.code in (200,201):info=json.load(response)
            else:
                received=response.headers.get('Range','')
                match=re.fullmatch(r'bytes=0-(\d+)',received)
                if received and not match:raise ValueError('Malformed acknowledged byte range; session preserved')
                state['acknowledged']=int(match.group(1))+1 if match else 0
        if not 0<=state['acknowledged']<=size:raise ValueError('Invalid acknowledged byte range')
        private_json(state_path,state)
    if info is None:
        with path.open('rb') as f:
            f.seek(state['acknowledged'])
            while state['acknowledged']<size:
                offset=state['acknowledged'];b=f.read(CHUNK);end=offset+len(b)-1
                with request(state['session'],b,'PUT',{'Content-Type':'application/octet-stream','Content-Length':str(len(b)),
                            'Content-Range':f'bytes {offset}-{end}/{size}'},True) as response:
                    if response.code==308:
                        if response.headers.get('Range')!=f'bytes=0-{end}':raise ValueError('Unexpected acknowledged byte range; stopped')
                    else:
                        assert response.code in (200,201) and end==size-1;info=json.load(response)
                state['acknowledged']=end+1
                if info is not None:state['completed_file']=info
                private_json(state_path,state)
                print(json.dumps(dict(uploaded_bytes=end+1,total_bytes=size)),flush=True)
    assert info and int(info['size'])==size
    state['completed_file']=info;private_json(state_path,state)
    ident=info['id']
    with request(BASE+'/drive/v3/files/'+ident+'?fields=id,size,permissions(type,role),webViewLink',headers=auth) as response:metadata=json.load(response)
    assert int(metadata['size'])==size
    permissions=metadata.get('permissions')
    if not permissions or any(p['type']!='user' or p['role']!='owner' for p in permissions):raise ValueError('Remote object is not confirmed owner-only private')
    h=hashlib.sha256();count=0
    with request(BASE+'/drive/v3/files/'+ident+'?alt=media',headers=auth) as response:
        assert response.status==200
        for b in iter(lambda:response.read(CHUNK),b''):h.update(b);count+=len(b)
    if count!=size or h.hexdigest()!=expected or digest(path)!=expected:raise ValueError('Full readback integrity check failed')
    receipt=dict(private=True,file_id=ident,bytes=size,sha256=expected,readback_sha256=h.hexdigest(),
                 verified_at_utc=datetime.now(timezone.utc).isoformat(),
                 webViewLink=metadata.get('webViewLink') or f'https://drive.google.com/file/d/{ident}/view',
                 verification='Authenticated full readback; exact size and SHA256; owner-only permissions')
    private_json(receipt_path,receipt);print(json.dumps(receipt,indent=2));return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('file');p.add_argument('--state',required=True);p.add_argument('--receipt',required=True)
    a=p.parse_args();upload(a.file,a.state,a.receipt)
