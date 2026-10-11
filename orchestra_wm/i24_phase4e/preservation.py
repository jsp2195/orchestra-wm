"""Private multipart preservation with authenticated readback and immutable hashes."""
import hashlib,json,os,tarfile,urllib.request,urllib.error
from pathlib import Path
from orchestra_wm.i24.data import sha
from orchestra_wm.i24_continuous.source import write_immutable
from .ranges import NoRedirect,SourceBlocked


def request(path, data=None, content_type=None):
    token=os.environ.get('GOOGLE_DRIVE_ACCESS_TOKEN')
    if not token:raise SourceBlocked('Missing secure Drive token')
    headers={'Authorization':'Bearer '+token}
    if content_type:headers['Content-Type']=content_type
    req=urllib.request.Request('https://www.googleapis.com/'+path,data=data,headers=headers)
    try:
        with urllib.request.build_opener(NoRedirect()).open(req,timeout=60) as response:
            return response.read()
    except urllib.error.HTTPError as e:
        status=e.code;e.close()
        raise SourceBlocked(f'Private preservation HTTP {status}; no authentication retry') from None
    except urllib.error.URLError:
        raise SourceBlocked('Private preservation transport failure') from None


def upload_verified(name,payload):
    boundary='orchestra_phase4e_private'
    meta=json.dumps(dict(name=name,mimeType='application/octet-stream'))
    body=('--'+boundary+'\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n'+meta+'\r\n--'+boundary+'\r\nContent-Type: application/octet-stream\r\n\r\n').encode()+payload+('\r\n--'+boundary+'--\r\n').encode()
    result=json.loads(request('upload/drive/v3/files?uploadType=multipart&fields=id,name,size,md5Checksum',body,'multipart/related; boundary='+boundary))
    remote=request('drive/v3/files/'+result['id']+'?alt=media')
    h=hashlib.sha256(payload).hexdigest()
    if len(remote)!=len(payload) or hashlib.sha256(remote).hexdigest()!=h:raise ValueError('Private readback checksum failure')
    permissions=json.loads(request('drive/v3/files/'+result['id']+'?fields=permissions(type,role),shared'))
    if permissions.get('shared',True) or not permissions.get('permissions') or not all(x['type']=='user' and x['role']=='owner' for x in permissions['permissions']):
        raise ValueError('Private backup owner-only permission check failed')
    return dict(file_id=result['id'],name=result['name'],bytes=len(payload),sha256=h,readback_sha256=h,owner_only=True)


def backup(roots,receipt_path,label,cache=None):
    receipt_path=Path(receipt_path);receipt_path.parent.mkdir(parents=True,exist_ok=True)
    files={}
    for root in map(Path,roots):
        for p in sorted(root.rglob('*')):
            if p.is_file():
                if p.is_symlink():raise ValueError('Symlinks forbidden in recovery bundle')
                files[str(p)]=dict(bytes=p.stat().st_size,sha256=sha(p))
    if not files:raise ValueError('Empty backup scope')
    archive=receipt_path.with_suffix('.tar.gz')
    if archive.exists():
        inventory=json.loads(archive.with_suffix('.inventory.json').read_text())
        if inventory!=files:raise ValueError('Backup source changed; use a new receipt')
    else:
        write_immutable(archive.with_suffix('.inventory.json'),files)
        partial=archive.with_suffix(archive.suffix+'.part')
        if partial.exists():raise ValueError('Incomplete recovery archive preserved; choose a new receipt path')
        with tarfile.open(partial,'w:gz',compresslevel=2) as tar:
            for name in files:tar.add(name,arcname=name,recursive=False)
        partial.replace(archive)
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers()
        if {m.name for m in members}!=set(files) or len(members)!=len(files):raise ValueError('Recovery member inventory differs')
        for member in members:
            if not member.isfile() or member.size!=files[member.name]['bytes']:raise ValueError('Recovery member type/size mismatch')
            if hashlib.file_digest(tar.extractfile(member),'sha256').hexdigest()!=files[member.name]['sha256']:
                raise ValueError('Recovery member checksum mismatch')
    archive_hash=sha(archive);parts=[];progress=receipt_path.with_suffix('.progress.json')
    if progress.exists():
        old=json.loads(progress.read_text())
        if old['archive_sha256']!=archive_hash:raise ValueError('Recovery archive changed')
        parts=old['parts']
    with archive.open('rb') as f:
        index=0
        while payload:=f.read(24*1024*1024):
            if index<len(parts):
                if hashlib.sha256(payload).hexdigest()!=parts[index]['sha256']:raise ValueError('Part changed')
            else:
                part=upload_verified(f'ORCHESTRA_PHASE4E_{label}_{archive_hash[:8]}.part{index:03}',payload)
                parts.append(part)
                tmp=progress.with_suffix('.tmp');tmp.write_text(json.dumps(dict(archive_sha256=archive_hash,parts=parts),indent=2)+'\n');tmp.replace(progress)
                print('Verified private part',index,flush=True)
            index+=1
    result=dict(status='COMPLETE_AUTHENTICATED_READBACK_SHA256_VERIFIED',archive_sha256=archive_hash,
                archive_bytes=archive.stat().st_size,files=files,parts=parts,owner_only=True,
                cache_manifest_sha256=sha(Path(cache)/'manifest.json') if cache else None)
    write_immutable(receipt_path,result)
    manifest_receipt=receipt_path.with_suffix('.manifest_receipt.json')
    if not manifest_receipt.exists():
        uploaded=upload_verified(f'ORCHESTRA_PHASE4E_{label}_{archive_hash[:8]}_manifest.json',receipt_path.read_bytes())
        write_immutable(manifest_receipt,uploaded)
    return result


def verify_preservation(receipt_path,cache):
    receipt=json.loads(Path(receipt_path).read_text())
    if receipt['status']!='COMPLETE_AUTHENTICATED_READBACK_SHA256_VERIFIED' or receipt['owner_only'] is not True:
        raise ValueError('Verified private recovery required before substantive training')
    if receipt['cache_manifest_sha256']!=sha(Path(cache)/'manifest.json'):raise ValueError('Preserved feature cache mismatch')
    manifest_receipt=json.loads(Path(receipt_path).with_suffix('.manifest_receipt.json').read_text())
    if not manifest_receipt['owner_only'] or manifest_receipt['readback_sha256']!=sha(receipt_path):raise ValueError('Private recovery manifest not verified')
    cache_manifest=json.loads((Path(cache)/'manifest.json').read_text())
    required={str(Path(cache)/'manifest.json')}
    required.update(str(Path(cache)/e['path']) for e in cache_manifest['entries'])
    for manifest_path in cache_manifest['config']['source_manifests']:
        required.add(manifest_path)
        source=json.loads(Path(manifest_path).read_text())
        required.update(str(Path(manifest_path).parent/p) for p in source['artifacts'])
    if not required.issubset(receipt['files']):raise ValueError('Required canonical/cache files missing from verified backup')
    for name,entry in receipt['files'].items():
        if sha(name)!=entry['sha256']:raise ValueError('Preserved data changed')
    for part in receipt['parts']:
        if not part['owner_only'] or part['sha256']!=part['readback_sha256']:raise ValueError('Unverified backup part')
    return receipt
