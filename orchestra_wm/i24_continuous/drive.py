"""Restricted, authenticated Google Drive archive transfer; no secret persistence."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import time
import urllib.error
import urllib.request
import zipfile

FILES = (
    ('1fbWITAB78jWrm0mQr_O4TX52aRxDMpLB', '11-21-2022.zip', 6248968002),
    ('16zYgDwdljmgyXrYsjPIkL-vw2_K9PsTn', 'auxiliary_information.zip', 4917485),
)
API = 'https://www.googleapis.com/drive/v3/files/'


class TransferBlocked(RuntimeError):
    """Sanitized diagnostic: never include request headers, bodies or credentials."""


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise TransferBlocked('Download redirect requires separate verified-host review; authorization was not forwarded')


def request(url, token, offset=None):
    headers = {'Authorization': 'Bearer ' + token, 'Accept-Encoding': 'identity'}
    if offset:
        headers['Range'] = f'bytes={offset}-'
    opener = urllib.request.build_opener(NoRedirect())
    for attempt in range(3):
        try:
            return opener.open(urllib.request.Request(url, headers=headers), timeout=60)
        except urllib.error.HTTPError as e:
            status = e.code
            e.close()
            if status in (429, 500, 502, 503, 504) and attempt < 2:
                time.sleep(2 ** attempt)
                continue
            raise TransferBlocked(f'Google Drive API HTTP {status}; verify scoped authorization, file permission or quota') from None
        except urllib.error.URLError:
            raise TransferBlocked('Google Drive API transport unavailable; check configured network access') from None


def metadata(file_id, token):
    fields = 'id,name,size,mimeType,md5Checksum,version,capabilities(canDownload)'
    with request(API + file_id + '?fields=' + fields + '&supportsAllDrives=true', token) as response:
        if response.status != 200:
            raise TransferBlocked('Unexpected metadata response')
        payload = response.read(65537)
    if len(payload) > 65536:
        raise TransferBlocked('Metadata exceeds bounded response budget')
    try:
        data = json.loads(payload)
    except ValueError:
        raise TransferBlocked('Metadata response is not JSON; HTML/login pages are not data') from None
    return data


def validate_metadata(data, expected):
    file_id, name, size = expected
    if data.get('id') != file_id or data.get('name') != name or str(data.get('size')) != str(size):
        raise TransferBlocked('Drive file identity/name/size differs from the user-specified archive')
    if data.get('mimeType') not in ('application/zip', 'application/x-zip-compressed', 'application/octet-stream'):
        raise TransferBlocked('Expected a binary archive, not a Google document or HTML response')
    if data.get('capabilities', {}).get('canDownload') is not True:
        raise TransferBlocked('Drive does not grant download permission')
    md5 = data.get('md5Checksum', '')
    if len(md5) != 32 or any(c not in '0123456789abcdef' for c in md5):
        raise TransferBlocked('Publisher-side Drive MD5 is required for this transfer')
    return {k: data[k] for k in ('id', 'name', 'size', 'mimeType', 'md5Checksum', 'version') if k in data}


def hashes(path):
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            sha.update(chunk); md5.update(chunk)
    return sha.hexdigest(), md5.hexdigest()


def inspect_zip(path):
    """List only; never extract arbitrary members or infer source semantics."""
    members = []
    try:
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                p = PurePosixPath(info.filename)
                if p.is_absolute() or '..' in p.parts or '\\' in info.filename or (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise TransferBlocked('Archive contains an unsafe path or symlink; extraction forbidden')
                members.append({'name':info.filename,'bytes':info.file_size,
                                'compressed_bytes':info.compress_size,'crc32':f'{info.CRC:08x}'})
    except zipfile.BadZipFile:
        raise TransferBlocked('Downloaded bytes are not a valid ZIP archive') from None
    return members


def transfer(expected, meta, root, token, min_free=10_000_000_000):
    file_id, name, size = expected
    dest, partial = root/name, root/(name+'.part')
    stamp = root/(name+'.transfer.json')
    for p in (dest, partial, stamp):
        if p.is_symlink():
            raise TransferBlocked('Transfer paths may not be symlinks')
    if stamp.exists():
        if json.loads(stamp.read_text()) != meta:
            raise TransferBlocked('Drive source metadata changed; preserve existing bytes for review')
    else:
        if partial.exists():
            raise TransferBlocked('Unidentified partial download; preserve it and use a separate data root')
        stamp.write_text(json.dumps(meta, indent=2)+'\n')
    if not dest.exists():
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > size:
            raise TransferBlocked('Partial file exceeds declared archive size')
        if shutil.disk_usage(root).free < size-offset+min_free:
            raise TransferBlocked('Transfer would violate the 10 GB free-disk reserve')
        if offset < size:
            with request(API+file_id+'?alt=media&supportsAllDrives=true', token, offset) as response:
                expected_status = 206 if offset else 200
                if response.status != expected_status:
                    raise TransferBlocked('Server did not honor the requested transfer/range; partial bytes preserved')
                if offset and response.headers.get('Content-Range') != f'bytes {offset}-{size-1}/{size}':
                    raise TransferBlocked('Resume Content-Range does not match the verified file size')
                if 'text/html' in response.headers.get('Content-Type','').lower():
                    raise TransferBlocked('HTML/login response rejected')
                length = response.headers.get('Content-Length')
                if length is not None and int(length) != size-offset:
                    raise TransferBlocked('HTTP transfer size mismatch')
                with partial.open('ab') as f:
                    while chunk := response.read(1024 * 1024):
                        if f.tell()+len(chunk) > size:
                            raise TransferBlocked('Response exceeds verified archive size')
                        if shutil.disk_usage(root).free-len(chunk) < min_free:
                            raise TransferBlocked('Disk reserve reached; partial bytes retained')
                        f.write(chunk)
                    f.flush(); os.fsync(f.fileno())
        source = partial
    else:
        source = dest
    if source.stat().st_size != size:
        raise TransferBlocked('Transfer incomplete; rerun to request a verified HTTP Range')
    sha, md5 = hashes(source)
    if md5 != meta['md5Checksum']:
        raise TransferBlocked('Drive MD5 mismatch; bytes preserved and not accepted')
    members = inspect_zip(source)
    if source != dest:
        source.replace(dest)
    return {'id':file_id,'name':name,'bytes':size,'sha256':sha,'md5':md5,
            'members':members,'uncompressed_bytes':sum(m['bytes'] for m in members),
            'status':'ARCHIVE_VERIFIED_NOT_YET_TRAJECTORY_VALIDATED'}


def run(root, max_download_bytes=6_300_000_000):
    token = os.environ.get('GOOGLE_DRIVE_ACCESS_TOKEN')
    if not token:
        raise TransferBlocked('Missing GOOGLE_DRIVE_ACCESS_TOKEN secure environment binding; do not paste credentials in chat')
    if sum(f[2] for f in FILES) > max_download_bytes:
        raise TransferBlocked('Declared transfer exceeds configured budget')
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    validated = [(f, validate_metadata(metadata(f[0],token),f)) for f in FILES]
    remaining = sum(f[2] - min(f[2], (root/(f[1]+'.part')).stat().st_size) if (root/(f[1]+'.part')).exists() else (0 if (root/f[1]).exists() else f[2]) for f in FILES)
    if shutil.disk_usage(root).free < remaining+10_000_000_000:
        raise TransferBlocked('Both archives would exceed free-disk reserve')
    return [transfer(f,m,root,token) for f,m in validated]
