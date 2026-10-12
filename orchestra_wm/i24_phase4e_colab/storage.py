"""Filesystem-only Drive persistence. No credential lookup or Drive HTTP requests."""
import hashlib
import errno
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import uuid

STATUS = 'MOUNTED_DRIVE_READBACK_SHA256_VERIFIED'
FREE_RESERVE_BYTES = 10_000_000_000


def digest(path, algorithm='sha256'):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, algorithm).hexdigest()


def flush_file(handle):
    handle.flush()
    try:
        os.fsync(handle.fileno())
    except OSError as error:
        # Some mounted filesystems implement close/readback but not POSIX fsync.
        # Keep the explicit byte/hash readback and do not claim server-side fsync.
        if error.errno not in {errno.EINVAL, errno.ENOTSUP, errno.ENOSYS}:
            raise


def require_space(path, additional_bytes):
    if shutil.disk_usage(path).free < int(additional_bytes) + FREE_RESERVE_BYTES:
        raise OSError('Frozen 10 GB free-space reserve would be crossed; existing files preserved')


def atomic_bytes(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require_space(path.parent,len(payload))
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.pending')
    with temporary.open('xb') as handle:
        handle.write(payload)
        flush_file(handle)
    expected = hashlib.sha256(payload).hexdigest()
    if digest(temporary) != expected:
        raise ValueError('Mounted Drive write/readback mismatch; temporary preserved')
    os.replace(temporary, path)
    if digest(path) != expected:
        raise ValueError('Mounted Drive committed readback mismatch')


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, sort_keys=True, indent=2) + '\n').encode())


def safe_name(name):
    p = PurePosixPath(name)
    if not name or p.is_absolute() or '..' in p.parts or '\\' in name:
        raise ValueError('Unsafe backup member')
    return p


def verify_files(files, root=Path('.')):
    for name, entry in files.items():
        p = root / safe_name(name)
        if p.is_symlink() or p.stat().st_size != entry['bytes'] or digest(p) != entry['sha256']:
            raise ValueError('Preserved file changed: ' + name)


def extract_verified(archive, files, destination):
    """Restore absent files only, compare existing bytes, refuse symlinks/traversal."""
    destination = Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    missing=sum(entry['bytes'] for name,entry in files.items() if not (destination/safe_name(name)).exists())
    require_space(destination,missing)
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        if len(members) != len(files) or {m.name for m in members} != set(files):
            raise ValueError('Backup member inventory differs')
        for member in members:
            name = safe_name(member.name)
            entry = files[member.name]
            target = destination / name
            if not member.isfile() or member.size != entry['bytes']:
                raise ValueError('Unexpected backup member type/size')
            for parent in [target, *target.parents]:
                if parent.is_symlink():
                    raise ValueError('Symlink in recovery destination')
            if target.exists():
                verify_files({member.name: entry}, destination)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_name(target.name + '.' + uuid.uuid4().hex + '.pending')
            with tar.extractfile(member) as source, temp.open('xb') as dest:
                shutil.copyfileobj(source, dest, 1024 * 1024)
                flush_file(dest)
            if digest(temp) != entry['sha256']:
                raise ValueError('Recovery member checksum mismatch')
            os.replace(temp, target)
    verify_files(files, destination)


def snapshot(roots, backup_root, label, cache=None):
    """Immutable compressed backup plus full mounted-filesystem readback verification."""
    files = {}
    for root in map(Path, roots):
        candidates = [root] if root.is_file() else sorted(root.rglob('*'))
        for path in candidates:
            if path.is_symlink():
                raise ValueError('Do not back up symlinks')
            if path.is_file() and not path.name.endswith('.pending'):
                safe_name(str(path))
                files[str(path)] = dict(bytes=path.stat().st_size, sha256=digest(path))
    if not files:
        raise ValueError('Empty backup')
    key = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    folder = Path(backup_root) / label / key
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / 'artifacts.tar.gz'
    receipt = folder / 'receipt.json'
    if not archive.exists():
        require_space(folder,int(sum(x['bytes'] for x in files.values())*1.02)+1024*len(files))
        temp = folder / (uuid.uuid4().hex + '.pending')
        with tarfile.open(temp, 'w:gz', compresslevel=2) as tar:
            for name in files:
                tar.add(name, arcname=name, recursive=False)
        os.replace(temp, archive)
    # Reopen after close; verify ALL member bytes, not only archive size/hash.
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        if len(members) != len(files) or {m.name for m in members} != set(files):
            raise ValueError('Backup inventory mismatch')
        for m in members:
            if not m.isfile() or m.size != files[m.name]['bytes']:
                raise ValueError('Backup member type/size mismatch')
            with tar.extractfile(m) as handle:
                if hashlib.file_digest(handle, 'sha256').hexdigest() != files[m.name]['sha256']:
                    raise ValueError('Backup readback mismatch')
    result = dict(status=STATUS, files=files, archive=str(archive), archive_sha256=digest(archive),
                  archive_bytes=archive.stat().st_size,
                  cache_manifest_sha256=digest(Path(cache)/'manifest.json') if cache else None,
                  privacy='New folder in mounted MyDrive; no sharing operations. ACL not independently queried.',
                  readback_scope='Closed/reopened mounted Drive filesystem; not an independent Drive API download')
    atomic_json(receipt, result)
    return receipt


def verify_snapshot(receipt, required=(), cache=None):
    r = json.loads(Path(receipt).read_text())
    if r['status'] != STATUS or digest(r['archive']) != r['archive_sha256']:
        raise ValueError('Verified mounted Drive backup required')
    if not set(map(str, required)).issubset(r['files']):
        raise ValueError('Required files missing from backup')
    if cache and r['cache_manifest_sha256'] != digest(Path(cache)/'manifest.json'):
        raise ValueError('Preserved feature cache mismatch')
    verify_files(r['files'])
    return r


def verify_training_backup(receipt, cache):
    m = json.loads((Path(cache)/'manifest.json').read_text())
    required = {str(Path(cache)/'manifest.json')}
    required.update(str(Path(cache)/e['path']) for e in m['entries'])
    for p in m['config']['source_manifests']:
        required.add(p)
        source = json.loads(Path(p).read_text())
        required.update(str(Path(p).parent / name) for name in source['artifacts'])
    return verify_snapshot(receipt, required, cache)
