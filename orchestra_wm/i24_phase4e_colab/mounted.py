"""Verified original ZIP reads from Colab's mounted Drive, with no network adapter."""
import hashlib
import json
import os
from pathlib import Path
import zipfile
from orchestra_wm.i24_phase4e.source import nested_records, SESSIONS
from .storage import atomic_json, digest


def index_drive(roots, exclude=()):
    """One filename inventory; content hashes resolve recovery parts without IDs."""
    excluded = {Path(p).resolve() for p in exclude}
    entries = [];seen_directories=set();seen_files=set()
    for root in roots:
        for directory, dirs, files in os.walk(root, followlinks=True):
            resolved=Path(directory).resolve()
            if resolved in excluded or resolved in seen_directories:
                dirs[:]=[]
                continue
            seen_directories.add(resolved)
            dirs[:] = sorted(d for d in dirs if (Path(directory)/d).resolve() not in excluded)
            for name in sorted(files):
                if name.endswith('.zip') or '.part' in name or name.endswith('RECOVERY_MANIFEST.json'):
                    path=Path(directory)/name
                    if path.resolve() not in seen_files:
                        entries.append(path);seen_files.add(path.resolve())
    return entries


def find_hashed(entries, name, expected, size=None):
    candidates = [p for p in entries if p.name == name] if name else list(entries)
    for path in candidates:
        if size is not None and path.stat().st_size != int(size):
            continue
        if digest(path) == expected:
            return path
    raise FileNotFoundError('Verified Drive backup file not found: ' + (name or 'part SHA256 ' + expected[:12]))


def source_path(entries, metadata):
    matches = [p for p in entries if p.name == metadata['name'] and p.stat().st_size == int(metadata['size'])]
    if len(matches) != 1:
        raise ValueError('Expected one original mounted archive: ' + metadata['name'] +
                         '. Narrow SOURCE_ROOT or add the source folder shortcut to MyDrive.')
    return matches[0]


def records(path, metadata, date, audit):
    """Check complete outer MD5 against original acquisition and both selected CRCs."""
    path = Path(path)
    before = path.stat()
    if before.st_size != int(metadata['size']):
        raise ValueError('Original archive size differs')
    print(date, 'verifying mounted original archive MD5', flush=True)
    if digest(path, 'md5') != metadata['md5Checksum']:
        raise ValueError('Original archive MD5 differs from acquired metadata')
    session = SESSIONS[date]
    with zipfile.ZipFile(path) as outer:
        readmes = [x for x in outer.infolist() if 'README' in x.filename]
        members = [x for x in outer.infolist() if x.filename.endswith('/'+session+'.zip')]
        if len(readmes) != 1 or readmes[0].file_size > 100000 or len(members) != 1:
            raise ValueError('Unrecognized original archive layout')
        text = outer.read(readmes[0]).decode('utf-8')
        if session not in text or 'Data Use Agreement' not in text:
            raise ValueError('Source session documentation mismatch')
        audit.update(readme_sha256=hashlib.sha256(text.encode()).hexdigest(),
                     selected_outer_member=dict(name=members[0].filename,bytes=members[0].file_size,
                                                crc32=f'{members[0].CRC:08x}'))
        with outer.open(members[0]) as handle:
            yield from nested_records(handle, session, audit)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Mounted source changed during ingestion')
    if date == '2022-11-21' and audit['nested_zip_sha256'] != 'dbabe7b3e665a7d48b27d89d9f55fa0190e2cbcd7f9953beae59a8663b6ec3b7':
        raise ValueError('Historical Nov21 nested ZIP differs')
    audit.update(full_outer_archive_md5_verified=True, source_transport='COLAB_MOUNTED_DRIVE',
                 mounted_stat_stable=True, source_adapter_sha256=digest(Path(__file__)),
                 integrity_scope='Original whole outer MD5; complete nested ZIP and JSON SHA256; both selected CRCs')


def verify_source(root):
    root = Path(root)
    native = json.loads((root/'native_manifest.json').read_text())
    if not all(native.get(k) is True for k in ['nested_members_crc_verified',
                                              'outer_selected_member_crc_verified',
                                              'full_outer_archive_md5_verified', 'mounted_stat_stable']):
        raise ValueError('Mounted ingestion integrity incomplete')
    selection = json.loads((root/'frozen_selection.json').read_text())
    result = dict(status='SOURCE_METADATA_STABLE_AND_SELECTED_STREAM_INTEGRITY_VERIFIED',
                  verification_method='MOUNTED_DRIVE_ORIGINAL_MD5_AND_STABLE_FILE_STAT',
                  metadata=selection['source_metadata'],native_manifest_sha256=digest(root/'native_manifest.json'),
                  nested_zip_sha256=native['nested_zip_sha256'],whole_outer_md5_verified=True)
    from orchestra_wm.i24_continuous.source import write_immutable
    write_immutable(root/'source_verification.json', result)
