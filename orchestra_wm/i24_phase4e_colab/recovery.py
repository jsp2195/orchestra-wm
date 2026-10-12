"""Restore the existing verified Nov21/22 backups; never reingest those dates."""
import hashlib
import json
import os
from pathlib import Path
import uuid
from .storage import digest, extract_verified, verify_files, atomic_json, flush_file, require_space
from .mounted import find_hashed

BASE = 'outputs/i24_phase4e'


def recover_date(date, entries, backup_status, staging):
    spec = backup_status
    manifest_path = find_hashed(entries, spec['manifest_name'], spec['manifest_sha256'],spec.get('manifest_bytes'))
    manifest = json.loads(manifest_path.read_text())
    if manifest['archive_sha256'] != spec['archive_sha256'] or manifest['archive_bytes'] != spec['archive_bytes']:
        raise ValueError('Original recovery archive identity differs')
    expected_prefix = f'{BASE}/data_v4/{date}/'
    if not any(name.startswith(expected_prefix+'canonical/') for name in manifest['files']):
        raise ValueError('Completed-date artifacts absent from recovery manifest')
    staging = Path(staging)
    staging.mkdir(parents=True, exist_ok=True)
    receipt = staging/(date+'-restored.json')
    try:
        verify_files(manifest['files'])
    except (FileNotFoundError, ValueError):
        archive = staging/(date+'-'+spec['archive_sha256']+'.tar.gz')
        if archive.exists():
            if digest(archive) != spec['archive_sha256']:
                raise ValueError('Existing recovery archive differs; preserved')
        else:
            require_space(staging,spec['archive_bytes'])
            temp = archive.with_name(archive.name+'.'+uuid.uuid4().hex+'.pending')
            combined = hashlib.sha256()
            with temp.open('xb') as dest:
                for i, part in enumerate(manifest['parts']):
                    if part.get('index',i) != i:
                        raise ValueError('Recovery part order differs')
                    size = part.get('bytes',part.get('remote_bytes'))
                    candidates = [p for p in entries if p.name.endswith(f'.part{i:03}')]
                    source = find_hashed(candidates,None,part['sha256'],size)
                    print('Restoring',date,'verified part',i+1,'of',len(manifest['parts']),flush=True)
                    with source.open('rb') as handle:
                        while block := handle.read(1024*1024):
                            dest.write(block);combined.update(block)
                flush_file(dest)
            if temp.stat().st_size != spec['archive_bytes'] or combined.hexdigest() != spec['archive_sha256']:
                raise ValueError('Reconstructed backup checksum differs')
            os.replace(temp,archive)
        extract_verified(archive,manifest['files'],Path('.'))
    root = Path(BASE)/'data_v4'/date
    data = json.loads((root/'data_manifest.json').read_text())
    verification = json.loads((root/'source_verification.json').read_text())
    if verification['native_manifest_sha256'] != digest(root/'native_manifest.json'):
        raise ValueError('Recovered native/source verification mismatch')
    if data['source_dates'] != [date] or data['scene_counts'] != {'train':92}:
        raise ValueError('Recovered completed date differs from frozen accepted dataset')
    atomic_json(receipt,dict(status='EXISTING_COMPLETED_DATE_REUSED',date=date,
                            recovery_manifest_sha256=spec['manifest_sha256'],archive_sha256=spec['archive_sha256'],
                            data_manifest_sha256=digest(root/'data_manifest.json')))
    return root
