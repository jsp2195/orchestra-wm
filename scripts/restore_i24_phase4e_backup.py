"""Verify connector-downloaded parts and restore to a new, isolated directory."""
import argparse, hashlib, json, shutil, tarfile
from pathlib import Path, PurePosixPath

p = argparse.ArgumentParser()
p.add_argument('--manifest', required=True)
p.add_argument('--parts', required=True)
p.add_argument('--output', required=True)
p.add_argument('--part-prefix', default='ORCHESTRA_PHASE4E_incomplete_ingestion',
               help='Local ordered part filename prefix; hashes remain authoritative')
a = p.parse_args()
m = json.loads(Path(a.manifest).read_text())
assert m['status'] == 'COMPLETE_AUTHENTICATED_READBACK_RECONSTRUCTION_SHA256_VERIFIED'
out = Path(a.output)
if out.exists():
    raise SystemExit('Output already exists; preserve it and select a new directory')
out.mkdir(parents=True)
archive = out / 'partial_ingestion.tar.gz'
combined = hashlib.sha256()
with archive.open('xb') as dst:
    for i, receipt in enumerate(m['parts']):
        assert receipt['index'] == i
        src = Path(a.parts) / f'{a.part_prefix}.part{i:03}'
        assert src.stat().st_size == receipt['remote_bytes']
        with src.open('rb') as handle:
            assert hashlib.file_digest(handle, 'sha256').hexdigest() == receipt['sha256']
        with src.open('rb') as handle:
            while block := handle.read(1024 * 1024):
                combined.update(block)
                dst.write(block)
assert archive.stat().st_size == m['archive_bytes']
assert combined.hexdigest() == m['archive_sha256']
staging = out / 'restored'
with tarfile.open(archive, 'r:gz') as tar:
    members = tar.getmembers()
    assert len({x.name for x in members}) == len(members)
    assert {x.name for x in members} == set(m['files'])
    for member in members:
        path = PurePosixPath(member.name)
        assert member.isfile() and not path.is_absolute() and '..' not in path.parts
        expected = m['files'][member.name]
        assert member.size == expected['bytes']
        target = staging / member.name
        target.parent.mkdir(parents=True, exist_ok=True)
        with tar.extractfile(member) as source, target.open('xb') as dest:
            shutil.copyfileobj(source, dest)
        with target.open('rb') as handle:
            assert hashlib.file_digest(handle, 'sha256').hexdigest() == expected['sha256']
print('Verified and restored artifacts to', staging)
print('Restoration alone does not establish training eligibility; verify source/QC manifests.')
