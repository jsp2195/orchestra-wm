"""Preserve existing experiment bytes in private, bounded transfer parts.

No datasets or predictions are regenerated. Existing packages fail closed.
Upload and authenticated readback receipts are separate from this local inventory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
p.add_argument('roots', nargs='+')
a = p.parse_args()
out = Path(a.output)
out.mkdir(parents=True, exist_ok=False)
archive = out/'recovery.tar.gz'
inventory = []
with tarfile.open(archive, 'w:gz') as tar:
    for root in map(Path, a.roots):
        assert root.exists() and not root.is_absolute() and '..' not in root.parts
        files = [root] if root.is_file() else sorted(root.rglob('*'))
        for f in files:
            if not f.is_file():
                continue
            data = f.read_bytes()
            inventory.append(dict(path=str(f), bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
            tar.add(f, arcname=str(f), recursive=False)
parts = []
digest = hashlib.sha256()
offset = 0
with archive.open('rb') as stream:
    while data := stream.read(24*1024*1024):
        digest.update(data)
        part = out/f'recovery.tar.gz.part{len(parts):03d}'
        part.write_bytes(data)
        parts.append(dict(filename=part.name, bytes=len(data), start=offset,
                          end_inclusive=offset+len(data)-1, sha256=hashlib.sha256(data).hexdigest()))
        offset += len(data)
manifest = dict(archive=archive.name, bytes=offset, sha256=digest.hexdigest(),
                parts=parts, files=inventory, verification='LOCAL_ONLY_UNTIL_AUTHENTICATED_REMOTE_READBACK',
                reassembly='Concatenate part000, part001, ... in numeric order, then verify full SHA256 before extraction.')
(out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
print(json.dumps(dict(directory=str(out), bytes=offset, sha256=digest.hexdigest(), parts=len(parts), files=len(inventory))))
