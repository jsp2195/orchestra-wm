"""Inventory and hash existing work; build a local recovery bundle, never upload."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import torch


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    root = Path.cwd()
    output = root/'outputs/i24_phase4b/audit'
    output.mkdir(parents=True, exist_ok=True)
    manifest = output/'preservation.json'
    if manifest.exists():
        data = json.loads(manifest.read_text())
        for p, item in data['files'].items():
            if p in ('.gitignore','README.md','AGENTS.md'):
                assert hashlib.sha256((root/p).read_bytes()[:item['bytes']]).hexdigest()==item['sha256'],p
            else:
                assert digest(root/p) == item['sha256'], p
        print('Verified preserved files', len(data['files']), flush=True)
        return
    paths = {Path(p) for p in subprocess.check_output(['git', 'ls-files'], text=True).splitlines()}
    for directory in ['data/i24_continuous', 'outputs/i24_continuous/pilot_0700_wb']:
        paths.update(p for p in Path(directory).rglob('*') if p.is_file())
    files = {}
    for p in sorted(paths):
        files[str(p)] = dict(bytes=p.stat().st_size, sha256=digest(p))
    pilot = Path('outputs/i24_continuous/pilot_0700_wb')
    checkpoints = json.loads((pilot/'training_metadata.json').read_text())
    for row in checkpoints:
        path = pilot/'checkpoints'/f'{row["variant"]}-101.pt'
        assert files[str(path)]['sha256'] == row['checkpoint_sha256']
    data = dict(starting_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                files=files, checkpoint_hashes_verified=6, device='cpu', cuda_available=torch.cuda.is_available(),
                free_bytes=shutil.disk_usage(root).free, persistent_mount_verified=False,
                durability='Existing overlay survived this task. Future-task persistence is not guaranteed.',
                remote_backup='NOT_CONFIGURED; Drive token is read-only; no upload attempted')
    bundle = Path('work/phase4b/original_pilot_recovery.tar.gz')
    bundle.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle, 'x:gz', compresslevel=1) as tar:
        tar.add(pilot, arcname=str(pilot))
    with tarfile.open(bundle) as tar:
        for member in tar:
            if member.isfile():
                h=hashlib.sha256()
                with tar.extractfile(member) as f:
                    for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
                assert h.hexdigest()==files[member.name]['sha256'], member.name
    data['local_recovery_bundle'] = dict(path=str(bundle), bytes=bundle.stat().st_size, sha256=digest(bundle), all_members_verified=True,
                                         scope='Canonical/native scenes, checkpoints, optimizer/RNG, evaluated arrays, manifests, reports, demo; original raw archives retained separately')
    manifest.write_text(json.dumps(data, indent=2)+'\n')
    print(json.dumps({k:v for k,v in data.items() if k!='files'}, indent=2), flush=True)


if __name__ == '__main__':main()
