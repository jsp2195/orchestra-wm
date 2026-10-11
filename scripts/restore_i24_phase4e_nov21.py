"""Restore only the historical Nov21 nested ZIP; retain all complete/partial bytes."""
import argparse,hashlib,json,os,shutil,time,zipfile
from pathlib import Path
from orchestra_wm.i24_phase4e.ranges import DriveRanges
from orchestra_wm.i24_continuous.source import write_immutable
p=argparse.ArgumentParser();p.add_argument('--metadata',required=True);a=p.parse_args()
meta=json.loads(Path(a.metadata).read_text())
if meta['name']!='11-21-2022.zip' or int(meta['size'])!=6248968002:raise ValueError('Wrong original archive metadata')
dest=Path('data/i24_continuous/raw/637b023440527bf2daa5932f__post1.zip');dest.parent.mkdir(parents=True,exist_ok=True)
expected='dbabe7b3e665a7d48b27d89d9f55fa0190e2cbcd7f9953beae59a8663b6ec3b7'
if dest.exists():
    with dest.open('rb') as handle:actual=hashlib.file_digest(handle,'sha256').hexdigest()
    if actual!=expected:raise ValueError('Existing archive differs; preserved')
    raise SystemExit('Existing historical archive SHA256 verified; no download')
partial=dest.with_suffix('.zip.part');offset=partial.stat().st_size if partial.exists() else 0
write_immutable(dest.with_suffix('.restore.json'),dict(source_metadata_sha256=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest(),expected_sha256=expected))
remote=DriveRanges(meta['id'],meta['size'],block_bytes=64*1024*1024)
with zipfile.ZipFile(remote) as archive:
    member=next(x for x in archive.infolist() if x.filename.endswith('/'+dest.name))
    if offset>member.file_size or shutil.disk_usage(dest.parent).free<member.file_size-offset+10_000_000_000:
        raise ValueError('Archive size or preservation reserve check failed')
    digest=hashlib.sha256();count=0;last=time.monotonic()
    with archive.open(member) as source,partial.open('a+b') as output:
        output.seek(0)
        while data:=source.read(8*1024*1024):
            prefix=min(len(data),max(0,offset-count))
            if prefix and output.read(prefix)!=data[:prefix]:raise ValueError('Existing partial prefix differs; preserved')
            remaining=data[prefix:]
            if remaining:
                if shutil.disk_usage(dest.parent).free<len(remaining)+10_000_000_000:raise ValueError('Disk reserve reached')
                output.write(remaining)
            digest.update(data);count+=len(data)
            if time.monotonic()-last>20:print('Restored nested archive bytes',count,'of',member.file_size,flush=True);last=time.monotonic()
        output.flush();os.fsync(output.fileno())
    if count!=member.file_size or digest.hexdigest()!=expected:raise ValueError('Historical nested ZIP size/SHA256 mismatch')
partial.replace(dest)
print('Historical nested archive SHA256 VERIFIED',count,expected,flush=True)
