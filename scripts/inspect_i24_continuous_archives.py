"""Verify local Drive archives and retain only the nested trajectory ZIP."""
import argparse
import json
from pathlib import Path
import shutil
import zipfile
import zlib

from orchestra_wm.i24_continuous.drive import FILES, TransferBlocked, hashes, verify_zip_crc
from orchestra_wm.i24_continuous.source import write_immutable


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',default='data/i24_continuous/raw')
    a=p.parse_args();root=Path(a.data_root)
    manifest=json.loads((root/'DRIVE_ARCHIVE_MANIFEST.json').read_text())
    verified=[]
    for expected in FILES:
        entries=[m for m in manifest if m['id']==expected[0]]
        if len(entries)!=1:raise TransferBlocked('Archive missing from Drive verification manifest')
        entry=entries[0];archive=root/expected[1]
        sha256,md5=hashes(archive)
        if archive.stat().st_size!=expected[2] or md5!=entry['md5'] or sha256!=entry['sha256']:
            raise TransferBlocked('Local archive integrity differs from verified Drive source')
        print(f'{archive.name}: verifying every ZIP member CRC without extraction',flush=True)
        members=verify_zip_crc(archive)
        verified.append(dict(name=archive.name,sha256=sha256,md5=md5,bytes=archive.stat().st_size,
                             zip_crc_verified=True,members=members))
    outer=root/'11-21-2022.zip'
    member='11-21-2022/637b023440527bf2daa5932f__post1.zip'
    dest=root/Path(member).name
    with zipfile.ZipFile(outer) as z:
        info=z.getinfo(member)
        if dest.exists():
            if dest.is_symlink() or dest.stat().st_size!=info.file_size:
                raise TransferBlocked('Existing nested archive differs; preserved')
            crc=0
            with dest.open('rb') as handle:
                for chunk in iter(lambda:handle.read(1024*1024),b''):crc=zlib.crc32(chunk,crc)
            if crc!=info.CRC:raise TransferBlocked('Existing nested archive CRC differs; preserved')
        else:
            if shutil.disk_usage(root).free-info.file_size<10_000_000_000:
                raise TransferBlocked('Nested ZIP extraction would violate disk reserve')
            partial=dest.with_name(dest.name+'.extract.part')
            if partial.exists():raise TransferBlocked('Extraction partial preserved; separate destination required')
            with z.open(member) as source,partial.open('xb') as target:
                shutil.copyfileobj(source,target,1024*1024)
            if partial.stat().st_size!=info.file_size:raise TransferBlocked('Nested ZIP size mismatch')
            partial.replace(dest)
    nested_sha,nested_md5=hashes(dest)
    with zipfile.ZipFile(dest) as z:
        members=[dict(name=i.filename,bytes=i.file_size,compressed_bytes=i.compress_size,
                      crc32=f'{i.CRC:08x}') for i in z.infolist()]
    write_immutable(root/'VERIFIED_ARCHIVE_INVENTORY.json',dict(
        archives=verified,nested=dict(name=dest.name,bytes=dest.stat().st_size,sha256=nested_sha,
                                     md5=nested_md5,outer_member_crc_verified=True,members=members),
        nested_uncompressed_bytes=sum(i['bytes'] for i in members),
        full_extraction='FORBIDDEN_WITH_CURRENT_DISK_RESERVE; use streaming selection',
        persistence='Present in this running workspace; ignored raw files are not in Git and persistence across fresh tasks is not guaranteed'))
    print('Archives and nested ZIP verified; full JSON remains compressed')
    return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except TransferBlocked as e:print('BLOCKED:',str(e));raise SystemExit(2)
