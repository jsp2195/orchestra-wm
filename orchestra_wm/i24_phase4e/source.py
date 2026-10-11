"""Sequential nested-ZIP ingestion using zipfile and stream-unzip CRC validation."""
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

import ijson
from stream_unzip import stream_unzip, NO_ENCRYPTION
from .ranges import DriveRanges, SourceBlocked, NoRedirect

SESSIONS = {
    '2022-11-21':'637b023440527bf2daa5932f__post1',
    '2022-11-22':'637c399add50d54aa5af0cf4__post2',
    '2022-11-23':'637d8ea678f0cb97981425dd__post3',
    '2022-11-25':'6380728cdd50d54aa5af0cf5__post5',
    '2022-11-28':'638450a3dd50d54aa5af0cf6__post8',
    '2022-11-29':'63858a2cfb3ff533c12df166__post9',
}


class ChunkReader(io.RawIOBase):
    def __init__(self, chunks):
        self.chunks=iter(chunks);self.pending=b''
    def readable(self): return True
    def readinto(self, target):
        while not self.pending:
            try:self.pending=next(self.chunks)
            except StopIteration:return 0
        n=min(len(target),len(self.pending));target[:n]=self.pending[:n];self.pending=self.pending[n:]
        return n


def nested_records(handle, session, audit):
    digest=hashlib.sha256();total=0;found=0;members=[]
    def chunks():
        nonlocal total
        while b:=handle.read(1024*1024):
            digest.update(b);total+=len(b);yield b
    for name,size,parts in stream_unzip(chunks(),allowed_encryption_mechanisms=(NO_ENCRYPTION,)):
        name=name.decode('utf-8')
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise SourceBlocked('Unsafe nested member name')
        member_digest=hashlib.sha256();member_bytes=0
        def verified_chunks():
            nonlocal member_bytes
            for b in parts:member_digest.update(b);member_bytes+=len(b);yield b
        reader=io.BufferedReader(ChunkReader(verified_chunks()))
        if name==session+'.json':
            found+=1
            if found!=1:raise SourceBlocked('Duplicate trajectory member')
            yield from ijson.items(reader,'item',use_float=True)
        while reader.read(1024*1024):pass
        # For data-descriptor ZIPs the header size is unknown (None). Exhausting
        # parts already validates the descriptor's final size and CRC in stream-unzip.
        if size is not None and member_bytes!=size:raise SourceBlocked('Nested member size mismatch')
        members.append(dict(name=name,bytes=member_bytes,sha256=member_digest.hexdigest(),crc_verified=True))
    # stream-unzip may buffer the central directory; drain remaining outer bytes.
    while b:=handle.read(1024*1024):digest.update(b);total+=len(b)
    if found!=1:raise SourceBlocked('Expected trajectory member missing')
    audit.update(nested_zip_sha256=digest.hexdigest(),nested_zip_bytes=total,members=members,
                 nested_members_crc_verified=True,outer_selected_member_crc_verified=True)


def remote_records(metadata, date, audit):
    session=SESSIONS[date]
    local=Path('data/i24_continuous/raw')/(session+'.zip')
    if date=='2022-11-21' and local.exists():
        with local.open('rb') as handle:
            yield from nested_records(handle,session,audit)
        if audit['nested_zip_sha256']!='dbabe7b3e665a7d48b27d89d9f55fa0190e2cbcd7f9953beae59a8663b6ec3b7':
            raise SourceBlocked('Restored November21 archive disagrees with historical SHA256')
        audit.update(local_restored_archive=str(local),range_requests=0,transferred_bytes=0,
                     outer_selected_member_crc_verified=False,historical_nested_sha256_verified=True,
                     full_outer_archive_md5_verified=False,
                     integrity_scope='Local complete nested ZIP matched historical SHA256; inner JSON CRC and SHA256 verified')
        return
    remote=DriveRanges(metadata['id'],metadata['size'],block_bytes=64*1024*1024)
    with zipfile.ZipFile(remote) as archive:
        readmes=[x for x in archive.infolist() if 'README' in x.filename]
        selected=[x for x in archive.infolist() if x.filename.endswith('/'+session+'.zip')]
        if len(readmes)!=1 or len(selected)!=1:raise SourceBlocked('Unrecognized source archive layout')
        if readmes[0].file_size>100000:raise SourceBlocked('Unexpected README size')
        readme=archive.read(readmes[0]).decode('utf-8')
        if session not in readme or 'Data Use Agreement' not in readme:raise SourceBlocked('Source session documentation mismatch')
        audit['readme_sha256']=hashlib.sha256(readme.encode()).hexdigest()
        audit['selected_outer_member']=dict(name=selected[0].filename,bytes=selected[0].file_size,crc32=f'{selected[0].CRC:08x}')
        with archive.open(selected[0]) as handle:
            yield from nested_records(handle,session,audit)
    if date=='2022-11-21' and audit['nested_zip_sha256']!='dbabe7b3e665a7d48b27d89d9f55fa0190e2cbcd7f9953beae59a8663b6ec3b7':
        raise SourceBlocked('November21 nested archive disagrees with historical SHA256')
    audit.update(range_requests=remote.requests,transferred_bytes=remote.transferred,
                 full_outer_archive_md5_verified=False,
                 integrity_scope='Selected nested ZIP and JSON SHA256 plus both member CRCs; unselected image bytes not downloaded')
