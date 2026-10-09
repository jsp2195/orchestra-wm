"""Fabricated acquisition fixtures only; none are authentic traffic evidence."""
import copy
import hashlib
import io
import json
from pathlib import Path
import struct
import tarfile
import urllib.error
import zipfile

import pytest

from orchestra_wm.i24_msd import DOI
from orchestra_wm.i24_msd import acquisition as a
from orchestra_wm.i24_msd.adapter import (I24MSDAdapter, SchemaUnverified, crc32c,
                                         masked_crc, inspect_file, tfrecords)


def metadata(content=b'UNIT-TEST ONLY; NOT REAL TRAFFIC', name='unit-test-only.bin'):
    return {'status': 'OK', 'data': {'id': 999, 'authority': '10.7910', 'identifier': 'DVN/DQOWQI',
        'latestVersion': {'versionState': 'RELEASED', 'versionNumber': 1, 'versionMinorNumber': 0,
        'files': [{'restricted': False, 'description': 'Fabricated unit test, not release metadata',
            'dataFile': {'id': 123, 'filename': name, 'filesize': len(content),
            'contentType': 'application/octet-stream',
            'checksum': {'type': 'MD5', 'value': hashlib.md5(content).hexdigest()}}}]}}}


class Response(io.BytesIO):
    def __init__(self, data, status=200, mime='application/octet-stream', url='https://dataverse.harvard.edu/api/access/datafile/123'):
        super().__init__(data)
        self.status = status
        self.headers = {'Content-Type': mime, 'Content-Length': str(len(data))}
        self.url = url

    def geturl(self):
        return self.url


def test_metadata_identity_schema_and_no_invented_splits():
    m = a.metadata_manifest(metadata())
    assert m['doi'] == DOI and m['files'][0]['split'] is None
    assert m['status'] == 'METADATA_VISIBLE' and m['total_downloaded_bytes'] == 0
    wrong = metadata(); wrong['data']['identifier'] = 'OTHER'
    with pytest.raises(a.AcquisitionError, match='Wrong dataset'):
        a.metadata_manifest(wrong)
    draft = metadata(); draft['data']['latestVersion']['versionState'] = 'DRAFT'
    with pytest.raises(a.AcquisitionError, match='RELEASED'):
        a.metadata_manifest(draft)
    with pytest.raises(a.AcquisitionError, match='Unsafe'):
        a.metadata_manifest(metadata(name='../escape'))


def test_public_authentication_distinction_and_retry(monkeypatch):
    monkeypatch.setattr(a.urllib.request, 'urlopen', lambda *args, **kwargs: Response(b'x', url='https://dataverse.harvard.edu/loginpage.xhtml'))
    with pytest.raises(a.AcquisitionError, match='authentication'):
        a.open_public('https://dataverse.harvard.edu/api/access/datafile/123')
    calls = []
    def denied(*args, **kwargs):
        calls.append(1)
        raise urllib.error.HTTPError('https://dataverse.harvard.edu', 403, 'Forbidden', {}, None)
    monkeypatch.setattr(a.urllib.request, 'urlopen', denied)
    with pytest.raises(a.AcquisitionError, match='403'):
        a.open_public('https://dataverse.harvard.edu')
    assert len(calls) == 1
    calls.clear()
    def transient(*args, **kwargs):
        calls.append(1)
        if len(calls) < 3:
            raise urllib.error.HTTPError('https://dataverse.harvard.edu', 429, 'Busy', {'Retry-After': '0'}, None)
        return Response(b'ok')
    monkeypatch.setattr(a.urllib.request, 'urlopen', transient)
    assert a.open_public('https://dataverse.harvard.edu').read() == b'ok'
    assert len(calls) == 3


def test_atomic_download_checksums_resume_and_changed_file(monkeypatch, tmp_path):
    content = b'UNIT-TEST ONLY; NOT REAL TRAFFIC'
    m = a.metadata_manifest(metadata(content))
    monkeypatch.setattr(a, 'open_public', lambda url: Response(content))
    a.acquire(m, tmp_path, [123], 1000, 2000)
    file = Path(m['files'][0]['local_path'])
    assert file.read_bytes() == content and not list(tmp_path.glob('*.part'))
    assert m['status'] == 'BYTES_VERIFIED_NOT_PARSED' and m['real_scenarios_parsed'] == 0
    monkeypatch.setattr(a, 'open_public', lambda url: pytest.fail('Resume must verify local file'))
    a.acquire(m, tmp_path, [123], 1000, 2000)
    file.write_bytes(b'x' * len(content))
    with pytest.raises(a.AcquisitionError, match='checksum mismatch'):
        a.acquire(m, tmp_path, [123], 1000, 2000)


@pytest.mark.parametrize('response,reason', [
    (lambda: Response(b'<html>login</html>', mime='text/html'), 'Content-Length|HTML'),
    (lambda: Response(b'too short'), 'Content-Length'),
    (lambda: Response(b'x' * 32), 'checksum'),
    (lambda: Response(b'x' * 32, status=206), 'complete HTTP 200'),
])
def test_bad_downloads_never_become_completed_files(monkeypatch, tmp_path, response, reason):
    m = a.metadata_manifest(metadata())
    monkeypatch.setattr(a, 'open_public', lambda url: response())
    with pytest.raises(a.AcquisitionError, match=reason):
        a.acquire(m, tmp_path, [123], 1000, 2000)
    assert not list(tmp_path.iterdir()) and m['total_downloaded_bytes'] == 0


def test_budgets_ids_restrictions_and_import(tmp_path):
    content = b'UNIT-TEST ONLY; NOT REAL TRAFFIC'
    m = a.metadata_manifest(metadata(content))
    with pytest.raises(a.AcquisitionError, match='file IDs'):
        a.acquire(m, tmp_path/'raw', [999], 1000, 2000)
    with pytest.raises(a.AcquisitionError, match='max-download'):
        a.acquire(m, tmp_path/'raw', [123], 1, 2000)
    with pytest.raises(a.AcquisitionError, match='Disk budget'):
        a.acquire(m, tmp_path/'raw', [123], 1000, 1)
    restricted = copy.deepcopy(m); restricted['files'][0]['restricted'] = True
    with pytest.raises(a.AcquisitionError, match='Restricted'):
        a.acquire(restricted, tmp_path/'raw', [123], 1000, 2000)
    origin = tmp_path/'manual'; origin.mkdir(); (origin/'unit-test-only.bin').write_bytes(content)
    a.acquire(m, tmp_path/'raw', [123], 1000, 2000, origin)
    assert m['files'][0]['acquisition'] == 'manual_import'
    assert m['files'][0]['hashes']['SHA-256'] == hashlib.sha256(content).hexdigest()


def test_archive_integrity_budget_traversal_and_links(tmp_path):
    good = tmp_path/'good.zip'
    with zipfile.ZipFile(good, 'w') as z:
        z.writestr('sub/test.bin', b'UNIT TEST')
    rows = a.safe_extract(good, tmp_path/'extracted', 10000)
    assert rows[0]['bytes'] == 9 and (tmp_path/'extracted/sub/test.bin').read_bytes() == b'UNIT TEST'
    with pytest.raises(a.AcquisitionError, match='already exists'):
        a.safe_extract(good, tmp_path/'extracted', 10000)
    with pytest.raises(a.AcquisitionError, match='budget'):
        a.safe_extract(good, tmp_path/'too-big', 1)
    for index, name in enumerate(['../escape', '/absolute', 'a/../../escape', 'a\\..\\escape']):
        bad = tmp_path/f'bad{index}.zip'
        with zipfile.ZipFile(bad, 'w') as z:
            z.writestr(name, b'bad')
        with pytest.raises(a.AcquisitionError, match='Unsafe'):
            a.safe_extract(bad, tmp_path/f'bad{index}', 100000)
    linked = tmp_path/'link.tar'
    with tarfile.open(linked, 'w') as t:
        member = tarfile.TarInfo('link'); member.type = tarfile.SYMTYPE; member.linkname = '/etc/passwd'; t.addfile(member)
    with pytest.raises(a.AcquisitionError, match='Unsafe'):
        a.safe_extract(linked, tmp_path/'link', 100000)
    assert not (tmp_path/'escape').exists()


def tfrecord(payload):
    size = struct.pack('<Q', len(payload))
    return size + struct.pack('<I', masked_crc(size)) + payload + struct.pack('<I', masked_crc(payload))


def test_tfrecord_crc_and_opaque_proto_type(tmp_path):
    assert crc32c(b'123456789') == 0xE3069283
    p = tmp_path/'unit.tfrecord'; p.write_bytes(tfrecord(b'\x0a\x03abc') + tfrecord(b'\x08\x01'))
    assert list(tfrecords(p)) == [b'\x0a\x03abc', b'\x08\x01']
    report = inspect_file(p)
    assert report['container'] == 'TFRECORD' and report['records_crc_verified'] == 2
    assert report['real_scenarios_parsed'] == 0
    assert all(m['wire']['message_type'] == 'UNKNOWN' for m in report['messages'])
    damaged = bytearray(p.read_bytes()); damaged[13] ^= 1; p.write_bytes(damaged)
    with pytest.raises(a.AcquisitionError, match='CRC32C'):
        list(tfrecords(p))
    p.write_bytes(tfrecord(b'test')[:-1])
    with pytest.raises(a.AcquisitionError, match='Truncated'):
        list(tfrecords(p))
    p.write_bytes(tfrecord(b'test'))
    with pytest.raises(a.AcquisitionError, match='budget'):
        list(tfrecords(p, max_record_bytes=1))


def test_unknown_schema_fixture_never_masquerades_as_real_adapter(tmp_path):
    content = b'[{"scenario_id":"SYNTHETIC UNIT TEST","agents":[1,2]}]'
    m = a.metadata_manifest(metadata(content, 'unit-test-only.json'))
    (tmp_path/'unit-test-only.json').write_bytes(content)
    a.acquire(m, tmp_path/'raw', [123], 1000, 2000, tmp_path)
    adapter = I24MSDAdapter(m)
    report = adapter.inspect()[0]
    assert report['container'] == 'JSON' and report['real_scenarios_parsed'] == 0
    with pytest.raises(SchemaUnverified, match='semantic schema remains unverified'):
        adapter.scenes()
    with pytest.raises(a.AcquisitionError, match='different source'):
        I24MSDAdapter({'doi': 'CONTINUOUS_I24'})
    with pytest.raises(a.AcquisitionError, match='No checksum-verified'):
        I24MSDAdapter({'doi': DOI, 'files': []}).inspect()


def test_acquisition_pipeline_and_pilot_gate(tmp_path, monkeypatch):
    from orchestra_wm.i24_msd import pipeline as p
    import yaml
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(p, 'preserve', lambda: 1)
    monkeypatch.setattr(p, 'resources', lambda root: {'test_fixture': True})
    content = b'UNIT-TEST ONLY; NOT REAL TRAFFIC'
    meta = tmp_path/'metadata.json'; meta.write_text(json.dumps(metadata(content)))
    cfg = tmp_path/'config.yaml'; cfg.write_text(yaml.safe_dump({'output': 'outputs/i24_msd/test', 'max_download_bytes': 1000, 'max_disk_bytes': 2000}))
    local = tmp_path/'manual'; local.mkdir(); (local/'unit-test-only.bin').write_bytes(content)
    m = p.run(cfg, stage='acquire', metadata_json=meta, import_dir=local, file_ids=[123])
    assert m['status'] == 'BYTES_VERIFIED_NOT_PARSED'
    with pytest.raises(a.AcquisitionError, match='verified pilot uses'):
        p.run(cfg, stage='pilot', metadata_json=meta)
    blocked = json.loads(Path('outputs/i24_msd/test/BLOCKED.json').read_text())
    assert blocked['gradient_steps'] == 0 and blocked['checkpoint'] is None and blocked['demo'] is None
    assert not list(Path('outputs').rglob('*.pt'))
    cfg.write_text(yaml.safe_dump({'output': 'outputs/i24_msd/research', 'max_download_bytes': 1000, 'max_disk_bytes': 2000, 'research_gate_required': True}))
    with pytest.raises(a.AcquisitionError, match='Research remains gated'):
        p.run(cfg)


def test_boundary_metric_uses_only_real_polyline_support():
    import numpy as np
    from orchestra_wm.i24_msd.evaluation import boundary_metrics
    p=np.zeros((1,1,2,8));p[0,0,:,0]=[5,50];p[0,0,:,1]=[0,20];p[:,:,:,6:]=[4,2]
    source={'travel_sign':1,'x_origin_m':0,'source_map':[
        {'type':'road_edge','polyline_xyz_m':[[0,-2,0],[10,-2,0]]},
        {'type':'road_edge','polyline_xyz_m':[[0,2,0],[10,2,0]]}]}
    result=boundary_metrics(p,np.ones((1,2),bool),source)
    assert result['road_boundary_coverage']==.5 and result['road_boundary_violation']==0
    p[0,0,0,1]=2
    assert boundary_metrics(p,np.ones((1,2),bool),source)['road_boundary_violation']==1
