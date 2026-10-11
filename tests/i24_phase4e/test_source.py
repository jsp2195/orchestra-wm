import io
import json
import zipfile
import pytest
from orchestra_wm.i24_phase4e.ranges import DriveRanges, SourceBlocked
from orchestra_wm.i24_phase4e.source import nested_records


class Response(io.BytesIO):
    def __init__(self,data,status,headers):super().__init__(data);self.status=status;self.headers=headers


class Server:
    def __init__(self,data,mode='valid'):self.data=data;self.mode=mode
    def open(self,request,timeout):
        start,end=map(int,request.get_header('Range').split('=')[1].split('-'))
        data=self.data[start:end+1]
        if self.mode=='short':data=data[:-1]
        headers={'Content-Range':f'bytes {start}-{end}/{len(self.data)}'}
        if self.mode=='wrong':headers['Content-Range']='bytes 0-1/2'
        return Response(data,200 if self.mode=='ignored' else 206,headers)


def test_ranges_random_seek_and_zip(monkeypatch):
    monkeypatch.setenv('GOOGLE_DRIVE_ACCESS_TOKEN','test-only')
    raw=io.BytesIO()
    with zipfile.ZipFile(raw,'w',zipfile.ZIP_DEFLATED) as z:z.writestr('records.json',json.dumps(list(range(1000))))
    data=raw.getvalue();remote=DriveRanges('fixture',len(data),opener=Server(data),block_bytes=17)
    with zipfile.ZipFile(remote) as z:assert json.loads(z.read('records.json'))==list(range(1000))
    for start,length in [(0,7),(41,91),(len(data)-5,8),(2,0),(100,33)]:
        remote.seek(start);assert remote.read(length)==data[start:start+length]


@pytest.mark.parametrize('mode',['short','wrong','ignored'])
def test_invalid_ranges_fail_closed(monkeypatch,mode):
    monkeypatch.setenv('GOOGLE_DRIVE_ACCESS_TOKEN','test-only')
    remote=DriveRanges('fixture',100,opener=Server(b'x'*100,mode))
    with pytest.raises(SourceBlocked):remote.read(10)


def test_nested_stream_matches_standard_zip_and_checks_crc():
    inner=io.BytesIO();records=[{'timestamp':[1,2,3],'_id':'v'}]
    with zipfile.ZipFile(inner,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('session.json',json.dumps(records));z.writestr('notes.txt','some notes')
    outer=io.BytesIO()
    with zipfile.ZipFile(outer,'w',zipfile.ZIP_DEFLATED) as z:z.writestr('nested.zip',inner.getvalue())
    outer.seek(0);audit={}
    with zipfile.ZipFile(outer) as z:
        with z.open('nested.zip') as f:assert list(nested_records(f,'session',audit))==records
    import hashlib
    assert audit['nested_zip_sha256']==hashlib.sha256(inner.getvalue()).hexdigest()
    assert audit['nested_zip_bytes']==len(inner.getvalue())
    assert len(audit['members'])==2
    corrupt=bytearray(inner.getvalue());corrupt[14]^=1
    with pytest.raises(Exception):list(nested_records(io.BytesIO(corrupt),'session',{}))


def test_zip64_data_descriptor_unknown_header_size():
    class Nonseekable(io.BytesIO):
        def seek(self,*args):raise io.UnsupportedOperation('stream')
    raw=Nonseekable();records=[{'timestamp':[1,2,3],'_id':'v'}]
    with zipfile.ZipFile(raw,'w',zipfile.ZIP_DEFLATED) as z:
        with z.open('session.json','w',force_zip64=True) as f:f.write(json.dumps(records).encode())
    payload=raw.getvalue();audit={}
    assert list(nested_records(io.BytesIO(payload),'session',audit))==records
    assert audit['nested_zip_bytes']==len(payload)
    import hashlib
    assert audit['nested_zip_sha256']==hashlib.sha256(payload).hexdigest()
    assert audit['members'][0]['crc_verified']


def test_date_roles_timezone_geometry_and_native_conversion(tmp_path,monkeypatch):
    import numpy as np
    from orchestra_wm.i24_phase4e import data
    from orchestra_wm.i24.data import load_scene
    metadata=dict(name='11-22-2022.zip',size='123',md5Checksum='0'*32,version='1')
    cfg=data.configuration('2022-11-22',metadata)
    assert cfg['blocks'][0]['start']==1669119300
    assert [b['seconds'] for b in cfg['blocks']]==[600]*4
    assert {b['split'] for b in cfg['blocks']}=={'train'}
    assert {b['split'] for b in data.configuration('2022-11-25',metadata)['blocks']}=={'validation'}
    assert {b['split'] for b in data.configuration('2022-11-28',metadata)['blocks']}=={'test'}
    start=cfg['blocks'][0]['start']
    def records(meta,date,audit):
        t=start+2+np.arange(2950)*.04
        for i,y in enumerate([18,30,42]):
            yield dict(_id=str(i),timestamp=t.tolist(),x_position=(320070-np.arange(len(t))*.8).tolist(),
                       y_position=[y]*len(t),length=15,width=6,direction=-1)
        audit.update(nested_zip_sha256='1'*64,nested_members_crc_verified=True,outer_selected_member_crc_verified=True)
    monkeypatch.setattr(data,'remote_records',records)
    from types import SimpleNamespace
    monkeypatch.setattr(data.shutil,'disk_usage',lambda _:SimpleNamespace(free=20_000_000_000))
    result=data.prepare('2022-11-22',metadata,tmp_path)
    assert result['scene_counts']['train']>0
    assert set(s['split'] for s in result['scenes'])=={'train'}
    scene=load_scene(tmp_path/result['scenes'][0]['path'])
    assert scene.provenance['recording_date']=='2022-11-22'
    assert scene.road.bins==8 and scene.road.field_count==32
    assert np.allclose(scene.values[:,:,2][scene.existence],6.096,atol=.001)
    assert len([e for e in result['excluded_windows'] if e.get('reason')=='no retained reconstruction in frozen slot'])==3
