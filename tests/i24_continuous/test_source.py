"""Bounded streaming extraction contracts; unit records are not scientific data."""
import json
import zipfile
import pytest
from orchestra_wm.i24_continuous.source import extract_interval
from orchestra_wm.i24_continuous.drive import TransferBlocked


def input_zip(tmp_path):
    p=tmp_path/'unit.zip'
    records=[]
    for ident,direction in [('wb',-1),('eb',1)]:
        records.append(dict(_id=ident,direction=direction,timestamp=[-3,-2,-1,0,1,2,601,602],
                            x_position=[105]*8,y_position=[18]*8,length=15,width=6))
    with zipfile.ZipFile(p,'w',compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('unit.json',json.dumps(records))
    return p


def test_selected_source_keeps_ids_and_exact_samples(tmp_path):
    archive=input_zip(tmp_path);dest=tmp_path/'selected'
    result=extract_interval(archive,dest,start=0,seconds=600,x_min=100,x_max=110,min_free=0)
    selected=json.loads((dest/'westbound_selected.json').read_text())
    assert len(selected)==1 and selected[0]['_id']=='wb'
    assert selected[0]['timestamp']==[-2,-1,0,1,2,601]
    assert result['selected_samples']==6 and result['nested_member_crc_verified']
    assert result['counts']['direction_1_tracks']==1
    assert result['min_tracks_per_5hz_frame']==0
    assert extract_interval(archive,dest,start=0,seconds=600,x_min=100,x_max=110,min_free=0)==result
    (dest/'westbound_selected.json').write_text('[]')
    with pytest.raises(TransferBlocked,match='checksum'):
        extract_interval(archive,dest,start=0,seconds=600,x_min=100,x_max=110,min_free=0)


def test_existing_partial_and_extraction_budget_preserved(tmp_path):
    archive=input_zip(tmp_path);dest=tmp_path/'selected'
    with pytest.raises(TransferBlocked,match='budget'):
        extract_interval(archive,dest,start=0,seconds=600,x_min=100,x_max=110,max_bytes=10,min_free=0)
    partial=dest/'westbound_selected.json.part'
    before=partial.read_bytes()
    with pytest.raises(TransferBlocked,match='Uncommitted'):
        extract_interval(archive,dest,start=0,seconds=600,x_min=100,x_max=110,min_free=0)
    assert partial.read_bytes()==before
