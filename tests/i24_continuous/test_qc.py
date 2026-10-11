import hashlib
import json
import numpy as np
from orchestra_wm.i24_continuous.qc import quality_control


def test_qc_reports_rejects_subsecond_support_and_transform(tmp_path):
    t=np.arange(-2,602,.04).tolist()
    records=[dict(_id=ident,timestamp=t,x_position=[320080-3*x for x in t],
                  y_position=[18]*len(t),length=15,width=6,direction=-1,flags=['UNIT_FIXTURE'])
             for ident in ['a','b']]
    records.append({**records[0],'_id':'bad-dimensions','width':100})
    payload=json.dumps(records).encode();(tmp_path/'westbound_selected.json').write_bytes(payload)
    meta=dict(dataset='I-24 MOTION',release='I24MOTION_PUBLIC_v1.0',format='json-array',direction=-1,
              terms_accepted=True,corrected_timestamps=True,source_url='https://i24motion.org/access_data',
              x_origin_ft=320080,session='UNIT_FIXTURE_NOT_REAL',recording_date='2022-11-21',
              start_unix_s=0,end_unix_s=600,
              files=[dict(path='westbound_selected.json',bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())])
    (tmp_path/'acquisition.json').write_text(json.dumps(meta))
    (tmp_path/'source_audit.json').write_text(json.dumps({'selection':{'x_min':316800,'x_max':320080}}))
    result=quality_control(tmp_path)
    assert result['minimum_usable_concurrent_tracks']==2
    assert result['rejected_tracks']=={'invalid_dimensions_track':1}
    assert result['coordinate_roundtrip_max_error_ft']['x']<1e-7
    assert result['training']=='NOT_RUN'
    assert result['counts']['accepted_tracks']==2
