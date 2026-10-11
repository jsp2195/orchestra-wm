"""Source-only dataset audit; does not instantiate models or score predictions."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from orchestra_wm.i24.data import load_scene,sha
from orchestra_wm.i24_continuous.source import write_immutable
from orchestra_wm.i24_phase4e.data import ROLES
p=argparse.ArgumentParser();p.add_argument('--data',default='outputs/i24_phase4e/data_v3');p.add_argument('--dates',nargs='+',required=True);p.add_argument('--output',required=True);a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=True);rows=[];identities={};geometry=None;sources={}
for date in a.dates:
    root=Path(a.data)/date;m=json.loads((root/'data_manifest.json').read_text());n=json.loads((root/'native_manifest.json').read_text())
    v=json.loads((root/'source_verification.json').read_text());assert v['native_manifest_sha256']==sha(root/'native_manifest.json')
    sources[date]=dict(data_manifest_sha256=sha(root/'data_manifest.json'),source_verification_sha256=sha(root/'source_verification.json'),
        nested_zip_sha256=n['nested_zip_sha256'],source_records=n['counts']['source_records_scanned'],source_samples=n['counts']['source_samples'],
        source_start_unix_s=n['source_start_unix_s'],source_end_unix_s=n['source_end_unix_s'],native_dt_median_s=n['native_dt_median_s'],
        role=ROLES[date],counts=n['counts'],excluded_windows=m['excluded_windows'])
    identities[date]=set()
    for entry in m['scenes']:
        assert entry['split']==ROLES[date]
        path=root/entry['path'];scene=load_scene(path);ids=set(scene.track_ids);identities[date].update(ids)
        current=(scene.road.s_min,scene.road.s_max,scene.road.lane_centers,scene.road.lane_width,scene.road.bins)
        if geometry is None:geometry=current
        assert current==geometry
        assert scene.existence.sum(1).min()>=2 and len(ids)<=2048
        assert np.isfinite(scene.values[scene.existence]).all() and np.allclose(np.diff(scene.time),.2,atol=1e-6)
        with np.load(path.with_suffix('.fields.npz')) as f:
            fields=f['fields'];support=f['support'];known=fields[support]
            assert np.isfinite(known).all()
            identity_error=float(abs(known[:,2]-3.6*known[:,0]*known[:,1]).max())
            rows.append(dict(date=date,split=entry['split'],block=entry['block'],scene=scene.scene_id,
                tracks=len(ids),usable_samples=int(scene.existence.sum()),minimum_concurrent=int(scene.existence.sum(1).min()),
                support_fraction=float(support.mean()),supported_cells=int(support.sum()),
                mean_speed_m_s=float(known[:,0].mean()),mean_density_veh_km_lane=float(known[:,1].mean()),mean_flow_veh_h_lane=float(known[:,2].mean()),
                congested_bin_fraction=float((known[:,0]<10).mean()),intermediate_bin_fraction=float(((known[:,0]>=10)&(known[:,0]<20)).mean()),free_bin_fraction=float((known[:,0]>=20).mean()),
                negative_speed_fraction=float((known[:,0]<0).mean()),flow_identity_max_abs_error=identity_error))
shared={}
for first in identities:
    for second in identities:
        if first<second:
            count=len(identities[first]&identities[second]);shared[first+'/'+second]=count
            if count:raise ValueError('Source trajectory ID shared across recording days')
frame=pd.DataFrame(rows);frame.to_csv(out/'windows.csv',index=False)
frame.groupby(['date','split']).agg(windows=('scene','size'),slots=('block','nunique'),
    mean_speed_m_s=('mean_speed_m_s','mean'),mean_density_veh_km_lane=('mean_density_veh_km_lane','mean'),
    mean_flow_veh_h_lane=('mean_flow_veh_h_lane','mean'),support_fraction=('support_fraction','mean'),
    congested_bin_fraction=('congested_bin_fraction','mean'),intermediate_bin_fraction=('intermediate_bin_fraction','mean'),
    free_bin_fraction=('free_bin_fraction','mean'),maximum_window_tracks=('tracks','max')).to_csv(out/'per_day.csv')
write_immutable(out/'audit.json',dict(status='SOURCE_ONLY_QC_PASSED',dates=a.dates,sources=sources,
    cross_day_shared_source_ids=shared,geometry=geometry,windows_csv_sha256=sha(out/'windows.csv'),
    interpretation='Correlated reconstructed retained-population windows; not a complete vehicle census; no forecasts evaluated'))
print(frame.groupby('date').size().to_string())
