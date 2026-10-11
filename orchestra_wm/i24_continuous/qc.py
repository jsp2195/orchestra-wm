"""Streaming authentic-source diagnostics using the existing v1.0 data adapter."""
import collections
import json
from pathlib import Path

import ijson
import numpy as np

from orchestra_wm.i24.data import I24Adapter, sha
from .source import write_immutable


def quality_control(root):
    root=Path(root)
    meta=json.loads((root/'acquisition.json').read_text())
    audit=json.loads((root/'source_audit.json').read_text())
    adapter=I24Adapter(meta)
    start,end=meta['start_unix_s'],meta['end_unix_s']
    frames=np.zeros(round((end-start)*5),dtype=np.int64)
    spatial=np.zeros((len(frames),10,4),dtype=np.int64)
    counts=collections.Counter();flags=collections.Counter();rejects=collections.Counter()
    seen=set();speeds=[];cadence=[];durations=[];x_error=0.;y_error=0.
    x_min,x_max=audit['selection']['x_min'],audit['selection']['x_max']
    for entry in meta['files']:
        path=root/entry['path']
        if path.stat().st_size!=entry['bytes'] or sha(path)!=entry['sha256']:
            raise ValueError('Source integrity mismatch')
        with path.open('rb') as handle:
            for record in ijson.items(handle,'item',use_float=True):
                counts['input_tracks']+=1
                counts['input_samples']+=len(record['timestamp'])
                flags.update(record.get('flags',[]))
                try:
                    frame,detail=adapter.record(record,meta['session'],entry['sha256'])
                except (ValueError,TypeError,KeyError):
                    rejects['schema_or_array_error']+=1
                    continue
                if frame is None:
                    rejects.update(detail)
                    continue
                ident=str(frame.track.iloc[0])
                if ident in seen: raise ValueError('Duplicate selected track identity')
                seen.add(ident)
                counts['accepted_tracks']+=1
                counts['usable_samples']+=int(frame.usable.sum())
                counts['rejected_samples']+=detail['rejected_samples']
                counts['gaps_over_0_5s']+=detail['gaps']
                counts['lane_jumps_over_one']+=detail['lane_jumps']
                cadence.append(detail['native_dt_median'])
                usable_time=frame.time.to_numpy()[frame.usable.to_numpy()]
                durations.append(float(usable_time[-1]-usable_time[0]) if len(usable_time)>1 else 0.)
                # Check exact inverse of WB back-center to travel-increasing center.
                reconstructed=adapter.origin_ft-(frame.s.to_numpy()-frame.length.to_numpy()/2)/.3048
                x_error=max(x_error,float(np.max(np.abs(reconstructed-np.asarray(record['x_position'])))))
                y_error=max(y_error,float(np.max(np.abs(frame.d.to_numpy()/.3048-np.asarray(record['y_position'])))))
                t=frame.time.to_numpy()
                use=frame.usable.to_numpy()&(t>=start)&(t<end)
                f=np.floor((t[use]-start)*5).astype(int)
                frames[np.unique(f)]+=1
                x=np.asarray(record['x_position'])[use]
                y=np.asarray(record['y_position'])[use]
                b=np.floor((x-x_min)/(x_max-x_min)*10).astype(int)
                lane=np.floor((y-12)/12).astype(int)
                valid=(b>=0)&(b<10)&(lane>=0)&(lane<4)
                triples=np.unique(np.stack([f[valid],b[valid],lane[valid]],axis=1),axis=0)
                if len(triples):np.add.at(spatial,(triples[:,0],triples[:,1],triples[:,2]),1)
                speeds.extend(frame.vs.to_numpy()[use][::5].tolist())
    support=bool(len(frames) and frames.min()>=2 and x_error<1e-7 and y_error<1e-7)
    result=dict(status='REGIONAL_SUBSECOND_SUPPORT_PASSED' if support else 'REGIONAL_SUBSECOND_SUPPORT_FAILED',
                source='ORIGINAL_I24_MOTION',session=meta['session'],recording_date=meta['recording_date'],
                source_sha256=meta['files'][0]['sha256'],counts=dict(counts),rejected_tracks=dict(rejects),
                source_flags=dict(flags),selected_seconds=end-start,sample_hz=5,
                minimum_usable_concurrent_tracks=int(frames.min()),
                concurrent_tracks_quantiles=np.quantile(frames,[0,.05,.5,.95,1]).tolist(),
                frames_with_fewer_than_two_tracks=int((frames<2).sum()),
                median_native_dt_s=float(np.median(cadence)) if cadence else None,
                usable_track_span_s_quantiles=np.quantile(durations,[0,.05,.5,.95,1]).tolist() if durations else [],
                tracks_with_25s_usable_span=int((np.asarray(durations)>=25).sum()),
                usable_speed_m_s_quantiles=np.quantile(speeds,[0,.05,.5,.95,1]).tolist() if speeds else [],
                coordinate_roundtrip_max_error_ft=dict(x=x_error,y=y_error),
                regional_length_m=(x_max-x_min)*.3048,
                lane_bin_observed_frame_fraction=(spatial>0).mean(0).tolist(),
                lane_bin_mean_concurrent_tracks=spatial.mean(0).tolist(),
                training='NOT_RUN',limitations=[
                    'Empty lane/bins are UNKNOWN, not zero density',
                    'Sample support does not establish unbiased census or tracking identity correctness',
                    'Lane boundaries are approximate per v1.x documentation; no calibrated sensor confidence',
                    'Corrected timestamp meaning is documented; raw synchronization corrections cannot be independently reconstructed',
                    'Regional single-day pilot cannot establish independent-session generalization'])
    write_immutable(root/'authentic_qc.json',result)
    return result
