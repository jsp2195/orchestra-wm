"""Bounded extraction from an authenticated INCEPTION ZIP; no full JSON extraction."""
import collections
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import zipfile

import ijson
import numpy as np

from .drive import TransferBlocked, hashes, inspect_zip


def write_immutable(path, value):
    path = Path(path)
    payload = json.dumps(value, indent=2, allow_nan=False) + '\n'
    if path.exists():
        if path.read_text() != payload:
            raise TransferBlocked('Existing provenance differs; choose a separate destination')
        return
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(payload)
    temporary.replace(path)


def extract_interval(archive, destination, *, start, seconds=600,
                     x_min=316800., x_max=320080., max_bytes=3_000_000_000,
                     min_free=10_000_000_000):
    """Scan all records/CRC; retain only declared WB time/space support.

    No track or timestamp interpolation. Two seconds of boundary support permit
    backward derivatives. Filtering and short-track exclusions are inventoried.
    Source identity is kept; full source trajectories remain in the input ZIP.
    """
    archive, destination = Path(archive), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    audit_path = destination/'source_audit.json'
    selected = destination/'westbound_selected.json'
    if audit_path.exists():
        previous = json.loads(audit_path.read_text())
        expected = dict(start=start, seconds=seconds, x_min=x_min, x_max=x_max)
        if previous['selection'] != expected or hashes(archive)[0] != previous['nested_zip_sha256']:
            raise TransferBlocked('Cached selection/source mismatch; existing files preserved')
        if hashes(selected)[0] != previous['selected_sha256']:
            raise TransferBlocked('Cached selected-source checksum mismatch')
        return previous
    partial = selected.with_name(selected.name+'.part')
    if selected.exists() or partial.exists():
        raise TransferBlocked('Uncommitted extraction exists; preserve it and choose a separate destination')
    if not np.isfinite([start,seconds,x_min,x_max]).all() or not 600 <= seconds <= 1800 or not x_min < x_max:
        raise TransferBlocked('Select a bounded 10–30 minute interval and finite roadway bounds')
    sha256, md5 = hashes(archive)
    members = inspect_zip(archive)
    json_members = [m for m in members if m['name'].endswith('.json')]
    if len(json_members) != 1:
        raise TransferBlocked('Review archive layout before selecting a trajectory member')
    member = json_members[0]
    counts = collections.Counter()
    native_dt = []
    source_start, source_end = float('inf'), float('-inf')
    selected_tracks, selected_samples = 0, 0
    frames = np.zeros((round(seconds*5), 10), dtype=np.int64)
    seen = set()
    last_report = time.monotonic()
    with zipfile.ZipFile(archive) as z, z.open(member['name']) as handle, partial.open('xb') as output:
        output.write(b'[')
        for record in ijson.items(handle, 'item', use_float=True):
            counts['source_tracks'] += 1
            if time.monotonic()-last_report>30:
                print(f"Scanned {counts['source_tracks']:,} tracks; retained {selected_tracks:,} / {selected_samples:,} samples",flush=True)
                last_report=time.monotonic()
            direction = int(record.get('direction', 0))
            counts[f'direction_{direction}_tracks'] += 1
            t = np.asarray(record.get('timestamp', []), dtype=float)
            x = np.asarray(record.get('x_position', []), dtype=float)
            y = np.asarray(record.get('y_position', []), dtype=float)
            counts['source_samples'] += len(t)
            if not len(t) or t.shape != x.shape or t.shape != y.shape:
                counts['invalid_array_tracks'] += 1
                continue
            if not np.isfinite(np.stack([t,x,y])).all():
                counts['nonfinite_tracks'] += 1
                continue
            source_start = min(source_start, float(t.min()))
            source_end = max(source_end, float(t.max()))
            dt = np.diff(t)
            if len(dt):
                native_dt.append(float(np.median(dt)))
                counts['source_gaps_over_0_5s'] += int((dt>.5).sum())
            if np.any(dt<=0):
                counts['nonmonotonic_tracks'] += 1
                continue
            if direction != -1:
                continue
            mask = (t>=start-2)&(t<start+seconds+2)&(x>=x_min)&(x<x_max)
            if not mask.any():
                continue
            if mask.sum()<3:
                counts['selected_short_tracks_excluded'] += 1
                continue
            ident = record['_id']
            ident = ident.get('$oid') if isinstance(ident, dict) else str(ident)
            if ident in seen:
                raise TransferBlocked('Duplicate selected source identity; reconciliation required')
            seen.add(ident)
            # Preserve every original field. Clip sample-aligned arrays only.
            chosen = {k: [v[i] for i in np.flatnonzero(mask)] if isinstance(v,list) and len(v)==len(t)
                      and k in ('timestamp','x_position','y_position') else v for k,v in record.items()}
            chosen['first_timestamp'] = float(t[mask][0])
            chosen['last_timestamp'] = float(t[mask][-1])
            chosen['starting_x'] = float(x[mask][0])
            chosen['ending_x'] = float(x[mask][-1])
            payload = json.dumps(chosen, separators=(',',':'), allow_nan=False).encode()
            if output.tell()+len(payload)+2>max_bytes or shutil.disk_usage(destination).free<len(payload)+min_free:
                raise TransferBlocked('Selected extraction reached byte/disk budget; partial preserved')
            if selected_tracks: output.write(b',')
            output.write(payload)
            selected_tracks += 1
            selected_samples += int(mask.sum())
            valid = mask&(t>=start)&(t<start+seconds)&(y>=12)&(y<60)
            # Each track counts once per 0.2-second frame/spatial bin.
            frame = np.floor((t[valid]-start)*5).astype(int)
            spatial = np.floor((x[valid]-x_min)/(x_max-x_min)*10).astype(int)
            pairs = np.unique(np.stack([frame,spatial],axis=1),axis=0)
            if len(pairs): np.add.at(frames,(pairs[:,0],pairs[:,1]),1)
        output.write(b']\n')
        output.flush(); os.fsync(output.fileno())
        # ijson consumes through end of member: ZIP CRC checked by ZipExtFile.
        while handle.read(1024*1024): pass
    if not selected_tracks:
        raise TransferBlocked('No westbound tracks in the declared selection; partial preserved')
    partial.replace(selected)
    result = dict(selection=dict(start=start,seconds=seconds,x_min=x_min,x_max=x_max),
                  nested_zip_name=archive.name,nested_zip_sha256=sha256,nested_zip_md5=md5,
                  source_member=member,nested_member_crc_verified=True,
                  source_start_unix_s=source_start,source_end_unix_s=source_end,
                  source_duration_s=source_end-source_start,counts=dict(counts),
                  native_dt_median_s=float(np.median(native_dt)),
                  selected_tracks=selected_tracks,selected_samples=selected_samples,
                  selected_bytes=selected.stat().st_size,selected_sha256=hashes(selected)[0],
                  min_tracks_per_5hz_frame=int(frames.sum(1).min()),
                  spatial_bins=10,spatial_bin_frame_coverage=(frames>0).mean(0).tolist(),
                  spatial_bin_mean_concurrent_tracks=frames.mean(0).tolist(),
                  source_recording_date=datetime.datetime.fromtimestamp(source_start,datetime.timezone.utc).date().isoformat(),
                  limitations=['Regional time/space subset, not complete corridor census',
                               'Raw simultaneous counts precede kinematic QC',
                               'Offline source reconstruction may use future imagery',
                               'Source date/session/release supported by bundled README and public v1.x documentation'])
    write_immutable(audit_path,result)
    return result
