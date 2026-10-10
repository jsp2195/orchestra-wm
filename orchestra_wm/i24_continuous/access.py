"""Read-only, bounded local import. Provenance declarations are not authentication."""
import json
import shutil
from pathlib import Path
from urllib.parse import urlsplit

import ijson
import numpy as np
from orchestra_wm.i24.data import I24Adapter, sha


class AccessBlocked(ValueError):
    pass


def inspect_local(root, budget):
    root = Path(root).resolve()
    if not (root / 'acquisition.json').is_file():
        raise AccessBlocked('Missing authorized source files and acquisition.json; see docs/I24_CONTINUOUS_ACQUISITION.md')
    if (root / 'acquisition.json').stat().st_size > 65536:
        raise AccessBlocked('Acquisition metadata exceeds size budget')
    meta = json.loads((root / 'acquisition.json').read_text())
    allowed = {'dataset','release','format','direction','terms_accepted','source_url',
               'x_origin_ft','files','corrected_timestamps','session','recording_date',
               'start_unix_s','end_unix_s'}
    if set(meta) - allowed:
        raise AccessBlocked('Unknown acquisition fields; never supply credentials or headers')
    u = urlsplit(meta.get('source_url', ''))
    if u.scheme != 'https' or u.netloc != 'i24motion.org' or u.path not in ('/data','/access_data') or u.query or u.fragment:
        raise AccessBlocked('Use only the public official landing URL, never a signed or credentialed URL')
    if meta.get('terms_accepted') is not True or meta.get('corrected_timestamps') is not True:
        raise AccessBlocked('Accepted terms and corrected timestamp stream must be declared')
    import datetime
    datetime.date.fromisoformat(meta['recording_date'])
    start, end = float(meta['start_unix_s']), float(meta['end_unix_s'])
    if not np.isfinite([start,end]).all() or not 600 <= end-start <= 1800:
        raise AccessBlocked('Select one continuous 10–30 minute recording interval')
    if not isinstance(meta['session'], str) or not meta['session']:
        raise AccessBlocked('A source session is required')
    adapter = I24Adapter(meta)
    if shutil.disk_usage(root).free < budget['min_free_bytes']:
        raise AccessBlocked('Insufficient free disk reserve')
    sources = []
    total = 0
    for entry in meta['files']:
        if set(entry) != {'path','bytes','sha256'}:
            raise AccessBlocked('Each source needs only relative path, bytes and SHA256')
        rel = Path(entry['path'])
        p = root / rel
        if rel.is_absolute() or '..' in rel.parts or not p.resolve().is_relative_to(root) or any((root / Path(*rel.parts[:i])).is_symlink() for i in range(1,len(rel.parts)+1)):
            raise AccessBlocked('Source paths must be local regular files without symlinks')
        if p.suffix != '.json' or not p.is_file():
            raise AccessBlocked('Only extracted original JSON-array sources are accepted')
        total += p.stat().st_size
        if total > budget['max_source_bytes']:
            raise AccessBlocked('Source-byte budget exceeded')
        if p.stat().st_size != entry['bytes'] or sha(p) != entry['sha256']:
            raise AccessBlocked('Source size or checksum mismatch')
        sources.append((p, entry))
    counts = np.zeros(int(end-start), dtype=np.int64)
    seen, samples, cadence = set(), 0, []
    for p, entry in sources:
        with p.open('rb') as f:
            # Reject HTML, error objects and JSONL before streaming records.
            first = b' '
            while first and first.isspace():
                first = f.read(1)
            if first != b'[':
                raise AccessBlocked('Source is not a JSON array (HTML/login responses are not data)')
            f.seek(0)
            for record in ijson.items(f, 'item', use_float=True):
                samples += len(record.get('timestamp', []))
                if samples > budget['max_samples']:
                    raise AccessBlocked('Sample budget exceeded; no partial success recorded')
                frame, qc = adapter.record(record, meta['session'], entry['sha256'])
                if frame is None:
                    raise AccessBlocked('Rejected source track; inspect release/schema before conversion')
                ident = str(frame.track.iloc[0])
                if ident in seen:
                    raise AccessBlocked('Duplicate source identity requires explicit reconciliation')
                seen.add(ident)
                t = frame.time.to_numpy()[frame.usable.to_numpy()]
                bins = np.unique(np.floor(t[(t >= start) & (t < end)]-start).astype(int))
                counts[bins] += 1
                cadence.append(qc['native_dt_median'])
    if not len(seen) or np.min(counts) < 2:
        raise AccessBlocked('Selected interval lacks multivehicle support in every second; no clip concatenation allowed')
    return {'status':'LOCAL_SCHEMA_QC_PASSED_PROVENANCE_REVIEW_REQUIRED',
            'source':'ORIGINAL_I24_MOTION','declared_session':meta['session'],
            'declared_date':meta['recording_date'],'source_bytes':total,
            'files':[e for _,e in sources],'source_tracks':len(seen),'source_samples':samples,
            'selected_seconds':end-start,'min_tracks_per_second':int(counts.min()),
            'median_native_dt_s':float(np.median(cadence)),
            'limitations':'Per-second support is not proof of frame continuity or authentic origin. Review source provenance, subsecond gaps, corridor coverage and source terms before G1 acceptance.',
            'training':'BLOCKED_PENDING_REAL_SOURCE_REVIEW'}
