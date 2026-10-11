"""Independent prepared-scene QC and compact local preservation bundle."""
import hashlib,json,tarfile,shutil
from pathlib import Path
import numpy as np
from orchestra_wm.i24.data import load_scene,sha
from orchestra_wm.i24_continuous.data import exposure_fields
from orchestra_wm.i24_continuous.source import write_immutable
root=Path('outputs/i24_phase4b/development_v1');m=json.loads((root/'data_manifest.json').read_text());ids={s:set() for s in ['train','validation','test']};stats=[]
for item in m['scenes']:
 p=root/item['path'];s=load_scene(p);ids[item['split']].update(s.track_ids)
 assert np.allclose(np.diff(s.time),.2,rtol=0,atol=1e-6)
 fields,support=exposure_fields(s.values,s.existence,s.road)
 with np.load(p.with_suffix('.fields.npz')) as a:
  assert np.array_equal(support,a['support']) and np.array_equal(fields,a['fields'])
 assert np.isfinite(s.values).all() and s.existence.sum(1).min()>=2
 stats.append(dict(block=item['block'],split=item['split'],start=item['start'],tracks=len(s.track_ids),usable_5hz_samples=int(s.existence.sum()),median_speed_m_s=float(np.median(s.values[:,:,2][s.existence])),support_fraction=float(support.mean())))
for a in ids:
 for b in ids:
  if a!=b:assert not ids[a].intersection(ids[b])
for a in m['scenes']:
 for b in m['scenes']:
  if a['split']!=b['split']:assert a['end']<b['start'] or b['end']<a['start']
for p,h in m['artifacts'].items():assert sha(root/p)==h
bundle=Path('work/phase4b/expanded_prepared_recovery.tar.gz')
if not bundle.exists():
 with tarfile.open(bundle,'x:gz',compresslevel=1) as tar:
  tar.add(root/'canonical',arcname=str(root/'canonical'))
  for p in root.glob('*.json'):tar.add(p,arcname=str(p))
with tarfile.open(bundle) as tar:
 for member in tar:
  if member.isfile():
   h=hashlib.sha256()
   with tar.extractfile(member) as f:
    for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
   assert h.hexdigest()==sha(Path(member.name))
counts=m['counts'];retained=sum(v for k,v in counts.items() if k.endswith('_samples') and 'unusable' not in k and 'omitted' not in k)
omitted=sum(v for k,v in counts.items() if k.endswith('boundary_samples_omitted'))
report=dict(status='EXPANDED_AUTHENTIC_DEVELOPMENT_QC_PASSED',new_training_updates=0,source_dates=m['source_dates'],scene_counts=m['scene_counts'],native_samples=retained,retained_tracks=sum(v for k,v in counts.items() if k.endswith('_tracks') and not any(t in k for t in ('boundary','short','invalid'))),
 shared_source_ids=0,nonoverlapping_split_support=True,macro_labels_exactly_recomputed=True,scenes=stats,
 boundary_omitted_samples=omitted,boundary_omission_fraction=omitted/(omitted+retained),free_bytes=shutil.disk_usage(root).free,
 local_recovery_bundle=dict(path=str(bundle),bytes=bundle.stat().st_size,sha256=sha(bundle),members_verified=True),
 data_manifest_sha256=sha(root/'data_manifest.json'),remote_backup='NOT_CONFIGURED')
write_immutable(root/'quality_and_recovery.json',report)
print(json.dumps({k:v for k,v in report.items() if k!='scenes'},indent=2))
