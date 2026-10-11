"""Exploratory history-defined traffic strata; never alter the primary all-window score."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from orchestra_wm.i24.data import load_scene,sha
from orchestra_wm.i24_continuous.data import exposure_fields
from orchestra_wm.i24_continuous.source import write_immutable

p=argparse.ArgumentParser();p.add_argument('--output',default='outputs/i24_phase4c/matched_macro_seed101_u300');args=p.parse_args()
root=Path(args.output);data_root=Path('outputs/i24_phase4b/development_v1');data=json.loads((data_root/'data_manifest.json').read_text())
tables=[];histories=[]
for split in ('validation','development_test'):
    evaluation=root/split
    if not (evaluation/'summary.json').exists():continue
    source_split='test' if split=='development_test' else split
    items=[i for i in data['scenes'] if i['split']==source_split]
    entries=json.loads((evaluation/'arrays_manifest.json').read_text());descriptors=[]
    for e in entries:
        if e['variant']!='full' or e['regime']!='dense':continue
        path=evaluation/e['path'];assert sha(path)==e['sha256'];scene=load_scene(data_root/items[e['scene_number']]['path'])
        with np.load(path) as a:fields,support=exposure_fields(a['history'],a['history_valid'],scene.road)
        speed=float(fields[...,0][support].mean());density=float(fields[...,1][support].mean())
        descriptors.append(dict(scene=e['scene'],split=split,history_supported_speed_m_s=speed,history_supported_density_veh_km=density,
                                history_field_support_fraction=float(support.mean()),
                                speed_regime='slow_<10' if speed<10 else 'intermediate_10_20' if speed<20 else 'fast_>=20',
                                density_regime='low_<20' if density<20 else 'medium_20_40' if density<40 else 'high_>=40'))
    d=pd.DataFrame(descriptors);histories.append(d)
    scores=pd.read_csv(evaluation/'evaluation.csv').merge(d,on='scene',validate='many_to_one')
    keys=['split','speed_regime','density_regime','variant','regime','horizon_seconds']
    table=scores.groupby(keys).agg(windows=('scene','size'),macro_energy=('macro_energy','mean'),micro_fde_m=('fde_m','mean'),macro_coverage90=('macro_coverage90','mean')).reset_index()
    tables.append(table)
pd.concat(histories).to_csv(root/'history_regimes.csv',index=False)
pd.concat(tables).to_csv(root/'regime_scores.csv',index=False)
write_immutable(root/'regime_analysis.json',dict(source_sha256=sha(Path(__file__)),history_table_sha256=sha(root/'history_regimes.csv'),score_table_sha256=sha(root/'regime_scores.csv'),
    scope='Exploratory post-hoc stratification by dense historical supported-bin means. No outcome-based model selection or change to primary all-window comparison.',
    limitations='Supported-bin reconstructed density is not complete roadway density. Same-day windows are not independent recording groups.'))
print('Saved history-defined regime tables; primary scores unchanged')
