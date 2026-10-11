"""Fixed central-road crop of saved forecasts; geometric links are not causal claims."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from orchestra_wm.i24.data import sha,load_scene
from orchestra_wm.i24_continuous.source import write_immutable

p=argparse.ArgumentParser();p.add_argument('--evaluation-output',default='outputs/i24_phase4c/matched_macro_seed101_u300/development_test')
p.add_argument('--output',default='outputs/i24_phase4c/gallery');args=p.parse_args()
root=Path(args.output);evaluation=Path(args.evaluation_output)
entries=json.loads((evaluation/'arrays_manifest.json').read_text())
entry=next(e for e in entries if e['scene_number']==0 and e['regime']=='dense' and e['variant']=='full')
path=evaluation/entry['path'];assert sha(path)==entry['sha256']
with np.load(path) as a:arrays={k:a[k].copy() for k in ('agents','truth','valid','predicted_valid')}
data_root=Path('outputs/i24_phase4b/development_v1');manifest=json.loads((data_root/'data_manifest.json').read_text())
split='test' if json.loads((evaluation/'summary.json').read_text())['split']=='development_test' else 'validation'
scene=load_scene(data_root/next(i['path'] for i in manifest['scenes'] if i['split']==split));road=scene.road
center=(road.s_min+road.s_max)/2;left=center-125;right=center+125
low=min(road.lane_centers)-road.lane_width/2;high=max(road.lane_centers)+road.lane_width/2
frames=[];steps=list(range(0,100,5))+[99]
for t in steps:
    im=Image.new('RGB',(1000,530),'#f6f8fc');d=ImageDraw.Draw(im)
    d.text((14,9),f'COHORT INTERACTIONS / +{(t+1)*.2:.1f}s / fixed central 250 m / sample 1',fill='#19334d')
    for row,(values,valid,label,color) in enumerate([(arrays['truth'][t],arrays['valid'][t],'Recorded cohort / comparison only','#ffb35b'),
                                                  (arrays['agents'][0,t],arrays['predicted_valid'][0,t],'Generated cohort / geometric neighborhoods under 100 m','#54dbea')]):
        x0,x1=40,960;y0=45+row*225;y1=y0+190
        d.rectangle((x0,y0,x1,y1),fill='#14293e');d.text((x0+8,y0+7),label,fill='white')
        sx=(x1-x0)/(right-left);sy=(y1-y0-34)/(high-low)
        point=lambda a:(x0+(a[0]-left)*sx,y0+29+(a[1]-low)*sy)
        for lane in road.lane_centers:
            yy=point([left,lane])[1];d.line((x0,yy,x1,yy),fill='#405d77')
        use=np.flatnonzero(valid&(values[:,0]>=left)&(values[:,0]<right)&(values[:,1]>=low)&(values[:,1]<high))
        if row:
            for ii,i in enumerate(use):
                for j in use[:ii]:
                    if np.linalg.norm(values[i,:2]-values[j,:2])<100:
                        d.line((*point(values[i]),*point(values[j])),fill='#355b72')
        for i in use:
            x,y=point(values[i]);length,width=values[i,6:8]
            d.rectangle((max(x0,x-length*sx/2),max(y0+27,y-width*sy/2),min(x1,x+length*sx/2),min(y1,y+width*sy/2)),fill=color)
        d.text((x0,y1+4),f'{left:.1f} m  -> westbound travel ->  {right:.1f} m',fill='#19334d')
    d.text((14,505),'Lines show a local subset of potential model messages, not causal interactions or verified collisions.',fill='#19334d')
    frames.append(im)
dest=root/'05_micro_interactions.gif'
frames[0].save(dest,save_all=True,append_images=frames[1:],duration=180,loop=0)
write_immutable(root/'interaction_provenance.json',dict(checkpoint_sha256=entry['checkpoint_sha256'],rollout_sha256=entry['sha256'],
    source_sha256=sha(Path(__file__)),gif_sha256=sha(dest),fixed_viewport_s_m=[left,right],sample=0,scene_number=0,frames=steps,
    qualification='Distance-based potential messages; source identity/census and causal-interaction limits remain'))
gallery=root/'VISUAL_GALLERY.md';text=gallery.read_text();addition='\n[Microscopic interaction view](05_micro_interactions.gif): fixed central 250 m, first prior sample, geometric message neighborhoods. These are not causal links.\n'
if addition not in text:gallery.write_text(text+addition)
print('Saved checkpoint-backed microscopic interaction view')
