import json
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from orchestra_wm.envs.traffic_env import TrafficEnv
from orchestra_wm.evaluation.common import infer,observe,write_json
from orchestra_wm.evaluation.action_sensitivity import named_plans
from orchestra_wm.models.training import load_model
from orchestra_wm.planning.orchestrator import Orchestrator
from orchestra_wm.planning.oracle import true_rollout
from orchestra_wm.sim.renderer import draw_scene

@torch.no_grad()
def make_demo(cfg,out):
    model=load_model(out/'checkpoints'/'orchestra.pt')
    c={**cfg,'vehicles':16,'connected':4,'plan_horizon':10}
    env=TrafficEnv(c,'intersection',70117);obs,_=env.reset(seed=70117)
    frames=[];warm=[];previous=None
    for t in range(4):
        frames.append((obs,previous,env.state_array()));warm.append(env.state_array().tolist())
        if t<3: previous=np.zeros(env.n,int);obs,*_=env.step(previous)
    state=infer(model,frames);planner=Orchestrator(model,c,71)
    optimized,_,_=planner.plan(state)
    fixed=named_plans(env,c['plan_horizon'])
    plans=np.array([fixed['aggressive'],fixed['conservative'],fixed['staggered'],optimized])
    predictions=model.imagine(state.repeat(4),torch.tensor(plans))
    costs=(predictions['components'].numpy()@c['objective_weights']).sum(1)
    selected=int(np.argmin(costs))
    imagined=predictions['agents'].numpy();truth=[];true_cost=[]
    for plan in plans:
        states,components=true_rollout(env,plan);truth.append(states.tolist());true_cost.append(float((components@c['objective_weights']).sum()))
    initial=env.state_array().copy();initial_obs=obs['agents'][:,:6].copy();visible=obs['mask'].copy()
    belief=state.agents[0].numpy().copy();executed=[initial.tolist()];executed_actions=[]
    for t in range(14):
        action=plans[selected,0] if t==0 else planner.plan(state)[0][0]
        executed_actions.append(action.tolist());obs,*_=env.step(action)
        executed.append(env.state_array().tolist());state=observe(model,state,obs,action)
    data={'roads':[l.centerline.tolist() for l in env.road.lanes], 'connected':[v.connected for v in env.vehicles],
          'visible':visible.tolist(),'initial':initial.tolist(),'observations':initial_obs.tolist(),'belief':belief.tolist(),
          'warm':warm,'imagined':imagined.tolist(),'truth':truth,'costs':costs.tolist(),'true_costs':true_cost,
          'selected':selected,'executed':executed,'actions':executed_actions,
          'labels':['A · AGGRESSIVE','B · CONSERVATIVE','C · STAGGERED','D · CEM OPTIMIZED']}
    write_json(out/'demo_data.json',data)
    html=HTML.replace('__DATA__',json.dumps(data))
    (out/'demo.html').write_text(html)
    # A portable animated overview uses exactly the same recorded predictions and execution.
    images=[]
    for t in range(32):
        fig,axes=plt.subplots(2,4,figsize=(12,7),facecolor='#101827')
        draw_scene(axes[0,0],env.road,initial,np.array(data['connected']),visible,'TRUE SCENE\nOrange = hidden from cameras')
        draw_scene(axes[0,1],env.road,initial_obs,np.array(data['connected']),title='PARTIAL OBSERVATIONS')
        draw_scene(axes[0,2],env.road,belief,np.array(data['connected']),title='ORCHESTRA BELIEF')
        k=min(max(t-15,0),len(executed)-1)
        draw_scene(axes[0,3],env.road,np.array(executed[k]),np.array(data['connected']),title='TRUE EXECUTION\nReplan every step')
        for i,ax in enumerate(axes[1]):
            h=min(max(t-3,0),c['plan_horizon']-1)
            draw_scene(ax,env.road,imagined[i,h],np.array(data['connected']),title=f'{data["labels"][i]}\nPredicted cost {costs[i]:.2f}'+(' • SELECTED' if i==selected else ''),trajectories=imagined[i,:h+1])
            if i==selected:
                for sp in ax.spines.values():sp.set_color('#41e2bc');sp.set_linewidth(2)
        fig.suptitle('ORCHESTRA-WM  /  ONE BELIEF → FOUR IMAGINED FUTURES → JOINT ACTION',color='white',fontsize=13)
        fig.text(.5,.025,'Synthetic research only • learned scores choose the action • simulator futures used only for evaluation',ha='center',color='#93a7bf',fontsize=9)
        for ax in axes.flat:
            ax.set_xticks([]);ax.set_yticks([]);ax.title.set_fontsize(9)
        fig.subplots_adjust(left=.025,right=.985,bottom=.08,top=.86,hspace=.32,wspace=.10)
        fig.canvas.draw()
        images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()).convert('P',palette=Image.Palette.ADAPTIVE,colors=128));plt.close(fig)
    images[0].save(out/'demo.gif',save_all=True,append_images=images[1:],duration=500,loop=0,optimize=True)
    return {'selected_plan':data['labels'][selected],'predicted_costs':costs.tolist(),'true_counterfactual_costs':true_cost}

HTML=r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ORCHESTRA-WM · Imagining coordination</title>
<style>
:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#0b1120;color:#e7eefb;font-family:system-ui,sans-serif}main{max-width:1500px;margin:auto;padding:32px}header{display:flex;justify-content:space-between;align-items:center;gap:24px}.eyebrow{font-size:11px;letter-spacing:3px;color:#41e2bc}h1{font-size:38px;margin:9px 0;letter-spacing:-1px}p{color:#a2b3ca;line-height:1.6}.badge{border:1px solid #34435b;padding:10px 14px;border-radius:20px;font-size:12px}.flow{display:flex;gap:12px;margin:24px 0;color:#a8bdd8;align-items:center}.flow b{color:#41e2bc}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.panel{background:#141e30;border:1px solid #2b3a51;border-radius:12px;overflow:hidden}.panel.selected{border:2px solid #41e2bc;box-shadow:0 0 22px #41e2bc15}.panel h3{font-size:12px;letter-spacing:1px;margin:16px 15px 6px}.panel p{font-size:12px;margin:8px 15px 15px}.panel canvas{width:100%;aspect-ratio:1/1;display:block}.label{display:flex;justify-content:space-between}.pill{color:#41e2bc;font-size:10px;margin:14px}section{margin:24px 0}h2{font-size:18px;font-weight:500}.controls{display:flex;gap:18px;align-items:center;background:#141e30;padding:16px;border-radius:10px}button{background:#41e2bc;color:#102723;border:0;border-radius:6px;padding:10px 18px;font-weight:600;cursor:pointer}input{flex:1;accent-color:#41e2bc}.legend{display:flex;gap:20px;font-size:12px;margin:14px 0;color:#b7c7da}.dot{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:6px}footer{font-size:12px;color:#8195af;border-top:1px solid #29364a;padding-top:18px}#phase{font-size:13px;color:#41e2bc;min-width:200px}label{font-size:12px;cursor:pointer}@media(max-width:900px){.grid{grid-template-columns:repeat(2,1fr)}h1{font-size:28px}main{padding:18px}header{display:block}}
</style><main><header><div><div class="eyebrow">MULTI-AGENT WORLD MODEL / RESEARCH PROTOTYPE</div><h1>Imagine together. Coordinate together.</h1><p>One partial view. Four possible futures. A joint decision made entirely inside a learned model.</p></div><div class="badge">ORCHESTRA-WM · CPU smoke experiment</div></header>
<div class="flow">OBSERVE <span>→</span> REMEMBER <span>→</span> <b>IMAGINE</b> <span>→</span> RANK <span>→</span> EXECUTE + REPLAN</div>
<div class="legend"><span><i class="dot" style="background:#41e2bc"></i>Connected AV</span><span><i class="dot" style="background:#a7b5c9"></i>Background vehicle</span><span><i class="dot" style="background:#faab62"></i>Hidden from cameras</span><span>Numbers are stable vehicle IDs</span></div>
<section class="grid" id="scene"></section><section><h2>Same inferred world state → four joint coordination plans</h2><div class="grid" id="branches"></div></section>
<div class="controls"><button id="play">Pause</button><input id="time" type="range" min="0" max="199" value="0" aria-label="Animation time"><span id="phase"></span><label><input type="checkbox" id="truth"> Show true counterfactuals</label></div>
<p id="explanation"></p><footer>Synthetic research testbed. Not intended or validated for real vehicles or traffic infrastructure. Predictions and costs are measured outputs of a smoke-trained model; selecting a plan does not establish its quality. Simulator branches are diagnostic and never available to the learned planner. Full evidence is in the adjacent CSV files and experiment summary.</footer></main>
<script>const D=__DATA__;
const sceneNames=['TRUE SCENE','PARTIAL OBSERVATIONS','INFERRED BELIEF','TRUE EXECUTION'];
function panel(title,id,description){return `<article class="panel" id="p${id}"><div class="label"><h3>${title}</h3><span class="pill"></span></div><canvas id="c${id}" width="420" height="420"></canvas><p>${description}</p></article>`}
document.querySelector('#scene').innerHTML=sceneNames.map((n,i)=>panel(n,i,['Complete synthetic state at the branching moment','Camera tracks + AV telemetry + delayed lane loops','Recurrent state inferred only from observation history','Execute first selected action, observe, and replan'][i])).join('');
document.querySelector('#branches').innerHTML=D.labels.map((n,i)=>panel(n,i+4,`Predicted cost <b>${D.costs[i].toFixed(3)}</b> · lower is better`)).join('');
document.querySelector('#p'+(D.selected+4)).classList.add('selected');document.querySelector('#p'+(D.selected+4)+' .pill').textContent='SELECTED';
function draw(id,state,hidden=false,trajectory=null){const canvas=document.querySelector('#c'+id),c=canvas.getContext('2d'),scale=2.25;const xy=(s)=>[210+s[0]*scale,210-s[1]*scale];c.fillStyle='#101827';c.fillRect(0,0,420,420);for(const road of D.roads){c.beginPath();road.forEach((p,i)=>{const q=xy(p);i?c.lineTo(...q):c.moveTo(...q)});c.strokeStyle='#27334a';c.lineWidth=17;c.stroke();c.setLineDash([5,7]);c.strokeStyle='#65758b';c.lineWidth=1;c.stroke();c.setLineDash([])}
if(trajectory){for(let i=0;i<state.length;i++){c.beginPath();trajectory.forEach((s,t)=>{let q=xy([s[i][0]*100,s[i][1]*100]);t?c.lineTo(...q):c.moveTo(...q)});c.strokeStyle=D.connected[i]?'#41e2bc80':'#a7b5c960';c.lineWidth=2;c.stroke()}}
state.forEach((s,i)=>{if(s[5]<.2)return;const p=xy([s[0]*100,s[1]*100]);c.save();c.translate(...p);c.rotate(-Math.atan2(s[4],s[3]));c.fillStyle=hidden&&!D.visible[i]?'#faab62':D.connected[i]?'#41e2bc':'#a7b5c9';c.fillRect(-6,-3,12,6);c.restore();c.fillStyle='#edf4ff';c.font='10px system-ui';c.fillText(i,p[0]+7,p[1]-5)})}
let tick=0,playing=true;const slider=document.querySelector('#time');document.querySelector('#play').onclick=()=>{playing=!playing;document.querySelector('#play').textContent=playing?'Pause':'Play'};slider.oninput=()=>{tick=+slider.value;render()};document.querySelector('#truth').onchange=render;
function render(){draw(0,D.initial,true);draw(1,D.observations);draw(2,D.belief);const h=Math.min(9,Math.max(0,Math.floor((tick-30)/9)));const t=Math.min(D.executed.length-1,Math.max(0,Math.floor((tick-120)/5)));draw(3,D.executed[t]);const showTruth=document.querySelector('#truth').checked;for(let i=0;i<4;i++){const tr=showTruth?D.truth[i]:D.imagined[i];draw(i+4,tr[h],false,tr.slice(0,h+1));document.querySelector('#p'+(i+4)+' p').innerHTML=`Predicted cost <b>${D.costs[i].toFixed(3)}</b>`+(showTruth?` · true ${D.true_costs[i].toFixed(3)}`:' · lower is better')}
document.querySelector('#phase').textContent=tick<30?'01 / OBSERVE + INFER':tick<120?'02 / BRANCH + IMAGINE':'03 / EXECUTE + REPLAN';document.querySelector('#explanation').textContent=tick<120?'The real scene is frozen. Every branch begins with the exact same learned state. Only the joint action sequence changes.':'The selected first action is applied to the true simulator. Each subsequent step receives a new observation and runs model-only planning again.';slider.value=tick}
setInterval(()=>{if(playing){tick=(tick+1)%200;render()}},100);render();</script></html>'''
