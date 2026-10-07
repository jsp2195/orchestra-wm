from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from orchestra_wm.sim.renderer import draw_scene
from orchestra_wm.envs.traffic_env import TrafficEnv


def generate_plots(cfg,out):
    directory=out/'figures';directory.mkdir(exist_ok=True)
    plt.rcParams.update({'figure.dpi':130,'axes.spines.top':False,'axes.spines.right':False,'font.size':9})
    def save(name):
        plt.tight_layout();plt.savefig(directory/f'{name}.png',bbox_inches='tight');plt.close()
    def line(frame,x,y,group,title,name):
        fig,ax=plt.subplots(figsize=(7,4))
        for label,g in frame.groupby(group):
            a=g.groupby(x)[y].agg(['mean','std','count']);err=a['std'].fillna(0)/np.sqrt(a['count'])
            ax.errorbar(a.index,a['mean'],yerr=err,label=label,marker='o',ms=3,lw=1.2)
        ax.set(xlabel=x.replace('_',' '),ylabel=y.replace('_',' '),title=title);ax.legend(fontsize=7,ncol=2);save(name)
    def bars(frame,x,y,title,name):
        fig,ax=plt.subplots(figsize=(8,4));a=frame.groupby(x)[y].agg(['mean','std','count'])
        ax.bar(a.index,a['mean'],yerr=a['std'].fillna(0)/np.sqrt(a['count']),color='#337b94',capsize=3)
        ax.set(ylabel=y.replace('_',' '),title=title);ax.tick_params(axis='x',rotation=30,labelsize=7);save(name)
    fig,axes=plt.subplots(2,2,figsize=(10,9),facecolor='#101827')
    for ax,family in zip(axes.flat,['intersection','merge','corridor','grid']):
        env=TrafficEnv(cfg,family,17);o,_=env.reset(seed=17)
        draw_scene(ax,env.road,env.state_array(),o['control_mask'],title=family)
    save('scenario_overview')
    fig,axes=plt.subplots(1,2,figsize=(10,5),facecolor='#101827');env=TrafficEnv(cfg,seed=17);o,_=env.reset(seed=17)
    draw_scene(axes[0],env.road,env.state_array(),o['control_mask'],o['mask'],'True state • orange = hidden')
    draw_scene(axes[1],env.road,o['agents'][:,:6],o['control_mask'],title='Partial structured observations');save('partial_observation')
    training=pd.read_csv(out/'training.csv')
    fig,ax=plt.subplots(figsize=(8,4))
    for name,g in training.groupby('variant'):
        ax.plot(g.epoch,g.loss,label=f'{name} train');ax.plot(g.epoch,g.validation_loss,ls='--',label=f'{name} validation')
    ax.set(xlabel='epoch',ylabel='multi-step loss',title='From-scratch training • dashed = held-out episodes');ax.legend(fontsize=6,ncol=2);save('training_curves')
    pred=pd.read_csv(out/'prediction.csv');line(pred,'horizon','position_error','model','Autonomous forecast position error (m) ± SE across seeds','rollout_error_vs_horizon')
    shuffle=pd.read_csv(out/'action_shuffle.csv');s=shuffle[shuffle.model=='orchestra']
    line(s,'horizon','position_error','condition','Same contexts and targets; only hypothetical actions change','correct_vs_shuffled_actions')
    inter=pd.read_csv(out/'interactions.csv')
    line(inter,'plan','predicted_cost','model','Joint intervention costs • compare simulator reference below','joint_action_interaction')
    fig,ax=plt.subplots(figsize=(7,4))
    for label,g in inter.groupby('model'): ax.scatter(g.true_cost,g.predicted_cost,label=label)
    ax.set(xlabel='true joint cost',ylabel='predicted joint cost',title='Joint-action outcomes');ax.legend();save('joint_action_costs')
    memory=pd.read_csv(out/'memory.csv');line(memory,'occlusion_steps','position_error','model','Forced hidden vehicle: prediction before reacquisition','hidden_agent_memory')
    costs=pd.read_csv(out/'plan_costs.csv');c=costs[costs.model=='orchestra']
    fig,ax=plt.subplots(figsize=(6,5))
    for family,g in c.groupby('scenario'):ax.scatter(g.true_cost,g.predicted_cost,label=family,alpha=.7,s=18)
    ax.set(xlabel='true simulator cost',ylabel='ORCHESTRA predicted cost',title='Candidate plans from matched inferred states');ax.legend();save('planning_rank_scatter')
    example=np.load(out/'counterfactual_example.npz')
    true=example['true_distances'];predicted=example['model_distances'];tri=np.triu_indices(len(true),1)
    fig,ax=plt.subplots(figsize=(5,4));ax.scatter(true[tri],predicted[tri],s=16,alpha=.7)
    ax.set(xlabel='true pairwise outcome distance (m)',ylabel='predicted pairwise distance (m)',title='Counterfactual outcome geometry');save('counterfactual_scatter')
    for name,array in [('model',predicted),('true',true)]:
        fig,ax=plt.subplots(figsize=(5,4));im=ax.imshow(array,cmap='viridis');fig.colorbar(im,ax=ax,label='joint outcome distance (m)')
        ax.set(xlabel='candidate plan',ylabel='candidate plan',title=f'Counterfactual distances: {name}');save(f'counterfactual_matrix_{name}')
    rank=pd.read_csv(out/'rank_fidelity.csv');bars(rank,'model','spearman','Plan cost ranking: mean ± SE across paired contexts','planning_rank_fidelity')
    bars(rank,'model','regret','Regret of model-selected candidate (lower is better)','planning_regret')
    planning=pd.read_csv(out/'planning.csv')
    bars(planning[planning.controller.isin(['independent','orchestra'])],'controller','objective','Independent local control vs shared model MPC','centralized_vs_independent')
    bars(planning[planning.controller.isin(['random','noop','heuristic','orchestra','privileged','oracle'])],'controller','objective','True simulator outcomes under feedback control','orchestra_vs_oracle')
    for family in cfg['scenarios']:
        bars(planning[planning.scenario==family],'controller','objective',f'{family.title()} coordination objective',f'{family}_task_summary')
    general=pd.read_csv(out/'generalization.csv')
    bars(general[general.condition.str.contains('OOD|ID')],'condition','position_error','Held-out conditions, same checkpoint','generalization_breakdown')
    bars(general[~general.condition.str.contains('OOD|stopped|blocked')],'condition','position_error','Information withholding, same checkpoint and simulator seeds','sensing_ablation')
    scaling=pd.read_csv(out/'scaling.csv');scaling['model']='orchestra'
    line(scaling,'agents','position_error','model','Variable-agent scaling: forecast error','performance_vs_agents')
    line(scaling,'agents','planner_ms','model','Variable-agent scaling: planning latency on this machine','scaling_latency')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,arr,label in [(axes[0],example['predicted'],'Imagined'),(axes[1],example['truth'],'True simulator')]:
        for i in range(arr.shape[2]):ax.plot(arr[0,:,i,0]*100,arr[0,:,i,1]*100,marker='.',lw=1)
        ax.set(title=label,xlabel='x (m)',ylabel='y (m)',aspect='equal')
    save('rollout_prediction')
    bars(pred[pred.horizon==10],'model','position_error','Performance summary: ten-step trajectory error','performance_summary')
    return sorted(p.name for p in directory.glob('*.png'))
