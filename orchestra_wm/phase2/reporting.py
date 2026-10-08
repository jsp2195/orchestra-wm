"""Phase-2 reports derived only from executed measurements; preserves Phase 1."""
from pathlib import Path
import json
import platform
import os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from orchestra_wm.phase2.scenarios import LABELS
from orchestra_wm.phase2.audits import verify_preservation
from orchestra_wm.evaluation.common import write_json

NAMES={'A_original':'A: Original data','B_factorial':'B: Factorial data','C_difference':'C: Factorial + CF','D_pairwise':'D: Pairwise + CF','independent':'Independent','privileged':'Privileged','oracle':'Oracle','heuristic':'Heuristic','noop':'Maintain'}


def table(frame):
    def cell(x):
        if pd.isna(x):return 'N/A'
        return f'{x:.4f}' if isinstance(x,(float,np.floating)) else str(x)
    return '| '+' | '.join(frame.columns)+' |\n|'+'|'.join(['---']*len(frame.columns))+'|\n'+'\n'.join('| '+' | '.join(cell(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None))


def figures(out):
    dest=out/'figures';dest.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140,'savefig.bbox':'tight','figure.constrained_layout.use':True})
    paths=[]
    def save(name):
        path=dest/f'{name}.png';plt.savefig(path);plt.close();paths.append(str(path))
    def bars(series,title,ylabel,name):
        fig,ax=plt.subplots(figsize=(9,4));series.plot.bar(ax=ax,color='#287f9c');ax.set(title=title,ylabel=ylabel);ax.tick_params(axis='x',rotation=25);save(name)
    raw=pd.read_csv(out/'phase1_action_choices_raw.csv')
    frequencies=pd.crosstab(raw.scenario,raw.action,normalize='index').reindex(columns=[-1,0,1],fill_value=0)
    frequencies.columns=['Yield','Maintain','Proceed'];frequencies.plot.bar(figsize=(7,4));plt.ylabel('Command frequency');plt.title('Frozen Phase-1 learned MPC');plt.xticks(rotation=0);save('01_phase1_action_choices')
    sim=pd.read_csv(out/'simulator_factorial_interactions.csv');fig,axs=plt.subplots(1,2,figsize=(9,4))
    for ax,task in zip(axs,['intersection','merge']):
        x=sim[sim.scenario==task].groupby('joint_action').objective.mean().reindex(LABELS).to_numpy().reshape(3,3)
        im=ax.imshow(x,cmap='viridis');ax.set(title=task,xlabel='AV B',ylabel='AV A',xticks=range(3),xticklabels=['Y','M','P'],yticks=range(3),yticklabels=['Y','M','P']);fig.colorbar(im,ax=ax)
        for i in range(3):
            for j in range(3):ax.text(j,i,f'{x[i,j]:.2f}',ha='center',color='white')
    fig.suptitle('True factorial objective (lower is better)');save('02_true_factorial')
    coverage=pd.read_csv(out/'factorial_action_coverage.csv');new=pd.read_csv(out/'factorial_v2_coverage.csv');fig,axs=plt.subplots(1,2,figsize=(10,4))
    for ax,frame,title in [(axs[0],coverage,'Phase 1: relevant pairs'),(axs[1],new[new.split=='train'],'Phase 2: sibling families')]:
        frame.pivot_table(index='joint_action',columns='scenario',values='count',aggfunc='sum').reindex(LABELS).plot.bar(ax=ax);ax.set(title=title,ylabel='Examples');ax.tick_params(axis='x',rotation=0)
    save('03_factorial_coverage')
    sensitivity=pd.read_csv(out/'cross_agent_action_sensitivity.csv');sensitivity.groupby(['model','relation']).cross_effect_m_per_action.mean().unstack().plot.bar(figsize=(8,4));plt.ylabel('Action A influence on B (m / action unit)');plt.title('Pre-training spatial-specificity audit');plt.xticks(rotation=0);save('04_cross_agent_sensitivity')
    prediction=pd.read_csv(out/'interaction_prediction.csv');bars(prediction.groupby('model').pair_position_error.mean().rename(index=NAMES),'Held-out interacting AVs: step 10','Position error (m)','05_interaction_error')
    costs=pd.read_csv(out/'factorial_plan_costs.csv');fig,axs=plt.subplots(2,3,figsize=(11,7))
    for row,task in enumerate(['intersection','merge']):
        subset=costs[(costs.scenario==task)&(costs.plan<9)]
        for col,(name,value) in enumerate([('C_difference','true_cost'),('independent','predicted_cost'),('C_difference','predicted_cost')]):
            x=subset[subset.model==name].groupby('plan')[value].mean().to_numpy().reshape(3,3);ax=axs[row,col];im=ax.imshow(x,cmap='viridis');fig.colorbar(im,ax=ax);ax.set(title=f'{task}: '+('True' if col==0 else NAMES[name]),xticks=range(3),xticklabels=['Y','M','P'],yticks=range(3),yticklabels=['Y','M','P'],xlabel='AV B',ylabel='AV A')
            for i in range(3):
                for j in range(3):ax.text(j,i,f'{x[i,j]:.1f}',ha='center',color='white')
        lo=min(float(ax.images[0].get_array().min()) for ax in axs[row]);hi=max(float(ax.images[0].get_array().max()) for ax in axs[row])
        for ax in axs[row]:ax.images[0].set_clim(lo,hi);ax.set_title(ax.get_title(),fontsize=10)
    save('06_true_predicted_matrices')
    shuffle=pd.read_csv(out/'joint_shuffle.csv');shuffle.groupby(['model','condition']).pair_position_error.mean().unstack().rename(index=NAMES).plot.bar(figsize=(10,4));plt.ylabel('Step-10 pair error (m)');plt.xticks(rotation=25);plt.title('Ordinary versus joint-pairing shuffle');save('07_joint_shuffle')
    subset=costs[(costs.model=='C_difference')&(costs.plan<9)];fig,ax=plt.subplots(figsize=(6,5))
    for task,frame in subset.groupby('scenario'):ax.scatter(frame.true_cost,frame.predicted_cost,s=12,alpha=.45,label=task)
    ax.set(xlabel='True factorial cost',ylabel='C predicted factorial cost',title='Factorial counterfactuals: all seeds and families');ax.legend();save('08_counterfactual_correlation')
    subset=costs[costs.model=='C_difference'];fig,ax=plt.subplots(figsize=(6,5))
    for task,frame in subset.groupby('scenario'):ax.scatter(frame.true_cost,frame.predicted_cost,s=12,alpha=.4,label=task)
    ax.set(xlabel='True cost',ylabel='Predicted cost',title='Interaction plan ranking: 18 candidates');ax.legend();save('09_plan_rank_scatter')
    actions=pd.read_csv(out/'mpc_actions.csv');long=pd.concat([actions[['model','a']].rename(columns={'a':'command'}),actions[['model','b']].rename(columns={'b':'command'})],ignore_index=True);frequency=pd.crosstab(long.model,long.command,normalize='index').reindex(columns=[-1,0,1],fill_value=0).rename(index=NAMES);frequency.columns=['Yield','Maintain','Proceed'];frequency.plot.bar(figsize=(10,4));plt.ylabel('Command frequency');plt.xticks(rotation=25);plt.title('Phase-2 executed MPC commands');save('10_phase2_action_choices')
    mpc=pd.read_csv(out/'mpc_results.csv');mpc.groupby(['model','scenario']).objective.mean().unstack().rename(index=NAMES).plot.bar(figsize=(10,4));plt.ylabel('12-step total objective (lower is better)');plt.xticks(rotation=25);save('11_centralized_independent_objective')
    gain=pd.read_csv(out/'coordination_gain.csv');c=gain[gain.model=='C_difference'];fig,ax=plt.subplots(figsize=(7,4));ax.errorbar(c.task,c['mean'],yerr=[c['mean']-c.ci_low,c.ci_high-c['mean']],fmt='o',capsize=6);ax.axhline(.1,color='red',ls='--',label='Preregistered minimum effect');ax.axhline(0,color='gray',lw=.5);ax.set(ylabel='Independent cost − centralized C cost',title='Coordination gain: training-seed t95 intervals');ax.legend();save('12_coordination_gain_ci')
    bars(mpc.groupby('model').regret_to_oracle.mean().rename(index=NAMES),'Gap to matched oracle MPC','Objective difference','13_oracle_gap')
    memory=pd.read_csv(out/'memory_coordination.csv');memory.groupby(['memory','scenario']).objective.mean().unstack().plot.bar(figsize=(7,4));plt.ylabel('Objective');plt.xticks(rotation=0);plt.title('Hidden-conflict memory: secondary experiment');save('14_hidden_conflict_memory')
    mpc[mpc.seed!=0].groupby(['model','seed']).objective.mean().unstack().rename(index=NAMES).plot.bar(figsize=(10,4));plt.ylabel('Mean episode objective');plt.xticks(rotation=25);plt.title('All three training seeds (no selection)');save('15_three_seed_summary')
    training=pd.read_csv(out/'training.csv');fig,axs=plt.subplots(1,2,figsize=(11,4))
    for name,frame in training.groupby('model'):
        v=frame.groupby('step')[['train_loss','factorial_validation_loss']].mean()
        for ax,key in zip(axs,['train_loss','factorial_validation_loss']):ax.plot(v.index,v[key],label=NAMES[name]);ax.set(xlabel='Optimizer updates',ylabel=key)
    axs[1].legend(fontsize=8);save('16_training_curves')
    return paths


def append_phase2(path,text):
    path=Path(path);prefix=path.read_bytes().split(b'\n<!-- PHASE2_RESULTS -->')[0]
    path.write_bytes(prefix+b'\n<!-- PHASE2_RESULTS -->\n'+text.encode())


def report(cfg,out,runtime=None):
    gates=json.loads((out/'gates.json').read_text());comparison=pd.read_csv(out/'definitive_comparison.csv');gain=pd.read_csv(out/'coordination_gain.csv');mpc=pd.read_csv(out/'mpc_results.csv');training=json.loads((out/'training_metadata.json').read_text())
    h=gates['INT-H'];passed=h['pass'];decision='YES' if passed else 'NO'
    statement=('ORCHESTRA learns interaction-dependent multi-agent dynamics and uses imagined joint futures to improve centralized coordination over independent learned control in the synthetic benchmark.' if passed else 'ORCHESTRA remains a functioning multi-agent world model, but the experiment did not establish a measurable centralized coordination advantage over an independent learned-controller baseline.')
    next_experiment='Keep the same factorial families, architecture, and update budget; replace the centered sibling loss with a loss on the double-centered interaction residual J, and test whether reduced objective-residual error improves plan ranking and the preregistered coordination gain.'
    figures_list=figures(out)
    preserved=verify_preservation(out)
    objective=mpc.groupby('model').objective.mean().to_dict()
    memory_table=pd.read_csv(out/'memory_coordination.csv').groupby('memory')[['objective','conflicts']].mean().reset_index()
    residuals=pd.read_csv(out/'interaction_prediction.csv').groupby('model').interaction_residual_error.mean()
    agreement=pd.read_csv(out/'factorial_residual_agreement.csv').groupby('model').mean(numeric_only=True)
    metrics={'config':cfg,'gates':gates,'comparison':comparison.to_dict('records'),'coordination_gain':gain.to_dict('records'),
       'training':training,'total_training_seconds':sum(x['seconds'] for x in training),'hardware':{'platform':platform.platform(),'cpu':platform.processor() or next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),platform.machine()),'logical_cpus':os.cpu_count(),'python':platform.python_version(),'torch':torch.__version__,'cuda_available':torch.cuda.is_available()},
       'runtime':runtime,'phase1_preserved_files':preserved,'figures':figures_list,'centralized_benefit':decision,
       'generalization':{'status':'UNTESTED','reason':'INT-H failed; preregistered OOD gate closed'} if not passed else {'status':'PENDING'},'next_experiment':next_experiment}
    write_json(out/'metrics.json',metrics)
    gates_table=pd.DataFrame([{'Gate':key,'Status':'PASS' if v['pass'] else 'FAIL','Measurements':json.dumps({k:x for k,x in v.items() if k!='pass'},sort_keys=True)} for key,v in gates.items()])
    display=comparison.copy();display['model']=display.model.map(NAMES);display.columns=['Model','Interaction Error (m)','Joint CF Corr.','Joint Shuffle Gap (m)','Plan Rank Spearman','MPC Objective','Coordination Gain']
    # Do not mistake time-summed objective residuals for position errors or shuffle gaps.
    text=f'''# Phase 2 — Joint Interaction and Coordination

{statement}

**CENTRALIZED ORCHESTRATION BENEFIT: {decision}.** C coordination gain = {h['mean']:.6f}, SD {h['sd']:.6f}, SE {h['se']:.6f}; Student-t 95% CI [{h['ci_low']:.6f}, {h['ci_high']:.6f}]. The unchanged acceptance criterion is lower bound >0.1. Independent MPC objective {objective['independent']:.6f}; C objective {objective['C_difference']:.6f}; oracle objective {objective['oracle']:.6f} (lower is better). The oracle is finite-budget CEM, not a proof of global optimality.

## Definitive comparison

{table(display)}

Interaction Error is final-step-10 Euclidean position error averaged over the two interacting AVs, nine siblings, 16 held-out families and three training seeds. Joint CF Corr. is the mean within-family nine-plan objective Pearson. Joint shuffle gap is absolute pair-error degradation in metres; INT-E instead uses each seed's ratio of mean errors. Oracle forecast/rank entries are self-comparison references, not learned predictions; its shuffle gap is undefined. A is the original-data Phase-1 architecture retrained for the matched Phase-2 budget; the frozen Phase-1 checkpoint appears only in audits. D and privileged are secondary diagnostics; neither can replace C.

## Preregistered gates

{table(gates_table)}

## Paired primary result by task

{table(gain[gain.model=='C_difference'].drop(columns=['model']))}

Confidence intervals use three training-seed means (df=2), not 48 independent trained models. The secondary hierarchical bootstrap resamples training seeds and matched episode pairs. No seed or checkpoint was selected after evaluation. All models receive 240 updates, batch 18, ten-step autonomous rollouts; factorial models use 75% sibling batches and 25% natural batches. A uses original data only. Training curves remain noisy, and this fixed integration-scale budget does not establish convergence.

## Audits and data

Phase-1 action frequency was 100% proceed in both audited tasks, including the recorded conflict/relevance strata. Numeric reproduction differences are zero for all three audited tables. {preserved} original artifact hashes remain unchanged; README and claim-ledger Phase-1 prefixes are checked separately from their appended Phase-2 sections. Simulator interaction RMS is {gates['INT-A']['interaction_rms']:.6f}; fraction above 0.05 is {gates['INT-A']['fraction_above_005']:.6f}. Each of 24 training and six validation families has nine siblings, giving 270 episodes / 3240 future transitions. All 16 test families are disjoint. Manifests record seeds, state hashes, config, and compressed data hash. Hidden states and sibling metadata remain targets or diagnostics, not primary model inputs.

D was triggered before Phase-2 training: frozen model mean cross-agent influence was 0.718145 m/action on interacting targets versus 0.724232 on distant targets. This audit did not demonstrate spatial specificity; it is not a precise estimate of structural impossibility. The operational direction rule is recorded in PHASE2_EXECUTION_NOTES.md. D adds a small relative-geometry action-message channel.

The independent baseline has separate per-agent dynamics and an additive cost head. Both learned controllers optimize joint candidate tensors, but the independent model cannot represent cross-agent effects. CEM uses identical balanced 18-plan initial pools, two iterations, horizons, weights and proposal randomness. Learned candidate scores use only imagination; real simulator clones belong solely to oracle and post-hoc evaluation. No action-label penalties were added.

## Scientific interpretation

Trajectory fidelity, counterfactual fidelity, and decision fidelity are distinct. Positive counterfactual correlation can reflect correct marginal action effects while missing nonlinear interaction residuals. Joint-pairing shuffle also harms an independent action-conditioned predictor and alone does not prove joint reasoning. The interaction-residual errors, rank results and closed-loop paired costs are therefore reported separately. The preregistered C result alone adjudicates centralized benefit. C's objective interaction-residual RMSE is {residuals['C_difference']:.6f}, versus {residuals['independent']:.6f} for the structurally additive independent model. C predicts a mean objective interaction RMS of {agreement.loc['C_difference','predicted_residual_rms']:.6f} against true {agreement.loc['C_difference','true_residual_rms']:.6f}, and its residual Pearson is {agreement.loc['C_difference','residual_pearson']:.6f} (secondary diagnostics). The independent residual RMS of about 6e−8 is floating-point roundoff, so its residual correlation is not scientifically interpretable. The centered sibling difference loss can primarily fit marginal effects without recovering the double-centered non-additive residual. This is the specific failure targeted by the next experiment.

OOD is {'gated pending limited evaluation' if passed else 'UNTESTED: INT-H failed, so no broad generalization experiments were run'}. Hidden-conflict memory is a secondary executed comparison in memory_coordination.csv; it cannot override INT-H. The four additional cases per seed give the following means (synthetic conflicts are cumulative simulator event counts):

{table(memory_table)}
 No real-world safety or infrastructure-control claim is made.

## Reproduction

From the repository root, with preserved Phase-1 local dataset/checkpoints available:

```sh
uv sync --frozen
uv run pytest -q
uv run python scripts/run_phase2.py --config configs/phase2.yaml
```

A fresh clone needs the original ignored Phase-1 dataset/checkpoints and TensorBoard files restored from the original workspace/archive before exact preservation verification can pass. The preservation manifest includes timestamped logs that cannot be reconstructed byte-for-byte by retraining. Do not run the Phase-1 pipeline over the preserved outputs or replace the manifest to bypass a mismatch. This is an archival portability limitation of the existing WIP manifest; the current workspace contains and verifies all originals. Checkpoints and dataset arrays are intentionally not in Git. Phase-2 stages reuse matching checkpoints and immutable dataset manifests. All 18 trained models, optimizer steps, parameter counts, hardware versions, stage timing and metrics are recorded in training_metadata.json / metrics.json. Aggregate CSVs retain every seed; no best-seed selection is supported.

## Figures

'''+''.join(f'- [{Path(p).stem}](figures/{Path(p).name})\n' for p in figures_list)+f'\nNEXT EXPERIMENT: {next_experiment}\n'
    (out/'EXPERIMENT_SUMMARY.md').write_text(text)
    append_phase2('README.md',text.replace('(figures/','(outputs/phase2/figures/'))
    claim_defs=[('joint interactions learned','INT-C'),('factorial counterfactuals learned','INT-D'),('joint pairing matters','INT-E'),('plan ranking works in interaction-heavy states','INT-F'),('planner avoids policy collapse','INT-G'),('centralized coordination beats independent learned MPC','INT-H')]
    claims=[]
    for claim,key in claim_defs:
        v=gates[key];status='SUPPORTED' if v['pass'] else 'NOT SUPPORTED'
        if key in ['INT-C','INT-D','INT-E'] and v['pass']:status='PARTIALLY SUPPORTED'
        claims.append({'Claim':claim,'Metric/gate':key,'Actual result':json.dumps(v,sort_keys=True),'Config/seeds':'phase2.yaml / 101,202,303','Status':status})
    ledger='# Phase 2 — measured claims\n\nPhase-1 entries above are preserved. C is primary. Counterfactual correlation and pairing degradation can occur without non-additive reasoning; passing those directional gates supports narrower claims only.\n\n'+table(pd.DataFrame(claims))+'\n\n'+statement+'\n\nGeneralization: UNTESTED unless INT-H passes. See [Phase-2 experiment report](../outputs/phase2/EXPERIMENT_SUMMARY.md) for all measurements.\n'
    append_phase2('docs/CLAIM_LEDGER.md',ledger)
    verify_preservation(out)
    return metrics
