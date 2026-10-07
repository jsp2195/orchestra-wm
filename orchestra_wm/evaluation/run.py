from orchestra_wm.models.training import load_model,VARIANTS
from orchestra_wm.evaluation.prediction import evaluate_prediction
from orchestra_wm.evaluation.counterfactuals import evaluate_counterfactuals
from orchestra_wm.evaluation.interactions import evaluate_interactions
from orchestra_wm.evaluation.memory import evaluate_memory
from orchestra_wm.evaluation.planning import evaluate_planning
from orchestra_wm.evaluation.generalization import evaluate_conditions,evaluate_scaling
from orchestra_wm.utils.seed import device_for,seed_everything

def evaluate(cfg,out):
    seed_everything(cfg['seed'],cfg['threads'])
    models={name:load_model(out/'checkpoints'/f'{name}.pt',device_for(cfg['device'])) for name in VARIANTS}
    for name,fn in [('prediction',evaluate_prediction),('interactions',evaluate_interactions),('memory',evaluate_memory),
                    ('counterfactuals',evaluate_counterfactuals),('planning',evaluate_planning)]:
        print('Evaluating',name,flush=True);fn(cfg,models,out)
    print('Evaluating OOD and sensing',flush=True);evaluate_conditions(cfg,models['orchestra'],out)
    print('Evaluating scaling',flush=True);evaluate_scaling(cfg,models['orchestra'],out)
