from pathlib import Path
import json
from orchestra_wm.utils.config import load_config
from orchestra_wm.utils.seed import seed_everything
from orchestra_wm.phase2.evaluation import evaluate_models,evaluate_mpc,evaluate_memory
from orchestra_wm.phase2.statistics import compute_statistics
if __name__=='__main__':
 c=load_config('configs/phase2.yaml');o=Path(c['output']);seed_everything(c['seed'],c['threads'])
 names=['A_original','B_factorial','C_difference','independent','privileged']
 if json.loads((o/'architecture_decision.json').read_text())['explicit_variant_eligible']:names.append('D_pairwise')
 evaluate_models(c,o,names);evaluate_mpc(c,o,names)
 if not (o/'memory_coordination.csv').exists():evaluate_memory(c,o)
 print(json.dumps(compute_statistics(c,o),indent=2))
