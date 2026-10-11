import argparse
import json
from pathlib import Path
from orchestra_wm.i24_phase4c.evaluation import evaluate, freeze_validation

p = argparse.ArgumentParser()
p.add_argument('--data-output', default='outputs/i24_phase4b/development_v1')
p.add_argument('--training-output', default='outputs/i24_phase4b/matched_macro_seed101_u300')
p.add_argument('--output', default='outputs/i24_phase4c/matched_macro_seed101_u300')
p.add_argument('--split', choices=['validation','development_test'], default='validation')
p.add_argument('--freeze-validation', action='store_true')
a = p.parse_args()
if a.freeze_validation:
    freeze_validation(a.output)
else:
    print(json.dumps(evaluate(a.data_output, a.training_output, Path(a.output)/a.split, a.split), indent=2))
