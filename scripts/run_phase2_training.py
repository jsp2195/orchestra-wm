from pathlib import Path
import json
import pandas as pd
from orchestra_wm.utils.config import load_config
from orchestra_wm.phase2.audits import verify_preservation
from orchestra_wm.phase2.data import collect_factorial
from orchestra_wm.phase2.training import train_phase2
if __name__=='__main__':
 c=load_config('configs/phase2.yaml');o=Path(c['output']);verify_preservation(o)
 strength=pd.read_csv(o/'simulator_interaction_strength.csv')
 assert strength.interaction_rms.mean()>.05 and (strength.interaction_rms>.05).mean()>=1/3
 decision=json.loads((o/'architecture_decision.json').read_text())
 collect_factorial(c,o)
 train_phase2(c,o,decision['explicit_variant_eligible'])
