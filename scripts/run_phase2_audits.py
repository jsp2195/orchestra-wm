from pathlib import Path
from orchestra_wm.utils.config import load_config
from orchestra_wm.phase2.audits import preserve_manifest,reproduce,action_choices,coverage,factorial_audit
if __name__=='__main__':
 c=load_config('configs/phase2.yaml');o=Path(c['output']);o.mkdir(exist_ok=True,parents=True)
 preserve_manifest(o);m=reproduce(o);action_choices(m,o);coverage(o);print(factorial_audit(c,m,o).groupby('scenario').interaction_rms.mean())
