import json
from orchestra_wm.i24_phase4c.diagnostics import diagnose
print(json.dumps(diagnose('outputs/i24_phase4b/development_v1',
                         'outputs/i24_phase4b/matched_macro_seed101_u300',
                         'outputs/i24_phase4c/matched_macro_seed101_u300/diagnostics'),indent=2))
