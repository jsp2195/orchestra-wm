import ast
import json
from pathlib import Path


def test_notebook_is_clean_python_and_uses_only_builtin_drive_authorization():
    nb=json.loads(Path('notebooks/ORCHESTRA_Phase4E_Colab.ipynb').read_text())
    assert nb['nbformat']==4
    cells=[c for c in nb['cells'] if c['cell_type']=='code']
    text='\n'.join(''.join(c['source']) for c in cells)
    for c in cells:
        assert c['outputs']==[] and c['execution_count'] is None
        ast.parse(''.join(c['source']))
    assert text.count('drive.mount(')==1
    assert 'GOOGLE_DRIVE_ACCESS_TOKEN' not in text and 'CLIENT_SECRET' not in text
    assert 'RUN_STAGE = "all"' in text
    assert 'uv' in text and "'--frozen'" in text and "'3.11'" in text
    assert 'run_i24_phase4e_colab.py' in text
    assert '2cf2330533f6630425116b636f1d7ba878862653' not in text, 'Notebook must pin the new implementation, not the old source commit'
