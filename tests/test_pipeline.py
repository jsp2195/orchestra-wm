import subprocess
import sys
from pathlib import Path
import pytest
from orchestra_wm.pipeline import validate_artifacts

def test_pipeline_cli_and_missing_outputs(tmp_path):
    result=subprocess.run([sys.executable,'scripts/run_pipeline.py','--help'],capture_output=True,text=True)
    assert result.returncode==0 and '--clean' in result.stdout
    with pytest.raises(AssertionError): validate_artifacts(tmp_path)
