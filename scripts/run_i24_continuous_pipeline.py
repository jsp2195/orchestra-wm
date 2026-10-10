"""Phase-4 access/QC entry point. Downstream science remains explicitly gated."""
import argparse
import json
from pathlib import Path
import yaml
from orchestra_wm.i24_continuous.access import inspect_local, AccessBlocked


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--report', default='outputs/i24_continuous/local_import_status.json')
    a = p.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    try:
        result = inspect_local(a.data_root, cfg['access'])
    except (AccessBlocked, ValueError, KeyError, OSError, TypeError) as e:
        result = {'source':'ORIGINAL_I24_MOTION','status':'BLOCKED',
                  'reason':str(e) if isinstance(e, AccessBlocked) else 'Invalid or unreadable source/metadata; inspect locally without logging credentials',
                  'training':'NOT_RUN'}
    out = Path(a.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Preserve earlier reports; changed inputs need an explicitly new report path.
    payload = json.dumps(result, indent=2, allow_nan=False)+'\n'
    if out.exists() and out.read_text() != payload:
        raise SystemExit('Report exists with different inputs/results; select a new --report path')
    tmp = out.with_suffix('.tmp'); tmp.write_text(payload); tmp.replace(out)
    print(result['status'])
    print('No continuous model training executed; see docs/I24_CONTINUOUS_ACQUISITION.md')
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
