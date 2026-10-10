"""Authenticated archive transfer; deliberately no automatic extraction/training."""
import argparse
import json
from pathlib import Path
from orchestra_wm.i24_continuous.drive import run, TransferBlocked


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',default='data/i24_continuous/raw')
    p.add_argument('--max-download-bytes',type=int,default=6_300_000_000)
    a=p.parse_args()
    try:
        results=run(a.data_root,a.max_download_bytes)
    except TransferBlocked as e:
        print('BLOCKED:',str(e));return 2
    except Exception:
        print('BLOCKED: transfer interrupted or response invalid; bytes retained. No request secrets logged.');return 2
    out=Path(a.data_root)/'DRIVE_ARCHIVE_MANIFEST.json'
    tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(results,indent=2)+'\n');tmp.replace(out)
    print('Archives verified and ZIP members inspected. Source schema/chronology review required before extraction and training.')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
