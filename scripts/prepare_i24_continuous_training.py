"""Build authentic, bounded, track-disjoint continuous pilot data; no training."""
import argparse
import json
from orchestra_wm.i24_continuous.data import prepare


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--max-agents',type=int,default=256)
    a=p.parse_args()
    result=prepare(a.data_root,a.output,a.max_agents)
    print(json.dumps({k:v for k,v in result.items() if k not in ('artifacts','scenes','rejected_windows')},indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
