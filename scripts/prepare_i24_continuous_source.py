"""Stream one bounded regional westbound subset; never extract the full archive."""
import argparse
import json
from orchestra_wm.i24_continuous.source import extract_interval
from orchestra_wm.i24_continuous.drive import TransferBlocked


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',required=True)
    p.add_argument('--destination',required=True)
    p.add_argument('--start',type=float,required=True)
    p.add_argument('--seconds',type=int,default=600)
    p.add_argument('--x-min-ft',type=float,default=316800.)
    p.add_argument('--x-max-ft',type=float,default=320080.)
    a=p.parse_args()
    try:
        result=extract_interval(a.archive,a.destination,start=a.start,seconds=a.seconds,
                                x_min=a.x_min_ft,x_max=a.x_max_ft)
    except TransferBlocked as e:
        print('BLOCKED:',str(e));return 2
    print(json.dumps(result,indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
