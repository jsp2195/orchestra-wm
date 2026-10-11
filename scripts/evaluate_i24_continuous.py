"""Matched final-checkpoint continuous-data pilot evaluation."""
import argparse
import json
from orchestra_wm.i24_continuous.evaluation import evaluate

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-output',required=True);p.add_argument('--output',required=True)
    p.add_argument('--samples',type=int,default=8);a=p.parse_args()
    print(json.dumps(evaluate(a.data_output,a.output,a.samples),indent=2))
