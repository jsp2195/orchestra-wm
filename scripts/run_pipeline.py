import argparse
from orchestra_wm.pipeline import run
if __name__=='__main__':
    p=argparse.ArgumentParser(description='Run the complete measured synthetic ORCHESTRA-WM experiment')
    p.add_argument('--config',default='configs/smoke.yaml');p.add_argument('--clean',action='store_true',help='Archive existing output before the run')
    p.add_argument('--output',help='Override the output directory')
    a=p.parse_args();run(a.config,a.clean,a.output)
