"""Create authentic source plots and optional exact-checkpoint GIF/HTML."""
import argparse
from orchestra_wm.i24_continuous.visuals import source_figures,forecast_demo

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',required=True);p.add_argument('--output',required=True);p.add_argument('--with-forecast',action='store_true');a=p.parse_args()
    print(source_figures(a.data_root,a.output))
    if a.with_forecast:print(forecast_demo(a.output))
