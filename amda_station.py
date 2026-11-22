"""Run AMDA on a user-provided station/reference CSV (does not download data).

CSV columns: date,station,reference. Dates ISO YYYY-MM-DD, temperature in Celsius.
Run: python amda_station.py --input your_data.csv --output-dir station_results
"""
from pathlib import Path
import argparse
import json
import pandas as pd
from amda import analyze

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input',required=True)
parser.add_argument('--output-dir',default='station_results')
parser.add_argument('--k',type=float,default=4.5)
parser.add_argument('--c',type=float,default=3.0)
parser.add_argument('--delta-min',type=float,default=0.5)
parser.add_argument('--h',type=int,choices=[5,11,21],default=None,
                    help='Omit for rolling-origin window selection; use a number to force a window.')
args=parser.parse_args()
out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
df=pd.read_csv(args.input)
result,daily,monthly=analyze(df,k=args.k,c=args.c,delta_min=args.delta_min,h=args.h)
daily.to_csv(out/'daily_flags.csv',index_label='date')
monthly.to_csv(out/'monthly_adjusted.csv',index_label='date')
(out/'summary.json').write_text(json.dumps(result,indent=2,default=float),encoding='utf8')
print(json.dumps(result,indent=2,default=float))
