#!/usr/bin/env python
import argparse, json
import pandas as pd
from graphtraffic.data.schema import infer_mapping
p=argparse.ArgumentParser(); p.add_argument("path"); a=p.parse_args(); df=pd.read_csv(a.path,nrows=1000)
print("Columns:",list(df.columns)); print("Inferred mapping:",json.dumps(infer_mapping(df),indent=2,ensure_ascii=False)); print(df.head().to_string())
