#!/usr/bin/env python
import argparse, hashlib, json
from pathlib import Path
import pandas as pd
from graphtraffic.data.schema import standardize_frame
from graphtraffic.data.features import add_calendar_features
p=argparse.ArgumentParser(); p.add_argument("--input",required=True); p.add_argument("--output",default="data/processed/traffic.csv"); p.add_argument("--source-url",default="https://data.ibb.gov.tr/en/dataset/hourly-traffic-density-data-set")
a=p.parse_args(); src=Path(a.input); raw=src.read_bytes(); sha=hashlib.sha256(raw).hexdigest(); df=standardize_frame(pd.read_csv(src)); df=add_calendar_features(df); out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); df.to_csv(out,index=False)
prov={"dataset":"IBB hourly traffic (user-supplied raw file)","source_url":a.source_url,"raw_filename":src.name,"sha256":sha,"rows":len(df),"sensors":int(df.sensor_id.nunique()),"start":str(df.timestamp.min()),"end":str(df.timestamp.max())}; Path("data/processed/provenance.json").write_text(json.dumps(prov,indent=2),encoding="utf-8"); print(json.dumps(prov,indent=2))
