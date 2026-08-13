#!/usr/bin/env python
import argparse
from pathlib import Path
from graphtraffic.data.synthetic import generate_synthetic
from graphtraffic.data.features import add_calendar_features

p=argparse.ArgumentParser(); p.add_argument("--hours",type=int,default=336); p.add_argument("--sensors",type=int,default=48); p.add_argument("--seed",type=int,default=42); p.add_argument("--output",default="data/processed/traffic.csv")
a=p.parse_args(); out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
df=generate_synthetic(a.hours,a.sensors,a.seed); df.to_csv(out,index=False)
Path("data/processed/provenance.json").write_text('{"dataset":"synthetic","purpose":"pipeline smoke test; not a scientific result"}\n',encoding="utf-8")
print(f"Wrote {len(df):,} rows to {out}")
