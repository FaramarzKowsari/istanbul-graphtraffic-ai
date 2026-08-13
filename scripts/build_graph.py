#!/usr/bin/env python
import argparse
from pathlib import Path
import pandas as pd
from graphtraffic.data.graph import sensor_metadata, knn_adjacency, save_graph
p=argparse.ArgumentParser(); p.add_argument("--traffic",default="data/processed/traffic.csv"); p.add_argument("--output",default="data/processed/graph.npz"); p.add_argument("--mode",choices=["knn","osm"],default="knn"); p.add_argument("--k",type=int,default=4); a=p.parse_args(); df=pd.read_csv(a.traffic); meta=sensor_metadata(df)
if a.mode=="knn": adj=knn_adjacency(meta,k=a.k)
else:
 from graphtraffic.data.osm_topology import osm_sensor_adjacency
 adj=osm_sensor_adjacency(meta)
Path(a.output).parent.mkdir(parents=True,exist_ok=True); save_graph(a.output,adj,meta,a.mode); print(f"Graph: {len(meta)} nodes; {(adj>0).sum()} weighted links -> {a.output}")
