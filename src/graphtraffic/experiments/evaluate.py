from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from graphtraffic.data.features import add_calendar_features
from graphtraffic.data.windows import frame_to_tensor, TrafficWindowDataset
from graphtraffic.data.splits import chronological_boundaries
from graphtraffic.data.graph import normalize_adjacency
from graphtraffic.models.factory import build_model
from graphtraffic.metrics import mae, rmse, mape, r2, interval_coverage, interval_width
from graphtraffic.experiments.failure import apply_sensor_failure
from graphtraffic.utils.io import save_json


def _predict(model, loader, A, failure=0.0, structured=False):
    ys=[]; ps=[]
    model.eval()
    with torch.no_grad():
        for x,y in loader:
            if failure: x,_=apply_sensor_failure(x,failure,seed=123,structured=structured)
            p=model(x,A); ys.append(y.numpy()); ps.append(p.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def run_evaluation(cfg: dict):
    out=Path(cfg["output"]["dir"]); ckpt=torch.load(out/"model.pt",map_location="cpu",weights_only=False)
    df=add_calendar_features(pd.read_csv(cfg["data"]["path"],parse_dates=["timestamp"]))
    bundle=frame_to_tensor(df,cfg["data"]["features"],cfg["data"]["target"])
    tr_end,va_end=chronological_boundaries(len(bundle.timestamps),cfg["data"]["train_fraction"],cfg["data"]["val_fraction"])
    ds=TrafficWindowDataset(bundle,cfg["data"]["history"],cfg["data"]["horizons"],va_end,len(bundle.timestamps))
    loader=DataLoader(ds,batch_size=cfg["training"]["batch_size"],shuffle=False)
    graph=np.load(cfg["data"]["graph_path"],allow_pickle=True); A=torch.tensor(normalize_adjacency(graph["adjacency"]),dtype=torch.float32)
    mcfg=cfg["model"]
    model=build_model(mcfg["name"],n_nodes=len(bundle.sensors),n_features=len(cfg["data"]["features"]),history=cfg["data"]["history"],horizons=len(cfg["data"]["horizons"]),hidden_dim=mcfg.get("hidden_dim",64),heads=mcfg.get("heads",4),dropout=mcfg.get("dropout",0.1),quantiles=mcfg.get("quantiles",[0.1,0.5,0.9])); model.load_state_dict(ckpt["model_state"])
    y,p=_predict(model,loader,A)
    target_transform = ckpt.get("target_transform")
    if target_transform and target_transform.get("name") == "zscore_train":
        p = p * float(target_transform["std"]) + float(target_transform["mean"])
    if p.ndim==4:
        qs=list(mcfg.get("quantiles",[0.1,0.5,0.9])); mid=int(np.argmin(np.abs(np.asarray(qs)-0.5))); point=p[...,mid]
    else: qs=None; point=p
    result={"dataset_label":"USER-PROVIDED OR SYNTHETIC—CHECK provenance.json","overall":{"mae":mae(y,point),"rmse":rmse(y,point),"mape":mape(y,point),"r2":r2(y,point)},"horizons":{}}
    for i,h in enumerate(cfg["data"]["horizons"]):
        row={"mae":mae(y[:,i],point[:,i]),"rmse":rmse(y[:,i],point[:,i]),"mape":mape(y[:,i],point[:,i]),"r2":r2(y[:,i],point[:,i])}
        if p.ndim==4 and len(qs)>=3:
            row["coverage"]=interval_coverage(y[:,i],p[:,i,:,0],p[:,i,:,-1]); row["interval_width"]=interval_width(p[:,i,:,0],p[:,i,:,-1])
        result["horizons"][str(h)]=row
    failures={}
    for frac in (0.1,0.2,0.3):
        fy,fp=_predict(model,loader,A,frac,False); fp = fp * float(target_transform["std"]) + float(target_transform["mean"]) if target_transform and target_transform.get("name") == "zscore_train" else fp; fpoint=fp[...,mid] if fp.ndim==4 else fp
        failures[str(frac)]={"random_mae":mae(fy,fpoint)}
        fy,fp=_predict(model,loader,A,frac,True); fp = fp * float(target_transform["std"]) + float(target_transform["mean"]) if target_transform and target_transform.get("name") == "zscore_train" else fp; fpoint=fp[...,mid] if fp.ndim==4 else fp
        failures[str(frac)]["structured_mae"]=mae(fy,fpoint)
    result["sensor_failure"]=failures
    save_json(result,out/"metrics.json")
    return result
