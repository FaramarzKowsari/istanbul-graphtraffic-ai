from __future__ import annotations
from pathlib import Path
import copy
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from graphtraffic.data.features import add_calendar_features
from graphtraffic.data.windows import frame_to_tensor, TrafficWindowDataset
from graphtraffic.data.splits import chronological_boundaries
from graphtraffic.data.graph import normalize_adjacency
from graphtraffic.models.factory import build_model
from graphtraffic.losses import pinball_loss
from graphtraffic.utils.seed import seed_everything
from graphtraffic.utils.io import save_json


def run_training(cfg: dict):
    seed_everything(int(cfg["seed"]))
    df = add_calendar_features(pd.read_csv(cfg["data"]["path"], parse_dates=["timestamp"]))
    features = cfg["data"]["features"]
    bundle = frame_to_tensor(df, features, cfg["data"]["target"])
    tr_end, va_end = chronological_boundaries(len(bundle.timestamps), cfg["data"]["train_fraction"], cfg["data"]["val_fraction"])
    target_transform = None
    if cfg["data"].get("target_normalization") == "zscore_train":
        train_target = bundle.target[:tr_end]
        mu = float(train_target.mean())
        sigma = float(train_target.std()) or 1.0
        bundle.target = ((bundle.target - mu) / sigma).astype(np.float32)
        target_transform = {"name": "zscore_train", "mean": mu, "std": sigma}
    ds_train = TrafficWindowDataset(bundle, cfg["data"]["history"], cfg["data"]["horizons"], 0, tr_end)
    ds_val = TrafficWindowDataset(bundle, cfg["data"]["history"], cfg["data"]["horizons"], tr_end, va_end)
    if not len(ds_train) or not len(ds_val):
        raise ValueError("Dataset is too short for requested history/horizons and splits")
    graph = np.load(cfg["data"]["graph_path"], allow_pickle=True)
    sensor_ids = graph["sensor_ids"].astype(str).tolist()
    if sensor_ids != bundle.sensors:
        raise ValueError("Sensor ordering mismatch between traffic tensor and graph")
    A = torch.tensor(normalize_adjacency(graph["adjacency"]), dtype=torch.float32)

    mcfg=cfg["model"]
    model=build_model(mcfg["name"], n_nodes=len(bundle.sensors), n_features=len(features), history=cfg["data"]["history"], horizons=len(cfg["data"]["horizons"]), hidden_dim=mcfg.get("hidden_dim",64), heads=mcfg.get("heads",4), dropout=mcfg.get("dropout",0.1), quantiles=mcfg.get("quantiles",[0.1,0.5,0.9]))
    opt=torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"].get("weight_decay",0.0))
    loader=DataLoader(ds_train,batch_size=cfg["training"]["batch_size"],shuffle=True)
    vloader=DataLoader(ds_val,batch_size=cfg["training"]["batch_size"],shuffle=False)
    quantiles=tuple(mcfg.get("quantiles",[0.1,0.5,0.9]))
    best=float("inf"); best_state=None; patience=0; hist=[]
    for epoch in range(1,int(cfg["training"]["epochs"])+1):
        model.train(); train_losses=[]
        for x,y in loader:
            opt.zero_grad(); pred=model(x,A)
            loss=pinball_loss(pred,y,quantiles) if pred.ndim==4 else torch.nn.functional.l1_loss(pred,y)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); train_losses.append(loss.item())
        model.eval(); val_losses=[]
        with torch.no_grad():
            for x,y in vloader:
                pred=model(x,A)
                loss=pinball_loss(pred,y,quantiles) if pred.ndim==4 else torch.nn.functional.l1_loss(pred,y)
                val_losses.append(loss.item())
        tv=float(np.mean(train_losses)); vv=float(np.mean(val_losses)); hist.append({"epoch":epoch,"train_loss":tv,"val_loss":vv})
        if vv < best-1e-6:
            best=vv; best_state=copy.deepcopy(model.state_dict()); patience=0
        else:
            patience+=1
            if patience>=int(cfg["training"].get("patience",10)): break
    if best_state is not None: model.load_state_dict(best_state)
    out=Path(cfg["output"]["dir"]); out.mkdir(parents=True,exist_ok=True)
    torch.save({"model_state":model.state_dict(),"config":cfg,"sensors":bundle.sensors,"features":features,"target_transform":target_transform}, out/"model.pt")
    save_json(hist,out/"training_history.json")
    return out/"model.pt"
