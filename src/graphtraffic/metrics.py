from __future__ import annotations
import numpy as np


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y)-np.asarray(p))))

def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y)-np.asarray(p))**2)))

def mape(y, p, eps=1e-3):
    y=np.asarray(y); p=np.asarray(p)
    mask=np.abs(y)>eps
    return float(np.mean(np.abs((y[mask]-p[mask])/y[mask]))*100) if mask.any() else float("nan")

def r2(y, p):
    y=np.asarray(y); p=np.asarray(p)
    ss_res=((y-p)**2).sum(); ss_tot=((y-y.mean())**2).sum()
    return float(1-ss_res/ss_tot) if ss_tot>0 else float("nan")

def interval_coverage(y, lower, upper):
    y=np.asarray(y); lower=np.asarray(lower); upper=np.asarray(upper)
    return float(np.mean((y>=lower)&(y<=upper)))

def interval_width(lower, upper):
    return float(np.mean(np.asarray(upper)-np.asarray(lower)))
