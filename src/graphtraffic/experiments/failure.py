from __future__ import annotations
import torch


def apply_sensor_failure(x: torch.Tensor, fraction: float, seed: int = 0, structured: bool = False):
    if not (0 <= fraction < 1):
        raise ValueError("fraction must be in [0,1)")
    n = x.shape[2]
    k = int(round(n*fraction))
    if k == 0:
        return x.clone(), torch.zeros(n, dtype=torch.bool, device=x.device)
    g = torch.Generator(device=x.device).manual_seed(seed)
    if structured:
        start = int(torch.randint(0, max(1,n-k+1), (1,), generator=g, device=x.device).item())
        idx = torch.arange(start, start+k, device=x.device)
    else:
        idx = torch.randperm(n, generator=g, device=x.device)[:k]
    out=x.clone(); out[:,:,idx,:]=0
    mask=torch.zeros(n,dtype=torch.bool,device=x.device); mask[idx]=True
    return out, mask
