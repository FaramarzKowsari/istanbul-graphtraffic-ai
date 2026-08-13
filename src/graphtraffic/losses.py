import torch


def pinball_loss(pred, target, quantiles):
    # pred B,H,N,Q ; target B,H,N
    target = target.unsqueeze(-1)
    losses=[]
    for i,q in enumerate(quantiles):
        e = target[...,0] - pred[...,i]
        losses.append(torch.maximum(q*e, (q-1)*e).mean())
    return torch.stack(losses).mean()
