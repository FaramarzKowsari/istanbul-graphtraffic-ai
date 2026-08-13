import torch
from graphtraffic.experiments.failure import apply_sensor_failure

def test_failure_mask():
    x=torch.ones(2,3,10,4); y,m=apply_sensor_failure(x,0.2,seed=1)
    assert m.sum().item()==2; assert (y[:,:,m,:]==0).all()
