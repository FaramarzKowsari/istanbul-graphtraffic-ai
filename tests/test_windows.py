from graphtraffic.data.synthetic import generate_synthetic
from graphtraffic.data.features import add_calendar_features
from graphtraffic.data.windows import frame_to_tensor, TrafficWindowDataset

def test_windows_shapes():
    df=add_calendar_features(generate_synthetic(hours=48,sensors=5))
    feats=["avg_speed","vehicle_count","traffic_density","hour_sin","hour_cos"]
    b=frame_to_tensor(df,feats,"avg_speed"); ds=TrafficWindowDataset(b,12,[1,2,3],0,48)
    x,y=ds[0]; assert x.shape==(12,5,5); assert y.shape==(3,5)
