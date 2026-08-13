import pandas as pd
from graphtraffic.data.schema import standardize_frame

def test_schema_aliases():
    df=pd.DataFrame({"datetime":["2026-01-01"],"location_id":["a"],"speed":[42]})
    out=standardize_frame(df)
    assert {"timestamp","sensor_id","avg_speed","vehicle_count","traffic_density"}.issubset(out.columns)
