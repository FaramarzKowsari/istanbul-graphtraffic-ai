import pandas as pd
import numpy as np
from graphtraffic.data.graph import knn_adjacency, normalize_adjacency

def test_knn_graph():
    meta=pd.DataFrame({"sensor_id":["a","b","c"],"latitude":[41,41.01,41.02],"longitude":[29,29.01,29.02]})
    a=knn_adjacency(meta,k=1)
    assert a.shape==(3,3) and np.allclose(np.diag(a),1)
    n=normalize_adjacency(a); assert np.isfinite(n).all()
