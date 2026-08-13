from __future__ import annotations
import numpy as np
import pandas as pd


def osm_sensor_adjacency(meta: pd.DataFrame, place: str = "Istanbul, Türkiye", max_route_km: float = 8.0) -> np.ndarray:
    """Build an approximate directed sensor graph from the drivable OSM network.

    Requires the optional `osmnx` dependency and network access. Sensors are snapped
    to nearest OSM nodes. A directed edge is added when the shortest-path road
    distance between two candidate nearby sensors is <= max_route_km.

    This is intentionally separated from the offline kNN graph so CI never depends
    on Overpass availability.
    """
    try:
        import osmnx as ox
        import networkx as nx
    except ImportError as exc:
        raise RuntimeError("Install requirements-geo.txt to build the OSM graph") from exc

    G = ox.graph.graph_from_place(place, network_type="drive", simplify=True, retain_all=False)
    G = ox.routing.add_edge_speeds(G)
    G = ox.routing.add_edge_travel_times(G)
    nodes = ox.distance.nearest_nodes(G, X=meta.longitude.to_numpy(), Y=meta.latitude.to_numpy())
    n = len(meta)
    a = np.zeros((n,n), dtype=np.float32)
    # Use geographic proximity to limit expensive route queries.
    xy = meta[["latitude","longitude"]].to_numpy()
    for i in range(n):
        dif = ((xy - xy[i]) ** 2).sum(axis=1)
        candidates = np.argsort(dif)[1:9]
        for j in candidates:
            try:
                meters = nx.shortest_path_length(G, nodes[i], nodes[j], weight="length")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            km = float(meters) / 1000.0
            if km <= max_route_km:
                a[i,j] = np.exp(-km / max_route_km)
    np.fill_diagonal(a, 1.0)
    return a
