from __future__ import annotations

import math
import numpy as np
import pandas as pd

from graphtraffic.data.sampling import (
    _haversine_vector_km,
    select_geographically_diverse_nodes,
)


def select_anchor_neighborhood_nodes(
    meta: pd.DataFrame,
    n_nodes: int,
    cluster_size: int = 4,
) -> pd.DataFrame:
    """Select citywide anchors plus local neighbors around each anchor.

    The goal is to preserve broad spatial coverage while avoiding the very sparse
    farthest-point-only sample used in Pilot #2/#4. Only sensor_id/latitude/longitude
    are used. No traffic target or volume value is used for selection.

    Strategy:
    1. Select geographically diverse anchors with deterministic FPS.
    2. Reserve every anchor so citywide coverage is preserved.
    3. Add nearest unused neighbors around anchors in round-robin order.
    4. If overlap leaves spare capacity, fill with nodes closest to any anchor.

    Returns a deterministic DataFrame with selection metadata.
    """
    required = {"sensor_id", "latitude", "longitude"}
    missing = required.difference(meta.columns)
    if missing:
        raise ValueError(f"meta missing required columns: {sorted(missing)}")
    if n_nodes < 4:
        raise ValueError("n_nodes must be >= 4")
    if cluster_size < 2:
        raise ValueError("cluster_size must be >= 2")

    work = (
        meta.dropna(subset=["sensor_id", "latitude", "longitude"])
        .drop_duplicates(subset=["sensor_id"], keep="first")
        .copy()
    )
    work["sensor_id"] = work["sensor_id"].astype(str)
    work = work.sort_values("sensor_id", kind="stable").reset_index(drop=True)

    if len(work) < n_nodes:
        raise ValueError(
            f"Only {len(work)} eligible nodes are available for n_nodes={n_nodes}"
        )

    n_anchors = max(2, int(math.ceil(n_nodes / cluster_size)))
    n_anchors = min(n_anchors, n_nodes)
    anchors = select_geographically_diverse_nodes(work, n_anchors).copy()
    anchor_ids = anchors["sensor_id"].astype(str).tolist()

    selected_ids = list(anchor_ids)
    selected_set = set(selected_ids)

    records = []
    for i, row in anchors.iterrows():
        records.append(
            {
                "sensor_id": str(row.sensor_id),
                "role": "anchor",
                "anchor_id": str(row.sensor_id),
                "anchor_order": int(i + 1),
                "neighbor_rank": 0,
                "distance_to_anchor_km": 0.0,
            }
        )

    lat = work["latitude"].to_numpy(float)
    lon = work["longitude"].to_numpy(float)
    sid = work["sensor_id"].astype(str).to_numpy()
    index_by_sid = {str(s): i for i, s in enumerate(sid)}

    # Round-robin: each anchor gets one local neighbor before any anchor gets a second.
    local_slots = max(cluster_size - 1, 1)
    for rank in range(1, local_slots + 1):
        for anchor_order, anchor_id in enumerate(anchor_ids, start=1):
            if len(selected_ids) >= n_nodes:
                break
            ai = index_by_sid[anchor_id]
            d = _haversine_vector_km(lat[ai], lon[ai], lat, lon)
            order = np.argsort(d, kind="stable")
            chosen = None
            for idx in order:
                candidate = str(sid[int(idx)])
                if candidate not in selected_set:
                    chosen = int(idx)
                    break
            if chosen is None:
                continue
            candidate = str(sid[chosen])
            selected_set.add(candidate)
            selected_ids.append(candidate)
            records.append(
                {
                    "sensor_id": candidate,
                    "role": "local_neighbor",
                    "anchor_id": anchor_id,
                    "anchor_order": int(anchor_order),
                    "neighbor_rank": int(rank),
                    "distance_to_anchor_km": float(d[chosen]),
                }
            )

    # Defensive fill if anchor neighborhoods overlap heavily.
    if len(selected_ids) < n_nodes:
        anchor_idx = np.asarray([index_by_sid[a] for a in anchor_ids], dtype=int)
        min_to_anchor = np.full(len(work), np.inf, dtype=float)
        for ai in anchor_idx:
            min_to_anchor = np.minimum(
                min_to_anchor,
                _haversine_vector_km(lat[ai], lon[ai], lat, lon),
            )
        for idx in np.argsort(min_to_anchor, kind="stable"):
            candidate = str(sid[int(idx)])
            if candidate in selected_set:
                continue
            nearest_anchor_pos = int(
                np.argmin(
                    [
                        _haversine_vector_km(
                            lat[int(idx)], lon[int(idx)],
                            np.asarray([lat[ai]]), np.asarray([lon[ai]])
                        )[0]
                        for ai in anchor_idx
                    ]
                )
            )
            anchor_id = anchor_ids[nearest_anchor_pos]
            selected_set.add(candidate)
            selected_ids.append(candidate)
            records.append(
                {
                    "sensor_id": candidate,
                    "role": "fill_neighbor",
                    "anchor_id": anchor_id,
                    "anchor_order": int(nearest_anchor_pos + 1),
                    "neighbor_rank": int(cluster_size),
                    "distance_to_anchor_km": float(min_to_anchor[int(idx)]),
                }
            )
            if len(selected_ids) >= n_nodes:
                break

    rec = pd.DataFrame(records).drop_duplicates("sensor_id", keep="first")
    rec["selection_order"] = np.arange(1, len(rec) + 1, dtype=int)

    selected = work[work["sensor_id"].isin(rec["sensor_id"])].copy()
    selected = selected.merge(rec, on="sensor_id", how="inner")
    selected = selected.sort_values("selection_order").reset_index(drop=True)
    if len(selected) != n_nodes:
        raise RuntimeError(
            f"Selection produced {len(selected)} unique nodes, expected {n_nodes}"
        )
    return selected
