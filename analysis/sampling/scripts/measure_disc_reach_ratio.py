#!/usr/bin/env python
"""
measure_disc_reach_ratio.py - Measure per-node Euclidean-disc-to-network-reach ratios.

The ablation's rung 2 (Section 4.2 of the paper) converts a Euclidean neighbour
count into a network-reach estimate by dividing by a fixed deflation factor
(cityseer.sampling.EUCLIDEAN_REACH_DEFLATION = 2.5). This script records the
measured per-node ratios that the deflation is judged against: for each live
node, the number of nodes within straight-line distance d (all nodes, KD-tree)
divided by the node's network reach at d (from the exact ground truth).

Requires the cached graphs and the per-node method caches
({network}_sampled_{dist}m_adaptive.pkl, for aligned live coordinates and reach).

Usage:
    python measure_disc_reach_ratio.py

Outputs:
    output/euclidean_reach_ratios.csv (per network/distance: median, p90, p99, max,
    and the fraction of live nodes whose ratio exceeds the deflation)
"""

import pickle

import numpy as np
import pandas as pd
from cityseer.sampling import EUCLIDEAN_REACH_DEFLATION
from scipy.spatial import KDTree
from utilities import CACHE_DIR, OUTPUT_DIR

NETWORKS = ["gla", "madrid", "cary", "woodlands"]
DISTANCES = [10000, 20000]


def all_node_coords(network: str) -> np.ndarray:
    """All-node coordinates (live + buffer) from the cached networkx graph."""
    with open(CACHE_DIR / f"{network}_graph.pkl", "rb") as f:
        G = pickle.load(f)
    return np.array([(d["x"], d["y"]) for _, d in G.nodes(data=True)], dtype=float)


def main() -> int:
    rows = []
    for network in NETWORKS:
        coords = all_node_coords(network)
        tree = KDTree(coords)
        print(f"{network}: {len(coords)} nodes")
        for dist in DISTANCES:
            cache = CACHE_DIR / f"{network}_sampled_{dist}m_adaptive.pkl"
            if not cache.exists():
                print(f"  {dist}m: no method cache, skipping")
                continue
            with open(cache, "rb") as f:
                data = pickle.load(f)
            live_pts = np.column_stack(
                [np.asarray(data["node_x"], dtype=float), np.asarray(data["node_y"], dtype=float)]
            )
            reach = np.asarray(data["node_reach"], dtype=float)
            disc = tree.query_ball_point(live_pts, r=float(dist), return_length=True, workers=-1)
            disc = np.asarray(disc, dtype=float)
            valid = reach > 0
            ratio = disc[valid] / reach[valid]
            rows.append(
                {
                    "network": network,
                    "distance": dist,
                    "n_live": int(valid.sum()),
                    "ratio_median": float(np.median(ratio)),
                    "ratio_p90": float(np.percentile(ratio, 90)),
                    "ratio_p99": float(np.percentile(ratio, 99)),
                    "ratio_max": float(np.max(ratio)),
                    "frac_above_deflation": float(np.mean(ratio > EUCLIDEAN_REACH_DEFLATION)),
                }
            )
            r = rows[-1]
            print(
                f"  {dist // 1000}km: median={r['ratio_median']:.2f} p90={r['ratio_p90']:.2f} "
                f"p99={r['ratio_p99']:.2f} max={r['ratio_max']:.2f} "
                f">{EUCLIDEAN_REACH_DEFLATION}: {r['frac_above_deflation']:.1%}"
            )
    df = pd.DataFrame(rows)
    out = OUTPUT_DIR / "euclidean_reach_ratios.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    exit(main())
