#!/usr/bin/env python
"""
11_frontier_woodlands.py - Distance-frontier validation on The Woodlands (30/40/50 km).

The main validation ends at 20 km because every network's buffered extent ends there by
construction. This script rebuilds the held-out Woodlands network with a 50 km buffer
(TIGER data in temp/tiger_woodlands50, fetched with fetch_tiger.py --buffer-km 50) and
validates the per-node method at 30, 40, and 50 km, cheapest distance first, so the cost
of each stage is known before the next starts. It answers one question: where does
betweenness rank fidelity actually cross the 0.95 line beyond the paper's envelope?

Everything uses frontier-specific cache keys (woodlands50_*); the paper's 20 km
artifacts are untouched.

Usage:
    python 11_frontier_woodlands.py                      # 30, 40, 50 km staged
    python 11_frontier_woodlands.py --distances 30000    # single stage
"""

import argparse
import pickle
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from cityseer.metrics import networks
from cityseer.tools import graphs, io
from shapely.geometry import Point
from utilities import CACHE_DIR, OUTPUT_DIR, assert_mask_within_data, compute_accuracy_metrics

SCRIPT_DIR = Path(__file__).parent
EDGES_DIR = SCRIPT_DIR.parent.parent.parent / "temp" / "tiger_woodlands50"
CRS = "EPSG:26915"
BUFFER_M = 50_000
FRONTIER_DISTANCES = [30000, 40000, 50000]
N_RUNS = 3


def load_frontier_network(force: bool = False):
    """Build (or load) the 50 km-buffered Woodlands network; live = Woodlands boundary."""
    import osmnx as ox

    boundary_cache = CACHE_DIR / "woodlands_boundary.geojson"
    if boundary_cache.exists():
        boundary = gpd.read_file(boundary_cache).geometry.iloc[0]
    else:
        boundary = ox.geocode_to_gdf("The Woodlands, Texas, USA").to_crs(CRS).geometry.iloc[0]
    buffered = boundary.buffer(BUFFER_M)

    graph_cache = CACHE_DIR / "woodlands50_graph.pkl"
    if graph_cache.exists() and not force:
        print(f"Loading cached frontier graph from {graph_cache}")
        with open(graph_cache, "rb") as f:
            G = pickle.load(f)
    else:
        edge_files = sorted(EDGES_DIR.glob("tl_2023_*_edges.zip"))
        if not edge_files:
            raise FileNotFoundError(f"No TIGER edge files in {EDGES_DIR}; run fetch_tiger.py --buffer-km 50 first.")
        print(f"Loading frontier network from {len(edge_files)} TIGER EDGES files")
        parts = [gpd.read_file(z).to_crs(CRS) for z in edge_files]
        edges_gdf = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs=CRS)
        edges_gdf = edges_gdf[(edges_gdf["ROADFLG"] == "Y") & edges_gdf.intersects(buffered)]
        edges_gdf = edges_gdf[edges_gdf.geometry.is_valid & ~edges_gdf.geometry.is_empty].explode(index_parts=False)
        edges_gdf.geometry = edges_gdf.geometry.map(shapely.force_2d)
        print(f"  Loaded: {len(edges_gdf)} road edges")
        assert_mask_within_data(buffered, edges_gdf, "Woodlands-50")
        print("  Building graph...")
        G = io.nx_from_generic_geopandas(edges_gdf)
        G = graphs.nx_remove_filler_nodes(G)
        G = graphs.nx_remove_dangling_nodes(G)
        G = graphs.nx_consolidate_nodes(G, buffer_dist=10)
        G = graphs.nx_remove_dangling_nodes(G, despine=20)
        with open(graph_cache, "wb") as f:
            pickle.dump(G, f)

    print(f"Frontier graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    n_live = 0
    for _n, data in G.nodes(data=True):
        data["live"] = boundary.contains(Point(data["x"], data["y"]))
        n_live += data["live"]
    print(f"  Live nodes: {n_live}/{G.number_of_nodes()} ({100 * n_live / G.number_of_nodes():.1f}%)")
    nodes_gdf, _, net = io.network_structure_from_nx(G)
    return net, nodes_gdf, nodes_gdf["live"].values, n_live


def run_distance(net, nodes_gdf, live_mask, dist: int) -> dict:
    """Exact ground truth (cached), then the method, at one distance."""
    gt_cache = CACHE_DIR / f"woodlands50_ground_truth_{dist}m.pkl"
    if gt_cache.exists():
        print(f"  Loading cached ground truth from {gt_cache}")
        with open(gt_cache, "rb") as f:
            gt = pickle.load(f)
    else:
        print(f"  Computing exact ground truth at {dist}m (closeness-only, then betweenness-only)...")
        t0 = time.time()
        gdf_c = networks.closeness_shortest(net, nodes_gdf.copy(), distances=[dist])
        t_close = time.time() - t0
        print(f"    closeness exact: {t_close / 60:.1f} min")
        t0 = time.time()
        gdf_b = networks.betweenness_shortest(net, nodes_gdf.copy(), distances=[dist])
        t_betw = time.time() - t0
        print(f"    betweenness exact: {t_betw / 60:.1f} min")
        gt = {
            "harmonic": gdf_c[f"cc_harmonic_{dist}"].to_numpy(float)[live_mask],
            "betweenness": gdf_b[f"cc_betweenness_{dist}"].to_numpy(float)[live_mask],
            "node_reach": gdf_c[f"cc_density_{dist}"].to_numpy(float)[live_mask],
            "baseline_time_closeness": t_close,
            "baseline_time_betweenness": t_betw,
        }
        with open(gt_cache, "wb") as f:
            pickle.dump(gt, f)

    true_h = np.asarray(gt["harmonic"], float)
    true_b = np.asarray(gt["betweenness"], float)
    rhos_h, rhos_b, times_c, times_b = [], [], [], []
    for seed in range(N_RUNS):
        t0 = time.time()
        gdf_c = networks.closeness_shortest(net, nodes_gdf.copy(), distances=[dist], random_seed=42 + seed, sample=True)
        times_c.append(time.time() - t0)
        sp_h, _, _, _, _ = compute_accuracy_metrics(true_h, gdf_c[f"cc_harmonic_{dist}"].to_numpy(float)[live_mask])
        rhos_h.append(sp_h)
        t0 = time.time()
        gdf_b = networks.betweenness_shortest(
            net, nodes_gdf.copy(), distances=[dist], random_seed=42 + seed, sample=True
        )
        times_b.append(time.time() - t0)
        sp_b, _, _, _, _ = compute_accuracy_metrics(true_b, gdf_b[f"cc_betweenness_{dist}"].to_numpy(float)[live_mask])
        rhos_b.append(sp_b)
        print(f"    seed {seed}: rho_c={sp_h:.4f} rho_b={sp_b:.4f}")

    row = {
        "distance": dist,
        "rho_closeness": float(np.nanmean(rhos_h)),
        "rho_closeness_std": float(np.nanstd(rhos_h)),
        "rho_betweenness": float(np.nanmean(rhos_b)),
        "rho_betweenness_std": float(np.nanstd(rhos_b)),
        "sampled_time_c": float(np.mean(times_c)),
        "sampled_time_b": float(np.mean(times_b)),
        "speedup_closeness": gt["baseline_time_closeness"] / float(np.mean(times_c)),
        "speedup_betweenness": gt["baseline_time_betweenness"] / float(np.mean(times_b)),
    }
    print(
        f"  {dist}m: rho_c={row['rho_closeness']:.4f} rho_b={row['rho_betweenness']:.4f} "
        f"sp_c={row['speedup_closeness']:.1f} sp_b={row['speedup_betweenness']:.1f}"
    )
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Frontier validation beyond 20 km on Woodlands-50")
    parser.add_argument("--distances", nargs="+", type=int, default=FRONTIER_DISTANCES)
    parser.add_argument("--force", action="store_true", help="Rebuild the frontier graph")
    args = parser.parse_args()

    net, nodes_gdf, live_mask, _ = load_frontier_network(force=args.force)
    out = OUTPUT_DIR / "woodlands_frontier.csv"
    rows = []
    for dist in sorted(args.distances):
        print(f"\n{'=' * 60}\nFRONTIER {dist}m\n{'=' * 60}")
        rows.append(run_distance(net, nodes_gdf, live_mask, dist))
        # write incrementally so partial progress survives interruption
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"  Saved {out}")
    return 0


if __name__ == "__main__":
    exit(main())
