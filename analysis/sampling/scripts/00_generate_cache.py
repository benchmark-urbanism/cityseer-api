#!/usr/bin/env python
"""
00_generate_cache.py - Generate cached data for the sampling model pipeline.

This script generates the cached data required by the analysis pipeline:
1. Synthetic network sampling results (for fitting k and min_eff_n)
2. GLA network validation data (for model validation)

Run this script first if caches don't exist or need regeneration.

Usage:
    python 00_generate_cache.py                  # Generate all caches
    python 00_generate_cache.py --synthetic      # Only synthetic data
    python 00_generate_cache.py --gla            # Only GLA validation data
    python 00_generate_cache.py --force          # Force regeneration

Outputs:
    - temp/sampling_cache/sampling_analysis_v17.pkl
    - analysis/sampling/.cache/gla_graph.pkl
    - analysis/sampling/.cache/ground_truth_*.pkl
    - analysis/sampling/output/model_validation.csv
"""

import argparse
import json
import math
import pickle
import sys
import time
import warnings
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from cityseer.tools import graphs, io
from scipy import stats as scipy_stats
from shapely.geometry import Point
from shapely.ops import unary_union

# Add analysis directory for imports (analysis/utils is a sibling of analysis/sampling)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.substrates import generate_keyed_template

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
SAMPLING_DIR = SCRIPT_DIR.parent  # analysis/sampling
CACHE_DIR = SAMPLING_DIR.parent.parent / "temp" / "sampling_cache"
LOCAL_CACHE_DIR = SAMPLING_DIR / ".cache"
OUTPUT_DIR = SAMPLING_DIR / "output"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Cache version - increment to force re-run
CACHE_VERSION = "v17"

# GLA network file
GLA_GPKG_FILE = SCRIPT_DIR.parent.parent / "temp" / "os_open_roads" / "gla.gpkg"

# =============================================================================
# SYNTHETIC DATA CONFIGURATION
# =============================================================================

N_RUNS = 3  # Multiple runs for variance estimation
TEMPLATE_NAMES = ["trellis", "tree", "linear"]
SUBSTRATE_TILES = 24  # ~12km network extent
DISTANCES = [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2500, 3000, 3500, 4000]
LIVE_INWARD_BUFFER_SYNTHETIC = 4000  # 4km buffer for synthetic networks
PROBS = [
    0.025,
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
    0.975,
    1.0,
]

# =============================================================================
# GLA VALIDATION CONFIGURATION
# =============================================================================

LIVE_INWARD_BUFFER_GLA = 20000  # 20km buffer for GLA network
GLA_DISTANCES = [5000, 10000, 20000]  # Validation distances
GLA_PROBS = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0]
GLA_N_RUNS = 3

# =============================================================================
# MADRID VALIDATION CONFIGURATION
# =============================================================================

MADRID_GPKG_URL = "https://github.com/songololo/ua-dataset-madrid/raw/main/data/street_network_w_edit.gpkg"
LIVE_INWARD_BUFFER_MADRID = 20000  # 20km buffer for Madrid network
MADRID_DISTANCES = [500, 1000, 2000, 4000, 5000, 10000, 20000]  # Validation distances
MADRID_N_RUNS = 3


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def load_cache(name: str):
    """Load cached results if available."""
    path = CACHE_DIR / f"{name}_{CACHE_VERSION}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def save_cache(name: str, data):
    """Save results to cache."""
    path = CACHE_DIR / f"{name}_{CACHE_VERSION}.pkl"
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"  Saved cache: {path}")


def apply_live_buffer_nx(G: nx.MultiGraph, buffer_dist: float) -> nx.MultiGraph:
    """
    Mark only interior nodes as live on NetworkX graph.

    Applies an inward buffer from the convex hull of all nodes.
    Nodes inside the buffered zone are marked as live=True,
    nodes in the buffer zone are marked as live=False.
    """
    coords = [(G.nodes[n]["x"], G.nodes[n]["y"]) for n in G.nodes()]
    all_points = [Point(x, y) for x, y in coords]
    hull = unary_union(all_points).convex_hull
    live_zone = hull.buffer(-buffer_dist)

    for n in G.nodes():
        pt = Point(G.nodes[n]["x"], G.nodes[n]["y"])
        G.nodes[n]["live"] = live_zone.contains(pt)

    n_live = sum(1 for n in G.nodes() if G.nodes[n]["live"])
    print(f"  Live nodes: {n_live}/{G.number_of_nodes()} ({100 * n_live / G.number_of_nodes():.1f}%)")

    return G


def compute_accuracy_metrics(true_vals: np.ndarray, est_vals: np.ndarray) -> tuple:
    """Compute ranking and magnitude accuracy metrics."""
    mask = (true_vals > 0) & np.isfinite(true_vals) & np.isfinite(est_vals)
    if mask.sum() < 10:
        return np.nan, np.nan, np.nan, np.nan

    true_masked = true_vals[mask]
    est_masked = est_vals[mask]

    spearman, _ = scipy_stats.spearmanr(true_masked, est_masked)

    k = max(1, int(len(true_masked) * 0.1))
    true_top_k = set(np.argsort(true_masked)[-k:])
    est_top_k = set(np.argsort(est_masked)[-k:])
    top_k_precision = len(true_top_k & est_top_k) / k

    ratios = est_masked / true_masked
    scale_ratio = float(np.median(ratios))
    scale_iqr = float(np.percentile(ratios, 75) - np.percentile(ratios, 25))

    return spearman, top_k_precision, scale_ratio, scale_iqr


# =============================================================================
# SYNTHETIC DATA GENERATION
# =============================================================================


def generate_synthetic_cache(force: bool = False):
    """Generate synthetic network sampling results cache."""
    cache_path = CACHE_DIR / f"sampling_analysis_{CACHE_VERSION}.pkl"

    if cache_path.exists() and not force:
        print(f"Synthetic cache already exists: {cache_path}")
        print("  Use --force to regenerate")
        return

    print("\n" + "=" * 70)
    print("GENERATING SYNTHETIC DATA CACHE")
    print("=" * 70)
    print(f"Templates: {TEMPLATE_NAMES}")
    print(f"Distances: {DISTANCES}")
    print(f"Probabilities: {len(PROBS)} values")
    print(f"Runs per config: {N_RUNS}")

    results = []

    for topo in TEMPLATE_NAMES:
        print(f"\n{'=' * 50}")
        print(f"Topology: {topo}")
        print(f"{'=' * 50}")

        # Generate substrate
        G, _, _ = generate_keyed_template(template_key=topo, tiles=SUBSTRATE_TILES, decompose=None, plot=False)
        if not nx.is_connected(G):
            largest = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest).copy()

        n_nodes = G.number_of_nodes()
        print(f"Nodes: {n_nodes}, Avg degree: {2 * G.number_of_edges() / n_nodes:.2f}")

        # Apply live buffer
        G = apply_live_buffer_nx(G, LIVE_INWARD_BUFFER_SYNTHETIC)

        # Convert to cityseer format
        ndf, edf, net = io.network_structure_from_nx(G)

        for dist in DISTANCES:
            # Ground truth (full computation)
            true_result = net.local_node_centrality_shortest(
                distances=[dist],
                compute_closeness=True,
                compute_betweenness=True,
                pbar_disabled=True,
            )

            true_harmonic = np.array(true_result.node_harmonic[dist])
            true_betweenness = np.array(true_result.node_betweenness[dist])
            reach = np.array(true_result.node_density[dist])
            mean_reach = float(np.mean(reach))

            if mean_reach < 5:
                continue

            print(f"  d={dist}m, reach={mean_reach:.0f}: ", end="", flush=True)

            for p in PROBS:
                if p == 1.0:
                    # Perfect accuracy at p=1.0
                    for metric in ["harmonic", "betweenness"]:
                        results.append(
                            {
                                "topology": topo,
                                "distance": dist,
                                "n_nodes": n_nodes,
                                "mean_reach": mean_reach,
                                "sample_prob": p,
                                "effective_n": mean_reach,
                                "metric": metric,
                                "spearman": 1.0,
                                "top_k_precision": 1.0,
                                "scale_ratio": 1.0,
                                "scale_iqr": 0.0,
                            }
                        )
                    continue

                # Multiple runs
                spearmans_h, spearmans_b = [], []

                for seed in range(N_RUNS):
                    r = net.local_node_centrality_shortest(
                        distances=[dist],
                        compute_closeness=True,
                        compute_betweenness=True,
                        sample_probability=p,
                        random_seed=seed,
                        pbar_disabled=True,
                    )

                    est_harmonic = np.array(r.node_harmonic[dist])
                    est_betweenness = np.array(r.node_betweenness[dist])

                    sp_h, prec_h, scale_h, iqr_h = compute_accuracy_metrics(true_harmonic, est_harmonic)
                    sp_b, prec_b, scale_b, iqr_b = compute_accuracy_metrics(true_betweenness, est_betweenness)

                    if not np.isnan(sp_h):
                        spearmans_h.append((sp_h, prec_h, scale_h, iqr_h))
                    if not np.isnan(sp_b):
                        spearmans_b.append((sp_b, prec_b, scale_b, iqr_b))

                effective_n = mean_reach * p

                if spearmans_h:
                    results.append(
                        {
                            "topology": topo,
                            "distance": dist,
                            "n_nodes": n_nodes,
                            "mean_reach": mean_reach,
                            "sample_prob": p,
                            "effective_n": effective_n,
                            "metric": "harmonic",
                            "spearman": np.mean([x[0] for x in spearmans_h]),
                            "top_k_precision": np.mean([x[1] for x in spearmans_h]),
                            "scale_ratio": np.mean([x[2] for x in spearmans_h]),
                            "scale_iqr": np.mean([x[3] for x in spearmans_h]),
                        }
                    )

                if spearmans_b:
                    results.append(
                        {
                            "topology": topo,
                            "distance": dist,
                            "n_nodes": n_nodes,
                            "mean_reach": mean_reach,
                            "sample_prob": p,
                            "effective_n": effective_n,
                            "metric": "betweenness",
                            "spearman": np.mean([x[0] for x in spearmans_b]),
                            "top_k_precision": np.mean([x[1] for x in spearmans_b]),
                            "scale_ratio": np.mean([x[2] for x in spearmans_b]),
                            "scale_iqr": np.mean([x[3] for x in spearmans_b]),
                        }
                    )

                print(".", end="", flush=True)
            print()

    # Save cache
    save_cache("sampling_analysis", results)
    print(f"\nGenerated {len(results)} data points")


# =============================================================================
# GLA VALIDATION DATA GENERATION
# =============================================================================


def generate_gla_cache(force: bool = False):
    """Generate GLA network validation data cache."""
    gla_cache = LOCAL_CACHE_DIR / "gla_graph.pkl"
    validation_csv = OUTPUT_DIR / "model_validation.csv"

    if validation_csv.exists() and not force:
        print(f"GLA validation data already exists: {validation_csv}")
        print("  Use --force to regenerate")
        return

    if not GLA_GPKG_FILE.exists():
        print(f"ERROR: GLA network file not found: {GLA_GPKG_FILE}")
        print("  Download the OS Open Roads dataset for Greater London")
        return

    print("\n" + "=" * 70)
    print("GENERATING GLA VALIDATION DATA")
    print("=" * 70)

    # Load or generate GLA graph
    if gla_cache.exists() and not force:
        print(f"Loading cached GLA graph from {gla_cache}")
        with open(gla_cache, "rb") as f:
            G = pickle.load(f)
    else:
        print(f"Loading GLA network from {GLA_GPKG_FILE}")
        edges_gdf = gpd.read_file(GLA_GPKG_FILE)
        edges_gdf = edges_gdf[edges_gdf.geometry.is_valid & ~edges_gdf.geometry.is_empty]
        edges_gdf = edges_gdf.explode(index_parts=False)

        print("  Building graph...")
        G = io.nx_from_generic_geopandas(edges_gdf)
        G = graphs.nx_remove_filler_nodes(G)
        G = graphs.nx_remove_dangling_nodes(G)
        G = graphs.nx_consolidate_nodes(G, buffer_dist=1)
        G = graphs.nx_remove_dangling_nodes(G, despine=20)

        print(f"  Caching to {gla_cache}")
        with open(gla_cache, "wb") as f:
            pickle.dump(G, f)

    print(f"GLA graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Apply live buffer
    print(f"Applying {LIVE_INWARD_BUFFER_GLA / 1000:.0f}km inward buffer...")
    all_points = [Point(G.nodes[n]["x"], G.nodes[n]["y"]) for n in G.nodes()]
    hull = unary_union(all_points).convex_hull
    live_zone = hull.buffer(-LIVE_INWARD_BUFFER_GLA)

    n_live = 0
    for n in G.nodes():
        pt = Point(G.nodes[n]["x"], G.nodes[n]["y"])
        is_live = live_zone.contains(pt)
        G.nodes[n]["live"] = is_live
        if is_live:
            n_live += 1
    print(f"  Live nodes: {n_live} / {G.number_of_nodes()} ({100 * n_live / G.number_of_nodes():.1f}%)")

    # Convert to cityseer
    print("Converting to cityseer format...")
    nodes_gdf, edges_gdf_out, net = io.network_structure_from_nx(G)

    results = []

    for dist in GLA_DISTANCES:
        print(f"\n{'=' * 50}")
        print(f"Distance: {dist}m")
        print(f"{'=' * 50}")

        # Check for cached ground truth
        gt_cache = LOCAL_CACHE_DIR / f"ground_truth_{dist}m.pkl"
        if gt_cache.exists() and not force:
            print(f"  Loading cached ground truth from {gt_cache}")
            with open(gt_cache, "rb") as f:
                gt_data = pickle.load(f)
            true_harmonic = gt_data["harmonic"]
            true_betweenness = gt_data["betweenness"]
            mean_reach = gt_data["mean_reach"]
        else:
            print("  Computing ground truth (this may take a while)...")
            true_result = net.local_node_centrality_shortest(
                distances=[dist],
                compute_closeness=True,
                compute_betweenness=True,
                pbar_disabled=False,
            )
            true_harmonic = np.array(true_result.node_harmonic[dist])
            true_betweenness = np.array(true_result.node_betweenness[dist])
            reach = np.array(true_result.node_density[dist])
            mean_reach = float(np.mean(reach))

            # Cache ground truth
            with open(gt_cache, "wb") as f:
                pickle.dump(
                    {
                        "harmonic": true_harmonic,
                        "betweenness": true_betweenness,
                        "mean_reach": mean_reach,
                    },
                    f,
                )
            print(f"  Cached ground truth to {gt_cache}")

        print(f"  Mean reach: {mean_reach:.0f}")

        for p in GLA_PROBS:
            effective_n = mean_reach * p
            print(f"    p={p:.2f} (eff_n={effective_n:.0f}): ", end="", flush=True)

            spearmans_h, spearmans_b = [], []

            for seed in range(GLA_N_RUNS):
                r = net.local_node_centrality_shortest(
                    distances=[dist],
                    compute_closeness=True,
                    compute_betweenness=True,
                    sample_probability=p,
                    random_seed=seed,
                    pbar_disabled=True,
                )

                est_harmonic = np.array(r.node_harmonic[dist])
                est_betweenness = np.array(r.node_betweenness[dist])

                sp_h, _, _, _ = compute_accuracy_metrics(true_harmonic, est_harmonic)
                sp_b, _, _, _ = compute_accuracy_metrics(true_betweenness, est_betweenness)

                if not np.isnan(sp_h):
                    spearmans_h.append(sp_h)
                if not np.isnan(sp_b):
                    spearmans_b.append(sp_b)

                print(".", end="", flush=True)

            if spearmans_h:
                results.append(
                    {
                        "distance": dist,
                        "mean_reach": mean_reach,
                        "sample_prob": p,
                        "effective_n": effective_n,
                        "metric": "harmonic",
                        "spearman": np.mean(spearmans_h),
                        "spearman_std": np.std(spearmans_h),
                    }
                )

            if spearmans_b:
                results.append(
                    {
                        "distance": dist,
                        "mean_reach": mean_reach,
                        "sample_prob": p,
                        "effective_n": effective_n,
                        "metric": "betweenness",
                        "spearman": np.mean(spearmans_b),
                        "spearman_std": np.std(spearmans_b),
                    }
                )

            print(f" rho_h={np.mean(spearmans_h):.3f}, rho_b={np.mean(spearmans_b):.3f}")

    # Save validation results
    df = pd.DataFrame(results)
    df.to_csv(validation_csv, index=False)
    print(f"\nSaved validation results: {validation_csv}")
    print(f"  {len(results)} data points")


# =============================================================================
# MADRID VALIDATION DATA GENERATION
# =============================================================================


def generate_madrid_cache(force: bool = False):
    """Generate Madrid network validation data cache."""
    madrid_cache = LOCAL_CACHE_DIR / "madrid_graph.pkl"
    validation_csv = OUTPUT_DIR / "madrid_validation.csv"

    if validation_csv.exists() and not force:
        print(f"Madrid validation data already exists: {validation_csv}")
        print("  Use --force to regenerate")
        return

    print("\n" + "=" * 70)
    print("GENERATING MADRID VALIDATION DATA")
    print("=" * 70)

    # Load or generate Madrid graph
    if madrid_cache.exists() and not force:
        print(f"Loading cached Madrid graph from {madrid_cache}")
        with open(madrid_cache, "rb") as f:
            G = pickle.load(f)
    else:
        print(f"Downloading Madrid network from: {MADRID_GPKG_URL}")
        print("  (This may take a minute...)")

        edges_gdf = gpd.read_file(MADRID_GPKG_URL)
        print(f"  Downloaded: {len(edges_gdf)} edges, CRS: {edges_gdf.crs}")

        # Convert multipart geoms to single
        edges_gdf = edges_gdf[edges_gdf.geometry.is_valid & ~edges_gdf.geometry.is_empty]
        edges_gdf = edges_gdf.explode(index_parts=False)

        print("  Building graph...")
        G = io.nx_from_generic_geopandas(edges_gdf)
        G = graphs.nx_remove_filler_nodes(G)
        G = graphs.nx_remove_dangling_nodes(G)

        print(f"  Caching to {madrid_cache}")
        with open(madrid_cache, "wb") as f:
            pickle.dump(G, f)

    print(f"Madrid graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Apply live buffer on NetworkX graph
    print(f"Applying {LIVE_INWARD_BUFFER_MADRID / 1000:.0f}km inward buffer...")
    all_points = [Point(G.nodes[n]["x"], G.nodes[n]["y"]) for n in G.nodes()]
    hull = unary_union(all_points).convex_hull
    live_zone = hull.buffer(-LIVE_INWARD_BUFFER_MADRID)

    n_live = 0
    for n in G.nodes():
        pt = Point(G.nodes[n]["x"], G.nodes[n]["y"])
        is_live = live_zone.contains(pt)
        G.nodes[n]["live"] = is_live
        if is_live:
            n_live += 1
    print(f"  Live nodes: {n_live} / {G.number_of_nodes()} ({100 * n_live / G.number_of_nodes():.1f}%)")

    # Convert to cityseer
    print("Converting to cityseer format...")
    nodes_gdf, edges_gdf_out, net = io.network_structure_from_nx(G)

    results = []

    for dist in MADRID_DISTANCES:
        print(f"\n{'=' * 50}")
        print(f"Distance: {dist}m")
        print(f"{'=' * 50}")

        # Check for cached ground truth
        gt_cache = LOCAL_CACHE_DIR / f"madrid_ground_truth_{dist}m.pkl"
        if gt_cache.exists() and not force:
            print(f"  Loading cached ground truth from {gt_cache}")
            with open(gt_cache, "rb") as f:
                gt_data = pickle.load(f)
            true_harmonic = gt_data["harmonic"]
            true_betweenness = gt_data["betweenness"]
            mean_reach = gt_data["mean_reach"]
            baseline_time = gt_data.get("baseline_time", None)
            if baseline_time is None:
                print("  Warning: cached data missing timing, will re-compute")
        else:
            baseline_time = None

        if not gt_cache.exists() or force or baseline_time is None:
            print("  Computing ground truth (this may take a while)...")
            t0 = time.perf_counter()
            true_result = net.local_node_centrality_shortest(
                distances=[dist],
                compute_closeness=True,
                compute_betweenness=True,
                pbar_disabled=False,
            )
            baseline_time = time.perf_counter() - t0
            true_harmonic = np.array(true_result.node_harmonic[dist])
            true_betweenness = np.array(true_result.node_betweenness[dist])
            reach = np.array(true_result.node_density[dist])
            mean_reach = float(np.mean(reach))

            # Cache ground truth with timing
            with open(gt_cache, "wb") as f:
                pickle.dump(
                    {
                        "harmonic": true_harmonic,
                        "betweenness": true_betweenness,
                        "mean_reach": mean_reach,
                        "baseline_time": baseline_time,
                    },
                    f,
                )
            print(f"  Cached ground truth to {gt_cache}")
            print(f"  Baseline time: {baseline_time:.2f}s")

        print(f"  Mean reach: {mean_reach:.0f}")

        # Test at model-recommended probability only (for validation)
        # Load model parameters
        model_path = OUTPUT_DIR / "sampling_model.json"
        if model_path.exists():
            with open(model_path) as f:
                model = json.load(f)
            k = model["model"]["k"]
            min_eff_n = model["model"]["min_eff_n"]

            # Compute model-recommended p
            eff_n = max(k * math.sqrt(mean_reach), min_eff_n)
            model_p = min(1.0, eff_n / mean_reach)
        else:
            # Fallback if model not yet fitted
            model_p = 0.1

        print(f"  Testing at model p = {model_p:.4f}")
        print(f"  Baseline time: {baseline_time:.2f}s")

        spearmans_h, spearmans_b = [], []
        sampled_times = []

        for seed in range(MADRID_N_RUNS):
            t0 = time.perf_counter()
            r = net.local_node_centrality_shortest(
                distances=[dist],
                compute_closeness=True,
                compute_betweenness=True,
                sample_probability=model_p,
                random_seed=seed,
                pbar_disabled=True,
            )
            sampled_times.append(time.perf_counter() - t0)

            est_harmonic = np.array(r.node_harmonic[dist])
            est_betweenness = np.array(r.node_betweenness[dist])

            sp_h, _, _, _ = compute_accuracy_metrics(true_harmonic, est_harmonic)
            sp_b, _, _, _ = compute_accuracy_metrics(true_betweenness, est_betweenness)

            if not np.isnan(sp_h):
                spearmans_h.append(sp_h)
            if not np.isnan(sp_b):
                spearmans_b.append(sp_b)

            print(".", end="", flush=True)

        effective_n = mean_reach * model_p
        avg_sampled_time = np.mean(sampled_times)
        speedup = baseline_time / avg_sampled_time if avg_sampled_time > 0 else float("inf")

        results.append(
            {
                "distance": dist,
                "mean_reach": mean_reach,
                "sample_prob": model_p,
                "effective_n": effective_n,
                "rho_closeness": np.mean(spearmans_h),
                "rho_closeness_std": np.std(spearmans_h),
                "rho_betweenness": np.mean(spearmans_b),
                "rho_betweenness_std": np.std(spearmans_b),
                "baseline_time": baseline_time,
                "sampled_time": avg_sampled_time,
                "speedup": speedup,
            }
        )

        print(f" rho_h={np.mean(spearmans_h):.3f}, rho_b={np.mean(spearmans_b):.3f}, speedup={speedup:.1f}x")

    # Save validation results
    df = pd.DataFrame(results)
    df.to_csv(validation_csv, index=False)
    print(f"\nSaved validation results: {validation_csv}")
    print(f"  {len(results)} data points")


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Generate cache data for sampling analysis")
    parser.add_argument("--synthetic", action="store_true", help="Generate only synthetic data cache")
    parser.add_argument("--gla", action="store_true", help="Generate only GLA validation data")
    parser.add_argument("--madrid", action="store_true", help="Generate only Madrid validation data")
    parser.add_argument("--force", action="store_true", help="Force regeneration of existing caches")
    args = parser.parse_args()

    print("=" * 70)
    print("00_generate_cache.py - Cache Generation for Sampling Analysis")
    print("=" * 70)

    # If none specified, run all (synthetic, GLA, and Madrid)
    any_specified = args.synthetic or args.gla or args.madrid
    do_synthetic = args.synthetic or not any_specified
    do_gla = args.gla or not any_specified
    do_madrid = args.madrid or not any_specified

    if do_synthetic:
        generate_synthetic_cache(force=args.force)

    if do_gla:
        generate_gla_cache(force=args.force)

    if do_madrid:
        generate_madrid_cache(force=args.force)

    print("\n" + "=" * 70)
    print("CACHE GENERATION COMPLETE")
    print("=" * 70)
    print("\nCache locations:")
    print(f"  Synthetic: {CACHE_DIR / f'sampling_analysis_{CACHE_VERSION}.pkl'}")
    print(f"  GLA graph: {LOCAL_CACHE_DIR / 'gla_graph.pkl'}")
    print(f"  GLA validation: {OUTPUT_DIR / 'model_validation.csv'}")
    print(f"  Madrid graph: {LOCAL_CACHE_DIR / 'madrid_graph.pkl'}")
    print(f"  Madrid validation: {OUTPUT_DIR / 'madrid_validation.csv'}")

    return 0


if __name__ == "__main__":
    exit(main())
