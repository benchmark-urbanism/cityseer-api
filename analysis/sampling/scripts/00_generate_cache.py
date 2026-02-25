#!/usr/bin/env python
"""
00_generate_cache.py - Generate synthetic network sampling data.

Both closeness and betweenness use the same unified framework:
  1. Hoeffding/EW bound determines sample count: k = log(2r/δ)/(2ε²), p = min(1, k/r)
  2. Spatial stratification selects which sources (via source_indices)
  3. IPW scaling: 1/p for closeness, 1/(2p) for betweenness

Each metric has its own epsilon sweep for independent tuning.

Usage:
    python 00_generate_cache.py           # Generate cache (skips if exists)
    python 00_generate_cache.py --force   # Force regeneration

Outputs:
    - .cache/sampling_analysis_{CACHE_VERSION}.pkl
"""

import argparse
import random
import sys
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
from cityseer.tools import io

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utilities import (
    CACHE_DIR,
    CACHE_VERSION,
    HOEFFDING_DELTA,
    QUARTILE_KEYS,
    apply_live_buffer_nx,
    compute_accuracy_metrics,
    compute_hoeffding_p,
    compute_quartile_accuracy,
    mean_quartiles,
    select_spatial_sources,
)

from utils.substrates import generate_keyed_template

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

N_RUNS = 2  # Sanity check on variance
TEMPLATE_NAMES = ["trellis", "tree", "linear"]
SUBSTRATE_TILES = 24  # ~12km network extent
DISTANCES = [250, 500, 1000, 1500, 2000, 3000, 4000]
LIVE_INWARD_BUFFER = 4000  # 4km buffer for synthetic networks

# Epsilon sweeps — separate for each metric
EPSILONS_CLOSENESS = [0.05, 0.1, 0.2]
EPSILONS_BETWEENNESS = [0.05, 0.1, 0.2]

BASE_SEED = 42


# =============================================================================
# HELPERS
# =============================================================================


def _make_perfect_row(
    topo: str,
    dist: int,
    n_nodes: int,
    mean_reach: float,
    node_reach: np.ndarray,
    eps: float,
    metric: str,
    true_vals: np.ndarray,
) -> dict:
    """Build a cache row for p >= 1.0 (full computation, perfect accuracy)."""
    q_perfect = {}
    for prefix in QUARTILE_KEYS:
        for q in range(1, 5):
            if prefix == "spearman":
                q_perfect[f"{prefix}_q{q}"] = 1.0
            elif prefix == "reach":
                q_perfect[f"{prefix}_q{q}"] = np.nan
            else:  # mae, max_error
                q_perfect[f"{prefix}_q{q}"] = 0.0
    true_f32 = true_vals.astype(np.float32)
    row = {
        "topology": topo,
        "distance": dist,
        "n_nodes": n_nodes,
        "mean_reach": mean_reach,
        "node_reach": node_reach,
        "epsilon": eps,
        "sample_prob": 1.0,
        "metric": metric,
        "spearman": 1.0,
        "top_k_precision": 1.0,
        "scale_ratio": 1.0,
        "scale_iqr": 0.0,
        "max_abs_error": 0.0,
        "node_true_vals": true_f32,
        "node_est_vals": true_f32,
    }
    row.update(q_perfect)
    return row


def _run_closeness_sample(
    net,
    dist: int,
    live_mask: np.ndarray,
    n_live: int,
    p: float,
    cell_size: float,
    seed: int,
) -> np.ndarray:
    """Run one closeness sample with spatial source selection + IPW."""
    rng = random.Random(seed)
    n_sources = max(1, int(p * n_live))
    sources = select_spatial_sources(net, n_sources, cell_size, rng)
    r = net.closeness_shortest(
        distances=[dist],
        source_indices=sources,
        sample_probability=p,
        pbar_disabled=True,
    )
    return np.array(r.node_harmonic[dist])[live_mask]


def _run_betweenness_sample(
    net,
    dist: int,
    live_mask: np.ndarray,
    n_live: int,
    p: float,
    cell_size: float,
    seed: int,
) -> np.ndarray:
    """Run one betweenness sample with spatial source selection + IPW."""
    rng = random.Random(seed)
    n_sources = max(1, int(p * n_live))
    sources = select_spatial_sources(net, n_sources, cell_size, rng)
    r = net.betweenness_shortest(
        distances=[dist],
        source_indices=sources,
        sample_probability=p,
        pbar_disabled=True,
    )
    return np.array(r.node_betweenness[dist])[live_mask]


def _collect_sampled_rows(
    runs_data: list[tuple],
    quartiles_list: list[dict],
    topo: str,
    dist: int,
    n_nodes: int,
    mean_reach: float,
    node_reach: np.ndarray,
    eps: float,
    p: float,
    metric: str,
    true_vals: np.ndarray,
    est_avg: np.ndarray,
) -> dict | None:
    """Build a cache row from multiple sampled runs."""
    q_avg = mean_quartiles(quartiles_list)
    if not runs_data:
        return None
    true_f32 = true_vals.astype(np.float32)
    est_f32 = est_avg.astype(np.float32)
    row = {
        "topology": topo,
        "distance": dist,
        "n_nodes": n_nodes,
        "mean_reach": mean_reach,
        "node_reach": node_reach,
        "epsilon": eps,
        "sample_prob": p,
        "metric": metric,
        "spearman": np.mean([x[0] for x in runs_data]),
        "top_k_precision": np.mean([x[1] for x in runs_data]),
        "scale_ratio": np.mean([x[2] for x in runs_data]),
        "scale_iqr": np.mean([x[3] for x in runs_data]),
        "max_abs_error": np.max([x[4] for x in runs_data]),
        "node_true_vals": true_f32,
        "node_est_vals": est_f32,
    }
    row.update(q_avg)
    return row


# =============================================================================
# MAIN
# =============================================================================


def generate_synthetic_cache(force: bool = False):
    """Generate synthetic network sampling results cache."""
    cache_path = CACHE_DIR / f"sampling_analysis_{CACHE_VERSION}.pkl"

    if cache_path.exists() and not force:
        print(f"Synthetic cache already exists: {cache_path}")
        print("  Use --force to regenerate")
        return

    import pickle

    print("\n" + "=" * 70)
    print("GENERATING SYNTHETIC DATA CACHE")
    print("=" * 70)
    print(f"Templates: {TEMPLATE_NAMES}")
    print(f"Distances: {DISTANCES}")
    print(f"Epsilons closeness: {EPSILONS_CLOSENESS}")
    print(f"Epsilons betweenness: {EPSILONS_BETWEENNESS}")
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
        G = apply_live_buffer_nx(G, LIVE_INWARD_BUFFER)

        # Convert to cityseer format
        ndf, edf, net = io.network_structure_from_nx(G)
        live_mask = ndf["live"].values
        n_live = int(live_mask.sum())

        # ---- Ground truth: compute all distances in one pass ----
        true_closeness = net.closeness_shortest(
            distances=DISTANCES,
            pbar_disabled=True,
        )
        true_betw = net.betweenness_shortest(
            distances=DISTANCES,
            pbar_disabled=True,
        )

        for dist in DISTANCES:
            true_harmonic = np.array(true_closeness.node_harmonic[dist])[live_mask]
            true_betw_arr = np.array(true_betw.node_betweenness[dist])[live_mask]
            node_reach = np.array(true_closeness.node_density[dist])[live_mask]
            mean_reach = float(np.mean(node_reach))

            if mean_reach < 5:
                continue

            cell_size = dist / 2.0

            # ---- Closeness: Hoeffding + spatial source_indices ----
            print(f"  d={dist}m, reach={mean_reach:.0f} closeness: ", end="", flush=True)

            for eps in EPSILONS_CLOSENESS:
                p = compute_hoeffding_p(mean_reach, epsilon=eps, delta=HOEFFDING_DELTA)

                if p >= 1.0:
                    results.append(
                        _make_perfect_row(topo, dist, n_nodes, mean_reach, node_reach, eps, "harmonic", true_harmonic)
                    )
                    print(".", end="", flush=True)
                    continue

                runs_data = []
                quartiles_list = []
                est_sum = np.zeros_like(true_harmonic, dtype=np.float64)

                for seed in range(N_RUNS):
                    est = _run_closeness_sample(net, dist, live_mask, n_live, p, cell_size, BASE_SEED + seed)
                    est_sum += est

                    sp, prec, scale, iqr, mae = compute_accuracy_metrics(true_harmonic, est)
                    if not np.isnan(sp):
                        runs_data.append((sp, prec, scale, iqr, mae))
                    quartiles_list.append(compute_quartile_accuracy(true_harmonic, est, node_reach))

                est_avg = (est_sum / N_RUNS).astype(np.float64)
                row = _collect_sampled_rows(
                    runs_data, quartiles_list, topo, dist, n_nodes, mean_reach, node_reach,
                    eps, p, "harmonic", true_harmonic, est_avg,
                )
                if row is not None:
                    results.append(row)
                print(".", end="", flush=True)
            print()

            # ---- Betweenness: Hoeffding + spatial source_indices ----
            nonzero_betw = np.sum(true_betw_arr > 0)
            if nonzero_betw < 10:
                print(f"  d={dist}m betweenness: skipped (only {nonzero_betw} nonzero)")
                continue

            print(f"  d={dist}m, reach={mean_reach:.0f} betweenness: ", end="", flush=True)

            for eps in EPSILONS_BETWEENNESS:
                p = compute_hoeffding_p(mean_reach, epsilon=eps, delta=HOEFFDING_DELTA)

                if p >= 1.0:
                    results.append(
                        _make_perfect_row(
                            topo, dist, n_nodes, mean_reach, node_reach, eps, "betweenness", true_betw_arr
                        )
                    )
                    print(".", end="", flush=True)
                    continue

                runs_data = []
                quartiles_list = []
                est_sum = np.zeros_like(true_betw_arr, dtype=np.float64)

                for seed in range(N_RUNS):
                    est = _run_betweenness_sample(net, dist, live_mask, n_live, p, cell_size, BASE_SEED + seed)
                    est_sum += est

                    sp, prec, scale, iqr, mae = compute_accuracy_metrics(true_betw_arr, est)
                    if not np.isnan(sp):
                        runs_data.append((sp, prec, scale, iqr, mae))
                    quartiles_list.append(compute_quartile_accuracy(true_betw_arr, est, node_reach))

                est_avg = (est_sum / N_RUNS).astype(np.float64)
                row = _collect_sampled_rows(
                    runs_data, quartiles_list, topo, dist, n_nodes, mean_reach, node_reach,
                    eps, p, "betweenness", true_betw_arr, est_avg,
                )
                if row is not None:
                    results.append(row)
                print(".", end="", flush=True)
            print()

    # Save cache
    path = CACHE_DIR / f"sampling_analysis_{CACHE_VERSION}.pkl"
    with open(path, "wb") as f:
        pickle.dump(results, f)
    print(f"  Saved cache: {path}")
    print(f"\nGenerated {len(results)} data points")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic sampling data cache")
    parser.add_argument("--force", action="store_true", help="Force regeneration of existing cache")
    args = parser.parse_args()

    print("=" * 70)
    print("00_generate_cache.py - Synthetic Data Cache Generation")
    print("=" * 70)

    generate_synthetic_cache(force=args.force)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"  Cache: {CACHE_DIR / f'sampling_analysis_{CACHE_VERSION}.pkl'}")

    return 0


if __name__ == "__main__":
    exit(main())
