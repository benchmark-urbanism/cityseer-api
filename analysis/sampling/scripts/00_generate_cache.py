#!/usr/bin/env python
"""
00_generate_cache.py - Generate synthetic network sampling data.

Generates cached sampling results for 3 topologies × 4 distances × epsilon targets × 2 metrics.
For each distance, inverts the Hoeffding bound to find the sample_prob corresponding to each
target epsilon, then runs sampling at those specific probabilities.

Both closeness and betweenness use the same pipeline:
  1. Invert Hoeffding bound: p = log(2r/δ) / (2ε²r)
  2. Bernoulli-sample sources with inclusion probability p
  3. Pass the target probability p to Rust for IPW-corrected sampling
  4. Record realised sample fraction as a diagnostic only

Usage:
    python 00_generate_cache.py           # Generate cache (skips if exists)
    python 00_generate_cache.py --force   # Force regeneration

Outputs:
    - .cache/sampling_analysis_{CACHE_VERSION}.pkl
"""

import argparse
import math
import pickle
import random as _random
import sys
import time
import warnings
from functools import partial
from pathlib import Path

import networkx as nx
import numpy as np
from cityseer.config import GRID_SPACING, HOEFFDING_EPSILON, compute_distance_p, compute_hoeffding_p
from cityseer.tools import io

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utilities import (
    CACHE_DIR,
    CACHE_VERSION,
    HOEFFDING_DELTA,
    apply_live_buffer_nx,
    compute_accuracy_metrics,
)
from utils.substrates import generate_keyed_template

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

TEMPLATE_NAMES = ["trellis", "tree", "linear"]
SUBSTRATE_TILES = 24  # ~12km network extent
DISTANCES = [500, 1000, 2000, 4000]
LIVE_INWARD_BUFFER = 4000  # 4km buffer for synthetic networks

SEED = 42

# Epsilon-targeted sweep: these epsilons are inverted via the Hoeffding bound to
# find the exact sample_prob needed at each distance's mean_reach.
# Spans the paper threshold (0.05, unified for both metrics) plus values
# above and below so Fig 1 can show where accuracy crosses 0.95.
EPS_CLOSENESS_TARGETS = [0.025, 0.05, 0.075, 0.1, 0.15, 0.2]
EPS_BETWEENNESS_TARGETS = [0.025, 0.05, 0.075, 0.1, 0.15, 0.2]


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

    print("\n" + "=" * 70)
    print("GENERATING SYNTHETIC DATA CACHE")
    print("=" * 70)
    print(f"Templates: {TEMPLATE_NAMES}")
    print(f"Distances: {DISTANCES}")
    print(f"Epsilon targets (closeness): {EPS_CLOSENESS_TARGETS}")
    print(f"Epsilon targets (betweenness): {EPS_BETWEENNESS_TARGETS}")

    results = []
    t_total = time.perf_counter()

    for topo in TEMPLATE_NAMES:
        print(f"\n{'=' * 50}")
        print(f"Topology: {topo}")
        print(f"{'=' * 50}")

        # Generate substrate
        t0 = time.perf_counter()
        G, _, _ = generate_keyed_template(template_key=topo, tiles=SUBSTRATE_TILES, decompose=None, plot=False)
        if not nx.is_connected(G):
            largest = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest).copy()

        n_nodes = G.number_of_nodes()
        G = apply_live_buffer_nx(G, LIVE_INWARD_BUFFER)
        ndf, edf, net = io.network_structure_from_nx(G)
        live_mask = ndf["live"].values
        n_live = int(live_mask.sum())
        if n_live == 0:
            print("  Skipping topology: no live nodes after buffer")
            continue
        print(f"  Setup: {time.perf_counter() - t0:.0f}s  |  {n_nodes} nodes, {n_live} live")

        # Ground truth (single combined pass for all distances)
        t0 = time.perf_counter()
        true_combined = net.centrality_shortest(
            distances=DISTANCES, compute_closeness=True, compute_betweenness=True, pbar_disabled=True
        )
        t_gt = time.perf_counter() - t0
        print(f"  Ground truth (combined): {t_gt:.0f}s")

        for dist in DISTANCES:
            true_harmonic = np.array(true_combined.node_harmonic[dist])[live_mask]
            true_betw_arr = np.array(true_combined.node_betweenness[dist])[live_mask]
            node_reach = np.array(true_combined.node_density[dist])[live_mask]
            mean_reach = float(np.mean(node_reach))

            if mean_reach < 5:
                continue

            live_indices = [i for i in net.node_indices() if net.is_node_live(i)]
            print(f"  d={dist}m, reach={mean_reach:.0f}: ", end="", flush=True)

            eps_spec = [
                ("harmonic", EPS_CLOSENESS_TARGETS, "node_harmonic"),
                ("betweenness", EPS_BETWEENNESS_TARGETS, "node_betweenness"),
            ]
            for metric_label, eps_targets, result_attr in eps_spec:
                true_arr = true_harmonic if metric_label == "harmonic" else true_betw_arr
                rust_fn = partial(net.centrality_shortest, compute_closeness=True, compute_betweenness=False) if metric_label == "harmonic" else partial(net.centrality_shortest, compute_closeness=False, compute_betweenness=True)

                for target_eps in eps_targets:
                    target_p = compute_hoeffding_p(mean_reach, target_eps, HOEFFDING_DELTA)
                    if target_p >= 1.0:
                        # Full sampling — record as perfect
                        true_f32 = true_arr.astype(np.float32)
                        # Synthetic experiment: use actual reach (not canonical)
                        eps_val = math.sqrt(math.log(2 * mean_reach / HOEFFDING_DELTA) / (2 * mean_reach))
                        row = {
                            "topology": topo,
                            "distance": dist,
                            "n_nodes": n_nodes,
                            "mean_reach": mean_reach,
                            "node_reach": node_reach,
                            "sample_prob": target_p,
                            "actual_sample_prob": 1.0,
                            "effective_n": mean_reach,
                            "epsilon": eps_val,
                            "target_epsilon": target_eps,
                            "sweep_type": "eps_targeted",
                            "metric": metric_label,
                            "spearman": 1.0,
                            "top_k_precision": 1.0,
                            "scale_ratio": 1.0,
                            "scale_iqr": 0.0,
                            "max_abs_error": 0.0,
                            "node_true_vals": true_f32,
                            "node_est_vals": true_f32,
                        }
                        results.append(row)
                        continue

                    rng = _random.Random(SEED)
                    sources = [idx for idx in live_indices if rng.random() < target_p]
                    if not sources and n_live > 0:
                        sources = [rng.choice(live_indices)]
                    actual_p = len(sources) / n_live if n_live > 0 else 1.0
                    effective_n = mean_reach * target_p
                    # Synthetic experiment: use actual reach (not canonical)
                    eps_val = math.sqrt(math.log(2 * mean_reach / HOEFFDING_DELTA) / (2 * effective_n))

                    rust_result = rust_fn(distances=[dist], source_indices=sources, sample_probability=target_p)
                    est = np.array(getattr(rust_result, result_attr)[dist])[live_mask]
                    sp, prec, scale, iqr, mae = compute_accuracy_metrics(true_arr, est)

                    if not np.isnan(sp):
                        row = {
                            "topology": topo,
                            "distance": dist,
                            "n_nodes": n_nodes,
                            "mean_reach": mean_reach,
                            "node_reach": node_reach,
                            "sample_prob": target_p,
                            "actual_sample_prob": actual_p,
                            "effective_n": effective_n,
                            "epsilon": eps_val,
                            "target_epsilon": target_eps,
                            "sweep_type": "eps_targeted",
                            "metric": metric_label,
                            "spearman": sp,
                            "top_k_precision": prec,
                            "scale_ratio": scale,
                            "scale_iqr": iqr,
                            "max_abs_error": mae,
                            "node_true_vals": true_arr.astype(np.float32),
                            "node_est_vals": est.astype(np.float32),
                        }
                        results.append(row)
                    print("e", end="", flush=True)
            print("  eps sweep done")

            # -----------------------------------------------------------
            # Distance-based deterministic sweep
            # -----------------------------------------------------------
            det_spec = [
                ("harmonic", HOEFFDING_EPSILON, "node_harmonic", true_harmonic),
                ("betweenness", HOEFFDING_EPSILON, "node_betweenness", true_betw_arr),
            ]
            for metric_label, epsilon, result_attr, true_arr in det_spec:
                rust_fn = partial(net.centrality_shortest, compute_closeness=True, compute_betweenness=False) if metric_label == "harmonic" else partial(net.centrality_shortest, compute_closeness=False, compute_betweenness=True)
                det_p = compute_distance_p(dist, epsilon=epsilon)
                r_canonical = math.pi * dist**2 / GRID_SPACING**2
                if det_p >= 1.0:
                    true_f32 = true_arr.astype(np.float32)
                    row = {
                        "topology": topo,
                        "distance": dist,
                        "n_nodes": n_nodes,
                        "mean_reach": mean_reach,
                        "node_reach": node_reach,
                        "sample_prob": det_p,
                        "actual_sample_prob": 1.0,
                        "effective_n": r_canonical,
                        "epsilon": 0.0,
                        "target_epsilon": None,
                        "sweep_type": "distance_based",
                        "metric": metric_label,
                        "spearman": 1.0,
                        "top_k_precision": 1.0,
                        "scale_ratio": 1.0,
                        "scale_iqr": 0.0,
                        "max_abs_error": 0.0,
                        "node_true_vals": true_f32,
                        "node_est_vals": true_f32,
                    }
                    results.append(row)
                    continue

                rng = _random.Random(SEED)
                sources = [idx for idx in live_indices if rng.random() < det_p]
                if not sources and n_live > 0:
                    sources = [rng.choice(live_indices)]
                actual_p = len(sources) / n_live if n_live > 0 else 1.0
                effective_n = r_canonical * det_p

                rust_result = rust_fn(distances=[dist], source_indices=sources, sample_probability=det_p)
                est = np.array(getattr(rust_result, result_attr)[dist])[live_mask]
                sp, prec, scale, iqr, mae = compute_accuracy_metrics(true_arr, est)

                if not np.isnan(sp):
                    row = {
                        "topology": topo,
                        "distance": dist,
                        "n_nodes": n_nodes,
                        "mean_reach": mean_reach,
                        "node_reach": node_reach,
                        "sample_prob": det_p,
                        "actual_sample_prob": actual_p,
                        "effective_n": effective_n,
                        "epsilon": epsilon,
                        "target_epsilon": None,
                        "sweep_type": "distance_based",
                        "metric": metric_label,
                        "spearman": sp,
                        "top_k_precision": prec,
                        "scale_ratio": scale,
                        "scale_iqr": iqr,
                        "max_abs_error": mae,
                        "node_true_vals": true_arr.astype(np.float32),
                        "node_est_vals": est.astype(np.float32),
                    }
                    results.append(row)
                print("d", end="", flush=True)
            print("  det sweep done")

    # Save cache
    path = CACHE_DIR / f"sampling_analysis_{CACHE_VERSION}.pkl"
    with open(path, "wb") as f:
        pickle.dump(results, f)

    elapsed = time.perf_counter() - t_total
    print(f"\nGenerated {len(results)} data points in {elapsed:.0f}s ({elapsed / 60:.1f}m)")
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic sampling data cache")
    parser.add_argument("--force", action="store_true", help="Force regeneration of existing cache")
    args = parser.parse_args()

    print("=" * 70)
    print("00_generate_cache.py - Synthetic Data Cache Generation")
    print("=" * 70)

    generate_synthetic_cache(force=args.force)

    return 0


if __name__ == "__main__":
    exit(main())
