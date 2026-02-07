#!/usr/bin/env python
"""
00_generate_cache.py - Generate synthetic network sampling data.

Generates cached sampling results for 3 topologies × 7 distances × 12 probabilities.
This is the first step in the pipeline; all subsequent scripts depend on this cache.

Usage:
    python 00_generate_cache.py           # Generate cache (skips if exists)
    python 00_generate_cache.py --force   # Force regeneration

Outputs:
    - .cache/sampling_analysis_{CACHE_VERSION}.pkl
"""

import argparse
import sys
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
from cityseer.tools import io

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utilities import CACHE_DIR, CACHE_VERSION, apply_live_buffer_nx, compute_accuracy_metrics
from utils.substrates import generate_keyed_template

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

N_RUNS = 3  # Multiple runs for variance estimation
TEMPLATE_NAMES = ["trellis", "tree", "linear"]
SUBSTRATE_TILES = 24  # ~12km network extent
DISTANCES = [250, 500, 1000, 1500, 2000, 3000, 4000]
LIVE_INWARD_BUFFER = 4000  # 4km buffer for synthetic networks
PROBS = [0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


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
        G = apply_live_buffer_nx(G, LIVE_INWARD_BUFFER)

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
                                "max_abs_error": 0.0,
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

                    sp_h, prec_h, scale_h, iqr_h, mae_h = compute_accuracy_metrics(true_harmonic, est_harmonic)
                    sp_b, prec_b, scale_b, iqr_b, mae_b = compute_accuracy_metrics(true_betweenness, est_betweenness)

                    if not np.isnan(sp_h):
                        spearmans_h.append((sp_h, prec_h, scale_h, iqr_h, mae_h))
                    if not np.isnan(sp_b):
                        spearmans_b.append((sp_b, prec_b, scale_b, iqr_b, mae_b))

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
                            "max_abs_error": np.mean([x[4] for x in spearmans_h]),
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
                            "max_abs_error": np.mean([x[4] for x in spearmans_b]),
                        }
                    )

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
