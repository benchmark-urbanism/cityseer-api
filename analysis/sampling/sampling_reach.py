# %% Sampling Analysis
"""
Sampling-based centrality approximation: accuracy analysis.

Key question: At what sampling probability can I trust BOTH the rankings
AND the magnitudes of centrality values?

This analysis measures two dimensions of accuracy:
1. RANKING: Are the relative orderings preserved? (Spearman correlation)
2. MAGNITUDE: Are the values accurate? (Scale ratio = estimated/true)

Key finding: Effective sample size (reach × p) drives both.
Rule of thumb: effective_n >= 200 gives reliable results for both dimensions.

NOTE: This analysis runs for BOTH shortest (metric) and simplest (angular)
distance heuristics, fitting separate models for each.
"""

import json
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from cityseer.tools import graphs, io
from cityseer.tools.mock import mock_graph
from scipy import optimize as scipy_optimize
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")

# Add parent directory to path for utils import
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.substrates import generate_keyed_template

# Output and cache directories
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR = SCRIPT_DIR.parent.parent / "temp" / "sampling_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Paper output directories
PAPER_DIR = SCRIPT_DIR / "paper"
FIGURES_DIR = PAPER_DIR / "figures"
TABLES_DIR = PAPER_DIR / "tables"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Cache version - increment to force re-run
CACHE_VERSION = "v12"  # Extended p range to 0.9 for complete p-dependency analysis


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


# %% Experimental Configuration
TEMPLATE_NAMES = ["trellis", "tree", "linear"]
SUBSTRATE_TILES = 10  # ~4900m network extent, supports d up to 5000m
# Distance range extended to 5000m for headline figure validation
DISTANCES = [100, 200, 300, 400, 500, 600, 700, 800, 1000, 1200, 1500, 2000, 2500, 3000, 4000, 5000]
# Comprehensive probability range with emphasis on low-p values for power correction fitting
# Low-p values (0.02-0.2) provide leverage for fitting the power correction penalty term
PROBS = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]
N_RUNS = 30  # Increased for better variance estimation at extreme p values


def generate_substrate(template_key: str) -> nx.MultiGraph:
    """Generate a tiled substrate network."""
    G, _, _ = generate_keyed_template(template_key=template_key, tiles=SUBSTRATE_TILES, decompose=None, plot=False)
    return G


def compute_accuracy_metrics(
    true_vals: np.ndarray, est_vals: np.ndarray, k_fraction: float = 0.1
) -> tuple[float, float, float, float]:
    """
    Compute ranking and magnitude accuracy metrics.

    Parameters
    ----------
    true_vals : np.ndarray
        Ground truth centrality values (from full computation)
    est_vals : np.ndarray
        Estimated centrality values (from sampled computation)
    k_fraction : float
        Fraction of nodes to use for top-k precision

    Returns
    -------
    spearman : float
        Spearman rank correlation (1.0 = perfect ranking preservation)
    top_k_precision : float
        Fraction of true top-k nodes found in estimated top-k
    scale_ratio : float
        median(estimated / true) - shows systematic over/undershoot
        1.0 = no bias, >1 = overestimating, <1 = underestimating
    scale_iqr : float
        IQR of (estimated / true) - shows consistency across nodes
    """
    # Filter out zeros/nans for valid comparison
    mask = (true_vals > 0) & np.isfinite(true_vals) & np.isfinite(est_vals)
    if mask.sum() < 10:
        return np.nan, np.nan, np.nan, np.nan

    true_masked = true_vals[mask]
    est_masked = est_vals[mask]

    # Spearman correlation (ranking preservation)
    spearman, _ = scipy_stats.spearmanr(true_masked, est_masked)

    # Top-k precision
    k = max(1, int(len(true_masked) * k_fraction))
    true_top_k = set(np.argsort(true_masked)[-k:])
    est_top_k = set(np.argsort(est_masked)[-k:])
    top_k_precision = len(true_top_k & est_top_k) / k

    # Scale ratio (magnitude accuracy)
    ratios = est_masked / true_masked
    scale_ratio = float(np.median(ratios))
    scale_iqr = float(np.percentile(ratios, 75) - np.percentile(ratios, 25))

    return spearman, top_k_precision, scale_ratio, scale_iqr


# %% Chapter 1: Correctness Verification
print("=" * 70)
print("CHAPTER 1: Correctness Verification")
print("=" * 70)
print("\nVerifying cityseer's centrality implementation against NetworkX...")

# Create test graph
G_test = mock_graph()
G_test = graphs.nx_simple_geoms(G_test)
ndf_test, edf_test, net_test = io.network_structure_from_nx(G_test)

# Use large distance to cover full graph for comparison
test_dist = 10000

# Run cityseer centrality (no sampling)
cs_result = net_test.local_node_centrality_shortest(
    distances=[test_dist], compute_closeness=True, compute_betweenness=True, pbar_disabled=True
)
cs_harmonic = np.array(cs_result.node_harmonic[test_dist])
cs_betweenness = np.array(cs_result.node_betweenness[test_dist])
cs_density = np.array(cs_result.node_density[test_dist])

# Compute NetworkX reference values
node_list = list(G_test.nodes())
node_to_idx = {n: i for i, n in enumerate(node_list)}


# Weight function that extracts length from LineString geometry
# For MultiGraph, d is a dict of {edge_key: edge_data}, so we take the min
def geom_weight(u, v, d):
    """Extract edge length from geom attribute (handles MultiGraph)."""
    return min(edge_data["geom"].length for edge_data in d.values())


# Harmonic closeness and density
nx_harmonic = np.zeros(len(G_test.nodes()))
nx_density = np.zeros(len(G_test.nodes()))
for src in G_test.nodes():
    lengths = nx.single_source_dijkstra_path_length(G_test, src, weight=geom_weight)
    for tgt, dist in lengths.items():
        if tgt != src and dist > 0:
            tgt_idx = node_to_idx[tgt]
            nx_harmonic[tgt_idx] += 1.0 / dist
            nx_density[tgt_idx] += 1

# Betweenness - use NetworkX built-in (matches cityseer at full graph distance)
nx_betw = nx.betweenness_centrality(G_test, weight=geom_weight, endpoints=False, normalized=False)
nx_betweenness = np.array([nx_betw[n] for n in node_list])

# Compare results
harmonic_diff = np.max(np.abs(cs_harmonic - nx_harmonic))
density_diff = np.max(np.abs(cs_density - nx_density))
betweenness_diff = np.max(np.abs(cs_betweenness - nx_betweenness))

correctness_results = {
    "Harmonic Closeness": {"max_diff": harmonic_diff, "status": "PASSED" if harmonic_diff < 1e-6 else "CHECK"},
    "Node Density": {"max_diff": density_diff, "status": "PASSED" if density_diff < 1e-6 else "CHECK"},
    "Betweenness": {"max_diff": betweenness_diff, "status": "PASSED" if betweenness_diff < 1e-6 else "CHECK"},
}

print("\nCorrectnessVerification Results:")
print("-" * 50)
print(f"{'Metric':<25} | {'Max Diff':<12} | Status")
print("-" * 50)
for metric, data in correctness_results.items():
    print(f"{metric:<25} | {data['max_diff']:<12.2e} | {data['status']}")
print("-" * 50)

# %% Visualize Test Network Topologies
print("\n" + "=" * 70)
print("TEST NETWORK TOPOLOGIES")
print("=" * 70)
print("\nGenerating topology visualizations (single tile of each)...")

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for i, topo in enumerate(TEMPLATE_NAMES):
    # Generate just ONE tile for visualization (not the full tiled network)
    G, _, _ = generate_keyed_template(template_key=topo, tiles=1, decompose=None, plot=False)
    if not nx.is_connected(G):
        largest = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest).copy()

    ax = axes[i]
    # Draw edges using actual geometries (not straight lines between nodes)
    for _u, _v, d in G.edges(data=True):
        if "geom" in d:
            coords = list(d["geom"].coords)
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            ax.plot(xs, ys, color="#666666", linewidth=1.0, alpha=0.8)
    # Draw nodes
    for n in G.nodes():
        ax.plot(G.nodes[n]["x"], G.nodes[n]["y"], "o", color="#333333", markersize=3, alpha=1.0)
    ax.set_title(f"{topo.title()}", fontsize=11, fontweight="bold")
    ax.set_aspect("equal")
    ax.axis("off")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "topologies.pdf", dpi=300, bbox_inches="tight")
print(f"Saved: {FIGURES_DIR / 'topologies.pdf'}")

# %% Run Sampling Analysis for BOTH distance types
print("\n" + "=" * 70)
print("CHAPTER 2: Sampling Accuracy Analysis")
print("=" * 70)
print(f"\nTemplates: {TEMPLATE_NAMES}")
print(f"Distances: {DISTANCES}")
print(f"Probabilities: {PROBS}")
print(f"Runs per config: {N_RUNS}")
print(f"Cache version: {CACHE_VERSION}")

cached = load_cache("sampling_analysis")
cached_angular = load_cache("sampling_analysis_angular")

if cached is not None and cached_angular is not None:
    print("\nLoading cached results...")
    results = cached
    results_angular = cached_angular
else:
    results = []
    results_angular = []

    for topo in TEMPLATE_NAMES:
        print(f"\n{'=' * 50}")
        print(f"Topology: {topo}")
        print(f"{'=' * 50}")

        G = generate_substrate(topo)
        if not nx.is_connected(G):
            largest = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest).copy()

        n_nodes = G.number_of_nodes()
        avg_degree = 2 * G.number_of_edges() / n_nodes
        print(f"Nodes: {n_nodes}, Avg degree: {avg_degree:.2f}")

        # Convert to cityseer format
        ndf, edf, net = io.network_structure_from_nx(G)

        for dist in DISTANCES:
            # Ground truth (full computation) for BOTH distance types
            true_shortest = net.local_node_centrality_shortest(
                distances=[dist],
                compute_closeness=True,
                compute_betweenness=True,
                pbar_disabled=True,
            )
            true_angular = net.local_node_centrality_simplest(
                distances=[dist],
                compute_closeness=True,
                compute_betweenness=True,
                pbar_disabled=True,
            )

            # Shortest distance metrics
            true_s_harmonic = np.array(true_shortest.node_harmonic[dist])
            true_s_betweenness = np.array(true_shortest.node_betweenness[dist])
            reach_s = np.array(true_shortest.node_density[dist])
            mean_reach_s = float(np.mean(reach_s))

            # Angular distance metrics
            true_a_harmonic = np.array(true_angular.node_harmonic[dist])
            true_a_betweenness = np.array(true_angular.node_betweenness[dist])
            reach_a = np.array(true_angular.node_density[dist])
            mean_reach_a = float(np.mean(reach_a))

            if mean_reach_s < 5 or mean_reach_a < 5:
                continue

            print(f"  d={dist}m, reach_shortest={mean_reach_s:.0f}, reach_angular={mean_reach_a:.0f}: ", end="", flush=True)

            for p in PROBS:
                if p == 1.0:
                    # Perfect accuracy at p=1.0 for BOTH distance types
                    for metric in ["harmonic", "betweenness"]:
                        # Shortest
                        results.append(
                            {
                                "topology": topo,
                                "distance": dist,
                                "mean_reach": mean_reach_s,
                                "sample_prob": p,
                                "effective_n": mean_reach_s,
                                "metric": metric,
                                "spearman": 1.0,
                                "spearman_std": 0.0,
                                "top_k_precision": 1.0,
                                "precision_std": 0.0,
                                "scale_ratio": 1.0,
                                "scale_ratio_std": 0.0,
                                "scale_iqr": 0.0,
                                "scale_iqr_std": 0.0,
                                "ci_width_median": 0.0,
                                "ci_width_90pct": 0.0,
                                "ci_width_95pct": 0.0,
                                "ci_width_abs_median": 0.0,
                                "ci_width_abs_90pct": 0.0,
                                "mean_centrality": np.nan,
                            }
                        )
                        # Angular
                        results_angular.append(
                            {
                                "topology": topo,
                                "distance": dist,
                                "mean_reach": mean_reach_a,
                                "sample_prob": p,
                                "effective_n": mean_reach_a,
                                "metric": metric,
                                "spearman": 1.0,
                                "spearman_std": 0.0,
                                "top_k_precision": 1.0,
                                "precision_std": 0.0,
                                "scale_ratio": 1.0,
                                "scale_ratio_std": 0.0,
                                "scale_iqr": 0.0,
                                "scale_iqr_std": 0.0,
                                "ci_width_median": 0.0,
                                "ci_width_90pct": 0.0,
                                "ci_width_95pct": 0.0,
                                "ci_width_abs_median": 0.0,
                                "ci_width_abs_90pct": 0.0,
                                "mean_centrality": np.nan,
                            }
                        )
                    continue

                # Multiple runs for variance estimation - BOTH distance types
                # Shortest
                metrics_data_s = {
                    "harmonic": {"spearmans": [], "precisions": [], "scale_ratios": [], "scale_iqrs": []},
                    "betweenness": {"spearmans": [], "precisions": [], "scale_ratios": [], "scale_iqrs": []},
                }
                true_vals_s = {"harmonic": true_s_harmonic, "betweenness": true_s_betweenness}
                per_node_estimates_s = {
                    "harmonic": np.zeros((N_RUNS, n_nodes)),
                    "betweenness": np.zeros((N_RUNS, n_nodes)),
                }

                # Angular
                metrics_data_a = {
                    "harmonic": {"spearmans": [], "precisions": [], "scale_ratios": [], "scale_iqrs": []},
                    "betweenness": {"spearmans": [], "precisions": [], "scale_ratios": [], "scale_iqrs": []},
                }
                true_vals_a = {"harmonic": true_a_harmonic, "betweenness": true_a_betweenness}
                per_node_estimates_a = {
                    "harmonic": np.zeros((N_RUNS, n_nodes)),
                    "betweenness": np.zeros((N_RUNS, n_nodes)),
                }

                for seed in range(N_RUNS):
                    # Run BOTH distance types with same seed
                    r_shortest = net.local_node_centrality_shortest(
                        distances=[dist],
                        compute_closeness=True,
                        compute_betweenness=True,
                        sample_probability=p,
                        random_seed=seed,
                        pbar_disabled=True,
                    )
                    r_angular = net.local_node_centrality_simplest(
                        distances=[dist],
                        compute_closeness=True,
                        compute_betweenness=True,
                        sample_probability=p,
                        random_seed=seed,
                        pbar_disabled=True,
                    )

                    # Extract shortest values
                    est_vals_s = {
                        "harmonic": np.array(r_shortest.node_harmonic[dist]),
                        "betweenness": np.array(r_shortest.node_betweenness[dist]),
                    }
                    per_node_estimates_s["harmonic"][seed] = est_vals_s["harmonic"]
                    per_node_estimates_s["betweenness"][seed] = est_vals_s["betweenness"]

                    # Extract angular values
                    est_vals_a = {
                        "harmonic": np.array(r_angular.node_harmonic[dist]),
                        "betweenness": np.array(r_angular.node_betweenness[dist]),
                    }
                    per_node_estimates_a["harmonic"][seed] = est_vals_a["harmonic"]
                    per_node_estimates_a["betweenness"][seed] = est_vals_a["betweenness"]

                    # Compute metrics for both
                    for metric in ["harmonic", "betweenness"]:
                        # Shortest
                        sp_s, prec_s, scale_s, iqr_s = compute_accuracy_metrics(true_vals_s[metric], est_vals_s[metric])
                        if not np.isnan(sp_s):
                            metrics_data_s[metric]["spearmans"].append(sp_s)
                            metrics_data_s[metric]["precisions"].append(prec_s)
                            metrics_data_s[metric]["scale_ratios"].append(scale_s)
                            metrics_data_s[metric]["scale_iqrs"].append(iqr_s)

                        # Angular
                        sp_a, prec_a, scale_a, iqr_a = compute_accuracy_metrics(true_vals_a[metric], est_vals_a[metric])
                        if not np.isnan(sp_a):
                            metrics_data_a[metric]["spearmans"].append(sp_a)
                            metrics_data_a[metric]["precisions"].append(prec_a)
                            metrics_data_a[metric]["scale_ratios"].append(scale_a)
                            metrics_data_a[metric]["scale_iqrs"].append(iqr_a)

                # Process results for SHORTEST
                effective_n_s = mean_reach_s * p
                for metric in ["harmonic", "betweenness"]:
                    data = metrics_data_s[metric]
                    if data["spearmans"]:
                        node_estimates = per_node_estimates_s[metric]
                        node_means = np.mean(node_estimates, axis=0)
                        node_stds = np.std(node_estimates, axis=0)

                        # Relative CI-width: 1.96 * std / mean (coefficient of variation)
                        with np.errstate(divide="ignore", invalid="ignore"):
                            node_rel_ci_widths = np.where(node_means > 0, 1.96 * node_stds / node_means, np.nan)

                        valid_ci_widths = node_rel_ci_widths[~np.isnan(node_rel_ci_widths)]
                        if len(valid_ci_widths) > 0:
                            ci_width_median = float(np.median(valid_ci_widths))
                            ci_width_90pct = float(np.percentile(valid_ci_widths, 90))
                            ci_width_95pct = float(np.percentile(valid_ci_widths, 95))
                        else:
                            ci_width_median = ci_width_90pct = ci_width_95pct = np.nan

                        # Absolute CI-width: 1.96 * std (actual uncertainty in centrality units)
                        node_abs_ci_widths = 1.96 * node_stds
                        valid_abs_ci = node_abs_ci_widths[~np.isnan(node_abs_ci_widths)]
                        if len(valid_abs_ci) > 0:
                            ci_width_abs_median = float(np.median(valid_abs_ci))
                            ci_width_abs_90pct = float(np.percentile(valid_abs_ci, 90))
                        else:
                            ci_width_abs_median = ci_width_abs_90pct = np.nan

                        # Mean centrality value (for context)
                        valid_means = node_means[node_means > 0]
                        mean_centrality = float(np.mean(valid_means)) if len(valid_means) > 0 else np.nan

                        results.append(
                            {
                                "topology": topo,
                                "distance": dist,
                                "mean_reach": mean_reach_s,
                                "sample_prob": p,
                                "effective_n": effective_n_s,
                                "metric": metric,
                                "spearman": np.mean(data["spearmans"]),
                                "spearman_std": np.std(data["spearmans"]),
                                "top_k_precision": np.mean(data["precisions"]),
                                "precision_std": np.std(data["precisions"]),
                                "scale_ratio": np.mean(data["scale_ratios"]),
                                "scale_ratio_std": np.std(data["scale_ratios"]),
                                "scale_iqr": np.mean(data["scale_iqrs"]),
                                "scale_iqr_std": np.std(data["scale_iqrs"]),
                                "ci_width_median": ci_width_median,
                                "ci_width_90pct": ci_width_90pct,
                                "ci_width_95pct": ci_width_95pct,
                                "ci_width_abs_median": ci_width_abs_median,
                                "ci_width_abs_90pct": ci_width_abs_90pct,
                                "mean_centrality": mean_centrality,
                            }
                        )

                # Process results for ANGULAR
                effective_n_a = mean_reach_a * p
                for metric in ["harmonic", "betweenness"]:
                    data = metrics_data_a[metric]
                    if data["spearmans"]:
                        node_estimates = per_node_estimates_a[metric]
                        node_means = np.mean(node_estimates, axis=0)
                        node_stds = np.std(node_estimates, axis=0)

                        # Relative CI-width: 1.96 * std / mean (coefficient of variation)
                        with np.errstate(divide="ignore", invalid="ignore"):
                            node_rel_ci_widths = np.where(node_means > 0, 1.96 * node_stds / node_means, np.nan)

                        valid_ci_widths = node_rel_ci_widths[~np.isnan(node_rel_ci_widths)]
                        if len(valid_ci_widths) > 0:
                            ci_width_median = float(np.median(valid_ci_widths))
                            ci_width_90pct = float(np.percentile(valid_ci_widths, 90))
                            ci_width_95pct = float(np.percentile(valid_ci_widths, 95))
                        else:
                            ci_width_median = ci_width_90pct = ci_width_95pct = np.nan

                        # Absolute CI-width: 1.96 * std (actual uncertainty in centrality units)
                        node_abs_ci_widths = 1.96 * node_stds
                        valid_abs_ci = node_abs_ci_widths[~np.isnan(node_abs_ci_widths)]
                        if len(valid_abs_ci) > 0:
                            ci_width_abs_median = float(np.median(valid_abs_ci))
                            ci_width_abs_90pct = float(np.percentile(valid_abs_ci, 90))
                        else:
                            ci_width_abs_median = ci_width_abs_90pct = np.nan

                        # Mean centrality value (for context)
                        valid_means = node_means[node_means > 0]
                        mean_centrality = float(np.mean(valid_means)) if len(valid_means) > 0 else np.nan

                        results_angular.append(
                            {
                                "topology": topo,
                                "distance": dist,
                                "mean_reach": mean_reach_a,
                                "sample_prob": p,
                                "effective_n": effective_n_a,
                                "metric": metric,
                                "spearman": np.mean(data["spearmans"]),
                                "spearman_std": np.std(data["spearmans"]),
                                "top_k_precision": np.mean(data["precisions"]),
                                "precision_std": np.std(data["precisions"]),
                                "scale_ratio": np.mean(data["scale_ratios"]),
                                "scale_ratio_std": np.std(data["scale_ratios"]),
                                "scale_iqr": np.mean(data["scale_iqrs"]),
                                "scale_iqr_std": np.std(data["scale_iqrs"]),
                                "ci_width_median": ci_width_median,
                                "ci_width_90pct": ci_width_90pct,
                                "ci_width_95pct": ci_width_95pct,
                                "ci_width_abs_median": ci_width_abs_median,
                                "ci_width_abs_90pct": ci_width_abs_90pct,
                                "mean_centrality": mean_centrality,
                            }
                        )

            print("done")

    save_cache("sampling_analysis", results)
    save_cache("sampling_analysis_angular", results_angular)
    print("\nResults cached.")

df = pd.DataFrame(results)
df_angular = pd.DataFrame(results_angular)
print(f"\nTotal observations (shortest): {len(df)}")
print(f"Total observations (angular): {len(df_angular)}")


# %% THE KEY TABLE: Combined Ranking + Magnitude by Effective N
print("\n" + "=" * 70)
print("KEY FINDING: Quality by Effective Sample Size (SHORTEST distances)")
print("=" * 70)
print("""
WHAT IS EFFECTIVE SAMPLE SIZE?

When sampling, each node's centrality is estimated from a subset of source nodes.
The accuracy depends on how many sampled sources contribute to each node's value:

  effective_n = reachability × sampling_probability

Where:
  - reachability: average number of nodes reachable within the distance threshold
  - sampling_probability (p): fraction of nodes used as sources (e.g., p=0.2 means 20%)

Example: If reachability=500 and p=0.4, then effective_n = 500 × 0.4 = 200
         Each node's value is estimated from ~200 sampled contributions.
""")

# Lookup table showing how reach and p combine - put it right after the explanation
print("LOOKUP TABLE: Effective Sample Size = Reachability × Probability")
print("-" * 55)
reach_values = [100, 200, 400, 600, 800, 1000]
p_values = [0.1, 0.2, 0.3, 0.4, 0.5]
header = r"Reach \ p"
print(f"{header:>10} |" + "".join([f" {p:>7.0%} |" for p in p_values]))
print("-" * 55)
for reach in reach_values:
    row = f"{reach:>10} |"
    for p in p_values:
        eff_n = reach * p
        row += f" {eff_n:>7.0f} |"
    print(row)
print("-" * 55)

print("\nThe tables below show measured accuracy at different effective_n values.\n")

# Compute combined metrics by effective_n bins
eff_n_bins = [(0, 50), (50, 100), (100, 200), (200, 400), (400, 1000)]
sampled = df[df["sample_prob"] < 1.0]


def print_accuracy_table(metric_name: str, metric_df: pd.DataFrame):
    """Print accuracy table for a specific metric."""
    print(f"\n{metric_name.upper()} - Accuracy by Effective Sample Size")
    print("-" * 52)
    header = f"{'Eff. N':>12} | {'Spearman ρ':>16} | {'Scale Ratio':>16}"
    print(header)
    print("-" * 52)

    for lo, hi in eff_n_bins:
        subset = metric_df[(metric_df["effective_n"] >= lo) & (metric_df["effective_n"] < hi)]
        if len(subset) < 3:
            continue

        sp_mean = subset["spearman"].mean()
        sp_std = subset["spearman"].std()
        scale_mean = subset["scale_ratio"].mean()
        scale_std = subset["scale_ratio"].std()

        print(f"{lo:>5}-{hi:<5} | {sp_mean:.3f} ± {sp_std:.3f}   | {scale_mean:.3f} ± {scale_std:.3f}")

    print("-" * 52)


# Print separate tables for each metric
harmonic_sampled = sampled[sampled["metric"] == "harmonic"]
betweenness_sampled = sampled[sampled["metric"] == "betweenness"]

print_accuracy_table("Harmonic Closeness", harmonic_sampled)
print_accuracy_table("Betweenness", betweenness_sampled)

print("\nINTERPRETATION:")
print("  Spearman ρ: 1.0 = perfect ranking preservation")
print("  Scale ratio: 1.0 = no bias, >1 = overestimate, <1 = underestimate")

# %% Angular distance tables
print("\n" + "=" * 70)
print("KEY FINDING: Quality by Effective Sample Size (ANGULAR distances)")
print("=" * 70)

sampled_angular = df_angular[df_angular["sample_prob"] < 1.0]
harmonic_sampled_angular = sampled_angular[sampled_angular["metric"] == "harmonic"]
betweenness_sampled_angular = sampled_angular[sampled_angular["metric"] == "betweenness"]

print_accuracy_table("Harmonic Closeness (Angular)", harmonic_sampled_angular)
print_accuracy_table("Betweenness (Angular)", betweenness_sampled_angular)


# %% Figure 1a: SHORTEST (metric) distances - Both Dimensions vs Effective N
print("\n" + "=" * 70)
print("FIGURE 1a: Accuracy vs Effective Sample Size (SHORTEST distances)")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
colors = {"trellis": "#1f77b4", "tree": "#d62728", "linear": "#ff7f0e"}

for col, metric in enumerate(["harmonic", "betweenness"]):
    metric_df = sampled[sampled["metric"] == metric]

    # Top row: Spearman (ranking)
    ax = axes[0, col]
    # Plot synthetic networks (circles)
    for topo in TEMPLATE_NAMES:
        subset = metric_df[metric_df["topology"] == topo]
        ax.scatter(subset["effective_n"], subset["spearman"], alpha=0.5, s=30, color=colors[topo], label=topo.title())

    ax.axhline(y=0.95, color="green", linestyle=":", linewidth=1.5, alpha=0.8, label="ρ = 0.95")
    ax.axhline(y=0.90, color="orange", linestyle=":", linewidth=1.5, alpha=0.8, label="ρ = 0.90")

    ax.set_xlabel("Effective Sample Size (reach × p)", fontsize=11)
    ax.set_ylabel("Spearman ρ (ranking)", fontsize=11)
    ax.set_title(f"{metric.title()}: Ranking Accuracy", fontsize=12, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlim(5, 3500)
    ax.set_ylim(0.4, 1.02)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom row: Scale ratio (magnitude)
    ax = axes[1, col]
    # Plot synthetic networks (circles)
    for topo in TEMPLATE_NAMES:
        subset = metric_df[metric_df["topology"] == topo]
        ax.scatter(
            subset["effective_n"], subset["scale_ratio"], alpha=0.5, s=30, color=colors[topo], label=topo.title()
        )

    ax.axhline(y=1.0, color="black", linestyle="-", alpha=0.7, linewidth=1.5)
    ax.axhline(y=1.05, color="orange", linestyle=":", linewidth=1.5, alpha=0.6)
    ax.axhline(y=0.95, color="orange", linestyle=":", linewidth=1.5, alpha=0.6)

    ax.set_xlabel("Effective Sample Size (reach × p)", fontsize=11)
    ax.set_ylabel("Scale Ratio (est/true)", fontsize=11)
    ax.set_title(f"{metric.title()}: Magnitude Accuracy", fontsize=12, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlim(5, 3500)
    ax.set_ylim(0.65, 1.15)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle(
    "SHORTEST Path: Sampling Accuracy vs Effective Sample Size\n(effective_n = reachability × sampling_probability)",
    fontsize=13,
    fontweight="bold",
    y=1.02,
)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "sampling_accuracy_shortest.pdf", dpi=300, bbox_inches="tight")
print(f"Saved: {FIGURES_DIR / 'sampling_accuracy_shortest.pdf'}")

# %% Figure 1b: ANGULAR (simplest) distances - Both Dimensions vs Effective N
print("\n" + "=" * 70)
print("FIGURE 1b: Accuracy vs Effective Sample Size (ANGULAR distances)")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for col, metric in enumerate(["harmonic", "betweenness"]):
    metric_df = sampled_angular[sampled_angular["metric"] == metric]

    # Top row: Spearman (ranking)
    ax = axes[0, col]
    # Plot synthetic networks (circles)
    for topo in TEMPLATE_NAMES:
        subset = metric_df[metric_df["topology"] == topo]
        ax.scatter(subset["effective_n"], subset["spearman"], alpha=0.5, s=30, color=colors[topo], label=topo.title())

    ax.axhline(y=0.95, color="green", linestyle=":", linewidth=1.5, alpha=0.8, label="ρ = 0.95")
    ax.axhline(y=0.90, color="orange", linestyle=":", linewidth=1.5, alpha=0.8, label="ρ = 0.90")

    ax.set_xlabel("Effective Sample Size (reach × p)", fontsize=11)
    ax.set_ylabel("Spearman ρ (ranking)", fontsize=11)
    ax.set_title(f"{metric.title()}: Ranking Accuracy", fontsize=12, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlim(5, 3500)
    ax.set_ylim(0.4, 1.02)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom row: Scale ratio (magnitude)
    ax = axes[1, col]
    # Plot synthetic networks (circles)
    for topo in TEMPLATE_NAMES:
        subset = metric_df[metric_df["topology"] == topo]
        ax.scatter(
            subset["effective_n"], subset["scale_ratio"], alpha=0.5, s=30, color=colors[topo], label=topo.title()
        )

    ax.axhline(y=1.0, color="black", linestyle="-", alpha=0.7, linewidth=1.5)
    ax.axhline(y=1.05, color="orange", linestyle=":", linewidth=1.5, alpha=0.6)
    ax.axhline(y=0.95, color="orange", linestyle=":", linewidth=1.5, alpha=0.6)

    ax.set_xlabel("Effective Sample Size (reach × p)", fontsize=11)
    ax.set_ylabel("Scale Ratio (est/true)", fontsize=11)
    ax.set_title(f"{metric.title()}: Magnitude Accuracy", fontsize=12, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlim(5, 3500)
    ax.set_ylim(0.65, 1.15)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle(
    "ANGULAR (Simplest) Path: Sampling Accuracy vs Effective Sample Size\n"
    "(effective_n = reachability × sampling_probability)",
    fontsize=13,
    fontweight="bold",
    y=1.02,
)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "sampling_accuracy_angular.pdf", dpi=300, bbox_inches="tight")
print(f"Saved: {FIGURES_DIR / 'sampling_accuracy_angular.pdf'}")


# %% Figure 2b: Expected Ranking Accuracy by Effective Sample Size (SHORTEST)
print("\n" + "=" * 70)
print("FIGURE 2b: Expected Ranking Accuracy by Effective Sample Size (SHORTEST)")
print("=" * 70)

# Plot B: What Spearman ρ can you expect at different effective_n values?

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Set fixed x-limit to focus on the relevant range where accuracy transitions
x_max = 2000  # Focus on 0-2000 where the accuracy transition happens

# Use adaptive binning: finer at low eff_n (dense data), coarser at high eff_n (sparse data)
eff_n_bins = np.array([
    0, 25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400,
    500, 600, 800, 1000, 1250, 1500, 2000
])

for col, metric in enumerate(["harmonic", "betweenness"]):
    metric_df = sampled[sampled["metric"] == metric]
    ax = axes[col]

    # Bin by effective_n and compute mean Spearman
    eff_n_centers = []
    spearman_means = []
    spearman_stds = []

    for i in range(len(eff_n_bins) - 1):
        lo, hi = eff_n_bins[i], eff_n_bins[i + 1]
        subset = metric_df[(metric_df["effective_n"] >= lo) & (metric_df["effective_n"] < hi)]
        if len(subset) >= 3:
            eff_n_centers.append((lo + hi) / 2)
            spearman_means.append(subset["spearman"].mean())
            spearman_stds.append(subset["spearman"].std())

    # Plot empirical data with error bars
    ax.errorbar(
        eff_n_centers,
        spearman_means,
        yerr=spearman_stds,
        fmt="o",
        markersize=5,
        capsize=3,
        alpha=0.7,
        color="steelblue",
    )

    # Add horizontal reference lines for target accuracies
    for target_rho, color in [(0.90, "green"), (0.95, "orange"), (0.99, "red")]:
        ax.axhline(y=target_rho, color=color, linestyle="--", alpha=0.5, linewidth=1)
        ax.text(x_max * 0.92, target_rho, f"ρ={target_rho}", fontsize=9, va="center", color=color)

    ax.set_xlabel("Effective Sample Size (reachability × p)", fontsize=11)
    ax.set_ylabel("Mean Spearman ρ", fontsize=11)
    ax.set_title(f"{metric.title()}", fontsize=12)
    ax.set_ylim(0.5, 1.02)
    ax.set_xlim(0, x_max)
    ax.grid(True, alpha=0.3)

plt.suptitle(
    "SHORTEST Path: Ranking Accuracy vs Effective Sample Size",
    fontsize=13,
    fontweight="bold",
    y=1.02,
)
fig.text(
    0.5,
    -0.02,
    "Points: mean Spearman ρ across all runs in each effective_n bin. "
    "Error bars: ±1 standard deviation. Dashed lines: accuracy targets.",
    ha="center",
    fontsize=9,
    style="italic",
)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "accuracy_vs_effn_shortest.pdf", dpi=300, bbox_inches="tight")
print(f"Saved: {FIGURES_DIR / 'accuracy_vs_effn_shortest.pdf'}")

# %% Figure 2c: Expected Ranking Accuracy by Effective Sample Size (ANGULAR)
print("\n" + "=" * 70)
print("FIGURE 2c: Expected Ranking Accuracy by Effective Sample Size (ANGULAR)")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Set fixed x-limit to focus on the relevant range where accuracy transitions
x_max_ang = 2000  # Focus on 0-2000 where the accuracy transition happens

# Use adaptive binning: finer at low eff_n (dense data), coarser at high eff_n (sparse data)
eff_n_bins_ang = np.array([
    0, 25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400,
    500, 600, 800, 1000, 1250, 1500, 2000
])

for col, metric in enumerate(["harmonic", "betweenness"]):
    metric_df = sampled_angular[sampled_angular["metric"] == metric]
    ax = axes[col]

    # Bin by effective_n and compute mean Spearman
    eff_n_centers = []
    spearman_means = []
    spearman_stds = []

    for i in range(len(eff_n_bins_ang) - 1):
        lo, hi = eff_n_bins_ang[i], eff_n_bins_ang[i + 1]
        subset = metric_df[(metric_df["effective_n"] >= lo) & (metric_df["effective_n"] < hi)]
        if len(subset) >= 3:
            eff_n_centers.append((lo + hi) / 2)
            spearman_means.append(subset["spearman"].mean())
            spearman_stds.append(subset["spearman"].std())

    # Plot empirical data with error bars
    ax.errorbar(
        eff_n_centers,
        spearman_means,
        yerr=spearman_stds,
        fmt="o",
        markersize=5,
        capsize=3,
        alpha=0.7,
        color="darkorange",
    )

    # Add horizontal reference lines for target accuracies
    for target_rho, color in [(0.90, "green"), (0.95, "orange"), (0.99, "red")]:
        ax.axhline(y=target_rho, color=color, linestyle="--", alpha=0.5, linewidth=1)
        ax.text(x_max_ang * 0.92, target_rho, f"ρ={target_rho}", fontsize=9, va="center", color=color)

    ax.set_xlabel("Effective Sample Size (reachability × p)", fontsize=11)
    ax.set_ylabel("Mean Spearman ρ", fontsize=11)
    ax.set_title(f"{metric.title()}", fontsize=12)
    ax.set_ylim(0.5, 1.02)
    ax.set_xlim(0, x_max_ang)
    ax.grid(True, alpha=0.3)

plt.suptitle(
    "ANGULAR Path: Ranking Accuracy vs Effective Sample Size",
    fontsize=13,
    fontweight="bold",
    y=1.02,
)
fig.text(
    0.5,
    -0.02,
    "Points: mean Spearman ρ across all runs in each effective_n bin. "
    "Error bars: ±1 standard deviation. Dashed lines: accuracy targets.",
    ha="center",
    fontsize=9,
    style="italic",
)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "accuracy_vs_effn_angular.pdf", dpi=300, bbox_inches="tight")
print(f"Saved: {FIGURES_DIR / 'accuracy_vs_effn_angular.pdf'}")


# %% Figure 2d: CI Width vs Effective Sample Size
print("\n" + "=" * 70)
print("FIGURE 2d: CI Width vs Effective Sample Size")
print("=" * 70)

# Check if CI-width columns exist in the data
if "ci_width_90pct" in sampled.columns:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Dynamically determine bin edges and x-limits based on data (with 10% margin)
    max_eff_n_ci = max(sampled["effective_n"].max(), sampled_angular["effective_n"].max())
    x_max_data = max_eff_n_ci * 1.1  # Data range with 10% margin
    x_max_data = np.ceil(x_max_data / 100) * 100  # Round up to nearest 100
    x_max_data = max(x_max_data, 500)  # Ensure minimum of 500

    # Extend x-limit for model curve extrapolation (3x the data range, up to 10000)
    x_max_ci = min(x_max_data * 3, 10000)
    x_max_ci = np.ceil(x_max_ci / 100) * 100

    # Create dynamic bins with more granularity at low eff_n, sparse at high eff_n
    bins = np.unique(np.concatenate([
        np.array([0, 10, 25, 50, 75, 100, 150, 200, 300, 400, 600, 800, 1200, 1600]),
        np.arange(2000, x_max_data + 200, 200)
    ]))

    # Load fitted model constants for plotting curves
    model_constants = None
    model_file = OUTPUT_DIR / "sampling_model_constants.json"
    if model_file.exists():
        import json
        with open(model_file) as f:
            model_constants = json.load(f)

    # Top row: SHORTEST distances
    for col, metric in enumerate(["harmonic", "betweenness"]):
        metric_df = sampled[sampled["metric"] == metric]
        ax = axes[0, col]

        # Bin by effective_n and compute 90th percentile CI widths
        bin_centers = []
        ci_90pct = []
        ci_median = []

        for i in range(len(bins) - 1):
            lo, hi = bins[i], bins[i + 1]
            subset = metric_df[(metric_df["effective_n"] >= lo) & (metric_df["effective_n"] < hi)]
            if len(subset) >= 5:
                bin_centers.append((lo + hi) / 2)
                # Use 90th percentile within bin to match the fitted model
                ci_90pct.append(np.percentile(subset["ci_width_90pct"], 90))
                ci_median.append(np.median(subset["ci_width_90pct"]))

        # Plot both median (matching fitted model) and 90th percentile (conservative) within bins
        ax.scatter(
            bin_centers,
            ci_median,
            marker="o",
            s=40,
            alpha=0.7,
            color="steelblue",
            label="Median within bins (fitted)",
            zorder=3,
        )
        ax.scatter(
            bin_centers,
            ci_90pct,
            marker="^",
            s=30,
            alpha=0.5,
            color="darkblue",
            label="90th pct within bins",
            zorder=3,
        )

        # Add fitted model curve (extending beyond data range)
        if model_constants is not None:
            shortest = model_constants.get("shortest", {})
            model_key = f"ci_width_{metric}_model"
            model = shortest.get(model_key, {})
            C, D = model.get("C"), model.get("D")

            if C is not None and D is not None:
                # Create smooth curve from 1 to x_max_ci
                eff_n_curve = np.linspace(1, x_max_ci, 500)
                ci_curve = C / np.sqrt(D + eff_n_curve)

                # Plot empirical range as solid line, extrapolated as dashed
                mask_empirical = eff_n_curve <= x_max_data
                mask_extrap = eff_n_curve > x_max_data

                ax.plot(eff_n_curve[mask_empirical], ci_curve[mask_empirical],
                       color="darkblue", linewidth=2, alpha=0.8, label="Fitted model", zorder=2)
                ax.plot(eff_n_curve[mask_extrap], ci_curve[mask_extrap],
                       color="darkblue", linewidth=2, alpha=0.5, linestyle="--",
                       label="Extrapolated", zorder=2)

                # Add shaded region for extrapolated area
                ax.axvspan(x_max_data, x_max_ci, alpha=0.05, color="gray", zorder=1)

        # Add horizontal reference lines for target CI widths
        for target_ci, color in [(0.05, "green"), (0.10, "orange"), (0.20, "red")]:
            ax.axhline(y=target_ci, color=color, linestyle="--", alpha=0.5, linewidth=1)
            ax.text(x_max_ci * 0.92, target_ci, f"±{target_ci:.0%}", fontsize=9, va="center", color=color)

        ax.set_xlabel("Effective Sample Size", fontsize=11)
        ax.set_ylabel("90th Percentile CI Width", fontsize=11)
        ax.set_title(f"{metric.title()} (Shortest Path)", fontsize=12)
        ax.set_ylim(0, 0.5)
        ax.set_xlim(0, x_max_ci)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.legend(loc="upper right", fontsize=9)

    # Bottom row: ANGULAR distances
    for col, metric in enumerate(["harmonic", "betweenness"]):
        metric_df = sampled_angular[sampled_angular["metric"] == metric]
        ax = axes[1, col]

        bin_centers = []
        ci_90pct = []
        ci_median = []

        for i in range(len(bins) - 1):
            lo, hi = bins[i], bins[i + 1]
            subset = metric_df[(metric_df["effective_n"] >= lo) & (metric_df["effective_n"] < hi)]
            if len(subset) >= 5:
                bin_centers.append((lo + hi) / 2)
                ci_90pct.append(np.percentile(subset["ci_width_90pct"], 90))
                ci_median.append(np.median(subset["ci_width_90pct"]))

        # Plot both median (matching fitted model) and 90th percentile (conservative) within bins
        ax.scatter(
            bin_centers,
            ci_median,
            marker="o",
            s=40,
            alpha=0.7,
            color="coral",
            label="Median within bins (fitted)",
            zorder=3,
        )
        ax.scatter(
            bin_centers,
            ci_90pct,
            marker="^",
            s=30,
            alpha=0.5,
            color="darkred",
            label="90th pct within bins",
            zorder=3,
        )

        # Add fitted model curve (extending beyond data range)
        if model_constants is not None:
            angular = model_constants.get("angular", {})
            model_key = f"ci_width_{metric}_model"
            model = angular.get(model_key, {})
            C, D = model.get("C"), model.get("D")

            if C is not None and D is not None:
                # Create smooth curve from 1 to x_max_ci
                eff_n_curve = np.linspace(1, x_max_ci, 500)
                ci_curve = C / np.sqrt(D + eff_n_curve)

                # Plot empirical range as solid line, extrapolated as dashed
                mask_empirical = eff_n_curve <= x_max_data
                mask_extrap = eff_n_curve > x_max_data

                ax.plot(eff_n_curve[mask_empirical], ci_curve[mask_empirical],
                       color="darkred", linewidth=2, alpha=0.8, label="Fitted model", zorder=2)
                ax.plot(eff_n_curve[mask_extrap], ci_curve[mask_extrap],
                       color="darkred", linewidth=2, alpha=0.5, linestyle="--",
                       label="Extrapolated", zorder=2)

                # Add shaded region for extrapolated area
                ax.axvspan(x_max_data, x_max_ci, alpha=0.05, color="gray", zorder=1)

        for target_ci, color in [(0.05, "green"), (0.10, "orange"), (0.20, "red")]:
            ax.axhline(y=target_ci, color=color, linestyle="--", alpha=0.5, linewidth=1)
            ax.text(x_max_ci * 0.92, target_ci, f"±{target_ci:.0%}", fontsize=9, va="center", color=color)

        ax.set_xlabel("Effective Sample Size", fontsize=11)
        ax.set_ylabel("90th Percentile CI Width", fontsize=11)
        ax.set_title(f"{metric.title()} (Angular Path)", fontsize=12)
        ax.set_ylim(0, 0.5)
        ax.set_xlim(0, x_max_ci)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.legend(loc="upper right", fontsize=9)

    plt.suptitle(
        "Per-Node Confidence Interval Width vs Effective Sample Size",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.text(
        0.5,
        -0.02,
        "Points: empirical 90th percentile CI width. Solid curves: fitted model within data range. "
        "Dashed curves: extrapolated predictions. Shaded region: extrapolation zone. "
        "Horizontal lines: precision targets (±5%, ±10%, ±20%).",
        ha="center",
        fontsize=9,
        style="italic",
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ci_width_vs_effn.pdf", dpi=300, bbox_inches="tight")
    print(f"Saved: {FIGURES_DIR / 'ci_width_vs_effn.pdf'}")
else:
    print("CI-width columns not found. Skipping CI-width figure.")


# %% Fit Continuous Models for Sampling Accuracy (SHORTEST distances)
print("\n" + "=" * 70)
print("FITTING CONTINUOUS MODELS - SHORTEST DISTANCES")
print("(Separate for Closeness and Betweenness)")
print("=" * 70)

# Fit models to predict Spearman ρ and std from effective_n
# Model for ρ: rho = 1 - A / (B + eff_n)  [approaches 1.0 as eff_n → ∞]
# Model for std: std = C / sqrt(D + eff_n)  [decreases with eff_n]
#
# Key insight: Betweenness has higher variance than closeness, so we need
# separate models. For conservative estimates, we use lower percentiles
# (10th percentile) instead of mean values per effective_n bin.

CONSERVATIVE_PERCENTILE = 10  # Use 10th percentile for conservative estimates


def rho_model(eff_n, a, b):
    """Model: rho = 1 - a / (b + eff_n)"""
    return 1 - a / (b + eff_n)


def std_model(eff_n, c, d):
    """Model: std = c / sqrt(d + eff_n)"""
    return c / np.sqrt(d + eff_n)


def bias_model(eff_n, e, f):
    """Model: scale = 1 - e / (f + eff_n), bias = 1 - scale"""
    return 1 - e / (f + eff_n)


def ci_width_model(eff_n, c, d):
    """Model: ci_width = c / sqrt(d + eff_n)

    CI width decreases with effective sample size following sqrt law.
    Same functional form as std_model but applied to relative CI widths.
    """
    return c / np.sqrt(d + eff_n)


def rho_model_power_corrected(eff_n_p_tuple, a, b, c, d):
    """Power correction model: rho = 1 - A/(B + eff_n) - C×(1-p)^D

    This model adds a penalty term for low sampling probabilities.
    At p=1, the penalty is zero and the model reduces to the baseline.
    At low p, the penalty increases, predicting lower accuracy.

    Parameters
    ----------
    eff_n_p_tuple : tuple of (eff_n, p) arrays
        Effective sample size and sampling probability values
    a, b : float
        Baseline model parameters
    c : float
        Penalty coefficient (typically 0.1-0.2)
    d : float
        Penalty exponent (typically 1.5-3.0)

    Returns
    -------
    rho : array
        Predicted Spearman correlation values
    """
    eff_n, p = eff_n_p_tuple
    baseline = 1 - a / (b + eff_n)
    penalty = c * np.power(1 - p, d)
    return baseline - penalty


def required_p_power_corrected(target_rho, reach, a, b, c, d):
    """Calculate required p for target rho using power correction model.

    The model is: rho = 1 - A/(B + eff_n) - C×(1-p)^D
    where eff_n = reach × p.

    This requires numerical solution since p appears in both eff_n and the penalty.
    We use Newton's method to find p.

    Parameters
    ----------
    target_rho : float
        Target Spearman correlation (e.g., 0.95)
    reach : float or array
        Mean reachability at the distance threshold
    a, b, c, d : float
        Power correction model parameters

    Returns
    -------
    p : float or array
        Required sampling probability (clipped to [0, 1])
    """
    reach = np.asarray(reach)
    scalar_input = reach.ndim == 0
    if scalar_input:
        reach = reach.reshape(1)

    result = np.zeros_like(reach, dtype=float)

    for i, r in enumerate(reach):
        if r <= 0:
            result[i] = 1.0
            continue

        # Define the equation: f(p) = rho_predicted - target_rho = 0
        def objective(p):
            if p <= 0:
                return -1.0  # Invalid
            if p >= 1:
                # At p=1, penalty is 0
                eff_n = r * 1.0
                return (1 - a / (b + eff_n)) - target_rho
            eff_n = r * p
            rho_pred = 1 - a / (b + eff_n) - c * np.power(1 - p, d)
            return rho_pred - target_rho

        # Binary search for p in [0, 1]
        p_lo, p_hi = 0.001, 1.0

        # Check if target is achievable
        if objective(1.0) < 0:
            # Even at p=1, can't achieve target
            result[i] = 1.0
            continue

        if objective(p_lo) > 0:
            # Even at very low p, target is exceeded
            result[i] = p_lo
            continue

        # Binary search
        for _ in range(50):
            p_mid = (p_lo + p_hi) / 2
            if objective(p_mid) < 0:
                p_lo = p_mid
            else:
                p_hi = p_mid
            if p_hi - p_lo < 1e-6:
                break

        result[i] = p_hi

    if scalar_input:
        return float(result[0])
    return np.clip(result, 0, 1)


def fit_power_corrected_rho_model_for_metric(
    metric_data: pd.DataFrame,
    metric_name: str,
    use_percentile: bool = False,
    percentile: int = 10,
) -> tuple[float, float, float, float, float, dict]:
    """
    Fit power correction rho model for a specific metric.

    Power correction model: rho = 1 - A/(B + eff_n) - C×(1-p)^D

    This model adds a penalty term for low sampling probabilities.
    At p=1, the penalty is zero and the model reduces to the baseline.
    At low p, the penalty increases, predicting lower accuracy.

    Also fits baseline model and computes model comparison statistics.

    Parameters
    ----------
    metric_data : pd.DataFrame
        Data with columns: effective_n, spearman, mean_reach, sample_prob
    metric_name : str
        Name for logging
    use_percentile : bool
        If True, bins data by effective_n and fits to percentile values
    percentile : int
        Percentile to use when binning (lower = more conservative)

    Returns
    -------
    a, b : float
        Baseline model parameters
    c, d : float
        Power penalty parameters (penalty = C × (1-p)^D)
    rmse : float
        RMSE of the power corrected model fit
    comparison : dict
        Model comparison statistics (AIC, likelihood ratio test, etc.)
    """
    # Extract the data we need
    eff_n_vals = metric_data["effective_n"].values
    spearman_vals = metric_data["spearman"].values
    p_vals = metric_data["sample_prob"].values

    # Filter valid data
    valid_mask = (
        (eff_n_vals > 0) &
        (spearman_vals > 0) &
        (spearman_vals <= 1) &
        np.isfinite(eff_n_vals) &
        np.isfinite(spearman_vals) &
        (p_vals > 0) &
        (p_vals < 1)  # Exclude p=1.0 since (1-p)^D = 0 there
    )
    eff_n_valid = eff_n_vals[valid_mask]
    spearman_valid = spearman_vals[valid_mask]
    p_valid = p_vals[valid_mask]

    if len(eff_n_valid) < 10:
        print(f"  Warning: Not enough valid data for {metric_name} power corrected model")
        return 0.0, 0.0, 0.0, 0.0, np.nan, {}

    eff_n_fit = eff_n_valid
    spearman_fit = spearman_valid
    p_fit = p_valid
    fit_type = "all data (unbinned for power correction model)"
    n_data = len(eff_n_fit)

    if n_data < 10:
        print(f"  Warning: Not enough data for {metric_name} power corrected model")
        return 0.0, 0.0, 0.0, 0.0, np.nan, {}

    print(f"\n{metric_name.upper()} POWER CORRECTION MODEL ({fit_type}):")
    print(f"  Data points: {n_data}")
    print(f"  Effective_n range: {eff_n_fit.min():.1f} - {eff_n_fit.max():.1f}")
    print(f"  Spearman range: {spearman_fit.min():.3f} - {spearman_fit.max():.3f}")
    print(f"  p range: {p_fit.min():.2f} - {p_fit.max():.2f}")

    # === Fit baseline model first ===
    try:
        params_baseline, _ = scipy_optimize.curve_fit(
            rho_model, eff_n_fit, spearman_fit,
            p0=[15, 20],
            maxfev=5000,
            bounds=([0, 0], [np.inf, np.inf])
        )
        a_baseline, b_baseline = params_baseline
        pred_baseline = rho_model(eff_n_fit, a_baseline, b_baseline)
        ss_res_baseline = np.sum((spearman_fit - pred_baseline) ** 2)
        rmse_baseline = np.sqrt(ss_res_baseline / n_data)
        n_params_baseline = 2

        # Check residual correlation with p
        corr_baseline, pval_baseline = scipy_stats.spearmanr(p_fit, spearman_fit - pred_baseline)

        print(f"  Baseline model: rho = 1 - {a_baseline:.2f} / ({b_baseline:.2f} + eff_n)")
        print(f"  Baseline RMSE: {rmse_baseline:.4f}")
        print(f"  Baseline residual-p correlation: r = {corr_baseline:.4f}, p = {pval_baseline:.4f}")
    except Exception as e:
        print(f"  Warning: Could not fit baseline model: {e}")
        return 0.0, 0.0, 0.0, 0.0, np.nan, {}

    # === Fit power correction model ===
    try:
        def power_wrapper(x, a, b, c, d):
            eff_n, p = x
            return rho_model_power_corrected((eff_n, p), a, b, c, d)

        params_corrected, _ = scipy_optimize.curve_fit(
            power_wrapper,
            (eff_n_fit, p_fit),
            spearman_fit,
            p0=[a_baseline, b_baseline, 0.1, 1.5],  # Start with baseline + small penalty
            maxfev=10000,
            bounds=([0, 0, 0, 0.5], [200, 300, 1.0, 4.0])  # Reasonable bounds
        )
        a_corr, b_corr, c_corr, d_corr = params_corrected
        pred_corrected = rho_model_power_corrected((eff_n_fit, p_fit), *params_corrected)
        ss_res_corrected = np.sum((spearman_fit - pred_corrected) ** 2)
        rmse_corrected = np.sqrt(ss_res_corrected / n_data)
        n_params_corrected = 4

        # Check residual correlation with p after correction
        corr_corrected, pval_corrected = scipy_stats.spearmanr(p_fit, spearman_fit - pred_corrected)

        print(f"  Power correction model: rho = 1 - {a_corr:.2f}/({b_corr:.2f} + eff_n) - {c_corr:.4f}×(1-p)^{d_corr:.2f}")
        print(f"  Corrected RMSE: {rmse_corrected:.4f}")
        print(f"  C (penalty coefficient): {c_corr:.4f}")
        print(f"  D (penalty exponent): {d_corr:.4f}")
        print(f"  Corrected residual-p correlation: r = {corr_corrected:.4f}, p = {pval_corrected:.4f}")
    except Exception as e:
        print(f"  Warning: Could not fit power correction model: {e}")
        # Fall back to baseline with no penalty
        return a_baseline, b_baseline, 0.0, 1.0, rmse_baseline, {
            "baseline_aic": np.nan,
            "corrected_aic": np.nan,
            "delta_aic": np.nan,
            "lr_statistic": np.nan,
            "lr_pvalue": np.nan,
            "correction_significant": False,
            "rmse_improvement": 0.0,
            "r2_baseline": np.nan,
            "r2_corrected": np.nan,
            "residual_p_corr_baseline": corr_baseline,
            "residual_p_corr_corrected": np.nan,
        }

    # === Model comparison statistics ===
    # AIC = n * ln(RSS/n) + 2k (for models fit by least squares)
    aic_baseline = n_data * np.log(ss_res_baseline / n_data) + 2 * n_params_baseline
    aic_corrected = n_data * np.log(ss_res_corrected / n_data) + 2 * n_params_corrected
    delta_aic = aic_corrected - aic_baseline  # Negative = corrected is better

    # Likelihood ratio test for nested models
    # LR = n * ln(RSS_reduced / RSS_full)
    # Under H0, LR ~ chi-squared with df = difference in parameters
    if ss_res_corrected < ss_res_baseline:
        lr_statistic = n_data * np.log(ss_res_baseline / ss_res_corrected)
    else:
        lr_statistic = 0.0
    lr_df = n_params_corrected - n_params_baseline  # = 2 (C and D parameters)
    lr_pvalue = 1 - scipy_stats.chi2.cdf(lr_statistic, lr_df) if lr_statistic > 0 else 1.0

    # R-squared
    ss_total = np.sum((spearman_fit - np.mean(spearman_fit)) ** 2)
    r2_baseline = 1 - ss_res_baseline / ss_total if ss_total > 0 else np.nan
    r2_corrected = 1 - ss_res_corrected / ss_total if ss_total > 0 else np.nan

    # RMSE improvement
    rmse_improvement = (rmse_baseline - rmse_corrected) / rmse_baseline * 100 if rmse_baseline > 0 else 0

    comparison = {
        "baseline_aic": aic_baseline,
        "corrected_aic": aic_corrected,
        "delta_aic": delta_aic,
        "lr_statistic": lr_statistic,
        "lr_pvalue": lr_pvalue,
        "correction_significant": bool(lr_pvalue < 0.05),
        "rmse_baseline": rmse_baseline,
        "rmse_corrected": rmse_corrected,
        "rmse_improvement": rmse_improvement,
        "r2_baseline": r2_baseline,
        "r2_corrected": r2_corrected,
        "a_baseline": a_baseline,
        "b_baseline": b_baseline,
        "residual_p_corr_baseline": corr_baseline,
        "residual_p_pval_baseline": pval_baseline,
        "residual_p_corr_corrected": corr_corrected,
        "residual_p_pval_corrected": pval_corrected,
    }

    print(f"\n  Model Comparison:")
    print(f"    AIC baseline: {aic_baseline:.2f}")
    print(f"    AIC corrected: {aic_corrected:.2f}")
    print(f"    Delta AIC: {delta_aic:.2f} ({'corrected better' if delta_aic < 0 else 'baseline better'})")
    print(f"    LR test: statistic={lr_statistic:.2f}, p-value={lr_pvalue:.4f}")
    print(f"    Correction significant at p<0.05: {comparison['correction_significant']}")
    print(f"    RMSE improvement: {rmse_improvement:.2f}%")
    print(f"    R2 baseline: {r2_baseline:.4f}")
    print(f"    R2 corrected: {r2_corrected:.4f}")
    print(f"    Residual-p corr removed: {abs(corr_baseline):.3f} -> {abs(corr_corrected):.3f}")

    return a_corr, b_corr, c_corr, d_corr, rmse_corrected, comparison


def fit_ci_width_model_for_metric(
    metric_data: pd.DataFrame,
    metric_name: str,
    ci_column: str = "ci_width_90pct",
    use_percentile: bool = True,
    percentile: int = 90,
) -> tuple[float, float, float]:
    """
    Fit CI-width model for a specific metric.

    Args:
        metric_data: DataFrame with effective_n and CI width columns
        metric_name: Name for logging
        ci_column: Which CI width column to use (median, 90pct, 95pct)
        use_percentile: If True, bin data and fit to upper percentile (conservative)
        percentile: Percentile to use when binning (higher = more conservative)

    Returns:
        Tuple of (c, d, rmse) for the model ci_width = c / sqrt(d + eff_n)
    """
    eff_n_vals = metric_data["effective_n"].values
    ci_vals = metric_data[ci_column].values

    # Filter valid data
    valid_mask = (eff_n_vals > 0) & (ci_vals > 0) & np.isfinite(eff_n_vals) & np.isfinite(ci_vals)
    eff_n_valid = eff_n_vals[valid_mask]
    ci_valid = ci_vals[valid_mask]

    if len(eff_n_valid) < 10:
        print(f"  Warning: Not enough valid data for {metric_name} CI-width model")
        return 1.0, 10.0, np.nan

    if use_percentile:
        # Bin data and compute upper percentiles for conservative fitting
        # Extended bins to cover full data range (up to ~3400)
        bins = np.array([0, 10, 25, 50, 75, 100, 150, 200, 300, 400, 600, 800, 1200, 1600, 2000, 2500, 3000])
        bin_centers = []
        bin_percentiles = []

        for i in range(len(bins) - 1):
            lo, hi = bins[i], bins[i + 1]
            mask = (eff_n_valid >= lo) & (eff_n_valid < hi)
            subset_ci = ci_valid[mask]

            if len(subset_ci) >= 5:  # Need enough samples
                bin_centers.append((lo + hi) / 2)
                # Use upper percentile for conservative CI width estimates
                bin_percentiles.append(np.percentile(subset_ci, percentile))

        eff_n_fit = np.array(bin_centers)
        ci_fit = np.array(bin_percentiles)
        fit_type = f"{percentile}th percentile"
    else:
        eff_n_fit = eff_n_valid
        ci_fit = ci_valid
        fit_type = "all data"

    if len(eff_n_fit) < 3:
        print(f"  Warning: Not enough bins for {metric_name} CI-width model")
        return 1.0, 10.0, np.nan

    print(f"\n{metric_name.upper()} CI-width model ({fit_type}, {ci_column}):")
    print(f"  Data points: {len(eff_n_fit)}")
    print(f"  Effective_n range: {eff_n_fit.min():.1f} - {eff_n_fit.max():.1f}")
    print(f"  CI width range: {ci_fit.min():.3f} - {ci_fit.max():.3f}")

    # Fit model with relative weighting (log-space fitting for better curve matching)
    # This gives equal importance to relative errors across the full range
    try:
        # Fit in log-space for better relative fit
        log_ci_fit = np.log(ci_fit)

        def log_ci_width_model(eff_n, c, d):
            return np.log(c / np.sqrt(d + eff_n))

        params, _ = scipy_optimize.curve_fit(log_ci_width_model, eff_n_fit, log_ci_fit, p0=[1.0, 10.0], maxfev=5000)
        c, d = params
        pred = ci_width_model(eff_n_fit, c, d)
        rmse = np.sqrt(np.mean((ci_fit - pred) ** 2))
        print(f"  Model: CI_width = {c:.3f} / sqrt({d:.2f} + eff_n)")
        print(f"  RMSE: {rmse:.4f}")
        return c, d, rmse
    except Exception as e:
        print(f"  Warning: Could not fit CI-width model: {e}")
        return 1.0, 10.0, np.nan


def fit_rho_model_for_metric(
    metric_data: pd.DataFrame,
    metric_name: str,
    use_percentile: bool = False,
    percentile: int = 10,
) -> tuple[float, float, float]:
    """
    Fit rho model for a specific metric.

    If use_percentile=True, bins data by effective_n and fits to the lower
    percentile values for conservative estimates.
    """
    eff_n_vals = metric_data["effective_n"].values
    spearman_vals = metric_data["spearman"].values

    # Filter valid data
    valid_mask = (eff_n_vals > 0) & (spearman_vals > 0) & np.isfinite(eff_n_vals) & np.isfinite(spearman_vals)
    eff_n_valid = eff_n_vals[valid_mask]
    spearman_valid = spearman_vals[valid_mask]

    if use_percentile:
        # Bin data and compute percentiles for conservative fitting
        # Extended bins to cover full data range (up to ~3400)
        bins = np.array([0, 10, 25, 50, 75, 100, 150, 200, 300, 400, 600, 800, 1200, 1600, 2000, 2500, 3000])
        bin_centers = []
        bin_percentiles = []

        for i in range(len(bins) - 1):
            lo, hi = bins[i], bins[i + 1]
            mask = (eff_n_valid >= lo) & (eff_n_valid < hi)
            subset_spearman = spearman_valid[mask]

            if len(subset_spearman) >= 10:  # Need enough samples for percentile
                bin_centers.append((lo + hi) / 2)
                bin_percentiles.append(np.percentile(subset_spearman, percentile))

        eff_n_fit = np.array(bin_centers)
        spearman_fit = np.array(bin_percentiles)
        fit_type = f"{percentile}th percentile"
    else:
        eff_n_fit = eff_n_valid
        spearman_fit = spearman_valid
        fit_type = "all data"

    print(f"\n{metric_name.upper()} model ({fit_type}):")
    print(f"  Data points: {len(eff_n_fit)}")
    print(f"  Effective_n range: {eff_n_fit.min():.1f} - {eff_n_fit.max():.1f}")
    print(f"  Spearman range: {spearman_fit.min():.3f} - {spearman_fit.max():.3f}")

    # Fit model
    try:
        params, _ = scipy_optimize.curve_fit(rho_model, eff_n_fit, spearman_fit, p0=[15, 20], maxfev=5000)
        a, b = params
        pred = rho_model(eff_n_fit, a, b)
        rmse = np.sqrt(np.mean((spearman_fit - pred) ** 2))
        print(f"  Model: ρ = 1 - {a:.2f} / ({b:.2f} + eff_n)")
        print(f"  RMSE: {rmse:.4f}")
        return a, b, rmse
    except Exception as e:
        print(f"  Warning: Could not fit model: {e}")
        return 14.11, 21.58, 0.0  # Fallback


# Separate data by metric
harmonic_data = sampled[sampled["metric"] == "harmonic"]
betweenness_data = sampled[sampled["metric"] == "betweenness"]

print(f"\nHarmonic (closeness) data points: {len(harmonic_data)}")
print(f"Betweenness data points: {len(betweenness_data)}")

# Fit models for each metric using lower percentiles (conservative)
print("\n--- Fitting Conservative Models (10th percentile) ---")

harmonic_a, harmonic_b, harmonic_rmse = fit_rho_model_for_metric(
    harmonic_data, "harmonic", use_percentile=True, percentile=CONSERVATIVE_PERCENTILE
)

betweenness_a, betweenness_b, betweenness_rmse = fit_rho_model_for_metric(
    betweenness_data, "betweenness", use_percentile=True, percentile=CONSERVATIVE_PERCENTILE
)

# Also fit mean-based models for comparison
print("\n--- Fitting Mean Models (for comparison) ---")

harmonic_mean_a, harmonic_mean_b, harmonic_mean_rmse = fit_rho_model_for_metric(
    harmonic_data, "harmonic (mean)", use_percentile=False
)

betweenness_mean_a, betweenness_mean_b, betweenness_mean_rmse = fit_rho_model_for_metric(
    betweenness_data, "betweenness (mean)", use_percentile=False
)

# === Fit Power Correction Models ===
print("\n" + "=" * 70)
print("FITTING POWER CORRECTION MODELS - SHORTEST")
print("Power correction model: rho = 1 - A/(B + eff_n) - C×(1-p)^D")
print("=" * 70)

harmonic_a_ext, harmonic_b_ext, harmonic_c_ext, harmonic_d_ext, harmonic_rmse_ext, harmonic_comparison = fit_power_corrected_rho_model_for_metric(
    harmonic_data, "harmonic", use_percentile=True, percentile=CONSERVATIVE_PERCENTILE
)

betweenness_a_ext, betweenness_b_ext, betweenness_c_ext, betweenness_d_ext, betweenness_rmse_ext, betweenness_comparison = fit_power_corrected_rho_model_for_metric(
    betweenness_data, "betweenness", use_percentile=True, percentile=CONSERVATIVE_PERCENTILE
)

# Store power correction model results for SHORTEST
power_correction_models_shortest = {
    "harmonic": {
        "A": harmonic_a_ext,
        "B": harmonic_b_ext,
        "C": harmonic_c_ext,
        "D": harmonic_d_ext,
        "rmse": harmonic_rmse_ext,
        "comparison": harmonic_comparison,
    },
    "betweenness": {
        "A": betweenness_a_ext,
        "B": betweenness_b_ext,
        "C": betweenness_c_ext,
        "D": betweenness_d_ext,
        "rmse": betweenness_rmse_ext,
        "comparison": betweenness_comparison,
    },
}

# Use betweenness conservative model as primary (it's more demanding)
# This ensures we meet accuracy targets for both metrics
rho_a, rho_b = betweenness_a, betweenness_b
rho_rmse = betweenness_rmse
print("\n*** Using BETWEENNESS conservative model for primary sampling config ***")
print("    (Betweenness has higher variance, so this ensures both metrics meet targets)")

# Compute std per effective_n bin for fitting (using betweenness for conservative)
std_bins = np.array([0, 10, 25, 50, 75, 100, 150, 200, 300, 400, 600, 1000])
bin_centers = []
bin_stds = []
for i in range(len(std_bins) - 1):
    lo, hi = std_bins[i], std_bins[i + 1]
    subset = betweenness_data[(betweenness_data["effective_n"] >= lo) & (betweenness_data["effective_n"] < hi)]
    if len(subset) > 10:
        bin_centers.append((lo + hi) / 2)
        bin_stds.append(subset["spearman"].std())

bin_centers = np.array(bin_centers)
bin_stds = np.array(bin_stds)

# Fit std model
try:
    std_params, _ = scipy_optimize.curve_fit(std_model, bin_centers, bin_stds, p0=[1, 10], maxfev=5000)
    std_c, std_d = std_params
    std_pred = std_model(bin_centers, std_c, std_d)
    std_rmse = np.sqrt(np.mean((bin_stds - std_pred) ** 2))
    print(f"\nStd model (betweenness): std = {std_c:.3f} / sqrt({std_d:.2f} + eff_n)")
    print(f"  RMSE: {std_rmse:.4f}")
except Exception as e:
    print(f"  Warning: Could not fit std model: {e}")
    std_c, std_d, std_rmse = 0.907, 10.06, None  # Fallback to previous values

# Fit bias model (scale ratio vs effective_n)
# Use all data (both metrics) for bias fitting since it's a general property
all_eff_n = sampled["effective_n"].values
all_scale = sampled["scale_ratio"].values
bias_mask = (all_eff_n > 0) & np.isfinite(all_scale) & (all_scale > 0)
eff_n_bias = all_eff_n[bias_mask]
scale_bias = all_scale[bias_mask]

try:
    bias_params, _ = scipy_optimize.curve_fit(bias_model, eff_n_bias, scale_bias, p0=[0.5, 0], maxfev=5000)
    bias_e, bias_f = bias_params
    bias_pred = bias_model(eff_n_bias, bias_e, bias_f)
    bias_rmse = np.sqrt(np.mean((scale_bias - bias_pred) ** 2))
    print(f"\nBias model: scale = 1 - {bias_e:.2f} / ({bias_f:.2f} + eff_n)")
    print(f"  RMSE: {bias_rmse:.4f}")
except Exception as e:
    print(f"  Warning: Could not fit bias model: {e}")
    bias_e, bias_f, bias_rmse = 0.46, -0.13, None  # Fallback values

# Fit CI-width models for each metric
print("\n" + "=" * 70)
print("FITTING CI-WIDTH MODELS (SHORTEST distances)")
print("=" * 70)

# Check if ci_width columns exist
if "ci_width_90pct" in harmonic_data.columns:
    # Fit CI-width models using median of 90th percentile CI widths (typical conservative)
    print("\n--- Fitting CI-width models (median of node 90th percentiles) ---")

    ci_harmonic_c, ci_harmonic_d, ci_harmonic_rmse = fit_ci_width_model_for_metric(
        harmonic_data, "harmonic", ci_column="ci_width_90pct", use_percentile=True, percentile=50
    )

    ci_betweenness_c, ci_betweenness_d, ci_betweenness_rmse = fit_ci_width_model_for_metric(
        betweenness_data, "betweenness", ci_column="ci_width_90pct", use_percentile=True, percentile=50
    )

    # Also fit median CI-width models for typical use case
    print("\n--- Fitting CI-width models (median for typical estimates) ---")

    ci_harmonic_median_c, ci_harmonic_median_d, ci_harmonic_median_rmse = fit_ci_width_model_for_metric(
        harmonic_data, "harmonic (median)", ci_column="ci_width_median", use_percentile=False
    )

    ci_betweenness_median_c, ci_betweenness_median_d, ci_betweenness_median_rmse = fit_ci_width_model_for_metric(
        betweenness_data, "betweenness (median)", ci_column="ci_width_median", use_percentile=False
    )

    # Print CI-width predictions
    print("\n" + "-" * 50)
    print("CI-WIDTH PREDICTIONS BY EFFECTIVE_N (median of node 90th percentiles)")
    print("-" * 50)
    print(f"{'eff_n':>8} | {'Harmonic CI':>12} | {'Betweenness CI':>15}")
    print("-" * 50)
    for n in [10, 25, 50, 100, 200, 400, 800]:
        ci_h = ci_width_model(n, ci_harmonic_c, ci_harmonic_d)
        ci_b = ci_width_model(n, ci_betweenness_c, ci_betweenness_d)
        print(f"{n:>8} | {ci_h:>11.1%} | {ci_b:>14.1%}")
else:
    print("\nCI-width columns not found in data. Run sampling analysis first to collect CI data.")
    ci_harmonic_c, ci_harmonic_d, ci_harmonic_rmse = 1.0, 10.0, None
    ci_betweenness_c, ci_betweenness_d, ci_betweenness_rmse = 1.0, 10.0, None
    ci_harmonic_median_c, ci_harmonic_median_d, ci_harmonic_median_rmse = 1.0, 10.0, None
    ci_betweenness_median_c, ci_betweenness_median_d, ci_betweenness_median_rmse = 1.0, 10.0, None

# Print model predictions comparing harmonic vs betweenness
print("\n" + "=" * 70)
print("MODEL COMPARISON: Harmonic vs Betweenness (Conservative / 10th percentile)")
print("=" * 70)
print(f"{'eff_n':>8} | {'Harmonic ρ':>12} | {'Betweenness ρ':>14} | {'Diff':>8}")
print("-" * 50)
for n in [10, 25, 50, 100, 200, 400, 800]:
    pred_harmonic = rho_model(n, harmonic_a, harmonic_b)
    pred_between = rho_model(n, betweenness_a, betweenness_b)
    diff = pred_harmonic - pred_between
    print(f"{n:>8} | {pred_harmonic:>12.3f} | {pred_between:>14.3f} | {diff:>+8.3f}")

# Print required effective_n for various target rho values
print("\n" + "=" * 70)
print("REQUIRED EFFECTIVE_N FOR TARGET ACCURACY (SHORTEST)")
print("=" * 70)
print(f"{'Target ρ':>10} | {'Harmonic eff_n':>15} | {'Betweenness eff_n':>18}")
print("-" * 50)
for target in [0.90, 0.95, 0.96, 0.97, 0.98, 0.99]:
    req_harmonic = harmonic_a / (1 - target) - harmonic_b
    req_between = betweenness_a / (1 - target) - betweenness_b
    print(f"{target:>10.2f} | {req_harmonic:>15.0f} | {req_between:>18.0f}")


# %% Fit Continuous Models for Sampling Accuracy (ANGULAR distances)
print("\n" + "=" * 70)
print("FITTING CONTINUOUS MODELS - ANGULAR DISTANCES")
print("(Separate for Closeness and Betweenness)")
print("=" * 70)

# Separate angular data by metric
harmonic_data_angular = sampled_angular[sampled_angular["metric"] == "harmonic"]
betweenness_data_angular = sampled_angular[sampled_angular["metric"] == "betweenness"]

print(f"\nAngular Harmonic (closeness) data points: {len(harmonic_data_angular)}")
print(f"Angular Betweenness data points: {len(betweenness_data_angular)}")

# Fit models for each metric using lower percentiles (conservative)
print("\n--- Fitting Angular Conservative Models (10th percentile) ---")

harmonic_a_angular, harmonic_b_angular, harmonic_rmse_angular = fit_rho_model_for_metric(
    harmonic_data_angular, "harmonic (angular)", use_percentile=True, percentile=CONSERVATIVE_PERCENTILE
)

betweenness_a_angular, betweenness_b_angular, betweenness_rmse_angular = fit_rho_model_for_metric(
    betweenness_data_angular, "betweenness (angular)", use_percentile=True, percentile=CONSERVATIVE_PERCENTILE
)

# === Fit Power Correction Models (ANGULAR) ===
print("\n" + "=" * 70)
print("FITTING POWER CORRECTION MODELS - ANGULAR")
print("Power correction model: rho = 1 - A/(B + eff_n) - C×(1-p)^D")
print("=" * 70)

harmonic_a_ext_ang, harmonic_b_ext_ang, harmonic_c_ext_ang, harmonic_d_ext_ang, harmonic_rmse_ext_ang, harmonic_comparison_ang = fit_power_corrected_rho_model_for_metric(
    harmonic_data_angular, "harmonic (angular)", use_percentile=True, percentile=CONSERVATIVE_PERCENTILE
)

betweenness_a_ext_ang, betweenness_b_ext_ang, betweenness_c_ext_ang, betweenness_d_ext_ang, betweenness_rmse_ext_ang, betweenness_comparison_ang = fit_power_corrected_rho_model_for_metric(
    betweenness_data_angular, "betweenness (angular)", use_percentile=True, percentile=CONSERVATIVE_PERCENTILE
)

# Store power correction model results for ANGULAR
power_correction_models_angular = {
    "harmonic": {
        "A": harmonic_a_ext_ang,
        "B": harmonic_b_ext_ang,
        "C": harmonic_c_ext_ang,
        "D": harmonic_d_ext_ang,
        "rmse": harmonic_rmse_ext_ang,
        "comparison": harmonic_comparison_ang,
    },
    "betweenness": {
        "A": betweenness_a_ext_ang,
        "B": betweenness_b_ext_ang,
        "C": betweenness_c_ext_ang,
        "D": betweenness_d_ext_ang,
        "rmse": betweenness_rmse_ext_ang,
        "comparison": betweenness_comparison_ang,
    },
}

# Compute std per effective_n bin for fitting (using betweenness for conservative)
std_bins_ang = np.array([0, 10, 25, 50, 75, 100, 150, 200, 300, 400, 600, 1000])
bin_centers_ang = []
bin_stds_ang = []
for i in range(len(std_bins_ang) - 1):
    lo, hi = std_bins_ang[i], std_bins_ang[i + 1]
    subset = betweenness_data_angular[
        (betweenness_data_angular["effective_n"] >= lo) & (betweenness_data_angular["effective_n"] < hi)
    ]
    if len(subset) > 10:
        bin_centers_ang.append((lo + hi) / 2)
        bin_stds_ang.append(subset["spearman"].std())

bin_centers_ang = np.array(bin_centers_ang)
bin_stds_ang = np.array(bin_stds_ang)

# Fit std model for angular
try:
    std_params_ang, _ = scipy_optimize.curve_fit(std_model, bin_centers_ang, bin_stds_ang, p0=[1, 10], maxfev=5000)
    std_c_angular, std_d_angular = std_params_ang
    std_pred_ang = std_model(bin_centers_ang, std_c_angular, std_d_angular)
    std_rmse_angular = np.sqrt(np.mean((bin_stds_ang - std_pred_ang) ** 2))
    print(f"\nAngular std model (betweenness): std = {std_c_angular:.3f} / sqrt({std_d_angular:.2f} + eff_n)")
    print(f"  RMSE: {std_rmse_angular:.4f}")
except Exception as e:
    print(f"  Warning: Could not fit angular std model: {e}")
    std_c_angular, std_d_angular, std_rmse_angular = 0.907, 10.06, None  # Fallback

# Fit bias model for angular
all_eff_n_ang = sampled_angular["effective_n"].values
all_scale_ang = sampled_angular["scale_ratio"].values
bias_mask_ang = (all_eff_n_ang > 0) & np.isfinite(all_scale_ang) & (all_scale_ang > 0)
eff_n_bias_ang = all_eff_n_ang[bias_mask_ang]
scale_bias_ang = all_scale_ang[bias_mask_ang]

try:
    bias_params_ang, _ = scipy_optimize.curve_fit(bias_model, eff_n_bias_ang, scale_bias_ang, p0=[0.5, 0], maxfev=5000)
    bias_e_angular, bias_f_angular = bias_params_ang
    bias_pred_ang = bias_model(eff_n_bias_ang, bias_e_angular, bias_f_angular)
    bias_rmse_angular = np.sqrt(np.mean((scale_bias_ang - bias_pred_ang) ** 2))
    print(f"\nAngular bias model: scale = 1 - {bias_e_angular:.2f} / ({bias_f_angular:.2f} + eff_n)")
    print(f"  RMSE: {bias_rmse_angular:.4f}")
except Exception as e:
    print(f"  Warning: Could not fit angular bias model: {e}")
    bias_e_angular, bias_f_angular, bias_rmse_angular = 0.46, -0.13, None  # Fallback

# Fit CI-width models for angular distances
print("\n" + "=" * 70)
print("FITTING CI-WIDTH MODELS (ANGULAR distances)")
print("=" * 70)

if "ci_width_90pct" in harmonic_data_angular.columns:
    print("\n--- Fitting Angular CI-width models (median of node 90th percentiles) ---")

    ci_harmonic_c_angular, ci_harmonic_d_angular, ci_harmonic_rmse_angular = fit_ci_width_model_for_metric(
        harmonic_data_angular, "harmonic (angular)", ci_column="ci_width_90pct", use_percentile=True, percentile=50
    )

    ci_betweenness_c_angular, ci_betweenness_d_angular, ci_betweenness_rmse_angular = fit_ci_width_model_for_metric(
        betweenness_data_angular, "betweenness (angular)", ci_column="ci_width_90pct", use_percentile=True, percentile=50
    )

    # Also fit median CI-width models
    print("\n--- Fitting Angular CI-width models (median) ---")

    ci_harmonic_median_c_angular, ci_harmonic_median_d_angular, ci_harmonic_median_rmse_angular = fit_ci_width_model_for_metric(
        harmonic_data_angular, "harmonic (angular, median)", ci_column="ci_width_median", use_percentile=False
    )

    ci_betweenness_median_c_angular, ci_betweenness_median_d_angular, ci_betweenness_median_rmse_angular = fit_ci_width_model_for_metric(
        betweenness_data_angular, "betweenness (angular, median)", ci_column="ci_width_median", use_percentile=False
    )

    # Print CI-width predictions for angular
    print("\n" + "-" * 50)
    print("CI-WIDTH PREDICTIONS BY EFFECTIVE_N (Angular, median of node 90th percentiles)")
    print("-" * 50)
    print(f"{'eff_n':>8} | {'Harmonic CI':>12} | {'Betweenness CI':>15}")
    print("-" * 50)
    for n in [10, 25, 50, 100, 200, 400, 800]:
        ci_h_ang = ci_width_model(n, ci_harmonic_c_angular, ci_harmonic_d_angular)
        ci_b_ang = ci_width_model(n, ci_betweenness_c_angular, ci_betweenness_d_angular)
        print(f"{n:>8} | {ci_h_ang:>11.1%} | {ci_b_ang:>14.1%}")
else:
    print("\nCI-width columns not found in angular data.")
    ci_harmonic_c_angular, ci_harmonic_d_angular, ci_harmonic_rmse_angular = 1.0, 10.0, None
    ci_betweenness_c_angular, ci_betweenness_d_angular, ci_betweenness_rmse_angular = 1.0, 10.0, None
    ci_harmonic_median_c_angular, ci_harmonic_median_d_angular, ci_harmonic_median_rmse_angular = 1.0, 10.0, None
    ci_betweenness_median_c_angular, ci_betweenness_median_d_angular, ci_betweenness_median_rmse_angular = 1.0, 10.0, None

# Print model comparison for angular
print("\n" + "=" * 70)
print("MODEL COMPARISON: Harmonic vs Betweenness (Angular, 10th percentile)")
print("=" * 70)
print(f"{'eff_n':>8} | {'Harmonic ρ':>12} | {'Betweenness ρ':>14} | {'Diff':>8}")
print("-" * 50)
for n in [10, 25, 50, 100, 200, 400, 800]:
    pred_harmonic_ang = rho_model(n, harmonic_a_angular, harmonic_b_angular)
    pred_between_ang = rho_model(n, betweenness_a_angular, betweenness_b_angular)
    diff_ang = pred_harmonic_ang - pred_between_ang
    print(f"{n:>8} | {pred_harmonic_ang:>12.3f} | {pred_between_ang:>14.3f} | {diff_ang:>+8.3f}")

# Print required effective_n for various target rho values (angular)
print("\n" + "=" * 70)
print("REQUIRED EFFECTIVE_N FOR TARGET ACCURACY (ANGULAR)")
print("=" * 70)
print(f"{'Target ρ':>10} | {'Harmonic eff_n':>15} | {'Betweenness eff_n':>18}")
print("-" * 50)
for target in [0.90, 0.95, 0.96, 0.97, 0.98, 0.99]:
    req_harmonic_ang = harmonic_a_angular / (1 - target) - harmonic_b_angular
    req_between_ang = betweenness_a_angular / (1 - target) - betweenness_b_angular
    print(f"{target:>10.2f} | {req_harmonic_ang:>15.0f} | {req_between_ang:>18.0f}")


# Save model constants to JSON for syncing to config.py
# Include both metric-specific and combined conservative models for BOTH distance types
model_constants = {
    "generated": datetime.now().isoformat(timespec="seconds"),
    "conservative_percentile": CONSERVATIVE_PERCENTILE,
    # === SHORTEST (metric) distance models ===
    "shortest": {
        "data_points": {
            "harmonic": len(harmonic_data),
            "betweenness": len(betweenness_data),
        },
        "harmonic_model": {
            "formula": "rho = 1 - A / (B + eff_n)",
            "A": round(harmonic_a, 2),
            "B": round(harmonic_b, 2),
            "rmse": round(harmonic_rmse, 4),
            "note": f"Fitted to {CONSERVATIVE_PERCENTILE}th percentile for conservative estimates",
        },
        "betweenness_model": {
            "formula": "rho = 1 - A / (B + eff_n)",
            "A": round(betweenness_a, 2),
            "B": round(betweenness_b, 2),
            "rmse": round(betweenness_rmse, 4),
            "note": f"Fitted to {CONSERVATIVE_PERCENTILE}th percentile for conservative estimates",
        },
        # Power correction models
        "harmonic_model_power_corrected": {
            "formula": "rho = 1 - A/(B + eff_n) - C×(1-p)^D",
            "A": round(power_correction_models_shortest["harmonic"]["A"], 2) if power_correction_models_shortest["harmonic"]["A"] else None,
            "B": round(power_correction_models_shortest["harmonic"]["B"], 2) if power_correction_models_shortest["harmonic"]["B"] else None,
            "C": round(power_correction_models_shortest["harmonic"]["C"], 4) if power_correction_models_shortest["harmonic"]["C"] else None,
            "D": round(power_correction_models_shortest["harmonic"]["D"], 4) if power_correction_models_shortest["harmonic"]["D"] else None,
            "rmse": round(power_correction_models_shortest["harmonic"]["rmse"], 4) if power_correction_models_shortest["harmonic"]["rmse"] and not np.isnan(power_correction_models_shortest["harmonic"]["rmse"]) else None,
            "comparison": {
                "baseline_aic": round(power_correction_models_shortest["harmonic"]["comparison"].get("baseline_aic", np.nan), 2) if power_correction_models_shortest["harmonic"]["comparison"].get("baseline_aic") and not np.isnan(power_correction_models_shortest["harmonic"]["comparison"].get("baseline_aic", np.nan)) else None,
                "corrected_aic": round(power_correction_models_shortest["harmonic"]["comparison"].get("corrected_aic", np.nan), 2) if power_correction_models_shortest["harmonic"]["comparison"].get("corrected_aic") and not np.isnan(power_correction_models_shortest["harmonic"]["comparison"].get("corrected_aic", np.nan)) else None,
                "delta_aic": round(power_correction_models_shortest["harmonic"]["comparison"].get("delta_aic", np.nan), 2) if power_correction_models_shortest["harmonic"]["comparison"].get("delta_aic") and not np.isnan(power_correction_models_shortest["harmonic"]["comparison"].get("delta_aic", np.nan)) else None,
                "lr_pvalue": round(power_correction_models_shortest["harmonic"]["comparison"].get("lr_pvalue", np.nan), 4) if power_correction_models_shortest["harmonic"]["comparison"].get("lr_pvalue") and not np.isnan(power_correction_models_shortest["harmonic"]["comparison"].get("lr_pvalue", np.nan)) else None,
                "correction_significant": power_correction_models_shortest["harmonic"]["comparison"].get("correction_significant", False),
                "rmse_improvement_pct": round(power_correction_models_shortest["harmonic"]["comparison"].get("rmse_improvement", 0), 2),
                "r2_baseline": round(power_correction_models_shortest["harmonic"]["comparison"].get("r2_baseline", np.nan), 4) if power_correction_models_shortest["harmonic"]["comparison"].get("r2_baseline") and not np.isnan(power_correction_models_shortest["harmonic"]["comparison"].get("r2_baseline", np.nan)) else None,
                "r2_corrected": round(power_correction_models_shortest["harmonic"]["comparison"].get("r2_corrected", np.nan), 4) if power_correction_models_shortest["harmonic"]["comparison"].get("r2_corrected") and not np.isnan(power_correction_models_shortest["harmonic"]["comparison"].get("r2_corrected", np.nan)) else None,
                "residual_p_corr_baseline": round(power_correction_models_shortest["harmonic"]["comparison"].get("residual_p_corr_baseline", np.nan), 4) if power_correction_models_shortest["harmonic"]["comparison"].get("residual_p_corr_baseline") and not np.isnan(power_correction_models_shortest["harmonic"]["comparison"].get("residual_p_corr_baseline", np.nan)) else None,
                "residual_p_corr_corrected": round(power_correction_models_shortest["harmonic"]["comparison"].get("residual_p_corr_corrected", np.nan), 4) if power_correction_models_shortest["harmonic"]["comparison"].get("residual_p_corr_corrected") and not np.isnan(power_correction_models_shortest["harmonic"]["comparison"].get("residual_p_corr_corrected", np.nan)) else None,
            },
            "note": f"Power correction model for low-p penalty, fitted to {CONSERVATIVE_PERCENTILE}th percentile",
        },
        "betweenness_model_power_corrected": {
            "formula": "rho = 1 - A/(B + eff_n) - C×(1-p)^D",
            "A": round(power_correction_models_shortest["betweenness"]["A"], 2) if power_correction_models_shortest["betweenness"]["A"] else None,
            "B": round(power_correction_models_shortest["betweenness"]["B"], 2) if power_correction_models_shortest["betweenness"]["B"] else None,
            "C": round(power_correction_models_shortest["betweenness"]["C"], 4) if power_correction_models_shortest["betweenness"]["C"] else None,
            "D": round(power_correction_models_shortest["betweenness"]["D"], 4) if power_correction_models_shortest["betweenness"]["D"] else None,
            "rmse": round(power_correction_models_shortest["betweenness"]["rmse"], 4) if power_correction_models_shortest["betweenness"]["rmse"] and not np.isnan(power_correction_models_shortest["betweenness"]["rmse"]) else None,
            "comparison": {
                "baseline_aic": round(power_correction_models_shortest["betweenness"]["comparison"].get("baseline_aic", np.nan), 2) if power_correction_models_shortest["betweenness"]["comparison"].get("baseline_aic") and not np.isnan(power_correction_models_shortest["betweenness"]["comparison"].get("baseline_aic", np.nan)) else None,
                "corrected_aic": round(power_correction_models_shortest["betweenness"]["comparison"].get("corrected_aic", np.nan), 2) if power_correction_models_shortest["betweenness"]["comparison"].get("corrected_aic") and not np.isnan(power_correction_models_shortest["betweenness"]["comparison"].get("corrected_aic", np.nan)) else None,
                "delta_aic": round(power_correction_models_shortest["betweenness"]["comparison"].get("delta_aic", np.nan), 2) if power_correction_models_shortest["betweenness"]["comparison"].get("delta_aic") and not np.isnan(power_correction_models_shortest["betweenness"]["comparison"].get("delta_aic", np.nan)) else None,
                "lr_pvalue": round(power_correction_models_shortest["betweenness"]["comparison"].get("lr_pvalue", np.nan), 4) if power_correction_models_shortest["betweenness"]["comparison"].get("lr_pvalue") and not np.isnan(power_correction_models_shortest["betweenness"]["comparison"].get("lr_pvalue", np.nan)) else None,
                "correction_significant": power_correction_models_shortest["betweenness"]["comparison"].get("correction_significant", False),
                "rmse_improvement_pct": round(power_correction_models_shortest["betweenness"]["comparison"].get("rmse_improvement", 0), 2),
                "r2_baseline": round(power_correction_models_shortest["betweenness"]["comparison"].get("r2_baseline", np.nan), 4) if power_correction_models_shortest["betweenness"]["comparison"].get("r2_baseline") and not np.isnan(power_correction_models_shortest["betweenness"]["comparison"].get("r2_baseline", np.nan)) else None,
                "r2_corrected": round(power_correction_models_shortest["betweenness"]["comparison"].get("r2_corrected", np.nan), 4) if power_correction_models_shortest["betweenness"]["comparison"].get("r2_corrected") and not np.isnan(power_correction_models_shortest["betweenness"]["comparison"].get("r2_corrected", np.nan)) else None,
                "residual_p_corr_baseline": round(power_correction_models_shortest["betweenness"]["comparison"].get("residual_p_corr_baseline", np.nan), 4) if power_correction_models_shortest["betweenness"]["comparison"].get("residual_p_corr_baseline") and not np.isnan(power_correction_models_shortest["betweenness"]["comparison"].get("residual_p_corr_baseline", np.nan)) else None,
                "residual_p_corr_corrected": round(power_correction_models_shortest["betweenness"]["comparison"].get("residual_p_corr_corrected", np.nan), 4) if power_correction_models_shortest["betweenness"]["comparison"].get("residual_p_corr_corrected") and not np.isnan(power_correction_models_shortest["betweenness"]["comparison"].get("residual_p_corr_corrected", np.nan)) else None,
            },
            "note": f"Power correction model for low-p penalty, fitted to {CONSERVATIVE_PERCENTILE}th percentile",
        },
        "std_model": {
            "formula": "std = C / sqrt(D + eff_n)",
            "C": round(std_c, 3),
            "D": round(std_d, 2),
            "rmse": round(std_rmse, 4) if std_rmse is not None else None,
        },
        "bias_model": {
            "formula": "scale = 1 - E / (F + eff_n)",
            "E": round(bias_e, 2),
            "F": round(bias_f, 2),
            "rmse": round(bias_rmse, 4) if bias_rmse is not None else None,
        },
        # CI-width models for per-node uncertainty estimation
        "ci_width_harmonic_model": {
            "formula": "ci_width = C / sqrt(D + eff_n)",
            "C": round(ci_harmonic_c, 3) if ci_harmonic_rmse is not None else None,
            "D": round(ci_harmonic_d, 2) if ci_harmonic_rmse is not None else None,
            "rmse": round(ci_harmonic_rmse, 4) if ci_harmonic_rmse is not None else None,
            "note": "Fitted to median of node 90th percentile CI widths (typical conservative)",
        },
        "ci_width_betweenness_model": {
            "formula": "ci_width = C / sqrt(D + eff_n)",
            "C": round(ci_betweenness_c, 3) if ci_betweenness_rmse is not None else None,
            "D": round(ci_betweenness_d, 2) if ci_betweenness_rmse is not None else None,
            "rmse": round(ci_betweenness_rmse, 4) if ci_betweenness_rmse is not None else None,
            "note": "Fitted to median of node 90th percentile CI widths (typical conservative)",
        },
        "ci_width_harmonic_median_model": {
            "formula": "ci_width = C / sqrt(D + eff_n)",
            "C": round(ci_harmonic_median_c, 3) if ci_harmonic_median_rmse is not None else None,
            "D": round(ci_harmonic_median_d, 2) if ci_harmonic_median_rmse is not None else None,
            "rmse": round(ci_harmonic_median_rmse, 4) if ci_harmonic_median_rmse is not None else None,
            "note": "Fitted to median node CI widths (typical case)",
        },
        "ci_width_betweenness_median_model": {
            "formula": "ci_width = C / sqrt(D + eff_n)",
            "C": round(ci_betweenness_median_c, 3) if ci_betweenness_median_rmse is not None else None,
            "D": round(ci_betweenness_median_d, 2) if ci_betweenness_median_rmse is not None else None,
            "rmse": round(ci_betweenness_median_rmse, 4) if ci_betweenness_median_rmse is not None else None,
            "note": "Fitted to median node CI widths (typical case)",
        },
    },
    # === ANGULAR (simplest) distance models ===
    "angular": {
        "data_points": {
            "harmonic": len(harmonic_data_angular),
            "betweenness": len(betweenness_data_angular),
        },
        "harmonic_model": {
            "formula": "rho = 1 - A / (B + eff_n)",
            "A": round(harmonic_a_angular, 2),
            "B": round(harmonic_b_angular, 2),
            "rmse": round(harmonic_rmse_angular, 4),
            "note": f"Fitted to {CONSERVATIVE_PERCENTILE}th percentile for conservative estimates",
        },
        "betweenness_model": {
            "formula": "rho = 1 - A / (B + eff_n)",
            "A": round(betweenness_a_angular, 2),
            "B": round(betweenness_b_angular, 2),
            "rmse": round(betweenness_rmse_angular, 4),
            "note": f"Fitted to {CONSERVATIVE_PERCENTILE}th percentile for conservative estimates",
        },
        # Power correction models (Angular)
        "harmonic_model_power_corrected": {
            "formula": "rho = 1 - A/(B + eff_n) - C×(1-p)^D",
            "A": round(power_correction_models_angular["harmonic"]["A"], 2) if power_correction_models_angular["harmonic"]["A"] else None,
            "B": round(power_correction_models_angular["harmonic"]["B"], 2) if power_correction_models_angular["harmonic"]["B"] else None,
            "C": round(power_correction_models_angular["harmonic"]["C"], 4) if power_correction_models_angular["harmonic"]["C"] else None,
            "D": round(power_correction_models_angular["harmonic"]["D"], 4) if power_correction_models_angular["harmonic"]["D"] else None,
            "rmse": round(power_correction_models_angular["harmonic"]["rmse"], 4) if power_correction_models_angular["harmonic"]["rmse"] and not np.isnan(power_correction_models_angular["harmonic"]["rmse"]) else None,
            "comparison": {
                "baseline_aic": round(power_correction_models_angular["harmonic"]["comparison"].get("baseline_aic", np.nan), 2) if power_correction_models_angular["harmonic"]["comparison"].get("baseline_aic") and not np.isnan(power_correction_models_angular["harmonic"]["comparison"].get("baseline_aic", np.nan)) else None,
                "corrected_aic": round(power_correction_models_angular["harmonic"]["comparison"].get("corrected_aic", np.nan), 2) if power_correction_models_angular["harmonic"]["comparison"].get("corrected_aic") and not np.isnan(power_correction_models_angular["harmonic"]["comparison"].get("corrected_aic", np.nan)) else None,
                "delta_aic": round(power_correction_models_angular["harmonic"]["comparison"].get("delta_aic", np.nan), 2) if power_correction_models_angular["harmonic"]["comparison"].get("delta_aic") and not np.isnan(power_correction_models_angular["harmonic"]["comparison"].get("delta_aic", np.nan)) else None,
                "lr_pvalue": round(power_correction_models_angular["harmonic"]["comparison"].get("lr_pvalue", np.nan), 4) if power_correction_models_angular["harmonic"]["comparison"].get("lr_pvalue") and not np.isnan(power_correction_models_angular["harmonic"]["comparison"].get("lr_pvalue", np.nan)) else None,
                "correction_significant": power_correction_models_angular["harmonic"]["comparison"].get("correction_significant", False),
                "rmse_improvement_pct": round(power_correction_models_angular["harmonic"]["comparison"].get("rmse_improvement", 0), 2),
                "r2_baseline": round(power_correction_models_angular["harmonic"]["comparison"].get("r2_baseline", np.nan), 4) if power_correction_models_angular["harmonic"]["comparison"].get("r2_baseline") and not np.isnan(power_correction_models_angular["harmonic"]["comparison"].get("r2_baseline", np.nan)) else None,
                "r2_corrected": round(power_correction_models_angular["harmonic"]["comparison"].get("r2_corrected", np.nan), 4) if power_correction_models_angular["harmonic"]["comparison"].get("r2_corrected") and not np.isnan(power_correction_models_angular["harmonic"]["comparison"].get("r2_corrected", np.nan)) else None,
                "residual_p_corr_baseline": round(power_correction_models_angular["harmonic"]["comparison"].get("residual_p_corr_baseline", np.nan), 4) if power_correction_models_angular["harmonic"]["comparison"].get("residual_p_corr_baseline") and not np.isnan(power_correction_models_angular["harmonic"]["comparison"].get("residual_p_corr_baseline", np.nan)) else None,
                "residual_p_corr_corrected": round(power_correction_models_angular["harmonic"]["comparison"].get("residual_p_corr_corrected", np.nan), 4) if power_correction_models_angular["harmonic"]["comparison"].get("residual_p_corr_corrected") and not np.isnan(power_correction_models_angular["harmonic"]["comparison"].get("residual_p_corr_corrected", np.nan)) else None,
            },
            "note": f"Power correction model for low-p penalty, fitted to {CONSERVATIVE_PERCENTILE}th percentile",
        },
        "betweenness_model_power_corrected": {
            "formula": "rho = 1 - A/(B + eff_n) - C×(1-p)^D",
            "A": round(power_correction_models_angular["betweenness"]["A"], 2) if power_correction_models_angular["betweenness"]["A"] else None,
            "B": round(power_correction_models_angular["betweenness"]["B"], 2) if power_correction_models_angular["betweenness"]["B"] else None,
            "C": round(power_correction_models_angular["betweenness"]["C"], 4) if power_correction_models_angular["betweenness"]["C"] else None,
            "D": round(power_correction_models_angular["betweenness"]["D"], 4) if power_correction_models_angular["betweenness"]["D"] else None,
            "rmse": round(power_correction_models_angular["betweenness"]["rmse"], 4) if power_correction_models_angular["betweenness"]["rmse"] and not np.isnan(power_correction_models_angular["betweenness"]["rmse"]) else None,
            "comparison": {
                "baseline_aic": round(power_correction_models_angular["betweenness"]["comparison"].get("baseline_aic", np.nan), 2) if power_correction_models_angular["betweenness"]["comparison"].get("baseline_aic") and not np.isnan(power_correction_models_angular["betweenness"]["comparison"].get("baseline_aic", np.nan)) else None,
                "corrected_aic": round(power_correction_models_angular["betweenness"]["comparison"].get("corrected_aic", np.nan), 2) if power_correction_models_angular["betweenness"]["comparison"].get("corrected_aic") and not np.isnan(power_correction_models_angular["betweenness"]["comparison"].get("corrected_aic", np.nan)) else None,
                "delta_aic": round(power_correction_models_angular["betweenness"]["comparison"].get("delta_aic", np.nan), 2) if power_correction_models_angular["betweenness"]["comparison"].get("delta_aic") and not np.isnan(power_correction_models_angular["betweenness"]["comparison"].get("delta_aic", np.nan)) else None,
                "lr_pvalue": round(power_correction_models_angular["betweenness"]["comparison"].get("lr_pvalue", np.nan), 4) if power_correction_models_angular["betweenness"]["comparison"].get("lr_pvalue") and not np.isnan(power_correction_models_angular["betweenness"]["comparison"].get("lr_pvalue", np.nan)) else None,
                "correction_significant": power_correction_models_angular["betweenness"]["comparison"].get("correction_significant", False),
                "rmse_improvement_pct": round(power_correction_models_angular["betweenness"]["comparison"].get("rmse_improvement", 0), 2),
                "r2_baseline": round(power_correction_models_angular["betweenness"]["comparison"].get("r2_baseline", np.nan), 4) if power_correction_models_angular["betweenness"]["comparison"].get("r2_baseline") and not np.isnan(power_correction_models_angular["betweenness"]["comparison"].get("r2_baseline", np.nan)) else None,
                "r2_corrected": round(power_correction_models_angular["betweenness"]["comparison"].get("r2_corrected", np.nan), 4) if power_correction_models_angular["betweenness"]["comparison"].get("r2_corrected") and not np.isnan(power_correction_models_angular["betweenness"]["comparison"].get("r2_corrected", np.nan)) else None,
                "residual_p_corr_baseline": round(power_correction_models_angular["betweenness"]["comparison"].get("residual_p_corr_baseline", np.nan), 4) if power_correction_models_angular["betweenness"]["comparison"].get("residual_p_corr_baseline") and not np.isnan(power_correction_models_angular["betweenness"]["comparison"].get("residual_p_corr_baseline", np.nan)) else None,
                "residual_p_corr_corrected": round(power_correction_models_angular["betweenness"]["comparison"].get("residual_p_corr_corrected", np.nan), 4) if power_correction_models_angular["betweenness"]["comparison"].get("residual_p_corr_corrected") and not np.isnan(power_correction_models_angular["betweenness"]["comparison"].get("residual_p_corr_corrected", np.nan)) else None,
            },
            "note": f"Power correction model for low-p penalty, fitted to {CONSERVATIVE_PERCENTILE}th percentile",
        },
        "std_model": {
            "formula": "std = C / sqrt(D + eff_n)",
            "C": round(std_c_angular, 3),
            "D": round(std_d_angular, 2),
            "rmse": round(std_rmse_angular, 4) if std_rmse_angular is not None else None,
        },
        "bias_model": {
            "formula": "scale = 1 - E / (F + eff_n)",
            "E": round(bias_e_angular, 2),
            "F": round(bias_f_angular, 2),
            "rmse": round(bias_rmse_angular, 4) if bias_rmse_angular is not None else None,
        },
        # CI-width models for angular distances
        "ci_width_harmonic_model": {
            "formula": "ci_width = C / sqrt(D + eff_n)",
            "C": round(ci_harmonic_c_angular, 3) if ci_harmonic_rmse_angular is not None else None,
            "D": round(ci_harmonic_d_angular, 2) if ci_harmonic_rmse_angular is not None else None,
            "rmse": round(ci_harmonic_rmse_angular, 4) if ci_harmonic_rmse_angular is not None else None,
            "note": "Fitted to median of node 90th percentile CI widths (typical conservative)",
        },
        "ci_width_betweenness_model": {
            "formula": "ci_width = C / sqrt(D + eff_n)",
            "C": round(ci_betweenness_c_angular, 3) if ci_betweenness_rmse_angular is not None else None,
            "D": round(ci_betweenness_d_angular, 2) if ci_betweenness_rmse_angular is not None else None,
            "rmse": round(ci_betweenness_rmse_angular, 4) if ci_betweenness_rmse_angular is not None else None,
            "note": "Fitted to median of node 90th percentile CI widths (typical conservative)",
        },
        "ci_width_harmonic_median_model": {
            "formula": "ci_width = C / sqrt(D + eff_n)",
            "C": round(ci_harmonic_median_c_angular, 3) if ci_harmonic_median_rmse_angular is not None else None,
            "D": round(ci_harmonic_median_d_angular, 2) if ci_harmonic_median_rmse_angular is not None else None,
            "rmse": round(ci_harmonic_median_rmse_angular, 4) if ci_harmonic_median_rmse_angular is not None else None,
            "note": "Fitted to median node CI widths (typical case)",
        },
        "ci_width_betweenness_median_model": {
            "formula": "ci_width = C / sqrt(D + eff_n)",
            "C": round(ci_betweenness_median_c_angular, 3) if ci_betweenness_median_rmse_angular is not None else None,
            "D": round(ci_betweenness_median_d_angular, 2) if ci_betweenness_median_rmse_angular is not None else None,
            "rmse": round(ci_betweenness_median_rmse_angular, 4) if ci_betweenness_median_rmse_angular is not None else None,
            "note": "Fitted to median node CI widths (typical case)",
        },
    },
    # === Legacy format (for backward compatibility) - uses shortest betweenness ===
    "data_points": {
        "harmonic": len(harmonic_data),
        "betweenness": len(betweenness_data),
    },
    "harmonic_model": {
        "formula": "rho = 1 - A / (B + eff_n)",
        "A": round(harmonic_a, 2),
        "B": round(harmonic_b, 2),
        "rmse": round(harmonic_rmse, 4),
        "note": f"Fitted to {CONSERVATIVE_PERCENTILE}th percentile for conservative estimates",
    },
    "betweenness_model": {
        "formula": "rho = 1 - A / (B + eff_n)",
        "A": round(betweenness_a, 2),
        "B": round(betweenness_b, 2),
        "rmse": round(betweenness_rmse, 4),
        "note": f"Fitted to {CONSERVATIVE_PERCENTILE}th percentile for conservative estimates",
    },
    # Primary model uses betweenness (more demanding) for safety
    "rho_model": {
        "formula": "rho = 1 - A / (B + eff_n)",
        "A": round(rho_a, 2),
        "B": round(rho_b, 2),
        "rmse": round(rho_rmse, 4),
        "note": "Uses betweenness conservative model (higher variance metric)",
    },
    "std_model": {
        "formula": "std = C / sqrt(D + eff_n)",
        "C": round(std_c, 3),
        "D": round(std_d, 2),
        "rmse": round(std_rmse, 4) if std_rmse is not None else None,
    },
    "bias_model": {
        "formula": "scale = 1 - E / (F + eff_n)",
        "E": round(bias_e, 2),
        "F": round(bias_f, 2),
        "rmse": round(bias_rmse, 4) if bias_rmse is not None else None,
    },
    # CI-width models (legacy format - uses betweenness as conservative)
    "ci_width_model": {
        "formula": "ci_width = C / sqrt(D + eff_n)",
        "C": round(ci_betweenness_c, 3) if ci_betweenness_rmse is not None else None,
        "D": round(ci_betweenness_d, 2) if ci_betweenness_rmse is not None else None,
        "rmse": round(ci_betweenness_rmse, 4) if ci_betweenness_rmse is not None else None,
        "note": "Uses betweenness model (higher variance metric), fitted to median of node 90th percentile CI widths",
    },
}

constants_path = OUTPUT_DIR / "sampling_model_constants.json"
with open(constants_path, "w") as f:
    json.dump(model_constants, f, indent=2)
print(f"\nModel constants saved to: {constants_path}")

# %% Generate Model Comparison Table (AIC analysis)
print("\n" + "=" * 70)
print("GENERATING MODEL COMPARISON TABLE (AIC)")
print("=" * 70)


def fit_and_compare_models(eff_n: np.ndarray, rho: np.ndarray, label: str) -> dict:
    """Fit multiple model forms and compare using AIC."""
    # Filter valid data
    mask = (eff_n > 0) & np.isfinite(rho) & (rho > 0) & (rho <= 1)
    eff_n_fit = eff_n[mask]
    rho_fit = rho[mask]
    n = len(eff_n_fit)

    if n < 10:
        print(f"  {label}: Insufficient data ({n} points)")
        return {}

    results = {}

    # Model 1: Hyperbolic (our main model)
    def hyperbolic(x, a, b):
        return 1 - a / (b + x)

    # Model 2: Power Law
    def power_law(x, a, b):
        return 1 - a * np.power(x + 1, -b)

    # Model 3: Exponential
    def exponential(x, a, b):
        return 1 - a * np.exp(-b * x)

    # Model 4: Logistic
    def logistic(x, a, b):
        return 1 / (1 + a * np.exp(-b * x))

    models = [
        ("Hyperbolic", hyperbolic, [15, 20]),
        ("Power Law", power_law, [1, 0.5]),
        ("Exponential", exponential, [1, 0.01]),
        ("Logistic", logistic, [10, 0.01]),
    ]

    for name, func, p0 in models:
        try:
            params, _ = scipy_optimize.curve_fit(
                func, eff_n_fit, rho_fit, p0=p0, maxfev=10000, bounds=(0, np.inf)
            )
            pred = func(eff_n_fit, *params)
            residuals = rho_fit - pred
            ss_res = np.sum(residuals**2)
            rmse = np.sqrt(ss_res / n)

            # Calculate AIC: AIC = n * ln(RSS/n) + 2k
            # where k is number of parameters (2 for all models)
            k = 2
            aic = n * np.log(ss_res / n) + 2 * k

            results[name] = {"aic": aic, "rmse": rmse, "params": params}
        except Exception as e:
            print(f"  {label} - {name}: Failed to fit ({e})")
            results[name] = {"aic": np.inf, "rmse": np.inf, "params": None}

    return results


# Combine data for model comparison (use all data, not just percentiles)
# Shortest path data
shortest_eff_n = sampled["effective_n"].values
shortest_rho = sampled["spearman"].values

# Angular data
angular_eff_n = sampled_angular["effective_n"].values
angular_rho = sampled_angular["spearman"].values

print("\nFitting models for SHORTEST path distances...")
shortest_results = fit_and_compare_models(shortest_eff_n, shortest_rho, "Shortest")

print("\nFitting models for ANGULAR distances...")
angular_results = fit_and_compare_models(angular_eff_n, angular_rho, "Angular")


def format_model_comparison_table(shortest_res: dict, angular_res: dict) -> str:
    """Generate LaTeX table for model comparison."""
    # Determine which model is best overall and compute max delta for Power Law
    # (Power Law consistently performs worst, so we report its delta)
    power_law_deltas = []
    best_models = []
    for results in [shortest_res, angular_res]:
        if results:
            best_aic = min(r["aic"] for r in results.values() if r["aic"] != np.inf)
            best_name = min(results.keys(), key=lambda x: results[x]["aic"])
            best_models.append(best_name)
            if "Power Law" in results:
                power_law_deltas.append(results["Power Law"]["aic"] - best_aic)

    # Build caption based on actual results
    min_power_law_delta = min(power_law_deltas) if power_law_deltas else 0
    hyperbolic_is_best_angular = "Hyperbolic" in best_models[-1] if best_models else False

    lines = []
    lines.append(f"% Auto-generated table: Model Comparison (AIC)")
    lines.append(f"% Generated by sampling_reach.py on {datetime.now().isoformat(timespec='seconds')}")
    lines.append("% DO NOT EDIT MANUALLY - regenerate with: python sampling_reach.py")
    lines.append("")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Model comparison using Akaike Information Criterion (AIC). "
        r"All models have 2 parameters. The hyperbolic, exponential, and logistic forms "
        r"perform similarly, while the power law shows weaker support "
        f"($\\Delta$AIC $\\geq$ {min_power_law_delta:.0f}). "
        r"We select the hyperbolic form for its desirable asymptotic properties "
        r"($\rho \to 1$ as $n_{\text{eff}} \to \infty$).}"
    )
    lines.append(r"\label{tab:model_comparison}")
    lines.append(r"\begin{tabular}{llrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Distance type & Model & AIC & $\Delta$AIC & Weight & RMSE \\")
    lines.append(r"\midrule")

    for dist_type, results in [("Shortest", shortest_res), ("Angular", angular_res)]:
        if not results:
            continue

        # Find best AIC
        best_aic = min(r["aic"] for r in results.values() if r["aic"] != np.inf)

        # Calculate delta AIC and weights
        deltas = {}
        for name, r in results.items():
            deltas[name] = r["aic"] - best_aic if r["aic"] != np.inf else np.inf

        # Akaike weights
        exp_deltas = {k: np.exp(-0.5 * v) for k, v in deltas.items() if v != np.inf}
        total_weight = sum(exp_deltas.values())
        weights = {k: v / total_weight for k, v in exp_deltas.items()}

        # Sort by AIC
        sorted_models = sorted(results.keys(), key=lambda x: results[x]["aic"])

        for name in sorted_models:
            r = results[name]
            if r["aic"] == np.inf:
                continue
            delta = deltas[name]
            weight = weights.get(name, 0)
            bold = r"\textbf{" + name + "}" if delta == 0 else name
            lines.append(
                f"{dist_type} & {bold} & {r['aic']:.0f} & {delta:.0f} & {weight:.2f} & {r['rmse']:.4f} \\\\"
            )

        if dist_type == "Shortest":
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


if shortest_results and angular_results:
    model_comparison_tex = format_model_comparison_table(shortest_results, angular_results)
    model_comparison_path = TABLES_DIR / "model_comparison.tex"
    with open(model_comparison_path, "w") as f:
        f.write(model_comparison_tex)
    print(f"\nSaved: {model_comparison_path}")

    # Print summary
    print("\nModel Comparison Summary:")
    for dist_type, results in [("Shortest", shortest_results), ("Angular", angular_results)]:
        if results:
            best = min(results.items(), key=lambda x: x[1]["aic"])
            print(f"  {dist_type}: Best model = {best[0]} (AIC = {best[1]['aic']:.0f})")
else:
    print("Warning: Could not generate model comparison table due to missing data")

# %% Generate Model Parameters LaTeX Table
print("\n" + "=" * 70)
print("GENERATING MODEL PARAMETERS TABLE")
print("=" * 70)


def required_eff_n(a, b_param, target):
    """Calculate required effective_n for target rho."""
    return a / (1 - target) - b_param


# Shortest path models (baseline)
sh = model_constants["shortest"]["harmonic_model"]
sb = model_constants["shortest"]["betweenness_model"]
sh_95 = required_eff_n(sh["A"], sh["B"], 0.95)
sh_99 = required_eff_n(sh["A"], sh["B"], 0.99)
sb_95 = required_eff_n(sb["A"], sb["B"], 0.95)
sb_99 = required_eff_n(sb["A"], sb["B"], 0.99)

# Angular models (baseline)
ah = model_constants["angular"]["harmonic_model"]
ab = model_constants["angular"]["betweenness_model"]
ah_95 = required_eff_n(ah["A"], ah["B"], 0.95)
ah_99 = required_eff_n(ah["A"], ah["B"], 0.99)
ab_95 = required_eff_n(ab["A"], ab["B"], 0.95)
ab_99 = required_eff_n(ab["A"], ab["B"], 0.99)

# Power correction models
sh_pwr = model_constants["shortest"]["harmonic_model_power_corrected"]
sb_pwr = model_constants["shortest"]["betweenness_model_power_corrected"]
ah_pwr = model_constants["angular"]["harmonic_model_power_corrected"]
ab_pwr = model_constants["angular"]["betweenness_model_power_corrected"]

# Helper to format C and D values (power correction parameters)
def fmt_cd(c, d):
    if c is None or d is None:
        return "---", "---"
    return f"{c:.4f}", f"{d:.2f}"

# Helper to format comparison stats
def fmt_delta_aic(comp):
    delta = comp.get("delta_aic") if comp else None
    if delta is None:
        return "---"
    return f"{delta:.1f}"

def fmt_pvalue(comp):
    pval = comp.get("lr_pvalue") if comp else None
    if pval is None:
        return "---"
    if pval < 0.001:
        return "$<$0.001"
    return f"{pval:.3f}"

# Format power correction parameters
sh_c, sh_d = fmt_cd(sh_pwr.get("C"), sh_pwr.get("D"))
sb_c, sb_d = fmt_cd(sb_pwr.get("C"), sb_pwr.get("D"))
ah_c, ah_d = fmt_cd(ah_pwr.get("C"), ah_pwr.get("D"))
ab_c, ab_d = fmt_cd(ab_pwr.get("C"), ab_pwr.get("D"))

model_params_tex = rf"""% Auto-generated table: Model Parameters
% Generated by sampling_reach.py on {datetime.now().isoformat(timespec="seconds")}
% DO NOT EDIT MANUALLY - regenerate with: python sampling_reach.py

\begin{{table}}[htbp]
\centering
\caption{{Fitted model parameters for predicting Spearman $\rho$ from effective sample size. Baseline model: $\rho = 1 - A/(B + \effn)$. Power correction model: $\rho = 1 - A/(B + \effn) - C \times (1-p)^D$ where $p$ is sampling probability. The correction term penalises low sampling probabilities where variance is higher. Models fitted to 10th percentile for conservative estimates. $\Delta$AIC compares power correction vs baseline (negative favours correction). $p$-value from likelihood ratio test.}}
\label{{tab:model_parameters}}
\resizebox{{\textwidth}}{{!}}{{
\begin{{tabular}}{{llrrrrrrrr}}
\toprule
Distance & Metric & $A$ & $B$ & $C$ & $D$ & RMSE & $\Delta$AIC & $\effn$ for $\rho\!=\!0.95$ & $\effn$ for $\rho\!=\!0.99$ \\
\midrule
Shortest & Harmonic & {sh_pwr["A"] or sh["A"]:.2f} & {sh_pwr["B"] or sh["B"]:.2f} & {sh_c} & {sh_d} & {sh_pwr["rmse"] or sh["rmse"]:.4f} & {fmt_delta_aic(sh_pwr.get("comparison"))} & {sh_95:.0f} & {sh_99:.0f} \\
Shortest & Betweenness & {sb_pwr["A"] or sb["A"]:.2f} & {sb_pwr["B"] or sb["B"]:.2f} & {sb_c} & {sb_d} & {sb_pwr["rmse"] or sb["rmse"]:.4f} & {fmt_delta_aic(sb_pwr.get("comparison"))} & {sb_95:.0f} & {sb_99:.0f} \\
\midrule
Angular & Harmonic & {ah_pwr["A"] or ah["A"]:.2f} & {ah_pwr["B"] or ah["B"]:.2f} & {ah_c} & {ah_d} & {ah_pwr["rmse"] or ah["rmse"]:.4f} & {fmt_delta_aic(ah_pwr.get("comparison"))} & {ah_95:.0f} & {ah_99:.0f} \\
Angular & Betweenness & {ab_pwr["A"] or ab["A"]:.2f} & {ab_pwr["B"] or ab["B"]:.2f} & {ab_c} & {ab_d} & {ab_pwr["rmse"] or ab["rmse"]:.4f} & {fmt_delta_aic(ab_pwr.get("comparison"))} & {ab_95:.0f} & {ab_99:.0f} \\
\bottomrule
\end{{tabular}}
}}
\end{{table}}
"""

model_params_path = TABLES_DIR / "model_parameters.tex"
with open(model_params_path, "w") as f:
    f.write(model_params_tex)
print(f"Saved: {model_params_path}")

# %% Generate Model Macros LaTeX File
print("\n" + "=" * 70)
print("GENERATING MODEL MACROS FILE")
print("=" * 70)

# Compute eff_n thresholds for all target rho values
rho_targets = [0.90, 0.95, 0.97, 0.99]
rho_names = {0.90: "Ninety", 0.95: "NinetyFive", 0.97: "NinetySeven", 0.99: "NinetyNine"}

# Calculate all thresholds
thresholds = {}
for dist_type, dist_key in [("Shortest", "shortest"), ("Angular", "angular")]:
    for metric, metric_key in [("Harmonic", "harmonic_model"), ("Betweenness", "betweenness_model")]:
        model = model_constants[dist_key][metric_key]
        for rho in rho_targets:
            key = f"{dist_type}{metric}{rho_names[rho]}"
            thresholds[key] = required_eff_n(model["A"], model["B"], rho)

# Calculate metric comparison ratios (harmonic/betweenness and betweenness/harmonic)
ratio_hb_shortest = thresholds["ShortestHarmonicNinetyFive"] / thresholds["ShortestBetweennessNinetyFive"]
ratio_bh_shortest = thresholds["ShortestBetweennessNinetyFive"] / thresholds["ShortestHarmonicNinetyFive"]
ratio_hb_angular = thresholds["AngularHarmonicNinetyFive"] / thresholds["AngularBetweennessNinetyFive"]
ratio_bh_angular = thresholds["AngularBetweennessNinetyFive"] / thresholds["AngularHarmonicNinetyFive"]

# Build macros content
macros_lines = [
    f"% Auto-generated macros: Model-Derived Values",
    f"% Generated by sampling_reach.py on {datetime.now().isoformat(timespec='seconds')}",
    f"% DO NOT EDIT MANUALLY - regenerate with: python sampling_reach.py",
    "",
]

# Add eff_n thresholds grouped by rho target
for rho in rho_targets:
    rho_name = rho_names[rho]
    rho_str = str(rho).replace("0.", "")
    macros_lines.append(f"% Effective sample size thresholds for rho={rho}")
    for dist_type in ["Shortest", "Angular"]:
        for metric in ["Harmonic", "Betweenness"]:
            key = f"{dist_type}{metric}{rho_name}"
            macro_name = f"\\newcommand{{\\effn{dist_type}{metric}{rho_name}}}{{{thresholds[key]:.0f}}}"
            macros_lines.append(macro_name)
    macros_lines.append("")

# Add model parameters (A and B) for baseline models
macros_lines.append("% Model parameters A and B (baseline)")
for dist_type, dist_key in [("Shortest", "shortest"), ("Angular", "angular")]:
    for metric, metric_key in [("Harmonic", "harmonic_model"), ("Betweenness", "betweenness_model")]:
        model = model_constants[dist_key][metric_key]
        macros_lines.append(f"\\newcommand{{\\paramA{dist_type}{metric}}}{{{model['A']:.2f}}}")
        macros_lines.append(f"\\newcommand{{\\paramB{dist_type}{metric}}}{{{model['B']:.2f}}}")
macros_lines.append("")

# Add power correction model parameters (A, B, C, D)
macros_lines.append("% Power correction model parameters A, B, C, D")
for dist_type, dist_key in [("Shortest", "shortest"), ("Angular", "angular")]:
    for metric, metric_key in [("Harmonic", "harmonic_model_power_corrected"), ("Betweenness", "betweenness_model_power_corrected")]:
        model = model_constants[dist_key][metric_key]
        a_val = model.get("A", 0) or 0
        b_val = model.get("B", 0) or 0
        c_val = model.get("C", 0) or 0
        d_val = model.get("D", 0) or 0
        metric_name = metric.replace("_model_power_corrected", "").title()
        macros_lines.append(f"\\newcommand{{\\paramAPwr{dist_type}{metric_name}}}{{{a_val:.2f}}}")
        macros_lines.append(f"\\newcommand{{\\paramBPwr{dist_type}{metric_name}}}{{{b_val:.2f}}}")
        macros_lines.append(f"\\newcommand{{\\paramCPwr{dist_type}{metric_name}}}{{{c_val:.4f}}}")
        macros_lines.append(f"\\newcommand{{\\paramDPwr{dist_type}{metric_name}}}{{{d_val:.2f}}}")
macros_lines.append("")

# Add power correction model comparison statistics
macros_lines.append("% Power correction model comparison statistics")
for dist_type, dist_key in [("Shortest", "shortest"), ("Angular", "angular")]:
    for metric, metric_key in [("Harmonic", "harmonic_model_power_corrected"), ("Betweenness", "betweenness_model_power_corrected")]:
        model = model_constants[dist_key][metric_key]
        comp = model.get("comparison", {})
        delta_aic = comp.get("delta_aic", 0) or 0
        lr_pval = comp.get("lr_pvalue", 1) or 1
        rmse_imp = comp.get("rmse_improvement", 0) or 0
        res_p_corr_base = comp.get("residual_p_corr_baseline", 0) or 0
        res_p_corr_corr = comp.get("residual_p_corr_corrected", 0) or 0
        metric_short = metric.replace("_model_power_corrected", "").title()
        macros_lines.append(f"\\newcommand{{\\deltaAIC{dist_type}{metric_short}}}{{{delta_aic:.1f}}}")
        macros_lines.append(f"\\newcommand{{\\lrPval{dist_type}{metric_short}}}{{{lr_pval:.4f}}}")
        macros_lines.append(f"\\newcommand{{\\rmseImp{dist_type}{metric_short}}}{{{rmse_imp:.1f}}}")
        macros_lines.append(f"\\newcommand{{\\resPCorrBase{dist_type}{metric_short}}}{{{res_p_corr_base:.3f}}}")
        macros_lines.append(f"\\newcommand{{\\resPCorrPwr{dist_type}{metric_short}}}{{{res_p_corr_corr:.3f}}}")
macros_lines.append("")

# Add metric comparison ratios
macros_lines.append("% Metric comparison ratios (at rho=0.95)")
macros_lines.append(f"\\newcommand{{\\ratioHBShortest}}{{{ratio_hb_shortest:.2f}}}  % harmonic/betweenness for shortest")
macros_lines.append(f"\\newcommand{{\\ratioBHShortest}}{{{ratio_bh_shortest:.2f}}}  % betweenness/harmonic for shortest")
macros_lines.append(f"\\newcommand{{\\ratioHBAngular}}{{{ratio_hb_angular:.2f}}}  % harmonic/betweenness for angular")
macros_lines.append(f"\\newcommand{{\\ratioBHAngular}}{{{ratio_bh_angular:.2f}}}  % betweenness/harmonic for angular")

model_macros_tex = "\n".join(macros_lines)

model_macros_path = TABLES_DIR / "model_macros.tex"
with open(model_macros_path, "w") as f:
    f.write(model_macros_tex)
print(f"Saved: {model_macros_path}")

# %% Figure 2a: Required Sampling Probability (SHORTEST distances)
print("\n" + "=" * 70)
print("FIGURE 2a: Required Sampling Probability by Reachability (SHORTEST)")
print("=" * 70)

# Plot A: Given a target Spearman, what p do you need at different reachabilities?
# Using the fitted model: ρ = 1 - A / (B + eff_n)
# Solving for eff_n: eff_n = A / (1 - ρ) - B
# Since eff_n = reach × p: p = eff_n / reach = (A / (1 - ρ) - B) / reach

# Create side-by-side plots for harmonic and betweenness models
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

spearman_targets = [0.90, 0.95, 0.97, 0.99]
target_colors = {
    "0.9": "steelblue",
    "0.95": "darkorange",
    "0.97": "mediumseagreen",
    "0.99": "teal",
}

# For each model, find the max required_eff_n (at ρ=0.99) to set appropriate x-axis
max_eff_n_harmonic = harmonic_a / (1 - 0.99) - harmonic_b
max_eff_n_betweenness = betweenness_a / (1 - 0.99) - betweenness_b

# Set x-axis to show full curves (where p drops below 1.0)
# x_max should be at least max_eff_n so curves can reach p=1
x_max = max(max_eff_n_harmonic, max_eff_n_betweenness) * 1.5
reach_range = np.linspace(50, x_max, 200)

models = [
    ("Harmonic (Closeness)", harmonic_a, harmonic_b, axes[0]),
    ("Betweenness", betweenness_a, betweenness_b, axes[1]),
]

for title, a, b, ax in models:
    for target_rho in spearman_targets:
        # Calculate required effective_n from model
        required_eff_n = a / (1 - target_rho) - b
        # Calculate required p for each reachability
        required_p = required_eff_n / reach_range
        # Clip to valid probability range [0, 1]
        required_p = np.clip(required_p, 0, 1)

        ax.plot(
            reach_range,
            required_p,
            "-",
            linewidth=2.5,
            label=f"ρ ≥ {target_rho} (eff_n ≥ {required_eff_n:.0f})",
            color=target_colors[str(target_rho)],
        )

    ax.set_xlabel("Mean Reachability", fontsize=12)
    ax.set_ylabel("Required Sampling Probability (p)", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, x_max)
    ax.legend(loc="lower left", title="Target Spearman ρ", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"{title}\nModel: ρ = 1 - {a:.1f}/({b:.1f} + reach×p)", fontsize=11)

fig.suptitle(
    "SHORTEST Path: Required Sampling Probability\n(Conservative 10th percentile models)",
    fontsize=13,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "required_probability_shortest.pdf", dpi=300, bbox_inches="tight")
print(f"Saved: {FIGURES_DIR / 'required_probability_shortest.pdf'}")

# %% Figure 2d: Required Sampling Probability (ANGULAR distances)
print("\n" + "=" * 70)
print("FIGURE 2d: Required Sampling Probability by Reachability (ANGULAR)")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# For angular models
max_eff_n_harmonic_ang = harmonic_a_angular / (1 - 0.99) - harmonic_b_angular
max_eff_n_betweenness_ang = betweenness_a_angular / (1 - 0.99) - betweenness_b_angular

x_max_ang = max(max_eff_n_harmonic_ang, max_eff_n_betweenness_ang) * 1.5
reach_range_ang = np.linspace(50, x_max_ang, 200)

models_ang = [
    ("Harmonic (Closeness)", harmonic_a_angular, harmonic_b_angular, axes[0]),
    ("Betweenness", betweenness_a_angular, betweenness_b_angular, axes[1]),
]

for title, a, b, ax in models_ang:
    for target_rho in spearman_targets:
        # Calculate required effective_n from model
        required_eff_n = a / (1 - target_rho) - b
        # Calculate required p for each reachability
        required_p = required_eff_n / reach_range_ang
        # Clip to valid probability range [0, 1]
        required_p = np.clip(required_p, 0, 1)

        ax.plot(
            reach_range_ang,
            required_p,
            "-",
            linewidth=2.5,
            label=f"ρ ≥ {target_rho} (eff_n ≥ {required_eff_n:.0f})",
            color=target_colors[str(target_rho)],
        )

    ax.set_xlabel("Mean Reachability", fontsize=12)
    ax.set_ylabel("Required Sampling Probability (p)", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, x_max_ang)
    ax.legend(loc="lower left", title="Target Spearman ρ", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"{title}\nModel: ρ = 1 - {a:.1f}/({b:.1f} + reach×p)", fontsize=11)

fig.suptitle(
    "ANGULAR Path: Required Sampling Probability\n(Conservative 10th percentile models)",
    fontsize=13,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "required_probability_angular.pdf", dpi=300, bbox_inches="tight")
print(f"Saved: {FIGURES_DIR / 'required_probability_angular.pdf'}")

# %% Generate README
print("\n" + "=" * 70)
print("GENERATING README")
print("=" * 70)

# Compute summary statistics for README
harmonic_df = sampled[sampled["metric"] == "harmonic"]
between_df = sampled[sampled["metric"] == "betweenness"]

# Build correctness results string for README
correctness_table = "\n".join(
    [f"| {metric} | {data['max_diff']:.2e} | {data['status']} |" for metric, data in correctness_results.items()]
)

# Compute key thresholds from the fitted model for documentation
eff_n_for_95 = rho_a / (1 - 0.95) - rho_b if rho_a and rho_b else 261
eff_n_for_90 = rho_a / (1 - 0.90) - rho_b if rho_a and rho_b else 120

# Precompute model predictions for README table
pred_rows = []
for n in [10, 25, 50, 100, 200, 400]:
    rho_pred = rho_model(n, betweenness_a, betweenness_b)
    std_pred = std_model(n, std_c, std_d)
    bias_pred = (1 - bias_model(n, bias_e, bias_f)) * 100
    pred_rows.append(f"| {n} | {rho_pred:.3f} | {std_pred:.3f} | {bias_pred:.1f}% |")
model_pred_table = "\n".join(pred_rows)

# Precompute required eff_n for README table
req_rows = []
for target, div in [(0.90, 0.1), (0.95, 0.05), (0.97, 0.03), (0.99, 0.01)]:
    h_req = harmonic_a / div - harmonic_b
    b_req = betweenness_a / div - betweenness_b
    req_rows.append(f"| {target} | {h_req:.0f} | {b_req:.0f} |")
req_eff_n_table = "\n".join(req_rows)

readme_content = f"""# Sampling Analysis: When Can You Trust Sampled Centrality?

Generated: {datetime.now().isoformat(timespec="seconds")}

This document summarises empirical observations on how sampling affects centrality
accuracy. Models fitted here are used by cityseer to provide runtime accuracy estimates.

---

## Summary

Based on experiments with three synthetic network topologies:

| Observed threshold | effective_n | Note |
|-------------------|-------------|------|
| Mean ρ ≈ 0.95 | ~{eff_n_for_95:.0f} | High variance at lower values |
| Mean ρ ≈ 0.90 | ~{eff_n_for_90:.0f} | Individual runs vary |

**Formula**: `effective_n = reachability × sampling_probability`

---

## Chapter 1: Correctness Verification

Before trusting sampled results, we verify that cityseer's centrality implementation
matches NetworkX (the reference implementation).

| Metric | Max Difference | Status |
|--------|----------------|--------|
{correctness_table}

All metrics pass verification, confirming cityseer computes correct centrality values.

---

## Chapter 2: Test Network Topologies

Three synthetic network topologies are used for testing:

![Test Network Topologies](output/topologies.png)

- **Trellis**: Dense grid-like networks (urban cores, high connectivity)
- **Tree**: Branching dendritic networks (suburban areas, hierarchical)
- **Linear**: Linear corridor networks (transit corridors, low connectivity)

These cover the range of real-world network structures.

---

## Chapter 3: Understanding Effective Sample Size

When using sampling to speed up centrality computation, accuracy depends on the
**effective sample size**:

```
effective_n = reachability × sampling_probability
```

Where:
- **reachability**: average number of nodes reachable within the distance threshold
- **sampling_probability (p)**: fraction of nodes used as sources (0 to 1)

### Concept

Each node's centrality value is computed from contributions by sampled source nodes.
The effective_n approximates how many sampled sources contribute to each node's estimate.

### Lookup Table: Effective Sample Size

| Reachability | p=10% | p=20% | p=30% | p=40% | p=50% |
|--------------|-------|-------|-------|-------|-------|
| 100 | 10 | 20 | 30 | 40 | 50 |
| 200 | 20 | 40 | 60 | 80 | 100 |
| 400 | 40 | 80 | 120 | 160 | 200 |
| 600 | 60 | 120 | 180 | 240 | 300 |
| 800 | 80 | 160 | 240 | 320 | 400 |
| 1000 | 100 | 200 | 300 | 400 | 500 |

---

## Chapter 4: Fitted Models

Empirical models fitted to the experimental data using the {CONSERVATIVE_PERCENTILE}th percentile
for conservative estimates (accounting for variance across network topologies).

### Spearman ρ Models (Ranking Accuracy)

Separate models are fitted for closeness (harmonic) and betweenness centrality,
as betweenness shows higher variance at the same effective_n.

**Harmonic (Closeness)**:
```
ρ = 1 - {harmonic_a:.2f} / ({harmonic_b:.2f} + effective_n)
```
- RMSE: {harmonic_rmse:.4f}

**Betweenness**:
```
ρ = 1 - {betweenness_a:.2f} / ({betweenness_b:.2f} + effective_n)
```
- RMSE: {betweenness_rmse:.4f}

When computing both metrics together, the betweenness (more conservative) model
is used to ensure both metrics meet accuracy targets.

### Standard Deviation Model (Uncertainty)

```
std = {std_c:.3f} / sqrt({std_d:.2f} + effective_n)
```

- Decreases as effective_n increases
- RMSE of fit: {f"{std_rmse:.4f}" if std_rmse is not None else "N/A"}

### Magnitude Bias Model

Observed tendency for magnitudes to be underestimated at low effective_n:

```
scale = 1 - {bias_e:.2f} / ({bias_f:.2f} + effective_n)
bias = 1 - scale
```

- RMSE of fit: {f"{bias_rmse:.4f}" if bias_rmse is not None else "N/A"}
- Note: Higher RMSE than ranking model; predictions less reliable

### Model Predictions (Betweenness / Conservative)

| effective_n | Expected ρ | Std Dev | Bias |
|-------------|------------|---------|------|
{model_pred_table}

### Required effective_n for Target Accuracy

| Target ρ | Harmonic | Betweenness |
|----------|----------|-------------|
{req_eff_n_table}

---

## Chapter 5: Results

### Figure 1: Accuracy vs Effective Sample Size

![Sampling Accuracy](output/sampling_accuracy.png)

Scatter plots of observed ranking (top) and magnitude (bottom) accuracy across
all experimental configurations. Points are coloured by network topology.

### Figure 2a: Required Sampling Probability

![Sampling Probability](output/sampling_probability.png)

Theoretical curves showing the sampling probability required to achieve each
target Spearman ρ, derived from the fitted model. The legend shows the
effective_n threshold needed for each accuracy level.

### Figure 2b: Expected Ranking Accuracy

![Sampling Accuracy vs Effective N](output/sampling_accuracy_vs_eff_n.png)

Mean observed Spearman ρ binned by effective sample size. Error bars show
±1 standard deviation within each bin.

---

## Chapter 6: Considerations

### Factors Affecting Accuracy

The models above were fitted on three synthetic network topologies. Real-world networks
may behave differently. The relationship between effective_n and accuracy appears
consistent across the tested topologies, but extrapolation to other network types
should be done with caution.

### Limitations

- Models are fitted on synthetic networks; real networks may vary
- Low effective_n (< 25) shows high variance in observed accuracy
- The bias model has higher RMSE than the ranking model
- Individual runs at the same effective_n can vary substantially

---

## Experimental Details

### Parameters

- Distances: {DISTANCES}
- Sampling probabilities: {[f"{p:.0%}" for p in PROBS]}
- Runs per configuration: {N_RUNS} (for variance estimation)
- Network topologies: {TEMPLATE_NAMES}

### Metrics

- **Harmonic closeness**: Sum of inverse distances to all reachable nodes
- **Betweenness**: Count of shortest paths passing through each node

### Model Constants

These are exported to `sampling_model_constants.json` and synced to
`pysrc/cityseer/config.py` for runtime accuracy estimation:

```json
{{
  "harmonic_model": {{"A": {harmonic_a:.2f}, "B": {harmonic_b:.2f}}},
  "betweenness_model": {{"A": {betweenness_a:.2f}, "B": {betweenness_b:.2f}}},
  "rho_model": {{"A": {rho_a:.2f}, "B": {rho_b:.2f}}},
  "std_model": {{"C": {std_c:.3f}, "D": {std_d:.2f}}},
  "bias_model": {{"E": {bias_e:.2f}, "F": {bias_f:.2f}}}
}}
```

Note: `rho_model` uses the betweenness (more conservative) model for backward compatibility
when computing both metrics together.

---

*Generated by `sampling_reach.py` — Run `poe sync_sampling_constants` to update config.py*
"""

results_md_path = SCRIPT_DIR / "sampling_reach_results.md"
with open(results_md_path, "w") as f:
    f.write(readme_content)

print(f"Results saved to: {results_md_path}")

# %% Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
eff_n_for_95_h = harmonic_a / 0.05 - harmonic_b
eff_n_for_95_b = betweenness_a / 0.05 - betweenness_b
print(f"""
KEY TAKEAWAY:

  effective_n = reachability × sampling_probability

  Higher effective_n = better accuracy for both ranking and magnitude.
  This relationship holds across all tested network topologies.

FITTED MODELS (Conservative {CONSERVATIVE_PERCENTILE}th percentile):

  Harmonic (closeness):
    Expected ρ = 1 - {harmonic_a:.2f} / ({harmonic_b:.2f} + effective_n)
    For ρ ≥ 0.95: eff_n ≥ {eff_n_for_95_h:.0f}

  Betweenness (higher variance):
    Expected ρ = 1 - {betweenness_a:.2f} / ({betweenness_b:.2f} + effective_n)
    For ρ ≥ 0.95: eff_n ≥ {eff_n_for_95_b:.0f}

  When computing both metrics, use the betweenness (more conservative) model.

PRACTICAL GUIDANCE:

  1. Estimate your reachability from network density and distance threshold
  2. Choose p to achieve your target effective_n:
     - Closeness only: p ≥ {eff_n_for_95_h:.0f} / reachability
     - Betweenness or both: p ≥ {eff_n_for_95_b:.0f} / reachability
  3. Check cityseer's runtime logs for actual accuracy estimates

OUTPUT FILES:

  - sampling_reach_results.md: Full documentation with figures
  - output/sampling_accuracy.png: Main results figure
  - output/sampling_probability.png: Required sampling probability curves
  - output/sampling_model_constants.json: Model parameters for config.py

NEXT STEPS:

  Run `poe sync_sampling_constants` to update config.py with new model parameters.
""")

# %%
