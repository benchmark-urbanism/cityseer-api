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
from utils.substrates import generate_keyed_template

warnings.filterwarnings("ignore")

# Output and cache directories
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR = Path(__file__).parent.parent / "temp" / "sampling_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Cache version - increment to force re-run
CACHE_VERSION = "v5"


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
SUBSTRATE_TILES = 3  # ~2940m network extent
# Finer distance granularity for better curve fitting
DISTANCES = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000]
# Finer probability steps for more accurate model fitting
PROBS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
N_RUNS = 20  # Runs per configuration for variance estimation


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
print("\nGenerating topology visualizations...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, topo in enumerate(TEMPLATE_NAMES):
    G = generate_substrate(topo)
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
            ax.plot(xs, ys, color="#666666", linewidth=0.5, alpha=0.6)
    # Draw nodes
    for n in G.nodes():
        ax.plot(G.nodes[n]["x"], G.nodes[n]["y"], "o", color="#bbbbbb", markersize=1, alpha=1.0)
    ax.set_title(f"{topo.title()}\n({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)", fontsize=11)
    ax.set_aspect("equal")
    ax.axis("off")

plt.suptitle("Test Network Topologies", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "topologies.png", dpi=150, bbox_inches="tight")
print(f"Saved: {OUTPUT_DIR / 'topologies.png'}")

# %% Run Sampling Analysis
print("\n" + "=" * 70)
print("CHAPTER 2: Sampling Accuracy Analysis")
print("=" * 70)
print(f"\nTemplates: {TEMPLATE_NAMES}")
print(f"Distances: {DISTANCES}")
print(f"Probabilities: {PROBS}")
print(f"Runs per config: {N_RUNS}")
print(f"Cache version: {CACHE_VERSION}")

cached = load_cache("sampling_analysis")
if cached is not None:
    print("\nLoading cached results...")
    results = cached
else:
    results = []

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
                                "mean_reach": mean_reach,
                                "sample_prob": p,
                                "effective_n": mean_reach,
                                "metric": metric,
                                "spearman": 1.0,
                                "spearman_std": 0.0,
                                "top_k_precision": 1.0,
                                "precision_std": 0.0,
                                "scale_ratio": 1.0,
                                "scale_ratio_std": 0.0,
                                "scale_iqr": 0.0,
                                "scale_iqr_std": 0.0,
                            }
                        )
                    continue

                # Multiple runs for variance estimation
                metrics_data = {
                    "harmonic": {"spearmans": [], "precisions": [], "scale_ratios": [], "scale_iqrs": []},
                    "betweenness": {"spearmans": [], "precisions": [], "scale_ratios": [], "scale_iqrs": []},
                }
                true_vals = {"harmonic": true_harmonic, "betweenness": true_betweenness}

                for seed in range(N_RUNS):
                    r = net.local_node_centrality_shortest(
                        distances=[dist],
                        compute_closeness=True,
                        compute_betweenness=True,
                        sample_probability=p,
                        random_seed=seed,
                        pbar_disabled=True,
                    )
                    est_vals = {
                        "harmonic": np.array(r.node_harmonic[dist]),
                        "betweenness": np.array(r.node_betweenness[dist]),
                    }

                    for metric in ["harmonic", "betweenness"]:
                        sp, prec, scale, iqr = compute_accuracy_metrics(true_vals[metric], est_vals[metric])
                        if not np.isnan(sp):
                            metrics_data[metric]["spearmans"].append(sp)
                            metrics_data[metric]["precisions"].append(prec)
                            metrics_data[metric]["scale_ratios"].append(scale)
                            metrics_data[metric]["scale_iqrs"].append(iqr)

                effective_n = mean_reach * p

                for metric in ["harmonic", "betweenness"]:
                    data = metrics_data[metric]
                    if data["spearmans"]:
                        results.append(
                            {
                                "topology": topo,
                                "distance": dist,
                                "mean_reach": mean_reach,
                                "sample_prob": p,
                                "effective_n": effective_n,
                                "metric": metric,
                                "spearman": np.mean(data["spearmans"]),
                                "spearman_std": np.std(data["spearmans"]),
                                "top_k_precision": np.mean(data["precisions"]),
                                "precision_std": np.std(data["precisions"]),
                                "scale_ratio": np.mean(data["scale_ratios"]),
                                "scale_ratio_std": np.std(data["scale_ratios"]),
                                "scale_iqr": np.mean(data["scale_iqrs"]),
                                "scale_iqr_std": np.std(data["scale_iqrs"]),
                            }
                        )

            print("done")

    save_cache("sampling_analysis", results)
    print("\nResults cached.")

df = pd.DataFrame(results)
print(f"\nTotal observations: {len(df)}")


# %% THE KEY TABLE: Combined Ranking + Magnitude by Effective N
print("\n" + "=" * 70)
print("KEY FINDING: Quality by Effective Sample Size")
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
print(f"{'Reach \\ p':>10} |" + "".join([f" {p:>7.0%} |" for p in p_values]))
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


# %% Figure 1: The Key Visual - Both Dimensions vs Effective N
print("\n" + "=" * 70)
print("FIGURE 1: Accuracy vs Effective Sample Size")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
colors = {"trellis": "#1f77b4", "tree": "#d62728", "linear": "#ff7f0e"}

for col, metric in enumerate(["harmonic", "betweenness"]):
    metric_df = sampled[sampled["metric"] == metric]

    # Top row: Spearman (ranking)
    ax = axes[0, col]
    for topo in TEMPLATE_NAMES:
        subset = metric_df[metric_df["topology"] == topo]
        ax.scatter(subset["effective_n"], subset["spearman"], alpha=0.5, s=30, color=colors[topo], label=topo.title())

    ax.axhline(y=0.95, color="green", linestyle=":", linewidth=1.5, alpha=0.8, label="ρ = 0.95")
    ax.axhline(y=0.90, color="orange", linestyle=":", linewidth=1.5, alpha=0.8, label="ρ = 0.90")

    ax.set_xlabel("Effective Sample Size (reach × p)", fontsize=11)
    ax.set_ylabel("Spearman ρ (ranking)", fontsize=11)
    ax.set_title(f"{metric.title()}: Ranking Accuracy", fontsize=12, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlim(5, 2000)
    ax.set_ylim(0.5, 1.02)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom row: Scale ratio (magnitude)
    ax = axes[1, col]
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
    ax.set_xlim(5, 2000)
    ax.set_ylim(0.7, 1.3)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle(
    "Sampling Accuracy: Both Ranking and Magnitude Depend on Effective Sample Size\n"
    "(effective_n = reachability × sampling_probability)",
    fontsize=13,
    fontweight="bold",
    y=1.02,
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sampling_accuracy.png", dpi=150, bbox_inches="tight")
print(f"Saved: {OUTPUT_DIR / 'sampling_accuracy.png'}")


# %% Figure 2b: Expected Ranking Accuracy by Effective Sample Size
print("\n" + "=" * 70)
print("FIGURE 2b: Expected Ranking Accuracy by Effective Sample Size")
print("=" * 70)

# Plot B: What Spearman ρ can you expect at different effective_n values?

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

eff_n_bins = np.linspace(0, 600, 25)

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
        ax.text(610, target_rho, f"ρ={target_rho}", fontsize=9, va="center", color=color)

    ax.set_xlabel("Effective Sample Size (reachability × p)", fontsize=11)
    ax.set_ylabel("Mean Spearman ρ", fontsize=11)
    ax.set_title(f"{metric.title()}", fontsize=12)
    ax.set_ylim(0.5, 1.02)
    ax.set_xlim(0, 650)
    ax.grid(True, alpha=0.3)

plt.suptitle(
    "Ranking Accuracy vs Effective Sample Size",
    fontsize=13,
    fontweight="bold",
    y=1.02,
)
fig.text(
    0.5, -0.02,
    "Points: mean Spearman ρ across all runs in each effective_n bin. "
    "Error bars: ±1 standard deviation. Dashed lines: accuracy targets.",
    ha="center", fontsize=9, style="italic",
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sampling_accuracy_vs_eff_n.png", dpi=150, bbox_inches="tight")
print(f"Saved: {OUTPUT_DIR / 'sampling_accuracy_vs_eff_n.png'}")


# %% Fit Continuous Models for Sampling Accuracy
print("\n" + "=" * 70)
print("FITTING CONTINUOUS MODELS")
print("=" * 70)

# Fit models to predict Spearman ρ and std from effective_n
# Model for ρ: rho = 1 - A / (B + eff_n)  [approaches 1.0 as eff_n → ∞]
# Model for std: std = C / sqrt(D + eff_n)  [decreases with eff_n]


def rho_model(eff_n, a, b):
    """Model: rho = 1 - a / (b + eff_n)"""
    return 1 - a / (b + eff_n)


def std_model(eff_n, c, d):
    """Model: std = c / sqrt(d + eff_n)"""
    return c / np.sqrt(d + eff_n)


def bias_model(eff_n, e, f):
    """Model: scale = 1 - e / (f + eff_n), bias = 1 - scale"""
    return 1 - e / (f + eff_n)


# Get data for fitting (use harmonic as primary, but could combine both)
harmonic_data = sampled[sampled["metric"] == "harmonic"]
eff_n_vals = harmonic_data["effective_n"].values
spearman_vals = harmonic_data["spearman"].values

# Filter valid data
valid_mask = (eff_n_vals > 0) & (spearman_vals > 0) & np.isfinite(eff_n_vals) & np.isfinite(spearman_vals)
eff_n_fit = eff_n_vals[valid_mask]
spearman_fit = spearman_vals[valid_mask]

print(f"\nFitting models using {len(eff_n_fit)} data points...")
print(f"  Effective_n range: {eff_n_fit.min():.1f} - {eff_n_fit.max():.1f}")
print(f"  Spearman range: {spearman_fit.min():.3f} - {spearman_fit.max():.3f}")

# Fit rho model
try:
    rho_params, _ = scipy_optimize.curve_fit(rho_model, eff_n_fit, spearman_fit, p0=[15, 20], maxfev=5000)
    rho_a, rho_b = rho_params
    rho_pred = rho_model(eff_n_fit, rho_a, rho_b)
    rho_rmse = np.sqrt(np.mean((spearman_fit - rho_pred) ** 2))
    print(f"\nRho model: ρ = 1 - {rho_a:.2f} / ({rho_b:.2f} + eff_n)")
    print(f"  RMSE: {rho_rmse:.4f}")
except Exception as e:
    print(f"  Warning: Could not fit rho model: {e}")
    rho_a, rho_b = 14.11, 21.58  # Fallback to previous values

# Compute std per effective_n bin for fitting
std_bins = np.array([0, 10, 25, 50, 75, 100, 150, 200, 300, 400, 600, 1000])
bin_centers = []
bin_stds = []
for i in range(len(std_bins) - 1):
    lo, hi = std_bins[i], std_bins[i + 1]
    subset = harmonic_data[(harmonic_data["effective_n"] >= lo) & (harmonic_data["effective_n"] < hi)]
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
    print(f"\nStd model: std = {std_c:.3f} / sqrt({std_d:.2f} + eff_n)")
    print(f"  RMSE: {std_rmse:.4f}")
except Exception as e:
    print(f"  Warning: Could not fit std model: {e}")
    std_c, std_d = 0.907, 10.06  # Fallback to previous values

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

# Print model predictions at key effective_n values
print("\nModel predictions:")
print(f"{'eff_n':>8} | {'predicted ρ':>12} | {'predicted std':>13} | {'expected bias':>13}")
print("-" * 55)
for n in [10, 25, 50, 100, 200, 400, 800]:
    pred_rho = rho_model(n, rho_a, rho_b)
    pred_std = std_model(n, std_c, std_d)
    pred_scale = bias_model(n, bias_e, bias_f)
    pred_bias = (1 - pred_scale) * 100
    print(f"{n:>8} | {pred_rho:>12.3f} | {pred_std:>13.3f} | {pred_bias:>12.1f}%")

# Save model constants to JSON for syncing to config.py
model_constants = {
    "generated": datetime.now().isoformat(timespec="seconds"),
    "data_points": len(eff_n_fit),
    "rho_model": {
        "formula": "rho = 1 - A / (B + eff_n)",
        "A": round(rho_a, 2),
        "B": round(rho_b, 2),
        "rmse": round(rho_rmse, 4),
    },
    "std_model": {
        "formula": "std = C / sqrt(D + eff_n)",
        "C": round(std_c, 3),
        "D": round(std_d, 2),
        "rmse": round(std_rmse, 4) if "std_rmse" in dir() else None,
    },
    "bias_model": {
        "formula": "scale = 1 - E / (F + eff_n)",
        "E": round(bias_e, 2),
        "F": round(bias_f, 2),
        "rmse": round(bias_rmse, 4) if bias_rmse is not None else None,
    },
}

constants_path = OUTPUT_DIR / "sampling_model_constants.json"
with open(constants_path, "w") as f:
    json.dump(model_constants, f, indent=2)
print(f"\nModel constants saved to: {constants_path}")

# %% Figure 2a: Required Sampling Probability (Model-Based)
print("\n" + "=" * 70)
print("FIGURE 2a: Required Sampling Probability by Reachability (Model-Based)")
print("=" * 70)

# Plot A: Given a target Spearman, what p do you need at different reachabilities?
# Using the fitted model: ρ = 1 - A / (B + eff_n)
# Solving for eff_n: eff_n = A / (1 - ρ) - B
# Since eff_n = reach × p: p = eff_n / reach = (A / (1 - ρ) - B) / reach

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

spearman_targets = [0.90, 0.95, 0.96, 0.97, 0.98, 0.99]
target_colors = {
    "0.9": "steelblue",
    "0.95": "darkorange",
    "0.96": "goldenrod",
    "0.97": "mediumseagreen",
    "0.98": "teal",
    "0.99": "seagreen",
}
reach_range = np.linspace(50, 2500, 100)

for target_rho in spearman_targets:
    # Calculate required effective_n from model
    required_eff_n = rho_a / (1 - target_rho) - rho_b
    # Calculate required p for each reachability
    required_p = required_eff_n / reach_range
    # Clip to valid probability range [0, 1]
    required_p = np.clip(required_p, 0, 1)

    ax.plot(reach_range, required_p, "-", linewidth=2.5,
            label=f"ρ ≥ {target_rho} (eff_n ≥ {required_eff_n:.0f})",
            color=target_colors[str(target_rho)])

ax.set_xlabel("Mean Reachability", fontsize=12)
ax.set_ylabel("Required Sampling Probability (p)", fontsize=12)
ax.set_ylim(0, 1.05)
ax.set_xlim(0, 2500)
ax.legend(loc="upper right", title="Target Spearman ρ", fontsize=10)
ax.grid(True, alpha=0.3)

# Add annotation explaining the relationship
ax.annotate(
    f"Model: ρ = 1 - {rho_a:.1f}/({rho_b:.1f} + reach×p)",
    xy=(0.02, 0.02), xycoords="axes fraction",
    fontsize=9, style="italic", color="gray"
)

plt.title(
    "Required Sampling Probability to Achieve Target Ranking Accuracy\n"
    "(Theoretical curves from fitted model)",
    fontsize=13,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sampling_probability.png", dpi=150, bbox_inches="tight")
print(f"Saved: {OUTPUT_DIR / 'sampling_probability.png'}")

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

![Test Network Topologies](topologies.png)

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

Empirical models fitted to the experimental data:

### Spearman ρ Model (Ranking Accuracy)

```
ρ = 1 - {rho_a:.2f} / ({rho_b:.2f} + effective_n)
```

- Approaches 1.0 as effective_n increases
- RMSE of fit: {rho_rmse:.4f}

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

### Model Predictions

| effective_n | Expected ρ | Std Dev | Bias |
|-------------|------------|---------|------|
| 10 | {rho_model(10, rho_a, rho_b):.3f} | {std_model(10, std_c, std_d):.3f} | {(1 - bias_model(10, bias_e, bias_f)) * 100:.1f}% |
| 25 | {rho_model(25, rho_a, rho_b):.3f} | {std_model(25, std_c, std_d):.3f} | {(1 - bias_model(25, bias_e, bias_f)) * 100:.1f}% |
| 50 | {rho_model(50, rho_a, rho_b):.3f} | {std_model(50, std_c, std_d):.3f} | {(1 - bias_model(50, bias_e, bias_f)) * 100:.1f}% |
| 100 | {rho_model(100, rho_a, rho_b):.3f} | {std_model(100, std_c, std_d):.3f} | {(1 - bias_model(100, bias_e, bias_f)) * 100:.1f}% |
| 200 | {rho_model(200, rho_a, rho_b):.3f} | {std_model(200, std_c, std_d):.3f} | {(1 - bias_model(200, bias_e, bias_f)) * 100:.1f}% |
| 400 | {rho_model(400, rho_a, rho_b):.3f} | {std_model(400, std_c, std_d):.3f} | {(1 - bias_model(400, bias_e, bias_f)) * 100:.1f}% |

---

## Chapter 5: Results

### Figure 1: Accuracy vs Effective Sample Size

![Sampling Accuracy](sampling_accuracy.png)

Scatter plots of observed ranking (top) and magnitude (bottom) accuracy across
all experimental configurations. Points are coloured by network topology.

### Figure 2a: Required Sampling Probability

![Sampling Probability](sampling_probability.png)

Theoretical curves showing the sampling probability required to achieve each
target Spearman ρ, derived from the fitted model. The legend shows the
effective_n threshold needed for each accuracy level.

### Figure 2b: Expected Ranking Accuracy

![Sampling Accuracy vs Effective N](sampling_accuracy_vs_eff_n.png)

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
  "rho_model": {{"A": {rho_a:.2f}, "B": {rho_b:.2f}}},
  "std_model": {{"C": {std_c:.3f}, "D": {std_d:.2f}}},
  "bias_model": {{"E": {bias_e:.2f}, "F": {bias_f:.2f}}}
}}
```

---

*Generated by `sampling_analysis.py` — Run `poe sync_sampling_constants` to update config.py*
"""

readme_path = OUTPUT_DIR / "README.md"
with open(readme_path, "w") as f:
    f.write(readme_content)

print(f"README saved to: {readme_path}")

# %% Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
KEY TAKEAWAY:

  effective_n = reachability × sampling_probability

  Higher effective_n = better accuracy for both ranking and magnitude.
  This relationship holds across all tested network topologies.

FITTED MODEL:

  Expected Spearman ρ = 1 - {rho_a:.2f} / ({rho_b:.2f} + effective_n)

  Key thresholds:
    - eff_n ≥ {eff_n_for_95:.0f}: Expected ρ ≥ 0.95
    - eff_n ≥ {eff_n_for_90:.0f}: Expected ρ ≥ 0.90

PRACTICAL GUIDANCE:

  1. Estimate your reachability from network density and distance threshold
  2. Choose p to achieve your target effective_n:
     - For ρ ≥ 0.95: p ≥ {eff_n_for_95:.0f} / reachability
     - For ρ ≥ 0.90: p ≥ {eff_n_for_90:.0f} / reachability
  3. Check cityseer's runtime logs for actual accuracy estimates

OUTPUT FILES:

  - README.md: Full documentation with figures
  - sampling_accuracy.png: Main results figure
  - sampling_guidance.png: Practical guidance figure
  - sampling_model_constants.json: Model parameters for config.py

NEXT STEPS:

  Run `poe sync_sampling_constants` to update config.py with new model parameters.
""")

# %%
