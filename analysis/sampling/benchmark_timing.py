# %% [markdown]
# # Timing Benchmark: Full vs Adaptive Sampling
#
# Measures wall-clock time for full computation (p=1.0) vs adaptive sampling
# across multiple distance thresholds. Demonstrates the speedup achievable
# with adaptive sampling, especially at larger distances where reachability
# is high enough to allow sampling.
#
# Uses the Madrid metropolitan street network to demonstrate dramatic speedup
# at regional distances (up to 20km).
#
# Note: Speedup is only achieved when reachability is high enough that the
# model allows sampling (p < 1.0). For small networks or short distances,
# full computation may still be required to meet accuracy targets.

# %% Imports
import time
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cityseer import config
from cityseer.tools import graphs, io

warnings.filterwarnings("ignore")

# %% Configuration
SCRIPT_DIR = Path(__file__).parent
FIGURES_DIR = SCRIPT_DIR / "paper" / "figures"
TABLES_DIR = SCRIPT_DIR / "paper" / "tables"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Benchmark parameters - use regional distances to show dramatic speedup
DISTANCES = [500, 1000, 2000, 5000, 10000, 20000]  # metres (up to 20km)
TARGET_RHO = 0.95
N_RUNS = 5  # Number of runs per configuration for timing stability

# Model parameters from sampling_reach.py (for computing required sample probability)
MODEL_HARMONIC_A = 32.40
MODEL_HARMONIC_B = 31.54
MODEL_BETWEENNESS_A = 48.31
MODEL_BETWEENNESS_B = 49.12

# Madrid network URL (regional-scale network)
MADRID_GPKG_URL = "https://github.com/songololo/ua-dataset-madrid/raw/main/data/street_network_w_edit.gpkg"


# %% Utility functions
def load_madrid_network():
    """Load the Madrid metropolitan network from GitHub."""
    print(f"Downloading Madrid network from: {MADRID_GPKG_URL}")
    print("  (This may take a minute...)")

    # Load directly from GitHub URL
    edges_gdf = gpd.read_file(MADRID_GPKG_URL)
    print(f"  Downloaded: {len(edges_gdf)} edges, CRS: {edges_gdf.crs}")

    # Convert multipart geoms to single
    edges_gdf_singles = edges_gdf.explode(index_parts=False)

    # Generate networkx graph
    G_nx = io.nx_from_generic_geopandas(edges_gdf_singles)

    # Clean graph: remove degree-2 nodes and danglers
    G_nx = graphs.nx_remove_filler_nodes(G_nx)
    G = graphs.nx_remove_dangling_nodes(G_nx)

    print(f"  Final graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Convert to cityseer network structure
    print("  Converting to cityseer format...")
    nodes_gdf, edges_gdf_out, network_structure = io.network_structure_from_nx(G)
    print(f"  Network structure ready: {len(nodes_gdf)} nodes")

    return nodes_gdf, edges_gdf_out, network_structure


def compute_required_effective_n(target_rho: float, metric: str = "both") -> float:
    """
    Compute the effective sample size required to achieve target accuracy.

    Uses the model: rho = 1 - A / (B + eff_n)
    Solving for eff_n: eff_n = A / (1 - target_rho) - B
    """
    if metric == "harmonic":
        a, b = MODEL_HARMONIC_A, MODEL_HARMONIC_B
    elif metric == "betweenness":
        a, b = MODEL_BETWEENNESS_A, MODEL_BETWEENNESS_B
    else:  # "both" - use betweenness (more conservative)
        a, b = MODEL_BETWEENNESS_A, MODEL_BETWEENNESS_B

    if target_rho >= 1.0:
        return float("inf")

    return a / (1 - target_rho) - b


def compute_sample_probability(reach: float, target_rho: float, metric: str = "both") -> float:
    """
    Compute the sampling probability required to achieve target accuracy.

    p = required_eff_n / reach
    """
    required_eff_n = compute_required_effective_n(target_rho, metric)
    if reach <= 0:
        return 1.0
    p = required_eff_n / reach
    return min(1.0, max(0.01, p))  # Clamp to [0.01, 1.0]


def run_timing_benchmark(
    network_structure,
    nodes_gdf,
    distances: list[int],
    target_rho: float,
    n_runs: int,
):
    """
    Run timing benchmark comparing full vs adaptive computation.

    Returns
    -------
    list[dict]
        Results for each distance with timing and speedup metrics.
    """
    results = []

    # Probe reachability for adaptive sampling plan
    print("\nProbing reachability...")
    reach_estimates = config.probe_reachability(network_structure, distances)
    for d in distances:
        print(f"  {d}m: reach = {reach_estimates[d]:.0f}")

    # Compute adaptive sampling probabilities
    sample_probs = config.compute_sample_probs_for_target_rho(reach_estimates, target_rho, metric="both")
    print("\nAdaptive sampling plan:")
    for d in distances:
        p = sample_probs.get(d)
        p_str = "full" if p is None or p >= 1.0 else f"{p:.1%}"
        print(f"  {d}m: p = {p_str}")

    for dist in distances:
        print(f"\n{'=' * 50}")
        print(f"Distance {dist}m")
        print(f"{'=' * 50}")

        reach = reach_estimates.get(dist, 100)
        sample_p = sample_probs.get(dist)
        effective_p = 1.0 if sample_p is None or sample_p >= 1.0 else sample_p

        # Full computation timing (multiple runs)
        full_times = []
        print(f"\n  Full computation (p=1.0):")
        for run in range(n_runs):
            t0 = time.perf_counter()
            _ = network_structure.local_node_centrality_shortest(
                distances=[dist],
                compute_closeness=True,
                compute_betweenness=True,
                pbar_disabled=True,
            )
            elapsed = time.perf_counter() - t0
            full_times.append(elapsed)
            print(f"    Run {run + 1}: {elapsed:.3f}s")

        full_time_mean = np.mean(full_times)
        full_time_std = np.std(full_times)
        print(f"    Mean: {full_time_mean:.3f}s (std: {full_time_std:.3f}s)")

        # Adaptive computation timing (multiple runs)
        adaptive_times = []
        print(f"\n  Adaptive computation (p={effective_p:.1%}):")
        for run in range(n_runs):
            t0 = time.perf_counter()
            _ = network_structure.local_node_centrality_shortest(
                distances=[dist],
                compute_closeness=True,
                compute_betweenness=True,
                sample_probability=effective_p,
                random_seed=run,
                pbar_disabled=True,
            )
            elapsed = time.perf_counter() - t0
            adaptive_times.append(elapsed)
            print(f"    Run {run + 1}: {elapsed:.3f}s")

        adaptive_time_mean = np.mean(adaptive_times)
        adaptive_time_std = np.std(adaptive_times)
        print(f"    Mean: {adaptive_time_mean:.3f}s (std: {adaptive_time_std:.3f}s)")

        # Compute speedup
        speedup = full_time_mean / adaptive_time_mean if adaptive_time_mean > 0 else 1.0
        print(f"\n  Speedup: {speedup:.2f}x")

        results.append({
            "distance_m": dist,
            "reach_estimate": reach,
            "sample_probability": effective_p,
            "full_time_mean": full_time_mean,
            "full_time_std": full_time_std,
            "adaptive_time_mean": adaptive_time_mean,
            "adaptive_time_std": adaptive_time_std,
            "speedup": speedup,
        })

    return results


def format_distance(d: int) -> str:
    """Format distance as meters or kilometers."""
    if d >= 1000:
        return f"{d // 1000}km"
    return f"{d}m"


def generate_speedup_figure(results: list[dict], n_nodes: int):
    """Generate speedup vs distance figure."""
    df = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left plot: Bar chart of timing comparison
    ax = axes[0]
    x = np.arange(len(df))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        df["full_time_mean"],
        width,
        yerr=df["full_time_std"],
        label="Full (p=1.0)",
        color="#e74c3c",
        capsize=3,
    )
    bars2 = ax.bar(
        x + width / 2,
        df["adaptive_time_mean"],
        width,
        yerr=df["adaptive_time_std"],
        label="Adaptive",
        color="#2ecc71",
        capsize=3,
    )

    ax.set_xlabel("Distance Threshold", fontsize=11)
    ax.set_ylabel("Computation Time (seconds)", fontsize=11)
    ax.set_title("Computation Time: Full vs Adaptive", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([format_distance(d) for d in df["distance_m"]])
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    # Add speedup annotations
    for i, (b1, b2) in enumerate(zip(bars1, bars2, strict=False)):
        speedup = df.iloc[i]["speedup"]
        # Color based on whether sampling was used
        p = df.iloc[i]["sample_probability"]
        color = "#2ecc71" if p < 1.0 else "#95a5a6"
        ax.annotate(
            f"{speedup:.1f}x",
            xy=(b1.get_x() + b1.get_width(), max(b1.get_height(), b2.get_height()) * 1.05),
            ha="center",
            fontsize=9,
            color=color,
            fontweight="bold",
        )

    # Right plot: Speedup vs distance with sampling probability context
    ax = axes[1]

    # Color bars based on whether sampling was used
    colors = ["#2ecc71" if p < 1.0 else "#95a5a6" for p in df["sample_probability"]]
    bars = ax.bar(
        x,
        df["speedup"],
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )

    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        p = df.iloc[i]["sample_probability"]
        ax.annotate(
            f"{height:.1f}x",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
        # Add sampling probability below
        ax.annotate(
            f"p={p:.0%}" if p < 1.0 else "full",
            xy=(bar.get_x() + bar.get_width() / 2, 0.05),
            ha="center",
            fontsize=8,
            color="white" if p < 1.0 else "gray",
            fontweight="bold",
        )

    ax.set_xlabel("Distance Threshold", fontsize=11)
    ax.set_ylabel("Speedup Factor (Full / Adaptive)", fontsize=11)
    ax.set_title(f"Speedup vs Distance (target rho={TARGET_RHO})", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([format_distance(d) for d in df["distance_m"]])
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7, label="No speedup")
    ax.set_ylim(0, max(df["speedup"]) * 1.2)
    ax.grid(True, alpha=0.3, axis="y")

    # Add legend for bar colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ecc71", edgecolor="black", label="Sampling used (p<1.0)"),
        Patch(facecolor="#95a5a6", edgecolor="black", label="Full computation (p=1.0)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

    plt.suptitle(
        f"Adaptive Sampling Speedup: Madrid Network ({n_nodes:,} nodes)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "speedup_vs_distance.pdf", dpi=300, bbox_inches="tight")
    print(f"\nSaved: {FIGURES_DIR / 'speedup_vs_distance.pdf'}")


def generate_report(results: list[dict], n_nodes: int):
    """Generate summary report and tables."""
    df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("TIMING BENCHMARK RESULTS")
    print("=" * 70)
    print(f"\nNetwork: Madrid ({n_nodes:,} nodes)")
    print(f"Target accuracy: rho >= {TARGET_RHO}")
    print(f"Runs per configuration: {N_RUNS}")
    print()

    # Summary table
    print("-" * 90)
    print(
        f"{'Distance':<10} {'Reach':<10} {'Sample p':<10} {'Full (s)':<12} {'Adaptive (s)':<14} {'Speedup':<10}"
    )
    print("-" * 90)

    for r in results:
        print(
            f"{r['distance_m']:<10} "
            f"{r['reach_estimate']:<10.0f} "
            f"{r['sample_probability']:<10.1%} "
            f"{r['full_time_mean']:<12.3f} "
            f"{r['adaptive_time_mean']:<14.3f} "
            f"{r['speedup']:<10.1f}x"
        )

    print("-" * 90)

    # Totals
    total_full = sum(r["full_time_mean"] for r in results)
    total_adaptive = sum(r["adaptive_time_mean"] for r in results)
    total_speedup = total_full / total_adaptive if total_adaptive > 0 else 1.0

    print(f"\nMulti-distance total ({len(results)} distances):")
    print(f"  Full computation:     {total_full:.2f}s")
    print(f"  Adaptive sampling:    {total_adaptive:.2f}s")
    print(f"  Overall speedup:      {total_speedup:.1f}x")

    # Save to CSV
    df.to_csv(TABLES_DIR / "timing_benchmark.csv", index=False)
    print(f"\nSaved: {TABLES_DIR / 'timing_benchmark.csv'}")


# %% Main execution
if __name__ == "__main__":
    print("=" * 70)
    print("TIMING BENCHMARK: FULL VS ADAPTIVE SAMPLING")
    print("=" * 70)
    print(f"Distances: {[format_distance(d) for d in DISTANCES]}")
    print(f"Target rho: {TARGET_RHO}")
    print(f"Runs per configuration: {N_RUNS}")
    print(f"Output: {FIGURES_DIR}")

    # Load network
    nodes_gdf, edges_gdf, network_structure = load_madrid_network()
    n_nodes = len(nodes_gdf)

    # Run benchmark
    results = run_timing_benchmark(
        network_structure,
        nodes_gdf,
        distances=DISTANCES,
        target_rho=TARGET_RHO,
        n_runs=N_RUNS,
    )

    # Generate outputs
    generate_speedup_figure(results, n_nodes)
    generate_report(results, n_nodes)

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
