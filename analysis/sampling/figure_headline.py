"""
Headline Figure: Uniform vs Adaptive Sampling Comparison

This figure distills the core insight into a compelling 2-panel comparison:
- Panel A: Uniform sampling FAILS at short distances (accuracy varies wildly)
- Panel B: Adaptive sampling SUCCEEDS (consistent accuracy across all distances)

The key insight:
With uniform sampling probability across all distances:
- Short distances have LOW reachability -> LOW effective_n -> POOR accuracy
- Long distances have HIGH reachability -> HIGH effective_n -> GOOD accuracy

With adaptive per-distance sampling:
- Probability is calibrated per distance to maintain consistent effective_n -> CONSISTENT accuracy
"""

import pickle
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from cityseer.tools import io
from cityseer.config import compute_required_p, get_extended_model_params
from scipy import stats as scipy_stats

# Configure matplotlib for publication quality
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})

# Directories
SCRIPT_DIR = Path(__file__).parent
CACHE_DIR = SCRIPT_DIR.parent.parent / "temp" / "sampling_cache"
FIGURES_DIR = SCRIPT_DIR / "paper" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Model parameters from sampling_reach.py
MODEL_HARMONIC_A = 32.40
MODEL_HARMONIC_B = 31.54
MODEL_BETWEENNESS_A = 48.31
MODEL_BETWEENNESS_B = 49.12

# Configuration
DISTANCES = [200, 500, 1000, 2000, 5000]
UNIFORM_PROB = 0.3
TARGET_RHO = 0.95
N_RUNS = 5  # Number of runs to average for accuracy estimate


def rho_model(eff_n: float, a: float, b: float) -> float:
    """Predict Spearman rho from effective sample size."""
    return 1 - a / (b + eff_n)


def required_eff_n_for_rho(target_rho: float, a: float, b: float) -> float:
    """Compute the effective_n required to achieve target rho."""
    # rho = 1 - a / (b + eff_n)
    # 1 - rho = a / (b + eff_n)
    # b + eff_n = a / (1 - rho)
    # eff_n = a / (1 - rho) - b
    return a / (1 - target_rho) - b


def compute_accuracy(true_vals: np.ndarray, est_vals: np.ndarray) -> float:
    """Compute Spearman correlation between true and estimated values."""
    mask = (true_vals > 0) & np.isfinite(true_vals) & np.isfinite(est_vals)
    if mask.sum() < 10:
        return np.nan
    rho, _ = scipy_stats.spearmanr(true_vals[mask], est_vals[mask])
    return rho


def load_london_network():
    """Load the London network from cache."""
    network_file = CACHE_DIR / "network_london_v1.pkl"

    if not network_file.exists():
        raise FileNotFoundError(
            f"London network cache not found at {network_file}. "
            "Please run the sampling analysis first to generate this cache."
        )

    print(f"Loading London network from {network_file}...")
    with open(network_file, "rb") as f:
        network_data = pickle.load(f)

    # Reconstruct network structure from geodataframes
    nodes_gdf = network_data["nodes_gdf"]
    edges_gdf = network_data["edges_gdf"]
    network_structure = io.network_structure_from_gpd(nodes_gdf, edges_gdf)

    print(f"  Loaded: {len(nodes_gdf)} nodes")
    return network_structure, nodes_gdf


def run_uniform_sampling_experiment(net, distances: list[int], sample_prob: float, n_runs: int):
    """
    Run centrality at each distance with uniform sampling probability.

    Returns dict mapping distance -> {"rho_harmonic": float, "rho_betweenness": float,
                                        "mean_reach": float, "eff_n": float}
    """
    results = {}

    for dist in distances:
        print(f"  Distance {dist}m: computing ground truth...")

        # Ground truth (full computation)
        true_result = net.local_node_centrality_shortest(
            distances=[dist],
            compute_closeness=True,
            compute_betweenness=True,
            pbar_disabled=True,
        )
        true_harmonic = np.array(true_result.node_harmonic[dist])
        true_betweenness = np.array(true_result.node_betweenness[dist])
        mean_reach = float(np.mean(np.array(true_result.node_density[dist])))

        print(f"    Mean reachability: {mean_reach:.0f}")

        # Sampled estimates (multiple runs)
        rhos_h = []
        rhos_b = []

        for seed in range(n_runs):
            r = net.local_node_centrality_shortest(
                distances=[dist],
                compute_closeness=True,
                compute_betweenness=True,
                sample_probability=sample_prob,
                random_seed=seed,
                pbar_disabled=True,
            )
            est_harmonic = np.array(r.node_harmonic[dist])
            est_betweenness = np.array(r.node_betweenness[dist])

            rho_h = compute_accuracy(true_harmonic, est_harmonic)
            rho_b = compute_accuracy(true_betweenness, est_betweenness)

            if not np.isnan(rho_h):
                rhos_h.append(rho_h)
            if not np.isnan(rho_b):
                rhos_b.append(rho_b)

        results[dist] = {
            "rho_harmonic": np.mean(rhos_h) if rhos_h else np.nan,
            "rho_harmonic_std": np.std(rhos_h) if rhos_h else np.nan,
            "rho_betweenness": np.mean(rhos_b) if rhos_b else np.nan,
            "rho_betweenness_std": np.std(rhos_b) if rhos_b else np.nan,
            "mean_reach": mean_reach,
            "eff_n": mean_reach * sample_prob,
            "sample_prob": sample_prob,
        }

        print(f"    Uniform p={sample_prob:.0%}: rho_H={results[dist]['rho_harmonic']:.3f}, "
              f"rho_B={results[dist]['rho_betweenness']:.3f}, eff_n={results[dist]['eff_n']:.0f}")

    return results


def run_adaptive_sampling_experiment(net, distances: list[int], target_rho: float, n_runs: int):
    """
    Run centrality at each distance with adaptive per-distance probability.

    The probability at each distance is calibrated to achieve target_rho.

    Returns dict mapping distance -> {"rho_harmonic": float, "rho_betweenness": float,
                                        "mean_reach": float, "sample_prob": float, "eff_n": float}
    """
    results = {}

    # First, get reachability at each distance
    reach_estimates = {}
    for dist in distances:
        true_result = net.local_node_centrality_shortest(
            distances=[dist],
            compute_closeness=True,
            compute_betweenness=False,  # Just need density
            pbar_disabled=True,
        )
        reach_estimates[dist] = float(np.mean(np.array(true_result.node_density[dist])))

    # Compute adaptive probabilities using extended model
    # For SHORTEST paths, harmonic closeness is MORE demanding than betweenness
    # (requires ~2.7x more samples), so use harmonic model for conservative estimates
    # Extended model: rho = 1 - A / (B + reach * p^(1+k))
    a, b, k = get_extended_model_params("harmonic", "shortest")
    print(f"\n  Using extended model (A={a}, B={b}, k={k}) for harmonic (shortest path)")

    adaptive_probs = {}
    for dist in distances:
        reach = reach_estimates[dist]
        # Use extended model inversion: p = ((A / (1 - rho) - B) / reach)^(1/(1+k))
        p = compute_required_p(reach, target_rho, "harmonic", "shortest")
        if p is None or p > 1.0:
            p = 1.0
        adaptive_probs[dist] = p
        print(f"    {dist}m: reach={reach:.0f}, p={p:.2%}")

    # Now run the experiment with adaptive probabilities
    for dist in distances:
        p = adaptive_probs[dist]

        print(f"\n  Distance {dist}m with adaptive p={p:.2%}...")

        # Ground truth
        true_result = net.local_node_centrality_shortest(
            distances=[dist],
            compute_closeness=True,
            compute_betweenness=True,
            pbar_disabled=True,
        )
        true_harmonic = np.array(true_result.node_harmonic[dist])
        true_betweenness = np.array(true_result.node_betweenness[dist])
        mean_reach = float(np.mean(np.array(true_result.node_density[dist])))

        # If p >= 1.0, skip sampling (it's full computation)
        if p >= 1.0:
            results[dist] = {
                "rho_harmonic": 1.0,
                "rho_harmonic_std": 0.0,
                "rho_betweenness": 1.0,
                "rho_betweenness_std": 0.0,
                "mean_reach": mean_reach,
                "eff_n": mean_reach,  # Full computation: eff_n = reach
                "sample_prob": 1.0,
            }
            print(f"    Full computation (p=100%): rho=1.0")
            continue

        # Sampled estimates
        rhos_h = []
        rhos_b = []

        for seed in range(n_runs):
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

            rho_h = compute_accuracy(true_harmonic, est_harmonic)
            rho_b = compute_accuracy(true_betweenness, est_betweenness)

            if not np.isnan(rho_h):
                rhos_h.append(rho_h)
            if not np.isnan(rho_b):
                rhos_b.append(rho_b)

        results[dist] = {
            "rho_harmonic": np.mean(rhos_h) if rhos_h else np.nan,
            "rho_harmonic_std": np.std(rhos_h) if rhos_h else np.nan,
            "rho_betweenness": np.mean(rhos_b) if rhos_b else np.nan,
            "rho_betweenness_std": np.std(rhos_b) if rhos_b else np.nan,
            "mean_reach": mean_reach,
            "eff_n": mean_reach * p,
            "sample_prob": p,
        }

        print(f"    Adaptive p={p:.2%}: rho_H={results[dist]['rho_harmonic']:.3f}, "
              f"rho_B={results[dist]['rho_betweenness']:.3f}, eff_n={results[dist]['eff_n']:.0f}")

    return results


def generate_headline_figure(uniform_results: dict, adaptive_results: dict, distances: list[int]):
    """Generate the 2-panel headline figure."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # Extract data for plotting
    x = np.array(distances)

    # Uniform sampling data
    uniform_rho_h = [uniform_results[d]["rho_harmonic"] for d in distances]
    uniform_rho_b = [uniform_results[d]["rho_betweenness"] for d in distances]
    uniform_rho_h_std = [uniform_results[d]["rho_harmonic_std"] for d in distances]
    uniform_rho_b_std = [uniform_results[d]["rho_betweenness_std"] for d in distances]
    uniform_eff_n = [uniform_results[d]["eff_n"] for d in distances]

    # Adaptive sampling data
    adaptive_rho_h = [adaptive_results[d]["rho_harmonic"] for d in distances]
    adaptive_rho_b = [adaptive_results[d]["rho_betweenness"] for d in distances]
    adaptive_rho_h_std = [adaptive_results[d]["rho_harmonic_std"] for d in distances]
    adaptive_rho_b_std = [adaptive_results[d]["rho_betweenness_std"] for d in distances]
    adaptive_eff_n = [adaptive_results[d]["eff_n"] for d in distances]
    adaptive_probs = [adaptive_results[d]["sample_prob"] for d in distances]

    # Colors
    color_h = "#2166AC"  # Blue for harmonic
    color_b = "#B2182B"  # Red for betweenness

    # =========================================================================
    # Panel A: Uniform Sampling
    # =========================================================================
    ax = axes[0]

    # Plot with error bars
    ax.errorbar(x, uniform_rho_h, yerr=uniform_rho_h_std,
                fmt="o-", color=color_h, linewidth=2, markersize=8,
                capsize=4, label="Closeness (harmonic)", alpha=0.9)
    ax.errorbar(x, uniform_rho_b, yerr=uniform_rho_b_std,
                fmt="s-", color=color_b, linewidth=2, markersize=8,
                capsize=4, label="Betweenness", alpha=0.9)

    # Target line
    ax.axhline(y=TARGET_RHO, color="green", linestyle="--", linewidth=2,
               alpha=0.7, label=f"Target rho = {TARGET_RHO}")

    # Add effective_n annotations
    for i, (d, eff_n) in enumerate(zip(distances, uniform_eff_n)):
        ax.annotate(f"eff_n={eff_n:.0f}",
                    xy=(d, min(uniform_rho_h[i], uniform_rho_b[i]) - 0.03),
                    ha="center", va="top", fontsize=8, color="gray")

    ax.set_xlabel("Distance Threshold (m)")
    ax.set_ylabel("Spearman rho (ranking accuracy)")
    ax.set_title("Uniform sampling", fontsize=11, fontweight="bold")
    ax.set_ylim(0.6, 1.02)
    ax.set_xscale("log")
    ax.set_xticks(distances)
    ax.set_xticklabels([str(d) for d in distances])
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Add failure zone annotation
    ax.fill_between([min(distances) * 0.8, max(distances) * 1.2],
                    0.6, TARGET_RHO, alpha=0.1, color="red", zorder=0)
    ax.text(300, 0.75, "FAILURE\nZONE", ha="center", va="center",
            fontsize=10, color="red", alpha=0.7, fontweight="bold")

    # =========================================================================
    # Panel B: Adaptive Sampling
    # =========================================================================
    ax = axes[1]

    # Plot with error bars
    ax.errorbar(x, adaptive_rho_h, yerr=adaptive_rho_h_std,
                fmt="o-", color=color_h, linewidth=2, markersize=8,
                capsize=4, label="Closeness (harmonic)", alpha=0.9)
    ax.errorbar(x, adaptive_rho_b, yerr=adaptive_rho_b_std,
                fmt="s-", color=color_b, linewidth=2, markersize=8,
                capsize=4, label="Betweenness", alpha=0.9)

    # Target line
    ax.axhline(y=TARGET_RHO, color="green", linestyle="--", linewidth=2,
               alpha=0.7, label=f"Target rho = {TARGET_RHO}")

    # Add probability annotations
    for i, (d, p) in enumerate(zip(distances, adaptive_probs)):
        ax.annotate(f"p={p:.0%}" if p < 1.0 else "p=100%",
                    xy=(d, max(adaptive_rho_h[i], adaptive_rho_b[i]) + 0.01),
                    ha="center", va="bottom", fontsize=8, color="gray")

    ax.set_xlabel("Distance Threshold (m)")
    ax.set_title("Adaptive sampling", fontsize=11, fontweight="bold")
    ax.set_ylim(0.6, 1.02)
    ax.set_xscale("log")
    ax.set_xticks(distances)
    ax.set_xticklabels([str(d) for d in distances])
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Add success zone annotation
    ax.fill_between([min(distances) * 0.8, max(distances) * 1.2],
                    TARGET_RHO, 1.02, alpha=0.1, color="green", zorder=0)
    ax.text(1000, 0.98, "SUCCESS ZONE", ha="center", va="center",
            fontsize=10, color="green", alpha=0.7, fontweight="bold")

    # =========================================================================
    # Overall title
    # =========================================================================
    # No suptitle - let the caption explain

    plt.tight_layout()

    # Save
    output_path = FIGURES_DIR / "headline_comparison.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\nSaved: {output_path}")

    plt.close()


def print_summary(uniform_results: dict, adaptive_results: dict, distances: list[int]):
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("SUMMARY: UNIFORM vs ADAPTIVE SAMPLING")
    print("=" * 70)

    print(f"\n{'Distance':<10} {'Uniform':<25} {'Adaptive':<25} {'Improvement':<15}")
    print(f"{'(m)':<10} {'rho_H    rho_B    eff_n':<25} {'rho_H    rho_B    p':<25} {'rho_B diff':<15}")
    print("-" * 75)

    for d in distances:
        u = uniform_results[d]
        a = adaptive_results[d]
        improvement = a["rho_betweenness"] - u["rho_betweenness"]

        print(f"{d:<10} "
              f"{u['rho_harmonic']:.3f}    {u['rho_betweenness']:.3f}    {u['eff_n']:>5.0f}     "
              f"{a['rho_harmonic']:.3f}    {a['rho_betweenness']:.3f}    {a['sample_prob']:>5.0%}     "
              f"{improvement:+.3f}")

    print("-" * 75)

    # Overall statistics
    uniform_min_rho = min(min(uniform_results[d]["rho_harmonic"], uniform_results[d]["rho_betweenness"])
                          for d in distances)
    adaptive_min_rho = min(min(adaptive_results[d]["rho_harmonic"], adaptive_results[d]["rho_betweenness"])
                           for d in distances)

    print(f"\nUniform sampling: worst-case rho = {uniform_min_rho:.3f}")
    print(f"Adaptive sampling: worst-case rho = {adaptive_min_rho:.3f}")
    print(f"Target rho: {TARGET_RHO}")

    n_uniform_failures = sum(1 for d in distances
                             if uniform_results[d]["rho_betweenness"] < TARGET_RHO)
    n_adaptive_failures = sum(1 for d in distances
                              if adaptive_results[d]["rho_betweenness"] < TARGET_RHO)

    print(f"\nDistances failing target (uniform): {n_uniform_failures}/{len(distances)}")
    print(f"Distances failing target (adaptive): {n_adaptive_failures}/{len(distances)}")


# =============================================================================
# Main execution
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("HEADLINE FIGURE: Uniform vs Adaptive Sampling")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Distances: {DISTANCES}")
    print(f"  Uniform probability: {UNIFORM_PROB:.0%}")
    print(f"  Target rho: {TARGET_RHO}")
    print(f"  Runs per config: {N_RUNS}")
    print(f"  Output: {FIGURES_DIR}")

    # Load network
    t0 = time.perf_counter()
    net, nodes_gdf = load_london_network()
    print(f"  Network loaded in {time.perf_counter() - t0:.1f}s")

    # Run uniform sampling experiment
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Uniform Sampling")
    print("=" * 70)
    uniform_results = run_uniform_sampling_experiment(net, DISTANCES, UNIFORM_PROB, N_RUNS)

    # Run adaptive sampling experiment
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Adaptive Sampling")
    print("=" * 70)
    adaptive_results = run_adaptive_sampling_experiment(net, DISTANCES, TARGET_RHO, N_RUNS)

    # Generate figure
    print("\n" + "=" * 70)
    print("GENERATING FIGURE")
    print("=" * 70)
    generate_headline_figure(uniform_results, adaptive_results, DISTANCES)

    # Print summary
    print_summary(uniform_results, adaptive_results, DISTANCES)

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
