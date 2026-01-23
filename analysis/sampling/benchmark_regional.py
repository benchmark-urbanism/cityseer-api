# %% [markdown]
# # Regional-Scale Network Benchmark
#
# Benchmarks adaptive sampling on a regional-scale street network.
# This script demonstrates the practical speedup achievable with adaptive sampling
# at regional distances (5km-20km) using the Madrid metropolitan street network.
#
# The Madrid network is sourced from:
# https://github.com/songololo/ua-dataset-madrid

# %% Imports
import time
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cityseer.metrics import networks
from cityseer.tools import graphs, io
from scipy import stats as scipy_stats
from scipy.spatial import KDTree

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).parent if "__file__" in dir() else Path(".")
PAPER_DIR = SCRIPT_DIR / "paper"
FIGURES_DIR = PAPER_DIR / "figures"
TABLES_DIR = PAPER_DIR / "tables"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Madrid network URL
MADRID_GPKG_URL = "https://github.com/songololo/ua-dataset-madrid/raw/main/data/street_network_w_edit.gpkg"

# %% Load Madrid Network
print(f"\n  Downloading Madrid network from: {MADRID_GPKG_URL}")
print("  (This may take a minute...)")

# Load directly from GitHub URL
edges_gdf = gpd.read_file(MADRID_GPKG_URL)
print(f"    Downloaded: {len(edges_gdf)} edges, CRS: {edges_gdf.crs}")

# Convert multipart geoms to single
edges_gdf_singles = edges_gdf.explode(drop=True)

# Generate networkx graph
G_nx = io.nx_from_generic_geopandas(edges_gdf_singles)

# Clean graph: remove degree-2 nodes and danglers
G_nx = graphs.nx_remove_filler_nodes(G_nx)
G = graphs.nx_remove_dangling_nodes(G_nx)

print(f"    Final graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# %% Convert to cityseer network structure
print("    Converting to cityseer format...")
nodes_gdf, edges_gdf_out, network_structure = io.network_structure_from_nx(G)
print(f"    Network structure ready: {len(nodes_gdf)} nodes")

# %% Helper functions


def compute_accuracy_metrics(true_vals: np.ndarray, est_vals: np.ndarray) -> tuple[float, float]:
    """Compute ranking accuracy metrics."""
    mask = (true_vals > 0) & np.isfinite(true_vals) & np.isfinite(est_vals)
    if mask.sum() < 10:
        return np.nan, np.nan

    true_masked = true_vals[mask]
    est_masked = est_vals[mask]

    spearman, _ = scipy_stats.spearmanr(true_masked, est_masked)
    ratios = est_masked / true_masked
    scale_ratio = float(np.median(ratios))

    return spearman, scale_ratio


def compute_morans_i(values: np.ndarray, coords: np.ndarray, k: int = 8) -> tuple[float, float]:
    """
    Compute Moran's I spatial autocorrelation statistic.

    Uses k-nearest neighbors for spatial weights.

    Returns
    -------
    tuple[float, float]
        (Moran's I, p-value)
    """
    n = len(values)
    if n < 10:
        return np.nan, np.nan

    # Standardize values
    mean_val = np.mean(values)
    dev = values - mean_val
    var = np.var(values)
    if var == 0:
        return np.nan, np.nan

    # Build k-nearest neighbor weights
    tree = KDTree(coords)
    _, indices = tree.query(coords, k=k + 1)  # +1 includes self
    indices = indices[:, 1:]  # Remove self

    # Compute Moran's I
    numerator = 0.0
    weight_sum = 0.0

    for i in range(n):
        for j in indices[i]:
            numerator += dev[i] * dev[j]
            weight_sum += 1.0

    if weight_sum == 0:
        return np.nan, np.nan

    morans_i = (n / weight_sum) * (numerator / (n * var))

    # Expected value under null hypothesis
    expected_i = -1.0 / (n - 1)

    # Approximate variance under randomization assumption
    s1 = 2 * weight_sum
    s2 = 4 * k * n
    b2 = np.mean(dev**4) / var**2  # Kurtosis

    variance_i = (
        (n * ((n**2 - 3 * n + 3) * s1 - n * s2 + 3 * weight_sum**2))
        - b2 * ((n**2 - n) * s1 - 2 * n * s2 + 6 * weight_sum**2)
    ) / ((n - 1) * (n - 2) * (n - 3) * weight_sum**2) - expected_i**2

    if variance_i <= 0:
        return morans_i, np.nan

    z_score = (morans_i - expected_i) / np.sqrt(variance_i)
    p_value = 2 * (1 - scipy_stats.norm.cdf(abs(z_score)))

    return morans_i, p_value


# %% Spatial Analysis Functions


def run_spatial_analysis(
    net,
    nodes_gdf: gpd.GeoDataFrame,
    dist: int,
    sample_prob: float,
    n_runs: int = 3,
):
    """
    Run spatial analysis comparing accuracy across areas with different reachability.

    Key question: Do sampling accuracy findings hold spatially in all areas,
    especially in sparse/low-reachability regions?

    Returns
    -------
    dict with:
        - reach_quartile_accuracy: accuracy by reachability quartile
        - morans_i: spatial autocorrelation of residuals
        - per_node_data: detailed per-node results
    """
    print(f"\n    Spatial analysis for {dist}m, p={sample_prob:.0%}...")

    # Get coordinates
    coords = np.column_stack([nodes_gdf.geometry.x.values, nodes_gdf.geometry.y.values])

    # Ground truth
    print("      Computing ground truth...")
    true_result = net.local_node_centrality_shortest(
        distances=[dist],
        compute_closeness=True,
        compute_betweenness=True,
        pbar_disabled=True,
    )
    true_harmonic = np.array(true_result.node_harmonic[dist])
    true_betweenness = np.array(true_result.node_betweenness[dist])
    reachability = np.array(true_result.node_density[dist])

    # Sampled estimates (averaged over runs)
    print(f"      Computing sampled estimates ({n_runs} runs)...")
    est_harmonic_runs = []
    est_betweenness_runs = []

    for seed in range(n_runs):
        r = net.local_node_centrality_shortest(
            distances=[dist],
            compute_closeness=True,
            compute_betweenness=True,
            sample_probability=sample_prob,
            random_seed=seed,
            pbar_disabled=True,
        )
        est_harmonic_runs.append(np.array(r.node_harmonic[dist]))
        est_betweenness_runs.append(np.array(r.node_betweenness[dist]))

    est_harmonic = np.mean(est_harmonic_runs, axis=0)
    est_betweenness = np.mean(est_betweenness_runs, axis=0)

    # Residuals (relative error)
    harmonic_residual = np.where(true_harmonic > 0, (est_harmonic - true_harmonic) / true_harmonic, 0)
    betweenness_residual = np.where(true_betweenness > 0, (est_betweenness - true_betweenness) / true_betweenness, 0)

    # Moran's I for spatial autocorrelation
    print("      Computing spatial autocorrelation...")
    morans_h, p_h = compute_morans_i(harmonic_residual, coords)
    morans_b, p_b = compute_morans_i(betweenness_residual, coords)

    # Analysis by reachability quartile
    print("      Analyzing by reachability quartile...")
    reach_quartiles = pd.qcut(reachability, 4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"])

    quartile_results = []
    for q_label in ["Q1 (low)", "Q2", "Q3", "Q4 (high)"]:
        mask = reach_quartiles == q_label
        if mask.sum() < 10:
            continue

        # Accuracy within this quartile
        q_true_h = true_harmonic[mask]
        q_est_h = est_harmonic[mask]
        q_true_b = true_betweenness[mask]
        q_est_b = est_betweenness[mask]

        valid_h = (q_true_h > 0) & np.isfinite(q_true_h) & np.isfinite(q_est_h)
        valid_b = (q_true_b > 0) & np.isfinite(q_true_b) & np.isfinite(q_est_b)

        rho_h = scipy_stats.spearmanr(q_true_h[valid_h], q_est_h[valid_h])[0] if valid_h.sum() > 10 else np.nan
        rho_b = scipy_stats.spearmanr(q_true_b[valid_b], q_est_b[valid_b])[0] if valid_b.sum() > 10 else np.nan

        mean_reach = reachability[mask].mean()
        eff_n = mean_reach * sample_prob

        quartile_results.append(
            {
                "quartile": q_label,
                "n_nodes": int(mask.sum()),
                "mean_reachability": mean_reach,
                "effective_n": eff_n,
                "rho_harmonic": rho_h,
                "rho_betweenness": rho_b,
            }
        )

    return {
        "distance": dist,
        "sample_prob": sample_prob,
        "morans_i_harmonic": morans_h,
        "morans_p_harmonic": p_h,
        "morans_i_betweenness": morans_b,
        "morans_p_betweenness": p_b,
        "quartile_results": quartile_results,
        "coords": coords,
        "reachability": reachability,
        "harmonic_residual": harmonic_residual,
        "betweenness_residual": betweenness_residual,
    }


# %% Benchmark function


def run_benchmark(
    net,
    nodes_gdf: gpd.GeoDataFrame,
    distances: list[int],
    target_rho: float = 0.95,
    n_accuracy_runs: int = 3,
):
    """
    Run benchmark comparing full computation vs adaptive sampling.

    Returns
    -------
    results : list[dict]
        Timing and accuracy results for each distance
    """
    results = []

    for dist in distances:
        print(f"\n  Distance {dist}m ({dist / 1000:.0f}km):")

        # Full computation timing
        print("    Running full computation...")
        t0 = time.perf_counter()
        full_result = net.local_node_centrality_shortest(
            distances=[dist],
            compute_closeness=True,
            compute_betweenness=True,
            pbar_disabled=True,
        )
        full_time = time.perf_counter() - t0
        print(f"      Full computation: {full_time:.2f}s")

        true_harmonic = np.array(full_result.node_harmonic[dist])
        true_betweenness = np.array(full_result.node_betweenness[dist])
        mean_reach = float(np.mean(np.array(full_result.node_density[dist])))

        print(f"      Mean reachability: {mean_reach:.0f} nodes")

        # Adaptive sampling timing
        print("    Running adaptive sampling...")
        t0 = time.perf_counter()
        adaptive_gdf = networks.node_centrality_shortest_adaptive(
            net,
            nodes_gdf,
            distances=[dist],
            target_rho=target_rho,
            compute_closeness=True,
            compute_betweenness=True,
        )
        adaptive_time = time.perf_counter() - t0
        print(f"      Adaptive sampling: {adaptive_time:.2f}s")

        est_harmonic = np.array(adaptive_gdf[f"cc_harmonic_{dist}"])
        est_betweenness = np.array(adaptive_gdf[f"cc_betweenness_{dist}"])

        # Single-run accuracy
        rho_h, _ = compute_accuracy_metrics(true_harmonic, est_harmonic)
        rho_b, _ = compute_accuracy_metrics(true_betweenness, est_betweenness)

        # Multi-run accuracy estimation (to verify consistency)
        rhos_h = []
        rhos_b = []
        for seed in range(n_accuracy_runs):
            r_gdf = networks.node_centrality_shortest_adaptive(
                net,
                nodes_gdf,
                distances=[dist],
                target_rho=target_rho,
                compute_closeness=True,
                compute_betweenness=True,
                random_seed=seed,
            )
            rho, _ = compute_accuracy_metrics(true_harmonic, np.array(r_gdf[f"cc_harmonic_{dist}"]))
            rhos_h.append(rho)
            rho, _ = compute_accuracy_metrics(true_betweenness, np.array(r_gdf[f"cc_betweenness_{dist}"]))
            rhos_b.append(rho)

        speedup = full_time / adaptive_time

        result = {
            "distance_m": dist,
            "distance_km": dist / 1000,
            "mean_reachability": mean_reach,
            "full_time_s": full_time,
            "adaptive_time_s": adaptive_time,
            "speedup": speedup,
            "rho_harmonic": np.mean(rhos_h),
            "rho_harmonic_std": np.std(rhos_h),
            "rho_betweenness": np.mean(rhos_b),
            "rho_betweenness_std": np.std(rhos_b),
            "target_rho": target_rho,
        }
        results.append(result)

        print(f"      Speedup: {speedup:.1f}x")
        print(f"      Accuracy: ρ_harmonic={np.mean(rhos_h):.3f}, ρ_betweenness={np.mean(rhos_b):.3f}")

    return results


# %% Report generation function


def generate_benchmark_report(results: list[dict], n_nodes: int):
    """Generate benchmark report and figures."""
    df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("REGIONAL BENCHMARK RESULTS")
    print("=" * 70)
    print(f"\nNetwork: {n_nodes:,} nodes")
    print(f"Target accuracy: ρ ≥ {results[0]['target_rho']}")
    print()

    # Summary table
    print("-" * 90)
    print(
        f"{'Distance':<12} {'Reachability':<14} {'Full (s)':<12} {'Adaptive (s)':<14} {'Speedup':<10} {'ρ (harm)':<10} {'ρ (betw)':<10}"
    )
    print("-" * 90)

    for r in results:
        print(
            f"{r['distance_km']:.0f} km"
            f"        {r['mean_reachability']:>8,.0f}"
            f"        {r['full_time_s']:>8.1f}"
            f"        {r['adaptive_time_s']:>8.1f}"
            f"        {r['speedup']:>6.1f}x"
            f"        {r['rho_harmonic']:>6.3f}"
            f"        {r['rho_betweenness']:>6.3f}"
        )

    print("-" * 90)

    # Calculate totals for multi-scale analysis
    total_full = sum(r["full_time_s"] for r in results)
    total_adaptive = sum(r["adaptive_time_s"] for r in results)
    total_speedup = total_full / total_adaptive

    print(f"\nMulti-scale total ({len(results)} distances):")
    print(f"  Full computation:     {total_full:.1f}s ({total_full / 60:.1f} min)")
    print(f"  Adaptive sampling:    {total_adaptive:.1f}s ({total_adaptive / 60:.1f} min)")
    print(f"  Overall speedup:      {total_speedup:.1f}x")

    # Generate figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Timing comparison
    ax = axes[0]
    x = np.arange(len(df))
    width = 0.35

    bars1 = ax.bar(x - width / 2, df["full_time_s"], width, label="Full computation", color="#2ecc71")
    bars2 = ax.bar(x + width / 2, df["adaptive_time_s"], width, label="Adaptive sampling", color="#3498db")

    ax.set_xlabel("Distance Threshold", fontsize=11)
    ax.set_ylabel("Computation Time (seconds)", fontsize=11)
    ax.set_title("Computation Time: Full vs Adaptive", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(d / 1000)} km" for d in df["distance_m"]])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Add speedup labels
    for i, (b1, b2) in enumerate(zip(bars1, bars2, strict=False)):
        speedup = df.iloc[i]["speedup"]
        ax.annotate(
            f"{speedup:.1f}x",
            xy=(b2.get_x() + b2.get_width() / 2, b2.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="#e74c3c",
            fontweight="bold",
        )

    # Right: Accuracy achieved
    ax = axes[1]
    ax.errorbar(
        df["distance_km"],
        df["rho_harmonic"],
        yerr=df["rho_harmonic_std"],
        fmt="o-",
        label="Harmonic closeness",
        capsize=4,
        color="#9b59b6",
    )
    ax.errorbar(
        df["distance_km"],
        df["rho_betweenness"],
        yerr=df["rho_betweenness_std"],
        fmt="s-",
        label="Betweenness",
        capsize=4,
        color="#e67e22",
    )
    ax.axhline(
        y=results[0]["target_rho"],
        color="green",
        linestyle="--",
        alpha=0.7,
        label=f"Target ρ = {results[0]['target_rho']}",
    )

    ax.set_xlabel("Distance Threshold (km)", fontsize=11)
    ax.set_ylabel("Spearman ρ (ranking accuracy)", fontsize=11)
    ax.set_title("Accuracy Achieved by Adaptive Sampling", fontsize=12, fontweight="bold")
    ax.set_ylim(0.9, 1.01)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Regional-Scale Benchmark: Madrid Network ({n_nodes:,} nodes)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "regional_benchmark.pdf", dpi=300, bbox_inches="tight")
    print(f"\nSaved: {FIGURES_DIR / 'regional_benchmark.pdf'}")

    # Save data
    df.to_csv(TABLES_DIR / "regional_benchmark.csv", index=False)
    print(f"Saved: {TABLES_DIR / 'regional_benchmark.csv'}")

    # Generate LaTeX table
    latex = (
        r"""% Auto-generated table: Regional-Scale Benchmark Results
% DO NOT EDIT MANUALLY - regenerate with: python benchmark_regional.py

\begin{table}[htbp]
\centering
\caption{Regional-scale benchmark on Madrid metropolitan network ("""
        + f"{n_nodes:,}"
        + r""" nodes). Full computation time compared to adaptive sampling with $\rhosp^* = """
        + f"{results[0]['target_rho']}"
        + r"""$.}
\label{tab:regional_benchmark}
\begin{tabular}{rrrrrr}
\toprule
Distance & Reachability & Full (s) & Adaptive (s) & Speedup & Observed $\rho$ \\
\midrule
"""
    )
    for r in results:
        rho_min = min(r["rho_harmonic"], r["rho_betweenness"])
        latex += f"{int(r['distance_km'])} km & {r['mean_reachability']:,.0f} & {r['full_time_s']:.1f} & {r['adaptive_time_s']:.1f} & {r['speedup']:.1f}$\\times$ & {rho_min:.3f} \\\\\n"

    latex += (
        r"""\midrule
\textbf{Total} & --- & \textbf{"""
        + f"{total_full:.1f}"
        + r"""} & \textbf{"""
        + f"{total_adaptive:.1f}"
        + r"""} & \textbf{"""
        + f"{total_speedup:.1f}"
        + r"""$\times$} & --- \\
\bottomrule
\end{tabular}
\end{table}
"""
    )

    with open(TABLES_DIR / "regional_benchmark.tex", "w") as f:
        f.write(latex)
    print(f"Saved: {TABLES_DIR / 'regional_benchmark.tex'}")


def generate_spatial_report(spatial_results: list[dict], n_nodes: int):
    """Generate spatial analysis report and figures."""
    print("\n" + "=" * 70)
    print("SPATIAL ANALYSIS RESULTS")
    print("=" * 70)
    print("\nKey question: Do accuracy findings hold spatially across all areas?")
    print("(Reachability varies spatially - low-reach areas have lower effective N)")

    # Collect all quartile data
    all_quartiles = []
    for sr in spatial_results:
        for qr in sr["quartile_results"]:
            all_quartiles.append(
                {
                    "distance": sr["distance"],
                    "sample_prob": sr["sample_prob"],
                    **qr,
                }
            )

    q_df = pd.DataFrame(all_quartiles)

    # Summary by quartile
    print("\n--- Accuracy by Reachability Quartile ---")
    print("(Q1 = lowest 25% reachability, Q4 = highest 25%)")
    print()

    for dist in q_df["distance"].unique():
        print(f"Distance {dist / 1000:.0f}km:")
        dist_df = q_df[q_df["distance"] == dist]
        summary = dist_df.groupby("quartile").agg(
            {
                "mean_reachability": "mean",
                "effective_n": "mean",
                "rho_harmonic": "mean",
                "rho_betweenness": "mean",
            }
        )
        print(summary.to_string())
        print()

    # Check accuracy gap
    q1_h = q_df[q_df["quartile"] == "Q1 (low)"]["rho_harmonic"].mean()
    q4_h = q_df[q_df["quartile"] == "Q4 (high)"]["rho_harmonic"].mean()
    q1_b = q_df[q_df["quartile"] == "Q1 (low)"]["rho_betweenness"].mean()
    q4_b = q_df[q_df["quartile"] == "Q4 (high)"]["rho_betweenness"].mean()

    print("--- Low vs High Reachability Accuracy Gap ---")
    print(f"Harmonic:    Q1={q1_h:.3f}, Q4={q4_h:.3f}, gap={q4_h - q1_h:+.3f}")
    print(f"Betweenness: Q1={q1_b:.3f}, Q4={q4_b:.3f}, gap={q4_b - q1_b:+.3f}")

    # Moran's I summary
    print("\n--- Spatial Autocorrelation (Moran's I) ---")
    print("(Values near 0 = random; positive = clustered residuals)")
    for sr in spatial_results:
        print(
            f"  {sr['distance'] / 1000:.0f}km, p={sr['sample_prob']:.0%}: "
            f"H={sr['morans_i_harmonic']:.3f}, B={sr['morans_i_betweenness']:.3f}"
        )

    # Generate figure: Accuracy by reachability quartile
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    quartile_order = ["Q1 (low)", "Q2", "Q3", "Q4 (high)"]
    colors = {5000: "#1f77b4", 10000: "#d62728", 20000: "#2ca02c"}

    for col, metric in enumerate(["harmonic", "betweenness"]):
        ax = axes[col]

        for dist in q_df["distance"].unique():
            dist_df = q_df[q_df["distance"] == dist]
            agg = dist_df.groupby("quartile")[f"rho_{metric}"].mean()
            values = [agg.get(q, np.nan) for q in quartile_order]
            ax.plot(
                range(4),
                values,
                "o-",
                label=f"{dist / 1000:.0f}km",
                color=colors.get(dist, "gray"),
                linewidth=2,
                markersize=8,
            )

        ax.set_xticks(range(4))
        ax.set_xticklabels(["Q1\n(low reach)", "Q2", "Q3", "Q4\n(high reach)"])
        ax.set_xlabel("Reachability Quartile")
        ax.set_ylabel("Spearman ρ")
        ax.set_title(f"{metric.title()}")
        ax.set_ylim(0.85, 1.01)
        ax.axhline(y=0.95, color="green", linestyle="--", alpha=0.7, label="Target ρ=0.95")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Spatial Validation: Accuracy by Local Reachability\n"
        f"Madrid Network ({n_nodes:,} nodes) - Does accuracy hold in low-reach areas?",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "spatial_reachability_accuracy.pdf", dpi=300, bbox_inches="tight")
    print(f"\nSaved: {FIGURES_DIR / 'spatial_reachability_accuracy.pdf'}")

    # Generate figure: Spatial residual map (for one distance)
    sr = spatial_results[0]  # Use first distance
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for col, metric in enumerate(["harmonic", "betweenness"]):
        ax = axes[col]
        residuals = sr[f"{metric}_residual"]

        vmax = np.percentile(np.abs(residuals), 95)
        clipped = np.clip(residuals, -vmax, vmax)

        scatter = ax.scatter(
            sr["coords"][:, 0],
            sr["coords"][:, 1],
            c=clipped,
            cmap="RdBu_r",
            s=1,
            alpha=0.6,
            vmin=-vmax,
            vmax=vmax,
        )
        plt.colorbar(scatter, ax=ax, label="Relative residual")

        morans_i = sr[f"morans_i_{metric}"]
        ax.annotate(
            f"Moran's I = {morans_i:.3f}",
            xy=(0.02, 0.98),
            xycoords="axes fraction",
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(f"{metric.title()} Residuals")
        ax.set_aspect("equal")

    plt.suptitle(
        f"Spatial Distribution of Sampling Residuals\n"
        f"(Distance={sr['distance'] / 1000:.0f}km, p={sr['sample_prob']:.0%})",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "spatial_residuals_map.pdf", dpi=150, bbox_inches="tight")
    print(f"Saved: {FIGURES_DIR / 'spatial_residuals_map.pdf'}")

    # Save data
    q_df.to_csv(TABLES_DIR / "spatial_reachability_accuracy.csv", index=False)
    print(f"Saved: {TABLES_DIR / 'spatial_reachability_accuracy.csv'}")

    # Generate LaTeX table
    latex = r"""% Auto-generated table: Spatial Analysis - Accuracy by Reachability
% DO NOT EDIT MANUALLY - regenerate with: python benchmark_regional.py

\begin{table}[htbp]
\centering
\caption{Accuracy by local reachability quartile. Q1 = lowest 25\% reachability (sparse areas),
Q4 = highest 25\%. Lower reachability means lower effective sample size.}
\label{tab:spatial_reachability}
\begin{tabular}{lrrrrrr}
\toprule
Distance & Quartile & Mean Reach & Eff. N & $\rho_H$ & $\rho_B$ \\
\midrule
"""
    for dist in sorted(q_df["distance"].unique()):
        dist_df = q_df[q_df["distance"] == dist]
        for q in quartile_order:
            q_data = dist_df[dist_df["quartile"] == q]
            if len(q_data) == 0:
                continue
            latex += (
                f"{dist / 1000:.0f} km & {q} & "
                f"{q_data['mean_reachability'].mean():,.0f} & "
                f"{q_data['effective_n'].mean():.0f} & "
                f"{q_data['rho_harmonic'].mean():.3f} & "
                f"{q_data['rho_betweenness'].mean():.3f} \\\\\n"
            )

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    with open(TABLES_DIR / "spatial_reachability.tex", "w") as f:
        f.write(latex)
    print(f"Saved: {TABLES_DIR / 'spatial_reachability.tex'}")


# %% Run benchmark
print("=" * 70)
print("REGIONAL-SCALE NETWORK BENCHMARK")
print("=" * 70)

DISTANCES = [5000, 10000, 20000]
TARGET_RHO = 0.95

print(f"Distances: {[f'{d / 1000:.0f}km' for d in DISTANCES]}")
print(f"Target ρ: {TARGET_RHO}")
print(f"Output: {PAPER_DIR}")

n_nodes = len(nodes_gdf)

results = run_benchmark(
    network_structure,
    nodes_gdf,
    distances=DISTANCES,
    target_rho=TARGET_RHO,
)

# %% Generate report
generate_benchmark_report(results, n_nodes)

# %% Run spatial analysis
print("\n" + "=" * 70)
print("SPATIAL ANALYSIS")
print("=" * 70)
print("Testing whether accuracy holds spatially across all areas...")
print("(Reachability varies spatially - low-reach areas may have lower accuracy)")

SPATIAL_SAMPLE_PROB = 0.3  # Use fixed p for spatial analysis
spatial_results = []

for dist in DISTANCES:
    sr = run_spatial_analysis(
        network_structure,
        nodes_gdf,
        dist=dist,
        sample_prob=SPATIAL_SAMPLE_PROB,
        n_runs=3,
    )
    spatial_results.append(sr)

generate_spatial_report(spatial_results, n_nodes)

print("\n" + "=" * 70)
print("BENCHMARK COMPLETE")
print("=" * 70)

# %%
