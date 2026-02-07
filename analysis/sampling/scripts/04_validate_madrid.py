#!/usr/bin/env python
"""
04_validate_madrid.py - External validation on Madrid regional network.

Generates validation data (if not cached) and produces figures.
Requires sampling_model.json from 01_fit_rank_model.py.

Usage:
    python 04_validate_madrid.py           # Run (skips cache if exists)
    python 04_validate_madrid.py --force   # Force regeneration of validation data

Outputs:
    - output/madrid_validation_{CACHE_VERSION}.csv
    - paper/figures/fig6_madrid_validation.pdf
"""

import argparse
import json
import math
import pickle
import time
from pathlib import Path

import geopandas as gpd
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from cityseer.tools import graphs, io
from utilities import (
    CACHE_DIR,
    CACHE_VERSION,
    FIGURES_DIR,
    MADRID_GPKG_URL,
    OUTPUT_DIR,
    apply_live_buffer_nx,
    compute_accuracy_metrics,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent

# Validation parameters
LIVE_INWARD_BUFFER = 20000  # 20km buffer
MADRID_DISTANCES = [1000, 2000, 5000, 10000, 20000]
N_RUNS = 3

# Matplotlib style
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


# =============================================================================
# EW BOUND FUNCTIONS
# =============================================================================

DELTA = 0.1  # Failure probability (90% confidence)


def ew_predicted_epsilon(n_eff: float, reach: float, delta: float = DELTA) -> float:
    """EW-predicted maximum normalised epsilon (Hoeffding form)."""
    if n_eff <= 0 or reach <= 0:
        return float("inf")
    return math.sqrt(math.log(2 * reach / delta) / (2 * n_eff))


def normalise_error(max_abs_error: float, reach: float, metric: str) -> float:
    """Normalise raw absolute error by theoretical maximum."""
    if reach <= 1:
        return float("inf")
    if metric == "betweenness":
        return max_abs_error / (reach * (reach - 1))
    else:  # harmonic closeness
        return max_abs_error / reach


def implied_epsilon_from_rank_model(reach: float, k: float, min_eff_n: float, delta: float = DELTA) -> float:
    """Additive error the rank model implicitly guarantees at this reach."""
    n_eff = max(k * math.sqrt(reach), min_eff_n)
    if n_eff <= 0 or reach <= 0:
        return float("inf")
    return math.sqrt(math.log(2 * reach / delta) / (2 * n_eff))


# =============================================================================
# DATA GENERATION
# =============================================================================


def load_model() -> tuple[float, int]:
    """Load the fitted model parameters."""
    model_path = OUTPUT_DIR / "sampling_model.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run 01_fit_rank_model.py first.")

    with open(model_path) as f:
        model = json.load(f)

    k = model["model"]["k"]
    min_eff_n = model["model"]["min_eff_n"]
    return k, min_eff_n


def compute_model_p(reach: float, k: float, min_eff_n: int) -> float:
    """Compute the model's recommended sampling probability."""
    eff_n = max(k * math.sqrt(reach), min_eff_n)
    return min(1.0, eff_n / reach)


def generate_validation_data(k: float, min_eff_n: int, force: bool = False) -> pd.DataFrame:
    """Generate Madrid validation data, or load from cache."""
    validation_csv = OUTPUT_DIR / f"madrid_validation_{CACHE_VERSION}.csv"

    if validation_csv.exists() and not force:
        print(f"Loading cached validation data from {validation_csv}")
        return pd.read_csv(validation_csv)

    print("\n" + "=" * 70)
    print("GENERATING MADRID VALIDATION DATA")
    print("=" * 70)

    # Load or build Madrid graph
    madrid_cache = CACHE_DIR / "madrid_graph.pkl"
    if madrid_cache.exists() and not force:
        print(f"Loading cached Madrid graph from {madrid_cache}")
        with open(madrid_cache, "rb") as f:
            G = pickle.load(f)
    else:
        print(f"Downloading Madrid regional network from: {MADRID_GPKG_URL}")
        print("  (This may take a minute...)")

        edges_gdf = gpd.read_file(MADRID_GPKG_URL)
        print(f"  Downloaded: {len(edges_gdf)} edges, CRS: {edges_gdf.crs}")

        edges_gdf_singles = edges_gdf.explode(index_parts=False)

        print("  Building graph...")
        G = io.nx_from_generic_geopandas(edges_gdf_singles)
        G = graphs.nx_remove_filler_nodes(G)
        G = graphs.nx_remove_dangling_nodes(G)

        print(f"  Caching to {madrid_cache}")
        with open(madrid_cache, "wb") as f:
            pickle.dump(G, f)

    print(f"Madrid graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Apply live buffer
    print(f"Applying {LIVE_INWARD_BUFFER / 1000:.0f}km inward buffer...")
    G = apply_live_buffer_nx(G, LIVE_INWARD_BUFFER)

    # Convert to cityseer
    print("Converting to cityseer format...")
    nodes_gdf, edges_gdf_out, net = io.network_structure_from_nx(G)

    results = []

    for dist in MADRID_DISTANCES:
        print(f"\n{'=' * 50}")
        print(f"Distance: {dist}m")
        print(f"{'=' * 50}")

        # Check for cached ground truth
        gt_cache = CACHE_DIR / f"madrid_ground_truth_{dist}m.pkl"
        if gt_cache.exists() and not force:
            print(f"  Loading cached ground truth from {gt_cache}")
            with open(gt_cache, "rb") as f:
                gt_data = pickle.load(f)
            true_harmonic = gt_data["harmonic"]
            true_betweenness = gt_data["betweenness"]
            mean_reach = gt_data["mean_reach"]
            baseline_time = gt_data.get("baseline_time", None)
        else:
            print("  Computing ground truth (this may take a while)...")
            t0 = time.time()
            true_result = net.local_node_centrality_shortest(
                distances=[dist],
                compute_closeness=True,
                compute_betweenness=True,
                pbar_disabled=False,
            )
            baseline_time = time.time() - t0

            true_harmonic = np.array(true_result.node_harmonic[dist])
            true_betweenness = np.array(true_result.node_betweenness[dist])
            reach = np.array(true_result.node_density[dist])
            mean_reach = float(np.mean(reach))

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

        print(f"  Mean reach: {mean_reach:.0f}")
        if baseline_time is not None:
            print(f"  Baseline time: {baseline_time:.1f}s")

        # Compute model-recommended p
        model_p = compute_model_p(mean_reach, k, min_eff_n)
        effective_n = mean_reach * model_p
        print(f"  Model p: {model_p:.3f} (eff_n={effective_n:.0f})")

        # Run sampled computations
        spearmans_h, spearmans_b = [], []
        maes_h, maes_b = [], []
        precs_h, precs_b = [], []
        scales_h, scales_b = [], []
        sampled_times = []

        print(f"  Running {N_RUNS} sampled runs: ", end="", flush=True)

        for seed in range(N_RUNS):
            t0 = time.time()
            r = net.local_node_centrality_shortest(
                distances=[dist],
                compute_closeness=True,
                compute_betweenness=True,
                sample_probability=model_p,
                random_seed=seed,
                pbar_disabled=True,
            )
            sampled_times.append(time.time() - t0)

            est_harmonic = np.array(r.node_harmonic[dist])
            est_betweenness = np.array(r.node_betweenness[dist])

            sp_h, prec_h, scale_h, _, mae_h = compute_accuracy_metrics(true_harmonic, est_harmonic)
            sp_b, prec_b, scale_b, _, mae_b = compute_accuracy_metrics(true_betweenness, est_betweenness)

            if not np.isnan(sp_h):
                spearmans_h.append(sp_h)
                maes_h.append(mae_h)
                precs_h.append(prec_h)
                scales_h.append(scale_h)
            if not np.isnan(sp_b):
                spearmans_b.append(sp_b)
                maes_b.append(mae_b)
                precs_b.append(prec_b)
                scales_b.append(scale_b)

            print(".", end="", flush=True)

        mean_sampled_time = np.mean(sampled_times)
        speedup = baseline_time / mean_sampled_time if baseline_time and mean_sampled_time > 0 else float("nan")

        print(
            f" rho_h={np.mean(spearmans_h):.3f}, rho_b={np.mean(spearmans_b):.3f}, "
            f"speedup={speedup:.1f}x"
        )

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
                "max_abs_error_h": np.mean(maes_h) if maes_h else float("nan"),
                "max_abs_error_b": np.mean(maes_b) if maes_b else float("nan"),
                "top_k_precision_h": np.mean(precs_h) if precs_h else float("nan"),
                "top_k_precision_b": np.mean(precs_b) if precs_b else float("nan"),
                "scale_ratio_h": np.mean(scales_h) if scales_h else float("nan"),
                "scale_ratio_b": np.mean(scales_b) if scales_b else float("nan"),
                "baseline_time": baseline_time if baseline_time else float("nan"),
                "sampled_time": mean_sampled_time,
                "speedup": speedup,
            }
        )

    df = pd.DataFrame(results)
    df.to_csv(validation_csv, index=False)
    print(f"\nSaved validation results: {validation_csv}")
    return df


# =============================================================================
# FIGURE GENERATION
# =============================================================================


def generate_validation_figure(df: pd.DataFrame):
    """
    Generate validation figure showing accuracy and speedup across distances.

    Two panels:
    - Left: Accuracy (rho) vs distance for closeness and betweenness
    - Right: Speedup vs distance
    """
    print("\nGenerating validation figure...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    distances_km = df["distance"] / 1000

    # -----------------------------------------------------------------
    # Panel A: Accuracy
    # -----------------------------------------------------------------
    ax1 = axes[0]

    # Closeness
    ax1.errorbar(
        distances_km,
        df["rho_closeness"],
        yerr=df["rho_closeness_std"],
        fmt="o-",
        color="#0072B2",
        linewidth=2,
        markersize=8,
        capsize=4,
        label="Closeness",
    )

    # Betweenness
    ax1.errorbar(
        distances_km,
        df["rho_betweenness"],
        yerr=df["rho_betweenness_std"],
        fmt="s-",
        color="#D55E00",
        linewidth=2,
        markersize=8,
        capsize=4,
        label="Betweenness",
    )

    # Target line
    ax1.axhline(0.95, color="green", linestyle="--", linewidth=1.5, alpha=0.7, label="Target (ρ=0.95)")

    ax1.set_xlabel("Distance (km)")
    ax1.set_ylabel("Spearman ρ (ranking accuracy)")
    ax1.set_title("(a) Accuracy vs Distance")
    ax1.legend(loc="lower right")
    ax1.set_ylim(0.90, 1.005)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale("log")
    ax1.set_xticks([1, 2, 5, 10, 20])
    ax1.set_xticklabels(["1", "2", "5", "10", "20"])

    # -----------------------------------------------------------------
    # Panel B: Speedup
    # -----------------------------------------------------------------
    ax2 = axes[1]

    ax2.bar(
        range(len(df)),
        df["speedup"],
        color="#009E73",
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )

    # Add speedup labels on bars
    for i, (spd, _dist) in enumerate(zip(df["speedup"], df["distance"], strict=True)):
        ax2.text(i, spd + 0.1, f"{spd:.1f}x", ha="center", va="bottom", fontsize=9)

    ax2.set_xlabel("Distance (km)")
    ax2.set_ylabel("Speedup (baseline / sampled)")
    ax2.set_title("(b) Computational Speedup")
    ax2.set_xticks(range(len(df)))
    ax2.set_xticklabels([f"{d / 1000:.0f}" if d >= 1000 else f"{d / 1000:.1f}" for d in df["distance"]])
    ax2.grid(True, alpha=0.3, axis="y")

    # Reference line at 1x
    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    fig.suptitle(
        "Madrid Regional Network Validation",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()

    output_path = FIGURES_DIR / "fig6_madrid_validation.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()

    return output_path


# =============================================================================
# THEORETICAL BOUNDS COMPARISON
# =============================================================================


def get_n_nodes(force: bool = False) -> int | None:
    """Get total node count from cached Madrid graph."""
    madrid_cache = CACHE_DIR / "madrid_graph.pkl"
    if not madrid_cache.exists():
        return None
    with open(madrid_cache, "rb") as f:
        G = pickle.load(f)
    # Apply same buffer as validation to get live node count
    G = apply_live_buffer_nx(G, LIVE_INWARD_BUFFER)
    return G.number_of_nodes()


def compute_theoretical_bounds(df: pd.DataFrame, n_nodes: int):
    """Compare empirical sample counts against Riondato, Bader, and Eppstein-Wang bounds."""
    print("\nComputing theoretical bounds comparison...")

    delta = 0.1  # Failure probability (90% confidence)
    rows = []

    for _, row in df.iterrows():
        reach = row["mean_reach"]
        our_eff_n = row["effective_n"]

        for metric, mae_col in [("harmonic", "max_abs_error_h"), ("betweenness", "max_abs_error_b")]:
            if mae_col not in row or np.isnan(row[mae_col]):
                continue

            raw_eps = row[mae_col]
            if metric == "betweenness":
                eps_normalised = raw_eps / (reach * (reach - 1)) if reach > 1 else float("inf")
            else:
                eps_normalised = raw_eps / reach if reach > 0 else float("inf")

            if eps_normalised <= 0 or not np.isfinite(eps_normalised):
                continue

            # Riondato & Kornaropoulos (2016): VC-dimension bound
            vd = max(3, int(np.sqrt(reach)))
            vc_dim = int(np.floor(np.log2(max(1, vd - 2)))) + 1
            riondato_samples = (1 / (2 * eps_normalised**2)) * (vc_dim + np.log(1 / delta))

            # Bader et al. (2007): per-vertex bound
            bader_samples = 1 / eps_normalised**2

            # Eppstein & Wang (2004): source-sampling bound
            # Global: O(log(n) / eps^2) using total node count
            eppstein_samples = np.log(n_nodes) / eps_normalised**2
            # Localised adaptation: O(log(r) / eps^2) using reach
            eppstein_local_samples = np.log(reach) / eps_normalised**2

            rows.append(
                {
                    "distance": row["distance"],
                    "metric": metric,
                    "reach": reach,
                    "our_eff_n": our_eff_n,
                    "raw_eps": raw_eps,
                    "eps_normalised": eps_normalised,
                    "vd_estimate": vd,
                    "vc_dim": vc_dim,
                    "riondato_samples": riondato_samples,
                    "bader_samples": bader_samples,
                    "eppstein_samples": eppstein_samples,
                    "eppstein_local_samples": eppstein_local_samples,
                    "ratio_riondato": riondato_samples / our_eff_n if our_eff_n > 0 else float("inf"),
                    "ratio_bader": bader_samples / our_eff_n if our_eff_n > 0 else float("inf"),
                    "ratio_eppstein": eppstein_samples / our_eff_n if our_eff_n > 0 else float("inf"),
                    "ratio_eppstein_local": eppstein_local_samples / our_eff_n if our_eff_n > 0 else float("inf"),
                }
            )

    if not rows:
        return None

    bounds_df = pd.DataFrame(rows)

    csv_path = OUTPUT_DIR / "madrid_theoretical_bounds_comparison.csv"
    bounds_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    print(
        f"\n  {'Distance':>10} | {'Metric':>12} | {'Our eff_n':>10} | {'EW global':>12} | {'EW local':>12} | "
        f"{'Riondato':>12} | {'Bader':>12} | {'R(EWg)':>8} | {'R(EWl)':>8}"
    )
    print("  " + "-" * 120)
    for _, r in bounds_df.iterrows():
        print(
            f"  {r['distance'] // 1000}km       | "
            f"{r['metric']:>12} | "
            f"{r['our_eff_n']:>10,.0f} | "
            f"{r['eppstein_samples']:>12,.0f} | "
            f"{r['eppstein_local_samples']:>12,.0f} | "
            f"{r['riondato_samples']:>12,.0f} | "
            f"{r['bader_samples']:>12,.0f} | "
            f"{r['ratio_eppstein']:>8.1f}x | "
            f"{r['ratio_eppstein_local']:>8.1f}x"
        )

    return bounds_df


# =============================================================================
# EW BOUND ANALYSIS
# =============================================================================


def compute_ew_analysis(df: pd.DataFrame, k: float, min_eff_n: int):
    """Evaluate the localised EW bound on Madrid validation configurations."""
    print("\n" + "=" * 70)
    print("LOCALISED EW BOUND ANALYSIS")
    print("=" * 70)

    rows = []
    for _, row in df.iterrows():
        reach = row["mean_reach"]
        sample_prob = row["sample_prob"]
        n_eff = reach * sample_prob

        for metric, mae_col in [("harmonic", "max_abs_error_h"), ("betweenness", "max_abs_error_b")]:
            if mae_col not in row or np.isnan(row[mae_col]):
                continue

            eps_obs = normalise_error(row[mae_col], reach, metric)
            eps_pred = ew_predicted_epsilon(n_eff, reach)
            bound_holds = eps_obs <= eps_pred

            rows.append(
                {
                    "distance": row["distance"],
                    "metric": metric,
                    "reach": reach,
                    "sample_prob": sample_prob,
                    "n_eff": n_eff,
                    "eps_observed": eps_obs,
                    "eps_predicted": eps_pred,
                    "bound_holds": bound_holds,
                }
            )

    if not rows:
        print("  No configurations to analyse")
        return None

    ew_df = pd.DataFrame(rows)

    # Overall success rate
    total = len(ew_df)
    holds = int(ew_df["bound_holds"].sum())
    print(f"\n  Overall: {holds}/{total} ({100 * holds / total:.1f}%) — expected >= {100 * (1 - DELTA):.0f}%")

    # By metric
    print(f"\n  {'Metric':<15} {'Holds':>8} {'Total':>8} {'Rate':>8}")
    print("  " + "-" * 45)
    for metric in sorted(ew_df["metric"].unique()):
        subset = ew_df[ew_df["metric"] == metric]
        h = int(subset["bound_holds"].sum())
        t = len(subset)
        print(f"  {metric:<15} {h:>8} {t:>8} {100 * h / t:>7.1f}%")

    # By distance
    print(f"\n  {'Distance':<15} {'Holds':>8} {'Total':>8} {'Rate':>8}")
    print("  " + "-" * 45)
    for dist in sorted(ew_df["distance"].unique()):
        subset = ew_df[ew_df["distance"] == dist]
        h = int(subset["bound_holds"].sum())
        t = len(subset)
        print(f"  {dist:<15} {h:>8} {t:>8} {100 * h / t:>7.1f}%")

    # Implied epsilon from rank model
    print("\n  Implied epsilon from rank model:")
    print(f"  {'Distance':>10} {'Reach':>10} {'Rank n_eff':>12} {'Implied eps':>12}")
    print("  " + "-" * 50)
    for dist in sorted(ew_df["distance"].unique()):
        reach = ew_df[ew_df["distance"] == dist]["reach"].iloc[0]
        impl_eps = implied_epsilon_from_rank_model(reach, k, min_eff_n)
        rank_eff_n = max(k * math.sqrt(reach), min_eff_n)
        print(f"  {dist // 1000}km       {reach:>10,.0f} {rank_eff_n:>12,.0f} {impl_eps:>12.4f}")

    # Save
    csv_path = OUTPUT_DIR / "madrid_ew_analysis.csv"
    ew_df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    return ew_df


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Validate sampling model on Madrid network")
    parser.add_argument("--force", action="store_true", help="Force regeneration of validation data")
    args = parser.parse_args()

    print("=" * 70)
    print("04_validate_madrid.py - External validation on Madrid network")
    print("=" * 70)

    # Load model
    k, min_eff_n = load_model()
    print(f"\nModel: eff_n = max({k} × sqrt(reach), {min_eff_n})")

    # Generate or load validation data
    df = generate_validation_data(k, min_eff_n, force=args.force)
    print(f"\nValidation data: {len(df)} rows")

    # Add target columns
    df["meets_target_close"] = df["rho_closeness"] >= 0.95
    df["meets_target_between"] = df["rho_betweenness"] >= 0.95

    # Generate figure
    fig_path = generate_validation_figure(df)

    # Theoretical bounds comparison
    n_nodes = get_n_nodes(force=args.force)
    bounds_df = None
    if n_nodes is not None:
        print(f"\nMadrid network: {n_nodes} live nodes")
        bounds_df = compute_theoretical_bounds(df, n_nodes)
    else:
        print("\n  Skipping theoretical bounds comparison (graph cache not found)")

    # EW bound analysis
    ew_df = compute_ew_analysis(df, k, min_eff_n)

    # Print summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    print(
        f"\n{'Distance':>10} | {'Reach':>10} | {'Model p':>10}"
        f" | {'ρ close':>10} | {'ρ between':>10} | {'Speedup':>10} | {'Pass?':>6}"
    )
    print("-" * 85)

    all_pass = True
    for _, row in df.iterrows():
        passes = row["meets_target_close"] and row["meets_target_between"]
        status = "PASS" if passes else "FAIL"
        if not passes:
            all_pass = False

        dist_str = f"{row['distance'] // 1000}km"
        print(
            f"{dist_str:>10} | "
            f"{row['mean_reach']:>10,.0f} | "
            f"{row['sample_prob']:>9.1%} | "
            f"{row['rho_closeness']:>10.4f} | "
            f"{row['rho_betweenness']:>10.4f} | "
            f"{row['speedup']:>9.1f}x | "
            f"{status:>6}"
        )

    print("-" * 85)

    if all_pass:
        print("\nALL DISTANCES PASS: Model achieves ρ >= 0.95 for both metrics at all distances.")
    else:
        print("\nWARNING: Some distances do not meet the ρ >= 0.95 target.")

    # Overall statistics
    print("\nOverall Statistics:")
    print(f"  Mean speedup: {df['speedup'].mean():.2f}x")
    print(f"  Max speedup:  {df['speedup'].max():.2f}x (at {df.loc[df['speedup'].idxmax(), 'distance'] / 1000:.0f}km)")
    print(f"  Mean ρ (closeness):   {df['rho_closeness'].mean():.4f}")
    print(f"  Mean ρ (betweenness): {df['rho_betweenness'].mean():.4f}")

    print("\n" + "=" * 70)
    print("OUTPUTS")
    print("=" * 70)
    print(f"  1. {OUTPUT_DIR / f'madrid_validation_{CACHE_VERSION}.csv'}")
    print(f"  2. {fig_path}")
    if bounds_df is not None:
        print(f"  3. {OUTPUT_DIR / 'madrid_theoretical_bounds_comparison.csv'}")
    if ew_df is not None:
        print(f"  4. {OUTPUT_DIR / 'madrid_ew_analysis.csv'}")

    return 0


if __name__ == "__main__":
    exit(main())
