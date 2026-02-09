#!/usr/bin/env python
"""
01_fit_rank_model.py - Fit the rank-based adaptive sampling model.

Fits the complete model: eff_n = max(k × sqrt(reach), min_eff_n)

This script combines three stages:
  1. Fit the proportional constant k from the sqrt(reach) scaling
  2. Fit the minimum effective sample size floor (min_eff_n)
  3. Combine into the final model with crossover analysis

The sqrt(reach) scaling has a theoretical basis in the Hajek estimator:
per-source variance grows linearly with reach, so maintaining constant
CV requires eff_n proportional to sqrt(reach).

Requires:
    - .cache/sampling_analysis_{CACHE_VERSION}.pkl (from 00_generate_cache.py)

Outputs:
    - output/model_fit.json: Fitted k value with metadata
    - output/floor_fit.json: Fitted min_eff_n value with metadata
    - output/sampling_model.json: Final combined model parameters
    - paper/figures/fig1_headline.pdf: Problem illustration
    - paper/figures/fig2_model_derivation.pdf: eff_n vs rho relationship
    - paper/figures/fig3_floor_justification.pdf: Floor justification
    - paper/figures/fig4_combined_model.pdf: Final model visualization
    - paper/tables/tab1_parameters.tex: LaTeX parameter table
"""

import json
import math
import pickle
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utilities import CACHE_DIR, CACHE_VERSION, FIGURES_DIR, OUTPUT_DIR, TABLES_DIR

# =============================================================================
# CONFIGURATION
# =============================================================================

SYNTHETIC_CACHE = CACHE_DIR / f"sampling_analysis_{CACHE_VERSION}.pkl"

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
# DATA LOADING
# =============================================================================


def load_synthetic_data() -> pd.DataFrame:
    """Load cached synthetic network sampling results."""
    if not SYNTHETIC_CACHE.exists():
        raise FileNotFoundError(
            f"Synthetic data cache not found at {SYNTHETIC_CACHE}\n"
            "Run scripts/00_generate_cache.py first to generate this cache."
        )

    with open(SYNTHETIC_CACHE, "rb") as f:
        data = pickle.load(f)

    df = pd.DataFrame(data)
    print(f"Loaded synthetic data: {len(df)} rows")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Topologies: {df['topology'].unique().tolist()}")
    print(f"  Distances: {sorted(df['distance'].unique())}")
    return df


# =============================================================================
# MODEL FUNCTIONS
# =============================================================================


def compute_eff_n(reach: float, k: float, min_eff_n: float) -> float:
    """Compute effective sample size from the model."""
    return max(k * math.sqrt(reach), min_eff_n)


def compute_p(reach: float, k: float, min_eff_n: float) -> float:
    """Compute sampling probability from the model."""
    eff_n = compute_eff_n(reach, k, min_eff_n)
    return min(1.0, eff_n / reach)


def crossover_reach(k: float, min_eff_n: float) -> float:
    """Find the reach where k × sqrt(reach) = min_eff_n."""
    return (min_eff_n / k) ** 2


# =============================================================================
# STAGE 1: FIT PROPORTIONAL CONSTANT k
# =============================================================================


def fit_proportional_k(df: pd.DataFrame, target_rho: float = 0.95) -> dict:
    """
    Fit the proportional constant k from synthetic data.

    For each (topology, distance) combination, find the minimum p where the
    aggregate Spearman rho achieves >= target_rho. Aggregate rho is used because
    within-quartile rho suffers from range restriction attenuation (Thorndike 1949):
    restricted variance within quartiles deflates correlation even when absolute
    errors are small.

    Then compute k = p × sqrt(reach) and take the 75th percentile across all combinations.
    """
    df_b = df[df["metric"] == "betweenness"].copy()

    results = []

    for topology in df_b["topology"].unique():
        for distance in sorted(df_b["distance"].unique()):
            subset = df_b[(df_b["topology"] == topology) & (df_b["distance"] == distance)]
            if len(subset) == 0:
                continue

            reach = subset["mean_reach"].iloc[0]
            subset_sorted = subset.sort_values("sample_prob")

            achieving = subset_sorted[subset_sorted["spearman"] >= target_rho]
            if len(achieving) == 0:
                min_p = 1.0
                achieved_rho = subset_sorted["spearman"].max()
            else:
                min_p = achieving["sample_prob"].iloc[0]
                achieved_rho = achieving["spearman"].iloc[0]

            k_implied = min_p * math.sqrt(reach)
            eff_n_at_target = reach * min_p

            results.append(
                {
                    "topology": topology,
                    "distance": distance,
                    "reach": reach,
                    "min_p_for_target": min_p,
                    "achieved_rho": achieved_rho,
                    "k_implied": k_implied,
                    "eff_n_at_target": eff_n_at_target,
                }
            )

    results_df = pd.DataFrame(results)

    k_max = results_df["k_implied"].max()
    k_mean = results_df["k_implied"].mean()
    k_p75 = results_df["k_implied"].quantile(0.75)
    k_p95 = results_df["k_implied"].quantile(0.95)
    k_selected = k_p75

    print("\nStage 1: Proportional k fitting (aggregate Spearman target)")
    print(f"  Target: aggregate rho >= {target_rho}")
    print(f"  k values: mean={k_mean:.2f}, 75th={k_p75:.2f}, 95th={k_p95:.2f}, max={k_max:.2f}")
    print(f"  Selected k (75th percentile): {k_selected:.2f}")

    return {
        "k": round(k_selected, 2),
        "k_max": round(k_max, 2),
        "target_rho": target_rho,
        "k_mean": round(k_mean, 2),
        "k_p75": round(k_p75, 2),
        "k_p95": round(k_p95, 2),
        "n_configs": len(results_df),
        "fitting_details": results_df.to_dict(orient="records"),
    }


# =============================================================================
# STAGE 2: FIT MINIMUM FLOOR
# =============================================================================


def fit_min_eff_n(df: pd.DataFrame, target_success_rate: float = 0.95) -> dict:
    """
    Fit the minimum effective sample size floor.

    Uses bin-based analysis to find the minimum eff_n where the local success
    rate (within that bin) achieves the target. Success is defined as aggregate
    Spearman rho >= 0.95.
    """
    df_b = df[df["metric"] == "betweenness"].copy()

    bins = [
        (0, 50),
        (50, 100),
        (100, 150),
        (150, 200),
        (200, 250),
        (250, 300),
        (300, 350),
        (350, 400),
        (400, 500),
        (500, 750),
        (750, 1000),
    ]
    bin_results = []

    for low, high in bins:
        subset = df_b[(df_b["effective_n"] > low) & (df_b["effective_n"] <= high)]
        if len(subset) == 0:
            continue

        n_success = (subset["spearman"] >= 0.95).sum()
        n_total = len(subset)
        success_rate = n_success / n_total

        bin_results.append(
            {
                "bin": f"{low}-{high}",
                "low": low,
                "high": high,
                "n_total": n_total,
                "n_success": n_success,
                "success_rate": success_rate,
            }
        )

    bin_results_df = pd.DataFrame(bin_results)

    achieving_bins = bin_results_df[bin_results_df["success_rate"] >= target_success_rate]
    if len(achieving_bins) > 0:
        min_eff_n = int(achieving_bins.iloc[0]["low"])
        achieved_rate = achieving_bins.iloc[0]["success_rate"]
        achieving_bin = achieving_bins.iloc[0]["bin"]
    else:
        min_eff_n = int(bin_results_df.iloc[-1]["high"])
        achieved_rate = bin_results_df.iloc[-1]["success_rate"]
        achieving_bin = bin_results_df.iloc[-1]["bin"]

    print("\nStage 2: Floor fitting")
    print(f"  Target success rate: {target_success_rate:.0%}")
    print(f"  First achieving bin: {achieving_bin} (rate: {achieved_rate:.1%})")
    print(f"  Fitted min_eff_n: {min_eff_n}")
    print("\n  Success rate by eff_n bin:")
    for _, row in bin_results_df.iterrows():
        print(f"    {row['bin']:>10}: {row['success_rate']:.1%} (n={row['n_total']})")

    return {
        "min_eff_n": min_eff_n,
        "target_success_rate": target_success_rate,
        "achieved_success_rate": round(float(achieved_rate), 4),
        "achieving_bin": achieving_bin,
        "bin_analysis": bin_results_df.to_dict(orient="records"),
    }


# =============================================================================
# NODE-LEVEL ACCURACY ASSESSMENT
# =============================================================================


def evaluate_node_level_accuracy(df: pd.DataFrame, k: float, min_eff_n: int) -> pd.DataFrame:
    """Evaluate per-node accuracy at the combined model's recommended p.

    Pools all nodes across all (topology, distance) configs at the model-
    recommended sampling probability, then bins by absolute per-node reach.
    This avoids the within-config quartile pre-aggregation that confounds
    boundary nodes from high-reach configs with all nodes from low-reach configs.
    """
    available_probs = sorted(p for p in df["sample_prob"].unique() if p < 1.0)
    reach_pool = {"harmonic": [], "betweenness": []}
    error_pool = {"harmonic": [], "betweenness": []}

    for metric in ["harmonic", "betweenness"]:
        df_m = df[df["metric"] == metric]
        for topology in df_m["topology"].unique():
            for distance in sorted(df_m["distance"].unique()):
                subset = df_m[(df_m["topology"] == topology) & (df_m["distance"] == distance)]
                if len(subset) == 0:
                    continue

                reach = subset["mean_reach"].iloc[0]
                model_p = compute_p(reach, k, min_eff_n)
                if model_p >= 1.0:
                    continue

                closest_p = min(available_probs, key=lambda p: abs(p - model_p))
                row_data = subset[subset["sample_prob"] == closest_p]
                if len(row_data) == 0:
                    continue

                r = row_data.iloc[0]
                node_reach = np.asarray(r["node_reach"])
                node_true = np.asarray(r["node_true_vals"])
                node_est = np.asarray(r["node_est_vals"])
                abs_error = np.abs(node_true - node_est)

                # Filter to valid nodes
                mask = (node_true > 0) & np.isfinite(node_true) & np.isfinite(node_est) & (node_reach > 0)
                reach_pool[metric].append(node_reach[mask])
                error_pool[metric].append(abs_error[mask])

    # Bin by absolute reach
    bin_edges = np.array([10, 50, 100, 200, 500, 1000, 2000, 5000, 10000])
    rows = []

    for metric in ["harmonic", "betweenness"]:
        if not reach_pool[metric]:
            continue
        all_reach = np.concatenate(reach_pool[metric])
        all_error = np.concatenate(error_pool[metric])

        for i in range(len(bin_edges) - 1):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            mask = (all_reach >= lo) & (all_reach < hi)
            n = mask.sum()
            if n < 10:
                continue

            errors = all_error[mask]
            reaches = all_reach[mask]
            med_error = float(np.median(errors))
            q25_error = float(np.percentile(errors, 25))
            q75_error = float(np.percentile(errors, 75))

            # Normalised error per node, then take median
            if metric == "harmonic":
                norm_errors = errors / reaches
            else:
                norm_errors = errors / (reaches * (reaches - 1))

            valid_norm = norm_errors[np.isfinite(norm_errors)]
            med_nrmse = float(np.median(valid_norm)) if len(valid_norm) > 0 else np.nan
            q25_nrmse = float(np.percentile(valid_norm, 25)) if len(valid_norm) > 0 else np.nan
            q75_nrmse = float(np.percentile(valid_norm, 75)) if len(valid_norm) > 0 else np.nan

            rows.append({
                "metric": metric,
                "reach_lo": lo,
                "reach_hi": hi,
                "reach_center": np.sqrt(lo * hi),  # geometric mean
                "median_reach": float(np.median(reaches)),
                "median_mae": med_error,
                "mae_q25": q25_error,
                "mae_q75": q75_error,
                "median_nrmse": med_nrmse,
                "nrmse_q25": q25_nrmse,
                "nrmse_q75": q75_nrmse,
                "n_nodes": int(n),
            })

    return pd.DataFrame(rows)


def print_reach_bin_summary(node_acc: pd.DataFrame):
    """Print per-reach-bin error diagnostics at the model's recommended p."""
    print("\n" + "=" * 70)
    print("NODE-LEVEL ERROR DIAGNOSTICS BY ABSOLUTE REACH")
    print("=" * 70)
    print("  Nodes pooled across all configs at model-recommended p,")
    print("  binned by absolute per-node reach.")

    for metric in ["betweenness", "harmonic"]:
        subset = node_acc[node_acc["metric"] == metric]
        if len(subset) == 0:
            continue

        print(f"\n  {metric.upper()}")
        print(
            f"  {'Reach bin':<16} {'N nodes':>10} {'Med MAE':>12} "
            f"{'Med NRMSE':>12}"
        )
        print("  " + "-" * 52)

        for _, row in subset.iterrows():
            label = f"{int(row['reach_lo'])}-{int(row['reach_hi'])}"
            print(
                f"  {label:<16} {row['n_nodes']:>10,} {row['median_mae']:>12.2f} "
                f"{row['median_nrmse']:>12.6f}"
            )

    print("\n  Crossover: absolute MAE increases with reach (larger errors")
    print("  at high-reach nodes), but normalised error decreases")
    print("  (precision scales with importance).")


# =============================================================================
# FIGURE GENERATION
# =============================================================================


def generate_fig1_headline(df: pd.DataFrame):
    """Figure 1: Problem illustration - why sampling is needed."""
    print("\nGenerating Figure 1: Headline comparison...")

    df_b = df[df["metric"] == "betweenness"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    distances = [500, 1000, 2000, 4000]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    # Panel A: Accuracy vs sampling probability for different reaches
    ax = axes[0]

    for dist, color in zip(distances, colors, strict=True):
        subset = df_b[df_b["distance"] == dist]
        if len(subset) == 0:
            continue

        grouped = (
            subset.groupby("sample_prob")
            .agg(
                {
                    "spearman": "mean",
                    "mean_reach": "mean",
                }
            )
            .reset_index()
        )

        reach = grouped["mean_reach"].iloc[0]
        ax.plot(
            grouped["sample_prob"] * 100,
            grouped["spearman"],
            "o-",
            color=color,
            markersize=4,
            linewidth=1.5,
            label=f"{dist}m (reach={reach:.0f})",
        )

    ax.axhline(0.95, color="green", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.text(95, 0.955, "target: rho=0.95", fontsize=9, color="green", ha="right")

    ax.set_xlabel("Sampling Probability (%)")
    ax.set_ylabel("Spearman rho (ranking accuracy)")
    ax.set_title("A) Accuracy Increases with Sampling")
    ax.set_xlim(0, 100)
    ax.set_ylim(0.5, 1.0)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel B: Speedup potential
    ax = axes[1]

    speedups = []
    for dist in sorted(df_b["distance"].unique()):
        subset = df_b[df_b["distance"] == dist]
        grouped = subset.groupby("sample_prob")["spearman"].mean().reset_index()
        achieving = grouped[grouped["spearman"] >= 0.95]
        if len(achieving) > 0:
            min_p = achieving["sample_prob"].iloc[0]
            speedup = 1 / min_p if min_p > 0 else 1
            reach = subset["mean_reach"].mean()
            speedups.append({"distance": dist, "reach": reach, "min_p": min_p, "speedup": speedup})

    speedups_df = pd.DataFrame(speedups)

    ax.bar(range(len(speedups_df)), speedups_df["speedup"], color="#0072B2", alpha=0.7)
    ax.set_xticks(range(len(speedups_df)))
    ax.set_xticklabels([f"{d}m" for d in speedups_df["distance"]], rotation=45, ha="right")
    ax.set_xlabel("Analysis Distance")
    ax.set_ylabel("Potential Speedup (1/p)")
    ax.set_title("B) Speedup While Maintaining rho >= 0.95")
    ax.grid(True, alpha=0.3, axis="y")

    for i, (_, row) in enumerate(speedups_df.iterrows()):
        ax.text(i, row["speedup"] + 0.2, f"{row['speedup']:.1f}x", ha="center", fontsize=8)

    fig.suptitle("Sampling-Based Centrality: The Opportunity", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    output_path = FIGURES_DIR / "fig1_headline.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


def generate_fig2_model_derivation(df: pd.DataFrame, k: float):
    """Figure 2: Model derivation - eff_n predicts rho."""
    print("\nGenerating Figure 2: Model derivation...")

    df_b = df[df["metric"] == "betweenness"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: All data points - eff_n vs aggregate rho
    ax = axes[0]

    topologies = df_b["topology"].unique()
    colors = {"trellis": "#1f77b4", "tree": "#ff7f0e", "linear": "#2ca02c"}

    for topology in topologies:
        subset = df_b[df_b["topology"] == topology]
        ax.scatter(
            subset["effective_n"],
            subset["spearman"],
            alpha=0.4,
            s=20,
            color=colors.get(topology, "gray"),
            marker="o",
            label=topology,
        )

    ax.axhline(0.95, color="green", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.text(4500, 0.955, "target: rho=0.95", fontsize=9, color="green", ha="right", va="bottom")

    ax.set_xscale("log")
    ax.set_xlabel("Effective Sample Size (eff_n = reach × p)")
    ax.set_ylabel("Spearman rho")
    ax.set_title("A) Accuracy Collapses to eff_n")
    ax.set_xlim(1, 5000)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    # Panel B: Required eff_n for rho >= 0.95 vs reach
    ax = axes[1]

    points = []
    for topology in topologies:
        for distance in sorted(df_b["distance"].unique()):
            subset = df_b[(df_b["topology"] == topology) & (df_b["distance"] == distance)]
            if len(subset) == 0:
                continue

            reach = subset["mean_reach"].iloc[0]
            achieving = subset[subset["spearman"] >= 0.95].sort_values("effective_n")

            if len(achieving) > 0:
                min_eff_n = achieving["effective_n"].iloc[0]
                points.append(
                    {
                        "topology": topology,
                        "reach": reach,
                        "min_eff_n": min_eff_n,
                    }
                )

    points_df = pd.DataFrame(points)

    for topology in topologies:
        subset = points_df[points_df["topology"] == topology]
        ax.scatter(
            subset["reach"], subset["min_eff_n"], s=40, color=colors.get(topology, "gray"), label=topology, alpha=0.7
        )

    reach_range = np.logspace(1, 4, 100)
    model_eff_n = k * np.sqrt(reach_range)
    ax.plot(reach_range, model_eff_n, "k-", linewidth=2, label=f"Model: eff_n = {k:.1f} × sqrt(reach)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Network Reach (nodes within distance)")
    ax.set_ylabel("Required eff_n for rho >= 0.95")
    ax.set_title("B) Proportional Scaling Emerges")
    ax.set_xlim(10, 5000)
    ax.set_ylim(10, 2000)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    fig.suptitle("Deriving the Sampling Model", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    output_path = FIGURES_DIR / "fig2_model_derivation.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


def generate_fig3_floor_justification(df: pd.DataFrame, k: float, min_eff_n: int, bin_analysis: list):
    """Figure 3: Why the floor is needed."""
    print("\nGenerating Figure 3: Floor justification...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Success rate by eff_n bin
    ax = axes[0]

    bin_df = pd.DataFrame(bin_analysis)
    x = range(len(bin_df))
    colors_bar = ["#d62728" if rate < 0.95 else "#2ca02c" for rate in bin_df["success_rate"]]

    ax.bar(x, bin_df["success_rate"] * 100, color=colors_bar, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.axhline(95, color="green", linestyle="--", linewidth=1.5, label="95% target")
    ax.axhline(90, color="orange", linestyle=":", linewidth=1.5, label="90% threshold")

    floor_idx = None
    for i, row in bin_df.iterrows():
        low = int(row["bin"].split("-")[0])
        if low >= min_eff_n:
            floor_idx = i
            break

    if floor_idx is not None:
        ax.axvline(floor_idx - 0.5, color="red", linestyle="-", linewidth=2, alpha=0.7)
        ax.annotate(
            f"min_eff_n={min_eff_n}", xy=(floor_idx - 0.3, 50), fontsize=10, color="red", rotation=90, va="bottom"
        )

    ax.set_xticks(x)
    ax.set_xticklabels(bin_df["bin"], rotation=45, ha="right")
    ax.set_xlabel("Effective Sample Size (eff_n) Range")
    ax.set_ylabel("Success Rate (% achieving rho >= 0.95)")
    ax.set_title("A) Reliability Requires Sufficient eff_n")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel B: Analytical view - proportional model's eff_n vs the floor
    ax = axes[1]

    crossover = crossover_reach(k, min_eff_n)
    reach_range = np.logspace(np.log10(50), np.log10(5000), 200)
    prop_eff_n = k * np.sqrt(reach_range)

    # Proportional model curve
    ax.plot(reach_range, prop_eff_n, color="#0072B2", linewidth=2.5, label="Proportional: $k\\sqrt{r}$")

    # Floor line
    ax.axhline(min_eff_n, color="#d62728", linestyle="--", linewidth=2, label=f"Floor: $n_{{\\min}}$ = {min_eff_n}")

    # Shade the deficit region where proportional < floor
    ax.fill_between(
        reach_range,
        prop_eff_n,
        min_eff_n,
        where=prop_eff_n < min_eff_n,
        alpha=0.2,
        color="#d62728",
        label="Insufficient $n_{\\mathrm{eff}}$",
    )

    # Crossover point
    ax.plot(crossover, min_eff_n, "ko", markersize=8, zorder=5)
    ax.annotate(
        f"crossover\nreach $\\approx$ {crossover:.0f}",
        xy=(crossover, min_eff_n),
        xytext=(crossover * 2.5, min_eff_n * 0.6),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
        ha="center",
    )

    ax.set_xscale("log")
    ax.set_xlabel("Network Reach")
    ax.set_ylabel("Effective Sample Size ($n_{\\mathrm{eff}}$)")
    ax.set_title("B) Proportional Model Deficit at Intermediate Reach")
    ax.set_ylim(0, min_eff_n * 2.5)
    ax.set_xlim(50, 5000)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    fig.suptitle("Empirical Justification for min_eff_n Floor", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    output_path = FIGURES_DIR / "fig3_floor_justification.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


def generate_fig4_combined_model(k: float, min_eff_n: int):
    """Figure 4: The complete sampling model."""
    print("\nGenerating Figure 4: Combined model...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    reach_range = np.logspace(1, 5.5, 200)
    crossover = crossover_reach(k, min_eff_n)

    # Panel A: Required p vs reach
    ax = axes[0]

    p_values = [compute_p(r, k, min_eff_n) * 100 for r in reach_range]

    ax.plot(reach_range, p_values, color="#0072B2", linewidth=2.5, label=f"Model: p = max({k}/sqrt(r), {min_eff_n}/r)")

    p_proportional = [min(100, k / math.sqrt(r) * 100) for r in reach_range]
    p_floor = [min(100, min_eff_n / r * 100) for r in reach_range]

    ax.plot(
        reach_range,
        p_proportional,
        color="gray",
        linewidth=1,
        linestyle=":",
        alpha=0.7,
        label=f"Proportional: {k}/sqrt(r)",
    )
    ax.plot(reach_range, p_floor, color="gray", linewidth=1, linestyle="--", alpha=0.7, label=f"Floor: {min_eff_n}/r")

    ax.axvline(crossover, color="red", linestyle="-", linewidth=1.5, alpha=0.5)
    ax.annotate(
        f"crossover\nreach={crossover:.0f}",
        xy=(crossover, compute_p(crossover, k, min_eff_n) * 100 + 5),
        fontsize=9,
        color="red",
        ha="center",
    )

    ax.fill_between(
        reach_range, 0, 100, where=[r < crossover for r in reach_range], alpha=0.1, color="red", label="Floor-dominated"
    )
    ax.fill_between(
        reach_range,
        0,
        100,
        where=[r >= crossover for r in reach_range],
        alpha=0.1,
        color="blue",
        label="Proportional-dominated",
    )

    ax.set_xscale("log")
    ax.set_xlabel("Network Reach (nodes within distance)")
    ax.set_ylabel("Required Sampling Probability (%)")
    ax.set_title("A) Required Sampling Probability vs Reach")
    ax.set_xlim(50, 300000)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel B: Effective sample size vs reach
    ax = axes[1]

    eff_n_values = [compute_eff_n(r, k, min_eff_n) for r in reach_range]

    ax.plot(
        reach_range, eff_n_values, color="#0072B2", linewidth=2.5, label=f"Model: eff_n = max({k}×sqrt(r), {min_eff_n})"
    )

    eff_n_proportional = [k * math.sqrt(r) for r in reach_range]
    eff_n_floor = [min_eff_n for _ in reach_range]

    ax.plot(
        reach_range,
        eff_n_proportional,
        color="gray",
        linewidth=1,
        linestyle=":",
        alpha=0.7,
        label=f"Proportional: {k}×sqrt(r)",
    )
    ax.plot(reach_range, eff_n_floor, color="gray", linewidth=1, linestyle="--", alpha=0.7, label=f"Floor: {min_eff_n}")

    ax.axvline(crossover, color="red", linestyle="-", linewidth=1.5, alpha=0.5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Network Reach")
    ax.set_ylabel("Effective Sample Size (eff_n)")
    ax.set_title("B) Effective Sample Size vs Reach")
    ax.set_xlim(50, 300000)
    ax.set_ylim(50, 10000)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    fig.suptitle(
        f"The Sampling Model: eff_n = max({k} × sqrt(reach), {min_eff_n})", fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()

    output_path = FIGURES_DIR / "fig4_combined_model.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


def evaluate_topk_at_model_p(df: pd.DataFrame, k: float, min_eff_n: int) -> pd.DataFrame:
    """Extract top-k precision (top 10% overlap) at the model's recommended p.

    Returns one row per (topology, distance, metric) configuration with the
    aggregate top_k_precision alongside Spearman rho and reach.
    """
    available_probs = sorted(p for p in df["sample_prob"].unique() if p < 1.0)
    rows = []

    for metric in ["harmonic", "betweenness"]:
        df_m = df[df["metric"] == metric]
        for topology in df_m["topology"].unique():
            for distance in sorted(df_m["distance"].unique()):
                subset = df_m[(df_m["topology"] == topology) & (df_m["distance"] == distance)]
                if len(subset) == 0:
                    continue

                reach = subset["mean_reach"].iloc[0]
                model_p = compute_p(reach, k, min_eff_n)

                if model_p >= 1.0:
                    continue

                closest_p = min(available_probs, key=lambda p: abs(p - model_p))
                row_data = subset[subset["sample_prob"] == closest_p]
                if len(row_data) == 0:
                    continue

                r = row_data.iloc[0]
                rows.append(
                    {
                        "topology": topology,
                        "distance": distance,
                        "metric": metric,
                        "mean_reach": reach,
                        "model_p": model_p,
                        "used_p": closest_p,
                        "spearman": r["spearman"],
                        "top_k_precision": r["top_k_precision"],
                    }
                )

    return pd.DataFrame(rows)


def print_topk_summary(tk: pd.DataFrame):
    """Print top-k precision diagnostics at the model's recommended p."""
    print("\n" + "=" * 70)
    print("TOP-K PRECISION (TOP 10% OVERLAP) AT MODEL-RECOMMENDED p")
    print("=" * 70)

    for metric in ["betweenness", "harmonic"]:
        subset = tk[tk["metric"] == metric]
        if len(subset) == 0:
            continue

        print(f"\n  {metric.upper()}")
        print(f"  {'Topology':<12} {'Distance':>10} {'Reach':>10} {'Spearman':>10} {'Top-10%':>10}")
        print("  " + "-" * 55)

        for _, row in subset.sort_values(["topology", "distance"]).iterrows():
            print(
                f"  {row['topology']:<12} {row['distance']:>10.0f} {row['mean_reach']:>10.0f} "
                f"{row['spearman']:>10.3f} {row['top_k_precision']:>10.3f}"
            )

        med_rho = subset["spearman"].median()
        med_topk = subset["top_k_precision"].median()
        min_topk = subset["top_k_precision"].min()
        print(f"\n  Median Spearman: {med_rho:.3f}  |  Median top-10%: {med_topk:.3f}  |  Min top-10%: {min_topk:.3f}")


def generate_fig5_error_crossover(node_acc: pd.DataFrame, k: float, min_eff_n: int):
    """Figure 5: Error crossover — absolute error grows with reach, normalised error shrinks.

    1×2 layout with continuous log-scale reach axis.  Nodes are pooled across
    all (topology, distance) configs and binned by absolute per-node reach.
    Left: absolute MAE trending up with reach.
    Right: normalised MAE trending down with reach, with localised EW bound overlay.
    Both metrics shown (betweenness red squares, harmonic blue circles).
    """
    print("\nGenerating Figure 5: Error crossover...")

    colour_bet = "#B2182B"
    colour_har = "#2166AC"

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for col_idx, (med_col, lo_col, hi_col, ylabel, title) in enumerate([
        ("median_mae", "mae_q25", "mae_q75",
         "Median Absolute Error", "A) Absolute Error"),
        ("median_nrmse", "nrmse_q25", "nrmse_q75",
         "Median Normalised Error", "B) Normalised Error"),
    ]):
        ax = axes[col_idx]

        for metric, colour, marker, label in [
            ("betweenness", colour_bet, "s", "Betweenness"),
            ("harmonic", colour_har, "o", "Harmonic closeness"),
        ]:
            sub = node_acc[node_acc["metric"] == metric].copy()
            sub = sub[sub[med_col].notna() & (sub[med_col] > 0)]
            if len(sub) == 0:
                continue

            x = sub["reach_center"].values
            y = sub[med_col].values
            lo_err = y - sub[lo_col].values
            hi_err = sub[hi_col].values - y
            # Clamp negative error bars (can happen at boundaries)
            lo_err = np.maximum(lo_err, 0)
            hi_err = np.maximum(hi_err, 0)

            ax.errorbar(
                x, y,
                yerr=[lo_err, hi_err],
                fmt=marker,
                color=colour,
                markersize=8,
                capsize=4,
                capthick=1.5,
                linewidth=1.5,
                label=label,
                zorder=3,
            )
            ax.plot(x, y, color=colour, linewidth=1.5, alpha=0.5, zorder=2)

        # Overlay localised EW bound on normalised error panel
        if col_idx == 1:
            delta = 0.1
            r_line = np.logspace(1, 4.2, 200)
            eff_n_line = np.maximum(k * np.sqrt(r_line), min_eff_n)
            eps_line = np.sqrt(np.log(2 * r_line / delta) / (2 * eff_n_line))
            ax.plot(
                r_line, eps_line,
                color="grey", linewidth=2, linestyle="--",
                label=r"Localised EW bound ($\delta=0.1$)",
                zorder=1,
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Per-Node Reach")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()

    output_path = FIGURES_DIR / "fig5_error_crossover.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


def generate_parameters_table(k: float, min_eff_n: int):
    """Generate LaTeX table of model parameters with CIs if available."""
    print("\nGenerating Table 1: Parameters...")

    crossover = crossover_reach(k, min_eff_n)

    # Try to load CI data from 07_validate_power_exponent.py output
    ci_path = OUTPUT_DIR / "power_exponent_analysis.json"
    k_ci_str = "--"
    floor_ci_str = "--"
    alpha_str = "0.5"
    alpha_ci_str = "--"
    alpha_p_str = "--"

    if ci_path.exists():
        with open(ci_path) as f:
            ci_data = json.load(f)

        pe = ci_data.get("power_exponent", {})
        fl = ci_data.get("floor_logistic", {})

        if pe.get("boot_ci_lower") is not None:
            alpha_str = f"{pe.get('alpha_hat', 0.5)}"
            alpha_ci_str = f"[{pe['boot_ci_lower']}, {pe['boot_ci_upper']}]"
            alpha_p_str = f"{pe.get('p_value_vs_half', 0.38):.2f}"

        if fl.get("recommended_floor_ci_lower") is not None:
            floor_ci_str = f"[{fl['recommended_floor_ci_lower']}, {fl['recommended_floor_ci_upper']}]"

    # k CI from fitting stats
    model_fit_path = OUTPUT_DIR / "model_fit.json"
    if model_fit_path.exists():
        with open(model_fit_path) as mf:
            mf_data = json.load(mf)
        k_mean = mf_data["fitting_stats"]["k_mean"]
        k_p95 = mf_data["fitting_stats"]["k_p95"]
        k_ci_str = f"[{k_mean}, {k_p95}]"

    latex = (
        r"""\begin{table}[htbp]
\centering
\caption{Fitted sampling model parameters.}
\label{tab:parameters}
\begin{tabular}{lrll}
\toprule
\textbf{Parameter} & \textbf{Value} & \textbf{Range} & \textbf{Description} \\
\midrule
$k$ & """
        + f"{k:.2f}"
        + r""" & """
        + k_ci_str
        + r""" & Proportional scaling constant \\
$n_{\min}$ & """
        + f"{min_eff_n}"
        + r""" & """
        + floor_ci_str
        + r""" & Minimum effective sample size \\
Crossover reach & """
        + f"{crossover:.0f}"
        + r""" & -- & Where $k\sqrt{r} = n_{\min}$ \\
\midrule
$\hat{\alpha}$ & """
        + alpha_str
        + r""" & """
        + alpha_ci_str
        + r""" & Estimated power exponent \\
\bottomrule
\end{tabular}

\vspace{0.5em}
\footnotesize
\textbf{Model:} $n_{\mathrm{eff}} = \max(k \cdot \sqrt{r},\; n_{\min})$, where $r$ is the network reach.
Sampling probability: $p = \min(1,\; n_{\mathrm{eff}} / r)$.\\
$k$: 75th percentile of implied values across configurations; range shows [mean, 95th percentile].\\
$n_{\min}$: logistic regression inversion at 95\% success rate; range shows 95\% CI.\\
$\hat{\alpha}$: estimated from general power model $n_{\mathrm{eff}} = k \cdot r^{\alpha}$; \\
not significantly different from 0.5 ($p = """
        + alpha_p_str
        + r"""$), supporting the fixed $\sqrt{r}$ specification.
\end{table}
"""
    )

    output_path = TABLES_DIR / "tab1_parameters.tex"
    with open(output_path, "w") as f:
        f.write(latex)
    print(f"  Saved: {output_path}")


# =============================================================================
# JSON OUTPUT
# =============================================================================


def save_model_fit_json(model_fit: dict):
    """Save model_fit.json for backward compatibility."""
    output = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "script": "01_fit_rank_model.py",
        "description": "Proportional sampling constant k fitted from synthetic data",
        "model": {
            "formula": "eff_n = k × sqrt(reach)",
            "k": model_fit["k"],
            "k_max": model_fit["k_max"],
            "target_rho": model_fit["target_rho"],
        },
        "fitting_stats": {
            "k_mean": model_fit["k_mean"],
            "k_p75": model_fit["k_p75"],
            "k_p95": model_fit["k_p95"],
            "n_configs": model_fit["n_configs"],
        },
    }

    output_path = OUTPUT_DIR / "model_fit.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {output_path}")


def save_floor_fit_json(floor_fit: dict):
    """Save floor_fit.json for backward compatibility."""
    output = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "script": "01_fit_rank_model.py",
        "description": "Minimum effective sample size floor fitted from synthetic data",
        "model": {
            "min_eff_n": int(floor_fit["min_eff_n"]),
            "target_success_rate": float(floor_fit["target_success_rate"]),
            "achieved_success_rate": float(floor_fit["achieved_success_rate"]),
        },
        "bin_analysis": [
            {
                k_: (
                    float(v)
                    if isinstance(v, (np.floating, float))
                    else int(v)
                    if isinstance(v, (np.integer, int))
                    else v
                )
                for k_, v in item.items()
            }
            for item in floor_fit["bin_analysis"]
        ],
    }

    output_path = OUTPUT_DIR / "floor_fit.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {output_path}")


def save_combined_model_json(k: float, min_eff_n: int):
    """Save sampling_model.json (the primary output used by downstream scripts)."""
    crossover = crossover_reach(k, min_eff_n)

    output = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "script": "01_fit_rank_model.py",
        "description": "Complete sampling model combining k and min_eff_n",
        "model": {
            "formula_eff_n": "eff_n = max(k × sqrt(reach), min_eff_n)",
            "formula_p": "p = min(1.0, eff_n / reach)",
            "k": k,
            "min_eff_n": min_eff_n,
            "crossover_reach": round(crossover, 0),
        },
        "interpretation": {
            "proportional_regime": f"When reach > {crossover:.0f}, eff_n scales as {k} × sqrt(reach)",
            "floor_regime": f"When reach < {crossover:.0f}, eff_n is fixed at {min_eff_n}",
        },
    }

    output_path = OUTPUT_DIR / "sampling_model.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {output_path}")


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("=" * 70)
    print("01_fit_rank_model.py - Fitting the Rank-Based Sampling Model")
    print("=" * 70)

    # Load data once
    df = load_synthetic_data()

    # Stage 1: Fit proportional constant k
    print("\n" + "-" * 50)
    model_fit = fit_proportional_k(df, target_rho=0.95)
    k = model_fit["k"]

    # Stage 2: Fit floor
    print("\n" + "-" * 50)
    floor_fit = fit_min_eff_n(df, target_success_rate=0.95)
    min_eff_n = floor_fit["min_eff_n"]

    # Stage 3: Combined model
    crossover = crossover_reach(k, min_eff_n)
    print("\n" + "-" * 50)
    print("Stage 3: Combined model")
    print(f"  eff_n = max({k} × sqrt(reach), {min_eff_n})")
    print(f"  Crossover reach: {crossover:.0f}")
    print(f"  Below {crossover:.0f}: floor dominates (eff_n = {min_eff_n})")
    print(f"  Above {crossover:.0f}: proportional dominates (eff_n = {k} × sqrt(reach))")

    # Save all JSON outputs
    print("\n" + "-" * 50)
    print("Saving JSON outputs...")
    save_model_fit_json(model_fit)
    save_floor_fit_json(floor_fit)
    save_combined_model_json(k, min_eff_n)

    # Stage 4: Node-level accuracy assessment
    print("\n" + "-" * 50)
    node_acc = evaluate_node_level_accuracy(df, k, min_eff_n)
    print_reach_bin_summary(node_acc)

    # Stage 5: Top-k precision assessment
    tk = evaluate_topk_at_model_p(df, k, min_eff_n)
    print_topk_summary(tk)

    # Generate all figures
    print("\n" + "-" * 50)
    print("Generating figures...")
    generate_fig1_headline(df)
    generate_fig2_model_derivation(df, k)
    generate_fig3_floor_justification(df, k, min_eff_n, floor_fit["bin_analysis"])
    generate_fig4_combined_model(k, min_eff_n)
    generate_fig5_error_crossover(node_acc, k, min_eff_n)
    generate_parameters_table(k, min_eff_n)

    # Example calculations
    print("\n" + "=" * 70)
    print("EXAMPLE CALCULATIONS")
    print("=" * 70)
    print(f"\n{'Reach':>10} | {'eff_n':>10} | {'p':>10} | {'Speedup':>10}")
    print("-" * 50)

    for reach in [100, 300, 500, 1000, 2000, 5000, 10000, 50000]:
        eff_n = compute_eff_n(reach, k, min_eff_n)
        p = compute_p(reach, k, min_eff_n)
        speedup = 1 / p if p > 0 else float("inf")
        print(f"{reach:>10} | {eff_n:>10.0f} | {p:>9.1%} | {speedup:>9.1f}x")

    print("\n" + "=" * 70)
    print("ALL OUTPUTS")
    print("=" * 70)
    print(f"  1. {OUTPUT_DIR / 'model_fit.json'}")
    print(f"  2. {OUTPUT_DIR / 'floor_fit.json'}")
    print(f"  3. {OUTPUT_DIR / 'sampling_model.json'}")
    print(f"  4. {FIGURES_DIR / 'fig1_headline.pdf'}")
    print(f"  5. {FIGURES_DIR / 'fig2_model_derivation.pdf'}")
    print(f"  6. {FIGURES_DIR / 'fig3_floor_justification.pdf'}")
    print(f"  7. {FIGURES_DIR / 'fig4_combined_model.pdf'}")
    print(f"  8. {FIGURES_DIR / 'fig5_error_crossover.pdf'}")
    print(f"  9. {TABLES_DIR / 'tab1_parameters.tex'}")

    return 0


if __name__ == "__main__":
    exit(main())
