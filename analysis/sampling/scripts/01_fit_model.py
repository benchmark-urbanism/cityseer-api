#!/usr/bin/env python
"""
01_fit_model.py - Fit the proportional sampling constant k.

This script fits the proportional constant k for the sampling model:
    eff_n = max(k × sqrt(reach), min_eff_n)

The proportional term k × sqrt(reach) comes from sampling theory:
standard error scales as 1/sqrt(n), so to maintain consistent precision
as reach grows, we need eff_n proportional to sqrt(reach).

Uses cached synthetic network data from 3 topologies × 12 distances × 22 probabilities.

Outputs:
    - output/model_fit.json: Fitted k value with metadata
    - paper/figures/fig1_headline.pdf: Problem illustration (full vs sampled)
    - paper/figures/fig2_model_derivation.pdf: eff_n vs rho relationship
"""

import json
import math
import pickle
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
SAMPLING_DIR = SCRIPT_DIR.parent  # analysis/sampling
OUTPUT_DIR = SAMPLING_DIR / "output"
FIGURES_DIR = SAMPLING_DIR / "paper" / "figures"
CACHE_DIR = SAMPLING_DIR.parent.parent / "temp" / "sampling_cache"

OUTPUT_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Cache file from old analysis (reused)
SYNTHETIC_CACHE = CACHE_DIR / "sampling_analysis_v17.pkl"

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
# MODEL FITTING
# =============================================================================


def fit_proportional_k(df: pd.DataFrame, target_rho: float = 0.95) -> dict:
    """
    Fit the proportional constant k from synthetic data.

    For each (topology, distance) combination, find the minimum p where rho >= target_rho.
    Then compute k = p × sqrt(reach) and take the maximum across all combinations.

    Parameters
    ----------
    df : pd.DataFrame
        Synthetic data with columns: topology, distance, mean_reach, sample_prob, spearman
    target_rho : float
        Target Spearman correlation (default 0.95)

    Returns
    -------
    dict
        Fitted model with k value and fitting details
    """
    # Filter to betweenness (more demanding metric)
    df_b = df[df["metric"] == "betweenness"].copy()

    results = []

    for topology in df_b["topology"].unique():
        for distance in sorted(df_b["distance"].unique()):
            subset = df_b[(df_b["topology"] == topology) & (df_b["distance"] == distance)]
            if len(subset) == 0:
                continue

            reach = subset["mean_reach"].iloc[0]

            # Sort by sample_prob to find minimum p achieving target
            subset_sorted = subset.sort_values("sample_prob")

            # Find minimum p where rho >= target_rho
            achieving = subset_sorted[subset_sorted["spearman"] >= target_rho]
            if len(achieving) == 0:
                # Target not achieved even at p=1.0
                min_p = 1.0
                achieved_rho = subset_sorted["spearman"].max()
            else:
                min_p = achieving["sample_prob"].iloc[0]
                achieved_rho = achieving["spearman"].iloc[0]

            # Compute implied k = p × sqrt(reach)
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

    # Compute k statistics across all configurations
    k_max = results_df["k_implied"].max()
    k_mean = results_df["k_implied"].mean()
    k_p75 = results_df["k_implied"].quantile(0.75)  # Upper quartile
    k_p95 = results_df["k_implied"].quantile(0.95)

    # Use upper quartile (75th percentile) for less conservative model
    # This allows ~25% of edge cases to potentially fall slightly below target
    # but provides meaningful speedup gains
    k_selected = k_p75

    print("\nProportional k fitting results:")
    print(f"  Target rho: {target_rho}")
    print("  k values across configs:")
    print(f"    mean={k_mean:.2f}, 75th={k_p75:.2f}, 95th={k_p95:.2f}, max={k_max:.2f}")
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
# FIGURE GENERATION
# =============================================================================


def generate_fig1_headline(df: pd.DataFrame):
    """
    Figure 1: Problem illustration - why sampling is needed.

    Shows that full centrality computation is expensive, but sampled
    centrality can achieve high accuracy with much less computation.
    """
    print("\nGenerating Figure 1: Headline comparison...")

    df_b = df[df["metric"] == "betweenness"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Accuracy vs sampling probability for different reaches
    ax = axes[0]

    # Select representative distances
    distances = [500, 1000, 2000, 4000]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for dist, color in zip(distances, colors, strict=True):
        subset = df_b[df_b["distance"] == dist]
        if len(subset) == 0:
            continue

        # Average across topologies
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

    # For each distance, find minimum p for rho >= 0.95
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

    # Add speedup values as text
    for i, (_, row) in enumerate(speedups_df.iterrows()):
        ax.text(i, row["speedup"] + 0.2, f"{row['speedup']:.1f}x", ha="center", fontsize=8)

    fig.suptitle("Sampling-Based Centrality: The Opportunity", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    output_path = FIGURES_DIR / "fig1_headline.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


def generate_fig2_model_derivation(df: pd.DataFrame, k: float):
    """
    Figure 2: Model derivation - eff_n predicts rho.

    Shows that effective sample size (reach × p) is the key predictor
    of accuracy, and derives the proportional relationship.
    """
    print("\nGenerating Figure 2: Model derivation...")

    df_b = df[df["metric"] == "betweenness"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: All data points - eff_n vs rho
    ax = axes[0]

    # Color by topology
    topologies = df_b["topology"].unique()
    colors = {"trellis": "#1f77b4", "tree": "#ff7f0e", "linear": "#2ca02c"}

    for topology in topologies:
        subset = df_b[df_b["topology"] == topology]
        ax.scatter(
            subset["effective_n"],
            subset["spearman"],
            alpha=0.3,
            s=15,
            color=colors.get(topology, "gray"),
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

    # For each (topology, distance), find min eff_n for rho >= 0.95
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

    # Plot the fitted model: eff_n = k × sqrt(reach)
    reach_range = np.logspace(1, 4, 100)
    model_eff_n = k * np.sqrt(reach_range)
    ax.plot(reach_range, model_eff_n, "k-", linewidth=2, label=f"Model: eff_n = {k:.1f} × sqrt(reach)")

    # Note: floor line not shown here - it's fitted in 02_fit_floor.py
    # and properly introduced in fig3_floor_justification.pdf

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


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("=" * 70)
    print("01_fit_model.py - Fitting proportional constant k")
    print("=" * 70)

    # Load data
    df = load_synthetic_data()

    # Fit model
    model_fit = fit_proportional_k(df, target_rho=0.95)

    # Save model fit
    output = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "script": "01_fit_model.py",
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
    print(f"\nSaved model fit: {output_path}")

    # Generate figures
    generate_fig1_headline(df)
    generate_fig2_model_derivation(df, model_fit["k"])

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nFitted proportional constant: k = {model_fit['k']} (75th percentile)")
    print(f"Conservative alternative (max): k = {model_fit['k_max']}")
    print(f"\nModel: eff_n = {model_fit['k']} × sqrt(reach)")
    print(f"       p = eff_n / reach = {model_fit['k']} / sqrt(reach)")

    print("\n" + "=" * 70)
    print("OUTPUTS")
    print("=" * 70)
    print(f"  1. {OUTPUT_DIR / 'model_fit.json'}")
    print(f"  2. {FIGURES_DIR / 'fig1_headline.pdf'}")
    print(f"  3. {FIGURES_DIR / 'fig2_model_derivation.pdf'}")

    return 0


if __name__ == "__main__":
    exit(main())
