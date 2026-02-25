#!/usr/bin/env python
"""
01_fit_rank_model.py - Analyse synthetic sampling data and generate figures.

Loads the synthetic sampling cache and generates:
  1. Figure 1 (headline): Sampling opportunity — 1x3 layout:
     A) Closeness accuracy vs epsilon (Hoeffding + spatial source sampling),
     B) Betweenness accuracy vs epsilon (same framework),
     C) Speedup comparison at the paper's default epsilons.
  2. Figure 3 (error crossover): Precision scales with importance — absolute
     and normalised MAE by per-node reach bin, with theoretical bound overlay
     for both closeness and betweenness.

Both metrics use the same unified framework:
  Hoeffding bound → p from (epsilon, reach) → spatial source_indices → IPW scaling.

Requires:
    - .cache/sampling_analysis_{CACHE_VERSION}.pkl (from 00_generate_cache.py)

Outputs:
    - paper/figures/fig1_headline.pdf: Sampling opportunity (main body)
    - paper/figures/fig3_error_crossover.pdf: Precision scales with importance (main body)
"""

import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cityseer.config import compute_hoeffding_p
from utilities import (
    CACHE_DIR,
    CACHE_VERSION,
    FIGURES_DIR,
    HOEFFDING_DELTA,
    ew_predicted_epsilon,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

SYNTHETIC_CACHE = CACHE_DIR / f"sampling_analysis_{CACHE_VERSION}.pkl"

# Paper default epsilons (separate for each metric)
PAPER_EPSILON_CLOSENESS = 0.1
PAPER_EPSILON_BETWEENNESS = 0.1

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
    print(f"  Topologies: {df['topology'].unique().tolist()}")
    print(f"  Distances: {sorted(df['distance'].unique())}")
    print(f"  Metrics: {df['metric'].unique().tolist()}")

    for metric_label in df["metric"].unique():
        subset = df[df["metric"] == metric_label]
        probs = sorted(subset["sample_prob"].dropna().unique())
        print(f"  {metric_label}: {len(subset)} rows, sample_probs: {probs}")

    return df


# =============================================================================
# NODE-LEVEL ACCURACY ASSESSMENT
# =============================================================================


def _pool_node_errors(df_subset: pd.DataFrame, metric_label: str) -> list[dict]:
    """Pool node-level errors from a DataFrame subset and bin by absolute reach."""
    reach_pool: list[np.ndarray] = []
    error_pool: list[np.ndarray] = []

    for _, row in df_subset.iterrows():
        node_reach = np.asarray(row["node_reach"])
        node_true = np.asarray(row["node_true_vals"])
        node_est = np.asarray(row["node_est_vals"])
        abs_error = np.abs(node_true - node_est)

        mask = (node_true > 0) & np.isfinite(node_true) & np.isfinite(node_est) & (node_reach > 0)
        reach_pool.append(node_reach[mask])
        error_pool.append(abs_error[mask])

    if not reach_pool:
        return []

    all_reach = np.concatenate(reach_pool)
    all_error = np.concatenate(error_pool)

    bin_edges = np.array([10, 50, 100, 200, 500, 1000, 2000, 5000, 10000])
    rows = []

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

        norm_errors = errors / reaches
        valid_norm = norm_errors[np.isfinite(norm_errors)]
        med_nrmse = float(np.median(valid_norm)) if len(valid_norm) > 0 else np.nan
        q25_nrmse = float(np.percentile(valid_norm, 25)) if len(valid_norm) > 0 else np.nan
        q75_nrmse = float(np.percentile(valid_norm, 75)) if len(valid_norm) > 0 else np.nan

        rows.append(
            {
                "metric": metric_label,
                "reach_lo": lo,
                "reach_hi": hi,
                "reach_center": np.sqrt(lo * hi),
                "median_reach": float(np.median(reaches)),
                "median_mae": med_error,
                "mae_q25": q25_error,
                "mae_q75": q75_error,
                "median_nrmse": med_nrmse,
                "nrmse_q25": q25_nrmse,
                "nrmse_q75": q75_nrmse,
                "n_nodes": int(n),
            }
        )

    return rows


def evaluate_node_level_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate per-node accuracy at a representative sample probability.

    Pools all nodes across all (topology, distance) configs at p=0.1
    for each metric, then bins by absolute per-node reach.
    """
    representative_p = 0.1
    all_rows: list[dict] = []

    # Closeness at representative p
    df_close = df[(df["metric"] == "harmonic") & (df["sample_prob"] == representative_p)]
    all_rows.extend(_pool_node_errors(df_close, "harmonic"))

    # Betweenness at representative p
    df_betw = df[(df["metric"] == "betweenness") & (df["sample_prob"] == representative_p)]
    all_rows.extend(_pool_node_errors(df_betw, "betweenness"))

    return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()


def print_reach_bin_summary(node_acc: pd.DataFrame):
    """Print per-reach-bin error diagnostics."""
    print("\n" + "=" * 70)
    print("NODE-LEVEL ERROR DIAGNOSTICS BY ABSOLUTE REACH")
    print("=" * 70)

    for metric_label, display_name in [
        ("harmonic", "HARMONIC CLOSENESS"),
        ("betweenness", "BETWEENNESS"),
    ]:
        subset = node_acc[node_acc["metric"] == metric_label]
        if len(subset) > 0:
            print(f"\n  {display_name} (p=0.1)")
            print(f"  {'Reach bin':<16} {'N nodes':>10} {'Med MAE':>12} {'Med NRMSE':>12}")
            print("  " + "-" * 52)

            for _, row in subset.iterrows():
                label = f"{int(row['reach_lo'])}-{int(row['reach_hi'])}"
                print(f"  {label:<16} {row['n_nodes']:>10,} {row['median_mae']:>12.2f} {row['median_nrmse']:>12.6f}")


# =============================================================================
# FIGURE GENERATION
# =============================================================================


def generate_fig1_headline(df: pd.DataFrame):
    """Figure 1: Sampling opportunity — 1x3 layout.

    Panel A: Closeness — Spearman rho vs sample probability.
    Panel B: Betweenness — Spearman rho vs sample probability (same framework).
    Panel C: Speedup comparison at paper epsilons — grouped bars (both use 1/p).
    """
    print("\nGenerating Figure 1: Headline comparison...")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    representative_dists = [500, 1000, 2000, 4000]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    # ------------------------------------------------------------------
    # Panel A: Closeness — rho vs sample probability
    # ------------------------------------------------------------------
    ax = axes[0]

    for dist, color in zip(representative_dists, colors, strict=True):
        subset = df[(df["metric"] == "harmonic") & (df["distance"] == dist)]
        if len(subset) > 0:
            grouped = (
                subset.groupby("sample_prob")
                .agg({"spearman": "mean", "mean_reach": "mean"})
                .reset_index()
                .sort_values("sample_prob")
            )
            reach = grouped["mean_reach"].iloc[0]
            ax.plot(
                grouped["sample_prob"],
                grouped["spearman"],
                "o-",
                color=color,
                markersize=4,
                linewidth=1.5,
                label=f"{dist}m (r={reach:.0f})",
            )

    ax.axhline(0.95, color="green", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.text(0.02, 0.955, r"target: $\rho$=0.95", fontsize=9, color="green", ha="left")

    ax.set_xlabel("Sample probability $p$")
    ax.set_ylabel(r"Spearman $\rho$ (ranking accuracy)")
    ax.set_title("A) Closeness: accuracy vs sample probability")
    ax.set_xscale("log")
    ax.set_ylim(0.5, 1.02)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Panel B: Betweenness — rho vs sample probability (same framework)
    # ------------------------------------------------------------------
    ax = axes[1]

    for dist, color in zip(representative_dists, colors, strict=True):
        subset = df[(df["metric"] == "betweenness") & (df["distance"] == dist)]
        if len(subset) > 0:
            grouped = (
                subset.groupby("sample_prob")
                .agg({"spearman": "mean", "mean_reach": "mean"})
                .reset_index()
                .sort_values("sample_prob")
            )
            reach = grouped["mean_reach"].iloc[0]
            ax.plot(
                grouped["sample_prob"],
                grouped["spearman"],
                "s-",
                color=color,
                markersize=4,
                linewidth=1.5,
                label=f"{dist}m (r={reach:.0f})",
            )

    ax.axhline(0.95, color="green", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.text(0.02, 0.955, r"target: $\rho$=0.95", fontsize=9, color="green", ha="left")

    ax.set_xlabel("Sample probability $p$")
    ax.set_ylabel(r"Spearman $\rho$ (ranking accuracy)")
    ax.set_title("B) Betweenness: accuracy vs sample probability")
    ax.set_xscale("log")
    ax.set_ylim(0.5, 1.02)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Panel C: Speedup comparison at paper epsilons (grouped bars, both 1/p)
    # ------------------------------------------------------------------
    ax = axes[2]

    all_distances = sorted(df["distance"].unique())
    bar_data = []

    for dist in all_distances:
        row_data: dict = {"distance": dist, "speedup_closeness": 1.0, "speedup_betweenness": 1.0}

        # Closeness speedup: 1/p
        subset_h = df[(df["metric"] == "harmonic") & (df["distance"] == dist) & (df["epsilon"] == PAPER_EPSILON_CLOSENESS)]
        if len(subset_h) > 0:
            mean_reach = subset_h["mean_reach"].mean()
            p = compute_hoeffding_p(mean_reach, epsilon=PAPER_EPSILON_CLOSENESS, delta=HOEFFDING_DELTA)
            row_data["speedup_closeness"] = 1.0 / p if 0 < p < 1.0 else 1.0

        # Betweenness speedup: 1/p
        subset_b = df[(df["metric"] == "betweenness") & (df["distance"] == dist) & (df["epsilon"] == PAPER_EPSILON_BETWEENNESS)]
        if len(subset_b) > 0:
            mean_reach = subset_b["mean_reach"].mean()
            p = compute_hoeffding_p(mean_reach, epsilon=PAPER_EPSILON_BETWEENNESS, delta=HOEFFDING_DELTA)
            row_data["speedup_betweenness"] = 1.0 / p if 0 < p < 1.0 else 1.0

        bar_data.append(row_data)

    bar_df = pd.DataFrame(bar_data)
    n_dists = len(bar_df)
    x = np.arange(n_dists)
    width = 0.35

    bars_h = ax.bar(
        x - width / 2,
        bar_df["speedup_closeness"],
        width,
        color="#2166AC",
        alpha=0.8,
        label=rf"Closeness ($\varepsilon$={PAPER_EPSILON_CLOSENESS})",
    )
    bars_b = ax.bar(
        x + width / 2,
        bar_df["speedup_betweenness"],
        width,
        color="#B2182B",
        alpha=0.8,
        label=rf"Betweenness ($\varepsilon$={PAPER_EPSILON_BETWEENNESS})",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}m" for d in bar_df["distance"]], rotation=45, ha="right")
    ax.set_xlabel("Analysis Distance")
    ax.set_ylabel("Speedup Factor (1/p)")
    ax.set_title("C) Speedup at paper defaults")
    ax.set_yscale("log")
    ax.axhline(1.0, color="grey", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y", which="both")

    for bars in [bars_h, bars_b]:
        for bar in bars:
            height = bar.get_height()
            if height > 1.05:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height * 1.1,
                    f"{height:.1f}x",
                    ha="center",
                    fontsize=7,
                )

    fig.suptitle(
        "Sampling-Based Centrality: The Opportunity",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    output_path = FIGURES_DIR / "fig1_headline.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


def generate_fig3_error_crossover(node_acc: pd.DataFrame, df: pd.DataFrame):
    """Figure 3: Error crossover — absolute error grows with reach, normalised error shrinks.

    1x2 layout. Nodes pooled at p=0.1 for each metric.
    Panel A: Absolute MAE trending up with reach.
    Panel B: Normalised MAE trending down, with Hoeffding bound overlay for both metrics.
    """
    print("\nGenerating Figure 3: Error crossover...")

    colour_har = "#2166AC"
    colour_betw = "#B2182B"

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for col_idx, (med_col, lo_col, hi_col, ylabel, title) in enumerate(
        [
            ("median_mae", "mae_q25", "mae_q75", "Median Absolute Error", "A) Absolute Error"),
            ("median_nrmse", "nrmse_q25", "nrmse_q75", "Median Normalised Error", "B) Normalised Error"),
        ]
    ):
        ax = axes[col_idx]

        for metric_label, colour, marker, display, paper_eps in [
            ("harmonic", colour_har, "o", "Harmonic closeness", PAPER_EPSILON_CLOSENESS),
            ("betweenness", colour_betw, "s", "Betweenness", PAPER_EPSILON_BETWEENNESS),
        ]:
            sub = node_acc[node_acc["metric"] == metric_label].copy()
            sub = sub[sub[med_col].notna() & (sub[med_col] > 0)]
            if len(sub) == 0:
                continue

            x = sub["reach_center"].values
            y = sub[med_col].values
            lo_err = np.maximum(y - sub[lo_col].values, 0)
            hi_err = np.maximum(sub[hi_col].values - y, 0)

            ax.errorbar(
                x, y, yerr=[lo_err, hi_err],
                fmt=marker, color=colour, markersize=7,
                capsize=4, capthick=1.5, linewidth=1.5,
                label=display, zorder=3,
            )
            ax.plot(x, y, color=colour, linewidth=1.5, alpha=0.5, zorder=2)

            # Hoeffding bound overlay on normalised error panel
            if col_idx == 1:
                r_line = np.logspace(np.log10(50), 4, 200)
                eps_bound = np.array(
                    [
                        ew_predicted_epsilon(
                            compute_hoeffding_p(r, paper_eps, HOEFFDING_DELTA) * r, r,
                        )
                        for r in r_line
                    ]
                )
                ax.plot(
                    r_line, eps_bound, color=colour, linewidth=2,
                    linestyle="--", alpha=0.5,
                    label=rf"Hoeffding ($\varepsilon$={paper_eps})",
                    zorder=1,
                )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Per-Node Reach")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.legend(loc="upper right" if col_idx == 1 else "lower right", fontsize=8)
        ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()

    output_path = FIGURES_DIR / "fig3_error_crossover.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("=" * 70)
    print("01_fit_rank_model.py - Synthetic Data Analysis & Figure Generation")
    print("=" * 70)

    print(f"\nPaper defaults:")
    print(f"  Closeness epsilon:   {PAPER_EPSILON_CLOSENESS}")
    print(f"  Betweenness epsilon: {PAPER_EPSILON_BETWEENNESS}")
    print(f"  Delta:               {HOEFFDING_DELTA}")
    print("  Both use: Hoeffding bound + spatial source_indices + IPW scaling")

    # Load data
    df = load_synthetic_data()

    # Node-level accuracy assessment
    print("\n" + "-" * 50)
    node_acc = evaluate_node_level_accuracy(df)
    print_reach_bin_summary(node_acc)

    # Generate figures
    print("\n" + "-" * 50)
    print("Generating figures...")
    generate_fig1_headline(df)
    generate_fig3_error_crossover(node_acc, df)

    # Summary of sampling budgets at paper epsilons
    for metric_label, display_name, paper_eps in [
        ("harmonic", "CLOSENESS", PAPER_EPSILON_CLOSENESS),
        ("betweenness", "BETWEENNESS", PAPER_EPSILON_BETWEENNESS),
    ]:
        print("\n" + "=" * 70)
        print(f"{display_name} SAMPLING BUDGETS AT EPSILON={paper_eps}")
        print("=" * 70)
        print(f"\n{'Reach':>10} | {'Hoeffding p':>12} | {'Speedup (1/p)':>14}")
        print("-" * 42)

        for reach in [100, 300, 500, 1000, 2000, 5000, 10000, 50000]:
            p = compute_hoeffding_p(reach, epsilon=paper_eps, delta=HOEFFDING_DELTA)
            speedup = 1 / p if 0 < p < 1 else 1.0
            print(f"{reach:>10,} | {p:>11.1%} | {speedup:>13.1f}x")

    print("\n" + "=" * 70)
    print("ALL OUTPUTS")
    print("=" * 70)
    print(f"  1. {FIGURES_DIR / 'fig1_headline.pdf'}")
    print(f"  2. {FIGURES_DIR / 'fig3_error_crossover.pdf'}")

    return 0


if __name__ == "__main__":
    exit(main())
