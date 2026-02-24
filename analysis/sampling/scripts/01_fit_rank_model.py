#!/usr/bin/env python
"""
01_fit_rank_model.py - Analyse synthetic sampling data and generate figures.

Loads the synthetic sampling cache and generates:
  1. Figure 1 (headline): Sampling opportunity — accuracy vs epsilon for both
     closeness (Hoeffding source sampling) and betweenness (Hoeffding path
     sampling), plus theoretical speedup at the paper's default epsilon.
  2. Figure 3 (error crossover): Precision scales with importance — absolute
     and normalised MAE by per-node reach bin, with theoretical bound overlays.

The cache sweep variable is epsilon (error tolerance).
  - Closeness rows contain `sample_prob` derived from Hoeffding.
  - Betweenness rows contain `n_samples` derived from Hoeffding.

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
from utilities import (
    CACHE_DIR,
    CACHE_VERSION,
    FIGURES_DIR,
    HOEFFDING_DELTA,
    compute_hoeffding_betw_budget,
    compute_hoeffding_p,
    ew_predicted_epsilon,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

SYNTHETIC_CACHE = CACHE_DIR / f"sampling_analysis_{CACHE_VERSION}.pkl"

# Paper default epsilon (matches \hoeffdingEpsilon in model_macros.tex)
PAPER_EPSILON = 0.1

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
    print(f"  Metrics: {df['metric'].unique().tolist()}")
    print(f"  Epsilons: {sorted(df['epsilon'].unique())}")
    return df


# =============================================================================
# NODE-LEVEL ACCURACY ASSESSMENT
# =============================================================================


def evaluate_node_level_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate per-node accuracy at the paper's default epsilon.

    Pools all nodes across all (topology, distance) configs at epsilon = PAPER_EPSILON,
    then bins by absolute per-node reach. This avoids the within-config quartile
    pre-aggregation that confounds boundary nodes from high-reach configs with all
    nodes from low-reach configs.
    """
    reach_pool = {"harmonic": [], "betweenness": []}
    error_pool = {"harmonic": [], "betweenness": []}

    for metric in ["harmonic", "betweenness"]:
        df_m = df[(df["metric"] == metric) & (df["epsilon"] == PAPER_EPSILON)]
        for _, row in df_m.iterrows():
            node_reach = np.asarray(row["node_reach"])
            node_true = np.asarray(row["node_true_vals"])
            node_est = np.asarray(row["node_est_vals"])
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

            rows.append(
                {
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
                }
            )

    return pd.DataFrame(rows)


def print_reach_bin_summary(node_acc: pd.DataFrame):
    """Print per-reach-bin error diagnostics at the paper's default epsilon."""
    print("\n" + "=" * 70)
    print(f"NODE-LEVEL ERROR DIAGNOSTICS BY ABSOLUTE REACH (epsilon={PAPER_EPSILON})")
    print("=" * 70)
    print("  Nodes pooled across all configs at paper epsilon,")
    print("  binned by absolute per-node reach.")

    for metric in ["betweenness", "harmonic"]:
        subset = node_acc[node_acc["metric"] == metric]
        if len(subset) == 0:
            continue

        print(f"\n  {metric.upper()}")
        print(f"  {'Reach bin':<16} {'N nodes':>10} {'Med MAE':>12} {'Med NRMSE':>12}")
        print("  " + "-" * 52)

        for _, row in subset.iterrows():
            label = f"{int(row['reach_lo'])}-{int(row['reach_hi'])}"
            print(f"  {label:<16} {row['n_nodes']:>10,} {row['median_mae']:>12.2f} {row['median_nrmse']:>12.6f}")

    print("\n  Crossover: absolute MAE increases with reach (larger errors")
    print("  at high-reach nodes), but normalised error decreases")
    print("  (precision scales with importance).")


# =============================================================================
# FIGURE GENERATION
# =============================================================================


def generate_fig1_headline(df: pd.DataFrame):
    """Figure 1: Sampling opportunity — accuracy vs epsilon and theoretical speedup.

    Panel A: Spearman rho vs epsilon for both closeness and betweenness at
    representative distances. Solid lines for closeness, dashed for betweenness.
    X-axis reversed (small epsilon = high accuracy on right).

    Panel B: Theoretical speedup vs distance at epsilon = PAPER_EPSILON.
    Paired bars: closeness (1/p) and betweenness (n_live / n_samples).
    """
    print("\nGenerating Figure 1: Headline comparison...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    representative_dists = [500, 1000, 2000, 4000]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    # ------------------------------------------------------------------
    # Panel A: rho vs epsilon for both metrics
    # ------------------------------------------------------------------
    ax = axes[0]

    for dist, color in zip(representative_dists, colors, strict=True):
        # Closeness (solid lines)
        subset_h = df[(df["metric"] == "harmonic") & (df["distance"] == dist)]
        if len(subset_h) > 0:
            grouped_h = (
                subset_h.groupby("epsilon")
                .agg({"spearman": "mean", "mean_reach": "mean"})
                .reset_index()
                .sort_values("epsilon")
            )
            reach = grouped_h["mean_reach"].iloc[0]
            ax.plot(
                grouped_h["epsilon"],
                grouped_h["spearman"],
                "o-",
                color=color,
                markersize=4,
                linewidth=1.5,
                label=f"{dist}m closeness (r={reach:.0f})",
            )

        # Betweenness (dashed lines)
        subset_b = df[(df["metric"] == "betweenness") & (df["distance"] == dist)]
        if len(subset_b) > 0:
            grouped_b = (
                subset_b.groupby("epsilon")
                .agg({"spearman": "mean", "mean_reach": "mean"})
                .reset_index()
                .sort_values("epsilon")
            )
            ax.plot(
                grouped_b["epsilon"],
                grouped_b["spearman"],
                "s--",
                color=color,
                markersize=4,
                linewidth=1.5,
                alpha=0.8,
                label=f"{dist}m betweenness",
            )

    ax.axhline(0.95, color="green", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.text(0.02, 0.955, r"target: $\rho$=0.95", fontsize=9, color="green", ha="left")

    ax.set_xlabel(r"$\varepsilon$ (error tolerance)")
    ax.set_ylabel(r"Spearman $\rho$ (ranking accuracy)")
    ax.set_title(r"A) Accuracy vs $\varepsilon$")
    ax.set_xlim(0.55, 0.005)  # Reversed: large epsilon on left, small on right
    ax.set_xscale("log")
    ax.set_ylim(0.5, 1.02)
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Panel B: Theoretical speedup at PAPER_EPSILON
    # ------------------------------------------------------------------
    ax = axes[1]

    all_distances = sorted(df["distance"].unique())
    bar_data = []

    for dist in all_distances:
        # Closeness speedup: 1/p (fraction of sources sampled)
        subset_h = df[(df["metric"] == "harmonic") & (df["distance"] == dist) & (df["epsilon"] == PAPER_EPSILON)]
        if len(subset_h) > 0:
            mean_reach_h = subset_h["mean_reach"].mean()
            p = compute_hoeffding_p(mean_reach_h, epsilon=PAPER_EPSILON, delta=HOEFFDING_DELTA)
            speedup_h = 1.0 / p if p > 0 and p < 1.0 else 1.0
        else:
            speedup_h = None

        # Betweenness speedup: mean_reach / m (per-source pair reduction)
        subset_b = df[(df["metric"] == "betweenness") & (df["distance"] == dist) & (df["epsilon"] == PAPER_EPSILON)]
        if len(subset_b) > 0:
            mean_reach_b = subset_b["mean_reach"].mean()
            m = compute_hoeffding_betw_budget(mean_reach_b, epsilon=PAPER_EPSILON, delta=HOEFFDING_DELTA)
            if m is not None and m > 0:
                speedup_b = mean_reach_b / m
            else:
                speedup_b = 1.0
        else:
            speedup_b = None

        if speedup_h is not None or speedup_b is not None:
            bar_data.append(
                {
                    "distance": dist,
                    "speedup_closeness": speedup_h if speedup_h is not None else 1.0,
                    "speedup_betweenness": speedup_b if speedup_b is not None else 1.0,
                }
            )

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
        label="Closeness (1/p)",
    )
    bars_b = ax.bar(
        x + width / 2,
        bar_df["speedup_betweenness"],
        width,
        color="#B2182B",
        alpha=0.8,
        label="Betweenness (reach/m)",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}m" for d in bar_df["distance"]], rotation=45, ha="right")
    ax.set_xlabel("Analysis Distance")
    ax.set_ylabel("Theoretical Speedup")
    ax.set_title(rf"B) Speedup at $\varepsilon$={PAPER_EPSILON}")
    ax.set_yscale("log")
    ax.axhline(1.0, color="grey", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y", which="both")

    # Add value labels on bars
    for bar_set in [bars_h, bars_b]:
        for bar in bar_set:
            height = bar.get_height()
            if height > 1.05:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height * 1.1,
                    f"{height:.1f}x",
                    ha="center",
                    fontsize=7,
                )

    fig.suptitle("Sampling-Based Centrality: The Opportunity", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    output_path = FIGURES_DIR / "fig1_headline.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


def generate_fig3_error_crossover(node_acc: pd.DataFrame, df: pd.DataFrame):
    """Figure 3: Error crossover — absolute error grows with reach, normalised error shrinks.

    1x2 layout with continuous log-scale reach axis. Nodes are pooled across
    all (topology, distance) configs at epsilon = PAPER_EPSILON and binned
    by absolute per-node reach.

    Panel A: Absolute MAE trending up with reach.
    Panel B: Normalised MAE trending down with reach, with Hoeffding bound
             overlay for closeness and R-K bound overlay for betweenness.

    Both metrics shown (betweenness red squares, harmonic blue circles).
    """
    print("\nGenerating Figure 3: Error crossover...")

    colour_bet = "#B2182B"
    colour_har = "#2166AC"

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Bound overlays use the actual Hoeffding/R-K budget at each reach level

    for col_idx, (med_col, lo_col, hi_col, ylabel, title) in enumerate(
        [
            ("median_mae", "mae_q25", "mae_q75", "Median Absolute Error", "A) Absolute Error"),
            ("median_nrmse", "nrmse_q25", "nrmse_q75", "Median Normalised Error", "B) Normalised Error"),
        ]
    ):
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
                x,
                y,
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

        # Overlay theoretical bounds on normalised error panel
        if col_idx == 1:
            r_line = np.logspace(np.log10(50), 4, 200)

            # Hoeffding bound for closeness at PAPER_EPSILON:
            # For each reach r, compute p from the Hoeffding budget, then
            # the predicted epsilon from that effective sample size.
            eps_hoeffding = np.array(
                [
                    ew_predicted_epsilon(
                        compute_hoeffding_p(r, PAPER_EPSILON, HOEFFDING_DELTA) * r,
                        r,
                    )
                    for r in r_line
                ]
            )
            ax.plot(
                r_line,
                eps_hoeffding,
                color="grey",
                linewidth=2,
                linestyle="--",
                label=rf"Hoeffding bound ($\varepsilon$={PAPER_EPSILON})",
                zorder=1,
            )

            # Hoeffding bound for betweenness at PAPER_EPSILON:
            # For each reach r, compute m from Hoeffding budget, then predicted epsilon.
            eps_betw = np.array(
                [
                    ew_predicted_epsilon(
                        compute_hoeffding_betw_budget(r, PAPER_EPSILON, HOEFFDING_DELTA) or 1,
                        r,
                    )
                    for r in r_line
                ]
            )
            ax.plot(
                r_line,
                eps_betw,
                color="#B2182B",
                linewidth=2,
                linestyle=":",
                alpha=0.7,
                label=rf"Betw. Hoeffding bound ($\varepsilon$={PAPER_EPSILON})",
                zorder=1,
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Per-Node Reach")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        if col_idx == 0:
            ax.legend(loc="lower right", fontsize=9)
        else:
            ax.legend(loc="upper right", fontsize=8)
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

    print(f"\nPaper default: epsilon = {PAPER_EPSILON}, delta = {HOEFFDING_DELTA}")
    print("  Closeness: Hoeffding source sampling  (p from epsilon + reach)")
    print("  Betweenness: Hoeffding path sampling   (m from epsilon + reach)")

    # Load data
    df = load_synthetic_data()

    # Node-level accuracy assessment at paper epsilon
    print("\n" + "-" * 50)
    node_acc = evaluate_node_level_accuracy(df)
    print_reach_bin_summary(node_acc)

    # Generate figures
    print("\n" + "-" * 50)
    print("Generating figures...")
    generate_fig1_headline(df)
    generate_fig3_error_crossover(node_acc, df)

    # Summary of key values at paper epsilon
    print("\n" + "=" * 70)
    print(f"SAMPLING BUDGETS AT EPSILON={PAPER_EPSILON}")
    print("=" * 70)
    print(f"\n{'Reach':>10} | {'Hoeffding p':>12} | {'Speedup (1/p)':>14} | {'Hoeff m':>10} | {'Betw speedup*':>14}")
    print("-" * 70)

    for reach in [100, 300, 500, 1000, 2000, 5000, 10000, 50000]:
        p = compute_hoeffding_p(reach, epsilon=PAPER_EPSILON, delta=HOEFFDING_DELTA)
        speedup_h = 1 / p if 0 < p < 1 else 1.0
        m = compute_hoeffding_betw_budget(reach, epsilon=PAPER_EPSILON, delta=HOEFFDING_DELTA)
        m_str = f"{m:,}" if m is not None else "N/A"
        speedup_b = reach / m if m is not None and m > 0 else float("inf")
        print(
            f"{reach:>10,} | {p:>11.1%} | {speedup_h:>13.1f}x | {m_str:>10} | {speedup_b:>13.1f}x"
        )

    print("\n  * Betw speedup = reach / m (approximate; actual depends on n_live)")

    print("\n" + "=" * 70)
    print("ALL OUTPUTS")
    print("=" * 70)
    print(f"  1. {FIGURES_DIR / 'fig1_headline.pdf'}")
    print(f"  2. {FIGURES_DIR / 'fig3_error_crossover.pdf'}")

    return 0


if __name__ == "__main__":
    exit(main())
