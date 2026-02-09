#!/usr/bin/env python
"""
01_fit_rank_model.py - Analyse synthetic sampling data and generate figures.

Loads the synthetic sampling cache and generates:
  1. Figure 1 (headline): Problem illustration — why sampling is needed
  2. Figure 3 (error crossover): Precision scales with importance
  3. Node-level and top-k precision diagnostics

All sampling probabilities are computed from the Hoeffding/EW model:
    k = log(2r / delta) / (2 * epsilon^2),  p = min(1, k / r)

Requires:
    - .cache/sampling_analysis_{CACHE_VERSION}.pkl (from 00_generate_cache.py)

Outputs:
    - paper/figures/fig1_headline.pdf: Problem illustration (main body)
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
    HOEFFDING_EPSILON,
    compute_hoeffding_p,
    ew_predicted_epsilon,
)

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
# NODE-LEVEL ACCURACY ASSESSMENT
# =============================================================================


def evaluate_node_level_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate per-node accuracy at the Hoeffding model's recommended p.

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
                model_p = compute_hoeffding_p(reach)
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
            norm_errors = errors / reaches if metric == "harmonic" else errors / (reaches * (reaches - 1))

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


def evaluate_topk_at_model_p(df: pd.DataFrame) -> pd.DataFrame:
    """Extract top-k precision (top 10% overlap) at the Hoeffding model's recommended p.

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
                model_p = compute_hoeffding_p(reach)

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


def generate_fig3_error_crossover(node_acc: pd.DataFrame):
    """Figure 3: Error crossover — absolute error grows with reach, normalised error shrinks.

    1×2 layout with continuous log-scale reach axis.  Nodes are pooled across
    all (topology, distance) configs and binned by absolute per-node reach.
    Left: absolute MAE trending up with reach.
    Right: normalised MAE trending down with reach, with localised EW bound overlay.
    Both metrics shown (betweenness red squares, harmonic blue circles).
    """
    print("\nGenerating Figure 3: Error crossover...")

    colour_bet = "#B2182B"
    colour_har = "#2166AC"

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

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

        # Overlay localised EW bound on normalised error panel
        if col_idx == 1:
            r_line = np.logspace(1, 4.2, 200)
            eps_line = np.array([ew_predicted_epsilon(compute_hoeffding_p(r) * r, r) for r in r_line])
            ax.plot(
                r_line,
                eps_line,
                color="grey",
                linewidth=2,
                linestyle="--",
                label="Hoeffding bound",
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
            ax.legend(loc="center right", fontsize=8)
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
    import math

    print("=" * 70)
    print("01_fit_rank_model.py - Synthetic Data Analysis & Figure Generation")
    print("=" * 70)

    print("\nModel: k = log(2r/δ) / (2ε²), p = min(1, k/r)")
    print(f"  ε = {HOEFFDING_EPSILON}, δ = {HOEFFDING_DELTA}")

    # Load data
    df = load_synthetic_data()

    # Node-level accuracy assessment
    print("\n" + "-" * 50)
    node_acc = evaluate_node_level_accuracy(df)
    print_reach_bin_summary(node_acc)

    # Top-k precision assessment
    tk = evaluate_topk_at_model_p(df)
    print_topk_summary(tk)

    # Generate figures
    print("\n" + "-" * 50)
    print("Generating figures...")
    generate_fig1_headline(df)
    generate_fig3_error_crossover(node_acc)

    # Example calculations
    print("\n" + "=" * 70)
    print("HOEFFDING MODEL — KEY VALUES")
    print("=" * 70)
    print(f"\n{'Reach':>10} | {'k':>10} | {'p':>10} | {'Speedup':>10}")
    print("-" * 50)

    for reach in [100, 300, 500, 1000, 2000, 5000, 10000, 50000]:
        k_val = math.log(2 * reach / HOEFFDING_DELTA) / (2 * HOEFFDING_EPSILON**2)
        p = compute_hoeffding_p(reach)
        speedup = 1 / p if p > 0 else float("inf")
        print(f"{reach:>10} | {k_val:>10.0f} | {p:>9.1%} | {speedup:>9.1f}x")

    print("\n" + "=" * 70)
    print("ALL OUTPUTS")
    print("=" * 70)
    print(f"  1. {FIGURES_DIR / 'fig1_headline.pdf'}")
    print(f"  2. {FIGURES_DIR / 'fig3_error_crossover.pdf'}")

    return 0


if __name__ == "__main__":
    exit(main())
