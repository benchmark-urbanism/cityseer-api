#!/usr/bin/env python
"""
01_analyse_synthetic.py - Analyse synthetic sampling results and generate figures.

Loads the synthetic sampling cache and generates:
  - fig1_rho_vs_epsilon.pdf: Ranking accuracy (Spearman rho) vs epsilon, both metrics.
  - fig3_hoeffding_bound.pdf: Hoeffding bound — epsilon determines p and speedup.
  - fig4_speedup.pdf: Speedup at paper default epsilons.
  - fig5_error_vs_reach.pdf: Absolute and normalised error vs reach.
  - tab1_ew_comparison.tex: Required k and p across epsilon values.

Requires:
    - .cache/sampling_analysis_{CACHE_VERSION}.pkl (from 00_generate_cache.py)
"""

import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cityseer.config import HOEFFDING_EPSILON, compute_distance_p, compute_hoeffding_p
from utilities import (
    CACHE_DIR,
    CACHE_VERSION,
    FIGURES_DIR,
    HOEFFDING_DELTA,
    TABLES_DIR,
    compute_hoeffding_eff_n,
    ew_predicted_epsilon,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

SYNTHETIC_CACHE = CACHE_DIR / f"sampling_analysis_{CACHE_VERSION}.pkl"

PAPER_EPSILON_CLOSENESS = 0.05
PAPER_EPSILON_BETWEENNESS = 0.05

EPSILON_TARGETS = [0.05, 0.1, 0.2]

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
# NODE-LEVEL ACCURACY (supports fig5)
# =============================================================================


def _pool_node_errors(df_subset: pd.DataFrame, metric_label: str) -> list[dict]:
    """Pool node-level errors and bin by absolute reach.

    Normalisation:
      - closeness: error / reach  (metric proportional to reach)
      - betweenness: error / true_value  (relative error; metric proportional to reach^2)
    """
    reach_pool: list[np.ndarray] = []
    error_pool: list[np.ndarray] = []
    true_pool: list[np.ndarray] = []

    for _, row in df_subset.iterrows():
        node_reach = np.asarray(row["node_reach"])
        node_true = np.asarray(row["node_true_vals"])
        node_est = np.asarray(row["node_est_vals"])
        abs_error = np.abs(node_true - node_est)

        mask = (node_true > 0) & np.isfinite(node_true) & np.isfinite(node_est) & (node_reach > 0)
        reach_pool.append(node_reach[mask])
        error_pool.append(abs_error[mask])
        true_pool.append(node_true[mask])

    if not reach_pool:
        return []

    all_reach = np.concatenate(reach_pool)
    all_error = np.concatenate(error_pool)
    all_true = np.concatenate(true_pool)

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
        trues = all_true[mask]

        if metric_label == "betweenness":
            norm_errors = errors / np.maximum(trues, 1e-10)
        else:
            norm_errors = errors / reaches
        valid_norm = norm_errors[np.isfinite(norm_errors)]

        rows.append(
            {
                "metric": metric_label,
                "reach_lo": lo,
                "reach_hi": hi,
                "reach_center": np.sqrt(lo * hi),
                "median_mae": float(np.median(errors)),
                "mae_q25": float(np.percentile(errors, 25)),
                "mae_q75": float(np.percentile(errors, 75)),
                "median_nrmse": float(np.median(valid_norm)) if len(valid_norm) > 0 else np.nan,
                "nrmse_q25": float(np.percentile(valid_norm, 25)) if len(valid_norm) > 0 else np.nan,
                "nrmse_q75": float(np.percentile(valid_norm, 75)) if len(valid_norm) > 0 else np.nan,
                "n_nodes": int(n),
            }
        )

    return rows


def evaluate_node_level_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate per-node accuracy at each metric's paper-default epsilon.

    Uses eps_targeted rows whose target_epsilon is closest to the paper default.
    """
    all_rows: list[dict] = []
    for metric, paper_eps in [("harmonic", PAPER_EPSILON_CLOSENESS), ("betweenness", PAPER_EPSILON_BETWEENNESS)]:
        available_eps = sorted(df[df["metric"] == metric]["target_epsilon"].dropna().unique())
        closest_eps = min(available_eps, key=lambda e: abs(e - paper_eps))
        subset = df[(df["metric"] == metric) & np.isclose(df["target_epsilon"], closest_eps)]
        print(
            f"  Node accuracy [{metric}]: paper eps={paper_eps}, "
            f"using target_epsilon={closest_eps:.3f} ({len(subset)} rows)"
        )
        all_rows.extend(_pool_node_errors(subset, metric))
    return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()


def compute_threshold_epsilons(df: pd.DataFrame, rho_target: float = 0.95) -> dict[str, float]:
    """Find the epsilon at which rho crosses rho_target for each metric.

    Reports per-topology and overall. Uses eps_targeted sweep rows only.
    """
    results = {}
    eps_df = df[(df["sweep_type"] == "eps_targeted") & (df["actual_sample_prob"] < 1.0)]
    print(f"\nEmpirical epsilon threshold (rho >= {rho_target}):")
    for metric in ["harmonic", "betweenness"]:
        subset = eps_df[eps_df["metric"] == metric]
        print(f"\n  {metric}:")

        # Per-topology breakdown
        for topo in sorted(subset["topology"].unique()):
            topo_sub = subset[subset["topology"] == topo]
            grouped = (
                topo_sub.groupby("target_epsilon")["spearman"]
                .mean()
                .reset_index()
                .sort_values("target_epsilon")
            )
            above = grouped[grouped["spearman"] >= rho_target]
            if len(above) == 0:
                print(f"    [{topo}]: never reaches rho={rho_target}")
            else:
                threshold_eps = above["target_epsilon"].max()
                threshold_rho = above.loc[above["target_epsilon"].idxmax(), "spearman"]
                print(f"    [{topo}]: threshold eps={threshold_eps:.3f}  (rho={threshold_rho:.3f})")

        # Overall mean
        grouped_all = (
            subset.groupby("target_epsilon")["spearman"]
            .mean()
            .reset_index()
            .sort_values("target_epsilon")
        )
        above_all = grouped_all[grouped_all["spearman"] >= rho_target]
        if len(above_all) == 0:
            print(f"    [overall mean]: never reaches rho={rho_target}")
            results[metric] = np.nan
        else:
            threshold_eps = above_all["target_epsilon"].max()
            threshold_rho = above_all.loc[above_all["target_epsilon"].idxmax(), "spearman"]
            print(f"    [overall mean]: threshold eps={threshold_eps:.3f}  (rho={threshold_rho:.3f})")
            results[metric] = threshold_eps
    return results


# =============================================================================
# FIGURES
# =============================================================================


def generate_fig1_rho_vs_epsilon(df: pd.DataFrame):
    """Figure 1: Spearman rho vs epsilon — two panels, one per metric.

    Uses the epsilon-targeted sweep rows (sweep_type="eps_targeted") where each
    row was sampled at exactly the p required to achieve a given target_epsilon.
    Groups by target_epsilon and averages across topologies and distances, giving
    a clean, unconfounded (epsilon, rho) relationship with a 95% CI band.
    Left panel: closeness. Right panel: betweenness.
    """
    print("\nGenerating Figure 1: rho vs epsilon...")

    eps_df = df[df["sweep_type"] == "eps_targeted"].copy() if "sweep_type" in df.columns else pd.DataFrame()

    if eps_df.empty:
        print("  WARNING: No eps_targeted rows found — cache needs regeneration with --force")
        print("  Falling back to prob-sweep data (confounded)")
        eps_df = df.copy()
        eps_df["target_epsilon"] = eps_df["epsilon"]

    panels = [
        ("harmonic", "Closeness", "A)", "#2166AC", PAPER_EPSILON_CLOSENESS),
        ("betweenness", "Betweenness", "B)", "#B2182B", PAPER_EPSILON_BETWEENNESS),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for panel_idx, (metric_label, metric_display, panel_label, colour, paper_eps) in enumerate(panels):
        ax = axes[panel_idx]

        subset = eps_df[eps_df["metric"] == metric_label]
        if len(subset) == 0:
            ax.set_title(f"{panel_label} {metric_display} (no data)")
            continue

        grouped = (
            subset.groupby("target_epsilon")["spearman"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .sort_values("target_epsilon")
        )
        grouped["ci"] = 1.96 * grouped["std"] / np.sqrt(grouped["count"])

        ax.fill_between(
            grouped["target_epsilon"],
            (grouped["mean"] - grouped["ci"]).clip(0, 1),
            (grouped["mean"] + grouped["ci"]).clip(0, 1),
            color=colour,
            alpha=0.2,
        )
        ax.plot(
            grouped["target_epsilon"],
            grouped["mean"],
            "o-",
            color=colour,
            markersize=5,
            linewidth=1.8,
        )

        ax.axhline(0.95, color="green", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.text(0.02, 0.955, r"$\rho$=0.95", fontsize=8, color="green", ha="left")

        ax.axvline(paper_eps, color="grey", linestyle=":", linewidth=1.2, alpha=0.8)
        ax.text(paper_eps + 0.002, 0.35, rf"$\varepsilon$={paper_eps}", fontsize=8, color="grey", ha="left")

        # Distance-based deterministic method: overlay achieved rho per distance
        det_eps = HOEFFDING_EPSILON
        det_subset = df[(df["metric"] == metric_label) & (df["sweep_type"] == "distance_based")]
        if not det_subset.empty:
            det_grouped = det_subset.groupby("distance").agg(
                mean_reach=("mean_reach", "mean"),
                spearman=("spearman", "mean"),
            ).reset_index()
            # Compute the effective epsilon for each distance using canonical grid model
            det_grouped["eff_eps"] = det_grouped["distance"].apply(
                lambda d: det_eps  # by design: we target this epsilon
            )
            ax.scatter(
                [det_eps] * len(det_grouped),
                det_grouped["spearman"],
                color=colour,
                marker="*",
                s=120,
                zorder=5,
                label="Distance-based method",
            )
            # Show mean achieved rho as a horizontal span
            det_mean_rho = det_grouped["spearman"].mean()
            ax.axhline(det_mean_rho, color=colour, linestyle="-.", linewidth=1.0, alpha=0.5)

        ax.set_xlabel(r"Target $\varepsilon$")
        ax.set_ylabel(r"Spearman $\rho$ (ranking accuracy)")
        ax.set_title(f"{panel_label} {metric_display}")
        ax.set_xlim(left=0)
        ax.set_ylim(0.3, 1.02)
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(r"Ranking Accuracy vs $\varepsilon$", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    output_path = FIGURES_DIR / "fig1_rho_vs_epsilon.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


def generate_fig3_hoeffding_bound():
    """Figure 3: The Hoeffding bound — how epsilon determines p and speedup.

    Panel A: Sample probability p vs reach, for several epsilon values.
    Panel B: Resulting speedup (1/p) vs reach.
    Paper defaults shown as solid coloured lines; other epsilons as grey dashes.
    """
    print("\nGenerating Figure 3: Hoeffding bound (epsilon -> p)...")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    reach_range = np.logspace(1.5, 5.5, 500)
    # (epsilon, colour, linewidth, linestyle, label)
    epsilon_specs = [
        (PAPER_EPSILON_BETWEENNESS, "#B2182B", 2.0, "-",  rf"$\varepsilon$={PAPER_EPSILON_BETWEENNESS} (betweenness)"),
        (PAPER_EPSILON_CLOSENESS,   "#2166AC", 2.0, "-",  rf"$\varepsilon$={PAPER_EPSILON_CLOSENESS} (closeness)"),
        (0.15,                      "#969696", 1.2, "--", r"$\varepsilon$=0.15"),
        (0.2,                       "#636363", 1.2, "--", r"$\varepsilon$=0.20"),
    ]

    ax = axes[0]
    for eps, colour, lw, ls, label in epsilon_specs:
        p_values = [compute_hoeffding_p(r, eps, HOEFFDING_DELTA) * 100 for r in reach_range]
        ax.plot(reach_range, p_values, linestyle=ls, color=colour, linewidth=lw, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("Network Reach ($r$)")
    ax.set_ylabel("Required Sample Probability $p$ (%)")
    ax.set_title(r"A) $\varepsilon \to p$")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for eps, colour, lw, ls, label in epsilon_specs:
        speedup_values = [1.0 / compute_hoeffding_p(r, eps, HOEFFDING_DELTA) for r in reach_range]
        ax.plot(reach_range, speedup_values, linestyle=ls, color=colour, linewidth=lw, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Network Reach ($r$)")
    ax.set_ylabel("Speedup (1/$p$)")
    ax.set_title(r"B) Resulting speedup")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    fig.suptitle(r"Hoeffding Bound: $\varepsilon \to$ Sampling Budget", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    output_path = FIGURES_DIR / "fig3_hoeffding_bound.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


def generate_fig4_speedup(df: pd.DataFrame):
    """Figure 4: Speedup at paper default epsilons.

    Both metrics use epsilon=0.05 (unified paper default).
    """
    print("\nGenerating Figure 4: Speedup...")

    fig, ax = plt.subplots(figsize=(7, 5))

    all_distances = sorted(df["distance"].unique())
    bar_data = []

    for dist in all_distances:
        row_data: dict = {"distance": dist, "speedup_closeness": 1.0, "speedup_betweenness": 1.0}

        for metric, paper_eps, col in [
            ("harmonic", PAPER_EPSILON_CLOSENESS, "speedup_closeness"),
            ("betweenness", PAPER_EPSILON_BETWEENNESS, "speedup_betweenness"),
        ]:
            subset = df[(df["metric"] == metric) & (df["distance"] == dist)]
            if len(subset) > 0:
                mean_reach = subset["mean_reach"].mean()
                p = compute_hoeffding_p(mean_reach, epsilon=paper_eps, delta=HOEFFDING_DELTA)
                row_data[col] = 1.0 / p if 0 < p < 1.0 else 1.0

        bar_data.append(row_data)

    bar_df = pd.DataFrame(bar_data)
    x = np.arange(len(bar_df))
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
    ax.set_ylabel("Speedup Factor (1/$p$)")
    ax.set_title("Speedup at paper default epsilons")
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

    plt.tight_layout()

    output_path = FIGURES_DIR / "fig4_speedup.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


def generate_fig5_error_vs_reach(node_acc: pd.DataFrame):
    """Figure 5: Absolute error grows with reach; normalised error bounded by epsilon."""
    print("\nGenerating Figure 5: Error vs reach...")

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
            ("harmonic", colour_har, "o", "Closeness", PAPER_EPSILON_CLOSENESS),
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

            if col_idx == 1:
                r_line = np.logspace(np.log10(50), 4, 200)
                eps_bound = np.array(
                    [
                        ew_predicted_epsilon(compute_hoeffding_p(r, paper_eps, HOEFFDING_DELTA) * r, r)
                        for r in r_line
                    ]
                )
                ax.plot(
                    r_line, eps_bound, color=colour, linewidth=2,
                    linestyle="--", alpha=0.5,
                    label=rf"Hoeffding bound ($\varepsilon$={paper_eps})",
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

    output_path = FIGURES_DIR / "fig5_error_vs_reach.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


# =============================================================================
# TABLE
# =============================================================================


def generate_tab1_ew_comparison():
    """Tab 1: Required k and p across epsilon values, with paper defaults noted in caption."""
    print("\nGenerating Table 1: EW comparison...")

    reach_values = [100, 500, 1000, 3000, 10000, 30000]

    def fmt_int(n):
        return f"{n:,}".replace(",", "{,}")

    def fmt_pct(p):
        return f"{p * 100:.1f}\\%"

    rows = []
    for reach in reach_values:
        row = {"reach": reach}
        for eps in EPSILON_TARGETS:
            row[f"k_{eps}"] = compute_hoeffding_eff_n(reach, eps, HOEFFDING_DELTA)
            row[f"p_{eps}"] = compute_hoeffding_p(reach, eps, HOEFFDING_DELTA)
        rows.append(row)

    n_eps = len(EPSILON_TARGETS)
    col_spec = "r" + "rr" * n_eps

    latex = r"""\begin{table}[htbp]
\centering
\caption{Required sample sizes under the Hoeffding/EW bound at different additive
  error tolerances ($\delta = 0.1$): $k = \log(2r/\delta) / (2\varepsilon^2)$,
  $p = \min(1, k/r)$. Paper default: $\varepsilon=0.05$ for both metrics.}
\label{tab:ew_comparison}
\small
"""
    latex += f"\\begin{{tabular}}{{@{{}}{col_spec}@{{}}}}\n"
    latex += "\\toprule\n"

    # Top header: epsilon group labels
    latex += "\\textbf{Reach}"
    for eps in EPSILON_TARGETS:
        latex += f" & \\multicolumn{{2}}{{c}}{{$\\varepsilon={eps}$}}"
    latex += " \\\\\n"

    # Sub-header: k and p columns
    latex += "         "
    for _ in EPSILON_TARGETS:
        latex += " & $k$ & $p$"
    latex += " \\\\\n"
    latex += "\\midrule\n"

    for row in rows:
        latex += fmt_int(row["reach"])
        for eps in EPSILON_TARGETS:
            k = int(row[f"k_{eps}"])
            p = row[f"p_{eps}"]
            if p >= 1.0:
                latex += " & \\multicolumn{2}{c}{exact}"
            else:
                latex += f" & {fmt_int(k)} & {fmt_pct(p)}"
        latex += " \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    output_path = TABLES_DIR / "tab1_ew_comparison.tex"
    with open(output_path, "w") as f:
        f.write(latex)
    print(f"  Saved: {output_path}")


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("=" * 70)
    print("01_analyse_synthetic.py")
    print("=" * 70)
    print(f"  Paper defaults: closeness eps={PAPER_EPSILON_CLOSENESS}, betweenness eps={PAPER_EPSILON_BETWEENNESS}")

    df = load_synthetic_data()

    compute_threshold_epsilons(df)

    node_acc = evaluate_node_level_accuracy(df)

    generate_fig1_rho_vs_epsilon(df)
    generate_fig3_hoeffding_bound()
    generate_fig4_speedup(df)
    generate_fig5_error_vs_reach(node_acc)
    generate_tab1_ew_comparison()

    print("\n" + "=" * 70)
    print("OUTPUTS")
    print("=" * 70)
    print(f"  {FIGURES_DIR / 'fig1_rho_vs_epsilon.pdf'}")
    print(f"  {FIGURES_DIR / 'fig3_hoeffding_bound.pdf'}")
    print(f"  {FIGURES_DIR / 'fig4_speedup.pdf'}")
    print(f"  {FIGURES_DIR / 'fig5_error_vs_reach.pdf'}")
    print(f"  {TABLES_DIR / 'tab1_ew_comparison.tex'}")

    return 0


if __name__ == "__main__":
    exit(main())
