#!/usr/bin/env python
"""
01_analyse_synthetic.py - Analyse synthetic sampling results and generate figures.

Loads the synthetic sampling cache and generates:
  - fig1_rho_vs_epsilon.pdf: Ranking accuracy (Spearman rho) vs epsilon, both metrics.
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
from cityseer.sampling import GRID_SPACING, compute_distance_p
from utilities import (
    CACHE_DIR,
    CACHE_VERSION,
    FIGURES_DIR,
    HOEFFDING_DELTA,
    TABLES_DIR,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

SYNTHETIC_CACHE = CACHE_DIR / f"sampling_analysis_{CACHE_VERSION}.pkl"

PAPER_EPSILON_CLOSENESS = 0.06
PAPER_EPSILON_BETWEENNESS = 0.06

EPSILON_TARGETS = [0.02, 0.04, 0.06, 0.08, 0.1]

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
            grouped = topo_sub.groupby("target_epsilon")["spearman"].mean().reset_index().sort_values("target_epsilon")
            above = grouped[grouped["spearman"] >= rho_target]
            if len(above) == 0:
                print(f"    [{topo}]: never reaches rho={rho_target}")
            else:
                threshold_eps = above["target_epsilon"].max()
                threshold_rho = above.loc[above["target_epsilon"].idxmax(), "spearman"]
                print(f"    [{topo}]: threshold eps={threshold_eps:.3f}  (rho={threshold_rho:.3f})")

        # Overall mean
        grouped_all = subset.groupby("target_epsilon")["spearman"].mean().reset_index().sort_values("target_epsilon")
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
    """Figure 1: Epsilon parameter sweep across synthetic network topologies.

    Uses the epsilon-targeted sweep rows (sweep_type="eps_targeted") where each
    row was sampled at exactly the p required to achieve a given target_epsilon.
    Shows per-topology lines (averaged over distances) plus a bold mean line,
    revealing how network structure affects the epsilon-accuracy relationship.
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

    topo_styles = {
        "trellis": ("--", "s", "Trellis"),
        "tree": ("-.", "^", "Tree"),
        "linear": (":", "o", "Linear"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for panel_idx, (metric_label, metric_display, panel_label, colour, paper_eps) in enumerate(panels):
        ax = axes[panel_idx]

        subset = eps_df[eps_df["metric"] == metric_label]
        if len(subset) == 0:
            ax.set_title(f"{panel_label} {metric_display} (no data)")
            continue

        # Per-topology lines (averaged over distances)
        for topo in sorted(subset["topology"].unique()):
            topo_sub = subset[subset["topology"] == topo]
            grouped = topo_sub.groupby("target_epsilon")["spearman"].mean().reset_index().sort_values("target_epsilon")
            ls, marker, label = topo_styles.get(topo, ("-", ".", topo))
            ax.plot(
                grouped["target_epsilon"],
                grouped["spearman"],
                linestyle=ls,
                marker=marker,
                markersize=4,
                color=colour,
                alpha=0.5,
                linewidth=1.0,
                label=label,
            )

        # Bold mean line across all topologies
        grouped_mean = subset.groupby("target_epsilon")["spearman"].mean().reset_index().sort_values("target_epsilon")
        ax.plot(
            grouped_mean["target_epsilon"],
            grouped_mean["spearman"],
            "-",
            color=colour,
            linewidth=2.2,
            label="Mean",
        )

        ax.axhline(0.95, color="green", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.text(0.02, 0.955, r"$\rho$=0.95", fontsize=8, color="green", ha="left")

        ax.axvline(paper_eps, color="grey", linestyle=":", linewidth=1.2, alpha=0.8)
        ax.text(paper_eps + 0.002, 0.35, rf"$\varepsilon$={paper_eps}", fontsize=8, color="grey", ha="left")

        ax.set_xlabel(r"Target $\varepsilon$")
        ax.set_ylabel(r"Spearman $\rho$ (ranking accuracy)")
        ax.set_title(f"{panel_label} {metric_display}")
        ax.set_xlim(left=0)
        ax.set_ylim(0.3, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower left")

    fig.suptitle(
        r"$\varepsilon$ parameter sweep: ranking accuracy across synthetic topologies",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    output_path = FIGURES_DIR / "fig1_rho_vs_epsilon.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


# =============================================================================
# TABLE
# =============================================================================


def generate_tab1_ew_comparison():
    """Tab 1: Deterministic sampling schedule at standard distances across epsilon values."""
    import math

    print("\nGenerating Table 1: Distance-based sampling comparison...")

    distance_values = [500, 1000, 2000, 5000, 10000, 20000]

    def fmt_int(n):
        return f"{n:,}".replace(",", "{,}")

    def fmt_pct(p):
        return f"{p * 100:.1f}\\%"

    rows = []
    for dist in distance_values:
        r = math.pi * dist**2 / GRID_SPACING**2
        row = {"distance": dist, "canonical_reach": r}
        for eps in EPSILON_TARGETS:
            p = compute_distance_p(dist, epsilon=eps, delta=HOEFFDING_DELTA)
            k = math.log(2 * r / HOEFFDING_DELTA) / (2 * eps**2) if r > 0 else 0
            row[f"k_{eps}"] = k
            row[f"p_{eps}"] = p
        rows.append(row)

    n_eps = len(EPSILON_TARGETS)
    col_spec = "r" + "rr" * n_eps

    latex = (
        r"""\begin{table}[htbp]
\centering
\caption{Deterministic sampling schedule at standard analysis distances under different
  error tolerances ($\delta = 0.1$, $s = """
        + f"{GRID_SPACING:.0f}"
        + r"""\,$m).
  Paper default: $\varepsilon=0.06$ for both metrics.}
\label{tab:ew_comparison}
\small
"""
    )
    latex += f"\\begin{{tabular}}{{@{{}}{col_spec}@{{}}}}\n"
    latex += "\\toprule\n"

    # Top header: epsilon group labels
    latex += "\\textbf{Distance}"
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
        d_str = f"{int(row['distance'] // 1000)}\\,km" if row["distance"] >= 1000 else f"{row['distance']}\\,m"
        latex += d_str
        for eps in EPSILON_TARGETS:
            k = math.ceil(row[f"k_{eps}"])
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

    generate_fig1_rho_vs_epsilon(df)
    generate_tab1_ew_comparison()

    print("\n" + "=" * 70)
    print("OUTPUTS")
    print("=" * 70)
    print(f"  {FIGURES_DIR / 'fig1_rho_vs_epsilon.pdf'}")
    print(f"  {TABLES_DIR / 'tab1_ew_comparison.tex'}")

    return 0


if __name__ == "__main__":
    exit(main())
