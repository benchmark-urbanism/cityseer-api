#!/usr/bin/env python
"""
05_practical_guide.py - Generate practical guidance for practitioners.

Creates visual and tabular guidance for users to quickly determine
the appropriate sampling budget for their analysis. Both closeness
and betweenness use the same Hoeffding/EW framework:
    k = log(2r/delta)/(2*eps^2), p = min(1, k/r)

Default: epsilon_closeness = 0.1, epsilon_betweenness = 0.1, delta = 0.1.

Outputs:
    - paper/figures/fig4_practical_guide.pdf: Visual lookup chart
    - paper/tables/tab3_practical_lookup.tex: Lookup table (both metrics)
"""

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from utilities import (
    FIGURES_DIR,
    HOEFFDING_DELTA,
    TABLES_DIR,
    compute_hoeffding_p,
)

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

# Paper defaults
EPSILON_CLOSENESS = 0.1
EPSILON_BETWEENNESS = 0.1

# GLA scenario reaches for annotation
GLA_SCENARIOS = [
    (185, "1 km"),
    (765, "2 km"),
    (5100, "5 km"),
    (20394, "10 km"),
    (69463, "20 km"),
]


# =============================================================================
# FIGURE GENERATION
# =============================================================================


def generate_practical_figure():
    """
    Figure 4: Practical guidance chart.

    Panel A: Required sampling probability p (%) vs reach at multiple epsilons.
    Panel B: Speedup (1/p) vs reach at paper default epsilons for both metrics.
    """
    print("\nGenerating Figure 4: Practical guide...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    reach_range = np.logspace(2, 5.5, 200)

    # ------------------------------------------------------------------
    # Panel A: Required sampling fraction vs reach (both metrics)
    # ------------------------------------------------------------------
    ax = axes[0]

    epsilon_values = [0.05, 0.1, 0.15, 0.2]
    colours = ["#d73027", "#0072B2", "#fc8d59", "#91bfdb"]
    linewidths = [1.5, 2.5, 1.5, 1.5]

    for eps, colour, lw in zip(epsilon_values, colours, linewidths, strict=True):
        p_values = [compute_hoeffding_p(r, eps) * 100 for r in reach_range]
        style = "-" if eps == 0.1 else "--"
        label = r"$\varepsilon$" + f" = {eps}"
        ax.plot(reach_range, p_values, linestyle=style, color=colour, linewidth=lw, label=label)

    # Annotate GLA network reaches at default epsilon
    for reach, label in GLA_SCENARIOS:
        p = compute_hoeffding_p(reach, epsilon=0.1) * 100
        ax.plot(reach, p, "o", color="#D55E00", markersize=7, zorder=5)
        if p >= 99.5:
            y_off, va = -12, "top"
        else:
            y_off, va = 12, "bottom"
        ax.annotate(
            f"{label}\n{p:.0f}%",
            xy=(reach, p),
            xytext=(0, y_off),
            textcoords="offset points",
            fontsize=7,
            ha="center",
            va=va,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="none"),
        )

    ax.set_xscale("log")
    ax.set_xlabel("Network Reach (nodes within distance)")
    ax.set_ylabel("Required Sampling Fraction (%)")
    ax.set_title("A) Required Sampling Budget")
    ax.set_xlim(100, 300000)
    ax.set_ylim(0, 100)

    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D([0], [0], color="#0072B2", linewidth=2.5, linestyle="-", label=r"$\varepsilon$ = 0.1 (default)"),
        Line2D([0], [0], color="grey", linewidth=1.5, linestyle="--", label=r"Other $\varepsilon$"),
        Line2D([0], [0], color="#D55E00", marker="o", linestyle="none", markersize=7, label="GLA scenarios"),
    ]
    ax.legend(handles=legend_handles, loc="center right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax.text(
        0.95, 0.95,
        "Both metrics: k = log(2r/\u03b4)/(2\u03b5\u00b2), p = min(1, k/r)\n"
        f"Default: \u03b4 = {HOEFFDING_DELTA}",
        transform=ax.transAxes, fontsize=7, ha="right", va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # ------------------------------------------------------------------
    # Panel B: Speedup (1/p) vs reach at paper defaults
    # ------------------------------------------------------------------
    ax = axes[1]

    speedups_c = [1 / compute_hoeffding_p(r, epsilon=EPSILON_CLOSENESS) for r in reach_range]
    speedups_b = [1 / compute_hoeffding_p(r, epsilon=EPSILON_BETWEENNESS) for r in reach_range]

    ax.plot(reach_range, speedups_c, color="#2166AC", linewidth=2.5,
            label=rf"Closeness ($\varepsilon$={EPSILON_CLOSENESS})")
    if EPSILON_CLOSENESS != EPSILON_BETWEENNESS:
        ax.plot(reach_range, speedups_b, color="#B2182B", linewidth=2.5, linestyle="--",
                label=rf"Betweenness ($\varepsilon$={EPSILON_BETWEENNESS})")

    # Annotate GLA scenarios
    for reach, _label in GLA_SCENARIOS:
        p = compute_hoeffding_p(reach, epsilon=EPSILON_CLOSENESS)
        speedup = 1 / p
        ax.plot(reach, speedup, "o", color="#D55E00", markersize=7, zorder=5)
        ax.annotate(f"{speedup:.1f}\u00d7", xy=(reach, speedup * 1.1), fontsize=9, ha="center", va="bottom")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Network Reach")
    ax.set_ylabel("Speedup (1/p)")
    ax.set_title(r"B) Speedup at Paper Defaults")
    ax.set_xlim(100, 300000)
    ax.set_ylim(1, 150)
    ax.grid(True, alpha=0.3, which="both")

    ax.axhspan(1, 2, alpha=0.1, color="red", label="<2\u00d7 (limited)")
    ax.axhspan(2, 10, alpha=0.1, color="yellow", label="2\u201310\u00d7 (moderate)")
    ax.axhspan(10, 100, alpha=0.1, color="green", label=">10\u00d7 (significant)")
    ax.legend(loc="lower right", fontsize=8)

    fig.suptitle("Practical Guide: Adaptive Sampling", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    output_path = FIGURES_DIR / "fig4_practical_guide.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


# =============================================================================
# TABLE GENERATION
# =============================================================================


def generate_practical_table():
    """Generate LaTeX lookup table showing both closeness and betweenness budgets."""
    print("\nGenerating Table 3: Practical lookup...")

    delta = HOEFFDING_DELTA
    reaches = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]

    def fmt_int(n):
        return f"{n:,}".replace(",", "{,}")

    # If both epsilons are the same, single column; otherwise separate
    same_eps = EPSILON_CLOSENESS == EPSILON_BETWEENNESS

    if same_eps:
        latex = rf"""\begin{{table}}[htbp]
\centering
\caption{{Practical sampling lookup table ($\varepsilon = {EPSILON_CLOSENESS}$, $\delta = {delta}$).
Both closeness and betweenness use the same Hoeffding budget.}}
\label{{tab:lookup}}
\begin{{tabular}}{{rrrr}}
\toprule
\textbf{{Reach}} & \textbf{{$k$}} & \textbf{{$p$ (\%)}} & \textbf{{Speedup}} \\
\midrule
"""
        for reach in reaches:
            k = math.log(2 * reach / delta) / (2 * EPSILON_CLOSENESS**2)
            p = compute_hoeffding_p(reach, EPSILON_CLOSENESS, delta)
            speedup = 1 / p
            if p >= 1.0:
                latex += f"{fmt_int(reach)} & {k:.0f} & 100.0 & 1.0$\\times$ \\\\\n"
            else:
                latex += f"{fmt_int(reach)} & {k:.0f} & {p * 100:.1f} & {speedup:.1f}$\\times$ \\\\\n"

        latex += rf"""\bottomrule
\end{{tabular}}

\vspace{{0.5em}}
\footnotesize
\textbf{{Usage:}} Find your network reach (nodes within analysis distance).
Read off $p$ — the same budget applies to both closeness and betweenness.\\
\textbf{{Model:}} $k = \log(2r/\delta) / (2\varepsilon^2)$, $p = \min(1, k/r)$.
\end{{table}}
"""
    else:
        latex = rf"""\begin{{table}}[htbp]
\centering
\caption{{Practical sampling lookup table ($\delta = {delta}$).
Closeness $\varepsilon_c = {EPSILON_CLOSENESS}$, betweenness $\varepsilon_b = {EPSILON_BETWEENNESS}$.}}
\label{{tab:lookup}}
\begin{{tabular}}{{rrrrr}}
\toprule
\textbf{{Reach}} & \textbf{{$p_c$ (\%)}} & \textbf{{Spd$_c$}} & \textbf{{$p_b$ (\%)}} & \textbf{{Spd$_b$}} \\
\midrule
"""
        for reach in reaches:
            p_c = compute_hoeffding_p(reach, EPSILON_CLOSENESS, delta)
            p_b = compute_hoeffding_p(reach, EPSILON_BETWEENNESS, delta)

            def fmt_row(p):
                if p >= 1.0:
                    return "100.0", "1.0$\\times$"
                return f"{p * 100:.1f}", f"{1/p:.1f}$\\times$"

            pc_str, sc_str = fmt_row(p_c)
            pb_str, sb_str = fmt_row(p_b)
            latex += f"{fmt_int(reach)} & {pc_str} & {sc_str} & {pb_str} & {sb_str} \\\\\n"

        latex += rf"""\bottomrule
\end{{tabular}}

\vspace{{0.5em}}
\footnotesize
\textbf{{Model:}} $k = \log(2r/\delta) / (2\varepsilon^2)$, $p = \min(1, k/r)$.
\end{{table}}
"""

    output_path = TABLES_DIR / "tab3_practical_lookup.tex"
    with open(output_path, "w") as f:
        f.write(latex)
    print(f"  Saved: {output_path}")


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("=" * 70)
    print("05_practical_guide.py - Generating practical guidance")
    print("  Both closeness and betweenness use Hoeffding/EW")
    print("=" * 70)

    print(f"\nModel: k = log(2r/delta) / (2*eps^2), p = min(1, k/r)")
    print(f"  Closeness eps:   {EPSILON_CLOSENESS}")
    print(f"  Betweenness eps: {EPSILON_BETWEENNESS}")
    print(f"  Delta:           {HOEFFDING_DELTA}")

    generate_practical_figure()
    generate_practical_table()

    # Print quick reference
    print("\n" + "=" * 70)
    print("QUICK REFERENCE")
    print("=" * 70)

    header = f"{'Reach':>10} | {'k':>8} | {'p_c':>7} | {'Spd_c':>7} | {'p_b':>7} | {'Spd_b':>7}"
    print(f"\n{header}")
    print("-" * len(header))

    delta = HOEFFDING_DELTA
    for reach in [100, 500, 1000, 5000, 10000, 50000]:
        p_c = compute_hoeffding_p(reach, EPSILON_CLOSENESS, delta)
        p_b = compute_hoeffding_p(reach, EPSILON_BETWEENNESS, delta)
        k = math.log(2 * reach / delta) / (2 * EPSILON_CLOSENESS**2)
        print(f"{reach:>10} | {k:>8.0f} | {p_c:>6.1%} | {1/p_c:>6.1f}x | {p_b:>6.1%} | {1/p_b:>6.1f}x")

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("\n1. No fitted parameters: eps and delta are user-chosen")
    print("2. Both metrics use the same framework: Hoeffding + spatial source selection")
    print("3. At low reach, p -> 1.0 (exact computation); speedup grows with reach")
    print("4. Tighter eps (e.g. 0.05) requires ~4x more samples")

    print("\n" + "=" * 70)
    print("OUTPUTS")
    print("=" * 70)
    print(f"  1. {FIGURES_DIR / 'fig4_practical_guide.pdf'}")
    print(f"  2. {TABLES_DIR / 'tab3_practical_lookup.tex'}")

    return 0


if __name__ == "__main__":
    exit(main())
