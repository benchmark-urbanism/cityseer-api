#!/usr/bin/env python
"""
05_practical_guide.py - Generate practical guidance for practitioners.

Creates visual and tabular guidance for users to quickly determine
the appropriate sampling budget for their analysis, using:
  - Closeness: Hoeffding/EW source sampling, k = log(2r/delta)/(2*eps^2), p = min(1, k/r)
  - Betweenness: R-K path sampling, VD = ceil(sqrt(r)),
        m = ceil((1/(2*eps^2)) * (floor(log2(VD-2)) + 1 + ln(1/delta)))

Default: epsilon = 0.1, delta = 0.1 (zero fitted parameters).

Outputs:
    - paper/figures/fig4_practical_guide.pdf: Visual lookup chart (2 panels)
    - paper/tables/tab3_practical_lookup.tex: Lookup table with both metrics
"""

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from utilities import (
    FIGURES_DIR,
    HOEFFDING_DELTA,
    HOEFFDING_EPSILON,
    TABLES_DIR,
    compute_hoeffding_p,
    compute_rk_budget,
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

    A visual lookup for practitioners showing:
    - Panel A: Required sampling budget vs reach for both closeness (p, %)
      and betweenness (m/r as fraction, %) at multiple epsilon values.
      GLA scenario reaches annotated.
    - Panel B: Closeness speedup (1/p) vs reach at epsilon = 0.1.
    """
    print("\nGenerating Figure 4: Practical guide...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    reach_range = np.logspace(2, 5.5, 200)

    # ------------------------------------------------------------------
    # Panel A: Required sampling fraction vs reach (both metrics)
    # ------------------------------------------------------------------
    ax = axes[0]

    epsilon_values = [0.05, 0.1, 0.15, 0.2]
    # Colours: red, blue, orange, light-blue
    colours = ["#d73027", "#0072B2", "#fc8d59", "#91bfdb"]
    linewidths = [1.5, 2.5, 1.5, 1.5]

    # --- Closeness: p (%) ---
    for eps, colour, lw in zip(epsilon_values, colours, linewidths, strict=True):
        p_values = [compute_hoeffding_p(r, eps) * 100 for r in reach_range]
        style = "-" if eps == 0.1 else "--"
        label = f"Closeness p, " + r"$\varepsilon$" + f" = {eps}"
        ax.plot(
            reach_range,
            p_values,
            linestyle=style,
            color=colour,
            linewidth=lw,
            label=label,
        )

    # --- Betweenness: m/r as percentage ---
    for eps, colour, lw in zip(epsilon_values, colours, linewidths, strict=True):
        betw_frac = []
        for r in reach_range:
            m = compute_rk_budget(r, eps)
            frac = min(1.0, m / r) * 100 if m is not None else 100.0
            betw_frac.append(frac)
        style = "-" if eps == 0.1 else "--"
        label = f"Betweenness m/r, " + r"$\varepsilon$" + f" = {eps}"
        ax.plot(
            reach_range,
            betw_frac,
            linestyle=style,
            color=colour,
            linewidth=lw,
            marker="",
            alpha=0.5,
            # Use dotted to distinguish from closeness solid/dashed
        )
        # Overlay tiny markers to visually separate betweenness lines
        ax.plot(
            reach_range[::20],
            [betw_frac[i] for i in range(0, len(betw_frac), 20)],
            linestyle="none",
            color=colour,
            marker="x",
            markersize=4,
            alpha=0.6,
        )

    # Annotate GLA network reaches at default epsilon (closeness)
    for reach, label in GLA_SCENARIOS:
        p = compute_hoeffding_p(reach, epsilon=0.1) * 100
        ax.plot(reach, p, "o", color="#D55E00", markersize=7, zorder=5)
        # Place saturated (100%) points below; others above
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

    # Build a concise legend: group by metric then note epsilon coding
    # Use custom legend entries for clarity
    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D([0], [0], color="#0072B2", linewidth=2.5, linestyle="-", label=r"Closeness $p$ ($\varepsilon$ = 0.1)"),
        Line2D(
            [0],
            [0],
            color="#0072B2",
            linewidth=2.5,
            linestyle="-",
            alpha=0.5,
            marker="x",
            markersize=4,
            label=r"Betweenness $m/r$ ($\varepsilon$ = 0.1)",
        ),
        Line2D([0], [0], color="grey", linewidth=1.5, linestyle="--", label=r"Other $\varepsilon$ (0.05, 0.15, 0.2)"),
        Line2D([0], [0], color="#D55E00", marker="o", linestyle="none", markersize=7, label="GLA scenarios"),
    ]
    ax.legend(handles=legend_handles, loc="center right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax.text(
        0.95,
        0.95,
        "Closeness: k = log(2r/\u03b4)/(2\u03b5\u00b2), p = min(1, k/r)\n"
        "Betweenness: R-K path sampling\n"
        f"Default: \u03b5 = 0.1, \u03b4 = {HOEFFDING_DELTA}",
        transform=ax.transAxes,
        fontsize=7,
        horizontalalignment="right",
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # ------------------------------------------------------------------
    # Panel B: Closeness speedup (1/p) vs reach at epsilon = 0.1
    # ------------------------------------------------------------------
    ax = axes[1]

    speedups = [1 / compute_hoeffding_p(r, epsilon=0.1) for r in reach_range]
    ax.plot(reach_range, speedups, color="#0072B2", linewidth=2.5, label="Closeness 1/p")

    # Annotate GLA scenarios
    for reach, _label in GLA_SCENARIOS:
        p = compute_hoeffding_p(reach, epsilon=0.1)
        speedup = 1 / p
        ax.plot(reach, speedup, "o", color="#D55E00", markersize=7, zorder=5)
        ax.annotate(f"{speedup:.1f}\u00d7", xy=(reach, speedup * 1.1), fontsize=9, ha="center", va="bottom")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Network Reach")
    ax.set_ylabel("Speedup (1/p)")
    ax.set_title(r"B) Closeness Speedup ($\varepsilon$ = 0.1)")
    ax.set_xlim(100, 300000)
    ax.set_ylim(1, 150)
    ax.grid(True, alpha=0.3, which="both")

    # Add helpful regions
    ax.axhspan(1, 2, alpha=0.1, color="red", label="<2\u00d7 (limited benefit)")
    ax.axhspan(2, 10, alpha=0.1, color="yellow", label="2\u201310\u00d7 (moderate)")
    ax.axhspan(10, 100, alpha=0.1, color="green", label=">10\u00d7 (significant)")
    ax.legend(loc="lower right", fontsize=8)

    fig.suptitle("Practical Guide: Adaptive Sampling for Closeness & Betweenness", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    output_path = FIGURES_DIR / "fig4_practical_guide.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


# =============================================================================
# TABLE GENERATION
# =============================================================================


def generate_practical_table():
    """
    Generate LaTeX lookup table for practitioners.

    Columns: Reach, k (Hoeffding), p (%), closeness speedup (1/p), m (R-K n_samples).
    At epsilon = 0.1, delta = 0.1.
    """
    print("\nGenerating Table 3: Practical lookup...")

    epsilon = 0.1
    delta = HOEFFDING_DELTA
    reaches = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]

    latex = r"""\begin{table}[htbp]
\centering
\caption{Practical sampling lookup table ($\varepsilon = 0.1$, $\delta = 0.1$).
Closeness uses Hoeffding source sampling; betweenness uses R-K path sampling.}
\label{tab:lookup}
\begin{tabular}{rrrrr}
\toprule
\textbf{Reach} & \textbf{$k$ (Hoeffding)} & \textbf{$p$ (\%)} & \textbf{Speedup ($1/p$)} & \textbf{$m$ (R-K)} \\
\midrule
"""

    def fmt_int(n):
        """Format integer with LaTeX-safe thousands separator."""
        return f"{n:,}".replace(",", "{,}")

    for reach in reaches:
        k = math.log(2 * reach / delta) / (2 * epsilon**2)
        p = compute_hoeffding_p(reach, epsilon, delta)
        speedup = 1 / p
        m = compute_rk_budget(reach, epsilon, delta)
        m_str = fmt_int(m) if m is not None else "--"

        if p >= 1.0:
            latex += f"{fmt_int(reach)} & {k:.0f} & 100.0 & 1.0$\\times$ & {m_str} \\\\\n"
        else:
            latex += f"{fmt_int(reach)} & {k:.0f} & {p * 100:.1f} & {speedup:.1f}$\\times$ & {m_str} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}

\vspace{0.5em}
\footnotesize
\textbf{Usage:} Find your network reach (nodes within analysis distance).
For closeness, read off the sampling probability $p$.
For betweenness, read off the R-K path budget $m$.\\
\textbf{Closeness model:} $k = \log(2r/\delta) / (2\varepsilon^2)$, $p = \min(1,\; k/r)$.
At low reach, $k > r$ so $p = 1$ (exact computation).\\
\textbf{Betweenness model:} $\mathrm{VD} = \lceil\sqrt{r}\rceil$,
$m = \lceil (1/(2\varepsilon^2))(\lfloor\log_2(\mathrm{VD}-2)\rfloor + 1 + \ln(1/\delta))\rceil$.
\end{table}
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
    print("  Closeness (Hoeffding/EW) + Betweenness (R-K path sampling)")
    print("=" * 70)

    epsilon = 0.1
    delta = HOEFFDING_DELTA

    print(f"\nCloseness model: k = log(2r/delta) / (2*eps^2), p = min(1, k/r)")
    print(f"Betweenness model: VD = ceil(sqrt(r)), m = ceil((1/(2*eps^2))*(floor(log2(VD-2))+1+ln(1/delta)))")
    print(f"  Default: eps = {epsilon}, delta = {delta}")

    # Generate outputs
    generate_practical_figure()
    generate_practical_table()

    # Print quick reference
    print("\n" + "=" * 70)
    print(f"QUICK REFERENCE (eps = {epsilon}, delta = {delta})")
    print("=" * 70)

    header = f"{'Reach':>10} | {'k':>8} | {'p':>8} | {'Speedup':>8} | {'m (R-K)':>8}"
    print(f"\n{header}")
    print("-" * len(header))

    for reach in [100, 500, 1000, 5000, 10000, 50000]:
        k = math.log(2 * reach / delta) / (2 * epsilon**2)
        p = compute_hoeffding_p(reach, epsilon, delta)
        speedup = 1 / p
        m = compute_rk_budget(reach, epsilon, delta)
        m_str = str(m) if m is not None else "--"
        print(f"{reach:>10} | {k:>8.0f} | {p:>7.1%} | {speedup:>7.1f}x | {m_str:>8}")

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("\n1. No fitted parameters: eps and delta are user-chosen (default 0.1, 0.1)")
    print("2. Closeness: at low reach, p -> 1.0 (exact computation); speedup grows with reach")
    print("3. Betweenness: R-K budget m grows logarithmically with reach (via VD = sqrt(r))")
    print("4. At reach > ~2,000, closeness sampling yields meaningful speedup")
    print("5. Tighter eps (e.g. 0.05) requires ~4x more samples for both metrics")

    print("\n" + "=" * 70)
    print("OUTPUTS")
    print("=" * 70)
    print(f"  1. {FIGURES_DIR / 'fig4_practical_guide.pdf'}")
    print(f"  2. {TABLES_DIR / 'tab3_practical_lookup.tex'}")

    return 0


if __name__ == "__main__":
    exit(main())
