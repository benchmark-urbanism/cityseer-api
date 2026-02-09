#!/usr/bin/env python
"""
05_practical_guide.py - Generate practical guidance for practitioners.

Creates visual and tabular guidance for users to quickly determine
the appropriate sampling probability for their analysis, using the
Hoeffding/EW bound model with epsilon as the user-facing parameter.

The model: k = log(2r / delta) / (2 * epsilon^2), p = min(1, k / r)
Default: epsilon = 0.1, delta = 0.1 (zero fitted parameters).

Outputs:
    - paper/figures/fig4_practical_guide.pdf: Visual lookup chart
    - paper/tables/tab3_practical_lookup.tex: Lookup table
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


# =============================================================================
# FIGURE GENERATION
# =============================================================================


def generate_practical_figure():
    """
    Figure 4: Practical guidance chart.

    A visual lookup for practitioners showing:
    - Panel A: Required p vs reach for different epsilon values
    - Panel B: Expected speedup vs reach at epsilon = 0.1
    With annotated common analysis scenarios.
    """
    print("\nGenerating Figure 4: Practical guide...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    reach_range = np.logspace(2, 5.5, 200)

    # Panel A: Required p for multiple epsilon values
    ax = axes[0]

    epsilon_values = [0.05, 0.1, 0.15, 0.2]
    colours = ["#d73027", "#0072B2", "#fc8d59", "#91bfdb"]
    linewidths = [1.5, 2.5, 1.5, 1.5]

    for eps, colour, lw in zip(epsilon_values, colours, linewidths, strict=True):
        p_values = [compute_hoeffding_p(r, eps) * 100 for r in reach_range]
        style = "-" if eps == HOEFFDING_EPSILON else "--"
        ax.plot(
            reach_range,
            p_values,
            linestyle=style,
            color=colour,
            linewidth=lw,
            label=f"ε = {eps}",
        )

    # Annotate common scenarios at default epsilon
    scenarios = [
        (500, "Local\n(500m)"),
        (1000, "Neighbourhood\n(1km)"),
        (5000, "District\n(5km)"),
        (10000, "City\n(10km)"),
        (50000, "Metro\n(20km)"),
    ]

    for reach, label in scenarios:
        p = compute_hoeffding_p(reach) * 100
        ax.plot(reach, p, "o", color="#D55E00", markersize=7, zorder=5)
        # Place "Local (500m)" below the dot; all others above
        if reach == 500:
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
    ax.set_ylabel("Required Sampling Probability (%)")
    ax.set_title("A) Required Sampling Probability")
    ax.set_xlim(100, 300000)
    ax.set_ylim(0, 100)
    ax.legend(loc="center right", fontsize=9)
    ax.grid(True, alpha=0.3)

    ax.text(
        0.95,
        0.95,
        f"Model: k = log(2r/δ) / (2ε²)\nDefault: ε = {HOEFFDING_EPSILON}, δ = {HOEFFDING_DELTA}",
        transform=ax.transAxes,
        fontsize=8,
        horizontalalignment="right",
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Panel B: Speedup potential at default epsilon
    ax = axes[1]

    speedups = [1 / compute_hoeffding_p(r) for r in reach_range]
    ax.plot(reach_range, speedups, color="#0072B2", linewidth=2.5)

    # Annotate scenarios
    for reach, _label in scenarios:
        p = compute_hoeffding_p(reach)
        speedup = 1 / p
        ax.plot(reach, speedup, "o", color="#D55E00", markersize=7, zorder=5)
        ax.annotate(f"{speedup:.1f}×", xy=(reach, speedup * 1.1), fontsize=9, ha="center", va="bottom")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Network Reach")
    ax.set_ylabel("Speedup (1/p)")
    ax.set_title(f"B) Expected Speedup (ε = {HOEFFDING_EPSILON})")
    ax.set_xlim(100, 300000)
    ax.set_ylim(1, 100)
    ax.grid(True, alpha=0.3, which="both")

    # Add helpful regions
    ax.axhspan(1, 2, alpha=0.1, color="red", label="<2× (limited benefit)")
    ax.axhspan(2, 10, alpha=0.1, color="yellow", label="2–10× (moderate)")
    ax.axhspan(10, 100, alpha=0.1, color="green", label=">10× (significant)")
    ax.legend(loc="lower right", fontsize=8)

    fig.suptitle("Practical Guide: Hoeffding-Based Adaptive Sampling", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    output_path = FIGURES_DIR / "fig4_practical_guide.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


def generate_practical_table():
    """Generate LaTeX lookup table for practitioners using Hoeffding model."""
    print("\nGenerating Table 3: Practical lookup...")

    reaches = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]

    latex = r"""\begin{table}[htbp]
\centering
\caption{Practical sampling lookup table ($\varepsilon = 0.1$, $\delta = 0.1$).}
\label{tab:lookup}
\begin{tabular}{rrrr}
\toprule
\textbf{Reach} & \textbf{$k$} & \textbf{$p$ (\%)} & \textbf{Speedup} \\
\midrule
"""

    def fmt_int(n):
        """Format integer with LaTeX-safe thousands separator."""
        return f"{n:,}".replace(",", "{,}")

    for reach in reaches:
        k = math.log(2 * reach / HOEFFDING_DELTA) / (2 * HOEFFDING_EPSILON**2)
        p = compute_hoeffding_p(reach)
        speedup = 1 / p

        if p >= 1.0:
            latex += f"{fmt_int(reach)} & {k:.0f} & 100.0 & 1.0$\\times$ \\\\\n"
        else:
            latex += f"{fmt_int(reach)} & {k:.0f} & {p * 100:.1f} & {speedup:.1f}$\\times$ \\\\\n"

    latex += r"""\bottomrule
\end{tabular}

\vspace{0.5em}
\footnotesize
\textbf{Usage:} Find your network reach (nodes within analysis distance), read off the required sampling probability.\\
\textbf{Model:} $k = \log(2r/\delta) / (2\varepsilon^2)$, $p = \min(1,\; k/r)$.
At low reach, $k > r$ so $p = 1$ (exact computation).
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
    print("05_practical_guide.py - Generating practical guidance (Hoeffding model)")
    print("=" * 70)

    print("\nModel: k = log(2r/δ) / (2ε²)")
    print(f"  Default: ε = {HOEFFDING_EPSILON}, δ = {HOEFFDING_DELTA}")

    # Generate outputs
    generate_practical_figure()
    generate_practical_table()

    # Print quick reference
    print("\n" + "=" * 70)
    print("QUICK REFERENCE (ε = 0.1, δ = 0.1)")
    print("=" * 70)

    print(f"\n{'Reach':>10} | {'k':>10} | {'p':>10} | {'Speedup':>10}")
    print("-" * 50)

    for reach in [100, 500, 1000, 5000, 10000, 50000]:
        k = math.log(2 * reach / HOEFFDING_DELTA) / (2 * HOEFFDING_EPSILON**2)
        p = compute_hoeffding_p(reach)
        speedup = 1 / p
        print(f"{reach:>10} | {k:>10.0f} | {p:>9.1%} | {speedup:>9.1f}×")

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("\n1. No fitted parameters: ε and δ are user-chosen (default 0.1, 0.1)")
    print("2. At low reach, p → 1.0 (floor emerges naturally from the bound)")
    print("3. At reach > ~2,000, meaningful speedup begins")
    print("4. At reach > 10,000, expect 10×+ speedup while maintaining ρ ≥ 0.95")
    print("5. Tighter ε (e.g. 0.05) requires ~4× more samples")

    print("\n" + "=" * 70)
    print("OUTPUTS")
    print("=" * 70)
    print(f"  1. {FIGURES_DIR / 'fig4_practical_guide.pdf'}")
    print(f"  2. {TABLES_DIR / 'tab3_practical_lookup.tex'}")

    return 0


if __name__ == "__main__":
    exit(main())
