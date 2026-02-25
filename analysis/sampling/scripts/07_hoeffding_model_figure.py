#!/usr/bin/env python
"""
07_hoeffding_model_figure.py - Generate the model derivation figure.

Creates Figure 2 for the paper: the Hoeffding/EW closeness sampling model.

Closeness (source sampling, Hoeffding/EW bound):
    k = log(2r / delta) / (2 * epsilon^2),  p = min(1, k / r),  speedup = 1/p

Both panels are parameterized by (epsilon, delta, reach).

Outputs:
    - paper/figures/fig2_hoeffding_model.pdf
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from cityseer.config import compute_hoeffding_p
from utilities import (
    FIGURES_DIR,
    HOEFFDING_DELTA,
    HOEFFDING_EPSILON,
    compute_hoeffding_eff_n,
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
# FIGURE
# =============================================================================


def generate_figure():
    """
    Figure 2: Model derivation -- Hoeffding (closeness).

    Two panels:
      A) Sample budget k vs reach at multiple epsilon values.
         Includes k=r diagonal (where p transitions from 1 to < 1). Log-log scale.
      B) Closeness speedup (1/p) vs reach for multiple epsilon values.
    """
    print("\nGenerating Figure 2: Model derivation (Hoeffding)...")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    reach_range = np.logspace(1.5, 5.5, 500)

    # Epsilon values and styling
    # Primary: eps=0.1 (thick, solid), secondary: eps=0.05/0.15/0.2 (thinner, dashed)
    PAPER_EPSILON = 0.1
    epsilon_values = [0.05, 0.1, 0.15, 0.2]
    colours = ["#d73027", "#0072B2", "#fc8d59", "#91bfdb"]
    linewidths = [1.5, 2.5, 1.5, 1.5]

    # ---- Panel A: Sample budget vs reach ----
    ax = axes[0]

    for eps, colour, lw in zip(epsilon_values, colours, linewidths, strict=True):
        # Hoeffding k (closeness) -- solid lines
        k_values = [compute_hoeffding_eff_n(r, eps) for r in reach_range]
        style = "-" if eps == PAPER_EPSILON else "--"
        ax.plot(
            reach_range,
            k_values,
            linestyle=style,
            color=colour,
            linewidth=lw,
            label=f"Closeness k, \u03b5 = {eps}",
        )

    # Add r = k line (where p transitions from 1 to < 1 for closeness)
    ax.plot(
        reach_range,
        reach_range,
        "k:",
        linewidth=1,
        alpha=0.5,
        label="k = r (p = 1 boundary)",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Network Reach (r)")
    ax.set_ylabel("Required Sample Budget (k)")
    ax.set_title("A) Closeness Sample Budget k")
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.3, which="both")

    # ---- Panel B: Closeness speedup (1/p) vs reach ----
    ax = axes[1]

    for eps, colour, lw in zip(epsilon_values, colours, linewidths, strict=True):
        speedup_values = [1.0 / compute_hoeffding_p(r, eps) for r in reach_range]
        style = "-" if eps == PAPER_EPSILON else "--"
        ax.plot(
            reach_range,
            speedup_values,
            linestyle=style,
            color=colour,
            linewidth=lw,
            label=f"\u03b5 = {eps}",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Network Reach (r)")
    ax.set_ylabel("Speedup (1/p)")
    ax.set_title("B) Closeness Computational Speedup")
    ax.set_xlim(reach_range[0], reach_range[-1])
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()

    output_path = FIGURES_DIR / "fig2_hoeffding_model.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("=" * 70)
    print("07_hoeffding_model_figure.py - Model derivation figure (Fig 2)")
    print("=" * 70)

    print("\nCloseness model: k = log(2r/\u03b4) / (2\u03b5\u00b2), p = min(1, k/r)")
    print(f"  Default: \u03b5 = {HOEFFDING_EPSILON}, \u03b4 = {HOEFFDING_DELTA}")

    generate_figure()

    # Print key values at representative reaches
    print("\n" + "-" * 70)
    print("KEY VALUES -- CLOSENESS (Hoeffding k)")
    print("-" * 70)
    header = f"  {'r':>6}  |  {'k (close)':>10}  {'p':>7}  {'speedup':>8}"
    print(header)
    print("  " + "-" * len(header.strip()))

    for reach in [100, 500, 1000, 5000, 10000, 50000]:
        k = compute_hoeffding_eff_n(reach)
        p = compute_hoeffding_p(reach)
        print(f"  {reach:>6,}  |  {k:>10,.0f}  {p:>7.3f}  {1 / p:>7.1f}\u00d7")

    # Also show for other epsilon values
    for eps in [0.05, 0.1, 0.15, 0.2]:
        print(f"\n  \u03b5 = {eps}:")
        for reach in [100, 500, 1000, 5000, 10000, 50000]:
            k = compute_hoeffding_eff_n(reach, eps)
            p = compute_hoeffding_p(reach, eps)
            print(f"    r={reach:>6,}: k={k:>8,.0f}, p={p:.3f}, speedup={1 / p:>7.1f}\u00d7")

    print(f"\n  Output: {FIGURES_DIR / 'fig2_hoeffding_model.pdf'}")
    return 0


if __name__ == "__main__":
    exit(main())
