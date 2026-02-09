#!/usr/bin/env python
"""
07_hoeffding_model_figure.py - Generate the Hoeffding model derivation figure.

Creates Figure 2 for the paper: a visual explanation of the localised Hoeffding
bound model showing how the sampling probability p varies with network reach r,
how the natural floor emerges at low reach (where k > r forces p = 1), and how
different epsilon values shift the curve.

The model: k = log(2r / delta) / (2 * epsilon^2), p = min(1, k / r)
Default: epsilon = 0.1, delta = 0.1 (zero fitted parameters).

Outputs:
    - paper/figures/fig2_hoeffding_model.pdf
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
# FIGURE
# =============================================================================


def generate_figure():
    """
    Figure 2: Hoeffding model derivation.

    Two panels:
      A) Required k (sample size) vs reach — shows log(2r/δ)/(2ε²) growth
      B) Sampling probability p vs reach — shows how p decreases, with floor at p=1
    """
    print("\nGenerating Figure 2: Hoeffding model derivation...")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    reach_range = np.logspace(1.5, 5.5, 500)

    # ---- Panel A: Required k vs reach for different epsilon ----
    ax = axes[0]

    epsilon_values = [0.05, 0.1, 0.15, 0.2]
    colours = ["#d73027", "#0072B2", "#fc8d59", "#91bfdb"]
    linewidths = [1.5, 2.5, 1.5, 1.5]

    for eps, colour, lw in zip(epsilon_values, colours, linewidths, strict=True):
        k_values = [math.log(2 * r / HOEFFDING_DELTA) / (2 * eps**2) for r in reach_range]
        style = "-" if eps == HOEFFDING_EPSILON else "--"
        ax.plot(
            reach_range,
            k_values,
            linestyle=style,
            color=colour,
            linewidth=lw,
            label=f"ε = {eps}",
        )

    # Add r = k line (where p transitions from 1 to < 1)
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
    ax.set_ylabel("Required Sample Size (k)")
    ax.set_title("A) Required Sample Size")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    # ---- Panel B: Speedup (1/p) vs reach ----
    ax = axes[1]

    for eps, colour, lw in zip(epsilon_values, colours, linewidths, strict=True):
        speedup_values = [1.0 / compute_hoeffding_p(r, eps) for r in reach_range]
        style = "-" if eps == HOEFFDING_EPSILON else "--"
        ax.plot(
            reach_range,
            speedup_values,
            linestyle=style,
            color=colour,
            linewidth=lw,
            label=f"ε = {eps}",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Network Reach (r)")
    ax.set_ylabel("Speedup (1/p)")
    ax.set_title("B) Computational Speedup")
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
    print("07_hoeffding_model_figure.py - Hoeffding model derivation (Fig 2)")
    print("=" * 70)

    print("\nModel: k = log(2r/δ) / (2ε²), p = min(1, k/r)")
    print(f"  Default: ε = {HOEFFDING_EPSILON}, δ = {HOEFFDING_DELTA}")

    generate_figure()

    # Print key values
    print("\n" + "-" * 70)
    print("KEY VALUES")
    print("-" * 70)
    for reach in [100, 500, 1000, 5000, 10000, 50000]:
        k = math.log(2 * reach / HOEFFDING_DELTA) / (2 * HOEFFDING_EPSILON**2)
        p = compute_hoeffding_p(reach)
        print(f"  r={reach:>6,}: k={k:>7.0f}, p={p:>6.3f}, speedup={1 / p:>6.1f}×")

    print(f"\n  Output: {FIGURES_DIR / 'fig2_hoeffding_model.pdf'}")
    return 0


if __name__ == "__main__":
    exit(main())
