#!/usr/bin/env python
"""
05_practical_guide.py - Generate practical guidance for practitioners.

Creates visual and tabular guidance for users to quickly determine
the appropriate sampling probability for their analysis.

Outputs:
    - paper/figures/fig6_practical_guide.pdf: Visual lookup chart
    - paper/tables/tab3_practical_lookup.tex: Lookup table
"""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
SAMPLING_DIR = SCRIPT_DIR.parent  # analysis/sampling
OUTPUT_DIR = SAMPLING_DIR / "output"
FIGURES_DIR = SAMPLING_DIR / "paper" / "figures"
TABLES_DIR = SAMPLING_DIR / "paper" / "tables"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Matplotlib style
plt.rcParams.update({
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
})


# =============================================================================
# DATA LOADING
# =============================================================================

def load_model() -> tuple[float, int]:
    """Load the fitted model parameters."""
    model_path = OUTPUT_DIR / "sampling_model.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run 03_combined_model.py first.")

    with open(model_path) as f:
        model = json.load(f)

    k = model["model"]["k"]
    min_eff_n = model["model"]["min_eff_n"]
    return k, min_eff_n


# =============================================================================
# MODEL FUNCTIONS
# =============================================================================

def compute_p(reach: float, k: float, min_eff_n: int) -> float:
    """Compute sampling probability from the model."""
    eff_n = max(k * math.sqrt(reach), min_eff_n)
    return min(1.0, eff_n / reach)


# =============================================================================
# FIGURE GENERATION
# =============================================================================

def generate_fig6_practical_guide(k: float, min_eff_n: int):
    """
    Figure 6: Practical guidance chart.

    A visual lookup for practitioners showing:
    - Required p for different reach values
    - Expected speedup
    - Common analysis scenarios
    """
    print("\nGenerating Figure 6: Practical guide...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    reach_range = np.logspace(2, 5.5, 200)

    # Panel A: Required p with annotated scenarios
    ax = axes[0]

    p_values = [compute_p(r, k, min_eff_n) * 100 for r in reach_range]
    ax.plot(reach_range, p_values, color="#0072B2", linewidth=2.5)

    # Annotate common scenarios
    scenarios = [
        (500, "Local (500m)"),
        (1000, "Neighborhood (1km)"),
        (5000, "District (5km)"),
        (10000, "City (10km)"),
        (50000, "Metro (20km)"),
    ]

    for reach, label in scenarios:
        p = compute_p(reach, k, min_eff_n) * 100
        ax.plot(reach, p, "o", color="#D55E00", markersize=8, zorder=5)
        ax.annotate(f"{label}\n{p:.0f}%", xy=(reach, p + 3),
                    fontsize=8, ha="center", va="bottom")

    ax.set_xscale("log")
    ax.set_xlabel("Network Reach (nodes within distance)")
    ax.set_ylabel("Required Sampling Probability (%)")
    ax.set_title("A) Quick Lookup: Required Sampling Probability")
    ax.set_xlim(100, 300000)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    # Add helpful text
    ax.text(0.05, 0.95, f"Model: eff_n = max({k}×√reach, {min_eff_n})",
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Panel B: Speedup potential
    ax = axes[1]

    speedups = [1 / compute_p(r, k, min_eff_n) for r in reach_range]
    ax.plot(reach_range, speedups, color="#2ca02c", linewidth=2.5)

    # Annotate scenarios
    for reach, label in scenarios:
        p = compute_p(reach, k, min_eff_n)
        speedup = 1 / p
        ax.plot(reach, speedup, "o", color="#D55E00", markersize=8, zorder=5)
        ax.annotate(f"{speedup:.1f}x", xy=(reach, speedup * 1.1),
                    fontsize=9, ha="center", va="bottom")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Network Reach")
    ax.set_ylabel("Speedup (1/p)")
    ax.set_title("B) Expected Computational Speedup")
    ax.set_xlim(100, 300000)
    ax.set_ylim(1, 100)
    ax.grid(True, alpha=0.3, which="both")

    # Add helpful regions
    ax.axhspan(1, 2, alpha=0.1, color="red", label="<2x (limited benefit)")
    ax.axhspan(2, 10, alpha=0.1, color="yellow", label="2-10x (moderate)")
    ax.axhspan(10, 100, alpha=0.1, color="green", label=">10x (significant)")
    ax.legend(loc="lower right", fontsize=8)

    fig.suptitle("Practical Guide: Sampling for Network Centrality",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    output_path = FIGURES_DIR / "fig6_practical_guide.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


def generate_practical_table(k: float, min_eff_n: int):
    """Generate LaTeX lookup table for practitioners."""
    print("\nGenerating Table 3: Practical lookup...")

    # Representative reach values
    reaches = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]

    latex = r"""\begin{table}[htbp]
\centering
\caption{Practical Sampling Lookup Table}
\label{tab:lookup}
\begin{tabular}{rrrrl}
\toprule
\textbf{Reach} & \textbf{eff\_n} & \textbf{$p$ (\%)} & \textbf{Speedup} & \textbf{Typical Scenario} \\
\midrule
"""

    scenarios = {
        100: "Very local",
        200: "Local",
        500: "500m walking",
        1000: "Neighborhood",
        2000: "District",
        5000: "5km cycling",
        10000: "City",
        20000: "Metro area",
        50000: "Regional",
        100000: "Large regional",
    }

    for reach in reaches:
        eff_n = max(k * math.sqrt(reach), min_eff_n)
        p = min(1.0, eff_n / reach)
        speedup = 1 / p
        scenario = scenarios.get(reach, "")

        latex += f"{reach:,} & {eff_n:.0f} & {p*100:.1f} & {speedup:.1f}$\\times$ & {scenario} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}

\vspace{0.5em}
\footnotesize
\textbf{Usage:} Find your network reach (nodes within analysis distance), read off the required sampling probability.\\
\textbf{Model:} $n_{\mathrm{eff}} = \max(""" + f"{k}" + r""" \cdot \sqrt{r}, """ + f"{min_eff_n}" + r""")$, $p = n_{\mathrm{eff}} / r$.
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
    print("=" * 70)

    # Load model
    k, min_eff_n = load_model()
    print(f"\nModel: eff_n = max({k} × sqrt(reach), {min_eff_n})")

    # Generate outputs
    generate_fig6_practical_guide(k, min_eff_n)
    generate_practical_table(k, min_eff_n)

    # Print quick reference
    print("\n" + "=" * 70)
    print("QUICK REFERENCE")
    print("=" * 70)

    print(f"\n{'Reach':>10} | {'eff_n':>10} | {'p':>10} | {'Speedup':>10}")
    print("-" * 50)

    for reach in [100, 500, 1000, 5000, 10000, 50000]:
        eff_n = max(k * math.sqrt(reach), min_eff_n)
        p = min(1.0, eff_n / reach)
        speedup = 1 / p
        print(f"{reach:>10} | {eff_n:>10.0f} | {p:>9.1%} | {speedup:>9.1f}x")

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print(f"\n1. Never sample fewer than {min_eff_n} nodes (the floor)")
    print(f"2. For large networks, sample {k} × sqrt(reach) nodes")
    print(f"3. At reach > {(min_eff_n/k)**2:.0f}, you can achieve >2x speedup")
    print(f"4. At reach > 10,000, expect 10x+ speedup while maintaining rho >= 0.95")

    print("\n" + "=" * 70)
    print("OUTPUTS")
    print("=" * 70)
    print(f"  1. {FIGURES_DIR / 'fig6_practical_guide.pdf'}")
    print(f"  2. {TABLES_DIR / 'tab3_practical_lookup.tex'}")

    return 0


if __name__ == "__main__":
    exit(main())
