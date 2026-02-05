#!/usr/bin/env python
"""
03_combined_model.py - Combine k and min_eff_n into final sampling model.

This script combines the fitted parameters from 01_fit_model.py and 02_fit_floor.py
into the complete sampling model:

    eff_n = max(k × sqrt(reach), min_eff_n)
    p = min(1.0, eff_n / reach)

Generates the key figure showing the final model behavior.

Outputs:
    - output/sampling_model.json: Final model parameters
    - paper/figures/fig4_combined_model.pdf: Model visualization
    - paper/tables/tab1_parameters.tex: LaTeX table of parameters
"""

import json
import math
from datetime import datetime
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

OUTPUT_DIR.mkdir(exist_ok=True)
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

def load_fitted_parameters() -> tuple[float, int]:
    """Load k and min_eff_n from previous scripts."""
    # Load k
    model_path = OUTPUT_DIR / "model_fit.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Model fit not found at {model_path}. Run 01_fit_model.py first.")

    with open(model_path) as f:
        model_fit = json.load(f)
    k = model_fit["model"]["k"]

    # Load min_eff_n
    floor_path = OUTPUT_DIR / "floor_fit.json"
    if not floor_path.exists():
        raise FileNotFoundError(f"Floor fit not found at {floor_path}. Run 02_fit_floor.py first.")

    with open(floor_path) as f:
        floor_fit = json.load(f)
    min_eff_n = floor_fit["model"]["min_eff_n"]

    return k, min_eff_n


# =============================================================================
# MODEL FUNCTIONS
# =============================================================================

def compute_eff_n(reach: float, k: float, min_eff_n: float) -> float:
    """Compute effective sample size from the model."""
    return max(k * math.sqrt(reach), min_eff_n)


def compute_p(reach: float, k: float, min_eff_n: float) -> float:
    """Compute sampling probability from the model."""
    eff_n = compute_eff_n(reach, k, min_eff_n)
    return min(1.0, eff_n / reach)


def crossover_reach(k: float, min_eff_n: float) -> float:
    """
    Find the reach where the two model components are equal.

    k × sqrt(reach) = min_eff_n
    reach = (min_eff_n / k)²
    """
    return (min_eff_n / k) ** 2


# =============================================================================
# FIGURE GENERATION
# =============================================================================

def generate_fig4_combined_model(k: float, min_eff_n: int):
    """
    Figure 4: The complete sampling model.

    Shows how the model behaves across different reach values,
    including the crossover point between proportional and floor regimes.
    """
    print("\nGenerating Figure 4: Combined model...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    reach_range = np.logspace(1, 5.5, 200)  # 10 to ~300,000
    crossover = crossover_reach(k, min_eff_n)

    # Panel A: Required p vs reach
    ax = axes[0]

    p_values = [compute_p(r, k, min_eff_n) * 100 for r in reach_range]

    # Plot the model
    ax.plot(reach_range, p_values, color="#0072B2", linewidth=2.5,
            label=f"Model: p = max({k}/sqrt(r), {min_eff_n}/r)")

    # Show the two components
    p_proportional = [min(100, k / math.sqrt(r) * 100) for r in reach_range]
    p_floor = [min(100, min_eff_n / r * 100) for r in reach_range]

    ax.plot(reach_range, p_proportional, color="gray", linewidth=1, linestyle=":",
            alpha=0.7, label=f"Proportional: {k}/sqrt(r)")
    ax.plot(reach_range, p_floor, color="gray", linewidth=1, linestyle="--",
            alpha=0.7, label=f"Floor: {min_eff_n}/r")

    # Mark crossover point
    ax.axvline(crossover, color="red", linestyle="-", linewidth=1.5, alpha=0.5)
    ax.annotate(f"crossover\nreach={crossover:.0f}",
                xy=(crossover, compute_p(crossover, k, min_eff_n) * 100 + 5),
                fontsize=9, color="red", ha="center")

    # Shade regimes
    ax.fill_between(reach_range, 0, 100, where=[r < crossover for r in reach_range],
                    alpha=0.1, color="red", label="Floor-dominated")
    ax.fill_between(reach_range, 0, 100, where=[r >= crossover for r in reach_range],
                    alpha=0.1, color="blue", label="Proportional-dominated")

    ax.set_xscale("log")
    ax.set_xlabel("Network Reach (nodes within distance)")
    ax.set_ylabel("Required Sampling Probability (%)")
    ax.set_title("A) Required Sampling Probability vs Reach")
    ax.set_xlim(50, 300000)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel B: Effective sample size vs reach
    ax = axes[1]

    eff_n_values = [compute_eff_n(r, k, min_eff_n) for r in reach_range]

    ax.plot(reach_range, eff_n_values, color="#0072B2", linewidth=2.5,
            label=f"Model: eff_n = max({k}×sqrt(r), {min_eff_n})")

    # Show components
    eff_n_proportional = [k * math.sqrt(r) for r in reach_range]
    eff_n_floor = [min_eff_n for _ in reach_range]

    ax.plot(reach_range, eff_n_proportional, color="gray", linewidth=1, linestyle=":",
            alpha=0.7, label=f"Proportional: {k}×sqrt(r)")
    ax.plot(reach_range, eff_n_floor, color="gray", linewidth=1, linestyle="--",
            alpha=0.7, label=f"Floor: {min_eff_n}")

    # Mark crossover
    ax.axvline(crossover, color="red", linestyle="-", linewidth=1.5, alpha=0.5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Network Reach")
    ax.set_ylabel("Effective Sample Size (eff_n)")
    ax.set_title("B) Effective Sample Size vs Reach")
    ax.set_xlim(50, 300000)
    ax.set_ylim(50, 10000)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    fig.suptitle(
        f"The Sampling Model: eff_n = max({k} × sqrt(reach), {min_eff_n})",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()

    output_path = FIGURES_DIR / "fig4_combined_model.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


def generate_parameters_table(k: float, min_eff_n: int):
    """Generate LaTeX table of model parameters."""
    print("\nGenerating Table 1: Parameters...")

    crossover = crossover_reach(k, min_eff_n)

    latex = r"""\begin{table}[htbp]
\centering
\caption{Fitted Sampling Model Parameters}
\label{tab:parameters}
\begin{tabular}{lrl}
\toprule
\textbf{Parameter} & \textbf{Value} & \textbf{Description} \\
\midrule
$k$ & """ + f"{k:.1f}" + r""" & Proportional scaling constant \\
$n_{\min}$ & """ + f"{min_eff_n}" + r""" & Minimum effective sample size \\
Crossover reach & """ + f"{crossover:.0f}" + r""" & Where $k\sqrt{r} = n_{\min}$ \\
\bottomrule
\end{tabular}

\vspace{0.5em}
\footnotesize
\textbf{Model:} $n_{\mathrm{eff}} = \max(k \cdot \sqrt{r}, n_{\min})$, where $r$ is the network reach.\\
Sampling probability: $p = n_{\mathrm{eff}} / r$ (capped at 1.0).
\end{table}
"""

    output_path = TABLES_DIR / "tab1_parameters.tex"
    with open(output_path, "w") as f:
        f.write(latex)
    print(f"  Saved: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("03_combined_model.py - Combining into final sampling model")
    print("=" * 70)

    # Load fitted parameters
    k, min_eff_n = load_fitted_parameters()
    print(f"\nLoaded parameters:")
    print(f"  k = {k}")
    print(f"  min_eff_n = {min_eff_n}")

    crossover = crossover_reach(k, min_eff_n)
    print(f"\nModel crossover point: reach = {crossover:.0f}")
    print(f"  Below {crossover:.0f}: floor dominates (eff_n = {min_eff_n})")
    print(f"  Above {crossover:.0f}: proportional dominates (eff_n = {k} × sqrt(reach))")

    # Save combined model
    output = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "script": "03_combined_model.py",
        "description": "Complete sampling model combining k and min_eff_n",
        "model": {
            "formula_eff_n": "eff_n = max(k × sqrt(reach), min_eff_n)",
            "formula_p": "p = min(1.0, eff_n / reach)",
            "k": k,
            "min_eff_n": min_eff_n,
            "crossover_reach": round(crossover, 0),
        },
        "interpretation": {
            "proportional_regime": f"When reach > {crossover:.0f}, eff_n scales as {k} × sqrt(reach)",
            "floor_regime": f"When reach < {crossover:.0f}, eff_n is fixed at {min_eff_n}",
        },
    }

    output_path = OUTPUT_DIR / "sampling_model.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved combined model: {output_path}")

    # Generate outputs
    generate_fig4_combined_model(k, min_eff_n)
    generate_parameters_table(k, min_eff_n)

    # Print example calculations
    print("\n" + "=" * 70)
    print("EXAMPLE CALCULATIONS")
    print("=" * 70)
    print(f"\n{'Reach':>10} | {'eff_n':>10} | {'p':>10} | {'Speedup':>10}")
    print("-" * 50)

    for reach in [100, 300, 500, 1000, 2000, 5000, 10000, 50000]:
        eff_n = compute_eff_n(reach, k, min_eff_n)
        p = compute_p(reach, k, min_eff_n)
        speedup = 1 / p if p > 0 else float("inf")
        print(f"{reach:>10} | {eff_n:>10.0f} | {p:>9.1%} | {speedup:>9.1f}x")

    print("\n" + "=" * 70)
    print("OUTPUTS")
    print("=" * 70)
    print(f"  1. {OUTPUT_DIR / 'sampling_model.json'}")
    print(f"  2. {FIGURES_DIR / 'fig4_combined_model.pdf'}")
    print(f"  3. {TABLES_DIR / 'tab1_parameters.tex'}")

    return 0


if __name__ == "__main__":
    exit(main())
