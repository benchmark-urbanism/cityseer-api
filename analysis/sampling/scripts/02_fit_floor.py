#!/usr/bin/env python
"""
02_fit_floor.py - Fit the minimum effective sample size floor.

This script determines the min_eff_n parameter for the sampling model:
    eff_n = max(k × sqrt(reach), min_eff_n)

The floor is needed because at low reach, the proportional term k × sqrt(reach)
gives too few samples for reliable estimates. Empirically, we find the minimum
eff_n that achieves a high success rate (>95% of configurations achieving rho >= 0.95).

Uses cached synthetic network data.

Outputs:
    - output/floor_fit.json: Fitted min_eff_n value with metadata
    - paper/figures/fig3_floor_justification.pdf: Why the floor is needed
"""

import json
import math
import pickle
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
SAMPLING_DIR = SCRIPT_DIR.parent  # analysis/sampling
OUTPUT_DIR = SAMPLING_DIR / "output"
FIGURES_DIR = SAMPLING_DIR / "paper" / "figures"
CACHE_DIR = SAMPLING_DIR.parent.parent / "temp" / "sampling_cache"

OUTPUT_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

SYNTHETIC_CACHE = CACHE_DIR / "sampling_analysis_v17.pkl"

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

def load_synthetic_data() -> pd.DataFrame:
    """Load cached synthetic network sampling results."""
    if not SYNTHETIC_CACHE.exists():
        raise FileNotFoundError(f"Synthetic data cache not found at {SYNTHETIC_CACHE}")

    with open(SYNTHETIC_CACHE, "rb") as f:
        data = pickle.load(f)

    df = pd.DataFrame(data)
    print(f"Loaded synthetic data: {len(df)} rows")
    return df


def load_model_fit() -> dict:
    """Load the fitted k value from 01_fit_model.py."""
    model_path = OUTPUT_DIR / "model_fit.json"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model fit not found at {model_path}\n"
            "Run 01_fit_model.py first."
        )

    with open(model_path) as f:
        return json.load(f)


# =============================================================================
# FLOOR FITTING
# =============================================================================

def fit_min_eff_n(df: pd.DataFrame, target_success_rate: float = 0.95) -> dict:
    """
    Fit the minimum effective sample size floor.

    Uses bin-based analysis to find the minimum eff_n where the LOCAL success
    rate (within that bin) achieves the target. This avoids the issue where
    cumulative rates get "pulled up" by high eff_n observations with 100% success.

    Parameters
    ----------
    df : pd.DataFrame
        Synthetic data with effective_n and spearman columns
    target_success_rate : float
        Target success rate (default 0.95 = 95%)

    Returns
    -------
    dict
        Fitted floor with min_eff_n value and analysis details
    """
    # Filter to betweenness
    df_b = df[df["metric"] == "betweenness"].copy()

    # Analyze by eff_n bins - use the LOWER bound of the first bin that achieves target
    bins = [
        (0, 50), (50, 100), (100, 150), (150, 200), (200, 250), (250, 300),
        (300, 350), (350, 400), (400, 500), (500, 750), (750, 1000)
    ]
    bin_results = []

    for low, high in bins:
        subset = df_b[(df_b["effective_n"] > low) & (df_b["effective_n"] <= high)]
        if len(subset) == 0:
            continue

        n_success = (subset["spearman"] >= 0.95).sum()
        n_total = len(subset)
        success_rate = n_success / n_total

        bin_results.append({
            "bin": f"{low}-{high}",
            "low": low,
            "high": high,
            "n_total": n_total,
            "n_success": n_success,
            "success_rate": success_rate,
        })

    bin_results_df = pd.DataFrame(bin_results)

    # Find the first bin where success rate >= target
    # The floor is the LOWER bound of that bin (conservative)
    achieving_bins = bin_results_df[bin_results_df["success_rate"] >= target_success_rate]
    if len(achieving_bins) > 0:
        # Use lower bound of first achieving bin
        min_eff_n = int(achieving_bins.iloc[0]["low"])
        achieved_rate = achieving_bins.iloc[0]["success_rate"]
        achieving_bin = achieving_bins.iloc[0]["bin"]
    else:
        # Use highest bin's upper bound if target not achieved
        min_eff_n = int(bin_results_df.iloc[-1]["high"])
        achieved_rate = bin_results_df.iloc[-1]["success_rate"]
        achieving_bin = bin_results_df.iloc[-1]["bin"]

    print(f"\nFloor fitting results:")
    print(f"  Target success rate: {target_success_rate:.0%}")
    print(f"  First achieving bin: {achieving_bin} (success rate: {achieved_rate:.1%})")
    print(f"  Fitted min_eff_n: {min_eff_n} (lower bound of achieving bin)")
    print(f"\n  Success rate by eff_n bin:")
    for _, row in bin_results_df.iterrows():
        print(f"    {row['bin']:>10}: {row['success_rate']:.1%} (n={row['n_total']})")

    return {
        "min_eff_n": min_eff_n,
        "target_success_rate": target_success_rate,
        "achieved_success_rate": round(float(achieved_rate), 4),
        "achieving_bin": achieving_bin,
        "bin_analysis": bin_results_df.to_dict(orient="records"),
    }


def analyze_proportional_breakdown(df: pd.DataFrame, k: float) -> pd.DataFrame:
    """
    Analyze where the pure proportional model breaks down.

    For each reach value, compute what eff_n the proportional model would give,
    and check the success rate at that eff_n.
    """
    df_b = df[df["metric"] == "betweenness"].copy()

    reach_values = [50, 100, 200, 300, 500, 1000, 2000]
    results = []

    for reach in reach_values:
        # Proportional model gives: eff_n = k × sqrt(reach)
        p_prop = min(1.0, k / math.sqrt(reach))
        eff_n_prop = reach * p_prop

        # Find observations near this eff_n
        tolerance = 0.2  # +/- 20%
        subset = df_b[
            (df_b["effective_n"] >= eff_n_prop * (1 - tolerance)) &
            (df_b["effective_n"] <= eff_n_prop * (1 + tolerance))
        ]

        if len(subset) > 0:
            success_rate = (subset["spearman"] >= 0.95).sum() / len(subset)
        else:
            success_rate = np.nan

        results.append({
            "reach": reach,
            "p_proportional": p_prop,
            "eff_n_proportional": eff_n_prop,
            "n_observations": len(subset),
            "success_rate": success_rate,
        })

    return pd.DataFrame(results)


# =============================================================================
# FIGURE GENERATION
# =============================================================================

def generate_fig3_floor_justification(df: pd.DataFrame, k: float, min_eff_n: int, bin_analysis: list):
    """
    Figure 3: Why the floor is needed.

    Shows that at low reach, the proportional model gives unreliable results,
    justifying the need for a minimum eff_n floor.
    """
    print("\nGenerating Figure 3: Floor justification...")

    df_b = df[df["metric"] == "betweenness"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Success rate by eff_n bin
    ax = axes[0]

    bin_df = pd.DataFrame(bin_analysis)
    x = range(len(bin_df))
    colors = ["#d62728" if rate < 0.95 else "#2ca02c" for rate in bin_df["success_rate"]]

    ax.bar(x, bin_df["success_rate"] * 100, color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.axhline(95, color="green", linestyle="--", linewidth=1.5, label="95% target")
    ax.axhline(90, color="orange", linestyle=":", linewidth=1.5, label="90% threshold")

    # Mark where floor is
    floor_idx = None
    for i, row in bin_df.iterrows():
        low = int(row["bin"].split("-")[0])
        if low >= min_eff_n:
            floor_idx = i
            break

    if floor_idx is not None:
        ax.axvline(floor_idx - 0.5, color="red", linestyle="-", linewidth=2, alpha=0.7)
        ax.annotate(f"min_eff_n={min_eff_n}", xy=(floor_idx - 0.3, 50),
                    fontsize=10, color="red", rotation=90, va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels(bin_df["bin"], rotation=45, ha="right")
    ax.set_xlabel("Effective Sample Size (eff_n) Range")
    ax.set_ylabel("Success Rate (% achieving rho >= 0.95)")
    ax.set_title("A) Reliability Requires Sufficient eff_n")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel B: Where proportional model fails
    ax = axes[1]

    breakdown = analyze_proportional_breakdown(df, k)

    x = range(len(breakdown))
    success_colors = ["#d62728" if rate < 0.90 else "#ff7f0e" if rate < 0.95 else "#2ca02c"
                      for rate in breakdown["success_rate"]]

    bars = ax.bar(x, breakdown["success_rate"] * 100, color=success_colors, alpha=0.7,
                  edgecolor="black", linewidth=0.5)

    ax.axhline(95, color="green", linestyle="--", linewidth=1.5)
    ax.axhline(90, color="orange", linestyle=":", linewidth=1.5)

    # Add eff_n annotations
    for i, row in breakdown.iterrows():
        eff_n = row["eff_n_proportional"]
        ax.annotate(f"eff_n={eff_n:.0f}", xy=(i, 5), fontsize=8, ha="center", color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in breakdown["reach"]])
    ax.set_xlabel("Network Reach")
    ax.set_ylabel("Success Rate (% achieving rho >= 0.95)")
    ax.set_title(f"B) Proportional Model (k={k:.1f}) Fails at Low Reach")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Empirical Justification for min_eff_n Floor", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    output_path = FIGURES_DIR / "fig3_floor_justification.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("02_fit_floor.py - Fitting minimum effective sample size floor")
    print("=" * 70)

    # Load data
    df = load_synthetic_data()

    # Load k from previous step
    model_fit = load_model_fit()
    k = model_fit["model"]["k"]
    print(f"\nUsing k = {k} from 01_fit_model.py")

    # Fit floor
    floor_fit = fit_min_eff_n(df, target_success_rate=0.95)

    # Save floor fit
    output = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "script": "02_fit_floor.py",
        "description": "Minimum effective sample size floor fitted from synthetic data",
        "model": {
            "min_eff_n": int(floor_fit["min_eff_n"]),
            "target_success_rate": float(floor_fit["target_success_rate"]),
            "achieved_success_rate": float(floor_fit["achieved_success_rate"]),
        },
        "bin_analysis": [
            {k: (float(v) if isinstance(v, (np.floating, float)) else
                 int(v) if isinstance(v, (np.integer, int)) else v)
             for k, v in item.items()}
            for item in floor_fit["bin_analysis"]
        ],
    }

    output_path = OUTPUT_DIR / "floor_fit.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved floor fit: {output_path}")

    # Generate figure
    generate_fig3_floor_justification(df, k, floor_fit["min_eff_n"], floor_fit["bin_analysis"])

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nFitted minimum effective sample size: min_eff_n = {floor_fit['min_eff_n']}")
    print(f"Success rate at this threshold: {floor_fit['achieved_success_rate']:.1%}")
    print(f"\nCombined with k = {k}:")
    print(f"  eff_n = max({k} × sqrt(reach), {floor_fit['min_eff_n']})")
    print(f"  p = eff_n / reach")

    print("\n" + "=" * 70)
    print("OUTPUTS")
    print("=" * 70)
    print(f"  1. {OUTPUT_DIR / 'floor_fit.json'}")
    print(f"  2. {FIGURES_DIR / 'fig3_floor_justification.pdf'}")

    return 0


if __name__ == "__main__":
    exit(main())
