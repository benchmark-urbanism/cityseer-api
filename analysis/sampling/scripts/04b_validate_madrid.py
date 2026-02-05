#!/usr/bin/env python
"""
04b_validate_madrid.py - External validation on Madrid regional network.

Loads pre-computed validation results from 00_generate_cache.py and generates
the validation figure for the paper.

Outputs:
    - paper/figures/fig_madrid_validation.pdf: Accuracy and speedup plots
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from utilities import FIGURES_DIR, OUTPUT_DIR

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent

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


def load_validation_data() -> pd.DataFrame:
    """Load validation results from 00_generate_cache.py output.

    Expected columns from cache: distance, mean_reach, sample_prob, effective_n,
    rho_closeness, rho_closeness_std, rho_betweenness, rho_betweenness_std,
    baseline_time, sampled_time, speedup
    """
    validation_path = OUTPUT_DIR / "madrid_validation.csv"
    if not validation_path.exists():
        raise FileNotFoundError(
            f"Madrid validation data not found at {validation_path}\n"
            "Run: python 00_generate_cache.py --madrid"
        )

    print(f"Loading validation data from {validation_path}")
    df = pd.read_csv(validation_path)

    # Rename columns to match expected format
    column_map = {
        "mean_reach": "reach",
        "sample_prob": "model_p",
    }
    df = df.rename(columns=column_map)

    # Add target columns if missing
    if "meets_target_close" not in df.columns:
        df["meets_target_close"] = df["rho_closeness"] >= 0.95
    if "meets_target_between" not in df.columns:
        df["meets_target_between"] = df["rho_betweenness"] >= 0.95

    return df


# =============================================================================
# FIGURE GENERATION
# =============================================================================


def generate_validation_figure(df: pd.DataFrame):
    """
    Generate validation figure showing accuracy and speedup across distances.

    Two panels:
    - Left: Accuracy (rho) vs distance for closeness and betweenness
    - Right: Speedup vs distance
    """
    print("\nGenerating validation figure...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    distances_km = df["distance"] / 1000

    # -----------------------------------------------------------------
    # Panel A: Accuracy
    # -----------------------------------------------------------------
    ax1 = axes[0]

    # Closeness
    ax1.errorbar(
        distances_km,
        df["rho_closeness"],
        yerr=df["rho_closeness_std"],
        fmt="o-",
        color="#0072B2",
        linewidth=2,
        markersize=8,
        capsize=4,
        label="Closeness",
    )

    # Betweenness
    ax1.errorbar(
        distances_km,
        df["rho_betweenness"],
        yerr=df["rho_betweenness_std"],
        fmt="s-",
        color="#D55E00",
        linewidth=2,
        markersize=8,
        capsize=4,
        label="Betweenness",
    )

    # Target line
    ax1.axhline(0.95, color="green", linestyle="--", linewidth=1.5, alpha=0.7, label="Target (ρ=0.95)")

    ax1.set_xlabel("Distance (km)")
    ax1.set_ylabel("Spearman ρ (ranking accuracy)")
    ax1.set_title("(a) Accuracy vs Distance")
    ax1.legend(loc="lower right")
    ax1.set_ylim(0.90, 1.005)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale("log")
    ax1.set_xticks([0.5, 1, 2, 5, 10, 20])
    ax1.set_xticklabels(["0.5", "1", "2", "5", "10", "20"])

    # -----------------------------------------------------------------
    # Panel B: Speedup
    # -----------------------------------------------------------------
    ax2 = axes[1]

    ax2.bar(
        range(len(df)),
        df["speedup"],
        color="#009E73",
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )

    # Add speedup labels on bars
    for i, (spd, dist) in enumerate(zip(df["speedup"], df["distance"])):
        ax2.text(i, spd + 0.1, f"{spd:.1f}x", ha="center", va="bottom", fontsize=9)

    ax2.set_xlabel("Distance (km)")
    ax2.set_ylabel("Speedup (baseline / sampled)")
    ax2.set_title("(b) Computational Speedup")
    ax2.set_xticks(range(len(df)))
    ax2.set_xticklabels([f"{d / 1000:.1f}" if d < 1000 else f"{d // 1000}" for d in df["distance"]])
    ax2.grid(True, alpha=0.3, axis="y")

    # Reference line at 1x
    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    fig.suptitle(
        "Madrid Regional Network Validation",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()

    output_path = FIGURES_DIR / "fig6_madrid_validation.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()

    return output_path


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("=" * 70)
    print("04b_validate_madrid.py - External validation on Madrid network")
    print("=" * 70)

    # Load model
    k, min_eff_n = load_model()
    print(f"\nModel: eff_n = max({k} × sqrt(reach), {min_eff_n})")

    # Load validation data from 00_generate_cache.py
    df = load_validation_data()
    print(f"  Loaded {len(df)} rows")

    # Generate figure
    fig_path = generate_validation_figure(df)

    # Print summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    print(
        f"\n{'Distance':>10} | {'Reach':>10} | {'Model p':>10} | {'ρ close':>10} | {'ρ between':>10} | {'Speedup':>10} | {'Pass?':>6}"
    )
    print("-" * 85)

    all_pass = True
    for _, row in df.iterrows():
        passes = row["meets_target_close"] and row["meets_target_between"]
        status = "PASS" if passes else "FAIL"
        if not passes:
            all_pass = False

        dist_str = f"{row['distance'] / 1000:.1f}km" if row["distance"] < 1000 else f"{row['distance'] // 1000}km"
        print(
            f"{dist_str:>10} | "
            f"{row['reach']:>10,.0f} | "
            f"{row['model_p']:>9.1%} | "
            f"{row['rho_closeness']:>10.4f} | "
            f"{row['rho_betweenness']:>10.4f} | "
            f"{row['speedup']:>9.1f}x | "
            f"{status:>6}"
        )

    print("-" * 85)

    if all_pass:
        print("\nALL DISTANCES PASS: Model achieves ρ >= 0.95 for both metrics at all distances.")
    else:
        print("\nWARNING: Some distances do not meet the ρ >= 0.95 target.")

    # Overall statistics
    print("\nOverall Statistics:")
    print(f"  Mean speedup: {df['speedup'].mean():.2f}x")
    print(f"  Max speedup:  {df['speedup'].max():.2f}x (at {df.loc[df['speedup'].idxmax(), 'distance'] / 1000:.0f}km)")
    print(f"  Mean ρ (closeness):   {df['rho_closeness'].mean():.4f}")
    print(f"  Mean ρ (betweenness): {df['rho_betweenness'].mean():.4f}")

    print("\n" + "=" * 70)
    print("OUTPUTS")
    print("=" * 70)
    print(f"  1. {OUTPUT_DIR / 'madrid_validation.csv'}")
    print(f"  2. {fig_path}")

    return 0


if __name__ == "__main__":
    exit(main())
