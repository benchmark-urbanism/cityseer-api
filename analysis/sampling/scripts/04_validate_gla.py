#!/usr/bin/env python
"""
04_validate_gla.py - Validate the sampling model on Greater London network.

Tests the fitted model on a real-world 294,000-node network at three
distance thresholds (5km, 10km, 20km). Uses cached ground truth and
sampled centrality data from previous analyses.

Uses 20km inward buffer for live nodes to eliminate edge artifacts.

Outputs:
    - output/gla_validation.csv: Validation results
    - paper/figures/fig5_gla_validation.pdf: Validation figure
    - paper/tables/tab2_validation.tex: Validation results table
"""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
SAMPLING_DIR = SCRIPT_DIR.parent  # analysis/sampling
OUTPUT_DIR = SAMPLING_DIR / "output"
FIGURES_DIR = SAMPLING_DIR / "paper" / "figures"
TABLES_DIR = SAMPLING_DIR / "paper" / "tables"
GLA_CACHE_DIR = SAMPLING_DIR / ".cache"

OUTPUT_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Validation distances
DISTANCES = [5000, 10000, 20000]

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
    """Load validation results from model_validation.csv.

    Handles both formats:
    - Old format: combined rows with obs_rho_b, obs_rho_h columns
    - New format (from 00_generate_cache.py): separate rows per metric
    """
    validation_path = OUTPUT_DIR / "model_validation.csv"
    if not validation_path.exists():
        raise FileNotFoundError(
            f"Validation CSV not found at {validation_path}\n"
            "Run: python 00_generate_cache.py --gla"
        )

    print(f"Loading validation data from {validation_path}")
    df = pd.read_csv(validation_path)

    # Check if this is the new format (has 'metric' column with separate rows)
    if "metric" in df.columns:
        print("  Converting from 00_generate_cache.py format...")
        # Pivot to combine harmonic/betweenness into columns
        harmonic = df[df["metric"] == "harmonic"].copy()
        betweenness = df[df["metric"] == "betweenness"].copy()

        # Rename columns for merge
        harmonic = harmonic.rename(columns={"spearman": "obs_rho_h", "spearman_std": "obs_rho_h_std"})
        betweenness = betweenness.rename(columns={"spearman": "obs_rho_b", "spearman_std": "obs_rho_b_std"})

        # Merge on distance and sample_prob
        merged = pd.merge(
            harmonic[["distance", "sample_prob", "mean_reach", "effective_n", "obs_rho_h", "obs_rho_h_std"]],
            betweenness[["distance", "sample_prob", "obs_rho_b", "obs_rho_b_std"]],
            on=["distance", "sample_prob"],
        )
        merged = merged.rename(columns={"mean_reach": "reach", "effective_n": "eff_n"})
        df = merged

    return df


# =============================================================================
# MODEL FUNCTIONS
# =============================================================================


def compute_model_p(reach: float, k: float, min_eff_n: int) -> float:
    """Compute the model's recommended sampling probability."""
    eff_n = max(k * math.sqrt(reach), min_eff_n)
    return min(1.0, eff_n / reach)


# =============================================================================
# FIGURE GENERATION
# =============================================================================


def generate_fig5_validation(df: pd.DataFrame, k: float, min_eff_n: int):
    """
    Figure 5: GLA validation results.

    Shows observed rho vs sampling probability for each distance,
    with model recommendations marked.
    """
    print("\nGenerating Figure 5: GLA validation...")

    distances = sorted(df["distance"].unique())
    n_distances = len(distances)

    fig, axes = plt.subplots(1, n_distances, figsize=(4 * n_distances, 5), sharey=True)
    if n_distances == 1:
        axes = [axes]

    colors = {5000: "#009E73", 10000: "#D55E00", 20000: "#CC79A7"}

    for i, dist in enumerate(distances):
        ax = axes[i]
        subset = df[df["distance"] == dist].sort_values("sample_prob")

        reach = subset["reach"].iloc[0]
        model_p = compute_model_p(reach, k, min_eff_n) * 100

        # Plot observed rho
        ax.plot(
            subset["sample_prob"] * 100,
            subset["obs_rho_b"],
            "o-",
            color=colors.get(dist, "gray"),
            linewidth=2,
            markersize=6,
            label="Observed rho (betweenness)",
        )

        # Target line
        ax.axhline(0.95, color="green", linestyle="--", linewidth=1.5, alpha=0.7)
        ax.text(95, 0.952, "target", fontsize=8, color="green", ha="right", va="bottom")

        # Model recommendation
        ax.axvline(model_p, color="#0072B2", linestyle="-", linewidth=2, alpha=0.8)
        ax.annotate(
            f"Model: {model_p:.1f}%", xy=(model_p + 2, 0.96), fontsize=9, color="#0072B2", rotation=90, va="bottom"
        )

        # Find observed rho at model recommendation
        closest_idx = (subset["sample_prob"] * 100 - model_p).abs().argmin()
        model_rho = subset.iloc[closest_idx]["obs_rho_b"]
        ax.plot(model_p, model_rho, "s", color="#0072B2", markersize=10, zorder=5)

        ax.set_xlabel("Sampling Probability (%)")
        if i == 0:
            ax.set_ylabel("Spearman rho (ranking accuracy)")
        ax.set_title(f"{dist // 1000}km (reach={reach:,.0f})")
        ax.set_xlim(0, 100)
        ax.set_ylim(0.95, 1.005)
        ax.grid(True, alpha=0.3)

        if i == n_distances - 1:
            ax.legend(loc="lower right", fontsize=9)

    fig.suptitle("Model Validation on Greater London Network (294k nodes)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    output_path = FIGURES_DIR / "fig5_gla_validation.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


def generate_validation_table(df: pd.DataFrame, k: float, min_eff_n: int):
    """Generate LaTeX table of validation results."""
    print("\nGenerating Table 2: Validation results...")

    rows = []
    for dist in sorted(df["distance"].unique()):
        subset = df[df["distance"] == dist]
        reach = subset["reach"].iloc[0]

        # Model prediction
        model_p = compute_model_p(reach, k, min_eff_n)
        model_eff_n = reach * model_p

        # Find observed rho at closest p to model recommendation
        closest_idx = (subset["sample_prob"] - model_p).abs().argmin()
        actual_p = subset.iloc[closest_idx]["sample_prob"]
        observed_rho = subset.iloc[closest_idx]["obs_rho_b"]

        # Speedup
        speedup = 1 / model_p if model_p > 0 else float("inf")

        rows.append(
            {
                "distance": dist,
                "reach": reach,
                "model_p": model_p,
                "model_eff_n": model_eff_n,
                "actual_p": actual_p,
                "observed_rho": observed_rho,
                "speedup": speedup,
                "meets_target": observed_rho >= 0.95,
            }
        )

    results_df = pd.DataFrame(rows)

    # Save as CSV
    csv_path = OUTPUT_DIR / "gla_validation_summary.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # Generate LaTeX
    latex = r"""\begin{table}[htbp]
\centering
\caption{Model Validation on Greater London Network}
\label{tab:validation}
\begin{tabular}{rrrrrr}
\toprule
\textbf{Distance} & \textbf{Reach} & \textbf{Model $p$} & \textbf{Observed $\rho$} & \textbf{Speedup} & \textbf{$\rho \geq 0.95$?} \\
\midrule
"""

    for _, row in results_df.iterrows():
        check = r"\checkmark" if row["meets_target"] else r"\texttimes"
        # Note: % must be escaped as \% in LaTeX (otherwise it starts a comment)
        model_p_pct = f"{row['model_p'] * 100:.1f}\\%"
        latex += f"{row['distance'] // 1000}km & {row['reach']:,.0f} & {model_p_pct} & "
        latex += f"{row['observed_rho']:.4f} & {row['speedup']:.1f}$\\times$ & {check} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}

\vspace{0.5em}
\footnotesize
Network: Greater London Area, 294,486 nodes, 20km live node buffer.
\end{table}
"""

    output_path = TABLES_DIR / "tab2_validation.tex"
    with open(output_path, "w") as f:
        f.write(latex)
    print(f"  Saved: {output_path}")

    return results_df


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("=" * 70)
    print("04_validate_gla.py - Validating model on Greater London network")
    print("=" * 70)

    # Load model
    k, min_eff_n = load_model()
    print(f"\nModel: eff_n = max({k} × sqrt(reach), {min_eff_n})")

    # Load validation data
    df = load_validation_data()
    print(f"\nValidation data: {len(df)} rows")
    print(f"Distances: {sorted(df['distance'].unique())}")

    # Generate outputs
    generate_fig5_validation(df, k, min_eff_n)
    results_df = generate_validation_table(df, k, min_eff_n)

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    print(f"\n{'Distance':>10} | {'Reach':>10} | {'Model p':>10} | {'Obs rho':>10} | {'Target?':>10}")
    print("-" * 60)

    all_pass = True
    for _, row in results_df.iterrows():
        status = "PASS" if row["meets_target"] else "FAIL"
        if not row["meets_target"]:
            all_pass = False
        print(
            f"{row['distance'] // 1000}km       | {row['reach']:>10,.0f} | {row['model_p']:>9.1%} | "
            f"{row['observed_rho']:>10.4f} | {status:>10}"
        )

    print("\n" + "-" * 60)
    if all_pass:
        print("ALL DISTANCES PASS: Model achieves rho >= 0.95 at all tested distances.")
    else:
        print("WARNING: Some distances do not meet the rho >= 0.95 target.")

    print("\n" + "=" * 70)
    print("OUTPUTS")
    print("=" * 70)
    print(f"  1. {OUTPUT_DIR / 'gla_validation_summary.csv'}")
    print(f"  2. {FIGURES_DIR / 'fig5_gla_validation.pdf'}")
    print(f"  3. {TABLES_DIR / 'tab2_validation.tex'}")

    return 0


if __name__ == "__main__":
    exit(main())
