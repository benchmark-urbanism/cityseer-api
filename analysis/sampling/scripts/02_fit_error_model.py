#!/usr/bin/env python
"""
02_fit_error_model.py - Validate concentration bounds on synthetic data.

Validates the Hoeffding/EW concentration bound on the synthetic sampling cache
for both closeness (harmonic) and betweenness. Both metrics use the same
unified framework:
    k = log(2r / delta) / (2 * epsilon^2),  p = min(1, k/r)

For each (topology, distance, epsilon) configuration of EACH metric, this script:
  1. Computes the observed normalised epsilon from max_abs_error
  2. Computes the predicted epsilon upper bound (Hoeffding/EW)
  3. Checks whether the bound holds (observed <= predicted)
  4. Reports success rates broken down by metric x epsilon x distance

Requires:
    - .cache/sampling_analysis_{CACHE_VERSION}.pkl (from 00_generate_cache.py)

Outputs:
    - output/error_model_synthetic.json
    - output/error_model_synthetic.csv
    - paper/figures/fig2_hoeffding_model.pdf
    - paper/tables/tab1_ew_comparison.tex (main body Tab 1)
"""

import json
import pickle
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utilities import (
    CACHE_DIR,
    CACHE_VERSION,
    FIGURES_DIR,
    HOEFFDING_DELTA,
    OUTPUT_DIR,
    TABLES_DIR,
    compute_hoeffding_eff_n,
    compute_hoeffding_p,
    ew_predicted_epsilon,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

DELTA = HOEFFDING_DELTA
EPSILON_TARGETS = [0.05, 0.1, 0.2]
REACH_THRESHOLD = 100

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
# CONVENIENCE WRAPPERS
# =============================================================================


def ew_required_n_eff(epsilon: float, reach: float, delta: float = DELTA) -> float:
    """Compute the n_eff required by the Hoeffding/EW bound."""
    return compute_hoeffding_eff_n(reach, epsilon, delta)


def ew_required_p(epsilon: float, reach: float, delta: float = DELTA) -> float:
    """Compute the sampling probability required by the Hoeffding/EW bound."""
    return compute_hoeffding_p(reach, epsilon, delta)


# =============================================================================
# DATA LOADING
# =============================================================================


def load_synthetic_cache() -> pd.DataFrame:
    """Load synthetic sampling results from cache."""
    cache_path = CACHE_DIR / f"sampling_analysis_{CACHE_VERSION}.pkl"
    if not cache_path.exists():
        raise FileNotFoundError(f"Synthetic cache not found at {cache_path}. Run 00_generate_cache.py first.")

    with open(cache_path, "rb") as f:
        results = pickle.load(f)

    df = pd.DataFrame(results)
    print(f"  Loaded {len(df)} rows from synthetic cache")
    print(f"  Topologies: {sorted(df['topology'].unique())}")
    print(f"  Distances: {sorted(df['distance'].unique())}")
    print(f"  Metrics: {sorted(df['metric'].unique())}")
    for metric in df["metric"].unique():
        subset = df[df["metric"] == metric]
        epsilons = sorted(subset["epsilon"].dropna().unique())
        print(f"    {metric}: {len(subset)} rows, epsilons: {epsilons}")
    return df


# =============================================================================
# ANALYSIS
# =============================================================================


def analyse_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """Analyse Hoeffding/EW concentration bounds on all synthetic configurations.

    Both closeness and betweenness use the same bound framework.
    """
    rows = []

    for _, row in df.iterrows():
        reach = row["mean_reach"]
        metric = row["metric"]
        max_abs_error = row["max_abs_error"]

        # Observed normalised epsilon: max_abs_error / reach
        eps_obs = max_abs_error / reach if reach > 0 else np.nan

        sample_prob = row["sample_prob"]
        # Skip exact computation rows (p=1.0)
        if sample_prob >= 1.0:
            continue

        n_eff = reach * sample_prob
        eps_pred = ew_predicted_epsilon(n_eff, reach)

        bound_holds = eps_obs <= eps_pred

        rows.append(
            {
                "topology": row["topology"],
                "distance": row["distance"],
                "metric": metric,
                "reach": reach,
                "epsilon": row.get("epsilon", None),
                "bound_type": "hoeffding",
                "sample_prob": sample_prob,
                "n_eff": n_eff,
                "max_abs_error": max_abs_error,
                "eps_observed": eps_obs,
                "eps_predicted": eps_pred,
                "bound_holds": bound_holds,
                "spearman": row["spearman"],
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# CONSOLE OUTPUT
# =============================================================================


def print_success_rates(results: pd.DataFrame):
    """Print bound success rates broken down by metric x epsilon x distance."""
    print("\n" + "=" * 70)
    print("BOUND SUCCESS RATES (Hoeffding/EW)")
    print("=" * 70)

    for metric, display_name in [("harmonic", "CLOSENESS"), ("betweenness", "BETWEENNESS")]:
        subset = results[results["metric"] == metric]
        if len(subset) == 0:
            continue

        print(f"\n  {display_name}:")
        for eps in sorted(subset["epsilon"].dropna().unique()):
            eps_sub = subset[subset["epsilon"] == eps]
            h = eps_sub["bound_holds"].sum()
            t = len(eps_sub)
            print(f"\n    eps={eps}:  {h}/{t} ({100 * h / t:.1f}%)")

            print(f"    {'Distance':<10} {'Reach':>8} {'Holds':>6} {'Total':>6} {'Rate':>7}  {'Med obs':>10} {'Predicted':>10}")
            print("    " + "-" * 65)
            for dist in sorted(eps_sub["distance"].unique()):
                ds = eps_sub[eps_sub["distance"] == dist]
                dh = ds["bound_holds"].sum()
                dt = len(ds)
                med_obs = ds["eps_observed"].median()
                med_pred = ds["eps_predicted"].median()
                reach = ds["reach"].iloc[0]
                print(f"    {dist:<10} {reach:>8.0f} {dh:>6} {dt:>6} {100 * dh / dt:>6.1f}%  {med_obs:>10.4f} {med_pred:>10.4f}")


# =============================================================================
# PLOTTING
# =============================================================================


def plot_bound_success(results: pd.DataFrame):
    """Plot bound success rate vs distance for both metrics, separated by epsilon."""
    n_metrics = len(results["metric"].unique())
    fig, axes = plt.subplots(1, n_metrics, figsize=(7 * n_metrics, 5), squeeze=False)

    for ax_idx, (metric, display_name) in enumerate([("harmonic", "Closeness"), ("betweenness", "Betweenness")]):
        subset = results[results["metric"] == metric]
        if len(subset) == 0:
            continue

        ax = axes[0, ax_idx]
        distances = sorted(subset["distance"].unique())

        for eps in sorted(subset["epsilon"].dropna().unique()):
            eps_sub = subset[subset["epsilon"] == eps]
            rates = []
            for dist in distances:
                ds = eps_sub[eps_sub["distance"] == dist]
                if len(ds) > 0:
                    rates.append(100 * ds["bound_holds"].mean())
                else:
                    rates.append(np.nan)
            ax.plot(distances, rates, "o-", label=f"$\\varepsilon$={eps}", markersize=6)

        ax.axhline(90, color="grey", linestyle="--", linewidth=0.8, label=f"$1-\\delta$={100*(1-DELTA):.0f}%")
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Bound success rate (%)")
        ax.set_title(display_name)
        ax.set_ylim(-5, 105)
        ax.legend()
        ax.set_xticks(distances)
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    output_path = FIGURES_DIR / "fig2_hoeffding_model.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


# =============================================================================
# TABLE GENERATION
# =============================================================================


def generate_comparison_table():
    """Generate LaTeX table showing Hoeffding/EW bounds for both metrics."""
    print("\nGenerating comparison table...")

    reach_values = [100, 500, 1000, 3000, 10000, 30000]

    def fmt_int(n):
        return f"{n:,}".replace(",", "{,}")

    def fmt_pct(p):
        return f"{p * 100:.1f}\\%"

    rows = []
    for reach in reach_values:
        row = {"reach": reach}
        for eps in EPSILON_TARGETS:
            row[f"k_{eps}"] = ew_required_n_eff(eps, reach)
            row[f"p_{eps}"] = ew_required_p(eps, reach)
        rows.append(row)

    n_eps = len(EPSILON_TARGETS)
    col_spec = "r" + "rr" * n_eps

    latex = r"""\begin{table}[htbp]
\centering
\caption{Required sample sizes under the Hoeffding/EW bound
  at different additive error tolerances ($\delta = 0.1$).
  Both closeness and betweenness use the same formula:
  $k = \log(2r/\delta) / (2\varepsilon^2)$, $p = \min(1, k/r)$.}
\label{tab:ew_comparison}
\small
"""
    latex += f"\\begin{{tabular}}{{@{{}}{col_spec}@{{}}}}\n"
    latex += "\\toprule\n"

    latex += "\\textbf{Reach}"
    for eps in EPSILON_TARGETS:
        latex += f" & \\textbf{{$k$}} & \\textbf{{$p$}}"
    latex += " \\\\\n"

    latex += ""
    for eps in EPSILON_TARGETS:
        latex += f" & \\multicolumn{{2}}{{c}}{{$\\varepsilon={eps}$}}"
    latex += " \\\\\n"
    latex += "\\midrule\n"

    for row in rows:
        latex += fmt_int(row["reach"])
        for eps in EPSILON_TARGETS:
            k = int(row[f"k_{eps}"])
            p = row[f"p_{eps}"]
            if p >= 1.0:
                latex += f" & \\multicolumn{{2}}{{c}}{{exact}}"
            else:
                latex += f" & {fmt_int(k)} & {fmt_pct(p)}"
        latex += " \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    output_path = TABLES_DIR / "tab1_ew_comparison.tex"
    with open(output_path, "w") as f:
        f.write(latex)
    print(f"  Saved: {output_path}")


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("=" * 70)
    print("02_fit_error_model.py - Concentration Bounds on Synthetic Data")
    print("  Hoeffding/EW for both closeness and betweenness")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    df = load_synthetic_cache()

    # Analyse
    print("\nAnalysing bounds...")
    results = analyse_bounds(df)
    print(f"  Analysed {len(results)} configurations")
    for metric in results["metric"].unique():
        n = (results["metric"] == metric).sum()
        print(f"    {metric}: {n}")

    # Console output
    print_success_rates(results)

    # Plots
    print("\nGenerating plots...")
    plot_bound_success(results)

    # Table
    generate_comparison_table()

    # Save CSV
    csv_path = OUTPUT_DIR / "error_model_synthetic.csv"
    results.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    # Save JSON summary
    json_output = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "script": "02_fit_error_model.py",
        "description": "Hoeffding/EW concentration bound analysis on synthetic data (both metrics)",
        "parameters": {
            "delta": DELTA,
            "epsilon_targets": EPSILON_TARGETS,
            "reach_threshold": REACH_THRESHOLD,
            "cache_version": CACHE_VERSION,
        },
    }

    for metric, display_name in [("harmonic", "closeness"), ("betweenness", "betweenness")]:
        metric_results = results[results["metric"] == metric]
        if len(metric_results) == 0:
            continue

        total = len(metric_results)
        holds = int(metric_results["bound_holds"].sum())

        high_reach = metric_results[metric_results["reach"] >= REACH_THRESHOLD]
        cond_total = len(high_reach)
        cond_holds = int(high_reach["bound_holds"].sum()) if cond_total > 0 else 0

        by_epsilon = {}
        for eps in sorted(metric_results["epsilon"].dropna().unique()):
            e_subset = metric_results[metric_results["epsilon"] == eps]
            by_dist = {}
            for dist in sorted(e_subset["distance"].unique()):
                d_subset = e_subset[e_subset["distance"] == dist]
                by_dist[str(dist)] = {
                    "total": len(d_subset),
                    "holds": int(d_subset["bound_holds"].sum()),
                    "rate": float(d_subset["bound_holds"].mean()),
                    "median_eps_observed": float(d_subset["eps_observed"].median()),
                    "median_eps_predicted": float(d_subset["eps_predicted"].median()),
                    "median_spearman": float(d_subset["spearman"].median()),
                }
            by_epsilon[str(eps)] = {
                "total": len(e_subset),
                "holds": int(e_subset["bound_holds"].sum()),
                "rate": float(e_subset["bound_holds"].mean()),
                "by_distance": by_dist,
            }

        json_output[display_name] = {
            "overall": {
                "total_configs": total,
                "bound_holds": holds,
                "success_rate": holds / total if total > 0 else None,
            },
            "conditional": {
                "reach_threshold": REACH_THRESHOLD,
                "total_configs": cond_total,
                "bound_holds": cond_holds,
                "success_rate": cond_holds / cond_total if cond_total > 0 else None,
            },
            "by_epsilon": by_epsilon,
        }

    json_path = OUTPUT_DIR / "error_model_synthetic.json"
    with open(json_path, "w") as f:
        json.dump(json_output, f, indent=2)
    print(f"  Saved: {json_path}")

    print("\n" + "=" * 70)
    print("OUTPUTS")
    print("=" * 70)
    print(f"  1. {csv_path}")
    print(f"  2. {json_path}")
    print(f"  3. {FIGURES_DIR / 'fig2_hoeffding_model.pdf'}")
    print(f"  4. {TABLES_DIR / 'tab1_ew_comparison.tex'}")

    return 0


if __name__ == "__main__":
    exit(main())
