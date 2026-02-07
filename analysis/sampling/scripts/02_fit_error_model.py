#!/usr/bin/env python
"""
02_fit_error_model.py - Analyse localised Eppstein-Wang error bound on synthetic data.

Complements the rank-based model (01_fit_rank_model.py) with an error-based analysis.
While the rank model targets rank preservation (Spearman rho >= 0.95), the localised
EW bound targets absolute accuracy (additive epsilon guarantee).

For each (topology, distance, sample_prob, metric) configuration in the synthetic
cache, this script:
  1. Computes the observed normalised epsilon from max_abs_error
  2. Computes the EW-predicted epsilon from the Hoeffding concentration bound
  3. Checks whether the bound holds (observed <= predicted)
  4. Computes the implied epsilon from the rank model at each reach
  5. Reports success rates, comparison tables, and figures

The localised EW bound adapts Eppstein & Wang (2004) to distance-bounded centrality:
    n_eff = log(2r / delta) / (2 * epsilon^2)
where r is the network reach and delta is the failure probability.

Requires:
    - .cache/sampling_analysis_{CACHE_VERSION}.pkl (from 00_generate_cache.py)
    - output/sampling_model.json (from 01_fit_rank_model.py)

Outputs:
    - output/error_model_synthetic.json
    - output/error_model_synthetic.csv
    - paper/figures/fig_ew_synthetic.pdf
    - paper/tables/tab_ew_comparison.tex
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

from utilities import CACHE_DIR, CACHE_VERSION, FIGURES_DIR, OUTPUT_DIR, TABLES_DIR

# =============================================================================
# CONFIGURATION
# =============================================================================

DELTA = 0.1  # Failure probability (90% confidence)
EPSILON_TARGETS = [0.01, 0.05, 0.1, 0.2]  # Tolerance levels to evaluate
REACH_THRESHOLD = 100  # Minimum reach for meaningful concentration bounds

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
# EW BOUND FUNCTIONS
# =============================================================================


def ew_predicted_epsilon(n_eff: float, reach: float, delta: float = DELTA) -> float:
    """Compute the EW-predicted maximum epsilon (Hoeffding form).

    eps = sqrt(log(2r / delta) / (2 * n_eff))
    """
    if n_eff <= 0 or reach <= 0:
        return float("inf")
    return math.sqrt(math.log(2 * reach / delta) / (2 * n_eff))


def ew_required_n_eff(epsilon: float, reach: float, delta: float = DELTA) -> float:
    """Compute the n_eff required by the EW bound for a given epsilon tolerance.

    n_eff = log(2r / delta) / (2 * epsilon^2)
    """
    if epsilon <= 0:
        return float("inf")
    return math.log(2 * reach / delta) / (2 * epsilon**2)


def ew_required_p(epsilon: float, reach: float, delta: float = DELTA) -> float:
    """Compute the sampling probability required by the EW bound."""
    n_eff = ew_required_n_eff(epsilon, reach, delta)
    return min(1.0, n_eff / reach)


def implied_epsilon_from_rank_model(reach: float, k: float, min_eff_n: float, delta: float = DELTA) -> float:
    """Compute the additive error the rank model implicitly guarantees.

    Given the rank model's n_eff = max(k*sqrt(r), min_eff_n), invert the
    Hoeffding bound to find what epsilon this n_eff would guarantee:
        eps = sqrt(log(2r/delta) / (2 * n_eff))
    """
    n_eff = max(k * math.sqrt(reach), min_eff_n)
    if n_eff <= 0 or reach <= 0:
        return float("inf")
    return math.sqrt(math.log(2 * reach / delta) / (2 * n_eff))


def normalise_error(max_abs_error: float, reach: float, metric: str) -> float:
    """Normalise raw absolute error by theoretical maximum.

    Betweenness: bounded by r*(r-1) pair-paths.
    Harmonic closeness: bounded by r (sum of 1/d for r nodes).
    """
    if reach <= 1:
        return float("inf")
    if metric == "betweenness":
        return max_abs_error / (reach * (reach - 1))
    else:  # harmonic closeness
        return max_abs_error / reach


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
    return df


def load_rank_model() -> tuple[float, float]:
    """Load the fitted rank-based model parameters."""
    model_path = OUTPUT_DIR / "sampling_model.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run 01_fit_rank_model.py first.")

    with open(model_path) as f:
        model = json.load(f)

    k = model["model"]["k"]
    min_eff_n = model["model"]["min_eff_n"]
    return k, min_eff_n


# =============================================================================
# ANALYSIS
# =============================================================================


def analyse_ew_bound(df: pd.DataFrame) -> pd.DataFrame:
    """Analyse the localised EW bound on all synthetic configurations."""
    rows = []

    for _, row in df.iterrows():
        reach = row["mean_reach"]
        sample_prob = row["sample_prob"]
        metric = row["metric"]
        max_abs_error = row["max_abs_error"]

        # Skip p=1.0 (exact computation, no sampling error)
        if sample_prob >= 1.0:
            continue

        n_eff = reach * sample_prob

        # Observed normalised epsilon
        eps_obs = normalise_error(max_abs_error, reach, metric)

        # EW predicted epsilon
        eps_pred = ew_predicted_epsilon(n_eff, reach)

        # Does the bound hold?
        bound_holds = eps_obs <= eps_pred

        rows.append(
            {
                "topology": row["topology"],
                "distance": row["distance"],
                "metric": metric,
                "reach": reach,
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
    """Print bound success rates, including conditional rates."""
    print("\n" + "=" * 70)
    print("EW BOUND SUCCESS RATES")
    print("=" * 70)

    total = len(results)
    holds = results["bound_holds"].sum()
    print(f"\n  Overall: {holds}/{total} ({100 * holds / total:.1f}%) — expected >= {100 * (1 - DELTA):.0f}%")

    # By metric
    print(f"\n  {'Metric':<15} {'Holds':>8} {'Total':>8} {'Rate':>8}")
    print("  " + "-" * 45)
    for metric in sorted(results["metric"].unique()):
        subset = results[results["metric"] == metric]
        h = subset["bound_holds"].sum()
        t = len(subset)
        print(f"  {metric:<15} {h:>8} {t:>8} {100 * h / t:>7.1f}%")

    # Conditional on reach >= threshold
    print(f"\n  Conditional on reach >= {REACH_THRESHOLD}:")
    high_reach = results[results["reach"] >= REACH_THRESHOLD]
    if len(high_reach) > 0:
        h = high_reach["bound_holds"].sum()
        t = len(high_reach)
        print(f"  Overall: {h}/{t} ({100 * h / t:.1f}%)")
        for metric in sorted(high_reach["metric"].unique()):
            subset = high_reach[high_reach["metric"] == metric]
            h = subset["bound_holds"].sum()
            t = len(subset)
            print(f"    {metric:<15} {h:>8}/{t:<8} {100 * h / t:>7.1f}%")

    # By topology
    print(f"\n  {'Topology':<15} {'Holds':>8} {'Total':>8} {'Rate':>8}")
    print("  " + "-" * 45)
    for topo in sorted(results["topology"].unique()):
        subset = results[results["topology"] == topo]
        h = subset["bound_holds"].sum()
        t = len(subset)
        print(f"  {topo:<15} {h:>8} {t:>8} {100 * h / t:>7.1f}%")

    # By distance
    print(f"\n  {'Distance':<15} {'Holds':>8} {'Total':>8} {'Rate':>8}")
    print("  " + "-" * 45)
    for dist in sorted(results["distance"].unique()):
        subset = results[results["distance"] == dist]
        h = subset["bound_holds"].sum()
        t = len(subset)
        print(f"  {dist:<15} {h:>8} {t:>8} {100 * h / t:>7.1f}%")


def print_epsilon_comparison(results: pd.DataFrame, k: float, min_eff_n: float):
    """Print comparison of EW required samples vs rank-based model, with implied epsilon."""
    print("\n" + "=" * 70)
    print("COMPARISON: EW BOUND vs RANK-BASED MODEL")
    print("=" * 70)

    configs = results.groupby(["topology", "distance"])["reach"].first().reset_index()
    configs = configs.sort_values(["topology", "distance"])

    header = f"  {'Topology':<10} {'Dist':>5} {'Reach':>8} | {'Rank n_eff':>10} | {'Implied eps':>11} |"
    for eps in EPSILON_TARGETS:
        header += f" {'EW@' + str(eps):>10}"
    header += " |"
    for eps in EPSILON_TARGETS:
        header += f" {'p@' + str(eps):>8}"

    print(header)
    print("  " + "-" * len(header))

    for _, cfg in configs.iterrows():
        reach = cfg["reach"]
        rank_eff_n = max(k * math.sqrt(reach), min_eff_n)
        impl_eps = implied_epsilon_from_rank_model(reach, k, min_eff_n)

        line = f"  {cfg['topology']:<10} {cfg['distance']:>5} {reach:>8.0f} | {rank_eff_n:>10.0f} | {impl_eps:>11.4f} |"
        for eps in EPSILON_TARGETS:
            ew_n = ew_required_n_eff(eps, reach)
            line += f" {ew_n:>10.0f}"
        line += " |"
        for eps in EPSILON_TARGETS:
            ew_p = ew_required_p(eps, reach)
            line += f" {ew_p:>7.1%}"

        print(line)


def print_observed_epsilon_summary(results: pd.DataFrame):
    """Print summary of observed epsilon values at each configuration."""
    print("\n" + "=" * 70)
    print("OBSERVED NORMALISED EPSILON")
    print("=" * 70)

    betw = results[results["metric"] == "betweenness"]
    if len(betw) == 0:
        print("  No betweenness data")
        return

    print(f"\n  Betweenness:")
    print(f"  {'Topology':<10} {'Dist':>5} {'Reach':>8} | ", end="")
    probs = sorted(betw["sample_prob"].unique())
    for p in probs:
        print(f"{'p=' + f'{p:.0%}':>10}", end="")
    print()
    print("  " + "-" * (35 + 10 * len(probs)))

    for topo in sorted(betw["topology"].unique()):
        for dist in sorted(betw["distance"].unique()):
            subset = betw[(betw["topology"] == topo) & (betw["distance"] == dist)]
            if len(subset) == 0:
                continue
            reach = subset.iloc[0]["reach"]
            line = f"  {topo:<10} {dist:>5} {reach:>8.0f} | "
            for p in probs:
                row = subset[subset["sample_prob"] == p]
                if len(row) > 0:
                    eps = row.iloc[0]["eps_observed"]
                    line += f"{eps:>10.4f}"
                else:
                    line += f"{'--':>10}"
            print(line)

    harm = results[results["metric"] == "harmonic"]
    if len(harm) > 0:
        print(f"\n  Harmonic closeness:")
        print(f"  {'Topology':<10} {'Dist':>5} {'Reach':>8} | ", end="")
        for p in probs:
            print(f"{'p=' + f'{p:.0%}':>10}", end="")
        print()
        print("  " + "-" * (35 + 10 * len(probs)))

        for topo in sorted(harm["topology"].unique()):
            for dist in sorted(harm["distance"].unique()):
                subset = harm[(harm["topology"] == topo) & (harm["distance"] == dist)]
                if len(subset) == 0:
                    continue
                reach = subset.iloc[0]["reach"]
                line = f"  {topo:<10} {dist:>5} {reach:>8.0f} | "
                for p in probs:
                    row = subset[subset["sample_prob"] == p]
                    if len(row) > 0:
                        eps = row.iloc[0]["eps_observed"]
                        line += f"{eps:>10.4f}"
                    else:
                        line += f"{'--':>10}"
                print(line)


# =============================================================================
# FIGURES
# =============================================================================


def generate_figure(results: pd.DataFrame, k: float, min_eff_n: float):
    """Generate the EW analysis figure with low/high reach distinction."""
    print("\nGenerating figure...")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- Panel A: Predicted vs Observed epsilon ---
    ax = axes[0]

    colours = {"betweenness": "#B2182B", "harmonic": "#2166AC"}

    for metric in ["betweenness", "harmonic"]:
        subset = results[results["metric"] == metric]
        valid = subset[(subset["eps_observed"] > 0) & (subset["eps_predicted"] > 0)]
        valid = valid[np.isfinite(valid["eps_observed"]) & np.isfinite(valid["eps_predicted"])]

        # Split into high-reach (filled) and low-reach (open)
        high = valid[valid["reach"] >= REACH_THRESHOLD]
        low = valid[valid["reach"] < REACH_THRESHOLD]

        if len(high) > 0:
            ax.scatter(
                high["eps_predicted"],
                high["eps_observed"],
                c=colours[metric],
                marker="o",
                s=20,
                alpha=0.5,
                label=f"{metric} (r >= {REACH_THRESHOLD})",
                edgecolors="none",
            )

        if len(low) > 0:
            ax.scatter(
                low["eps_predicted"],
                low["eps_observed"],
                facecolors="none",
                edgecolors=colours[metric],
                marker="o",
                s=20,
                alpha=0.7,
                label=f"{metric} (r < {REACH_THRESHOLD})",
                linewidths=0.8,
            )

    # Identity line (bound boundary)
    lims = [1e-8, 100]
    ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5, label="bound = observed")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("EW predicted epsilon (upper bound)")
    ax.set_ylabel("Observed epsilon")
    ax.set_title("A) Bound validation: predicted vs observed")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    ax.fill_between(lims, [1e-10, 1e-10], lims, alpha=0.05, color="green")
    ax.text(
        0.95,
        0.05,
        "bound holds",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="green",
        alpha=0.7,
    )

    # --- Panel B: Required p by reach for various epsilon tolerances ---
    ax = axes[1]

    reach_range = np.logspace(1.5, 4.5, 200)

    ew_colours = ["#d73027", "#fc8d59", "#fee090", "#91bfdb"]
    for eps_target, colour in zip(EPSILON_TARGETS, ew_colours):
        p_values = [ew_required_p(eps_target, r) * 100 for r in reach_range]
        ax.plot(reach_range, p_values, linewidth=2, color=colour, label=f"EW eps={eps_target}")

    # Overlay the rank-based model
    rank_p = [min(1.0, max(k * math.sqrt(r), min_eff_n) / r) * 100 for r in reach_range]
    ax.plot(reach_range, rank_p, "k-", linewidth=2.5, label="Rank model (rho>=0.95)")

    ax.set_xscale("log")
    ax.set_xlabel("Network reach (nodes)")
    ax.set_ylabel("Required sampling probability (%)")
    ax.set_title("B) Required p: EW bound vs rank model")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = FIGURES_DIR / "fig_ew_synthetic.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


# =============================================================================
# TABLE GENERATION
# =============================================================================


def generate_comparison_table(k: float, min_eff_n: float):
    """Generate LaTeX comparison table with implied epsilon from rank model."""
    print("\nGenerating comparison table...")

    reach_values = [100, 500, 1000, 3000, 10000, 30000]

    rows = []
    for reach in reach_values:
        rank_eff_n = max(k * math.sqrt(reach), min_eff_n)
        impl_eps = implied_epsilon_from_rank_model(reach, k, min_eff_n)

        row = {"reach": reach, "rank_eff_n": rank_eff_n, "implied_eps": impl_eps}
        for eps in [0.05, 0.1, 0.2]:
            row[f"ew_{eps}"] = ew_required_n_eff(eps, reach)
        rows.append(row)

    latex = r"""\begin{table}[htbp]
\centering
\caption{Required effective sample sizes under the rank-based model ($\rho \geq \targetRho$) and the localised Eppstein--Wang bound at different additive error tolerances ($\delta = 0.1$). The final column shows the normalised additive error that the rank model's sample size would guarantee under the EW bound.}
\label{tab:ew_comparison}
\small
\begin{tabular}{@{}rrrrrr@{}}
\toprule
\textbf{Reach} & \textbf{Rank $\effn$} & \textbf{EW $\varepsilon{=}0.05$} & \textbf{EW $\varepsilon{=}0.1$} & \textbf{EW $\varepsilon{=}0.2$} & \textbf{Implied $\varepsilon$} \\
\midrule
"""

    for row in rows:
        latex += (
            f"{row['reach']:,} & "
            f"{row['rank_eff_n']:.0f} & "
            f"{row['ew_0.05']:,.0f} & "
            f"{row['ew_0.1']:,.0f} & "
            f"{row['ew_0.2']:,.0f} & "
            f"{row['implied_eps']:.3f} \\\\\n"
        )

    latex += r"""\bottomrule
\end{tabular}

\vspace{0.5em}
\footnotesize
The rank model requires substantially fewer samples than the EW bound at tight tolerances ($\varepsilon \leq 0.05$), but the two converge at $\varepsilon \approx 0.1$ for large reach. At reach above 3{,}000, the rank model implicitly guarantees $\varepsilon < 0.1$.
\end{table}
"""

    output_path = TABLES_DIR / "tab_ew_comparison.tex"
    with open(output_path, "w") as f:
        f.write(latex)
    print(f"  Saved: {output_path}")


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("=" * 70)
    print("02_fit_error_model.py - Localised EW Bound on Synthetic Data")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    df = load_synthetic_cache()
    k, min_eff_n = load_rank_model()
    print(f"  Rank model: k={k}, min_eff_n={min_eff_n}")

    # Analyse
    print("\nAnalysing EW bound...")
    results = analyse_ew_bound(df)
    print(f"  Analysed {len(results)} configurations (excluding p=1.0)")

    # Console output
    print_success_rates(results)
    print_observed_epsilon_summary(results)
    print_epsilon_comparison(results, k, min_eff_n)

    # Figure and table
    generate_figure(results, k, min_eff_n)
    generate_comparison_table(k, min_eff_n)

    # Save CSV
    csv_path = OUTPUT_DIR / "error_model_synthetic.csv"
    results.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    # Save JSON summary with conditional success rates
    total = len(results)
    holds = int(results["bound_holds"].sum())

    # Conditional success rate for reach >= threshold
    high_reach = results[results["reach"] >= REACH_THRESHOLD]
    cond_total = len(high_reach)
    cond_holds = int(high_reach["bound_holds"].sum()) if cond_total > 0 else 0

    by_metric = {}
    for metric in results["metric"].unique():
        subset = results[results["metric"] == metric]
        high_subset = subset[subset["reach"] >= REACH_THRESHOLD]
        by_metric[metric] = {
            "total": len(subset),
            "holds": int(subset["bound_holds"].sum()),
            "rate": float(subset["bound_holds"].mean()),
            "conditional_total": len(high_subset),
            "conditional_holds": int(high_subset["bound_holds"].sum()),
            "conditional_rate": float(high_subset["bound_holds"].mean()) if len(high_subset) > 0 else None,
            "median_eps_observed": float(subset["eps_observed"].median()),
            "median_eps_predicted": float(subset["eps_predicted"].median()),
        }

    # Implied epsilon at representative reach values
    implied_eps_table = {}
    for reach in [100, 500, 1000, 3000, 10000, 30000]:
        implied_eps_table[str(reach)] = round(implied_epsilon_from_rank_model(reach, k, min_eff_n), 4)

    json_output = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "script": "02_fit_error_model.py",
        "description": "Localised EW bound analysis on synthetic data",
        "parameters": {
            "delta": DELTA,
            "epsilon_targets": EPSILON_TARGETS,
            "reach_threshold": REACH_THRESHOLD,
            "cache_version": CACHE_VERSION,
        },
        "overall": {
            "total_configs": total,
            "bound_holds": holds,
            "success_rate": holds / total,
        },
        "conditional": {
            "reach_threshold": REACH_THRESHOLD,
            "total_configs": cond_total,
            "bound_holds": cond_holds,
            "success_rate": cond_holds / cond_total if cond_total > 0 else None,
        },
        "by_metric": by_metric,
        "rank_model": {"k": k, "min_eff_n": min_eff_n},
        "implied_epsilon_from_rank_model": implied_eps_table,
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
    print(f"  3. {FIGURES_DIR / 'fig_ew_synthetic.pdf'}")
    print(f"  4. {TABLES_DIR / 'tab_ew_comparison.tex'}")

    return 0


if __name__ == "__main__":
    exit(main())
