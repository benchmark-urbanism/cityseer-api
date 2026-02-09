#!/usr/bin/env python
"""
02_fit_error_model.py - Validate localised Hoeffding/EW bound on synthetic data.

This is the PRIMARY calibration script for the Hoeffding-based sampling model.
The localised Hoeffding bound (adapting Eppstein & Wang 2004) prescribes:
    k = log(2r / delta) / (2 * epsilon^2)
    p = min(1, k / r)
where r is network reach, epsilon is normalised error tolerance, delta is failure
probability. This model has ZERO fitted parameters for conventional (epsilon=0.1,
delta=0.1).

For each (topology, distance, sample_prob, metric) configuration in the synthetic
cache, this script:
  1. Computes the observed normalised epsilon from max_abs_error
  2. Computes the Hoeffding-predicted epsilon upper bound
  3. Checks whether the bound holds (observed <= predicted)
  4. Reports success rates, comparison tables, and figures

Requires:
    - .cache/sampling_analysis_{CACHE_VERSION}.pkl (from 00_generate_cache.py)

Outputs:
    - output/error_model_synthetic.json
    - output/error_model_synthetic.csv
    - paper/tables/tab1_ew_comparison.tex (main body Tab 1)
"""

import json
import pickle
from datetime import datetime

import pandas as pd
from utilities import (
    CACHE_DIR,
    CACHE_VERSION,
    HOEFFDING_DELTA,
    OUTPUT_DIR,
    TABLES_DIR,
    compute_hoeffding_eff_n,
    compute_hoeffding_p,
    ew_predicted_epsilon,
    normalise_error,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

DELTA = HOEFFDING_DELTA  # Failure probability (90% confidence)
EPSILON_TARGETS = [0.01, 0.05, 0.1, 0.2]  # Tolerance levels to evaluate
REACH_THRESHOLD = 100  # Minimum reach for meaningful concentration bounds


# =============================================================================
# EW BOUND FUNCTIONS (local wrappers for convenience)
# =============================================================================


def ew_required_n_eff(epsilon: float, reach: float, delta: float = DELTA) -> float:
    """Compute the n_eff required by the EW bound for a given epsilon tolerance."""
    return compute_hoeffding_eff_n(reach, epsilon, delta)


def ew_required_p(epsilon: float, reach: float, delta: float = DELTA) -> float:
    """Compute the sampling probability required by the EW bound."""
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
    return df


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


def print_epsilon_comparison(results: pd.DataFrame):
    """Print EW bound required samples at different epsilon tolerances."""
    print("\n" + "=" * 70)
    print("EW BOUND: REQUIRED SAMPLES BY EPSILON")
    print("=" * 70)

    configs = results.groupby(["topology", "distance"])["reach"].first().reset_index()
    configs = configs.sort_values(["topology", "distance"])

    header = f"  {'Topology':<10} {'Dist':>5} {'Reach':>8} |"
    for eps in EPSILON_TARGETS:
        header += f" {'EW@' + str(eps):>10}"
    header += " |"
    for eps in EPSILON_TARGETS:
        header += f" {'p@' + str(eps):>8}"

    print(header)
    print("  " + "-" * len(header))

    for _, cfg in configs.iterrows():
        reach = cfg["reach"]

        line = f"  {cfg['topology']:<10} {cfg['distance']:>5} {reach:>8.0f} |"
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

    print("\n  Betweenness:")
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
        print("\n  Harmonic closeness:")
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
# TABLE GENERATION
# =============================================================================


def generate_comparison_table():
    """Generate LaTeX table of EW bound required samples at different epsilon tolerances."""
    print("\nGenerating comparison table...")

    reach_values = [100, 500, 1000, 3000, 10000, 30000]

    rows = []
    for reach in reach_values:
        row = {"reach": reach}
        for eps in [0.05, 0.1, 0.2]:
            row[f"ew_{eps}"] = ew_required_n_eff(eps, reach)
            row[f"p_{eps}"] = ew_required_p(eps, reach)
        rows.append(row)

    latex = r"""\begin{table}[htbp]
\centering
\caption{Required effective sample sizes and sampling probabilities under the localised
  Eppstein--Wang bound at different additive error tolerances ($\delta = 0.1$).}
\label{tab:ew_comparison}
\small
\begin{tabular}{@{}rrrrrrr@{}}
\toprule
\textbf{Reach} & \textbf{EW $\varepsilon{=}0.05$} & \textbf{$p$} & \textbf{EW $\varepsilon{=}0.1$} &
  \textbf{$p$} & \textbf{EW $\varepsilon{=}0.2$} & \textbf{$p$} \\\\
\midrule
"""

    def fmt_int(n):
        """Format integer with LaTeX-safe thousands separator."""
        return f"{n:,}".replace(",", "{,}")

    def fmt_pct(p):
        """Format probability as percentage with escaped % sign."""
        return f"{p * 100:.1f}\\%"

    for row in rows:
        latex += (
            f"{fmt_int(row['reach'])} & "
            f"{fmt_int(int(row['ew_0.05']))} & {fmt_pct(row['p_0.05'])} & "
            f"{fmt_int(int(row['ew_0.1']))} & {fmt_pct(row['p_0.1'])} & "
            f"{fmt_int(int(row['ew_0.2']))} & {fmt_pct(row['p_0.2'])} \\\\\n"
        )

    latex += r"""\bottomrule
\end{tabular}

\vspace{0.5em}
\footnotesize
The Hoeffding model prescribes $k = \log(2r/\delta) / (2\varepsilon^2)$, $p = \min(1, k/r)$.
At the default $\varepsilon = 0.1$, $\delta = 0.1$, the model has zero fitted parameters.
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
    print("02_fit_error_model.py - Localised EW Bound on Synthetic Data")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    df = load_synthetic_cache()

    # Analyse
    print("\nAnalysing EW bound...")
    results = analyse_ew_bound(df)
    print(f"  Analysed {len(results)} configurations (excluding p=1.0)")

    # Console output
    print_success_rates(results)
    print_observed_epsilon_summary(results)
    print_epsilon_comparison(results)

    # Table
    generate_comparison_table()

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
    print(f"  3. {TABLES_DIR / 'tab1_ew_comparison.tex'}")

    return 0


if __name__ == "__main__":
    exit(main())
