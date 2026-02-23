#!/usr/bin/env python
"""
02_fit_error_model.py - Validate concentration bounds on synthetic data.

Validates two distinct concentration bounds on the synthetic sampling cache:
  - **Closeness (harmonic):** Localised Hoeffding/EW bound (source sampling)
        k = log(2r / delta) / (2 * epsilon^2),  p = min(1, k/r)
  - **Betweenness:** Riondato-Kornaropoulos (R-K) bound (path sampling)
        VD = ceil(sqrt(r)),  m = ceil((floor(log2(VD-2)) + 1 + ln(1/delta)) / (2*eps^2))

For each (topology, distance, epsilon, metric) configuration in the synthetic
cache, this script:
  1. Computes the observed normalised epsilon from max_abs_error
  2. Computes the predicted epsilon upper bound (Hoeffding for closeness, R-K
     for betweenness)
  3. Checks whether the bound holds (observed <= predicted)
  4. Reports success rates, comparison tables, and outputs

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
    compute_rk_budget,
    ew_predicted_epsilon,
    normalise_error,
    rk_predicted_epsilon,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

DELTA = HOEFFDING_DELTA  # Failure probability (90% confidence)
EPSILON_TARGETS = [0.01, 0.05, 0.1, 0.2]  # Tolerance levels to evaluate
REACH_THRESHOLD = 100  # Minimum reach for meaningful concentration bounds


# =============================================================================
# CONVENIENCE WRAPPERS
# =============================================================================


def ew_required_n_eff(epsilon: float, reach: float, delta: float = DELTA) -> float:
    """Compute the n_eff required by the Hoeffding/EW bound for a given epsilon tolerance."""
    return compute_hoeffding_eff_n(reach, epsilon, delta)


def ew_required_p(epsilon: float, reach: float, delta: float = DELTA) -> float:
    """Compute the sampling probability required by the Hoeffding/EW bound."""
    return compute_hoeffding_p(reach, epsilon, delta)


def rk_required_m(epsilon: float, reach: float, delta: float = DELTA) -> int | None:
    """Compute the R-K path-sampling budget for a given epsilon tolerance."""
    return compute_rk_budget(reach, epsilon, delta)


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
    if "epsilon" in df.columns:
        print(f"  Epsilon sweep: {sorted(df['epsilon'].unique())}")
    return df


# =============================================================================
# ANALYSIS
# =============================================================================


def analyse_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """Analyse concentration bounds on all synthetic configurations.

    Uses Hoeffding/EW for closeness (harmonic) and R-K for betweenness.
    """
    rows = []

    for _, row in df.iterrows():
        reach = row["mean_reach"]
        metric = row["metric"]
        max_abs_error = row["max_abs_error"]

        # Observed normalised epsilon
        eps_obs = normalise_error(max_abs_error, reach, metric)

        if metric == "harmonic":
            # Closeness: Hoeffding/EW bound via source sampling
            sample_prob = row["sample_prob"]
            # Skip p=1.0 (exact computation, no sampling error)
            if sample_prob >= 1.0:
                continue
            n_eff = reach * sample_prob
            eps_pred = ew_predicted_epsilon(n_eff, reach)
            bound_type = "hoeffding"
            n_samples = None
        elif metric == "betweenness":
            # Betweenness: R-K bound via path sampling
            n_samples = row["n_samples"]
            if n_samples <= 0:
                continue
            n_eff = None
            sample_prob = None
            eps_pred = rk_predicted_epsilon(n_samples, reach)
            bound_type = "rk"
        else:
            continue

        # Does the bound hold?
        bound_holds = eps_obs <= eps_pred

        rows.append(
            {
                "topology": row["topology"],
                "distance": row["distance"],
                "metric": metric,
                "reach": reach,
                "epsilon": row.get("epsilon", None),
                "bound_type": bound_type,
                "sample_prob": sample_prob,
                "n_eff": n_eff,
                "n_samples": n_samples,
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
    print("BOUND SUCCESS RATES (Hoeffding for closeness, R-K for betweenness)")
    print("=" * 70)

    total = len(results)
    holds = results["bound_holds"].sum()
    print(f"\n  Overall: {holds}/{total} ({100 * holds / total:.1f}%) — expected >= {100 * (1 - DELTA):.0f}%")

    # By metric
    print(f"\n  {'Metric':<15} {'Bound':<12} {'Holds':>8} {'Total':>8} {'Rate':>8}")
    print("  " + "-" * 55)
    for metric in sorted(results["metric"].unique()):
        subset = results[results["metric"] == metric]
        h = subset["bound_holds"].sum()
        t = len(subset)
        bt = subset["bound_type"].iloc[0] if len(subset) > 0 else "?"
        print(f"  {metric:<15} {bt:<12} {h:>8} {t:>8} {100 * h / t:>7.1f}%")

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
    """Print required samples at different epsilon tolerances for both bounds."""
    print("\n" + "=" * 70)
    print("REQUIRED SAMPLES BY EPSILON (Hoeffding k/p + R-K m)")
    print("=" * 70)

    configs = results.groupby(["topology", "distance"])["reach"].first().reset_index()
    configs = configs.sort_values(["topology", "distance"])

    header = f"  {'Topology':<10} {'Dist':>5} {'Reach':>8} |"
    for eps in EPSILON_TARGETS:
        header += f" {'k@' + str(eps):>10} {'p@' + str(eps):>8} {'m@' + str(eps):>10}"
    print(header)
    print("  " + "-" * len(header))

    for _, cfg in configs.iterrows():
        reach = cfg["reach"]
        line = f"  {cfg['topology']:<10} {cfg['distance']:>5} {reach:>8.0f} |"
        for eps in EPSILON_TARGETS:
            ew_n = ew_required_n_eff(eps, reach)
            ew_p = ew_required_p(eps, reach)
            rk_m = rk_required_m(eps, reach)
            line += f" {ew_n:>10.0f} {ew_p:>7.1%}"
            line += f" {rk_m:>10}" if rk_m is not None else f" {'N/A':>10}"
        print(line)


def print_observed_epsilon_summary(results: pd.DataFrame):
    """Print summary of observed epsilon values at each configuration."""
    print("\n" + "=" * 70)
    print("OBSERVED NORMALISED EPSILON")
    print("=" * 70)

    # Betweenness (by n_samples)
    betw = results[results["metric"] == "betweenness"]
    if len(betw) > 0:
        print("\n  Betweenness (R-K path sampling):")
        epsilons = sorted(betw["epsilon"].dropna().unique())
        print(f"  {'Topology':<10} {'Dist':>5} {'Reach':>8} | ", end="")
        for eps in epsilons:
            print(f"{'eps=' + f'{eps}':>10}", end="")
        print()
        print("  " + "-" * (35 + 10 * len(epsilons)))

        for topo in sorted(betw["topology"].unique()):
            for dist in sorted(betw["distance"].unique()):
                subset = betw[(betw["topology"] == topo) & (betw["distance"] == dist)]
                if len(subset) == 0:
                    continue
                reach = subset.iloc[0]["reach"]
                line = f"  {topo:<10} {dist:>5} {reach:>8.0f} | "
                for eps in epsilons:
                    row = subset[subset["epsilon"] == eps]
                    if len(row) > 0:
                        eps_obs = row.iloc[0]["eps_observed"]
                        line += f"{eps_obs:>10.4f}"
                    else:
                        line += f"{'--':>10}"
                print(line)

    # Harmonic closeness (by sample_prob)
    harm = results[results["metric"] == "harmonic"]
    if len(harm) > 0:
        print("\n  Harmonic closeness (Hoeffding source sampling):")
        epsilons = sorted(harm["epsilon"].dropna().unique())
        print(f"  {'Topology':<10} {'Dist':>5} {'Reach':>8} | ", end="")
        for eps in epsilons:
            print(f"{'eps=' + f'{eps}':>10}", end="")
        print()
        print("  " + "-" * (35 + 10 * len(epsilons)))

        for topo in sorted(harm["topology"].unique()):
            for dist in sorted(harm["distance"].unique()):
                subset = harm[(harm["topology"] == topo) & (harm["distance"] == dist)]
                if len(subset) == 0:
                    continue
                reach = subset.iloc[0]["reach"]
                line = f"  {topo:<10} {dist:>5} {reach:>8.0f} | "
                for eps in epsilons:
                    row = subset[subset["epsilon"] == eps]
                    if len(row) > 0:
                        eps_obs = row.iloc[0]["eps_observed"]
                        line += f"{eps_obs:>10.4f}"
                    else:
                        line += f"{'--':>10}"
                print(line)


# =============================================================================
# TABLE GENERATION
# =============================================================================


def generate_comparison_table():
    """Generate LaTeX table comparing Hoeffding and R-K bounds at different epsilon tolerances."""
    print("\nGenerating comparison table...")

    reach_values = [100, 500, 1000, 3000, 10000, 30000]
    eps_values = [0.05, 0.1, 0.2]

    def fmt_int(n):
        """Format integer with LaTeX-safe thousands separator."""
        return f"{n:,}".replace(",", "{,}")

    def fmt_pct(p):
        """Format probability as percentage with escaped % sign."""
        return f"{p * 100:.1f}\\%"

    # Build table rows
    rows = []
    for reach in reach_values:
        row = {"reach": reach}
        for eps in eps_values:
            row[f"k_{eps}"] = ew_required_n_eff(eps, reach)
            row[f"p_{eps}"] = ew_required_p(eps, reach)
            m = rk_required_m(eps, reach)
            row[f"m_{eps}"] = m if m is not None else 0
        rows.append(row)

    # LaTeX output — structure: Reach | for each eps: k, p, m
    n_eps = len(eps_values)
    col_spec = "r" + "rrr" * n_eps  # reach + (k, p, m) per epsilon

    latex = r"""\begin{table}[htbp]
\centering
\caption{Required effective sample sizes under the Hoeffding (closeness) and
  Riondato--Kornaropoulos (betweenness) bounds at different additive error
  tolerances ($\delta = 0.1$).}
\label{tab:ew_comparison}
\small
"""
    latex += f"\\begin{{tabular}}{{@{{}}{col_spec}@{{}}}}\n"
    latex += "\\toprule\n"

    # Header row 1: epsilon groupings
    latex += "& "
    for i, eps in enumerate(eps_values):
        latex += f"\\multicolumn{{3}}{{c}}{{$\\varepsilon = {eps}$}}"
        if i < n_eps - 1:
            latex += " & "
    latex += " \\\\\n"

    # Cmidrules under each epsilon group
    for i, eps in enumerate(eps_values):
        col_start = 2 + 3 * i
        col_end = col_start + 2
        latex += f"\\cmidrule(lr){{{col_start}-{col_end}}} "
    latex += "\n"

    # Header row 2: column labels
    latex += "\\textbf{Reach}"
    for eps in eps_values:
        latex += " & \\textbf{$k$} & \\textbf{$p$} & \\textbf{$m$}"
    latex += " \\\\\n"
    latex += "\\midrule\n"

    # Data rows
    for row in rows:
        latex += fmt_int(row["reach"])
        for eps in eps_values:
            k = int(row[f"k_{eps}"])
            p = row[f"p_{eps}"]
            m = int(row[f"m_{eps}"])
            latex += f" & {fmt_int(k)} & {fmt_pct(p)} & {fmt_int(m)}"
        latex += " \\\\\n"

    latex += r"""\bottomrule
\end{tabular}

\vspace{0.5em}
\footnotesize
\textbf{Closeness} (source sampling): $k = \log(2r/\delta) / (2\varepsilon^2)$, $p = \min(1, k/r)$.
\textbf{Betweenness} (path sampling): $m = \lceil(\lfloor\log_2(\text{VD}-2)\rfloor + 1 + \ln(1/\delta)) / (2\varepsilon^2)\rceil$,
where $\text{VD} = \lceil\sqrt{r}\rceil$.
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
    print("  Hoeffding/EW for closeness, R-K for betweenness")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    df = load_synthetic_cache()

    # Analyse
    print("\nAnalysing bounds...")
    results = analyse_bounds(df)
    n_hoeff = (results["bound_type"] == "hoeffding").sum()
    n_rk = (results["bound_type"] == "rk").sum()
    print(f"  Analysed {len(results)} configurations ({n_hoeff} Hoeffding, {n_rk} R-K)")

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
        bt = subset["bound_type"].iloc[0] if len(subset) > 0 else "unknown"
        by_metric[metric] = {
            "bound_type": bt,
            "total": len(subset),
            "holds": int(subset["bound_holds"].sum()),
            "rate": float(subset["bound_holds"].mean()),
            "conditional_total": len(high_subset),
            "conditional_holds": int(high_subset["bound_holds"].sum()),
            "conditional_rate": float(high_subset["bound_holds"].mean()) if len(high_subset) > 0 else None,
            "median_eps_observed": float(subset["eps_observed"].median()),
            "median_eps_predicted": float(subset["eps_predicted"].median()),
        }

    by_topology = {}
    for topo in results["topology"].unique():
        subset = results[results["topology"] == topo]
        by_topology[topo] = {
            "total": len(subset),
            "holds": int(subset["bound_holds"].sum()),
            "rate": float(subset["bound_holds"].mean()),
        }

    by_distance = {}
    for dist in results["distance"].unique():
        subset = results[results["distance"] == dist]
        by_distance[str(dist)] = {
            "total": len(subset),
            "holds": int(subset["bound_holds"].sum()),
            "rate": float(subset["bound_holds"].mean()),
        }

    json_output = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "script": "02_fit_error_model.py",
        "description": "Concentration bound analysis on synthetic data (Hoeffding for closeness, R-K for betweenness)",
        "parameters": {
            "delta": DELTA,
            "epsilon_targets": EPSILON_TARGETS,
            "reach_threshold": REACH_THRESHOLD,
            "cache_version": CACHE_VERSION,
        },
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
        "by_metric": by_metric,
        "by_topology": by_topology,
        "by_distance": by_distance,
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
