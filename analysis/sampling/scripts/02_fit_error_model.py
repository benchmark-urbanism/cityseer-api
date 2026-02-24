#!/usr/bin/env python
"""
02_fit_error_model.py - Validate concentration bounds on synthetic data.

Validates the Hoeffding/EW concentration bound on the synthetic sampling cache
for both metrics:
  - **Closeness (harmonic):** Hoeffding/EW bound (source sampling)
        k = log(2r / delta) / (2 * epsilon^2),  p = min(1, k/r)
  - **Betweenness:** Hoeffding/EW bound (path sampling)
        m = ceil(log(2r / delta) / (2 * epsilon^2)),  capped at r(r-1)/2

Both metrics use the same concentration inequality (Hoeffding + union bound
over r nodes), differing only in the sampling primitive (sources vs paths).

For each (topology, distance, epsilon, metric) configuration in the synthetic
cache, this script:
  1. Computes the observed normalised epsilon from max_abs_error
  2. Computes the predicted epsilon upper bound (Hoeffding/EW for both)
  3. Checks whether the bound holds (observed <= predicted)
  4. Reports success rates broken down by metric x epsilon x distance

Requires:
    - .cache/sampling_analysis_{CACHE_VERSION}.pkl (from 00_generate_cache.py)

Outputs:
    - output/error_model_synthetic.json
    - output/error_model_synthetic.csv
    - paper/figures/fig_bound_success.pdf
    - paper/figures/fig_observed_vs_predicted.pdf
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
    compute_hoeffding_betw_budget,
    compute_hoeffding_eff_n,
    compute_hoeffding_p,
    ew_predicted_epsilon,
    normalise_error,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

DELTA = HOEFFDING_DELTA  # Failure probability (90% confidence)
EPSILON_TARGETS_CLOSENESS = [0.05, 0.1, 0.2]  # Closeness comparison
EPSILON_TARGETS_BETWEENNESS = [0.001, 0.002, 0.003]  # Fine sweep for betweenness
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
# CONVENIENCE WRAPPERS
# =============================================================================


def ew_required_n_eff(epsilon: float, reach: float, delta: float = DELTA) -> float:
    """Compute the n_eff required by the Hoeffding/EW bound for a given epsilon tolerance."""
    return compute_hoeffding_eff_n(reach, epsilon, delta)


def ew_required_p(epsilon: float, reach: float, delta: float = DELTA) -> float:
    """Compute the sampling probability required by the Hoeffding/EW bound."""
    return compute_hoeffding_p(reach, epsilon, delta)


def hoeffding_required_m(epsilon: float, reach: float, delta: float = DELTA) -> int | None:
    """Compute the Hoeffding path-sampling budget for a given epsilon tolerance."""
    return compute_hoeffding_betw_budget(reach, epsilon, delta)


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

    Uses Hoeffding/EW for both closeness (source sampling) and betweenness
    (path sampling). Same concentration inequality, different primitives.
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
            # Betweenness: Hoeffding/EW bound via path sampling
            n_samples = row["n_samples"]
            if n_samples <= 0:
                continue
            n_eff = None
            sample_prob = None
            eps_pred = ew_predicted_epsilon(n_samples, reach)
            bound_type = "hoeffding"
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
    """Print bound success rates broken down by metric x epsilon x distance."""
    print("\n" + "=" * 70)
    print("BOUND SUCCESS RATES (Hoeffding/EW)")
    print("=" * 70)

    # --- Closeness ---
    harm = results[results["metric"] == "harmonic"]
    if len(harm) > 0:
        print("\n  CLOSENESS (source sampling):")
        for eps in sorted(harm["epsilon"].dropna().unique()):
            subset = harm[harm["epsilon"] == eps]
            h = subset["bound_holds"].sum()
            t = len(subset)
            print(f"\n    eps={eps}:  {h}/{t} ({100 * h / t:.1f}%)")

            print(f"    {'Distance':<10} {'Reach':>8} {'Holds':>6} {'Total':>6} {'Rate':>7}  {'Med obs':>10} {'Predicted':>10}")
            print("    " + "-" * 65)
            for dist in sorted(subset["distance"].unique()):
                ds = subset[subset["distance"] == dist]
                dh = ds["bound_holds"].sum()
                dt = len(ds)
                med_obs = ds["eps_observed"].median()
                med_pred = ds["eps_predicted"].median()
                reach = ds["reach"].iloc[0]
                print(f"    {dist:<10} {reach:>8.0f} {dh:>6} {dt:>6} {100 * dh / dt:>6.1f}%  {med_obs:>10.4f} {med_pred:>10.4f}")

    # --- Betweenness ---
    betw = results[results["metric"] == "betweenness"]
    if len(betw) > 0:
        print("\n  BETWEENNESS (path sampling):")
        for eps in sorted(betw["epsilon"].dropna().unique()):
            subset = betw[betw["epsilon"] == eps]
            h = subset["bound_holds"].sum()
            t = len(subset)
            print(f"\n    eps={eps}:  {h}/{t} ({100 * h / t:.1f}%)")

            print(f"    {'Distance':<10} {'Reach':>8} {'Holds':>6} {'Total':>6} {'Rate':>7}  {'Med obs':>10} {'Predicted':>10} {'Samples':>10}")
            print("    " + "-" * 80)
            for dist in sorted(subset["distance"].unique()):
                ds = subset[subset["distance"] == dist]
                dh = ds["bound_holds"].sum()
                dt = len(ds)
                med_obs = ds["eps_observed"].median()
                med_pred = ds["eps_predicted"].median()
                reach = ds["reach"].iloc[0]
                med_samples = ds["n_samples"].median()
                print(f"    {dist:<10} {reach:>8.0f} {dh:>6} {dt:>6} {100 * dh / dt:>6.1f}%  {med_obs:>10.4f} {med_pred:>10.4f} {med_samples:>10.0f}")

    # --- By topology (betweenness only, per epsilon) ---
    if len(betw) > 0:
        print("\n  BETWEENNESS by topology:")
        for eps in sorted(betw["epsilon"].dropna().unique()):
            subset = betw[betw["epsilon"] == eps]
            print(f"\n    eps={eps}:")
            for topo in sorted(subset["topology"].unique()):
                ts = subset[subset["topology"] == topo]
                h = ts["bound_holds"].sum()
                t = len(ts)
                print(f"      {topo:<12} {h:>4}/{t:<4} ({100 * h / t:.1f}%)")


# =============================================================================
# PLOTTING
# =============================================================================


def plot_bound_success(results: pd.DataFrame):
    """Plot bound success rate vs distance, separated by metric and epsilon."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    distances = sorted(results["distance"].unique())

    # --- Left panel: Closeness ---
    harm = results[results["metric"] == "harmonic"]
    if len(harm) > 0:
        for eps in sorted(harm["epsilon"].dropna().unique()):
            subset = harm[harm["epsilon"] == eps]
            rates = []
            for dist in distances:
                ds = subset[subset["distance"] == dist]
                if len(ds) > 0:
                    rates.append(100 * ds["bound_holds"].mean())
                else:
                    rates.append(np.nan)
            ax1.plot(distances, rates, "o-", label=f"$\\varepsilon$={eps}", markersize=6)

    ax1.axhline(90, color="grey", linestyle="--", linewidth=0.8, label=f"$1-\\delta$={100*(1-DELTA):.0f}%")
    ax1.set_xlabel("Distance (m)")
    ax1.set_ylabel("Bound success rate (%)")
    ax1.set_title("Closeness (source sampling)")
    ax1.set_ylim(-5, 105)
    ax1.legend()
    ax1.set_xticks(distances)
    ax1.tick_params(axis="x", rotation=45)

    # --- Right panel: Betweenness ---
    betw = results[results["metric"] == "betweenness"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    if len(betw) > 0:
        for i, eps in enumerate(sorted(betw["epsilon"].dropna().unique())):
            subset = betw[betw["epsilon"] == eps]
            rates = []
            for dist in distances:
                ds = subset[subset["distance"] == dist]
                if len(ds) > 0:
                    rates.append(100 * ds["bound_holds"].mean())
                else:
                    rates.append(np.nan)
            color = colors[i % len(colors)]
            ax2.plot(distances, rates, "o-", label=f"$\\varepsilon$={eps}", markersize=6, color=color)

    ax2.axhline(90, color="grey", linestyle="--", linewidth=0.8, label=f"$1-\\delta$={100*(1-DELTA):.0f}%")
    ax2.set_xlabel("Distance (m)")
    ax2.set_ylabel("Bound success rate (%)")
    ax2.set_title("Betweenness (path sampling)")
    ax2.set_ylim(-5, 105)
    ax2.legend()
    ax2.set_xticks(distances)
    ax2.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    output_path = FIGURES_DIR / "fig_bound_success.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


def plot_observed_vs_predicted(results: pd.DataFrame):
    """Plot observed epsilon vs distance for each betweenness epsilon, with predicted overlay."""
    betw = results[results["metric"] == "betweenness"]
    if len(betw) == 0:
        return

    epsilons = sorted(betw["epsilon"].dropna().unique())
    n_eps = len(epsilons)
    fig, axes = plt.subplots(1, n_eps, figsize=(5 * n_eps, 5), squeeze=False)
    axes = axes[0]

    distances = sorted(betw["distance"].unique())
    topo_markers = {"trellis": "o", "linear": "s", "tree": "^"}
    topo_colors = {"trellis": "#1f77b4", "linear": "#ff7f0e", "tree": "#2ca02c"}

    for i, eps in enumerate(epsilons):
        ax = axes[i]
        subset = betw[betw["epsilon"] == eps]

        for topo in sorted(subset["topology"].unique()):
            ts = subset[subset["topology"] == topo]
            ax.scatter(
                ts["distance"],
                ts["eps_observed"],
                marker=topo_markers.get(topo, "o"),
                color=topo_colors.get(topo, "grey"),
                label=f"{topo} (obs)",
                alpha=0.7,
                s=40,
            )

        # Predicted line (median across topologies per distance)
        pred_by_dist = []
        for dist in distances:
            ds = subset[subset["distance"] == dist]
            if len(ds) > 0:
                pred_by_dist.append(ds["eps_predicted"].median())
            else:
                pred_by_dist.append(np.nan)
        ax.plot(distances, pred_by_dist, "k--", linewidth=1.5, label="predicted $\\varepsilon$")

        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Normalised $\\varepsilon$")
        ax.set_title(f"Betweenness $\\varepsilon$={eps}")
        ax.legend(fontsize=8)
        ax.set_xticks(distances)
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    output_path = FIGURES_DIR / "fig_observed_vs_predicted.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


# =============================================================================
# TABLE GENERATION
# =============================================================================


def generate_comparison_table():
    """Generate LaTeX table comparing Hoeffding bounds for closeness and betweenness."""
    print("\nGenerating comparison table...")

    reach_values = [100, 500, 1000, 3000, 10000, 30000]
    eps_closeness = EPSILON_TARGETS_CLOSENESS
    eps_betweenness = EPSILON_TARGETS_BETWEENNESS

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
        for eps in eps_closeness:
            row[f"k_{eps}"] = ew_required_n_eff(eps, reach)
            row[f"p_{eps}"] = ew_required_p(eps, reach)
        for eps in eps_betweenness:
            m = hoeffding_required_m(eps, reach)
            T = int(reach * (reach - 1) / 2)
            row[f"m_{eps}"] = m if m is not None else 0
            row[f"frac_{eps}"] = (m / T) if m is not None and T > 0 else 0
        rows.append(row)

    # LaTeX output
    n_close = len(eps_closeness)
    n_betw = len(eps_betweenness)
    col_spec = "r" + "rr" * n_close + "rr" * n_betw

    latex = r"""\begin{table}[htbp]
\centering
\caption{Required sample sizes under the Hoeffding/EW bound for
  closeness (source sampling) and betweenness (path sampling) at different
  additive error tolerances ($\delta = 0.1$).}
\label{tab:ew_comparison}
\small
"""
    latex += f"\\begin{{tabular}}{{@{{}}{col_spec}@{{}}}}\n"
    latex += "\\toprule\n"

    # Header row 1: metric groupings
    latex += " & "
    latex += f"\\multicolumn{{{2 * n_close}}}{{c}}{{Closeness (source)}} & "
    latex += f"\\multicolumn{{{2 * n_betw}}}{{c}}{{Betweenness (path)}}"
    latex += " \\\\\n"
    latex += f"\\cmidrule(lr){{2-{1 + 2 * n_close}}} "
    latex += f"\\cmidrule(lr){{{2 + 2 * n_close}-{1 + 2 * n_close + 2 * n_betw}}}\n"

    # Header row 2: epsilon values
    latex += "\\textbf{Reach}"
    for eps in eps_closeness:
        latex += f" & \\textbf{{$k$}} & \\textbf{{$p$}}"
    for eps in eps_betweenness:
        latex += f" & \\textbf{{$m$}} & \\textbf{{frac}}"
    latex += " \\\\\n"

    # Header row 3: epsilon labels
    latex += ""
    for eps in eps_closeness:
        latex += f" & \\multicolumn{{2}}{{c}}{{$\\varepsilon={eps}$}}"
    for eps in eps_betweenness:
        latex += f" & \\multicolumn{{2}}{{c}}{{$\\varepsilon={eps}$}}"
    latex += " \\\\\n"
    latex += "\\midrule\n"

    # Data rows
    for row in rows:
        latex += fmt_int(row["reach"])
        for eps in eps_closeness:
            k = int(row[f"k_{eps}"])
            p = row[f"p_{eps}"]
            if p >= 1.0:
                latex += f" & \\multicolumn{{2}}{{c}}{{exact}}"
            else:
                latex += f" & {fmt_int(k)} & {fmt_pct(p)}"
        for eps in eps_betweenness:
            m = int(row[f"m_{eps}"])
            frac = row[f"frac_{eps}"]
            if frac >= 1.0:
                latex += f" & \\multicolumn{{2}}{{c}}{{exact}}"
            else:
                latex += f" & {fmt_int(m)} & {fmt_pct(frac)}"
        latex += " \\\\\n"

    latex += r"""\bottomrule
\end{tabular}

\vspace{0.5em}
\footnotesize
\textbf{Closeness} (source sampling): $k = \log(2r/\delta) / (2\varepsilon^2)$, $p = \min(1, k/r)$.
\textbf{Betweenness} (path sampling): $m = \min\bigl(\lceil\log(2r/\delta) / (2\varepsilon^2)\rceil,\; r(r{-}1)/2\bigr)$.
Both use the same Hoeffding/EW concentration inequality.
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
    n_hoeff = (results["bound_type"] == "hoeffding").sum()
    print(f"  Analysed {len(results)} configurations ({n_hoeff} Hoeffding/EW)")

    # Console output
    print_success_rates(results)

    # Plots
    print("\nGenerating plots...")
    plot_bound_success(results)
    plot_observed_vs_predicted(results)

    # Table
    generate_comparison_table()

    # Save CSV
    csv_path = OUTPUT_DIR / "error_model_synthetic.csv"
    results.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    # Save JSON summary broken down by metric x epsilon x distance
    total = len(results)
    holds = int(results["bound_holds"].sum())

    # Conditional success rate for reach >= threshold
    high_reach = results[results["reach"] >= REACH_THRESHOLD]
    cond_total = len(high_reach)
    cond_holds = int(high_reach["bound_holds"].sum()) if cond_total > 0 else 0

    # By metric x epsilon
    by_metric_epsilon = {}
    for metric in results["metric"].unique():
        m_subset = results[results["metric"] == metric]
        by_metric_epsilon[metric] = {}
        for eps in sorted(m_subset["epsilon"].dropna().unique()):
            e_subset = m_subset[m_subset["epsilon"] == eps]
            # By distance within this metric x epsilon
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
            # By topology
            by_topo = {}
            for topo in sorted(e_subset["topology"].unique()):
                t_subset = e_subset[e_subset["topology"] == topo]
                by_topo[topo] = {
                    "total": len(t_subset),
                    "holds": int(t_subset["bound_holds"].sum()),
                    "rate": float(t_subset["bound_holds"].mean()),
                }
            by_metric_epsilon[metric][str(eps)] = {
                "total": len(e_subset),
                "holds": int(e_subset["bound_holds"].sum()),
                "rate": float(e_subset["bound_holds"].mean()),
                "median_eps_observed": float(e_subset["eps_observed"].median()),
                "median_eps_predicted": float(e_subset["eps_predicted"].median()),
                "by_distance": by_dist,
                "by_topology": by_topo,
            }

    json_output = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "script": "02_fit_error_model.py",
        "description": "Concentration bound analysis on synthetic data (Hoeffding/EW for both closeness and betweenness)",
        "parameters": {
            "delta": DELTA,
            "epsilon_targets_closeness": EPSILON_TARGETS_CLOSENESS,
            "epsilon_targets_betweenness": EPSILON_TARGETS_BETWEENNESS,
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
        "by_metric_epsilon": by_metric_epsilon,
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
    print(f"  3. {FIGURES_DIR / 'fig_bound_success.pdf'}")
    print(f"  4. {FIGURES_DIR / 'fig_observed_vs_predicted.pdf'}")
    print(f"  5. {TABLES_DIR / 'tab1_ew_comparison.tex'}")

    return 0


if __name__ == "__main__":
    exit(main())
