#!/usr/bin/env python
"""
03_validate_gla.py - Validate the sampling model on Greater London network.

Generates validation data (if not cached) and produces figures/tables.
Requires sampling_model.json from 01_fit_rank_model.py.

Usage:
    python 03_validate_gla.py           # Run (skips cache if exists)
    python 03_validate_gla.py --force   # Force regeneration of validation data

Outputs:
    - output/gla_validation_{CACHE_VERSION}.csv
    - output/gla_validation_summary.csv
    - output/gla_theoretical_bounds_comparison.csv
    - paper/figures/fig6_gla_validation.pdf
    - paper/tables/tab2_validation.tex
"""

import argparse
import json
import math
import pickle
import time
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from cityseer.tools import graphs, io
from utilities import (
    CACHE_DIR,
    CACHE_VERSION,
    FIGURES_DIR,
    OUTPUT_DIR,
    QUARTILE_KEYS,
    TABLES_DIR,
    apply_live_buffer_nx,
    compute_accuracy_metrics,
    compute_quartile_accuracy,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent

# GLA network file
GLA_GPKG_FILE = SCRIPT_DIR.parent.parent.parent / "temp" / "os_open_roads" / "gla.gpkg"

# Validation parameters
LIVE_INWARD_BUFFER = 20000  # 20km buffer
GLA_DISTANCES = [1000, 2000, 5000, 10000, 20000]
GLA_PROBS = [0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
N_RUNS = 1

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

DELTA = 0.1  # Failure probability (90% confidence)


def ew_predicted_epsilon(n_eff: float, reach: float, delta: float = DELTA) -> float:
    """EW-predicted maximum normalised epsilon (Hoeffding form)."""
    if n_eff <= 0 or reach <= 0:
        return float("inf")
    return math.sqrt(math.log(2 * reach / delta) / (2 * n_eff))


def normalise_error(max_abs_error: float, reach: float, metric: str) -> float:
    """Normalise raw absolute error by theoretical maximum."""
    if reach <= 1:
        return float("inf")
    if metric == "betweenness":
        return max_abs_error / (reach * (reach - 1))
    else:  # harmonic closeness
        return max_abs_error / reach


def implied_epsilon_from_rank_model(reach: float, k: float, min_eff_n: float, delta: float = DELTA) -> float:
    """Additive error the rank model implicitly guarantees at this reach."""
    n_eff = max(k * math.sqrt(reach), min_eff_n)
    if n_eff <= 0 or reach <= 0:
        return float("inf")
    return math.sqrt(math.log(2 * reach / delta) / (2 * n_eff))


# =============================================================================
# DATA GENERATION
# =============================================================================


def load_model() -> tuple[float, int]:
    """Load the fitted model parameters."""
    model_path = OUTPUT_DIR / "sampling_model.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run 01_fit_rank_model.py first.")

    with open(model_path) as f:
        model = json.load(f)

    k = model["model"]["k"]
    min_eff_n = model["model"]["min_eff_n"]
    return k, min_eff_n


def generate_validation_data(force: bool = False) -> pd.DataFrame:
    """Generate GLA validation data, or load from cache."""
    validation_csv = OUTPUT_DIR / f"gla_validation_{CACHE_VERSION}.csv"

    if validation_csv.exists() and not force:
        print(f"Loading cached validation data from {validation_csv}")
        return pd.read_csv(validation_csv)

    if not GLA_GPKG_FILE.exists():
        raise FileNotFoundError(
            f"GLA network file not found: {GLA_GPKG_FILE}\n  Download the OS Open Roads dataset for Greater London"
        )

    print("\n" + "=" * 70)
    print("GENERATING GLA VALIDATION DATA")
    print("=" * 70)

    # Load or build GLA graph
    gla_cache = CACHE_DIR / "gla_graph.pkl"
    if gla_cache.exists() and not force:
        print(f"Loading cached GLA graph from {gla_cache}")
        with open(gla_cache, "rb") as f:
            G = pickle.load(f)
    else:
        print(f"Loading GLA network from {GLA_GPKG_FILE}")
        edges_gdf = gpd.read_file(GLA_GPKG_FILE)
        edges_gdf = edges_gdf[edges_gdf.geometry.is_valid & ~edges_gdf.geometry.is_empty]
        edges_gdf = edges_gdf.explode(index_parts=False)

        print("  Building graph...")
        G = io.nx_from_generic_geopandas(edges_gdf)
        G = graphs.nx_remove_filler_nodes(G)
        G = graphs.nx_remove_dangling_nodes(G)
        G = graphs.nx_consolidate_nodes(G, buffer_dist=1)
        G = graphs.nx_remove_dangling_nodes(G, despine=20)

        print(f"  Caching to {gla_cache}")
        with open(gla_cache, "wb") as f:
            pickle.dump(G, f)

    print(f"GLA graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Apply live buffer
    print(f"Applying {LIVE_INWARD_BUFFER / 1000:.0f}km inward buffer...")
    G = apply_live_buffer_nx(G, LIVE_INWARD_BUFFER)

    # Convert to cityseer
    print("Converting to cityseer format...")
    nodes_gdf, edges_gdf_out, net = io.network_structure_from_nx(G)

    results = []

    for dist in GLA_DISTANCES:
        print(f"\n{'=' * 50}")
        print(f"Distance: {dist}m")
        print(f"{'=' * 50}")

        # Check for cached ground truth
        gt_cache = CACHE_DIR / f"gla_ground_truth_{dist}m.pkl"
        if gt_cache.exists() and not force:
            print(f"  Loading cached ground truth from {gt_cache}")
            with open(gt_cache, "rb") as f:
                gt_data = pickle.load(f)
            true_harmonic = gt_data["harmonic"]
            true_betweenness = gt_data["betweenness"]
            mean_reach = gt_data["mean_reach"]
            node_reach = gt_data.get("node_reach", None)
            baseline_time = gt_data.get("baseline_time", None)
        else:
            print("  Computing ground truth (this may take a while)...")
            t0 = time.time()
            true_result = net.local_node_centrality_shortest(
                distances=[dist],
                compute_closeness=True,
                compute_betweenness=True,
                pbar_disabled=False,
            )
            baseline_time = time.time() - t0
            true_harmonic = np.array(true_result.node_harmonic[dist])
            true_betweenness = np.array(true_result.node_betweenness[dist])
            node_reach = np.array(true_result.node_density[dist])
            mean_reach = float(np.mean(node_reach))

            with open(gt_cache, "wb") as f:
                pickle.dump(
                    {
                        "harmonic": true_harmonic,
                        "betweenness": true_betweenness,
                        "node_reach": node_reach,
                        "mean_reach": mean_reach,
                        "baseline_time": baseline_time,
                    },
                    f,
                )
            print(f"  Cached ground truth to {gt_cache}")

        print(f"  Mean reach: {mean_reach:.0f}")

        for p in GLA_PROBS:
            effective_n = mean_reach * p
            print(f"    p={p:.2f} (eff_n={effective_n:.0f}): ", end="", flush=True)

            spearmans_h, spearmans_b = [], []
            maes_h, maes_b = [], []
            precs_h, precs_b = [], []
            scales_h, scales_b = [], []
            quartiles_h, quartiles_b = [], []
            sampled_times = []

            for seed in range(N_RUNS):
                t0 = time.time()
                r = net.local_node_centrality_shortest(
                    distances=[dist],
                    compute_closeness=True,
                    compute_betweenness=True,
                    sample_probability=p,
                    random_seed=seed,
                    pbar_disabled=True,
                )
                sampled_times.append(time.time() - t0)

                est_harmonic = np.array(r.node_harmonic[dist])
                est_betweenness = np.array(r.node_betweenness[dist])

                sp_h, prec_h, scale_h, iqr_h, mae_h = compute_accuracy_metrics(true_harmonic, est_harmonic)
                sp_b, prec_b, scale_b, iqr_b, mae_b = compute_accuracy_metrics(true_betweenness, est_betweenness)

                if not np.isnan(sp_h):
                    spearmans_h.append(sp_h)
                    maes_h.append(mae_h)
                    precs_h.append(prec_h)
                    scales_h.append(scale_h)
                if not np.isnan(sp_b):
                    spearmans_b.append(sp_b)
                    maes_b.append(mae_b)
                    precs_b.append(prec_b)
                    scales_b.append(scale_b)

                # Per-reachability-quartile accuracy
                if node_reach is not None:
                    quartiles_h.append(compute_quartile_accuracy(true_harmonic, est_harmonic, node_reach))
                    quartiles_b.append(compute_quartile_accuracy(true_betweenness, est_betweenness, node_reach))

                print(".", end="", flush=True)

            mean_sampled_time = np.mean(sampled_times) if sampled_times else float("nan")

            # Average quartile results across runs
            def _mean_quartiles(quartile_list):
                if not quartile_list:
                    result = {}
                    for prefix in QUARTILE_KEYS:
                        for q in range(1, 5):
                            result[f"{prefix}_q{q}"] = np.nan
                    return result
                result = {}
                for key in quartile_list[0]:
                    vals = [q[key] for q in quartile_list if not np.isnan(q[key])]
                    result[key] = float(np.mean(vals)) if vals else np.nan
                return result

            q_h = _mean_quartiles(quartiles_h)
            q_b = _mean_quartiles(quartiles_b)

            if spearmans_h:
                row_h = {
                    "distance": dist,
                    "mean_reach": mean_reach,
                    "sample_prob": p,
                    "effective_n": effective_n,
                    "metric": "harmonic",
                    "spearman": np.mean(spearmans_h),
                    "spearman_std": np.std(spearmans_h),
                    "max_abs_error": np.mean(maes_h),
                    "max_abs_error_std": np.std(maes_h),
                    "top_k_precision": np.mean(precs_h),
                    "scale_ratio": np.mean(scales_h),
                    "baseline_time": baseline_time if baseline_time is not None else float("nan"),
                    "sampled_time": mean_sampled_time,
                    "speedup": (
                        baseline_time / mean_sampled_time
                        if baseline_time and mean_sampled_time > 0
                        else float("nan")
                    ),
                }
                row_h.update(q_h)
                results.append(row_h)

            if spearmans_b:
                row_b = {
                    "distance": dist,
                    "mean_reach": mean_reach,
                    "sample_prob": p,
                    "effective_n": effective_n,
                    "metric": "betweenness",
                    "spearman": np.mean(spearmans_b),
                    "spearman_std": np.std(spearmans_b),
                    "max_abs_error": np.mean(maes_b),
                    "max_abs_error_std": np.std(maes_b),
                    "top_k_precision": np.mean(precs_b),
                    "scale_ratio": np.mean(scales_b),
                    "baseline_time": baseline_time if baseline_time is not None else float("nan"),
                    "sampled_time": mean_sampled_time,
                    "speedup": baseline_time / mean_sampled_time
                    if baseline_time and mean_sampled_time > 0
                    else float("nan"),
                }
                row_b.update(q_b)
                results.append(row_b)

            print(f" rho_h={np.mean(spearmans_h):.3f}, rho_b={np.mean(spearmans_b):.3f}")

    df = pd.DataFrame(results)
    df.to_csv(validation_csv, index=False)
    print(f"\nSaved validation results: {validation_csv}")
    return df


def pivot_validation_data(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot per-metric rows into combined columns for analysis."""
    harmonic = df[df["metric"] == "harmonic"].copy()
    betweenness = df[df["metric"] == "betweenness"].copy()

    harmonic = harmonic.rename(
        columns={
            "spearman": "obs_rho_h",
            "spearman_std": "obs_rho_h_std",
            "max_abs_error": "max_abs_error_h",
        }
    )
    betweenness = betweenness.rename(
        columns={
            "spearman": "obs_rho_b",
            "spearman_std": "obs_rho_b_std",
            "max_abs_error": "max_abs_error_b",
        }
    )

    h_cols = ["distance", "sample_prob", "mean_reach", "effective_n", "obs_rho_h", "obs_rho_h_std"]
    b_cols = ["distance", "sample_prob", "obs_rho_b", "obs_rho_b_std"]
    if "max_abs_error_h" in harmonic.columns:
        h_cols.append("max_abs_error_h")
    if "max_abs_error_b" in betweenness.columns:
        b_cols.append("max_abs_error_b")

    merged = pd.merge(harmonic[h_cols], betweenness[b_cols], on=["distance", "sample_prob"])
    merged = merged.rename(columns={"mean_reach": "reach", "effective_n": "eff_n"})
    return merged


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
    """Figure 5: GLA validation results."""
    print("\nGenerating Figure 5: GLA validation...")

    distances = sorted(df["distance"].unique())
    n_distances = len(distances)

    fig, axes = plt.subplots(1, n_distances, figsize=(4 * n_distances, 5), sharey=True)
    if n_distances == 1:
        axes = [axes]

    color_b = "#B2182B"  # red for betweenness
    color_h = "#2166AC"  # blue for closeness

    for i, dist in enumerate(distances):
        ax = axes[i]
        subset = df[df["distance"] == dist].sort_values("sample_prob")

        reach = subset["reach"].iloc[0]
        model_p = compute_model_p(reach, k, min_eff_n) * 100

        ax.plot(
            subset["sample_prob"] * 100,
            subset["obs_rho_b"],
            "o-",
            color=color_b,
            linewidth=2,
            markersize=5,
            label="Betweenness",
        )
        ax.plot(
            subset["sample_prob"] * 100,
            subset["obs_rho_h"],
            "s-",
            color=color_h,
            linewidth=2,
            markersize=5,
            label="Closeness",
        )

        ax.axhline(0.95, color="green", linestyle="--", linewidth=1.5, alpha=0.7)
        ax.text(95, 0.952, "target", fontsize=8, color="green", ha="right", va="bottom")

        ax.axvline(model_p, color="#0072B2", linestyle="-", linewidth=2, alpha=0.8)
        ax.annotate(
            f"Model: {model_p:.1f}%", xy=(model_p + 2, 0.96), fontsize=9, color="#0072B2", rotation=90, va="bottom"
        )

        closest_idx = (subset["sample_prob"] * 100 - model_p).abs().argmin()
        model_rho_b = subset.iloc[closest_idx]["obs_rho_b"]
        ax.plot(model_p, model_rho_b, "s", color="#0072B2", markersize=10, zorder=5)

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

    output_path = FIGURES_DIR / "fig6_gla_validation.pdf"
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

        model_p = compute_model_p(reach, k, min_eff_n)
        model_eff_n = reach * model_p

        closest_idx = (subset["sample_prob"] - model_p).abs().argmin()
        actual_p = subset.iloc[closest_idx]["sample_prob"]
        observed_rho_b = subset.iloc[closest_idx]["obs_rho_b"]
        observed_rho_h = subset.iloc[closest_idx]["obs_rho_h"]

        speedup = 1 / model_p if model_p > 0 else float("inf")

        row_data = {
            "distance": dist,
            "reach": reach,
            "model_p": model_p,
            "model_eff_n": model_eff_n,
            "actual_p": actual_p,
            "observed_rho_h": observed_rho_h,
            "observed_rho_b": observed_rho_b,
            "speedup": speedup,
            "meets_target": observed_rho_b >= 0.95 and observed_rho_h >= 0.95,
        }
        rows.append(row_data)

    results_df = pd.DataFrame(rows)

    csv_path = OUTPUT_DIR / "gla_validation_summary.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # Generate LaTeX
    latex = r"""\begin{table}[htbp]
\centering
\caption{Model Validation on Greater London Network}
\label{tab:validation}
\begin{tabular}{rrrrrrrr}
\toprule
\textbf{Distance} & \textbf{Reach} & \textbf{Model $p$} &
\textbf{$\rho_\text{close}$} & \textbf{$\rho_\text{betw}$} & \textbf{Speedup} & \textbf{$\rho \geq 0.95$?} \\
\midrule
"""

    for _, row in results_df.iterrows():
        check = r"\checkmark" if row["meets_target"] else r"\texttimes"
        model_p_pct = f"{row['model_p'] * 100:.1f}\\%"
        latex += f"{row['distance'] // 1000}km & {row['reach']:,.0f} & {model_p_pct} & "
        latex += f"{row['observed_rho_h']:.4f} & {row['observed_rho_b']:.4f} & "
        latex += f"{row['speedup']:.1f}$\\times$ & {check} \\\\\n"

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
# THEORETICAL BOUNDS COMPARISON
# =============================================================================


def get_n_nodes(force: bool = False) -> int | None:
    """Get total node count from cached GLA graph."""
    gla_cache = CACHE_DIR / "gla_graph.pkl"
    if not gla_cache.exists():
        return None
    with open(gla_cache, "rb") as f:
        G = pickle.load(f)
    # Apply same buffer as validation to get live node count
    G = apply_live_buffer_nx(G, LIVE_INWARD_BUFFER)
    return G.number_of_nodes()


def compute_theoretical_bounds(results_df: pd.DataFrame, n_nodes: int):
    """Compare empirical sample counts against Riondato, Bader, and Eppstein-Wang bounds."""
    if "max_abs_error" not in results_df.columns:
        print("\n  Skipping theoretical bounds comparison (no max_abs_error data)")
        return None

    print("\nComputing theoretical bounds comparison...")

    delta = 0.1  # Failure probability (90% confidence)
    rows = []

    for _, row in results_df.iterrows():
        reach = row["reach"]
        our_eff_n = row["model_eff_n"]
        raw_eps = row["max_abs_error"]

        eps_normalised = raw_eps / (reach * (reach - 1)) if reach > 1 else float("inf")

        if eps_normalised <= 0 or not np.isfinite(eps_normalised):
            continue

        # Riondato & Kornaropoulos (2016): VC-dimension bound for betweenness.
        # VD estimate: for planar graphs, vertex-diameter is O(sqrt(n));
        # we use sqrt(reach) as a heuristic for the distance-bounded subgraph.
        vd = max(3, int(np.sqrt(reach)))
        vc_dim = int(np.floor(np.log2(max(1, vd - 2)))) + 1
        riondato_samples = (1 / (2 * eps_normalised**2)) * (vc_dim + np.log(1 / delta))

        # Bader et al. (2007): per-vertex bound for betweenness.
        bader_samples = 1 / eps_normalised**2

        # Eppstein & Wang (2004): source-sampling bound for closeness.
        # Global: O(log(n) / eps^2) using total node count.
        eppstein_samples = np.log(n_nodes) / eps_normalised**2
        # Localised adaptation: O(log(r) / eps^2) using reach instead of n.
        eppstein_local_samples = np.log(reach) / eps_normalised**2

        rows.append(
            {
                "distance": row["distance"],
                "reach": reach,
                "our_eff_n": our_eff_n,
                "raw_eps": raw_eps,
                "eps_normalised": eps_normalised,
                "vd_estimate": vd,
                "vc_dim": vc_dim,
                "riondato_samples": riondato_samples,
                "bader_samples": bader_samples,
                "eppstein_samples": eppstein_samples,
                "eppstein_local_samples": eppstein_local_samples,
                "ratio_riondato": riondato_samples / our_eff_n if our_eff_n > 0 else float("inf"),
                "ratio_bader": bader_samples / our_eff_n if our_eff_n > 0 else float("inf"),
                "ratio_eppstein": eppstein_samples / our_eff_n if our_eff_n > 0 else float("inf"),
                "ratio_eppstein_local": eppstein_local_samples / our_eff_n if our_eff_n > 0 else float("inf"),
            }
        )

    if not rows:
        return None

    bounds_df = pd.DataFrame(rows)

    csv_path = OUTPUT_DIR / "gla_theoretical_bounds_comparison.csv"
    bounds_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    print(
        f"\n  {'Distance':>10} | {'Our eff_n':>10} | {'EW global':>12} | {'EW local':>12} | "
        f"{'Riondato':>12} | {'Bader':>12} | {'R(EWg)':>8} | {'R(EWl)':>8} | {'R(Rio)':>8} | {'R(Bad)':>8}"
    )
    print("  " + "-" * 130)
    for _, r in bounds_df.iterrows():
        print(
            f"  {r['distance'] // 1000}km       | "
            f"{r['our_eff_n']:>10,.0f} | "
            f"{r['eppstein_samples']:>12,.0f} | "
            f"{r['eppstein_local_samples']:>12,.0f} | "
            f"{r['riondato_samples']:>12,.0f} | "
            f"{r['bader_samples']:>12,.0f} | "
            f"{r['ratio_eppstein']:>8.1f}x | "
            f"{r['ratio_eppstein_local']:>8.1f}x | "
            f"{r['ratio_riondato']:>8.1f}x | "
            f"{r['ratio_bader']:>8.1f}x"
        )

    return bounds_df


# =============================================================================
# EW BOUND ANALYSIS
# =============================================================================


def compute_ew_analysis(raw_df: pd.DataFrame, k: float, min_eff_n: int):
    """Evaluate the localised EW bound on all GLA validation configurations."""
    print("\n" + "=" * 70)
    print("LOCALISED EW BOUND ANALYSIS")
    print("=" * 70)

    rows = []
    for _, row in raw_df.iterrows():
        reach = row["mean_reach"]
        sample_prob = row["sample_prob"]
        metric = row["metric"]
        max_abs_error = row["max_abs_error"]

        if sample_prob >= 1.0:
            continue

        n_eff = reach * sample_prob
        eps_obs = normalise_error(max_abs_error, reach, metric)
        eps_pred = ew_predicted_epsilon(n_eff, reach)
        bound_holds = eps_obs <= eps_pred

        rows.append(
            {
                "distance": row["distance"],
                "metric": metric,
                "reach": reach,
                "sample_prob": sample_prob,
                "n_eff": n_eff,
                "eps_observed": eps_obs,
                "eps_predicted": eps_pred,
                "bound_holds": bound_holds,
                "spearman": row["spearman"],
            }
        )

    if not rows:
        print("  No configurations to analyse")
        return None

    ew_df = pd.DataFrame(rows)

    # Overall success rate
    total = len(ew_df)
    holds = int(ew_df["bound_holds"].sum())
    print(f"\n  Overall: {holds}/{total} ({100 * holds / total:.1f}%) — expected >= {100 * (1 - DELTA):.0f}%")

    # By metric
    print(f"\n  {'Metric':<15} {'Holds':>8} {'Total':>8} {'Rate':>8}")
    print("  " + "-" * 45)
    for metric in sorted(ew_df["metric"].unique()):
        subset = ew_df[ew_df["metric"] == metric]
        h = int(subset["bound_holds"].sum())
        t = len(subset)
        print(f"  {metric:<15} {h:>8} {t:>8} {100 * h / t:>7.1f}%")

    # By distance
    print(f"\n  {'Distance':<15} {'Holds':>8} {'Total':>8} {'Rate':>8}")
    print("  " + "-" * 45)
    for dist in sorted(ew_df["distance"].unique()):
        subset = ew_df[ew_df["distance"] == dist]
        h = int(subset["bound_holds"].sum())
        t = len(subset)
        print(f"  {dist:<15} {h:>8} {t:>8} {100 * h / t:>7.1f}%")

    # Implied epsilon from rank model
    print("\n  Implied epsilon from rank model:")
    print(f"  {'Distance':>10} {'Reach':>10} {'Rank n_eff':>12} {'Implied eps':>12}")
    print("  " + "-" * 50)
    for dist in sorted(ew_df["distance"].unique()):
        reach = ew_df[ew_df["distance"] == dist]["reach"].iloc[0]
        impl_eps = implied_epsilon_from_rank_model(reach, k, min_eff_n)
        rank_eff_n = max(k * math.sqrt(reach), min_eff_n)
        print(f"  {dist // 1000}km       {reach:>10,.0f} {rank_eff_n:>12,.0f} {impl_eps:>12.4f}")

    # Conservatism: median predicted/observed ratio
    for metric in sorted(ew_df["metric"].unique()):
        subset = ew_df[(ew_df["metric"] == metric) & (ew_df["eps_observed"] > 0)]
        if len(subset) > 0:
            ratio = subset["eps_predicted"] / subset["eps_observed"]
            print(f"\n  Conservatism ({metric}): median predicted/observed = {ratio.median():.1f}x")

    # Save
    csv_path = OUTPUT_DIR / "gla_ew_analysis.csv"
    ew_df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    return ew_df


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Validate sampling model on GLA network")
    parser.add_argument("--force", action="store_true", help="Force regeneration of validation data")
    args = parser.parse_args()

    print("=" * 70)
    print("03_validate_gla.py - Validating model on Greater London network")
    print("=" * 70)

    # Load model
    k, min_eff_n = load_model()
    print(f"\nModel: eff_n = max({k} × sqrt(reach), {min_eff_n})")

    # Generate or load validation data
    raw_df = generate_validation_data(force=args.force)

    # Pivot to combined format
    df = pivot_validation_data(raw_df)
    print(f"\nValidation data: {len(df)} rows")
    print(f"Distances: {sorted(df['distance'].unique())}")

    # Generate outputs
    generate_fig5_validation(df, k, min_eff_n)
    results_df = generate_validation_table(df, k, min_eff_n)

    # Theoretical bounds comparison
    n_nodes = get_n_nodes(force=args.force)
    bounds_df = None
    if n_nodes is not None:
        print(f"\nGLA network: {n_nodes} live nodes")
        bounds_df = compute_theoretical_bounds(results_df, n_nodes)
    else:
        print("\n  Skipping theoretical bounds comparison (graph cache not found)")

    # EW bound analysis
    ew_df = compute_ew_analysis(raw_df, k, min_eff_n)

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    print(f"\n{'Distance':>10} | {'Reach':>10} | {'Model p':>10} | {'rho_close':>10} | {'rho_betw':>10} | {'Target?':>10}")
    print("-" * 75)

    all_pass = True
    for _, row in results_df.iterrows():
        status = "PASS" if row["meets_target"] else "FAIL"
        if not row["meets_target"]:
            all_pass = False
        print(
            f"{row['distance'] // 1000}km       | {row['reach']:>10,.0f} | {row['model_p']:>9.1%} | "
            f"{row['observed_rho_h']:>10.4f} | {row['observed_rho_b']:>10.4f} | {status:>10}"
        )

    print("\n" + "-" * 75)
    if all_pass:
        print("ALL DISTANCES PASS: Model achieves rho >= 0.95 for both metrics at all distances.")
    else:
        print("WARNING: Some distances do not meet the rho >= 0.95 target.")

    print("\n" + "=" * 70)
    print("OUTPUTS")
    print("=" * 70)
    print(f"  1. {OUTPUT_DIR / f'gla_validation_{CACHE_VERSION}.csv'}")
    print(f"  2. {OUTPUT_DIR / 'gla_validation_summary.csv'}")
    print(f"  3. {FIGURES_DIR / 'fig6_gla_validation.pdf'}")
    print(f"  4. {TABLES_DIR / 'tab2_validation.tex'}")
    if bounds_df is not None:
        print(f"  5. {OUTPUT_DIR / 'gla_theoretical_bounds_comparison.csv'}")
    if ew_df is not None:
        print(f"  6. {OUTPUT_DIR / 'gla_ew_analysis.csv'}")

    return 0


if __name__ == "__main__":
    exit(main())
