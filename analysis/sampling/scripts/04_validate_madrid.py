#!/usr/bin/env python
"""
04_validate_madrid.py - External validation on Madrid regional network.

Validates the Hoeffding sampling model on an independent network.

Usage:
    python 04_validate_madrid.py           # Run (skips cache if exists)
    python 04_validate_madrid.py --force   # Force regeneration of validation data

Outputs:
    - output/madrid_validation.csv
    - paper/tables/tab4_madrid_validation.tex
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
import shapely
from cityseer.tools import graphs, io
from shapely.geometry import Point
from utilities import (
    CACHE_DIR,
    HOEFFDING_DELTA,
    HOEFFDING_EPSILON,
    OUTPUT_DIR,
    TABLES_DIR,
    compute_accuracy_metrics,
    compute_hoeffding_p,
    compute_quartile_accuracy,
    ew_predicted_epsilon,
    mean_quartiles,
    normalise_error,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent

# Madrid network file
MADRID_GPKG_FILE = SCRIPT_DIR.parent.parent.parent / "temp" / "RT_MADRID_gpkg" / "red_viaria.gpkg"

# Validation parameters
MADRID_DISTANCES = [1000, 2000, 5000, 10000, 20000]
N_RUNS = 5


DELTA = HOEFFDING_DELTA


def get_madrid_mask(force: bool = False):
    """
    Return Community of Madrid boundary and 20km buffered version in EPSG:25830, cached as GeoJSON.

    Returns
    -------
    boundary : shapely geometry
        Simplified Community of Madrid boundary — used to mark live nodes.
    buffered : shapely geometry
        Community of Madrid boundary buffered by 20km — used as spatial filter when loading roads.
    """
    boundary_cache = CACHE_DIR / "madrid_boundary.geojson"
    buffered_cache = CACHE_DIR / "madrid_buffered.geojson"
    if boundary_cache.exists() and buffered_cache.exists() and not force:
        boundary = gpd.read_file(boundary_cache).geometry.iloc[0]
        buffered = gpd.read_file(buffered_cache).geometry.iloc[0]
        return boundary, buffered
    print("Downloading Community of Madrid boundary from OSM...")
    gdf = ox.geocode_to_gdf("Community of Madrid, Spain")
    gdf = gdf.to_crs("EPSG:25830")
    boundary = gdf.geometry.iloc[0].simplify(100)  # 100m tolerance
    buffered = boundary.buffer(20_000)
    gpd.GeoDataFrame(geometry=[boundary], crs="EPSG:25830").to_file(boundary_cache, driver="GeoJSON")
    gpd.GeoDataFrame(geometry=[buffered], crs="EPSG:25830").to_file(buffered_cache, driver="GeoJSON")
    print(f"  Cached Madrid boundary to {boundary_cache}")
    print(f"  Cached Madrid buffered to {buffered_cache}")
    return boundary, buffered


def generate_validation_data(force: bool = False) -> pd.DataFrame:
    """Generate Madrid validation data, or load from cache."""
    validation_csv = OUTPUT_DIR / "madrid_validation.csv"

    if validation_csv.exists() and not force:
        df = pd.read_csv(validation_csv)
        if "rho_closeness" in df.columns and len(df) > 0:
            print(f"Loading cached validation data from {validation_csv}")
            return df
        print(f"Stale/empty cache at {validation_csv}, regenerating...")

    print("\n" + "=" * 70)
    print("GENERATING MADRID VALIDATION DATA")
    print("=" * 70)

    # Load Madrid boundary (cached after first download)
    madrid_boundary, madrid_buffered = get_madrid_mask(force=force)

    # Load or build Madrid graph
    madrid_cache = CACHE_DIR / "madrid_graph.pkl"
    if madrid_cache.exists() and not force:
        print(f"Loading cached Madrid graph from {madrid_cache}")
        with open(madrid_cache, "rb") as f:
            G = pickle.load(f)
    else:
        print(f"Loading Madrid network from {MADRID_GPKG_FILE}")
        # Reproject buffered mask to file CRS (EPSG:4258) for spatial filtering
        buffered_4258 = gpd.GeoDataFrame(geometry=[madrid_buffered], crs="EPSG:25830").to_crs("EPSG:4258").geometry.iloc[0]
        edges_gdf = gpd.read_file(MADRID_GPKG_FILE, layer="rt_tramo_vial", mask=buffered_4258)
        edges_gdf = edges_gdf[edges_gdf.geometry.is_valid & ~edges_gdf.geometry.is_empty]
        edges_gdf = edges_gdf.to_crs("EPSG:25830").explode(index_parts=False)
        edges_gdf.geometry = edges_gdf.geometry.map(shapely.force_2d)
        print(f"  Loaded: {len(edges_gdf)} edges")

        print("  Building graph...")
        G = io.nx_from_generic_geopandas(edges_gdf)
        G = graphs.nx_remove_filler_nodes(G)
        G = graphs.nx_remove_dangling_nodes(G)

        print(f"  Caching to {madrid_cache}")
        with open(madrid_cache, "wb") as f:
            pickle.dump(G, f)

    print(f"Madrid graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Mark live nodes: inside Madrid boundary = live, buffer zone = not live
    print("Marking live nodes using Madrid boundary...")
    n_live = 0
    for _n, data in G.nodes(data=True):
        data["live"] = madrid_boundary.contains(Point(data["x"], data["y"]))
        n_live += data["live"]
    print(f"  Live nodes: {n_live}/{G.number_of_nodes()} ({100 * n_live / G.number_of_nodes():.1f}%)")

    # Convert to cityseer
    print("Converting to cityseer format...")
    nodes_gdf, _, net = io.network_structure_from_nx(G)
    live_mask = nodes_gdf["live"].values
    n_live = int(live_mask.sum())
    print(f"Live nodes: {n_live}/{len(live_mask)}")

    results = []

    for dist in MADRID_DISTANCES:
        print(f"\n{'=' * 50}")
        print(f"Distance: {dist}m")
        print(f"{'=' * 50}")

        # Check for cached ground truth
        gt_cache = CACHE_DIR / f"madrid_ground_truth_{dist}m.pkl"
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

            # Extract only live nodes for ground truth
            true_harmonic = np.array(true_result.node_harmonic[dist])[live_mask]
            true_betweenness = np.array(true_result.node_betweenness[dist])[live_mask]
            node_reach = np.array(true_result.node_density[dist])[live_mask]
            mean_reach = float(np.mean(node_reach))

            with open(gt_cache, "wb") as f:
                pickle.dump(
                    {
                        "harmonic": true_harmonic,
                        "betweenness": true_betweenness,
                        "node_reach": node_reach,
                        "mean_reach": mean_reach,
                        "n_live": n_live,
                        "baseline_time": baseline_time,
                    },
                    f,
                )
            print(f"  Cached ground truth to {gt_cache}")

        print(f"  Mean reach: {mean_reach:.0f}")
        if baseline_time is not None:
            print(f"  Baseline time: {baseline_time:.1f}s")

        # Compute Hoeffding model probability
        hoeffding_p = compute_hoeffding_p(mean_reach, epsilon=HOEFFDING_EPSILON)

        probs_to_test = [("hoeffding_0.1", hoeffding_p)]

        for label, test_p in probs_to_test:
            effective_n = mean_reach * test_p
            print(f"  {label}: p={test_p:.4f} (eff_n={effective_n:.0f})")

            # Run sampled computations
            spearmans_h, spearmans_b = [], []
            maes_h, maes_b = [], []
            precs_h, precs_b = [], []
            scales_h, scales_b = [], []
            quartiles_h, quartiles_b = [], []
            sampled_times = []

            print(f"    Running {N_RUNS} sampled runs: ", end="", flush=True)

            for seed in range(N_RUNS):
                t0 = time.time()
                r = net.local_node_centrality_shortest(
                    distances=[dist],
                    compute_closeness=True,
                    compute_betweenness=True,
                    sample_probability=test_p,
                    random_seed=seed,
                    pbar_disabled=True,
                )
                sampled_times.append(time.time() - t0)

                est_harmonic = np.array(r.node_harmonic[dist])[live_mask]
                est_betweenness = np.array(r.node_betweenness[dist])[live_mask]

                sp_h, prec_h, scale_h, _, mae_h = compute_accuracy_metrics(true_harmonic, est_harmonic)
                sp_b, prec_b, scale_b, _, mae_b = compute_accuracy_metrics(true_betweenness, est_betweenness)

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

            mean_sampled_time = np.mean(sampled_times)
            speedup = baseline_time / mean_sampled_time if baseline_time and mean_sampled_time > 0 else float("nan")

            # Average quartile results across runs
            q_h = mean_quartiles(quartiles_h)
            q_b = mean_quartiles(quartiles_b)

            print(f" rho_h={np.mean(spearmans_h):.3f}, rho_b={np.mean(spearmans_b):.3f}, speedup={speedup:.1f}x")

            row = {
                "distance": dist,
                "mean_reach": mean_reach,
                "model": label,
                "sample_prob": test_p,
                "effective_n": effective_n,
                "rho_closeness": np.mean(spearmans_h),
                "rho_closeness_min": np.min(spearmans_h) if spearmans_h else float("nan"),
                "rho_closeness_median": np.median(spearmans_h) if spearmans_h else float("nan"),
                "rho_closeness_max": np.max(spearmans_h) if spearmans_h else float("nan"),
                "rho_closeness_std": np.std(spearmans_h),
                "rho_betweenness": np.mean(spearmans_b),
                "rho_betweenness_min": np.min(spearmans_b) if spearmans_b else float("nan"),
                "rho_betweenness_median": np.median(spearmans_b) if spearmans_b else float("nan"),
                "rho_betweenness_max": np.max(spearmans_b) if spearmans_b else float("nan"),
                "rho_betweenness_std": np.std(spearmans_b),
                "max_abs_error_h": np.max(maes_h) if maes_h else float("nan"),
                "mean_abs_error_h": np.mean(maes_h) if maes_h else float("nan"),
                "median_abs_error_h": np.median(maes_h) if maes_h else float("nan"),
                "min_abs_error_h": np.min(maes_h) if maes_h else float("nan"),
                "max_abs_error_b": np.max(maes_b) if maes_b else float("nan"),
                "mean_abs_error_b": np.mean(maes_b) if maes_b else float("nan"),
                "median_abs_error_b": np.median(maes_b) if maes_b else float("nan"),
                "min_abs_error_b": np.min(maes_b) if maes_b else float("nan"),
                "top_k_precision_h": np.mean(precs_h) if precs_h else float("nan"),
                "top_k_precision_b": np.mean(precs_b) if precs_b else float("nan"),
                "scale_ratio_h": np.mean(scales_h) if scales_h else float("nan"),
                "scale_ratio_b": np.mean(scales_b) if scales_b else float("nan"),
                "baseline_time": baseline_time if baseline_time else float("nan"),
                "sampled_time": mean_sampled_time,
                "speedup": speedup,
            }
            # Add per-quartile columns (prefixed by metric)
            for k_q, v_q in q_h.items():
                row[f"h_{k_q}"] = v_q
            for k_q, v_q in q_b.items():
                row[f"b_{k_q}"] = v_q
            results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(validation_csv, index=False)
    print(f"\nSaved validation results: {validation_csv}")
    return df


# =============================================================================
# THEORETICAL BOUNDS COMPARISON
# =============================================================================


def get_n_nodes(force: bool = False) -> int | None:
    """Get live node count from cached Madrid graph."""
    n_nodes_cache = CACHE_DIR / "madrid_n_nodes.json"
    if n_nodes_cache.exists() and not force:
        with open(n_nodes_cache) as f:
            return json.load(f)["n_nodes"]
    madrid_cache = CACHE_DIR / "madrid_graph.pkl"
    if not madrid_cache.exists():
        return None
    with open(madrid_cache, "rb") as f:
        G = pickle.load(f)
    madrid_boundary, _ = get_madrid_mask(force=force)
    n_nodes = sum(1 for _n, d in G.nodes(data=True) if madrid_boundary.contains(Point(d["x"], d["y"])))
    with open(n_nodes_cache, "w") as f:
        json.dump({"n_nodes": n_nodes}, f)
    return n_nodes


def compute_theoretical_bounds(df: pd.DataFrame, n_nodes: int):
    """Compare empirical sample counts against Riondato, Bader, and Eppstein-Wang bounds."""
    print("\nComputing theoretical bounds comparison...")

    delta = 0.1  # Failure probability (90% confidence)
    rows = []

    for _, row in df.iterrows():
        reach = row["mean_reach"]
        our_eff_n = row["effective_n"]

        for metric, mae_col in [("harmonic", "max_abs_error_h"), ("betweenness", "max_abs_error_b")]:
            if mae_col not in row or np.isnan(row[mae_col]):
                continue

            raw_eps = row[mae_col]
            if metric == "betweenness":
                eps_normalised = raw_eps / (reach * (reach - 1)) if reach > 1 else float("inf")
            else:
                eps_normalised = raw_eps / reach if reach > 0 else float("inf")

            if eps_normalised <= 0 or not np.isfinite(eps_normalised):
                continue

            # Riondato & Kornaropoulos (2016): VC-dimension bound
            vd = max(3, int(np.sqrt(reach)))
            vc_dim = int(np.floor(np.log2(max(1, vd - 2)))) + 1
            riondato_samples = (1 / (2 * eps_normalised**2)) * (vc_dim + np.log(1 / delta))

            # Bader et al. (2007): per-vertex bound
            bader_samples = 1 / eps_normalised**2

            # Eppstein & Wang (2004): source-sampling bound
            # Global: O(log(n) / eps^2) using total node count
            eppstein_samples = np.log(n_nodes) / eps_normalised**2
            # Localised adaptation: O(log(r) / eps^2) using reach
            eppstein_local_samples = np.log(reach) / eps_normalised**2

            rows.append(
                {
                    "distance": row["distance"],
                    "metric": metric,
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

    csv_path = OUTPUT_DIR / "madrid_theoretical_bounds_comparison.csv"
    bounds_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    print(
        f"\n  {'Distance':>10} | {'Metric':>12} | {'Our eff_n':>10} | {'EW global':>12} | {'EW local':>12} | "
        f"{'Riondato':>12} | {'Bader':>12} | {'R(EWg)':>8} | {'R(EWl)':>8}"
    )
    print("  " + "-" * 120)
    for _, r in bounds_df.iterrows():
        print(
            f"  {r['distance'] // 1000}km       | "
            f"{r['metric']:>12} | "
            f"{r['our_eff_n']:>10,.0f} | "
            f"{r['eppstein_samples']:>12,.0f} | "
            f"{r['eppstein_local_samples']:>12,.0f} | "
            f"{r['riondato_samples']:>12,.0f} | "
            f"{r['bader_samples']:>12,.0f} | "
            f"{r['ratio_eppstein']:>8.1f}x | "
            f"{r['ratio_eppstein_local']:>8.1f}x"
        )

    return bounds_df


# =============================================================================
# EW BOUND ANALYSIS
# =============================================================================


def compute_ew_analysis(df: pd.DataFrame):
    """Evaluate the localised EW bound on Madrid validation configurations."""
    print("\n" + "=" * 70)
    print("LOCALISED EW BOUND ANALYSIS")
    print("=" * 70)

    rows = []
    for _, row in df.iterrows():
        reach = row["mean_reach"]
        sample_prob = row["sample_prob"]
        n_eff = reach * sample_prob

        for metric, mae_col in [("harmonic", "max_abs_error_h"), ("betweenness", "max_abs_error_b")]:
            if mae_col not in row or np.isnan(row[mae_col]):
                continue

            eps_obs = normalise_error(row[mae_col], reach, metric)
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

    # Save
    csv_path = OUTPUT_DIR / "madrid_ew_analysis.csv"
    ew_df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    return ew_df


def generate_validation_table(df: pd.DataFrame, n_nodes: int | None):
    """Generate LaTeX table of Madrid validation results (Hoeffding rows only)."""
    print("\nGenerating Table: Madrid validation results...")

    # Filter to Hoeffding rows only
    df_hoeff = df[df["model"] == "hoeffding_0.1"] if "model" in df.columns else df

    latex = r"""\begin{table}[htbp]
\centering
\caption{Hoeffding model validation on Greater Madrid network ($\varepsilon = 0.1$, $\delta = 0.1$).}
\label{tab:madrid_validation}
\begin{tabular}{rrrrrrr}
\toprule
\textbf{Distance} & \textbf{Reach} & \textbf{Hoeff.\ $p$} &
\textbf{$\rho_\text{close}$} & \textbf{$\rho_\text{betw}$} & \textbf{Speedup} & \textbf{$\rho \geq 0.95$?} \\
\midrule
"""

    for _, row in df_hoeff.iterrows():
        reach = row["mean_reach"]
        hoeff_p = compute_hoeffding_p(reach)
        speedup = 1 / hoeff_p if hoeff_p > 0 else float("inf")
        meets = row["rho_closeness"] >= 0.95 and row["rho_betweenness"] >= 0.95
        check = r"\checkmark" if meets else r"\texttimes"
        hoeff_p_pct = f"{hoeff_p * 100:.1f}\\%"
        latex += f"{row['distance'] // 1000}km & {reach:,.0f} & {hoeff_p_pct} & "
        latex += f"{row['rho_closeness']:.4f} & {row['rho_betweenness']:.4f} & "
        latex += f"{speedup:.1f}$\\times$ & {check} \\\\\n"

    n_nodes_str = f"{n_nodes:,}" if n_nodes else "\\texttildelow 99,000"
    latex += rf"""\bottomrule
\end{{tabular}}

\vspace{{0.5em}}
\footnotesize
Network: Greater Madrid, {n_nodes_str} nodes, Community of Madrid boundary live node mask.
Hoeffding model: $k = \log(2r/\delta) / (2\varepsilon^2)$, $p = \min(1, k/r)$.
\end{{table}}
"""

    output_path = TABLES_DIR / "tab4_madrid_validation.tex"
    with open(output_path, "w") as f:
        f.write(latex)
    print(f"  Saved: {output_path}")


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Validate sampling model on Madrid network")
    parser.add_argument("--force", action="store_true", help="Force regeneration of validation data")
    args = parser.parse_args()

    print("=" * 70)
    print("04_validate_madrid.py - External validation on Madrid network")
    print("=" * 70)

    print("\nModel: k = log(2r/δ) / (2ε²), p = min(1, k/r)")
    print(f"  ε = {HOEFFDING_EPSILON}, δ = {HOEFFDING_DELTA}")

    # Generate or load validation data
    df = generate_validation_data(force=args.force)
    print(f"\nValidation data: {len(df)} rows")

    # Add target columns
    df["meets_target_close"] = df["rho_closeness"] >= 0.95
    df["meets_target_between"] = df["rho_betweenness"] >= 0.95

    # Theoretical bounds comparison
    df_hoeff = df[df["model"] == "hoeffding_0.1"] if "model" in df.columns else df
    n_nodes = get_n_nodes(force=args.force)
    bounds_df = None
    if n_nodes is not None:
        print(f"\nMadrid network: {n_nodes} live nodes")
        bounds_df = compute_theoretical_bounds(df_hoeff, n_nodes)
    else:
        print("\n  Skipping theoretical bounds comparison (graph cache not found)")

    # Generate validation table
    generate_validation_table(df, n_nodes)

    # EW bound analysis
    ew_df = compute_ew_analysis(df)

    # Print summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    print(
        f"\n{'Distance':>10} | {'Model':>15} | {'Reach':>10} | {'p':>8}"
        f" | {'ρ close':>10} | {'ρ between':>10} | {'Speedup':>10} | {'Pass?':>6}"
    )
    print("-" * 100)

    all_pass = True
    for _, row in df.iterrows():
        passes = row["meets_target_close"] and row["meets_target_between"]
        status = "PASS" if passes else "FAIL"
        if not passes:
            all_pass = False

        dist_str = f"{row['distance'] // 1000}km"
        model_str = row.get("model", "hoeffding_0.1")
        print(
            f"{dist_str:>10} | "
            f"{model_str:>15} | "
            f"{row['mean_reach']:>10,.0f} | "
            f"{row['sample_prob']:>7.1%} | "
            f"{row['rho_closeness']:>10.4f} | "
            f"{row['rho_betweenness']:>10.4f} | "
            f"{row['speedup']:>9.1f}x | "
            f"{status:>6}"
        )

    print("-" * 100)

    if all_pass:
        print("\nALL CONFIGURATIONS PASS: Hoeffding model achieves ρ >= 0.95 for both metrics at all distances.")
    else:
        print("\nWARNING: Some configurations do not meet the ρ >= 0.95 target.")

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
    print(f"  2. {TABLES_DIR / 'tab4_madrid_validation.tex'}")
    if bounds_df is not None:
        print(f"  3. {OUTPUT_DIR / 'madrid_theoretical_bounds_comparison.csv'}")
    if ew_df is not None:
        print(f"  4. {OUTPUT_DIR / 'madrid_ew_analysis.csv'}")

    return 0


if __name__ == "__main__":
    exit(main())
