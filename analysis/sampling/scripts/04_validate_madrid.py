#!/usr/bin/env python
"""
04_validate_madrid.py - External validation on Madrid regional network.

Validates both closeness and betweenness sampling models on an independent network.
Both metrics use the unified framework: Hoeffding bound + spatial source_indices + IPW.

Usage:
    python 04_validate_madrid.py           # Run (skips cache if exists)
    python 04_validate_madrid.py --force   # Force regeneration of validation data

Outputs:
    - output/madrid_validation.csv
    - output/madrid_theoretical_bounds_comparison.csv
    - output/madrid_bound_analysis.csv
    - paper/tables/tab4_madrid_validation.tex
"""

import argparse
import json
import pickle
import random
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
    OUTPUT_DIR,
    TABLES_DIR,
    compute_accuracy_metrics,
    compute_hoeffding_p,
    compute_quartile_accuracy,
    ew_predicted_epsilon,
    mean_quartiles,
    select_spatial_sources,
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

# Hoeffding + spatial source sampling (both metrics)
MADRID_EPSILON_CLOSENESS = 0.1
MADRID_EPSILON_BETWEENNESS = 0.1

DELTA = HOEFFDING_DELTA


def get_madrid_mask(force: bool = False):
    """Return Community of Madrid boundary and 20km buffered version in EPSG:25830."""
    boundary_cache = CACHE_DIR / "madrid_boundary.geojson"
    buffered_cache = CACHE_DIR / "madrid_buffered.geojson"
    if boundary_cache.exists() and buffered_cache.exists() and not force:
        boundary = gpd.read_file(boundary_cache).geometry.iloc[0]
        buffered = gpd.read_file(buffered_cache).geometry.iloc[0]
        return boundary, buffered
    print("Downloading Community of Madrid boundary from OSM...")
    gdf = ox.geocode_to_gdf("Community of Madrid, Spain")
    gdf = gdf.to_crs("EPSG:25830")
    boundary = gdf.geometry.iloc[0].simplify(100)
    buffered = boundary.buffer(20_000)
    gpd.GeoDataFrame(geometry=[boundary], crs="EPSG:25830").to_file(boundary_cache, driver="GeoJSON")
    gpd.GeoDataFrame(geometry=[buffered], crs="EPSG:25830").to_file(buffered_cache, driver="GeoJSON")
    print(f"  Cached Madrid boundary to {boundary_cache}")
    return boundary, buffered


def generate_validation_data(force: bool = False) -> pd.DataFrame:
    """Generate Madrid validation data, or load from cache."""
    validation_csv = OUTPUT_DIR / "madrid_validation.csv"

    if validation_csv.exists() and not force:
        df = pd.read_csv(validation_csv)
        if "model" in df.columns:
            print(f"Stale cache detected at {validation_csv}, regenerating...")
        elif "rho_closeness" in df.columns and "rho_betweenness" in df.columns and len(df) > 0:
            print(f"Loading cached validation data from {validation_csv}")
            return df
        else:
            print(f"Stale/empty cache at {validation_csv}, regenerating...")

    print("\n" + "=" * 70)
    print("GENERATING MADRID VALIDATION DATA")
    print("=" * 70)

    madrid_boundary, madrid_buffered = get_madrid_mask(force=force)

    # Load or build Madrid graph
    madrid_cache = CACHE_DIR / "madrid_graph.pkl"
    if madrid_cache.exists() and not force:
        print(f"Loading cached Madrid graph from {madrid_cache}")
        with open(madrid_cache, "rb") as f:
            G = pickle.load(f)
    else:
        print(f"Loading Madrid network from {MADRID_GPKG_FILE}")
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

    # Mark live nodes
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

    results = []

    for dist in MADRID_DISTANCES:
        print(f"\n{'=' * 50}")
        print(f"Distance: {dist}m")
        print(f"{'=' * 50}")

        # Ground truth (cached per distance)
        gt_cache = CACHE_DIR / f"madrid_ground_truth_{dist}m.pkl"
        if gt_cache.exists() and not force:
            print(f"  Loading cached ground truth from {gt_cache}")
            with open(gt_cache, "rb") as f:
                gt_data = pickle.load(f)
            true_harmonic = gt_data["harmonic"]
            true_betweenness = gt_data["betweenness"]
            mean_reach = gt_data["mean_reach"]
            node_reach = gt_data.get("node_reach", None)
            baseline_time_closeness = gt_data.get("baseline_time_closeness", gt_data.get("baseline_time", None))
            baseline_time_betweenness = gt_data.get("baseline_time_betweenness", None)
        else:
            print("  Computing ground truth closeness...")
            t0 = time.time()
            close_result = net.closeness_shortest(distances=[dist], pbar_disabled=False)
            baseline_time_closeness = time.time() - t0
            true_harmonic = np.array(close_result.node_harmonic[dist])[live_mask]
            node_reach = np.array(close_result.node_density[dist])[live_mask]
            mean_reach = float(np.mean(node_reach))
            print(f"  Closeness: {baseline_time_closeness:.1f}s")

            print("  Computing ground truth betweenness...")
            t0 = time.time()
            betw_result = net.betweenness_shortest(distances=[dist], pbar_disabled=False)
            baseline_time_betweenness = time.time() - t0
            true_betweenness = np.array(betw_result.node_betweenness[dist])[live_mask]
            print(f"  Betweenness: {baseline_time_betweenness:.1f}s")

            with open(gt_cache, "wb") as f:
                pickle.dump(
                    {
                        "harmonic": true_harmonic,
                        "betweenness": true_betweenness,
                        "node_reach": node_reach,
                        "mean_reach": mean_reach,
                        "n_live": n_live,
                        "baseline_time_closeness": baseline_time_closeness,
                        "baseline_time_betweenness": baseline_time_betweenness,
                    },
                    f,
                )
            print(f"  Cached ground truth to {gt_cache}")

        print(f"  Mean reach: {mean_reach:.0f}")
        cell_size = dist / 2.0

        # ---------------------------------------------------------------
        # Closeness: Hoeffding + spatial source sampling
        # ---------------------------------------------------------------
        p_close = compute_hoeffding_p(mean_reach, epsilon=MADRID_EPSILON_CLOSENESS, delta=DELTA)
        print(f"  Closeness eps={MADRID_EPSILON_CLOSENESS}: p={p_close:.4f}")

        spearmans_h, maes_h, precs_h, scales_h, quartiles_h = [], [], [], [], []
        close_times = []

        print(f"    Running {N_RUNS} runs: ", end="", flush=True)
        for seed in range(N_RUNS):
            rng = random.Random(42 + seed)
            n_sources = max(1, int(p_close * n_live))
            sources = select_spatial_sources(net, n_sources, cell_size, rng)

            t0 = time.time()
            r_close = net.closeness_shortest(
                distances=[dist],
                source_indices=sources,
                sample_probability=p_close,
                pbar_disabled=True,
            )
            close_times.append(time.time() - t0)
            est_harmonic = np.array(r_close.node_harmonic[dist])[live_mask]

            sp_h, prec_h, scale_h, _, mae_h = compute_accuracy_metrics(true_harmonic, est_harmonic)
            if not np.isnan(sp_h):
                spearmans_h.append(sp_h)
                maes_h.append(mae_h)
                precs_h.append(prec_h)
                scales_h.append(scale_h)
            if node_reach is not None:
                quartiles_h.append(compute_quartile_accuracy(true_harmonic, est_harmonic, node_reach))
            print(".", end="", flush=True)

        mean_close_time = np.mean(close_times) if close_times else float("nan")
        q_h = mean_quartiles(quartiles_h)
        n_eff_h = mean_reach * p_close
        eps_pred_h = ew_predicted_epsilon(n_eff_h, mean_reach, delta=DELTA)
        max_mae_h = np.max(maes_h) if maes_h else float("nan")
        eps_obs_h = max_mae_h / mean_reach if not np.isnan(max_mae_h) else float("nan")
        rho_h = np.mean(spearmans_h) if spearmans_h else float("nan")
        speedup_h = baseline_time_closeness / mean_close_time if baseline_time_closeness and mean_close_time > 0 else float("nan")
        print(f" rho={rho_h:.3f}, speedup={speedup_h:.1f}x")

        # ---------------------------------------------------------------
        # Betweenness: Hoeffding + spatial source sampling
        # ---------------------------------------------------------------
        nonzero_betw = np.sum(true_betweenness > 0)
        rho_b = float("nan")
        speedup_b = float("nan")
        p_betw = float("nan")
        max_mae_b = float("nan")
        eps_obs_b = float("nan")
        eps_pred_b = float("nan")
        q_b = {}

        if nonzero_betw < 10:
            print(f"  Betweenness: skipped (only {nonzero_betw} nonzero)")
        else:
            p_betw = compute_hoeffding_p(mean_reach, epsilon=MADRID_EPSILON_BETWEENNESS, delta=DELTA)
            print(f"  Betweenness eps={MADRID_EPSILON_BETWEENNESS}: p={p_betw:.4f}")

            spearmans_b, maes_b, precs_b, scales_b, quartiles_b = [], [], [], [], []
            betw_times = []

            print(f"    Running {N_RUNS} runs: ", end="", flush=True)
            for seed in range(N_RUNS):
                rng = random.Random(42 + seed)
                n_sources = max(1, int(p_betw * n_live))
                sources = select_spatial_sources(net, n_sources, cell_size, rng)

                t0 = time.time()
                r_betw = net.betweenness_shortest(
                    distances=[dist],
                    source_indices=sources,
                    sample_probability=p_betw,
                    pbar_disabled=True,
                )
                betw_times.append(time.time() - t0)
                est_betweenness = np.array(r_betw.node_betweenness[dist])[live_mask]

                sp_b, prec_b, scale_b, _, mae_b = compute_accuracy_metrics(true_betweenness, est_betweenness)
                if not np.isnan(sp_b):
                    spearmans_b.append(sp_b)
                    maes_b.append(mae_b)
                    precs_b.append(prec_b)
                    scales_b.append(scale_b)
                if node_reach is not None:
                    quartiles_b.append(compute_quartile_accuracy(true_betweenness, est_betweenness, node_reach))
                print(".", end="", flush=True)

            mean_betw_time = np.mean(betw_times) if betw_times else float("nan")
            q_b = mean_quartiles(quartiles_b)
            n_eff_b = mean_reach * p_betw
            eps_pred_b = ew_predicted_epsilon(n_eff_b, mean_reach, delta=DELTA)
            max_mae_b = np.max(maes_b) if maes_b else float("nan")
            eps_obs_b = max_mae_b / mean_reach if not np.isnan(max_mae_b) else float("nan")
            rho_b = np.mean(spearmans_b) if spearmans_b else float("nan")
            speedup_b = baseline_time_betweenness / mean_betw_time if baseline_time_betweenness and mean_betw_time > 0 else float("nan")
            print(f" rho={rho_b:.3f}, speedup={speedup_b:.1f}x")

        # Build result row
        row = {
            "distance": dist,
            "mean_reach": mean_reach,
            # Closeness
            "epsilon_closeness": MADRID_EPSILON_CLOSENESS,
            "hoeffding_p_close": p_close,
            "rho_closeness": rho_h,
            "max_abs_error_h": max_mae_h,
            "eps_obs_h": eps_obs_h,
            "eps_pred_h": eps_pred_h,
            "top_k_precision_h": np.mean(precs_h) if precs_h else float("nan"),
            "baseline_time_h": baseline_time_closeness if baseline_time_closeness else float("nan"),
            "sampled_time_h": mean_close_time,
            "speedup_closeness": speedup_h,
            # Betweenness
            "epsilon_betweenness": MADRID_EPSILON_BETWEENNESS,
            "hoeffding_p_betw": p_betw,
            "rho_betweenness": rho_b,
            "max_abs_error_b": max_mae_b,
            "eps_obs_b": eps_obs_b,
            "eps_pred_b": eps_pred_b,
            "baseline_time_b": baseline_time_betweenness if baseline_time_betweenness else float("nan"),
            "speedup_betweenness": speedup_b,
        }
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
    """Compare empirical sample counts against Eppstein-Wang bounds."""
    print("\nComputing theoretical bounds comparison...")

    rows = []
    for _, row in df.iterrows():
        reach = row["mean_reach"]

        for metric, p_col, mae_col in [
            ("harmonic", "hoeffding_p_close", "max_abs_error_h"),
            ("betweenness", "hoeffding_p_betw", "max_abs_error_b"),
        ]:
            if mae_col not in row or np.isnan(row[mae_col]) or np.isnan(row[p_col]):
                continue
            raw_eps = row[mae_col]
            eps_norm = raw_eps / reach if reach > 0 else float("inf")
            if eps_norm <= 0 or not np.isfinite(eps_norm):
                continue
            our_n = reach * row[p_col]
            eppstein_global = np.log(n_nodes) / eps_norm**2
            eppstein_local = np.log(reach) / eps_norm**2
            rows.append(
                {
                    "distance": row["distance"],
                    "metric": metric,
                    "reach": reach,
                    "our_n": our_n,
                    "eps_normalised": eps_norm,
                    "eppstein_global": eppstein_global,
                    "eppstein_local": eppstein_local,
                    "ratio_global": eppstein_global / our_n if our_n > 0 else float("inf"),
                    "ratio_local": eppstein_local / our_n if our_n > 0 else float("inf"),
                }
            )

    if not rows:
        return None

    bounds_df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "madrid_theoretical_bounds_comparison.csv"
    bounds_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    return bounds_df


def compute_bound_analysis(df: pd.DataFrame):
    """Evaluate Hoeffding bounds on both metrics."""
    print("\n" + "=" * 70)
    print("BOUND ANALYSIS (Hoeffding for closeness + betweenness)")
    print("=" * 70)

    rows = []
    for _, row in df.iterrows():
        reach = row["mean_reach"]
        for metric, obs_col, pred_col, p_col in [
            ("harmonic", "eps_obs_h", "eps_pred_h", "hoeffding_p_close"),
            ("betweenness", "eps_obs_b", "eps_pred_b", "hoeffding_p_betw"),
        ]:
            if np.isnan(row[obs_col]) or np.isnan(row[pred_col]):
                continue
            rows.append(
                {
                    "distance": row["distance"],
                    "metric": metric,
                    "reach": reach,
                    "budget": row[p_col],
                    "eps_observed": row[obs_col],
                    "eps_predicted": row[pred_col],
                    "bound_holds": row[obs_col] <= row[pred_col],
                }
            )

    if not rows:
        print("  No configurations to analyse")
        return None

    bound_df = pd.DataFrame(rows)
    total = len(bound_df)
    holds = int(bound_df["bound_holds"].sum())
    print(f"\n  Overall: {holds}/{total} ({100 * holds / total:.1f}%)")

    for metric in sorted(bound_df["metric"].unique()):
        subset = bound_df[bound_df["metric"] == metric]
        h = int(subset["bound_holds"].sum())
        t = len(subset)
        print(f"    {metric}: {h}/{t} ({100 * h / t:.1f}%)")

    csv_path = OUTPUT_DIR / "madrid_bound_analysis.csv"
    bound_df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")
    return bound_df


def generate_validation_table(df: pd.DataFrame, n_nodes: int | None):
    """Generate LaTeX table of Madrid validation results (both metrics)."""
    print("\nGenerating Table: Madrid validation results...")

    eps_c = MADRID_EPSILON_CLOSENESS
    eps_b = MADRID_EPSILON_BETWEENNESS

    latex = rf"""\begin{{table}}[htbp]
\centering
\caption{{Sampling validation on Greater Madrid network
  ($\varepsilon_c = {eps_c}$, $\varepsilon_b = {eps_b}$, $\delta = 0.1$).}}
\label{{tab:madrid_validation}}
\begin{{tabular}}{{rrrrrrrr}}
\toprule
\textbf{{Dist.}} & \textbf{{Reach}} &
\textbf{{$p_c$}} & \textbf{{$\rho_c$}} & \textbf{{Spd$_c$}} &
\textbf{{$p_b$}} & \textbf{{$\rho_b$}} & \textbf{{Spd$_b$}} \\
\midrule
"""

    for _, row in df.iterrows():
        p_c_pct = f"{row['hoeffding_p_close'] * 100:.1f}\\%"
        rho_c = f"{row['rho_closeness']:.4f}"
        spd_c = f"{row['speedup_closeness']:.1f}$\\times$" if np.isfinite(row['speedup_closeness']) else "---"

        if np.isfinite(row['hoeffding_p_betw']):
            p_b_pct = f"{row['hoeffding_p_betw'] * 100:.1f}\\%"
            rho_b = f"{row['rho_betweenness']:.4f}" if np.isfinite(row['rho_betweenness']) else "---"
            spd_b = f"{row['speedup_betweenness']:.1f}$\\times$" if np.isfinite(row['speedup_betweenness']) else "---"
        else:
            p_b_pct = "---"
            rho_b = "---"
            spd_b = "---"

        latex += f"{row['distance'] // 1000}km & {row['mean_reach']:,.0f} & "
        latex += f"{p_c_pct} & {rho_c} & {spd_c} & "
        latex += f"{p_b_pct} & {rho_b} & {spd_b} \\\\\n"

    n_nodes_str = f"{n_nodes:,}" if n_nodes else r"\texttildelow 99,000"
    latex += rf"""\bottomrule
\end{{tabular}}

\vspace{{0.5em}}
\footnotesize
Network: Greater Madrid, {n_nodes_str} nodes. Both metrics use Hoeffding bound
with spatial source selection: $p = \min(1, k/r)$.
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

    print(f"\nCloseness:   Hoeffding + spatial, eps={MADRID_EPSILON_CLOSENESS}")
    print(f"Betweenness: Hoeffding + spatial, eps={MADRID_EPSILON_BETWEENNESS}")
    print(f"Delta: {DELTA}")

    # Generate or load validation data
    df = generate_validation_data(force=args.force)
    print(f"\nValidation data: {len(df)} rows")

    # Theoretical bounds comparison
    n_nodes = get_n_nodes(force=args.force)
    if n_nodes is not None:
        print(f"\nMadrid network: {n_nodes} live nodes")
        compute_theoretical_bounds(df, n_nodes)

    # Generate validation table
    generate_validation_table(df, n_nodes)

    # Bound analysis
    compute_bound_analysis(df)

    # Print summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    print(
        f"\n{'Dist':>6} | {'Reach':>8} | {'p_c':>7} | {'rho_c':>7} | {'Spd_c':>7}"
        f" | {'p_b':>7} | {'rho_b':>7} | {'Spd_b':>7} | {'OK?':>5}"
    )
    print("-" * 80)

    all_pass = True
    for _, row in df.iterrows():
        passes_c = row["rho_closeness"] >= 0.95
        passes_b = np.isnan(row["rho_betweenness"]) or row["rho_betweenness"] >= 0.95
        passes = passes_c and passes_b
        status = "PASS" if passes else "FAIL"
        if not passes:
            all_pass = False

        rho_b_str = f"{row['rho_betweenness']:.4f}" if np.isfinite(row['rho_betweenness']) else "n/a"
        p_b_str = f"{row['hoeffding_p_betw']:.1%}" if np.isfinite(row['hoeffding_p_betw']) else "n/a"
        spd_b_str = f"{row['speedup_betweenness']:.1f}x" if np.isfinite(row['speedup_betweenness']) else "n/a"

        print(
            f"{row['distance'] // 1000}km   | {row['mean_reach']:>8,.0f} | {row['hoeffding_p_close']:>6.1%} | "
            f"{row['rho_closeness']:>7.4f} | {row['speedup_closeness']:>6.1f}x | "
            f"{p_b_str:>7} | {rho_b_str:>7} | {spd_b_str:>7} | {status:>5}"
        )

    print("-" * 80)
    if all_pass:
        print("ALL PASS: rho >= 0.95 for both metrics at all distances.")
    else:
        print("WARNING: Some distances do not meet rho >= 0.95.")

    print("\nOverall Statistics:")
    print(f"  Mean rho (closeness):       {df['rho_closeness'].mean():.4f}")
    print(f"  Mean speedup (closeness):   {df['speedup_closeness'].mean():.2f}x")
    betw_valid = df["rho_betweenness"].dropna()
    if len(betw_valid) > 0:
        print(f"  Mean rho (betweenness):     {betw_valid.mean():.4f}")
        print(f"  Mean speedup (betweenness): {df['speedup_betweenness'].dropna().mean():.2f}x")

    print("\n" + "=" * 70)
    print("OUTPUTS")
    print("=" * 70)
    print(f"  1. {OUTPUT_DIR / 'madrid_validation.csv'}")
    print(f"  2. {TABLES_DIR / 'tab4_madrid_validation.tex'}")
    print(f"  3. {OUTPUT_DIR / 'madrid_theoretical_bounds_comparison.csv'}")
    print(f"  4. {OUTPUT_DIR / 'madrid_bound_analysis.csv'}")

    return 0


if __name__ == "__main__":
    exit(main())
