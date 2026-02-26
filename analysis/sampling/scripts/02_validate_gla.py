#!/usr/bin/env python
"""
02_validate_gla.py - Validate sampling models on Greater London network.

Tests closeness and betweenness (both using Hoeffding + spatial source sampling)
at their respective epsilon values across five distance thresholds.

Usage:
    python 02_validate_gla.py           # Run (skips cache if exists)
    python 02_validate_gla.py --force   # Force regeneration of validation data

Outputs:
    - output/gla_validation.csv
    - output/gla_validation_summary.csv
    - output/gla_ew_analysis.csv
    - output/gla_theoretical_bounds_comparison.csv
    - paper/tables/tab2_validation.tex
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
from cityseer.config import compute_distance_p
from cityseer.metrics import networks
from cityseer.tools import graphs, io
from shapely.geometry import Point
from utilities import (
    CACHE_DIR,
    HOEFFDING_DELTA,
    OUTPUT_DIR,
    TABLES_DIR,
    compute_accuracy_metrics,
    compute_quartile_accuracy,
    ew_predicted_epsilon,
    mean_quartiles,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent

# GLA network file
GLA_GPKG_FILE = SCRIPT_DIR.parent.parent.parent / "temp" / "os_open_roads" / "oproad_gb.gpkg"

# Validation parameters
GLA_DISTANCES = [1000, 2000, 5000, 10000, 20000]
# Hoeffding + spatial source sampling (both metrics)
GLA_EPSILON_CLOSENESS = 0.05
GLA_EPSILON_BETWEENNESS = 0.05
N_RUNS = 1  # GLA is huge; keep runs minimal

DELTA = HOEFFDING_DELTA


def get_gla_mask(force: bool = False):
    """
    Return Greater London boundary and 20km buffered version in EPSG:27700, cached as GeoJSON.

    Returns
    -------
    boundary : shapely geometry
        Simplified GLA boundary polygon -- used to mark live nodes.
    buffered : shapely geometry
        GLA boundary buffered by 20km -- used as spatial filter when loading roads.
    """
    boundary_cache = CACHE_DIR / "gla_boundary.geojson"
    buffered_cache = CACHE_DIR / "gla_buffered.geojson"
    if boundary_cache.exists() and buffered_cache.exists() and not force:
        boundary = gpd.read_file(boundary_cache).geometry.iloc[0]
        buffered = gpd.read_file(buffered_cache).geometry.iloc[0]
        return boundary, buffered
    print("Downloading Greater London boundary from OSM...")
    gdf = ox.geocode_to_gdf("Greater London, England")
    gdf = gdf.to_crs("EPSG:27700")
    boundary = gdf.geometry.iloc[0].simplify(100)  # 100m tolerance
    buffered = boundary.buffer(20_000)
    gpd.GeoDataFrame(geometry=[boundary], crs="EPSG:27700").to_file(boundary_cache, driver="GeoJSON")
    gpd.GeoDataFrame(geometry=[buffered], crs="EPSG:27700").to_file(buffered_cache, driver="GeoJSON")
    print(f"  Cached GLA boundary to {boundary_cache}")
    print(f"  Cached GLA buffered to {buffered_cache}")
    return boundary, buffered


def _is_stale_csv(csv_path: Path) -> bool:
    """Check if a cached CSV has the old format (sample_prob column instead of epsilon)."""
    if not csv_path.exists():
        return False
    try:
        df = pd.read_csv(csv_path, nrows=1)
        if "sample_prob" in df.columns and "epsilon" not in df.columns:
            return True
    except Exception:
        return True
    return False


def _missing_validation_columns(df: pd.DataFrame) -> set[str]:
    """Required columns for downstream summary + bound analysis."""
    required = {
        "distance",
        "mean_reach",
        "epsilon",
        "metric",
        "budget_param",
        "spearman",
        "max_abs_error",
        "eps_observed",
        "eps_predicted",
        "bound_holds",
        "speedup",
    }
    return required - set(df.columns)


def generate_validation_data(force: bool = False) -> pd.DataFrame:
    """Generate GLA validation data, or load from cache."""
    validation_csv = OUTPUT_DIR / "gla_validation.csv"

    # Detect stale format
    if _is_stale_csv(validation_csv):
        print(f"Stale CSV format detected at {validation_csv}, regenerating...")
        force = True

    if validation_csv.exists() and not force:
        df = pd.read_csv(validation_csv)
        missing = _missing_validation_columns(df)
        if len(df) > 0 and not missing:
            print(f"Loading cached validation data from {validation_csv}")
            return df
        if missing:
            print(f"Stale cache columns at {validation_csv}: missing {sorted(missing)}")
        else:
            print(f"Stale/empty cache at {validation_csv}, regenerating...")

    if not GLA_GPKG_FILE.exists():
        raise FileNotFoundError(
            f"GLA network file not found: {GLA_GPKG_FILE}\n  Download the OS Open Roads dataset for Greater London"
        )

    print("\n" + "=" * 70)
    print("GENERATING GLA VALIDATION DATA")
    print("=" * 70)

    # Load GLA boundary (cached after first download)
    gla_boundary, gla_buffered = get_gla_mask(force=force)

    # Load or build GLA graph
    gla_cache = CACHE_DIR / "gla_graph.pkl"
    if gla_cache.exists() and not force:
        print(f"Loading cached GLA graph from {gla_cache}")
        with open(gla_cache, "rb") as f:
            G = pickle.load(f)
    else:
        print(f"Loading GLA network from {GLA_GPKG_FILE}")
        edges_gdf = gpd.read_file(GLA_GPKG_FILE, layer="road_link", mask=gla_buffered)
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

    # Mark live nodes: inside GLA boundary = live, buffer zone = not live
    print("Marking live nodes using GLA boundary...")
    n_live = 0
    for _n, data in G.nodes(data=True):
        data["live"] = gla_boundary.contains(Point(data["x"], data["y"]))
        n_live += data["live"]
    print(f"  Live nodes: {n_live}/{G.number_of_nodes()} ({100 * n_live / G.number_of_nodes():.1f}%)")

    # Convert to cityseer
    print("Converting to cityseer format...")
    nodes_gdf, _, net = io.network_structure_from_nx(G)
    live_mask = nodes_gdf["live"].values

    results = []
    for dist in GLA_DISTANCES:
        print(f"\n{'=' * 50}")
        print(f"Distance: {dist}m")
        print(f"{'=' * 50}")

        # ---------------------------------------------------------------
        # Ground truth: closeness + exact betweenness (cached per distance)
        # ---------------------------------------------------------------
        gt_cache = CACHE_DIR / f"gla_ground_truth_{dist}m.pkl"
        if gt_cache.exists() and not force:
            print(f"  Loading cached ground truth from {gt_cache}")
            with open(gt_cache, "rb") as f:
                gt_data = pickle.load(f)
            true_harmonic = gt_data["harmonic"]
            true_betweenness = gt_data["betweenness"]
            mean_reach = gt_data["mean_reach"]
            node_reach = gt_data.get("node_reach", None)
            baseline_close_time = gt_data.get("baseline_close_time", gt_data.get("baseline_time", None))
            baseline_betw_time = gt_data.get("baseline_betw_time", None)
        else:
            print("  Computing ground truth closeness (this may take a while)...")
            t0 = time.time()
            close_result = net.closeness_shortest(
                distances=[dist],
                pbar_disabled=False,
            )
            baseline_close_time = time.time() - t0
            true_harmonic = np.array(close_result.node_harmonic[dist])[live_mask]
            node_reach = np.array(close_result.node_density[dist])[live_mask]
            mean_reach = float(np.mean(node_reach))
            print(f"  Closeness ground truth: {baseline_close_time:.1f}s")

            print("  Computing ground truth betweenness (exact Brandes)...")
            t0 = time.time()
            betw_result = net.betweenness_shortest(
                distances=[dist],
                pbar_disabled=False,
            )
            baseline_betw_time = time.time() - t0
            true_betweenness = np.array(betw_result.node_betweenness[dist])[live_mask]
            print(f"  Betweenness ground truth: {baseline_betw_time:.1f}s")

            with open(gt_cache, "wb") as f:
                pickle.dump(
                    {
                        "harmonic": true_harmonic,
                        "betweenness": true_betweenness,
                        "node_reach": node_reach,
                        "mean_reach": mean_reach,
                        "n_live": n_live,
                        "baseline_close_time": baseline_close_time,
                        "baseline_betw_time": baseline_betw_time,
                    },
                    f,
                )
            print(f"  Cached ground truth to {gt_cache}")

        print(f"  Mean reach: {mean_reach:.0f}")

        # ---------------------------------------------------------------
        # Closeness: distance-based sampling
        # ---------------------------------------------------------------
        eps_close = GLA_EPSILON_CLOSENESS
        n_live = int(live_mask.sum())
        actual_p_close = compute_distance_p(dist, epsilon=eps_close)
        print(f"\n  Closeness eps={eps_close}: p={actual_p_close:.4f}")

        spearmans_h, maes_h, precs_h, scales_h, quartiles_h = [], [], [], [], []
        close_times = []

        for seed in range(N_RUNS):
            t0 = time.time()
            nodes_gdf_close = networks.closeness_shortest(
                net,
                nodes_gdf.copy(),
                distances=[dist],
                random_seed=42 + seed,
                sample=True,
            )
            close_times.append(time.time() - t0)
            col_key = f"cc_harmonic_{dist}"
            est_harmonic = nodes_gdf_close[col_key].values[live_mask]

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
        n_eff_close = mean_reach * actual_p_close
        eps_pred_close = ew_predicted_epsilon(n_eff_close, mean_reach)

        if spearmans_h:
            eps_obs_h = np.max(maes_h) / mean_reach if maes_h else float("nan")
            row_h = {
                "distance": dist,
                "mean_reach": mean_reach,
                "epsilon": eps_close,
                "metric": "harmonic",
                "budget_param": actual_p_close,
                "spearman": np.mean(spearmans_h),
                "spearman_min": np.min(spearmans_h),
                "spearman_max": np.max(spearmans_h),
                "spearman_std": np.std(spearmans_h) if len(spearmans_h) > 1 else 0.0,
                "max_abs_error": np.max(maes_h) if maes_h else float("nan"),
                "mean_abs_error": np.mean(maes_h) if maes_h else float("nan"),
                "median_abs_error": np.median(maes_h) if maes_h else float("nan"),
                "eps_observed": eps_obs_h,
                "eps_predicted": eps_pred_close,
                "bound_holds": eps_obs_h <= eps_pred_close if np.isfinite(eps_obs_h) else False,
                "top_k_precision": np.mean(precs_h),
                "scale_ratio": np.mean(scales_h),
                "baseline_time": baseline_close_time if baseline_close_time is not None else float("nan"),
                "sampled_time": mean_close_time,
                "speedup": (
                    baseline_close_time / mean_close_time
                    if baseline_close_time and mean_close_time > 0
                    else float("nan")
                ),
            }
            row_h.update(q_h)
            results.append(row_h)

        rho_h_str = f"{np.mean(spearmans_h):.3f}" if spearmans_h else "n/a"
        print(f" rho_h={rho_h_str}")

        # ---------------------------------------------------------------
        # Betweenness: distance-based sampling
        # ---------------------------------------------------------------
        nonzero_betw = np.sum(true_betweenness > 0)
        actual_p_b = float("nan")
        est_betweenness = None
        if nonzero_betw < 10:
            print(f"  Betweenness: skipped (only {nonzero_betw} nonzero)")
        else:
            eps_betw = GLA_EPSILON_BETWEENNESS
            actual_p_b = compute_distance_p(dist, epsilon=eps_betw)
            print(f"\n  Betweenness eps={eps_betw}: p={actual_p_b:.4f}")

            spearmans_b, maes_b, precs_b, scales_b, quartiles_b = [], [], [], [], []
            betw_times = []

            for seed in range(N_RUNS):
                t0 = time.time()
                nodes_gdf_betw = networks.betweenness_shortest(
                    net,
                    nodes_gdf.copy(),
                    distances=[dist],
                    random_seed=42 + seed,
                    sample=True,
                )
                betw_times.append(time.time() - t0)
                col_key = f"cc_betweenness_{dist}"
                est_betweenness = nodes_gdf_betw[col_key].values[live_mask]

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
            n_eff_betw = mean_reach * actual_p_b
            eps_pred_betw = ew_predicted_epsilon(n_eff_betw, mean_reach)

            if spearmans_b:
                eps_obs_b = np.max(maes_b) / mean_reach if maes_b else float("nan")
                row_b = {
                    "distance": dist,
                    "mean_reach": mean_reach,
                    "epsilon": eps_betw,
                    "metric": "betweenness",
                    "budget_param": actual_p_b,
                    "spearman": np.mean(spearmans_b),
                    "spearman_min": np.min(spearmans_b),
                    "spearman_max": np.max(spearmans_b),
                    "spearman_std": np.std(spearmans_b) if len(spearmans_b) > 1 else 0.0,
                    "max_abs_error": np.max(maes_b) if maes_b else float("nan"),
                    "mean_abs_error": np.mean(maes_b) if maes_b else float("nan"),
                    "median_abs_error": np.median(maes_b) if maes_b else float("nan"),
                    "eps_observed": eps_obs_b,
                    "eps_predicted": eps_pred_betw,
                    "bound_holds": eps_obs_b <= eps_pred_betw if np.isfinite(eps_obs_b) else False,
                    "top_k_precision": np.mean(precs_b),
                    "scale_ratio": np.mean(scales_b),
                    "baseline_time": baseline_betw_time if baseline_betw_time is not None else float("nan"),
                    "sampled_time": mean_betw_time,
                    "speedup": (
                        baseline_betw_time / mean_betw_time
                        if baseline_betw_time and mean_betw_time > 0
                        else float("nan")
                    ),
                }
                row_b.update(q_b)
                results.append(row_b)

            rho_b_str = f"{np.mean(spearmans_b):.3f}" if spearmans_b else "n/a"
            print(f" rho_b={rho_b_str}")

        # ---------------------------------------------------------------
        # Save per-node results (exact + sampled) for this distance
        # ---------------------------------------------------------------
        sampled_cache = CACHE_DIR / f"gla_sampled_{dist}m.pkl"
        sampled_data = {
            "distance": dist,
            "mean_reach": mean_reach,
            "node_reach": node_reach,
            "true_harmonic": true_harmonic,
            "est_harmonic": est_harmonic,
            "epsilon_closeness": GLA_EPSILON_CLOSENESS,
            "hoeffding_p": actual_p_close,
            "true_betweenness": true_betweenness,
            "est_betweenness": est_betweenness if est_betweenness is not None else None,
            "epsilon_betweenness": GLA_EPSILON_BETWEENNESS,
            "hoeffding_p_betw": actual_p_b if nonzero_betw >= 10 else None,
        }
        with open(sampled_cache, "wb") as f:
            pickle.dump(sampled_data, f)
        print(f"  Saved per-node results: {sampled_cache}")

    df = pd.DataFrame(results)
    df.to_csv(validation_csv, index=False)
    print(f"\nSaved validation results: {validation_csv}")
    return df


# =============================================================================
# SUMMARY TABLE GENERATION
# =============================================================================


def generate_validation_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Generate summary CSV (closeness + betweenness)."""
    print("\nGenerating validation summary...")

    rows = []

    for dist in sorted(df["distance"].unique()):
        h_rows = df[(df["distance"] == dist) & (df["metric"] == "harmonic")]
        b_rows = df[(df["distance"] == dist) & (df["metric"] == "betweenness")]

        if h_rows.empty:
            continue

        h_row = h_rows.iloc[0]
        reach = h_row["mean_reach"]

        rho_close = h_row["spearman"]
        p_close = h_row["budget_param"]
        speedup_close = h_row["speedup"] if np.isfinite(h_row["speedup"]) else float("nan")

        rho_betw = float("nan")
        p_betw = float("nan")
        speedup_betw = float("nan")
        if not b_rows.empty:
            b_row = b_rows.iloc[0]
            rho_betw = b_row["spearman"]
            p_betw = b_row["budget_param"]
            speedup_betw = b_row["speedup"] if np.isfinite(b_row["speedup"]) else float("nan")

        meets = rho_close >= 0.95 and (np.isnan(rho_betw) or rho_betw >= 0.95)

        rows.append(
            {
                "distance": dist,
                "reach": reach,
                "epsilon_closeness": GLA_EPSILON_CLOSENESS,
                "epsilon_betweenness": GLA_EPSILON_BETWEENNESS,
                "hoeffding_p_close": p_close,
                "hoeffding_p_betw": p_betw,
                "rho_closeness": rho_close,
                "rho_betweenness": rho_betw,
                "speedup_closeness": speedup_close,
                "speedup_betweenness": speedup_betw,
                "meets_target": meets,
            }
        )

    summary_df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "gla_validation_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    return summary_df


# =============================================================================
# LATEX TABLE GENERATION
# =============================================================================


def generate_validation_table(summary_df: pd.DataFrame):
    """Generate LaTeX table of validation results (both metrics)."""
    print("\nGenerating Table 2: Validation results...")

    eps_c = GLA_EPSILON_CLOSENESS
    eps_b = GLA_EPSILON_BETWEENNESS

    latex = rf"""\begin{{table}}[htbp]
\centering
\caption{{Sampling validation on Greater London network
  ($\varepsilon_c = {eps_c}$, $\varepsilon_b = {eps_b}$, $\delta = 0.1$).}}
\label{{tab:validation}}
\begin{{tabular}}{{rrrrrrrr}}
\toprule
\textbf{{Dist.}} & \textbf{{Reach}} &
\textbf{{$p_c$}} & \textbf{{$\rho_c$}} & \textbf{{Spd$_c$}} &
\textbf{{$p_b$}} & \textbf{{$\rho_b$}} & \textbf{{Spd$_b$}} \\
\midrule
"""

    for _, row in summary_df.iterrows():
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

        latex += f"{row['distance'] // 1000}km & {row['reach']:,.0f} & "
        latex += f"{p_c_pct} & {rho_c} & {spd_c} & "
        latex += f"{p_b_pct} & {rho_b} & {spd_b} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}

\vspace{0.5em}
\footnotesize
Network: Greater London Area. Both metrics use Hoeffding bound
with spatial source selection: $p = \min(1, k/r)$.
Subscripts: $c$ = closeness, $b$ = betweenness.
\end{table}
"""

    output_path = TABLES_DIR / "tab2_validation.tex"
    with open(output_path, "w") as f:
        f.write(latex)
    print(f"  Saved: {output_path}")


# =============================================================================
# THEORETICAL BOUNDS COMPARISON
# =============================================================================


def get_n_nodes(force: bool = False) -> int | None:
    """Get live node count from cached GLA graph."""
    n_nodes_cache = CACHE_DIR / "gla_n_nodes.json"
    if n_nodes_cache.exists() and not force:
        with open(n_nodes_cache) as f:
            return json.load(f)["n_nodes"]
    gla_cache = CACHE_DIR / "gla_graph.pkl"
    if not gla_cache.exists():
        return None
    with open(gla_cache, "rb") as f:
        G = pickle.load(f)
    gla_boundary, _ = get_gla_mask(force=force)
    n_nodes = sum(1 for _n, d in G.nodes(data=True) if gla_boundary.contains(Point(d["x"], d["y"])))
    with open(n_nodes_cache, "w") as f:
        json.dump({"n_nodes": n_nodes}, f)
    return n_nodes


def compute_theoretical_bounds(summary_df: pd.DataFrame, n_nodes: int, raw_df: pd.DataFrame):
    """Compare empirical sample counts against Eppstein-Wang bounds for closeness."""
    print("\nComputing theoretical bounds comparison...")

    rows = []

    for _, srow in summary_df.iterrows():
        dist = srow["distance"]
        reach = srow["reach"]

        raw_subset = raw_df[
            (raw_df["distance"] == dist) & (raw_df["metric"] == "harmonic")
        ]
        if raw_subset.empty:
            continue
        raw_row = raw_subset.iloc[0]
        raw_eps = raw_row["max_abs_error"]
        if np.isnan(raw_eps):
            continue

        eps_normalised = raw_eps / reach if reach > 0 else float("inf")

        if eps_normalised <= 0 or not np.isfinite(eps_normalised):
            continue

        # Our budget
        our_eff_n = reach * srow["hoeffding_p_close"]

        # Eppstein & Wang (2004): source-sampling bound
        eppstein_samples = np.log(n_nodes) / eps_normalised**2
        eppstein_local_samples = np.log(reach) / eps_normalised**2

        rows.append(
            {
                "distance": dist,
                "metric": "harmonic",
                "reach": reach,
                "our_eff_n": our_eff_n,
                "raw_eps": raw_eps,
                "eps_normalised": eps_normalised,
                "eppstein_samples": eppstein_samples,
                "eppstein_local_samples": eppstein_local_samples,
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
        f"\n  {'Distance':>10} | {'Metric':>12} | {'Our budget':>10} | {'EW global':>12} | {'EW local':>12} | "
        f"{'R(EWg)':>8} | {'R(EWl)':>8}"
    )
    print("  " + "-" * 90)
    for _, r in bounds_df.iterrows():
        print(
            f"  {r['distance'] // 1000}km       | "
            f"{r['metric']:>12} | "
            f"{r['our_eff_n']:>10,.0f} | "
            f"{r['eppstein_samples']:>12,.0f} | "
            f"{r['eppstein_local_samples']:>12,.0f} | "
            f"{r['ratio_eppstein']:>8.1f}x | "
            f"{r['ratio_eppstein_local']:>8.1f}x"
        )

    return bounds_df


# =============================================================================
# EW BOUND ANALYSIS
# =============================================================================


def compute_bound_analysis(raw_df: pd.DataFrame):
    """Evaluate Hoeffding bounds on all configurations (both metrics)."""
    print("\n" + "=" * 70)
    print("BOUND ANALYSIS (Hoeffding for closeness + betweenness)")
    print("=" * 70)

    rows = []
    for _, row in raw_df.iterrows():
        metric = row["metric"]
        eps_obs = row["eps_observed"]
        eps_pred = row["eps_predicted"]
        bound_holds = row["bound_holds"]

        if not np.isfinite(eps_obs):
            continue

        rows.append(
            {
                "distance": row["distance"],
                "metric": metric,
                "bound_type": "Hoeffding",
                "reach": row["mean_reach"],
                "epsilon": row["epsilon"],
                "budget_param": row["budget_param"],
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
    print(f"\n  Overall: {holds}/{total} ({100 * holds / total:.1f}%) -- expected >= {100 * (1 - DELTA):.0f}%")

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

    # Conservatism
    subset = ew_df[ew_df["eps_observed"] > 0]
    if len(subset) > 0:
        ratio = subset["eps_predicted"] / subset["eps_observed"]
        print(f"\n  Conservatism (Hoeffding): median predicted/observed = {ratio.median():.1f}x")

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
    print("02_validate_gla.py - Validating sampling models on Greater London network")
    print("=" * 70)

    print(f"\nCloseness:   Hoeffding + spatial, eps={GLA_EPSILON_CLOSENESS}")
    print(f"Betweenness: Hoeffding + spatial, eps={GLA_EPSILON_BETWEENNESS}")
    print(f"  Delta: {DELTA}")
    print(f"  Distances: {GLA_DISTANCES}")
    print(f"  N_RUNS: {N_RUNS}")

    # Generate or load validation data
    raw_df = generate_validation_data(force=args.force)
    print(f"\nValidation data: {len(raw_df)} rows")
    print(f"Distances: {sorted(raw_df['distance'].unique())}")

    # Generate summary
    summary_df = generate_validation_summary(raw_df)

    # Generate LaTeX table
    generate_validation_table(summary_df)

    # Theoretical bounds comparison
    n_nodes = get_n_nodes(force=args.force)
    bounds_df = None
    if n_nodes is not None:
        print(f"\nGLA network: {n_nodes} live nodes")
        bounds_df = compute_theoretical_bounds(summary_df, n_nodes, raw_df)
    else:
        print("\n  Skipping theoretical bounds comparison (graph cache not found)")

    # Bound analysis (Hoeffding for closeness)
    ew_df = compute_bound_analysis(raw_df)

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    print(
        f"\n{'Distance':>10} | {'Reach':>10} | {'p_c':>8} | {'rho_c':>8}"
        f" | {'Spd_c':>8} | {'p_b':>8} | {'rho_b':>8} | {'Spd_b':>8} | {'OK?':>5}"
    )
    print("-" * 95)

    all_pass = True
    for _, row in summary_df.iterrows():
        status = "PASS" if row["meets_target"] else "FAIL"
        if not row["meets_target"]:
            all_pass = False

        rho_b_str = f"{row['rho_betweenness']:.4f}" if np.isfinite(row['rho_betweenness']) else "n/a"
        p_b_str = f"{row['hoeffding_p_betw']:.1%}" if np.isfinite(row['hoeffding_p_betw']) else "n/a"
        spd_b_str = f"{row['speedup_betweenness']:.1f}x" if np.isfinite(row['speedup_betweenness']) else "n/a"

        print(
            f"{row['distance'] // 1000}km       | {row['reach']:>10,.0f} | {row['hoeffding_p_close']:>7.1%} | "
            f"{row['rho_closeness']:>8.4f} | {row['speedup_closeness']:>7.1f}x | "
            f"{p_b_str:>8} | {rho_b_str:>8} | {spd_b_str:>8} | {status:>5}"
        )

    print("\n" + "-" * 95)
    if all_pass:
        print("ALL DISTANCES PASS: rho >= 0.95 for both metrics at all distances.")
    else:
        print("WARNING: Some distances do not meet the rho >= 0.95 target.")

    # Overall statistics
    print("\nOverall Statistics:")
    print(f"  Mean speedup (closeness):   {summary_df['speedup_closeness'].mean():.2f}x")
    print(f"  Mean rho (closeness):       {summary_df['rho_closeness'].mean():.4f}")
    betw_valid = summary_df["rho_betweenness"].dropna()
    if len(betw_valid) > 0:
        print(f"  Mean speedup (betweenness): {summary_df['speedup_betweenness'].dropna().mean():.2f}x")
        print(f"  Mean rho (betweenness):     {betw_valid.mean():.4f}")

    print("\n" + "=" * 70)
    print("OUTPUTS")
    print("=" * 70)
    print(f"  1. {OUTPUT_DIR / 'gla_validation.csv'}")
    print(f"  2. {OUTPUT_DIR / 'gla_validation_summary.csv'}")
    print(f"  3. {TABLES_DIR / 'tab2_validation.tex'}")
    if bounds_df is not None:
        print(f"  4. {OUTPUT_DIR / 'gla_theoretical_bounds_comparison.csv'}")
    if ew_df is not None:
        print(f"  5. {OUTPUT_DIR / 'gla_ew_analysis.csv'}")

    return 0


if __name__ == "__main__":
    exit(main())
