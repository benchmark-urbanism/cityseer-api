#!/usr/bin/env python
"""
03_validate_cary.py - External validation on Cary, NC (suburban) network.

Third validation network: a low-connectivity planned suburb (Raleigh / Research
Triangle), the sparsest of the three (cf. dense Greater London and Greater Madrid)
and the real-world analog of the synthetic "tree" topology — the regime where the
epsilon schedule's conservatism matters most.

Builds the exact ground truths and the ablation rung-1 record: closeness and
betweenness under the canonical schedule p = compute_distance_p(d), driven
explicitly through the low-level API with the schedule's p >= phi fallback to
exact computation. Since cityseer 4.25 the high-level sample=True flag runs the
per-node polled-pilot method (validate_adaptive.py), not this schedule.

Data: TIGER/Line 2023 EDGES (topologically integrated, natively noded) for the
seven NC counties intersecting Cary + 20 km. Fetch with fetch_tiger.py first (its header gives
the Cary invocation).

Usage:
    python 03_validate_cary.py           # Run (skips cache if exists)
    python 03_validate_cary.py --force   # Force regeneration of validation data

Outputs:
    - output/cary_validation.csv
    - output/cary_theoretical_bounds_comparison.csv
    - output/cary_bound_analysis.csv
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
from cityseer.metrics import networks
from cityseer.sampling import compute_distance_p
from cityseer.tools import graphs, io
from shapely.geometry import Point
from utilities import (
    CACHE_DIR,
    HOEFFDING_DELTA,
    OUTPUT_DIR,
    assert_mask_within_data,
    canonical_reach,
    compute_accuracy_metrics,
    compute_quartile_accuracy,
    ew_predicted_epsilon,
    mean_quartiles,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent

# Cary TIGER/Line EDGES directory (per-county zips from fetch_tiger.py)
CARY_EDGES_DIR = SCRIPT_DIR.parent.parent.parent / "temp" / "tiger_cary"
CARY_CRS = "EPSG:32119"  # NAD83 / North Carolina (metres)

# Validation parameters
CARY_DISTANCES = [1000, 2000, 5000, 10000, 20000]
N_RUNS = 3

# Hoeffding + deterministic distance-based source sampling (both metrics)
CARY_EPSILON_CLOSENESS = 0.05
CARY_EPSILON_BETWEENNESS = 0.05

DELTA = HOEFFDING_DELTA

# Sensitivity analysis: grid spacings to test (default s=175m is the paper default)
DEFAULT_GRID_SPACINGS = [125, 150, 175, 200, 225]

def get_cary_mask(force: bool = False):
    """Return Cary boundary (marks live) and its 20km buffered version (road-load mask) in EPSG:32119.

    Mirrors the GLA pattern: the analysis area (Cary) is a sub-region carved out of a
    much larger dataset (seven NC counties), so the buffer is built *outward* and is
    fully covered by the data (asserted by the loader guard).
    """
    boundary_cache = CACHE_DIR / "cary_boundary.geojson"
    buffered_cache = CACHE_DIR / "cary_buffered.geojson"
    if boundary_cache.exists() and buffered_cache.exists() and not force:
        boundary = gpd.read_file(boundary_cache).geometry.iloc[0]
        buffered = gpd.read_file(buffered_cache).geometry.iloc[0]
        return boundary, buffered
    print("Downloading Cary, NC boundary from OSM...")
    gdf = ox.geocode_to_gdf("Cary, North Carolina, USA").to_crs(CARY_CRS)
    boundary = gdf.geometry.iloc[0].simplify(100)
    buffered = boundary.buffer(20_000)
    gpd.GeoDataFrame(geometry=[boundary], crs=CARY_CRS).to_file(boundary_cache, driver="GeoJSON")
    gpd.GeoDataFrame(geometry=[buffered], crs=CARY_CRS).to_file(buffered_cache, driver="GeoJSON")
    print(f"  Cached Cary boundary to {boundary_cache}")
    return boundary, buffered


def load_cary_network(force: bool = False):
    """Load (or build) the Cary network and return (net, nodes_gdf, live_mask, n_live).

    Uses the TIGER EDGES layer (ROADFLG=='Y'), which is natively noded — no planar
    noding required, and correct at grade-separated crossings.
    """
    cary_boundary, cary_buffered = get_cary_mask(force=force)

    cary_cache = CACHE_DIR / "cary_graph.pkl"
    if cary_cache.exists() and not force:
        print(f"Loading cached Cary graph from {cary_cache}")
        with open(cary_cache, "rb") as f:
            G = pickle.load(f)
    else:
        edge_files = sorted(CARY_EDGES_DIR.glob("tl_2023_*_edges.zip"))
        if not edge_files:
            raise FileNotFoundError(
                f"No TIGER edge files in {CARY_EDGES_DIR}. "
                "Run fetch_tiger.py first (its header gives the Cary invocation)."
            )
        print(f"Loading Cary network from {len(edge_files)} TIGER EDGES files")
        parts = [gpd.read_file(z).to_crs(CARY_CRS) for z in edge_files]
        edges_gdf = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs=CARY_CRS)
        # ROADFLG=='Y' selects the road network from the topologically-integrated EDGES layer
        edges_gdf = edges_gdf[(edges_gdf["ROADFLG"] == "Y") & edges_gdf.intersects(cary_buffered)]
        edges_gdf = edges_gdf[edges_gdf.geometry.is_valid & ~edges_gdf.geometry.is_empty].explode(index_parts=False)
        edges_gdf.geometry = edges_gdf.geometry.map(shapely.force_2d)
        print(f"  Loaded: {len(edges_gdf)} road edges")
        # Guard: the 20km road-mask must lie within the available data (mask and edges both EPSG:32119)
        assert_mask_within_data(cary_buffered, edges_gdf, "Cary")

        print("  Building graph...")
        G = io.nx_from_generic_geopandas(edges_gdf)
        G = graphs.nx_remove_filler_nodes(G)
        G = graphs.nx_remove_dangling_nodes(G)
        G = graphs.nx_consolidate_nodes(G, buffer_dist=10)
        G = graphs.nx_remove_dangling_nodes(G, despine=20)

        print(f"  Caching to {cary_cache}")
        with open(cary_cache, "wb") as f:
            pickle.dump(G, f)

    print(f"Cary graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Mark live nodes
    print("Marking live nodes using Cary boundary...")
    n_live = 0
    for _n, data in G.nodes(data=True):
        data["live"] = cary_boundary.contains(Point(data["x"], data["y"]))
        n_live += data["live"]
    print(f"  Live nodes: {n_live}/{G.number_of_nodes()} ({100 * n_live / G.number_of_nodes():.1f}%)")

    # Convert to cityseer
    print("Converting to cityseer format...")
    nodes_gdf, _, net = io.network_structure_from_nx(G)
    live_mask = nodes_gdf["live"].values

    return net, nodes_gdf, live_mask, n_live


def generate_validation_data(net, nodes_gdf, live_mask, force: bool = False) -> pd.DataFrame:
    """Generate Cary validation data, or load from cache."""
    validation_csv = OUTPUT_DIR / "cary_validation.csv"
    required_cols = {
        "distance",
        "mean_reach",
        "rho_closeness",
        "hoeffding_p_close",
        "speedup_closeness",
        "rho_betweenness",
        "hoeffding_p_betw",
        "speedup_betweenness",
        "eps_obs_h",
        "eps_pred_h",
    }

    if validation_csv.exists() and not force:
        df = pd.read_csv(validation_csv)
        missing = required_cols - set(df.columns)
        if len(df) > 0 and not missing:
            print(f"Loading cached validation data from {validation_csv}")
            return df
        if missing:
            print(f"Stale cache columns at {validation_csv}: missing {sorted(missing)}, regenerating...")
        else:
            print(f"Stale/empty cache at {validation_csv}, regenerating...")

    print("\n" + "=" * 70)
    print("GENERATING CARY VALIDATION DATA")
    print("=" * 70)

    n_live = int(live_mask.sum())

    # Extract live node coordinates for spatial plots
    node_x = nodes_gdf["x"].values[live_mask]
    node_y = nodes_gdf["y"].values[live_mask]

    results = []

    for dist in CARY_DISTANCES:
        print(f"\n{'=' * 50}")
        print(f"Distance: {dist}m")
        print(f"{'=' * 50}")

        # Ground truth (cached per distance)
        gt_cache = CACHE_DIR / f"cary_ground_truth_{dist}m.pkl"
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
            # Exact baselines are measured PER METRIC: the 4.25 redesign gives them different
            # source sets (closeness-exact sources only live nodes; betweenness-exact sources
            # every node), so a 50/50 split of one combined run would misstate both speedups.
            # Use the same high-level wrappers as the sampled path (sample=False) so exact and
            # sampled share an identical code path and the speedup ratio is apples-to-apples.
            print("  Computing exact ground truth (closeness-only, then betweenness-only)...")
            t0 = time.time()
            gdf_close = networks.closeness_shortest(net, nodes_gdf.copy(), distances=[dist])
            baseline_time_closeness = time.time() - t0  # exact closeness sources n_live
            true_harmonic = gdf_close[f"cc_harmonic_{dist}"].to_numpy()[live_mask]
            node_reach = gdf_close[f"cc_density_{dist}"].to_numpy()[live_mask]
            mean_reach = float(np.mean(node_reach))

            t0 = time.time()
            gdf_betw = networks.betweenness_shortest(net, nodes_gdf.copy(), distances=[dist])
            baseline_time_betweenness = time.time() - t0  # exact betweenness sources n_total
            true_betweenness = gdf_betw[f"cc_betweenness_{dist}"].to_numpy()[live_mask]
            print(
                f"  Ground truth exact: closeness={baseline_time_closeness:.1f}s (n_live), "
                f"betweenness={baseline_time_betweenness:.1f}s (n_total)"
            )

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

        # ---------------------------------------------------------------
        # Closeness: distance-based sampling
        # ---------------------------------------------------------------
        phi = float(np.mean(live_mask))
        actual_p_close = compute_distance_p(dist, epsilon=CARY_EPSILON_CLOSENESS)
        close_exact = actual_p_close >= phi  # the schedule's fallback: p >= phi runs the whole call exact
        print(
            f"  Closeness eps={CARY_EPSILON_CLOSENESS}: p={actual_p_close:.4f} "
            f"({'exact' if close_exact else 'sampled'})"
        )

        spearmans_h, maes_h, precs_h, scales_h, quartiles_h = [], [], [], [], []
        close_times = []

        print(f"    Running {N_RUNS} runs: ", end="", flush=True)
        for seed in range(N_RUNS):
            t0 = time.time()
            if close_exact:
                nodes_gdf_close = networks.closeness_shortest(net, nodes_gdf.copy(), distances=[dist])
                est_harmonic = nodes_gdf_close[f"cc_harmonic_{dist}"].values[live_mask]
            else:
                # Canonical schedule via the low-level API: sample=True now runs the
                # per-node method, so the schedule's p is supplied explicitly.
                result = net.centrality_shortest(
                    distances=[dist],
                    closeness_exprs=[("harmonic", "1/c")],
                    betweenness_exprs=[],
                    compute_cycles=False,
                    sample_probability=float(actual_p_close),
                    random_seed=42 + seed,
                    pbar_disabled=True,
                )
                est_harmonic = np.array(result.metrics["harmonic"][dist])[live_mask]
            close_times.append(time.time() - t0)

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
        r_canonical = canonical_reach(dist)
        eps_pred_h = ew_predicted_epsilon(dist, actual_p_close, delta=DELTA)
        max_mae_h = np.max(maes_h) if maes_h else float("nan")
        eps_obs_h = max_mae_h / r_canonical if not np.isnan(max_mae_h) else float("nan")
        rho_h = np.mean(spearmans_h) if spearmans_h else float("nan")
        speedup_h = (
            baseline_time_closeness / mean_close_time
            if baseline_time_closeness and mean_close_time > 0
            else float("nan")
        )
        print(f" rho={rho_h:.3f}, speedup={speedup_h:.1f}x")

        # ---------------------------------------------------------------
        # Betweenness: distance-based sampling
        # ---------------------------------------------------------------
        nonzero_betw = np.sum(true_betweenness > 0)
        rho_b = float("nan")
        speedup_b = float("nan")
        actual_p_betw = float("nan")
        max_mae_b = float("nan")
        eps_obs_b = float("nan")
        eps_pred_b = float("nan")
        q_b = {}
        est_betweenness = None
        spearmans_b_list = []

        if nonzero_betw < 10:
            print(f"  Betweenness: skipped (only {nonzero_betw} nonzero)")
        else:
            actual_p_betw = compute_distance_p(dist, epsilon=CARY_EPSILON_BETWEENNESS)
            betw_exact = actual_p_betw >= phi  # the schedule's fallback applies to the whole call
            print(
                f"  Betweenness eps={CARY_EPSILON_BETWEENNESS}: p={actual_p_betw:.4f} "
                f"({'exact' if betw_exact else 'sampled'})"
            )

            spearmans_b, maes_b, precs_b, scales_b, quartiles_b = [], [], [], [], []
            betw_times = []

            print(f"    Running {N_RUNS} runs: ", end="", flush=True)
            for seed in range(N_RUNS):
                t0 = time.time()
                if betw_exact:
                    nodes_gdf_betw = networks.betweenness_shortest(net, nodes_gdf.copy(), distances=[dist])
                    est_betweenness = nodes_gdf_betw[f"cc_betweenness_{dist}"].values[live_mask]
                else:
                    result = net.centrality_shortest(
                        distances=[dist],
                        closeness_exprs=[],
                        betweenness_exprs=[("betweenness", "1")],
                        compute_cycles=False,
                        sample_probability=float(actual_p_betw),
                        random_seed=42 + seed,
                        pbar_disabled=True,
                    )
                    est_betweenness = np.array(result.metrics["betweenness"][dist])[live_mask]
                betw_times.append(time.time() - t0)

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
            eps_pred_b = ew_predicted_epsilon(dist, actual_p_betw, delta=DELTA)
            max_mae_b = np.max(maes_b) if maes_b else float("nan")
            eps_obs_b = max_mae_b / r_canonical if not np.isnan(max_mae_b) else float("nan")
            rho_b = np.mean(spearmans_b) if spearmans_b else float("nan")
            speedup_b = (
                baseline_time_betweenness / mean_betw_time
                if baseline_time_betweenness and mean_betw_time > 0
                else float("nan")
            )
            spearmans_b_list = spearmans_b
            print(f" rho={rho_b:.3f}, speedup={speedup_b:.1f}x")

        # ---------------------------------------------------------------
        # Save per-node results (exact + sampled) for this distance
        # ---------------------------------------------------------------
        sampled_cache = CACHE_DIR / f"cary_sampled_{dist}m.pkl"
        sampled_data = {
            "distance": dist,
            "mean_reach": mean_reach,
            "node_reach": node_reach,
            "node_x": node_x,
            "node_y": node_y,
            "true_harmonic": true_harmonic,
            "est_harmonic": est_harmonic,
            "epsilon_closeness": CARY_EPSILON_CLOSENESS,
            "hoeffding_p": actual_p_close,
            "spearmans_closeness": spearmans_h,
            "true_betweenness": true_betweenness,
            "est_betweenness": est_betweenness,
            "epsilon_betweenness": CARY_EPSILON_BETWEENNESS,
            "hoeffding_p_betw": actual_p_betw if nonzero_betw >= 10 else None,
            "spearmans_betweenness": spearmans_b_list if nonzero_betw >= 10 else None,
        }
        with open(sampled_cache, "wb") as f:
            pickle.dump(sampled_data, f)
        print(f"  Saved per-node results: {sampled_cache}")

        # Build result row
        row = {
            "distance": dist,
            "mean_reach": mean_reach,
            # Closeness
            "epsilon_closeness": CARY_EPSILON_CLOSENESS,
            "hoeffding_p_close": actual_p_close,
            "rho_closeness": rho_h,
            "max_abs_error_h": max_mae_h,
            "eps_obs_h": eps_obs_h,
            "eps_pred_h": eps_pred_h,
            "top_k_precision_h": np.mean(precs_h) if precs_h else float("nan"),
            "baseline_time_h": baseline_time_closeness if baseline_time_closeness else float("nan"),
            "sampled_time_h": mean_close_time,
            "speedup_closeness": speedup_h,
            # Betweenness
            "epsilon_betweenness": CARY_EPSILON_BETWEENNESS,
            "hoeffding_p_betw": actual_p_betw,
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


def get_n_nodes(force: bool = False) -> dict | None:
    """Get live and total node counts from cached Cary graph.

    Returns dict with 'n_nodes' (live), 'n_total', and 'live_fraction'.
    """
    n_nodes_cache = CACHE_DIR / "cary_n_nodes.json"
    if n_nodes_cache.exists() and not force:
        with open(n_nodes_cache) as f:
            data = json.load(f)
            if "n_total" in data:
                return data
    cary_cache = CACHE_DIR / "cary_graph.pkl"
    if not cary_cache.exists():
        return None
    with open(cary_cache, "rb") as f:
        G = pickle.load(f)
    cary_boundary, _ = get_cary_mask(force=force)
    n_total = G.number_of_nodes()
    n_nodes = sum(1 for _n, d in G.nodes(data=True) if cary_boundary.contains(Point(d["x"], d["y"])))
    data = {"n_nodes": n_nodes, "n_total": n_total, "live_fraction": n_nodes / n_total if n_total > 0 else 1.0}
    with open(n_nodes_cache, "w") as f:
        json.dump(data, f)
    return data


def compute_theoretical_bounds(df: pd.DataFrame, n_nodes: int):
    """Compare empirical sample counts against Eppstein-Wang bounds."""
    print("\nComputing theoretical bounds comparison...")

    rows = []
    for _, row in df.iterrows():
        dist = row["distance"]
        r = canonical_reach(dist)

        for metric, p_col, mae_col in [
            ("harmonic", "hoeffding_p_close", "max_abs_error_h"),
            ("betweenness", "hoeffding_p_betw", "max_abs_error_b"),
        ]:
            if mae_col not in row or np.isnan(row[mae_col]) or np.isnan(row[p_col]):
                continue
            raw_eps = row[mae_col]
            eps_norm = raw_eps / r if r > 0 else float("inf")
            if eps_norm <= 0 or not np.isfinite(eps_norm):
                continue
            our_n = r * row[p_col]
            eppstein_global = np.log(n_nodes) / eps_norm**2
            eppstein_local = np.log(r) / eps_norm**2
            rows.append(
                {
                    "distance": dist,
                    "metric": metric,
                    "canonical_reach": r,
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
    csv_path = OUTPUT_DIR / "cary_theoretical_bounds_comparison.csv"
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
                    "canonical_reach": canonical_reach(row["distance"]),
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

    csv_path = OUTPUT_DIR / "cary_bound_analysis.csv"
    bound_df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")
    return bound_df


# =============================================================================
# SENSITIVITY ANALYSIS: GRID SPACING
# =============================================================================


def run_sensitivity_analysis(
    net,
    nodes_gdf: gpd.GeoDataFrame,
    live_mask: np.ndarray,
    grid_spacings: list[float],
    distances: list[int] | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Run sampling at multiple grid spacings to test sensitivity of ρ to s.

    Reuses cached ground truth; only re-runs sampled centrality at each (distance, s).
    Uses the Rust API with sample_probability + random_seed.
    """
    if distances is None:
        distances = [10000, 20000]  # only long distances where sampling matters

    sensitivity_csv = OUTPUT_DIR / "cary_sensitivity.csv"
    if sensitivity_csv.exists() and not force:
        print(f"\nSensitivity results already exist: {sensitivity_csv}")
        return pd.read_csv(sensitivity_csv)

    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS: GRID SPACING")
    print(f"  Spacings: {grid_spacings}")
    print(f"  Distances: {distances}")
    print("=" * 70)

    rows = []
    for dist in distances:
        # Load ground truth
        gt_cache = CACHE_DIR / f"cary_ground_truth_{dist}m.pkl"
        if not gt_cache.exists():
            print(f"  Skipping {dist}m — no ground truth cache")
            continue
        with open(gt_cache, "rb") as f:
            gt_data = pickle.load(f)
        true_harmonic = gt_data["harmonic"]
        true_betweenness = gt_data["betweenness"]
        mean_reach = gt_data["mean_reach"]

        for s in grid_spacings:
            p = compute_distance_p(dist, epsilon=CARY_EPSILON_CLOSENESS, grid_spacing=float(s))
            print(f"\n  d={dist}m, s={s}m: p={p:.4f}")

            if p >= 1.0:
                print("    p=1.0 (exact), skipping")
                rows.append(
                    {
                        "distance": dist,
                        "grid_spacing": s,
                        "p": p,
                        "rho_closeness": 1.0,
                        "rho_betweenness": 1.0,
                        "mean_reach": mean_reach,
                    }
                )
                continue

            # Closeness
            spearmans_h = []
            for seed in range(N_RUNS):
                result = net.centrality_shortest(
                    distances=[dist],
                    closeness_exprs=[("harmonic", "1/c")],
                    betweenness_exprs=[],
                    compute_cycles=False,
                    sample_probability=p,
                    random_seed=42 + seed,
                    pbar_disabled=True,
                )
                est_h = np.array(result.metrics["harmonic"][dist])[live_mask]
                sp_h, _, _, _, _ = compute_accuracy_metrics(true_harmonic, est_h)
                if not np.isnan(sp_h):
                    spearmans_h.append(sp_h)
                print(".", end="", flush=True)

            # Betweenness
            spearmans_b = []
            if np.sum(true_betweenness > 0) >= 10:
                for seed in range(N_RUNS):
                    result = net.centrality_shortest(
                        distances=[dist],
                        closeness_exprs=[],
                        betweenness_exprs=[("betweenness", "1")],
                        compute_cycles=False,
                        sample_probability=p,
                        random_seed=42 + seed,
                        pbar_disabled=True,
                    )
                    est_b = np.array(result.metrics["betweenness"][dist])[live_mask]
                    sp_b, _, _, _, _ = compute_accuracy_metrics(true_betweenness, est_b)
                    if not np.isnan(sp_b):
                        spearmans_b.append(sp_b)
                    print(".", end="", flush=True)

            rho_h = np.mean(spearmans_h) if spearmans_h else float("nan")
            rho_b = np.mean(spearmans_b) if spearmans_b else float("nan")
            print(f" rho_c={rho_h:.4f}, rho_b={rho_b:.4f}")

            rows.append(
                {
                    "distance": dist,
                    "grid_spacing": s,
                    "p": p,
                    "rho_closeness": rho_h,
                    "rho_betweenness": rho_b,
                    "mean_reach": mean_reach,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(sensitivity_csv, index=False)
    print(f"\nSaved sensitivity results: {sensitivity_csv}")
    return df


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Validate sampling model on Cary, NC network")
    parser.add_argument("--force", action="store_true", help="Force regeneration of validation data")
    parser.add_argument(
        "--sensitivity",
        action="store_true",
        help="Run grid spacing sensitivity analysis after validation",
    )
    parser.add_argument(
        "--grid-spacings",
        type=float,
        nargs="+",
        default=None,
        help=f"Grid spacings for sensitivity analysis (default: {DEFAULT_GRID_SPACINGS})",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("03_validate_cary.py - External validation on Cary, NC (suburban) network")
    print("=" * 70)

    print(f"\nCloseness:   Hoeffding + deterministic distance-based, eps={CARY_EPSILON_CLOSENESS}")
    print(f"Betweenness: Hoeffding + deterministic distance-based, eps={CARY_EPSILON_BETWEENNESS}")
    print(f"Delta: {DELTA}")

    # Load network (needed for both validation and sensitivity)
    net, nodes_gdf, live_mask, n_live_count = load_cary_network(force=args.force)

    # Generate or load validation data
    df = generate_validation_data(net, nodes_gdf, live_mask, force=args.force)
    print(f"\nValidation data: {len(df)} rows")

    # Theoretical bounds comparison
    node_info = get_n_nodes(force=args.force)
    n_nodes = None
    if node_info is not None:
        n_nodes = node_info["n_nodes"]
        print(f"\nCary network: {n_nodes} live / {node_info['n_total']} total (φ={node_info['live_fraction']:.3f})")
        compute_theoretical_bounds(df, n_nodes)

    # Bound analysis
    compute_bound_analysis(df)

    # Print summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    print(f"\n{'Dist':>6} | {'p':>7} | {'rho_c':>7} | {'Spd_c':>7} | {'rho_b':>7} | {'Spd_b':>7} | {'OK?':>5}")
    print("-" * 65)

    all_pass = True
    for _, row in df.iterrows():
        passes_c = row["rho_closeness"] >= 0.95
        passes_b = np.isnan(row["rho_betweenness"]) or row["rho_betweenness"] >= 0.95
        passes = passes_c and passes_b
        status = "PASS" if passes else "FAIL"
        if not passes:
            all_pass = False

        rho_b_str = f"{row['rho_betweenness']:.4f}" if np.isfinite(row["rho_betweenness"]) else "n/a"
        spd_b_str = f"{row['speedup_betweenness']:.1f}x" if np.isfinite(row["speedup_betweenness"]) else "n/a"

        print(
            f"{row['distance'] // 1000}km   | {row['hoeffding_p_close']:>6.1%} | "
            f"{row['rho_closeness']:>7.4f} | {row['speedup_closeness']:>6.1f}x | "
            f"{rho_b_str:>7} | {spd_b_str:>7} | {status:>5}"
        )

    print("-" * 65)
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
    print(f"  1. {OUTPUT_DIR / 'cary_validation.csv'}")
    print(f"  2. {OUTPUT_DIR / 'cary_theoretical_bounds_comparison.csv'}")
    print(f"  3. {OUTPUT_DIR / 'cary_bound_analysis.csv'}")

    # Sensitivity analysis (optional)
    if args.sensitivity:
        spacings = args.grid_spacings or DEFAULT_GRID_SPACINGS
        run_sensitivity_analysis(
            net,
            nodes_gdf,
            live_mask,
            grid_spacings=spacings,
            force=args.force,
        )
        print(f"  5. {OUTPUT_DIR / 'cary_sensitivity.csv'}")

    return 0


if __name__ == "__main__":
    exit(main())
