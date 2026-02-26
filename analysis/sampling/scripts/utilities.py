"""
Utility Functions for Sampling Analysis

Provides shared configuration, statistical utilities, and helper functions
for all sampling analysis scripts.

Sections:
1. Configuration and paths
2. Statistical utilities (accuracy metrics, quartile analysis)
3. Network helpers (live buffer)
4. Hoeffding / EW bound utilities (analysis-specific)

Note: Distance-based sampling probability uses cityseer.config.compute_distance_p.
      Hoeffding probability (for cache sweep) uses cityseer.config.compute_hoeffding_p.
"""

import math
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
from scipy import stats as scipy_stats
from shapely.geometry import Point
from shapely.ops import unary_union

warnings.filterwarnings("ignore")

# =============================================================================
# SECTION 1: Configuration and Paths
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
SAMPLING_DIR = SCRIPT_DIR.parent  # analysis/sampling
CACHE_DIR = SAMPLING_DIR / ".cache"
OUTPUT_DIR = SAMPLING_DIR / "output"
PAPER_DIR = SAMPLING_DIR / "paper"
FIGURES_DIR = PAPER_DIR / "figures"
TABLES_DIR = PAPER_DIR / "tables"

# Ensure directories exist
for d in [CACHE_DIR, OUTPUT_DIR, FIGURES_DIR, TABLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Cache version for invalidation — bump this to force all caches to regenerate
# Versioned filenames (synthetic pkl, validation CSVs) auto-regenerate on bump.
# Network graphs (gla_graph.pkl, gla_ground_truth_*.pkl) are unversioned and persist.
CACHE_VERSION = "v33"

# Canonical per-quartile key prefixes — single source of truth for fallback/perfect blocks
QUARTILE_KEYS = ("spearman", "mae", "max_error", "reach")

# Analysis-specific Hoeffding defaults (stricter than library default of 0.1)
HOEFFDING_EPSILON = 0.05  # Normalised additive error tolerance
HOEFFDING_DELTA = 0.1  # Failure probability (90% confidence)


# =============================================================================
# SECTION 2: Statistical Utilities
# =============================================================================


def compute_accuracy_metrics(true_vals: np.ndarray, est_vals: np.ndarray) -> tuple:
    """
    Compute ranking and magnitude accuracy metrics.

    Parameters
    ----------
    true_vals : np.ndarray
        Ground truth values
    est_vals : np.ndarray
        Estimated values

    Returns
    -------
    tuple
        (spearman, top_k_precision, scale_ratio, scale_iqr, max_abs_error)
    """
    mask = (true_vals > 0) & np.isfinite(true_vals) & np.isfinite(est_vals)
    if mask.sum() < 10:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    true_masked = true_vals[mask]
    est_masked = est_vals[mask]

    spearman, _ = scipy_stats.spearmanr(true_masked, est_masked)

    k = max(1, int(len(true_masked) * 0.1))
    true_top_k = set(np.argsort(true_masked)[-k:])
    est_top_k = set(np.argsort(est_masked)[-k:])
    top_k_precision = len(true_top_k & est_top_k) / k

    ratios = est_masked / true_masked
    scale_ratio = float(np.median(ratios))
    scale_iqr = float(np.percentile(ratios, 75) - np.percentile(ratios, 25))

    max_abs_error = float(np.max(np.abs(true_masked - est_masked)))

    return spearman, top_k_precision, scale_ratio, scale_iqr, max_abs_error


def compute_quartile_accuracy(
    true_vals: np.ndarray,
    est_vals: np.ndarray,
    node_reach: np.ndarray,
) -> dict:
    """
    Compute accuracy metrics within reachability quartiles.

    Splits nodes into quartiles by their individual reachability and computes
    Spearman correlation and error metrics within each quartile, testing whether
    low-reachability nodes have systematically worse accuracy.

    Parameters
    ----------
    true_vals : np.ndarray
        Ground truth centrality values (per node)
    est_vals : np.ndarray
        Estimated centrality values (per node)
    node_reach : np.ndarray
        Per-node reachability (node density at this distance)

    Returns
    -------
    dict
        Keys per quartile (q1-q4):
          spearman_q{i}    - Spearman rho
          mae_q{i}         - median absolute error
          max_error_q{i}   - max absolute error
          reach_q{i}       - median reach in quartile
    """
    nan_result = {}
    for q in range(1, 5):
        nan_result[f"spearman_q{q}"] = np.nan
        nan_result[f"mae_q{q}"] = np.nan
        nan_result[f"max_error_q{q}"] = np.nan
        nan_result[f"reach_q{q}"] = np.nan

    mask = (true_vals > 0) & np.isfinite(true_vals) & np.isfinite(est_vals) & np.isfinite(node_reach)
    if mask.sum() < 40:
        return nan_result

    true_m = true_vals[mask]
    est_m = est_vals[mask]
    reach_m = node_reach[mask]

    quartile_edges = np.percentile(reach_m, [0, 25, 50, 75, 100])
    result = {}

    for q in range(4):
        lo, hi = quartile_edges[q], quartile_edges[q + 1]
        q_mask = (reach_m >= lo) & (reach_m < hi) if q < 3 else (reach_m >= lo) & (reach_m <= hi)

        if q_mask.sum() < 10:
            result[f"spearman_q{q + 1}"] = np.nan
            result[f"mae_q{q + 1}"] = np.nan
            result[f"max_error_q{q + 1}"] = np.nan
            result[f"reach_q{q + 1}"] = np.nan
        else:
            abs_errors = np.abs(true_m[q_mask] - est_m[q_mask])
            rho, _ = scipy_stats.spearmanr(true_m[q_mask], est_m[q_mask])
            result[f"spearman_q{q + 1}"] = rho
            result[f"mae_q{q + 1}"] = float(np.median(abs_errors))
            result[f"max_error_q{q + 1}"] = float(np.max(abs_errors))
            result[f"reach_q{q + 1}"] = float(np.median(reach_m[q_mask]))

    return result


def mean_quartiles(quartile_list: list[dict], quartile_keys: tuple = QUARTILE_KEYS) -> dict:
    """
    Average quartile accuracy results across multiple runs.

    Parameters
    ----------
    quartile_list : list[dict]
        List of quartile dicts from compute_quartile_accuracy
    quartile_keys : tuple
        Quartile key prefixes (default: QUARTILE_KEYS)

    Returns
    -------
    dict
        Averaged quartile results
    """
    if not quartile_list:
        result = {}
        for prefix in quartile_keys:
            for q in range(1, 5):
                result[f"{prefix}_q{q}"] = np.nan
        return result
    result = {}
    for key in quartile_list[0]:
        vals = [q[key] for q in quartile_list if not np.isnan(q[key])]
        result[key] = float(np.mean(vals)) if vals else np.nan
    return result


# =============================================================================
# SECTION 3: Network Helpers
# =============================================================================


def apply_live_buffer_nx(G: nx.MultiGraph, buffer_dist: float) -> nx.MultiGraph:
    """
    Mark only interior nodes as live on NetworkX graph.

    Applies an inward buffer from the convex hull of all nodes.
    Nodes inside the buffered zone are marked as live=True,
    nodes in the buffer zone are marked as live=False.

    Parameters
    ----------
    G : nx.MultiGraph
        NetworkX graph with 'x' and 'y' node attributes
    buffer_dist : float
        Inward buffer distance in metres

    Returns
    -------
    nx.MultiGraph
        Graph with 'live' attribute set on each node
    """
    coords = [(G.nodes[n]["x"], G.nodes[n]["y"]) for n in G.nodes()]
    all_points = [Point(x, y) for x, y in coords]
    hull = unary_union(all_points).convex_hull
    live_zone = hull.buffer(-buffer_dist)

    for n in G.nodes():
        pt = Point(G.nodes[n]["x"], G.nodes[n]["y"])
        G.nodes[n]["live"] = live_zone.contains(pt)

    n_live = sum(1 for n in G.nodes() if G.nodes[n]["live"])
    print(f"  Live nodes: {n_live}/{G.number_of_nodes()} ({100 * n_live / G.number_of_nodes():.1f}%)")

    return G


# =============================================================================
# SECTION 4: Hoeffding / EW Bound Utilities (analysis-specific)
# =============================================================================


def compute_hoeffding_eff_n(
    reach: float,
    epsilon: float = HOEFFDING_EPSILON,
    delta: float = HOEFFDING_DELTA,
) -> float:
    """
    Compute effective sample size from the Hoeffding/EW bound.

    k = log(2r / delta) / (2 * epsilon^2)

    Parameters
    ----------
    reach : float
        Mean network reach (nodes within distance)
    epsilon : float
        Normalised additive error tolerance
    delta : float
        Failure probability

    Returns
    -------
    float
        Required effective sample size
    """
    if reach <= 0 or epsilon <= 0:
        return reach
    return math.log(2 * reach / delta) / (2 * epsilon**2)


def ew_predicted_epsilon(
    n_eff: float,
    reach: float,
    delta: float = HOEFFDING_DELTA,
) -> float:
    """
    Compute the EW-predicted maximum normalised epsilon.

    eps = sqrt(log(2r / delta) / (2 * n_eff))

    Parameters
    ----------
    n_eff : float
        Effective sample size
    reach : float
        Mean network reach
    delta : float
        Failure probability

    Returns
    -------
    float
        Predicted maximum additive error
    """
    if n_eff <= 0 or reach <= 0:
        return float("inf")
    return math.sqrt(math.log(2 * reach / delta) / (2 * n_eff))
