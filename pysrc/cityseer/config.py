from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from queue import Queue

import numpy as np
from tqdm import tqdm

from . import rustalgos

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

np.seterr(invalid="ignore")


def prep_gdf_key(key: str, dist: int, angular: bool = False, weighted: bool | None = None) -> str:
    """Format a column label for GeoPandas."""
    key = key.replace(".0", "")
    key = key.replace(".0_", "_")
    key = f"cc_{key}_{dist}"
    if angular is True:
        key += "_ang"
    if weighted is True:
        key += "_wt"
    elif weighted is False:
        key += "_nw"
    return key


def check_quiet() -> bool:
    """Check whether to enable quiet mode."""
    if "GCP_PROJECT" in os.environ:
        return True
    return "CITYSEER_QUIET_MODE" in os.environ and os.environ["CITYSEER_QUIET_MODE"].lower() in [
        "true",
        "1",
    ]


QUIET_MODE = check_quiet()


def check_debug() -> bool:
    """Check whether to enable debug mode."""
    return "CITYSEER_DEBUG_MODE" in os.environ and os.environ["CITYSEER_DEBUG_MODE"].lower() in [
        "true",
        "1",
    ]


DEBUG_MODE: bool = check_debug()
# for turning off validation
SKIP_VALIDATION: bool = False
# for calculating default betas vs. distances
MIN_THRESH_WT: float = 0.01831563888873418
SPEED_M_S = 1.33333
# for all_close equality checks
ATOL: float = 0.01
RTOL: float = 0.0001

# Empirical model parameters for sampling accuracy prediction
# Fitted from analysis across network topologies (trellis, tree, linear)
# effective_n = mean_reachability × sample_probability
# See analysis/output/README.md for methodology
#
# Models fitted to 10th percentile for conservative estimates.
# Betweenness has higher variance than closeness, so separate models are provided.
#
# Model for expected Spearman ρ: rho = 1 - A / (B + eff_n)
# Model for standard deviation: std = C / sqrt(D + eff_n)
# Model for expected scale (magnitude bias): scale = 1 - E / (F + eff_n)

# === SHORTEST (metric) distance models ===
# Generated from analysis/sampling/sampling_reach.py v11 on 2026-01-30
# Harmonic (closeness) model - requires more samples than betweenness
SAMPLING_MODEL_SHORTEST_HARMONIC_A: float = 138.83
SAMPLING_MODEL_SHORTEST_HARMONIC_B: float = 136.62

# Betweenness model - lower variance for shortest path distances
SAMPLING_MODEL_SHORTEST_BETWEENNESS_A: float = 79.73
SAMPLING_MODEL_SHORTEST_BETWEENNESS_B: float = 81.50

# Additional models for std deviation and bias estimation
SAMPLING_MODEL_SHORTEST_STD_C: float = 1.487  # Numerator coefficient for std model
SAMPLING_MODEL_SHORTEST_STD_D: float = 26.28  # Denominator offset for std model
SAMPLING_MODEL_SHORTEST_BIAS_E: float = 0.84  # Numerator coefficient for bias model
SAMPLING_MODEL_SHORTEST_BIAS_F: float = 0.95  # Denominator offset for bias model

# === ANGULAR (simplest) distance models ===
# Generated from analysis/sampling/sampling_reach.py v11 on 2026-01-30
# Angular paths show different accuracy characteristics vs shortest paths
SAMPLING_MODEL_ANGULAR_HARMONIC_A: float = 86.93
SAMPLING_MODEL_ANGULAR_HARMONIC_B: float = 87.37

SAMPLING_MODEL_ANGULAR_BETWEENNESS_A: float = 147.09
SAMPLING_MODEL_ANGULAR_BETWEENNESS_B: float = 155.26

SAMPLING_MODEL_ANGULAR_STD_C: float = 2.269
SAMPLING_MODEL_ANGULAR_STD_D: float = 61.91
SAMPLING_MODEL_ANGULAR_BIAS_E: float = 0.93
SAMPLING_MODEL_ANGULAR_BIAS_F: float = 1.83

# === EXTENDED MODEL PARAMETERS ===
# Extended model: rho = 1 - A / (B + eff_n * p^k)
# Equivalent to: rho = 1 - A / (B + reach * p^(1+k))
# NOTE: k is NOT statistically significant for harmonic closeness (k≈0).
# The baseline model is sufficient for harmonic; extended model helps for betweenness.

# Extended model k parameters (p exponent correction)
# Harmonic k≈0 (not significant), betweenness k>0 (significant at p<0.05)
SAMPLING_MODEL_SHORTEST_HARMONIC_K: float = 0.0  # p=0.92, not significant
SAMPLING_MODEL_SHORTEST_BETWEENNESS_K: float = 0.1813  # p=0.023, significant
SAMPLING_MODEL_ANGULAR_HARMONIC_K: float = 0.0  # p=0.25, not significant
SAMPLING_MODEL_ANGULAR_BETWEENNESS_K: float = 0.2573  # p=0.011, significant

# Extended model A/B parameters (different from baseline)
SAMPLING_MODEL_EXT_SHORTEST_HARMONIC_A: float = 43.90
SAMPLING_MODEL_EXT_SHORTEST_HARMONIC_B: float = 61.57
SAMPLING_MODEL_EXT_SHORTEST_BETWEENNESS_A: float = 22.87
SAMPLING_MODEL_EXT_SHORTEST_BETWEENNESS_B: float = 30.69
SAMPLING_MODEL_EXT_ANGULAR_HARMONIC_A: float = 23.13
SAMPLING_MODEL_EXT_ANGULAR_HARMONIC_B: float = 32.64
SAMPLING_MODEL_EXT_ANGULAR_BETWEENNESS_A: float = 30.19
SAMPLING_MODEL_EXT_ANGULAR_BETWEENNESS_B: float = 42.15


def get_expected_spearman(
    effective_n: float,
    metric: str = "betweenness",
    distance_type: str = "shortest",
) -> tuple[float, float]:
    """
    Get expected Spearman ρ and standard deviation for given effective sample size.

    Uses fitted empirical models:
    - Expected ρ = 1 - A / (B + eff_n)
    - Std dev = C / sqrt(D + eff_n)

    Parameters
    ----------
    effective_n : float
        The effective sample size (reach × p)
    metric : str
        Which metric model to use: "harmonic", "betweenness", or "both".
        Use "both" when computing both metrics together (uses betweenness/conservative).
        Default "betweenness" for backward compatibility.
    distance_type : str
        Which distance type model to use: "shortest" (metric) or "angular" (simplest).
        Default "shortest" for backward compatibility.

    Returns
    -------
    tuple[float, float]
        (expected_spearman, std_dev)
    """
    # Ensure minimum effective_n of 1 to avoid edge cases
    eff_n = max(1.0, effective_n)

    # Select model parameters based on distance_type and metric
    if distance_type == "angular":
        if metric == "harmonic":
            a, b = SAMPLING_MODEL_ANGULAR_HARMONIC_A, SAMPLING_MODEL_ANGULAR_HARMONIC_B
        else:
            # "betweenness" or "both" - use conservative model
            a, b = SAMPLING_MODEL_ANGULAR_BETWEENNESS_A, SAMPLING_MODEL_ANGULAR_BETWEENNESS_B
        std_c, std_d = SAMPLING_MODEL_ANGULAR_STD_C, SAMPLING_MODEL_ANGULAR_STD_D
    else:  # shortest (default)
        if metric == "harmonic":
            a, b = SAMPLING_MODEL_SHORTEST_HARMONIC_A, SAMPLING_MODEL_SHORTEST_HARMONIC_B
        else:
            # "betweenness" or "both" - use conservative model
            a, b = SAMPLING_MODEL_SHORTEST_BETWEENNESS_A, SAMPLING_MODEL_SHORTEST_BETWEENNESS_B
        std_c, std_d = SAMPLING_MODEL_SHORTEST_STD_C, SAMPLING_MODEL_SHORTEST_STD_D

    # Expected Spearman ρ
    expected_rho = 1.0 - a / (b + eff_n)
    expected_rho = max(0.0, min(1.0, expected_rho))  # Clamp to [0, 1]

    # Standard deviation
    std_dev = std_c / (std_d + eff_n) ** 0.5
    std_dev = max(0.001, std_dev)  # Minimum std

    return expected_rho, std_dev


def get_expected_bias(effective_n: float, distance_type: str = "shortest") -> float:
    """
    Get expected magnitude bias for given effective sample size.

    Uses fitted empirical model: scale = 1 - E / (F + eff_n)
    Bias is reported as (1 - scale), i.e., the fraction by which
    magnitudes are expected to be underestimated.

    Parameters
    ----------
    effective_n : float
        The effective sample size (reach × p)
    distance_type : str
        Which distance type model to use: "shortest" (metric) or "angular" (simplest).
        Default "shortest" for backward compatibility.

    Returns
    -------
    float
        Expected bias as a fraction (e.g., 0.05 means ~5% underestimate)
    """
    eff_n = max(1.0, effective_n)

    # Select model parameters based on distance_type
    if distance_type == "angular":
        bias_e, bias_f = SAMPLING_MODEL_ANGULAR_BIAS_E, SAMPLING_MODEL_ANGULAR_BIAS_F
    else:  # shortest (default)
        bias_e, bias_f = SAMPLING_MODEL_SHORTEST_BIAS_E, SAMPLING_MODEL_SHORTEST_BIAS_F

    expected_scale = 1.0 - bias_e / (bias_f + eff_n)
    expected_scale = max(0.0, min(1.0, expected_scale))  # Clamp to [0, 1]
    return 1.0 - expected_scale


def get_extended_model_params(
    metric: str = "betweenness",
    distance_type: str = "shortest",
) -> tuple[float, float, float]:
    """
    Get extended model parameters (A, B, k) for the given metric and distance type.

    The extended model is: rho = 1 - A / (B + eff_n * p^k)
    Which is equivalent to: rho = 1 - A / (B + reach * p^(1+k))

    Parameters
    ----------
    metric : str
        Which metric model to use: "harmonic", "betweenness", or "both".
        Use "both" when computing both metrics together (uses betweenness/conservative).
        Default "betweenness" for backward compatibility.
    distance_type : str
        Which distance type model to use: "shortest" (metric) or "angular" (simplest).
        Default "shortest" for backward compatibility.

    Returns
    -------
    tuple[float, float, float]
        (A, B, k) parameters for the extended model.
    """
    if distance_type == "angular":
        if metric == "harmonic":
            return (
                SAMPLING_MODEL_EXT_ANGULAR_HARMONIC_A,
                SAMPLING_MODEL_EXT_ANGULAR_HARMONIC_B,
                SAMPLING_MODEL_ANGULAR_HARMONIC_K,
            )
        else:
            # "betweenness" or "both" - use conservative model
            return (
                SAMPLING_MODEL_EXT_ANGULAR_BETWEENNESS_A,
                SAMPLING_MODEL_EXT_ANGULAR_BETWEENNESS_B,
                SAMPLING_MODEL_ANGULAR_BETWEENNESS_K,
            )
    else:  # shortest (default)
        if metric == "harmonic":
            return (
                SAMPLING_MODEL_EXT_SHORTEST_HARMONIC_A,
                SAMPLING_MODEL_EXT_SHORTEST_HARMONIC_B,
                SAMPLING_MODEL_SHORTEST_HARMONIC_K,
            )
        else:
            # "betweenness" or "both" - use conservative model
            return (
                SAMPLING_MODEL_EXT_SHORTEST_BETWEENNESS_A,
                SAMPLING_MODEL_EXT_SHORTEST_BETWEENNESS_B,
                SAMPLING_MODEL_SHORTEST_BETWEENNESS_K,
            )


def get_required_effective_n(
    target_spearman: float,
    metric: str = "betweenness",
    distance_type: str = "shortest",
) -> float | None:
    """
    Get the minimum effective_n required to achieve a target Spearman ρ.

    Inverts the model: eff_n = A / (1 - target) - B

    Parameters
    ----------
    target_spearman : float
        Target Spearman ρ (e.g., 0.95)
    metric : str
        Which metric model to use: "harmonic", "betweenness", or "both".
        Use "both" when computing both metrics together (uses betweenness/conservative).
        Default "betweenness" for backward compatibility.
    distance_type : str
        Which distance type model to use: "shortest" (metric) or "angular" (simplest).
        Default "shortest" for backward compatibility.

    Returns
    -------
    float | None
        Required effective_n, or None if target is impossible (≥1.0)
    """
    if target_spearman >= 1.0:
        return None
    if target_spearman <= 0.0:
        return 0.0

    # Select model parameters based on distance_type and metric
    if distance_type == "angular":
        if metric == "harmonic":
            a, b = SAMPLING_MODEL_ANGULAR_HARMONIC_A, SAMPLING_MODEL_ANGULAR_HARMONIC_B
        else:
            # "betweenness" or "both" - use conservative model
            a, b = SAMPLING_MODEL_ANGULAR_BETWEENNESS_A, SAMPLING_MODEL_ANGULAR_BETWEENNESS_B
    else:  # shortest (default)
        if metric == "harmonic":
            a, b = SAMPLING_MODEL_SHORTEST_HARMONIC_A, SAMPLING_MODEL_SHORTEST_HARMONIC_B
        else:
            # "betweenness" or "both" - use conservative model
            a, b = SAMPLING_MODEL_SHORTEST_BETWEENNESS_A, SAMPLING_MODEL_SHORTEST_BETWEENNESS_B

    # Invert: rho = 1 - A / (B + n) => n = A / (1 - rho) - B
    required_n = a / (1.0 - target_spearman) - b
    return max(1.0, required_n)


def compute_required_p(
    mean_reachability: float,
    target_spearman: float = 0.95,
    metric: str = "betweenness",
    distance_type: str = "shortest",
) -> float | None:
    """
    Compute the sampling probability required to achieve target accuracy.

    When k is significant (betweenness), uses extended model:
        rho = 1 - A / (B + reach * p^(1+k))
        Inverted: p = ((A / (1 - rho) - B) / reach)^(1/(1+k))

    When k is NOT significant (harmonic, k≈0), uses baseline model
    for conservative estimates:
        rho = 1 - A / (B + eff_n)
        Inverted: p = (A / (1 - rho) - B) / reach

    Parameters
    ----------
    mean_reachability : float
        Average number of nodes reachable within distance threshold
    target_spearman : float
        Target Spearman ρ (default 0.95)
    metric : str
        Which metric model to use: "harmonic", "betweenness", or "both".
        Use "both" when computing both metrics together (uses betweenness/conservative).
        Default "betweenness" for backward compatibility.
    distance_type : str
        Which distance type model to use: "shortest" (metric) or "angular" (simplest).
        Default "shortest" for backward compatibility.

    Returns
    -------
    float | None
        Required sampling probability, or None if impossible
    """
    if target_spearman >= 1.0:
        return None
    if mean_reachability <= 0:
        return None

    # Get extended model parameters
    a_ext, b_ext, k = get_extended_model_params(metric, distance_type)

    # When k is NOT significant (≈0), use baseline model for conservative estimates
    # The extended model with k≈0 was fitted to all data (median), not 10th percentile
    if abs(k) < 0.05:  # k not significant
        # Use baseline (conservative 10th percentile) model parameters
        if distance_type == "angular":
            if metric == "harmonic":
                a, b = SAMPLING_MODEL_ANGULAR_HARMONIC_A, SAMPLING_MODEL_ANGULAR_HARMONIC_B
            else:
                a, b = SAMPLING_MODEL_ANGULAR_BETWEENNESS_A, SAMPLING_MODEL_ANGULAR_BETWEENNESS_B
        else:  # shortest
            if metric == "harmonic":
                a, b = SAMPLING_MODEL_SHORTEST_HARMONIC_A, SAMPLING_MODEL_SHORTEST_HARMONIC_B
            else:
                a, b = SAMPLING_MODEL_SHORTEST_BETWEENNESS_A, SAMPLING_MODEL_SHORTEST_BETWEENNESS_B

        # Baseline model inversion: p = (A / (1 - rho) - B) / reach
        numerator = a / (1.0 - target_spearman) - b
        if numerator <= 0:
            return None
        p = numerator / mean_reachability
    else:
        # Extended model with significant k
        a, b = a_ext, b_ext
        numerator = a / (1.0 - target_spearman) - b
        if numerator <= 0:
            return None
        # Extended model inversion: p = (numerator / reach)^(1/(1+k))
        p = (numerator / mean_reachability) ** (1.0 / (1.0 + k))

    return min(1.0, max(0.0, p))


def log_thresholds(
    distances: list[int] | None = None,
    betas: list[float] | None = None,
    minutes: list[float] | None = None,
    min_threshold_wt: float = MIN_THRESH_WT,
    speed_m_s: float = SPEED_M_S,
):
    # pair distances, betas, and time for logging - DO AFTER PARTIAL FUNC
    distances, betas, seconds = rustalgos.pair_distances_betas_time(
        speed_m_s, distances, betas, minutes, min_threshold_wt=min_threshold_wt
    )
    # log distances, betas, minutes
    logger.info("Metrics computed for:")
    for distance, beta, walking_time in zip(distances, betas, seconds, strict=True):
        logger.info(f"Distance: {distance}m, Beta: {round(beta, 5)}, Walking Time: {walking_time / 60} minutes.")
    return distances


def log_sampling(
    sample_probability: float | None = None,
    distances: list[int] | None = None,
    reachability_totals: list[int] | None = None,
    sampled_source_count: int | None = None,
) -> None:
    """
    Log sampling statistics when sampling is enabled.

    Provides users with:
    - Expected ranking accuracy (Spearman ρ) based on effective sample size
    - Recommendations for achieving target accuracy levels where applicable
    - Warnings when effective sample size is very low

    The effective sample size (eff_n = reachability × p) determines accuracy:
    - eff_n ≥ 260: Expected ρ ≥ 0.95
    - eff_n ≥ 120: Expected ρ ≥ 0.90
    - eff_n < 50:  High variance expected
    """
    if sample_probability is None:
        return

    # Log sampling overview
    logger.info("")  # Visual separator
    speedup = 1 / sample_probability
    logger.info(f"Sampling enabled: p={sample_probability:.0%}, theoretical speedup ~{speedup:.1f}x")
    logger.info("  Spearman ρ measures ranking preservation (0=uncorrelated, 1=identical)")
    logger.info("  Bias shows expected magnitude underestimate (0%=unbiased)")

    # If we have actual reachability data, log effective_n and expected accuracy
    if reachability_totals and sampled_source_count and sampled_source_count > 0 and distances:
        logger.info("  Per-distance accuracy estimates:")

        for dist, total_reach in zip(distances, reachability_totals, strict=True):
            mean_reach = total_reach / sampled_source_count
            if mean_reach <= 0:
                continue

            effective_n = mean_reach * sample_probability
            expected_rho, std_rho = get_expected_spearman(effective_n)

            # Compute required p for different accuracy targets
            p_for_90 = compute_required_p(mean_reach, 0.90)
            p_for_95 = compute_required_p(mean_reach, 0.95)

            # Build recommendation based on achievable targets
            # Check if targets are achievable (p < 1.0 means sampling still provides speedup)
            can_reach_95 = p_for_95 is not None and p_for_95 < 1.0
            can_reach_90 = p_for_90 is not None and p_for_90 < 1.0

            if expected_rho >= 0.95:
                # Already at ρ≥0.95
                recommendation = ""
            elif can_reach_95:
                # Can achieve ρ≥0.95 with higher p
                recommendation = f" → p≥{p_for_95:.0%} for ρ≥0.95"
            elif can_reach_90:
                # Can achieve ρ≥0.90 with higher p, but ρ≥0.95 requires full computation
                recommendation = f" → p≥{p_for_90:.0%} for ρ≥0.90"
            else:
                # Neither target achievable with sampling - reach is too low
                recommendation = " (reach too low for ρ≥0.90 with sampling)"

            # Get expected bias
            expected_bias = get_expected_bias(effective_n)
            bias_str = f", bias={expected_bias:.0%}" if expected_bias >= 0.01 else ""

            # Log the main info line
            logger.info(
                f"    {dist}m: reach={mean_reach:.0f}, eff_n={effective_n:.0f}, "
                f"expected ρ={expected_rho:.2f}±{std_rho:.2f}{bias_str}{recommendation}"
            )


# =============================================================================
# Adaptive Sampling Helpers
# =============================================================================
# These functions support per-distance adaptive sampling, where sampling
# probability is calibrated separately for each distance threshold based on
# reachability. This allows aggressive sampling at large distances (high reach)
# while maintaining accuracy at short distances (low reach).


def spatial_sample(
    network_structure: rustalgos.graph.NetworkStructure,
    n_samples: int,
) -> tuple[list[int], float]:
    """
    Sample nodes with spatial distribution using grid stratification.

    Divides the network bounding box into 1km² grid cells and samples round-robin
    from each cell, ensuring spatial coverage across the network.

    Parameters
    ----------
    network_structure
        The network to sample from.
    n_samples
        Number of nodes to sample.

    Returns
    -------
    tuple[list[int], float]
        Indices of sampled nodes (spatially distributed) and network area in km².
    """
    import random

    # Get live nodes
    live_indices = [i for i in network_structure.node_indices() if network_structure.is_node_live(i)]

    # Get coordinates for live nodes
    all_xs = network_structure.node_xs
    all_ys = network_structure.node_ys
    coords = np.array([(all_xs[i], all_ys[i]) for i in live_indices])

    # Compute bounding box and area
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    x_range = max(x_max - x_min, 1.0)  # metres
    y_range = max(y_max - y_min, 1.0)  # metres
    area_km2 = (x_range * y_range) / 1_000_000  # convert m² to km²

    if len(live_indices) <= n_samples:
        return live_indices, area_km2

    # Grid with ~1km cells (1000m)
    grid_side_x = max(1, int(np.ceil(x_range / 1000)))
    grid_side_y = max(1, int(np.ceil(y_range / 1000)))

    # Compute grid cell for each node
    cell_x = ((coords[:, 0] - x_min) / x_range * (grid_side_x - 0.001)).astype(int)
    cell_y = ((coords[:, 1] - y_min) / y_range * (grid_side_y - 0.001)).astype(int)
    cell_ids = cell_x * grid_side_y + cell_y

    # Group nodes by cell
    cells: dict[int, list[int]] = {}
    for idx, cell_id in zip(live_indices, cell_ids, strict=False):
        cells.setdefault(cell_id, []).append(idx)

    # Sample round-robin from cells
    selected = []
    cell_lists = list(cells.values())
    random.shuffle(cell_lists)

    while len(selected) < n_samples:
        for cell_nodes in cell_lists:
            if cell_nodes and len(selected) < n_samples:
                idx = random.randrange(len(cell_nodes))
                selected.append(cell_nodes.pop(idx))

    return selected, area_km2


DEFAULT_PROBE_DENSITY: float = 4.0  # Probes per km² for reachability estimation
MIN_PROBES: int = 20  # Minimum probes regardless of area
MAX_PROBES: int = 200  # Maximum probes to limit computation


def probe_reachability(
    network_structure: rustalgos.graph.NetworkStructure,
    distances: list[int],
    probe_density: float = DEFAULT_PROBE_DENSITY,
    speed_m_s: float = SPEED_M_S,
) -> dict[int, float]:
    """
    Estimate reachability per distance by probing spatially distributed nodes.

    This is a lightweight pre-computation step that runs Dijkstra from a sample
    of nodes to estimate how many nodes are typically reachable at each distance
    threshold. The estimates are used to compute appropriate sampling probabilities
    for adaptive sampling.

    Uses spatial stratification with ~1km² grid cells to ensure probes are
    distributed across the network rather than clustered in one area.

    Parameters
    ----------
    network_structure
        The network to probe.
    distances
        Distance thresholds in metres.
    probe_density
        Number of probes per km² of network area. Default 4.0.
        The actual probe count is bounded by MIN_PROBES (20) and MAX_PROBES (200).
    speed_m_s
        Walking speed for converting distance to seconds.

    Returns
    -------
    dict[int, float]
        25th percentile reachability (node count) for each distance threshold.
        Using a lower percentile provides conservative estimates that account
        for spatial variation in reachability across the network.
    """
    # Get live nodes
    live_indices = [i for i in network_structure.node_indices() if network_structure.is_node_live(i)]

    if not live_indices:
        return {d: 0.0 for d in distances}

    # Compute n_probes from network area and density
    all_xs = network_structure.node_xs
    all_ys = network_structure.node_ys
    xs = [all_xs[i] for i in live_indices]
    ys = [all_ys[i] for i in live_indices]
    x_range = max(xs) - min(xs)
    y_range = max(ys) - min(ys)
    area_km2 = max(x_range, 1.0) * max(y_range, 1.0) / 1_000_000
    n_probes = int(np.clip(area_km2 * probe_density, MIN_PROBES, MAX_PROBES))

    n_probes = min(n_probes, len(live_indices))
    probe_indices, _ = spatial_sample(network_structure, n_probes)

    # Accumulate reach counts per distance
    reach_counts: dict[int, list[int]] = {d: [] for d in distances}
    max_seconds = int(max(distances) / speed_m_s)

    for src_idx in probe_indices:
        # Run single Dijkstra to max distance
        visited, tree_map = network_structure.dijkstra_tree_shortest(
            src_idx, max_seconds, speed_m_s, jitter_scale=None, random_seed=None
        )

        # Count nodes reachable at each distance threshold
        for d in distances:
            count = sum(1 for v_idx in visited if tree_map[v_idx].short_dist <= d and v_idx != src_idx)
            reach_counts[d].append(count)

    # Return 25th percentile reachability (conservative estimate for spatial variation)
    return {d: float(np.percentile(counts, 25)) if counts else 0.0 for d, counts in reach_counts.items()}


def compute_sample_probs_for_target_rho(
    reach_estimates: dict[int, float],
    target_rho: float = 0.95,
    metric: str = "both",
    distance_type: str = "shortest",
) -> dict[int, float | None]:
    """
    Compute the sampling probability required at each distance to achieve target accuracy.

    Uses extended model: rho = 1 - A / (B + reach * p^(1+k))
    Inverted: p = ((A / (1 - rho) - B) / reach)^(1/(1+k))

    Conservative estimates are achieved through:
    1. Reachability probing uses 25th percentile (handles spatial variation)
    2. Accuracy model is fitted to 10th percentile of observations

    Parameters
    ----------
    reach_estimates
        Reachability per distance (from probe_reachability, 25th percentile).
    target_rho
        Target Spearman ρ correlation. Default 0.95.
    metric
        Which metric model to use: "harmonic", "betweenness", or "both".
        - "harmonic": Use closeness model (less conservative, more speedup)
        - "betweenness": Use betweenness model (more conservative)
        - "both": Use betweenness model to ensure both metrics meet target
    distance_type
        Which distance type model to use: "shortest" (metric) or "angular" (simplest).
        Default "shortest" for backward compatibility.

    Returns
    -------
    dict[int, float | None]
        Sampling probability for each distance.
        Returns None for distances where reach is too low to achieve target with any p ≤ 1.0.
    """
    result = {}
    for d, reach in reach_estimates.items():
        p = compute_required_p(reach, target_rho, metric=metric, distance_type=distance_type)
        result[d] = p

    return result


def log_adaptive_sampling_plan(
    distances: list[int],
    reach_estimates: dict[int, float],
    sample_probs: dict[int, float | None],
    target_rho: float,
    metric: str = "both",
    safety_margin: float = 0.02,
    distance_type: str = "shortest",
) -> None:
    """
    Log the adaptive sampling plan before execution.

    Parameters
    ----------
    distances
        Distance thresholds.
    reach_estimates
        Estimated reachability per distance.
    sample_probs
        Computed sampling probabilities per distance.
    target_rho
        Target accuracy level.
    metric
        Which metric model is being used: "harmonic", "betweenness", or "both".
    safety_margin
        Safety margin applied to target_rho internally.
    distance_type
        Which distance type model to use: "shortest" (metric) or "angular" (simplest).
    """
    logger.info("")  # Visual separator
    metric_label = {"harmonic": "closeness", "betweenness": "betweenness", "both": "both metrics"}.get(
        metric, "both metrics"
    )
    dist_label = "angular" if distance_type == "angular" else "shortest"
    effective_target = min(0.99, target_rho + safety_margin)
    logger.info(
        f"Adaptive sampling plan ({dist_label}, target ρ ≥ {target_rho:.2f}, "
        f"internal target {effective_target:.2f} for {metric_label}):"
    )
    logger.info("  Distance │  Reach │ Sample p │ Expected ρ")
    logger.info("  ─────────┼────────┼──────────┼───────────")

    for d in sorted(distances):
        reach = reach_estimates.get(d, 0)
        p = sample_probs.get(d)

        # Full computation (p >= 1.0 or None) means exact results
        if p is None or p >= 1.0:
            logger.info(f"  {d:>7}m │ {reach:>6.0f} │     full │ 1.00 (exact)")
        else:
            eff_n = reach * p
            exp_rho, _ = get_expected_spearman(eff_n, metric=metric, distance_type=distance_type)
            logger.info(f"  {d:>7}m │ {reach:>6.0f} │ {p:>7.0%} │ {exp_rho:.2f} (eff_n={eff_n:.0f})")


RustResults = (
    rustalgos.centrality.CentralityShortestResult
    | rustalgos.centrality.CentralitySimplestResult
    | rustalgos.centrality.CentralitySegmentResult
    | rustalgos.data.AccessibilityResult
    | rustalgos.data.MixedUsesResult
    | rustalgos.data.StatsResult
)


def wrap_progress(
    total: int,
    rust_struct: rustalgos.graph.NetworkStructure | rustalgos.data.DataMap | rustalgos.viewshed.Viewshed,
    partial_func: Callable,
) -> RustResults:
    """Wraps long running parallelised rust functions with a progress counter."""

    def wrapper(queue: Queue[RustResults | Exception]):
        try:
            result: RustResults = partial_func()
            queue.put(result)
        except Exception as e:
            queue.put(e)

    result_queue: Queue[RustResults | Exception] = Queue()
    thread = threading.Thread(target=wrapper, args=(result_queue,))
    pbar = tqdm(
        total=total,
        disable=QUIET_MODE,
    )
    thread.start()
    while thread.is_alive():
        time.sleep(0.1)
        pbar.update(rust_struct.progress() - pbar.n)  # type: ignore
    pbar.update(total - pbar.n)
    pbar.close()
    result = result_queue.get()
    thread.join()

    if isinstance(result, Exception):
        raise result

    return result
