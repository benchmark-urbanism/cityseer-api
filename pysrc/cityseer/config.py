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

# === SAMPLING MODEL: Hoeffding / Eppstein-Wang Bound ===
# Zero-parameter model based on Hoeffding's inequality adapted from the
# Eppstein-Wang (2004) source-sampling framework for closeness estimation.
#
# Given reach r (nodes within distance threshold), the required effective
# sample size k and sampling probability p are:
#   k = log(2r / δ) / (2ε²)
#   p = min(1, k / r)
#
# Default parameters:
#   ε = 0.1  (normalised additive error tolerance)
#   δ = 0.1  (failure probability → 90% confidence)
#
# At ε = 0.1, this delivers Spearman ρ ≥ 0.98 on real street networks
# with speedups of 5–63× depending on distance threshold.
# Validated on GLA (294k nodes) and Madrid (99k nodes) networks.
HOEFFDING_EPSILON: float = 0.1
HOEFFDING_DELTA: float = 0.1


def compute_hoeffding_p(
    mean_reachability: float,
    epsilon: float = HOEFFDING_EPSILON,
    delta: float = HOEFFDING_DELTA,
) -> float | None:
    """
    Compute sampling probability from the Hoeffding/Eppstein-Wang bound.

    k = log(2r / δ) / (2ε²)
    p = min(1, k / r)

    Parameters
    ----------
    mean_reachability : float
        Average number of nodes reachable within distance threshold.
    epsilon : float
        Normalised additive error tolerance. Default 0.1.
    delta : float
        Failure probability (1 - confidence). Default 0.1.

    Returns
    -------
    float | None
        Required sampling probability in [0, 1], or None if reach is invalid.
    """
    if mean_reachability <= 0:
        return None

    import math

    k = math.log(2 * mean_reachability / delta) / (2 * epsilon**2)
    return min(1.0, k / mean_reachability)


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

    Reports the Hoeffding bound epsilon for each distance threshold,
    showing the theoretical additive error guarantee.
    """
    if sample_probability is None:
        return

    import math

    # Log sampling overview
    logger.info("")  # Visual separator
    speedup = 1 / sample_probability
    logger.info(f"Sampling enabled: p={sample_probability:.0%}, theoretical speedup ~{speedup:.1f}x")

    # If we have actual reachability data, log effective_n and epsilon bound
    if reachability_totals and sampled_source_count and sampled_source_count > 0 and distances:
        logger.info("  Per-distance Hoeffding bound (ε = normalised additive error):")

        for dist, total_reach in zip(distances, reachability_totals, strict=True):
            mean_reach = total_reach / sampled_source_count
            if mean_reach <= 0:
                continue

            effective_n = mean_reach * sample_probability
            # Compute the Hoeffding epsilon achieved at this eff_n
            if effective_n > 0 and mean_reach > 0:
                eps = math.sqrt(math.log(2 * mean_reach / HOEFFDING_DELTA) / (2 * effective_n))
            else:
                eps = float("inf")

            # Compute what the Hoeffding model would recommend
            p_hoeffding = compute_hoeffding_p(mean_reach)
            recommendation = ""
            if p_hoeffding is not None and eps > HOEFFDING_EPSILON:
                if p_hoeffding < 1.0:
                    recommendation = f" → p≥{p_hoeffding:.0%} for ε≤{HOEFFDING_EPSILON}"
                else:
                    recommendation = " (full computation needed for ε≤0.1)"

            logger.info(f"    {dist}m: reach={mean_reach:.0f}, eff_n={effective_n:.0f}, ε={eps:.3f}{recommendation}")


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
        Median reachability (node count) for each distance threshold.
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

    # Return median reachability (50th percentile)
    return {d: float(np.percentile(counts, 50)) if counts else 0.0 for d, counts in reach_counts.items()}


def compute_sample_probs(
    reach_estimates: dict[int, float],
    epsilon: float = HOEFFDING_EPSILON,
    delta: float = HOEFFDING_DELTA,
) -> dict[int, float | None]:
    """
    Compute sampling probability for each distance using the Hoeffding/EW bound.

    Parameters
    ----------
    reach_estimates
        Reachability per distance (from probe_reachability, median).
    epsilon
        Normalised additive error tolerance. Default 0.1.
    delta
        Failure probability. Default 0.1.

    Returns
    -------
    dict[int, float | None]
        Sampling probability for each distance.
        Returns None for distances where reach is zero or negative.
    """
    return {d: compute_hoeffding_p(reach, epsilon, delta) for d, reach in reach_estimates.items()}


def log_adaptive_sampling_plan(
    distances: list[int],
    reach_estimates: dict[int, float],
    sample_probs: dict[int, float | None],
    epsilon: float = HOEFFDING_EPSILON,
    delta: float = HOEFFDING_DELTA,
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
    epsilon
        Normalised additive error tolerance.
    delta
        Failure probability.
    """
    logger.info("")  # Visual separator
    logger.info(f"Adaptive sampling plan (Hoeffding bound: ε={epsilon}, δ={delta}):")
    logger.info("  Distance │  Reach │ Sample p │ Speedup")
    logger.info("  ─────────┼────────┼──────────┼────────")

    for d in sorted(distances):
        reach = reach_estimates.get(d, 0)
        p = sample_probs.get(d)

        # Full computation (p >= 1.0 or None) means exact results
        if p is None or p >= 1.0:
            logger.info(f"  {d:>7}m │ {reach:>6.0f} │     full │    1.0x")
        else:
            speedup = 1.0 / p
            logger.info(f"  {d:>7}m │ {reach:>6.0f} │ {p:>7.0%} │ {speedup:>5.1f}x")


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
    desc: str | None = None,
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
        desc=desc,
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
