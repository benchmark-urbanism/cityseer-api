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
HOEFFDING_EPSILON_BETWEENNESS: float = 0.05
HOEFFDING_DELTA: float = 0.1


def compute_hoeffding_p(
    mean_reachability: float,
    epsilon: float = HOEFFDING_EPSILON,
    delta: float = HOEFFDING_DELTA,
) -> float:
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
    float
        Required sampling probability in [0, 1]. Returns 1.0 if reach is invalid.
    """
    if (
        not np.isfinite(mean_reachability)
        or not np.isfinite(epsilon)
        or not np.isfinite(delta)
        or mean_reachability <= 0
        or epsilon <= 0
        or delta <= 0
        or delta >= 1
    ):
        return 1.0

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


def min_spatial_samples(
    network_structure: rustalgos.graph.NetworkStructure,
    cell_size: float,
) -> int:
    """
    Return the theoretical minimum number of samples for spatial coverage.

    Computes ``ceil(x_range / cell_size) * ceil(y_range / cell_size)`` from
    the live-node bounding box — one sample per grid cell.

    Parameters
    ----------
    network_structure
        The network to sample from.
    cell_size
        Grid cell side length in metres.

    Returns
    -------
    int
        Minimum sample count for full grid coverage.
    """
    import math

    if not np.isfinite(cell_size) or cell_size <= 0:
        raise ValueError("cell_size must be a finite positive number")

    live_indices = [i for i in network_structure.node_indices() if network_structure.is_node_live(i)]
    if not live_indices:
        return 0
    all_xs = network_structure.node_xs
    all_ys = network_structure.node_ys
    xs = [all_xs[i] for i in live_indices]
    ys = [all_ys[i] for i in live_indices]
    x_range = max(max(xs) - min(xs), 1.0)
    y_range = max(max(ys) - min(ys), 1.0)
    return max(1, math.ceil(x_range / cell_size)) * max(1, math.ceil(y_range / cell_size))


def spatial_sample(
    network_structure: rustalgos.graph.NetworkStructure,
    n_samples: int,
    *,
    cell_size: float = 1000.0,
    random_seed: int | None = None,
) -> tuple[list[int], float]:
    """
    Sample nodes using proportional spatial allocation across grid cells.

    Divides the network into grid cells of ``cell_size`` metres and allocates
    samples to each cell proportionally to its share of live nodes. Within each
    cell, nodes are selected by uniform random draw.

    ``n_samples >= min_spatial_samples(...)`` provides a conservative coverage
    budget, but proportional allocation does not guarantee one sample in every
    occupied cell.

    Note: the downstream IPW correction uses a single scalar probability
    ``actual_p = n_sources / n_live`` rather than per-node inclusion
    probabilities. Because proportional allocation closely tracks the global
    rate, the estimator is approximately (not strictly) unbiased. This is an
    intentional design trade-off for simplicity and empirical performance.

    Parameters
    ----------
    network_structure
        The network to sample from.
    n_samples
        Number of nodes to sample.
    cell_size
        Grid cell side length in metres. Typically ``distance / 2`` so that
        spatial stratification operates at half the analysis distance.
    random_seed
        Optional seed for reproducibility.

    Returns
    -------
    tuple[list[int], float]
        Indices of sampled nodes and network area in km².
    """
    import math
    import random as _random

    if n_samples < 0:
        raise ValueError("n_samples must be non-negative")
    if not np.isfinite(cell_size) or cell_size <= 0:
        raise ValueError("cell_size must be a finite positive number")

    rng = _random.Random(random_seed)

    # Get live nodes
    live_indices = [i for i in network_structure.node_indices() if network_structure.is_node_live(i)]
    n_live = len(live_indices)
    if n_live == 0:
        return [], 0.0

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

    if n_samples == 0:
        return [], area_km2

    if n_live <= n_samples:
        return live_indices, area_km2

    # Build grid and group nodes by cell
    grid_side_x = max(1, int(np.ceil(x_range / cell_size)))
    grid_side_y = max(1, int(np.ceil(y_range / cell_size)))
    cell_x = ((coords[:, 0] - x_min) / x_range * (grid_side_x - 0.001)).astype(int)
    cell_y = ((coords[:, 1] - y_min) / y_range * (grid_side_y - 0.001)).astype(int)
    cell_ids = cell_x * grid_side_y + cell_y

    cells: dict[int, list[int]] = {}
    for idx, cell_id in zip(live_indices, cell_ids, strict=False):
        cells.setdefault(cell_id, []).append(idx)

    cell_items = [(cell_id, nodes) for cell_id, nodes in cells.items() if nodes]

    # Proportional allocation: each cell gets floor(n_samples * cell_count / n_live)
    # Remaining samples distributed by largest fractional remainder
    allocations: list[tuple[int, int, float]] = []
    for cell_id, nodes in cell_items:
        exact = n_samples * len(nodes) / n_live
        floor_alloc = math.floor(exact)
        allocations.append((cell_id, floor_alloc, exact - floor_alloc))

    total_floor = sum(a[1] for a in allocations)
    leftover = n_samples - total_floor
    # Sort by remainder descending, break ties randomly
    rng.shuffle(allocations)
    allocations.sort(key=lambda a: a[2], reverse=True)
    cell_alloc: dict[int, int] = {}
    for i, (cell_id, floor_alloc, _remainder) in enumerate(allocations):
        cell_alloc[cell_id] = floor_alloc + (1 if i < leftover else 0)

    # Sample within each cell
    selected: list[int] = []
    for cell_id, nodes in cell_items:
        k = min(cell_alloc[cell_id], len(nodes))
        if k > 0:
            selected.extend(rng.sample(nodes, k))

    return selected, area_km2


DEFAULT_PROBE_DENSITY: float = 4.0  # Probes per km² for reachability estimation
MIN_PROBES: int = 20  # Minimum probes regardless of area
MAX_PROBES: int = 200  # Maximum probes to limit computation


def probe_reachability(
    network_structure: rustalgos.graph.NetworkStructure,
    distances: list[int],
    probe_density: float = DEFAULT_PROBE_DENSITY,
    speed_m_s: float = SPEED_M_S,
    random_seed: int | None = None,
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
    random_seed
        Optional seed for reproducible probe selection.

    Returns
    -------
    dict[int, float]
        Median reachability (node count) for each distance threshold.
        Using a lower percentile provides conservative estimates that account
        for spatial variation in reachability across the network.
    """
    if not distances:
        return {}

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
    probe_indices, _ = spatial_sample(
        network_structure,
        n_probes,
        cell_size=max(distances) / 2,
        random_seed=random_seed,
    )

    # Accumulate reach counts per distance
    reach_counts: dict[int, list[int]] = {d: [] for d in distances}
    max_seconds = int(max(distances) / speed_m_s)

    for src_idx in probe_indices:
        # Run single Dijkstra to max distance
        visited, tree_map = network_structure.dijkstra_tree_shortest(
            src_idx, max_seconds, speed_m_s
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
) -> dict[int, float]:
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
    dict[int, float]
        Sampling probability for each distance. Returns 1.0 for invalid reach.
    """
    return {d: compute_hoeffding_p(reach, epsilon, delta) for d, reach in reach_estimates.items()}


def log_adaptive_sampling_plan(
    distances: list[int],
    reach_estimates: dict[int, float],
    sample_probs: dict[int, float],
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

        # Full computation (p >= 1.0) means exact results
        if p is None or p >= 1.0:  # None check for backward compat
            logger.info(f"  {d:>7}m │ {reach:>6.0f} │     full │    1.0x")
        else:
            speedup = 1.0 / p
            logger.info(f"  {d:>7}m │ {reach:>6.0f} │ {p:>7.0%} │ {speedup:>5.1f}x")


RustResults = (
    rustalgos.centrality.ClosenessShortestResult
    | rustalgos.centrality.ClosenessSimplestResult
    | rustalgos.centrality.BetweennessShortestResult
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
