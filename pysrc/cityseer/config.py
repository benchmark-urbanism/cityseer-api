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

# === SAMPLING MODEL: Distance-based Hoeffding / Eppstein-Wang Bound ===
# Sampling probability derived from distance alone using a canonical grid network model.
# Reachability is estimated as r = π * d² / s² for grid spacing s (metres).
# The Hoeffding bound then gives:
#   k = log(2r / δ) / (2ε²)
#   p = min(1, k / r)
#
# Using a fixed grid spacing produces deterministic p values for any distance,
# enabling reach-agnostic comparison across networks.
#
# Default parameters:
#   ε = 0.05  (normalised additive error tolerance; unified for closeness and betweenness)
#   δ = 0.1   (failure probability → 90% confidence)
#   s = 175m  (canonical sparse street network inter-node spacing)
HOEFFDING_EPSILON: float = 0.05
HOEFFDING_DELTA: float = 0.1
GRID_SPACING: float = 175.0  # metres — canonical sparse street network inter-node spacing


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
        Normalised additive error tolerance. Default 0.05 (via HOEFFDING_EPSILON).
    delta : float
        Failure probability (1 - confidence). Default 0.1 (via HOEFFDING_DELTA).

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


def compute_distance_p(
    distance: float,
    epsilon: float = HOEFFDING_EPSILON,
    delta: float = HOEFFDING_DELTA,
    grid_spacing: float = GRID_SPACING,
) -> float:
    """
    Compute sampling probability from distance using a canonical grid network model.

    Estimates reachability as r = π * d² / s² for grid spacing s, then applies
    the Hoeffding/Eppstein-Wang bound. This produces deterministic p values for
    any distance, independent of the actual network, enabling cross-network comparison.

    Parameters
    ----------
    distance : float
        Distance threshold in metres.
    epsilon : float
        Normalised additive error tolerance. Default 0.05 (unified for closeness and betweenness).
    delta : float
        Failure probability (1 - confidence). Default 0.1.
    grid_spacing : float
        Canonical inter-node spacing in metres. Default 175m (sparse street network).

    Returns
    -------
    float
        Required sampling probability in [0, 1].
    """
    import math

    if distance <= 0 or grid_spacing <= 0:
        return 1.0
    r = math.pi * distance**2 / grid_spacing**2
    return compute_hoeffding_p(r, epsilon=epsilon, delta=delta)


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


RustResults = (
    rustalgos.centrality.CentralityShortestResult
    | rustalgos.centrality.CentralitySimplestResult
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
