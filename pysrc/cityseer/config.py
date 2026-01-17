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
# Empirical k values from sampling analysis (see analysis/output/README.md)
# RMSE = k × √((1-p) / effective_n) where effective_n = mean_reachability × p
SAMPLING_K_HARMONIC: float = 1.35
SAMPLING_K_BETWEENNESS: float = 2.14


def compute_expected_rmse(
    sample_probability: float,
    mean_reachability: float,
    k: float = SAMPLING_K_BETWEENNESS,
) -> float:
    """
    Compute expected RMSE for given sampling probability and reachability.

    Based on Horvitz-Thompson estimator variance analysis.
    Formula: RMSE = k × √((1-p) / effective_n)

    Parameters
    ----------
    sample_probability : float
        The sampling probability used (0 < p <= 1)
    mean_reachability : float
        Mean number of nodes reachable from sampled sources
    k : float
        Empirical constant from sampling analysis.
        Default uses betweenness k (2.14) as conservative worst-case.
        Harmonic closeness has lower k (1.35).

    Returns
    -------
    float
        Expected RMSE as a fraction (e.g., 0.05 = 5%)
    """
    if sample_probability <= 0 or mean_reachability <= 0:
        return float("inf")
    if sample_probability >= 1.0:
        return 0.0
    effective_n = mean_reachability * sample_probability
    return k * ((1 - sample_probability) / effective_n) ** 0.5


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
    """Log sampling statistics when sampling is enabled."""
    if sample_probability is None:
        return
    # Log sampling info
    logger.info(f"Sampling enabled: probability = {sample_probability}")
    logger.info(f"  Expected speedup: ~{1 / sample_probability:.1f}x")
    # If we have actual reachability data, log post-hoc expected RMSE
    if reachability_totals and sampled_source_count and sampled_source_count > 0 and distances:
        for dist, total_reach in zip(distances, reachability_totals, strict=True):
            mean_reach = total_reach / sampled_source_count
            if mean_reach > 0:
                rmse_harmonic = compute_expected_rmse(sample_probability, mean_reach, SAMPLING_K_HARMONIC)
                rmse_betweenness = compute_expected_rmse(sample_probability, mean_reach, SAMPLING_K_BETWEENNESS)
                logger.info(
                    f"  {dist}m: {sampled_source_count} sources, "
                    f"mean_reach={mean_reach:.0f}, "
                    f"RMSE ~{rmse_harmonic:.0%} (closeness) / ~{rmse_betweenness:.0%} (betweenness)"
                )


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
    partial_func: Callable,  # type: ignore
) -> RustResults:
    """Wraps long running parallelised rust functions with a progress counter."""

    def wrapper(queue: Queue[RustResults | Exception]):
        try:
            result: RustResults = partial_func()  # type: ignore
            queue.put(result)  # type: ignore
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
        time.sleep(1)
        pbar.update(rust_struct.progress() - pbar.n)  # type: ignore
    pbar.update(total - pbar.n)
    pbar.close()
    result = result_queue.get()
    thread.join()

    if isinstance(result, Exception):
        raise result

    return result
