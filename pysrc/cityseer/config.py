from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from queue import Queue

import numpy as np
from tqdm import tqdm

from cityseer import rustalgos

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


# for calculating default betas vs. distances
MIN_THRESH_WT: float = 0.01831563888873418
SPEED_M_S = 1.33333
# for all_close equality checks
ATOL: float = 0.001
RTOL: float = 0.0001


def plot_thresholds(
    distances: list[int] | None = None,
    betas: list[float] | None = None,
    walking_times: list[float] | None = None,
    min_threshold_wt: float = MIN_THRESH_WT,
    speed_m_s: float = SPEED_M_S,
):
    # pair distances, betas, and walking times for logging - DO AFTER PARTIAL FUNC
    distances, betas, walking_times = rustalgos.pair_distances_betas_walking_times(
        distances, betas, walking_times, min_threshold_wt=min_threshold_wt, speed_m_s=speed_m_s
    )
    # log distances, betas, walking times
    if not QUIET_MODE:
        logger.info("Metrics computed for:")
        for distance, beta, walking_time in zip(distances, betas, walking_times, strict=True):
            logger.info(f"Distance: {distance}m, Beta: {round(beta, 5)}, Walking Time: {walking_time} minutes.")
    return distances


RustResults = (
    rustalgos.CentralityShortestResult
    | rustalgos.CentralitySimplestResult
    | rustalgos.CentralitySegmentResult
    | rustalgos.AccessibilityResult
    | rustalgos.MixedUsesResult
    | rustalgos.StatsResult
)


def wrap_progress(
    total: int,
    rust_struct: rustalgos.NetworkStructure | rustalgos.DataMap | rustalgos.Viewshed,
    partial_func: Callable,  # type: ignore
) -> RustResults:
    """Wraps long running parallelised rust functions with a progress counter."""

    def wrapper(queue: Queue[RustResults]):
        result: RustResults = partial_func()  # type: ignore
        queue.put(result)  # type: ignore

    result_queue: Queue[RustResults] = Queue()
    thread = threading.Thread(target=wrapper, args=(result_queue,))
    pbar = tqdm(total=total)
    thread.start()
    while thread.is_alive():
        time.sleep(1)
        pbar.update(rust_struct.progress() - pbar.n)
    pbar.update(total - pbar.n)
    pbar.close()
    result = result_queue.get()
    thread.join()
    return result
