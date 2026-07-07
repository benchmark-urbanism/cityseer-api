from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from queue import Queue
from typing import TypeVar

import numpy as np
from tqdm import tqdm

from . import rustalgos

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

np.seterr(invalid="ignore")


def prep_gdf_key(key: str, dist: int, angular: bool = False) -> str:
    """Format a column label for GeoPandas."""
    key = key.replace(".0", "")
    key = key.replace(".0_", "_")
    key = f"cc_{key}_{dist}"
    if angular is True:
        key += "_ang"
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
if QUIET_MODE:
    # quiet mode silences progress bars and routine INFO chatter alike, including
    # chatty third-party loggers (e.g. pyogrio) via the root logger level
    logging.getLogger("cityseer").setLevel(logging.WARNING)
    logging.getLogger().setLevel(logging.WARNING)


def check_debug() -> bool:
    """Check whether to enable debug mode."""
    return "CITYSEER_DEBUG_MODE" in os.environ and os.environ["CITYSEER_DEBUG_MODE"].lower() in [
        "true",
        "1",
    ]


DEBUG_MODE: bool = check_debug()
# for turning off validation
SKIP_VALIDATION: bool = False
# default min threshold weight for beta-distance conversions (matches Rust MIN_THRESH_WT)
MIN_THRESH_WT: float = 0.01831563888873418
SPEED_M_S = 1.33333
# for all_close equality checks
ATOL: float = 0.01
RTOL: float = 0.0001


def resolve_distances(
    distances: list[int] | None = None,
    minutes: list[float] | None = None,
    speed_m_s: float = SPEED_M_S,
) -> tuple[list[int], list[int]]:
    """Resolve distance and time thresholds from distances or minutes.

    Exactly one of ``distances`` or ``minutes`` must be provided.

    Returns
    -------
    tuple[list[int], list[int]]
        (distances, seconds).
    """
    distances, seconds = rustalgos.pair_distances_and_time(speed_m_s, distances, minutes)
    return distances, seconds


def log_thresholds(
    distances: list[int] | None = None,
    minutes: list[float] | None = None,
    speed_m_s: float = SPEED_M_S,
):
    """Resolve and log distance thresholds."""
    distances, seconds = resolve_distances(distances=distances, minutes=minutes, speed_m_s=speed_m_s)
    logger.info("Metrics computed for:")
    for distance, walking_time in zip(distances, seconds, strict=True):
        logger.info(f"Distance: {distance}m, Walking Time: {walking_time / 60} minutes.")
    return distances


# Result type of the wrapped rust call; bound per-call (single result or a list of results).
_RustResult = TypeVar("_RustResult")


def wrap_progress(
    total: int,
    rust_struct: rustalgos.graph.NetworkStructure | rustalgos.data.DataMap | rustalgos.viewshed.Viewshed,
    partial_func: Callable[[], _RustResult],
    desc: str | None = None,
) -> _RustResult:
    """Wraps long running parallelised rust functions with a progress counter."""

    def wrapper(queue: Queue[_RustResult | Exception]):
        try:
            result: _RustResult = partial_func()
            queue.put(result)
        except Exception as e:
            queue.put(e)

    result_queue: Queue[_RustResult | Exception] = Queue()
    thread = threading.Thread(target=wrapper, args=(result_queue,))
    pbar = tqdm(
        total=total,
        disable=QUIET_MODE,
        desc=desc,
        mininterval=0.1,
    )
    thread.start()
    while thread.is_alive():
        time.sleep(0.01)
        pbar.update(rust_struct.progress() - pbar.n)  # type: ignore
    pbar.update(total - pbar.n)
    pbar.close()
    result = result_queue.get()
    thread.join()

    if isinstance(result, Exception):
        raise result

    return result
