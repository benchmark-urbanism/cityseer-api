from __future__ import annotations

import os
import threading
import time
from queue import Queue
from typing import Callable, Union

import numpy as np
from tqdm import tqdm

from cityseer import rustalgos

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
    if "CITYSEER_QUIET_MODE" in os.environ:
        if os.environ["CITYSEER_QUIET_MODE"].lower() in ["true", "1"]:
            return True
    return False


QUIET_MODE = check_quiet()


def check_debug() -> bool:
    """Check whether to enable debug mode."""
    if "CITYSEER_DEBUG_MODE" in os.environ:
        if os.environ["CITYSEER_DEBUG_MODE"].lower() in ["true", "1"]:
            return True
    return False


DEBUG_MODE: bool = check_debug()


# for calculating default betas vs. distances
MIN_THRESH_WT: float = 0.01831563888873418
# for all_close equality checks
ATOL: float = 0.001
RTOL: float = 0.0001


RustResults = Union[
    rustalgos.CentralityShortestResult,
    rustalgos.CentralitySimplestResult,
    rustalgos.CentralitySegmentResult,
    rustalgos.AccessibilityResult,
    rustalgos.MixedUsesResult,
    rustalgos.StatsResult,
]


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
