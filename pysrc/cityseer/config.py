from __future__ import annotations

import os
from queue import Queue
import threading
import time
from typing import Callable
import numpy as np
from tqdm import tqdm

from cityseer import rustalgos

np.seterr(invalid="ignore")


def prep_gdf_key(key: str) -> str:
    """Format a column label for GeoPandas."""
    key = key.replace(".0", "")
    key = key.replace(".0_", "_")
    return f"cc_metric_{key}"


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


def wrap_progress(
    total: int, rust_struct: rustalgos.NetworkStructure | rustalgos.DataMap, partial_func: Callable
) -> (
    rustalgos.CentralityShortestResult
    | rustalgos.CentralitySimplestResult
    | rustalgos.CentralitySegmentResult
    | rustalgos.AccessibilityResult
    | rustalgos.MixedUsesResult
    | rustalgos.StatsResult
):
    """Wraps long running parallelised rust functions with a progress counter."""

    def wrapper(queue):
        result = partial_func()
        queue.put(result)

    result_queue = Queue()
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
