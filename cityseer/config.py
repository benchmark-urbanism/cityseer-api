from __future__ import annotations

import os

import numpy as np

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
# fastmath flags
FASTMATH: set[str] = {"ninf", "nsz", "arcp", "contract", "afn", "reassoc"}
