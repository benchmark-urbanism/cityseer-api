# pylint: disable=duplicate-code

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numba import njit  # type: ignore

from cityseer import config


@njit(cache=True, fastmath=config.FASTMATH)
def check_numerical_data(data_arr: npt.NDArray[np.float32]) -> None:
    """Check the integrity of numeric data arrays."""
    if not data_arr.ndim == 2:
        raise ValueError(
            "The numeric data array must have a dimensionality 2, "
            "consisting of the number of respective data arrays x the length of data points."
        )
    for num in np.nditer(data_arr):
        if np.isinf(num):
            raise ValueError("The numeric data values must consist of either floats or NaNs.")


@njit(cache=True, fastmath=config.FASTMATH)
def check_categorical_data(data_arr: npt.NDArray[np.int_]) -> None:
    """Check the integrity of categoric data arrays."""
    for cat in data_arr:
        if not np.isfinite(cat) or not cat >= 0:
            raise ValueError("Data map contains points with missing data classes.")
        if int(cat) != cat:
            raise ValueError("Data map contains non-integer class-codes.")


@njit(cache=True, fastmath=config.FASTMATH)
def check_distances_and_betas(distances: npt.NDArray[np.int_], betas: npt.NDArray[np.float32]) -> None:
    """Check integrity across distances and betas."""
    if len(distances) == 0:
        raise ValueError("No distances provided.")
    if len(betas) == 0:
        raise ValueError("No betas provided.")
    if not len(distances) == len(betas):
        raise ValueError("The number of distances and betas should be equal.")
    for i, i_dist in enumerate(distances):
        for j, j_dist in enumerate(distances):
            if i > j:
                if i_dist == j_dist:
                    raise ValueError("Duplicate distances provided. Please provide only one of each.")
    for d in distances:
        if d <= 0:
            raise ValueError("Please provide a positive distance value.")
    for b in betas:
        if b < 0:
            raise ValueError("Please provide the beta value without the leading negative.")
    # set threshold_min from first entry
    threshold_min = np.exp(distances[0] * -betas[0])
    # then check that all entries match the same distance to beta relationship
    for d, b in zip(distances, betas):
        # use a small tolerance for rounding point issues
        if not np.abs(np.exp(-b * d) - threshold_min) < config.ATOL:
            # handles edge case for infinity
            if not np.isinf(d) and b != 0:
                raise ValueError(
                    "Inconsistent threshold minimums, indicating that the relationship between the betas and distances "
                    "is not consistent for all distance / beta pairs."
                )


@njit(cache=True, fastmath=config.FASTMATH, nogil=True)
def clipped_beta_wt(
    beta: np.float32,
    max_curve_wt: np.float32,
    data_dist: np.int_,
) -> np.float32:
    """Calculates negative exponential clipped to the max_curve_wt parameter."""
    raw_wt = np.exp(-beta * data_dist)
    clipped_wt = min(raw_wt, max_curve_wt) / max_curve_wt
    return clipped_wt
