from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numba import njit
from numba.typed import Dict  # pylint: disable=no-name-in-module

from cityseer import config


@njit(cache=True, fastmath=config.FASTMATH)
def check_numerical_data(data_arr: npt.NDArray[np.float32]):
    """Checks the integrity of numeric data arrays."""
    if not data_arr.ndim == 2:
        raise ValueError(
            "The numeric data array must have a dimensionality 2, "
            "consisting of the number of respective data arrays x the length of data points."
        )
    for num in np.nditer(data_arr):
        if np.isinf(num):
            raise ValueError("The numeric data values must consist of either floats or NaNs.")


@njit(cache=True, fastmath=config.FASTMATH)
def check_categorical_data(data_arr: npt.NDArray[np.float32]):
    """Checks the integrity of categoric data arrays."""
    for cat in data_arr:
        if not np.isfinite(float(cat)) or not cat >= 0:
            raise ValueError("Data map contains points with missing data classes.")
        if int(cat) != cat:
            raise ValueError("Data map contains non-integer class-codes.")


@njit(cache=True, fastmath=config.FASTMATH)
def check_data_map(data_map: npt.NDArray[np.float32], check_assigned=True):
    """
    Checks the integrity of data maps.

    Notes
    -----
    DATA MAP:
    0 - x
    1 - y
    2 - assigned network index - nearest
    3 - assigned network index - next-nearest

    """
    # catch zero length data maps
    if data_map.shape[0] == 0:
        raise ValueError("Zero length data map")
    # other checks - e.g. checking for single dimensional arrays, are tricky with numba
    if not data_map.ndim == 2 or not data_map.shape[1] == 4:
        raise ValueError(
            "The data map must have a dimensionality of Nx4, with the first two indices consisting of x, y coordinates."
            "Indices 2 and 3, if populated, correspond to the nearest and next-nearest network nodes."
        )
    if check_assigned:
        # check that data map has been assigned
        if np.all(np.isnan(data_map[:, 2])):
            raise ValueError(
                "Data map has not been assigned to a network. (Else data-points were not assignable "
                "for the given max_dist parameter passed to assign_to_network."
            )


@njit(cache=True, fastmath=config.FASTMATH)
def check_network_maps(node_data: npt.NDArray[np.float32], edge_data: npt.NDArray[np.float32], node_edge_map: Dict):
    """
    Checks the integrity of network maps.

    NODE MAP:
    0 - x
    1 - y
    2 - live
    EDGE MAP:
    0 - start node
    1 - end node
    2 - length in metres
    3 - sum of angular travel along length
    4 - impedance factor
    5 - entry bearing
    6 - exit bearing

    """
    # catch zero length node or edge maps
    if len(node_data) == 0:
        raise ValueError("Zero length node map")
    if len(edge_data) == 0:
        raise ValueError("Zero length edge map")
    if not node_data.ndim == 2 or not node_data.shape[1] == 3:
        raise ValueError("The node map must have a dimensionality of Nx4.")
    if not edge_data.ndim == 2 or not edge_data.shape[1] == 7:
        raise ValueError("The edge map must have a dimensionality of Nx7")
    # check sequential and reciprocal node to edge map indices
    edge_counts = np.full(len(edge_data), 0)
    for n_idx in range(node_data.shape[0]):
        # zip through all edges for current node
        for edge_idx in node_edge_map[n_idx]:
            # get the edge
            edge = edge_data[edge_idx]
            # check that the start node matches the current node index
            start_nd_idx, end_nd_idx = edge[:2]
            if start_nd_idx != n_idx:
                raise ValueError("Start node does not match current node index")
            # check that each edge has a matching pair in the opposite direction
            paired = False
            for return_edge_idx in node_edge_map[int(end_nd_idx)]:
                if edge_data[return_edge_idx][1] == n_idx:
                    paired = True
                    break
            if not paired:
                raise ValueError("Missing matching edge pair in opposite direction.")
            # add to the counter
            edge_counts[edge_idx] += 1
    if not np.all(edge_counts == 1):
        raise ValueError("Mismatched node and edge maps encountered.")
    if not np.all(np.isfinite(edge_data[:, 0])) or not np.all(edge_data[:, 0] >= 0):
        raise ValueError("Missing or invalid start node index encountered.")
    if not np.all(np.isfinite(edge_data[:, 1])) or not np.all(edge_data[:, 1] >= 0):
        raise ValueError("Missing or invalid end node index encountered.")
    if not np.all(np.isfinite(edge_data[:, 2])) or not np.all(edge_data[:, 2] >= 0):
        raise ValueError("Invalid edge length encountered. Should be finite number greater than or equal to zero.")
    if not np.all(np.isfinite(edge_data[:, 3])) or not np.all(edge_data[:, 3] >= 0):
        raise ValueError("Invalid edge angle sum encountered. Should be finite number greater than or equal to zero.")
    if not np.all(np.isfinite(edge_data[:, 4])) or not np.all(edge_data[:, 4] >= 0):
        raise ValueError("Invalid impedance factor encountered. Should be finite number greater than or equal to zero.")


@njit(cache=True, fastmath=config.FASTMATH)
def check_distances_and_betas(distances: npt.NDArray[np.float32], betas: npt.NDArray[np.float32]):
    """Checks integrity across distances and betas."""
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
