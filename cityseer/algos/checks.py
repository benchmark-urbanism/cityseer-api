import os
import numpy as np
from numba import njit, typeof
from numba.typed import Dict

def_min_thresh_wt = 0.01831563888873418

quiet_mode = False
if 'GCP_PROJECT' in os.environ:
    quiet_mode = True

if 'CITYSEER_QUIET_MODE' in os.environ:
    if os.environ['CITYSEER_QUIET_MODE'].lower() in ['true', '1']:
        quiet_mode = True


@njit(cache=True)
def _print_msg(hash_count, void_count, percentage):
    msg = '|'
    for n in range(int(hash_count)):
        msg += '#'
    for n in range(int(void_count)):
        msg += ' '
    msg += '|'
    print(msg, percentage, '%')


@njit(cache=False)
def progress_bar(current: int, total: int, steps: int = 20):
    '''
    Printing carries a performance penalty
    Cache has to be set to false per Numba issue:
    https://github.com/numba/numba/issues/3555
    TODO: set cache to True once resolved - likely 2020
    '''
    if steps == 0:
        return
    if current + 1 == total:
        _print_msg(steps, 0, 100)
        return
    if total <= steps:
        step_size = 1
    else:
        step_size = int(total / steps)
    if current % step_size == 0:
        percentage = np.round(current / total * 100, 2)
        hash_count = int(percentage / 100 * steps)
        void_count = steps - hash_count
        _print_msg(hash_count, void_count, percentage)


@njit(cache=True)
def check_numerical_data(data_arr: np.ndarray):
    if not data_arr.ndim == 2:
        raise ValueError('The numeric data array must have a dimensionality 2, '
                         'consisting of the number of respective data arrays x the length of data points.')
    for num in np.nditer(data_arr):
        if np.isinf(num):
            raise ValueError('The numeric data values must consist of either floats or NaNs.')


@njit(cache=True)
def check_categorical_data(data_arr: np.ndarray):
    for cl in data_arr:
        if not np.isfinite(np.float(cl)) or not cl >= 0:
            raise ValueError('Data map contains points with missing data classes.')
        if int(cl) != cl:
            raise ValueError('Data map contains non-integer class-codes.')


@njit(cache=True)
def check_data_map(data_map: np.ndarray, check_assigned=True):
    '''
    DATA MAP:
    0 - x
    1 - y
    2 - assigned network index - nearest
    3 - assigned network index - next-nearest
    '''
    # catch zero length data maps
    if len(data_map) == 0:
        raise ValueError('Zero length data map')

    # other checks - e.g. checking for single dimensional arrays, are tricky with numba
    if not data_map.ndim == 2 or not data_map.shape[1] == 4:
        raise ValueError(
            'The data map must have a dimensionality of Nx4, with the first two indices consisting of x, y coordinates. '
            'Indices 2 and 3, if populated, correspond to the nearest and next-nearest network nodes.')

    if check_assigned:
        # check that data map has been assigned
        if np.all(np.isnan(data_map[:, 2])):
            raise ValueError('Data map has not been assigned to a network.')


@njit(cache=True)
def check_network_maps(node_data: np.ndarray,
                       edge_data: np.ndarray,
                       node_edge_map: Dict):
    '''
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
    '''
    # catch zero length node or edge maps
    if len(node_data) == 0:
        raise ValueError('Zero length node map')
    if len(edge_data) == 0:
        raise ValueError('Zero length edge map')
    if not node_data.ndim == 2 or not node_data.shape[1] == 3:
        raise ValueError('The node map must have a dimensionality of Nx4.')
    if not edge_data.ndim == 2 or not edge_data.shape[1] == 7:
        raise ValueError('The edge map must have a dimensionality of Nx7')
    # check sequential and reciprocal node to edge map indices
    edge_counts = np.full(len(edge_data), 0)
    for n_idx in range(len(node_data)):
        # zip through all edges for current node
        for edge_idx in node_edge_map[n_idx]:
            # get the edge
            edge = edge_data[edge_idx]
            # check that the start node matches the current node index
            start_nd_idx, end_nd_idx = edge[:2]
            assert start_nd_idx == n_idx
            # check that each edge has a matching pair in the opposite direction
            paired = False
            for return_edge_idx in node_edge_map[int(end_nd_idx)]:
                if edge_data[return_edge_idx][1] == n_idx:
                    paired = True
                    break
            if not paired:
                raise ValueError('Missing matching edge pair in opposite direction.')
            # add to the counter
            edge_counts[edge_idx] += 1
    if not np.all(edge_counts == 1):
        raise ValueError('Mismatched node and edge maps encountered.')
    if not np.all(np.isfinite(edge_data[:, 0])) or not np.all(edge_data[:, 0] >= 0):
        raise ValueError('Missing or invalid start node index encountered.')
    if not np.all(np.isfinite(edge_data[:, 1])) or not np.all(edge_data[:, 1] >= 0):
        raise ValueError('Missing or invalid end node index encountered.')
    if not np.all(np.isfinite(edge_data[:, 2])) or not np.all(edge_data[:, 2] >= 0):
        raise ValueError('Invalid edge length encountered. Should be finite number greater than or equal to zero.')
    if not np.all(np.isfinite(edge_data[:, 3])) or not np.all(edge_data[:, 3] >= 0):
        raise ValueError(
            'Invalid edge angle sum encountered. Should be finite number greater than or equal to zero.')
    if not np.all(np.isfinite(edge_data[:, 4])) or not np.all(edge_data[:, 4] >= 0):
        raise ValueError(
            'Invalid impedance factor encountered. Should be finite number greater than or equal to zero.')


@njit(cache=True)
def check_distances_and_betas(distances: np.ndarray, betas: np.ndarray):
    if len(distances) == 0:
        raise ValueError('No distances provided.')

    if len(betas) == 0:
        raise ValueError('No betas provided.')

    if not len(distances) == len(betas):
        raise ValueError('The number of distances and betas should be equal.')

    for i in range(len(distances)):
        for j in range(len(distances)):
            if i > j:
                if distances[i] == distances[j]:
                    raise ValueError('Duplicate distances provided. Please provide only one of each.')

    for d in distances:
        if d <= 0:
            raise ValueError('Please provide a positive distance value.')

    for b in betas:
        if b > 0:
            raise ValueError('Please provide the beta value with the leading negative.')

    threshold_min = np.exp(distances[0] * betas[0])
    for d, b in zip(distances, betas):
        if not np.exp(d * b) == threshold_min:
            # handles edge case for infinity
            if not (d, b) == (np.inf, -0.0):
                raise ValueError(
                    'Inconsistent threshold minimums, indicating that the relationship between the betas and distances '
                    'is not consistent for all distance / beta pairs.')
