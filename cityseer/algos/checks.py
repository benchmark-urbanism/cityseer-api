import numpy as np
from numba import njit


def_min_thresh_wt = 0.01831563888873418


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
        if not np.isfinite(cl) or not cl >= 0:
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
            'The data map must have a dimensionality of Nx4, with the first two indices consisting of x, y coordinates. Indices 2 and 3, if populated, correspond to the nearest and next-nearest network nodes.')

    if check_assigned:
        # check that data map has been assigned
        if np.all(np.isnan(data_map[:, 2])):
            raise ValueError('Data map has not been assigned to a network.')


@njit(cache=True)
def check_trim_maps(trim_to_full: np.ndarray, full_to_trim: np.ndarray):
    if len(trim_to_full) > len(full_to_trim):
        raise ValueError(
            'The trim_to_full map is longer than the full_to_trim map. Check that these have not been switched.')

    counter = 0
    # test for round-trip in the indices
    # i.e. full_to_trim indices should be reciprocated by the trim_to_full indices in the other direction
    for i in range(len(full_to_trim)):
        # NaN values indicate points that are filtered out
        # Non NaN values, on the other-hand, should point to the next index on the trim map
        if not np.isnan(full_to_trim[i]):
            trim_idx = int(full_to_trim[i])
            # if the index exceeds the length of the trim map
            if trim_idx >= len(trim_to_full):
                raise ValueError('Trim index exceeds range of trim_to_full map.')
            # indices should increase by one
            if trim_idx > counter:
                raise ValueError('Non-sequential index in full_to_trim map.')
            # if the reciprocal trim_to_full index doesn't match the current i
            full_idx = trim_to_full[trim_idx]
            if full_idx != i:
                raise ValueError('Mismatching trim-to-full and full-to-trim maps.')
            counter += 1
    # the counter (number of reciprocal indices) should match the length of the trim_to_full map
    if counter != len(trim_to_full):
        raise ValueError(
            'The length of the trim-to-full map does not match the number of active elements in the full-to-trim map.')


@njit(cache=True)
def check_network_maps(node_map: np.ndarray, edge_map: np.ndarray):
    '''
    NODE MAP:
    0 - x
    1 - y
    2 - live
    3 - edge index
    4 - weight

    EDGE MAP:
    0 - start node
    1 - end node
    2 - length in metres
    3 - impedance
    '''

    # catch zero length node or edge maps
    if len(node_map) == 0:
        raise ValueError('Zero length node map')
    if len(edge_map) == 0:
        raise ValueError('Zero length edge map')

    if not node_map.ndim == 2 or not node_map.shape[1] == 5:
        raise ValueError(
            'The node map must have a dimensionality of Nx5, consisting of x, y, live, edge idx, and weight attributes.')

    if not edge_map.ndim == 2 or not edge_map.shape[1] == 4:
        raise ValueError(
            'The edge map must have a dimensionality of Nx4, consisting of start, end, length, and impedance attributes.')

    # check sequential and reciprocal node to edge map indices
    edge_counter = 0
    for n_idx in range(len(node_map)):
        # in the event of isolated nodes, there will be no corresponding edge index
        e_idx = node_map[n_idx][3]
        if np.isnan(e_idx):
            continue
        # the edge index should match the sequential edge counter
        if e_idx != edge_counter:
            raise ValueError('Mismatched node / edge maps encountered.')
        while edge_counter < len(edge_map):
            start = edge_map[edge_counter][0]
            if start != n_idx:
                break
            edge_counter += 1
    if edge_counter != len(edge_map):
        raise ValueError('Mismatched node and edge maps encountered.')

    if not np.all(np.isfinite(node_map[:, 4])) or not np.all(node_map[:, 4] >= 0):
        raise ValueError('Invalid node weights encountered. All weights should be greater than or equal to zero.')

    if not np.all(np.isfinite(edge_map[:, 2])) or not np.all(edge_map[:, 2] >= 0):
        raise ValueError('Invalid edge length encountered. All edge lengths should be greater than or equal to zero.')

    if not np.all(np.isfinite(edge_map[:, 3])) or not np.all(edge_map[:, 3] >= 0):
        raise ValueError(
            'Invalid edge impedance encountered. All edge impedances should be greater than or equal to zero.')


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
            if i != j:
                if distances[i] == distances[j]:
                    raise ValueError('Duplicate distances provided. Please provide only one of each.')

    for b in betas:
        if b >= 0:
            raise ValueError('Please provide the beta values with the leading negative.')

    threshold_min = np.exp(distances[0] * -betas[0])
    for d, b in zip(distances, betas):
        if not np.exp(d * -b) == threshold_min:
            raise ValueError(
                'Inconsistent threshold minimums, indicating that the relationship between the betas and distances is not consistent for all distance / beta pairs.')
