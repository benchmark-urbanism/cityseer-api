import numpy as np
from numba import njit


# cc = CC('checks')


#@cc.export('check_data_map', 'void(float64[:,:])')
@njit
def check_data_map(data_map: np.ndarray):
    if not data_map.ndim == 2 or data_map.shape[1] != 6:
        raise ValueError(
            'The data map must have a dimensionality of Nx6, with the first four indices consisting of x, y, live, and class attributes. Indices 5 and 6, if populated, correspond to the nearest and next-nearest network nodes.')


#@cc.export('check_trim_maps', '(float64[:], float64[:])')
@njit
def check_trim_maps(trim_to_full: np.ndarray, full_to_trim: np.ndarray):
    counter = 0
    for idx in range(len(full_to_trim)):
        ref = full_to_trim[idx]
        if not np.isnan(ref):
            if trim_to_full[int(ref)] != idx or idx > len(trim_to_full):
                raise ValueError('Mismatching trim-to-full and full-to-trim maps.')
            counter += 1
    if counter != len(trim_to_full):
        raise ValueError(
            'The length of the trim-to-full map does not match the number of active elements in the full-to-trim map.')


#@cc.export('check_network_types', '(float64[:,:], float64[:,:])')
@njit
def check_network_types(node_map: np.ndarray, edge_map: np.ndarray):
    if not node_map.ndim == 2 or node_map.shape[1] != 5:
        raise ValueError(
            'The node map must have a dimensionality of Nx5, consisting of x, y, live, link idx, and weight attributes.')

    if not edge_map.ndim == 2 or edge_map.shape[1] != 4:
        raise ValueError(
            'The link map must have a dimensionality of Nx4, consisting of start, end, length, and impedance attributes.')


#@cc.export('check_distances_and_betas', '(float64[:], float64[:])')
@njit
def check_distances_and_betas(distances: np.ndarray, betas: np.ndarray):
    if len(distances) != len(betas):
        raise ValueError('The number of distances and betas should be equal.')

    for b in betas:
        if b >= 0:
            raise ValueError('Please provide the beta values with the leading negative.')

    threshold_min = np.exp(distances[0] * -betas[0])
    for d, b in zip(distances, betas):
        if np.exp(d * -b) != threshold_min:
            raise ValueError(
                'Inconsistent threshold minimums, indicating that the relationship between the betas and distances is not consistent for all distance / beta pairs.')
