from typing import Tuple

import numpy as np
from numba import njit
from numba.pycc import CC

from cityseer.algos import centrality, types

cc = CC('data', source_module='cityseer.algos.data')

@cc.export('tiered_sort', '(float64[:,:], uint8)')
@njit
def tiered_sort(arr: np.ndarray, tier: int) -> np.ndarray:
    if tier > arr.shape[1] - 1:
        raise ValueError('The selected tier for sorting exceeds the available tiers.')

    # don't modify the arrays in place
    sort_order = arr[:, tier].argsort()
    return arr[sort_order]


@cc.export('binary_search', '(float64[:], float64, float64)')
@njit
def binary_search(arr: np.ndarray, min: float, max: float) -> Tuple[int, int]:
    if min > max:
        raise ValueError('Max must be greater than min.')

    left_idx = np.searchsorted(arr, min)
    right_idx = np.searchsorted(arr, max, side='right')

    return left_idx, right_idx


@cc.export('generate_index', '(float64[:], float64[:])')
@njit
def generate_index(x_arr: np.ndarray, y_arr: np.ndarray) -> np.ndarray:
    '''
    Create a 2d numpy array:
    0 - x_arr
    1 - x_idx - corresponds to original index of non-sorted x_arr
    2 - y_arr
    3 - y_idx
    '''

    if len(x_arr) != len(y_arr):
        raise ValueError('x and y arrays must match in length')

    index_map = np.full((len(x_arr), 4), np.nan)

    x_idx = np.arange(len(x_arr))
    x_stacked = np.vstack((x_arr, x_idx)).T
    index_map[:, :2] = tiered_sort(x_stacked, tier=0)

    y_idx = np.arange(len(y_arr))
    y_stacked = np.vstack((y_arr, y_idx)).T
    index_map[:, 2:] = tiered_sort(y_stacked, tier=0)

    return index_map


@cc.export('_slice_index', '(float64[:,:], float64, float64, float64)')
@njit
def _slice_index(index_map: np.ndarray, x: float, y: float, max_dist: float) -> Tuple[np.ndarray, np.ndarray]:
    '''
    0 - x_arr
    1 - x_idx - corresponds to original index of non-sorted x_arr
    2 - y_arr
    3 - y_idx
    '''

    # find the x and y ranges
    x_arr = index_map[:, 0]
    y_arr = index_map[:, 2]
    x_start, x_end = binary_search(x_arr, x - max_dist, x + max_dist)
    y_start, y_end = binary_search(y_arr, y - max_dist, y + max_dist)

    # slice the x and y data based on min and max - then sort to index order
    x_range = index_map[x_start:x_end, :2]
    y_range = index_map[y_start:y_end, 2:]

    x_idx_sorted = tiered_sort(x_range, tier=1)
    y_idx_sorted = tiered_sort(y_range, tier=1)

    return x_idx_sorted, y_idx_sorted


@cc.export('_generate_trim_to_full_map', '(float64[:], uint64)')
@njit
def _generate_trim_to_full_map(full_to_trim_map: np.ndarray, trim_count: int) -> np.ndarray:
    # prepare the trim to full map
    trim_to_full_idx_map = np.full(trim_count, np.nan)

    for idx in range(len(full_to_trim_map)):
        trim_map = full_to_trim_map[idx]
        # if the full map has a finite value, then respectively map from the trimmed index to the full index
        if not np.isnan(trim_map):
            trim_to_full_idx_map[int(trim_map)] = idx
        # no need to iterate remainder if last value already written
        if trim_map == len(trim_to_full_idx_map) - 1:
            break
    return trim_to_full_idx_map


@cc.export('distance_filter', '(float64[:,:], float64, float64, float64, boolean)')
@njit
def distance_filter(index_map: np.ndarray, x: float, y: float, max_dist: float, radial=True) -> Tuple[
    np.ndarray, np.ndarray]:
    x_idx_sorted, y_idx_sorted = _slice_index(index_map, x, y, max_dist)

    # prepare the full to trim output map
    total_count = len(index_map)
    full_to_trim_idx_map = np.full(total_count, np.nan)

    # iterate the slices
    trim_count = 0
    y_cursor = 0
    for idx in range(len(x_idx_sorted)):
        # disaggregate this way to avoid numba typing issues
        x_coord = x_idx_sorted[idx][0]
        x_key = x_idx_sorted[idx][1]
        # see if the same key is in the y array
        l, r = binary_search(y_idx_sorted[y_cursor:, 1], x_key, x_key)
        # l is the index relative to the cropped range
        # this can be incremented regardless of matches
        y_cursor += l
        # if the left and right indices are not the same, the element was found
        if l < r:
            # if crow-flies - check the distance
            if radial:
                y_coord = y_idx_sorted[y_cursor, 0]
                dist = np.hypot(x_coord - x, y_coord - y)
                if dist > max_dist:
                    continue
            full_to_trim_idx_map[int(x_key)] = trim_count
            trim_count += 1

    trim_to_full_idx_map = _generate_trim_to_full_map(full_to_trim_idx_map, trim_count)

    return trim_to_full_idx_map, full_to_trim_idx_map


@cc.export('nearest_idx', '(float64[:,:], float64, float64, float64)')
@njit
def nearest_idx(index_map: np.ndarray, x: float, y: float, max_dist: float) -> Tuple[int, float]:
    # get the x and y ranges spanning the max distance
    x_idx_sorted, y_idx_sorted = _slice_index(index_map, x, y, max_dist)

    min_idx = np.nan
    min_dist = np.inf
    y_cursor = 0
    for idx in range(len(x_idx_sorted)):
        # disaggregate this way to avoid numba typing issues
        x_coord = x_idx_sorted[idx][0]
        x_key = x_idx_sorted[idx][1]
        # see if the same key is in the y array
        l, r = binary_search(y_idx_sorted[y_cursor:, 1], x_key, x_key)
        # l is the index relative to the cropped range
        # this can be incremented regardless of matches
        y_cursor += l
        # if the left and right indices are not the same, the element was found
        if l < r:
            # check if it is less than the current minimum
            y_coord = y_idx_sorted[y_cursor, 0]
            dist = np.hypot(x_coord - x, y_coord - y)
            if dist < min_dist:
                min_idx = x_key
                min_dist = dist

    return min_idx, min_dist


# @cc.export('assign_to_network', '(float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64)')
# @njit
def assign_to_network(data_map: np.ndarray, node_map: np.ndarray, edge_map: np.ndarray, netw_index: np.ndarray,
                      max_dist: float) -> np.ndarray:
    '''
    To save unnecessary computation - this is done once and written to the data map.

    1 - find the closest network node from each data point
    2A - wind clockwise along the network to preferably find a block cycle surrounding the node
    2B - in event of topological traps, try anti-clockwise as well
    3A - select the closest block cycle node
    3B - if no enclosing cycle - simply use the closest node
    4 - find the neighbouring node that minimises the distance between the data point on "street-front"

    NODE MAP:
    0 - x
    1 - y
    2 - live
    3 - edge indx
    4 - weight

    EDGE MAP:
    0 - start node
    1 - end node
    2 - length in metres
    3 - impedance

    DATA MAP:
    0 - x
    1 - y
    2 - live
    3 - data class
    4 - assigned network index - nearest
    5 - assigned network index - next-nearest
    '''

    types.check_data_map(data_map)

    types.check_network_types(node_map, edge_map, index_map=netw_index)

    # iterate each data point
    for idx in range(len(data_map)):

        # numba no object mode can only handle basic printing
        if idx % 10000 == 0:
            print('...progress:', round(idx / len(data_map) * 100, 2), '%')

        # iterate each network id
        d_x = data_map[idx][0]
        d_y = data_map[idx][1]
        min_idx, min_dist = nearest_idx(netw_index, d_x, d_y, max_dist)

        # wind along the network - try to find a block cycle around the point
        visited = np.full(len(node_map), False)
        # follow neighbours that maximise the clockwise winding
        node_idx = int(min_idx)
        total_wind = 0
        while True:
            visited[node_idx] = True
            nb_wind = 0
            max_nb_idx = None
            n_x = node_map[node_idx][0]
            n_y = node_map[node_idx][1]
            edge_idx = int(node_map[node_idx][3])
            while edge_idx < len(edge_map):
                # get the edge's start and end node indices
                start, end = edge_map[edge_idx][:2]
                # if the start index no longer matches it means all neighbours have been visited
                if start != node_idx:
                    break
                # increment idx for next loop
                edge_idx += 1
                # cast to int for indexing
                nb = int(end)
                nb_x = node_map[nb][0]
                nb_y = node_map[nb][1]
                # calculate the wind
                wind = np.random.uniform(1, 10)
                # if greater than, update
                if wind > nb_wind:
                    total_wind += wind
                    max_nb_idx = nb

            # break conditions
            # 1 - if the original node is re-encountered after circling a block
            if max_nb_idx == min_idx:
                break
            # 2 - if no neighbour is found
            if max_nb_idx is None:
                break
            # 3 - if the max wind node had already been visited, then a non-encircling loop was found...
            if visited[max_nb_idx]:
                break
            node_idx = max_nb_idx

        # if clockwise fails, try counter-clockwise

        # if encircling succeeded: iterate visited nodes and select closest
        # NOTE -> this is not necessarily the starting node if the nearest node is not on the encircling block

        # select any neighbours on the encircling (visited) route and choose the one with the shallowest adjacent street

        # if encircling failed: simply select the closest and next closest neighbour

        # set in the data map
        data_map[idx][4] = 0  # adj_idx
        data_map[idx][5] = 0  # next_adj_idx

    print('...done')

    return data_map


# @cc.export('aggregate_to_src_idx', '(uint64, float64, float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], boolean)')
# @njit
def aggregate_to_src_idx(src_idx: int, max_dist: float, node_map: np.ndarray, edge_map: np.ndarray,
                         netw_index: np.ndarray,
                         data_map: np.ndarray, angular: bool = False):
    netw_x_arr = node_map[:, 0]
    netw_y_arr = node_map[:, 1]
    src_x = netw_x_arr[src_idx]
    src_y = netw_y_arr[src_idx]

    data_x_arr = data_map[:, 0]
    data_y_arr = data_map[:, 1]
    data_classes = data_map[:, 3]
    data_assign_map = data_map[:, 4]
    data_assign_dist = data_map[:, 5]

    # filter the network by distance
    netw_trim_to_full_idx_map, netw_full_to_trim_idx_map = \
        distance_filter(netw_index, src_x, src_y, max_dist)

    # run the shortest tree dijkstra
    # keep in mind that predecessor map is based on impedance heuristic - which can be different from metres
    # NOTE -> use np.inf for max distance so as to explore all paths
    # In some cases the predecessor nodes will be within reach even if the closest node is not
    # Total distance is check later
    map_impedance_trim, map_distance_trim, map_pred_trim, _cycles_trim = \
        centrality.shortest_path_tree(node_map, edge_map, src_idx, netw_trim_to_full_idx_map, netw_full_to_trim_idx_map,
                                      max_dist=np.inf, angular=angular)

    # STEP A - SLICE THE DATA POINTS

    # STEP B - LOOKUP EACH DATA POINTS NEAREST AND NEXT NEAREST ADJACENCIES

    # STEP C - AGGREGATE TO LEAST DISTANCE VIA SHORTEST PATH MAP

    # filter the data by distance
    # in this case, the source x, y is the same as for the networks
    data_trim_to_full_idx_map, _data_full_to_trim_idx_map = distance_filter(data_index, src_x, src_y, max_dist)

    # prepare the new data arrays
    reachable_classes_trim = np.full(len(data_trim_to_full_idx_map), np.nan)
    reachable_classes_dist_trim = np.full(len(data_trim_to_full_idx_map), np.inf)

    # iterate the distance trimmed data points
    for idx, data_idx_full in enumerate(data_trim_to_full_idx_map):
        # cast to int
        data_idx_full = int(data_idx_full)

        # TODO: error here, cannot convert float NaN to int

        # find the full and trim indices of the assigned network node
        assigned_netw_idx_full = int(data_assign_map[data_idx_full])
        assigned_netw_idx_trim = int(netw_full_to_trim_idx_map[assigned_netw_idx_full])
        # calculate the distance via the assigned network node - use the distance map, not the impedance map
        # in some cases the distance will exceed the max distance, though the predecessor may offer a closer route
        dist_calc = map_distance_trim[assigned_netw_idx_trim] + data_assign_dist[data_idx_full]
        # get the predecessor node so that distance can be compared
        # in some cases this will provide access via a closer corner, especially for non-decomposed networks
        prev_netw_idx_trim = int(map_pred_trim[assigned_netw_idx_trim])
        prev_netw_full_idx = int(netw_trim_to_full_idx_map[prev_netw_idx_trim])
        # calculate the distance - in this case the tail-end from the network node to data point is computed manually
        prev_dist_calc = map_distance_trim[prev_netw_idx_trim]
        prev_dist_calc += np.hypot(netw_x_arr[prev_netw_full_idx] - data_x_arr[data_idx_full],
                                   netw_y_arr[prev_netw_full_idx] - data_y_arr[data_idx_full])
        # use the shorter distance between the current and prior nodes
        # but only if less than the maximum distance
        if dist_calc <= max_dist and dist_calc <= prev_dist_calc:
            reachable_classes_trim[idx] = data_classes[data_idx_full]
            reachable_classes_dist_trim[idx] = dist_calc
        elif prev_dist_calc <= max_dist and prev_dist_calc < dist_calc:
            reachable_classes_trim[idx] = data_classes[data_idx_full]
            reachable_classes_dist_trim[idx] = prev_dist_calc

    # note that some entries will be nan values if the max distance was exceeded
    return reachable_classes_trim, reachable_classes_dist_trim, data_trim_to_full_idx_map
