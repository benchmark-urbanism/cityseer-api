import numpy as np
from typing import Tuple
from numba.pycc import CC
from numba import njit
from cityseer.algos import networks


cc = CC('data')


@cc.export('_top_down_split_merge_1d', '(float64[:], uint64, uint64, float64[:])')
@njit
def _top_down_split_merge_1d(work_arr:np.ndarray, begin_idx:int, end_idx:int, arr:np.ndarray):

    if end_idx - begin_idx < 2:
        return

    split_idx = int((end_idx + begin_idx) / 2)

    _top_down_split_merge_1d(arr, begin_idx, split_idx, work_arr)
    _top_down_split_merge_1d(arr, split_idx, end_idx, work_arr)
    _top_down_merge_1d(work_arr, begin_idx, split_idx, end_idx, arr)


@cc.export('_top_down_split_merge_2d', '(float64[:,:], uint64, uint64, float64[:,:])')
@njit
def _top_down_split_merge_2d(work_arr:np.ndarray, begin_idx:int, end_idx:int, arr:np.ndarray):

    if end_idx - begin_idx < 2:
        return

    split_idx = int((end_idx + begin_idx) / 2)

    _top_down_split_merge_2d(arr, begin_idx, split_idx, work_arr)
    _top_down_split_merge_2d(arr, split_idx, end_idx, work_arr)
    _top_down_merge_2d(work_arr, begin_idx, split_idx, end_idx, arr)


@cc.export('_top_down_merge_1d', '(float64[:], uint64, uint64, uint64, float64[:])')
@njit
def _top_down_merge_1d(arr:np.ndarray, begin_idx:int, mid_idx:int, end_idx:int, work_arr:np.ndarray):
    i = begin_idx
    j = mid_idx

    k = begin_idx
    while k < end_idx:
        if i < mid_idx and (j >= end_idx or arr[i] <= arr[j]):
            work_arr[k] = arr[i]
            i += 1
        else:
            work_arr[k] = arr[j]
            j += 1
        k += 1


@cc.export('_top_down_merge_2d', '(float64[:,:], uint64, uint64, uint64, float64[:,:])')
@njit
def _top_down_merge_2d(arr:np.ndarray, begin_idx:int, mid_idx:int, end_idx:int, work_arr:np.ndarray):
    i = begin_idx
    j = mid_idx

    k = begin_idx
    while k < end_idx:
        if i < mid_idx and (j >= end_idx or arr[i][0] <= arr[j][0]):
            work_arr[k] = arr[i]
            i += 1
        else:
            work_arr[k] = arr[j]
            j += 1
        k += 1


@cc.export('merge_sort_1d', 'float64[:](float64[:])')
@njit
def merge_sort_1d(arr:np.ndarray) -> np.ndarray:
    '''
    https://en.wikipedia.org/wiki/Merge_sort
    NOTE -> this version is for single deep arrays
    '''
    work_arr = arr.copy()
    n = len(arr)
    _top_down_split_merge_1d(work_arr, 0, n, arr)
    return arr


@cc.export('merge_sort_2d', 'float64[:,:](float64[:,:])')
@njit
def merge_sort_2d(arr:np.ndarray) -> np.ndarray:
    '''
    https://en.wikipedia.org/wiki/Merge_sort
    NOTE -> this version is for double deep arrays - second level keeps track of original indices
    '''
    work_arr = arr.copy()
    n = len(arr)
    _top_down_split_merge_2d(work_arr, 0, n, arr)
    return arr


@cc.export('generate_idx', '(float64[:], float64[:])')
@njit
def generate_index(x_arr:np.ndarray, y_arr:np.ndarray) -> np.ndarray:
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
    index_map[:,:2] = merge_sort_2d(x_stacked)

    y_idx = np.arange(len(y_arr))
    y_stacked = np.vstack((y_arr, y_idx)).T
    index_map[:, 2:] = merge_sort_2d(y_stacked)

    return index_map


@cc.export('_binary_search_leftmost', '(float64[:], float64)')
@njit
def _binary_search_leftmost(arr:np.ndarray, target_min:float) -> int:
    L = 0
    R = len(arr)
    while L < R:
        m = int(np.floor((L + R) / 2))
        if arr[m] < target_min:
            L = m + 1
        else:
            R = m
    return L


@cc.export('_binary_search_rightmost', '(float64[:], float64)')
@njit
def _binary_search_rightmost(arr:np.ndarray, target_max:float) -> int:
    L = 0
    R = len(arr)
    while L < R:
        m = int(np.floor((L + R) / 2))
        if arr[m] <= target_max:
            L = m + 1
        else:
            R = m
    return R


@cc.export('binary_search', '(float64[:], float64, float64)')
@njit
def binary_search(arr:np.ndarray, min:float, max:float) -> Tuple[int, int]:
    '''
    https://en.wikipedia.org/wiki/Merge_sort
    NOTE -> this version is for double deep arrays - second level keeps track of original indices
    '''

    if min > max:
        raise ValueError('Max must be greater than min.')

    left_idx = _binary_search_leftmost(arr, min)
    right_idx = _binary_search_rightmost(arr, max)

    return left_idx, right_idx


@cc.export('spatial_filter', '(float64[:,:], float64, float64, uint64)')
@njit
def spatial_filter(spatial_index:np.ndarray, x:float, y:float, distance:float, radial=True) -> np.ndarray:
    '''
    0 - x_arr
    1 - x_idx - corresponds to original index of non-sorted x_arr
    2 - y_arr
    3 - y_idx
    '''
    x_arr = spatial_index[:,0]
    x_idx = spatial_index[:,1]
    y_arr = spatial_index[:,2]
    y_idx = spatial_index[:,3]
    x_start, x_end = binary_search(x_arr, x - distance, x + distance)
    y_start, y_end = binary_search(y_arr, y - distance, y + distance)

    x_keys = x_idx[x_start, x_end].copy()
    x_keys_sorted = merge_sort_1d(x_keys)

    y_keys = y_idx[y_start, y_end].copy()
    y_keys_sorted = merge_sort_1d(y_keys)

    y_cursor = 0
    for x_cursor, x_key in enumerate(x_keys_sorted):
        l, r = binary_search(y_keys_sorted[y_cursor:], x_key, x_key)
        # if the left and right indices are the same, the element was not found -> so remove
        if l == r:
            x_keys_sorted[x_cursor] = np.nan
        # but if they are different, then it was found and the y_cursor can be bumped
        else:
            y_cursor = l
            # if crow-flies - check the distance
            if radial:


    # create the


@cc.export('assign_to_network', '(float64[:,:], float64[:,:], float64)')
@njit
def assign_to_network(data_map:np.ndarray, node_map:np.ndarray, may_dist:float) -> np.ndarray:
    '''
    Each data point is assigned to the closest network node.

    This is designed to be done once prior to windowed iteration of the graph.

    Crow-flies operations are performed inside the iterative data aggregation step because pre-computation would be memory-prohibitive due to an N*M matrix.

    Note that the assignment to a network index is a starting reference for the data aggregation step, and that if the prior point on the shortest path is closer, then the distance will be calculated via the prior point instead.

    NODE MAP:
    0 - x
    1 - y
    2 - live
    3 - edge indx
    4 - weight

    DATA MAP:
    0 - x
    1 - y
    2 - live
    3 - data class
    4 - assigned network index
    5 - distance from assigned network index
    '''

    if data_map.shape[1] != 6:
        raise AttributeError('The data map must have a dimensionality of Nx6, with the first four indices consisting of x, y, live, and class attributes. This method will populate indices 5 and 6.')

    if node_map.shape[1] != 5:
        raise AttributeError('The node map must have a dimensionality of Nx5, consisting of x, y, live, link idx, and weight attributes.')

    netw_x_arr = node_map[:,0]
    netw_y_arr = node_map[:,1]
    data_x_arr = data_map[:,0]
    data_y_arr = data_map[:,1]

    # iterate each data point
    for idx, data_idx in enumerate(range(len(data_map))):

        # numba no object mode can only handle basic printing
        if idx % 10000 == 0:
            print('...progress:', round(idx / len(data_map) * 100, 2), '%')

        # iterate each network id
        for network_idx in range(len(node_map)):
            # get the distance
            dist = np.sqrt(
                (netw_x_arr[network_idx] - data_x_arr[data_idx]) ** 2 +
                (netw_y_arr[network_idx] - data_y_arr[data_idx]) ** 2)
            # only proceed if it is less than the max dist cutoff
            if dist > max_dist:
                continue
            # if no adjacent network point has yet been assigned for this data point
            # then proceed to record this adjacency and the corresponding distance
            elif np.isnan(data_map[data_idx][5]):
                data_map[data_idx][5] = dist
                data_map[data_idx][4] = network_idx
            # otherwise, only update if the new distance is less than any prior distances
            elif dist < data_map[data_idx][5]:
                data_map[data_idx][5] = dist
                data_map[data_idx][4] = network_idx

    print('...done')

    return data_map


#@cc.export('aggregate_to_src_idx', '(float64[:,:], float64[:,:], float64[:,:], uint64, float64, boolean)')
#@njit
def aggregate_to_src_idx(node_map:np.ndarray, edge_map:np.ndarray, data_map:np.ndarray, src_idx:int, max_dist:float, angular:bool=False):

    netw_x_arr = node_map[:,0]
    netw_y_arr = node_map[:,1]
    src_x = netw_x_arr[src_idx]
    src_y = netw_y_arr[src_idx]

    data_x_arr = data_map[:,0]
    data_y_arr = data_map[:,1]
    data_classes = data_map[:,3]
    data_assign_map = data_map[:,4]
    data_assign_dist = data_map[:,5]

    # filter the network by distance
    netw_trim_to_full_idx_map, netw_full_to_trim_idx_map = \
        networks.crow_flies(src_x, src_y, netw_x_arr, netw_y_arr, max_dist)

    # run the shortest tree dijkstra
    # keep in mind that predecessor map is based on impedance heuristic - which can be different from metres
    # NOTE -> use np.inf for max distance so as to explore all paths
    # In some cases the predecessor nodes will be within reach even if the closest node is not
    # Total distance is check later
    map_impedance_trim, map_distance_trim, map_pred_trim, _cycles_trim = \
        networks.shortest_path_tree(node_map, edge_map, src_idx, netw_trim_to_full_idx_map, netw_full_to_trim_idx_map,
                                    max_dist=np.inf, angular=angular)

    # filter the data by distance
    # in this case, the source x, y is the same as for the networks
    data_trim_to_full_idx_map, _data_full_to_trim_idx_map = \
        networks.crow_flies(src_x, src_y, data_x_arr, data_y_arr, max_dist)

    # prepare the new data arrays
    reachable_classes_trim = np.full(len(data_trim_to_full_idx_map), np.nan)
    reachable_classes_dist_trim = np.full(len(data_trim_to_full_idx_map), np.inf)

    # iterate the distance trimmed data points
    for idx, data_idx_full in enumerate(data_trim_to_full_idx_map):
        # cast to int
        data_idx_full = int(data_idx_full)
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
        prev_dist_calc += np.sqrt(
            (netw_x_arr[prev_netw_full_idx] - data_x_arr[data_idx_full]) ** 2 +
            (netw_y_arr[prev_netw_full_idx] - data_y_arr[data_idx_full]) ** 2)
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
