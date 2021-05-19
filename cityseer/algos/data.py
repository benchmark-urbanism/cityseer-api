from typing import Tuple

import numpy as np
from numba import njit
from numba.typed import Dict

from cityseer.algos import centrality, checks, diversity


@njit(cache=True, nogil=True)
def radial_filter(src_x: float, src_y: float, x_arr: np.ndarray, y_arr: np.ndarray, max_dist: float) -> np.ndarray:
    if len(x_arr) != len(y_arr):
        raise ValueError('Mismatching x and y array lengths.')

    # filter by distance
    total_count = len(x_arr)
    data_filter = np.full(total_count, False)

    # if infinite max, then no need to check distances
    if max_dist == np.inf:
        data_filter[:] = True
        return data_filter

    else:
        for i in range(total_count):
            dist = np.hypot(x_arr[i] - src_x, y_arr[i] - src_y)
            if dist <= max_dist:
                data_filter[i] = True

    return data_filter


@njit(cache=True, nogil=True)
def find_nearest(src_x: float, src_y: float, x_arr: np.ndarray, y_arr: np.ndarray, max_dist: float) -> Tuple[
    int, float]:
    if len(x_arr) != len(y_arr):
        raise ValueError('Mismatching x and y array lengths.')

    # filter by distance
    total_count = len(x_arr)
    min_idx = np.nan
    min_dist = np.inf

    for i in range(total_count):
        dist = np.hypot(x_arr[i] - src_x, y_arr[i] - src_y)
        if dist <= max_dist and dist < min_dist:
            min_idx = i
            min_dist = dist

    return min_idx, min_dist


@njit(cache=False, nogil=True)
def assign_to_network(data_map: np.ndarray,
                      node_data: np.ndarray,
                      edge_data: np.ndarray,
                      node_edge_map: Dict,
                      max_dist: float,
                      suppress_progress: bool = False) -> np.ndarray:
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
    EDGE MAP:
    0 - start node
    1 - end node
    2 - length in metres
    3 - sum of angular travel along length
    4 - impedance factor
    5 - entry bearing
    6 - exit bearing
    DATA MAP:
    0 - x
    1 - y
    2 - assigned network index - nearest
    3 - assigned network index - next-nearest
    '''
    checks.check_network_maps(node_data, edge_data, node_edge_map)

    def calculate_rotation(point_a, point_b):
        # https://stackoverflow.com/questions/37459121/calculating-angle-between-three-points-but-only-anticlockwise-in-python
        # these two points / angles are relative to the origin - so pass in difference between the points and origin as vectors
        ang_a = np.arctan2(point_a[1], point_a[0])  # arctan is in y/x order
        ang_b = np.arctan2(point_b[1], point_b[0])
        return np.rad2deg((ang_a - ang_b) % (2 * np.pi))

    def calculate_rotation_smallest(point_a, point_b):
        # smallest difference angle
        ang_a = np.rad2deg(np.arctan2(point_a[1], point_a[0]))
        ang_b = np.rad2deg(np.arctan2(point_b[1], point_b[0]))
        return np.abs((ang_b - ang_a + 180) % 360 - 180)

    def road_distance(d_coords, netw_idx_a, netw_idx_b):
        a_coords = node_data[netw_idx_a, :2]
        b_coords = node_data[netw_idx_b, :2]
        # get the angles from either intersection node to the data point
        ang_a = calculate_rotation_smallest(d_coords - a_coords, b_coords - a_coords)
        ang_b = calculate_rotation_smallest(d_coords - b_coords, a_coords - b_coords)
        # assume offset street segment if either is significantly greater than 90 (in which case sideways offset from road)
        if ang_a > 110 or ang_b > 110:
            return np.inf, np.nan, np.nan
        # calculate height from two sides and included angle
        side_a = np.hypot(d_coords[0] - a_coords[0], d_coords[1] - a_coords[1])
        side_b = np.hypot(d_coords[0] - b_coords[0], d_coords[1] - b_coords[1])
        base = np.hypot(a_coords[0] - b_coords[0], a_coords[1] - b_coords[1])
        # forestall potential division by zero
        if base == 0:
            return np.inf, np.nan, np.nan
        # heron's formula
        s = (side_a + side_b + base) / 2  # perimeter / 2
        a = np.sqrt(s * (s - side_a) * (s - side_b) * (s - base))
        # area is 1/2 base * h, so h = area / (0.5 * base)
        h = a / (0.5 * base)
        # NOTE - the height of the triangle may be less than the distance to the nodes
        # happens due to offset segments: can cause wrong assignment where adjacent segments have same triangle height
        # in this case, set to length of closest node so that h (minimum distance) is still meaningful
        # return indices in order of nearest then next nearest
        if side_a < side_b:
            if ang_a > 90:
                h = side_a
            return h, netw_idx_a, netw_idx_b
        else:
            if ang_b > 90:
                h = side_b
            return h, netw_idx_b, netw_idx_a

    def closest_intersections(d_coords, pr_map, end_node):
        if len(pr_map) == 1:
            return np.inf, end_node, np.nan
        current_idx = end_node
        next_idx = int(pr_map[int(end_node)])
        if len(pr_map) == 2:
            return road_distance(d_coords, current_idx, next_idx)
        nearest_idx = np.nan
        next_nearest_idx = np.nan
        min_d = np.inf
        first_pred = next_idx  # for finding end of loop
        while True:
            h, n_idx, n_n_idx = road_distance(d_coords, current_idx, next_idx)
            if h < min_d:
                min_d = h
                nearest_idx = n_idx
                next_nearest_idx = n_n_idx
            # if the next in the chain is nan, then break
            if np.isnan(pr_map[next_idx]):
                break
            current_idx = next_idx
            next_idx = int(pr_map[next_idx])
            if next_idx == first_pred:
                break
        return min_d, nearest_idx, next_nearest_idx

    pred_map = np.full(len(node_data), np.nan)
    netw_coords = node_data[:, :2]
    netw_x_arr = node_data[:, 0]
    netw_y_arr = node_data[:, 1]
    data_coords = data_map[:, :2]
    data_x_arr = data_map[:, 0]
    data_y_arr = data_map[:, 1]
    total_count = len(data_map)
    # setup progress bar params
    steps = int(total_count / 10000)
    for data_idx in range(total_count):
        if not suppress_progress:
            checks.progress_bar(data_idx, total_count, steps)
        # find the nearest network node
        min_idx, min_dist = find_nearest(data_x_arr[data_idx], data_y_arr[data_idx], netw_x_arr, netw_y_arr, max_dist)
        # in some cases no network node will be within max_dist... so accept NaN
        if np.isnan(min_idx):
            continue
        # nearest is initially set for this nearest node, but if a nearer street-edge is found, it will be overriden
        nearest = min_idx
        next_nearest = np.nan
        # set start node to nearest network node
        node_idx = int(min_idx)
        # keep track of visited nodes
        pred_map.fill(np.nan)
        # state
        reversing = False
        # keep track of previous indices
        prev_idx = np.nan
        # iterate neighbours
        while True:
            # reset neighbour rotation and index counters
            rotation = np.nan
            nb_idx = np.nan
            # iterate the edges
            for edge_idx in node_edge_map[node_idx]:
                # get the edge's start and end node indices
                start, end = edge_data[edge_idx, :2]
                # cast to int for indexing
                new_idx = int(end)
                # don't follow self-loops
                if new_idx == node_idx:
                    continue
                # check that this isn't the previous node (already visited as neighbour from other direction)
                if np.isfinite(prev_idx) and new_idx == prev_idx:
                    continue
                # look for the new neighbour with the smallest rightwards (anti-clockwise arctan2) angle
                # measure the angle relative to the data point for the first node
                if np.isnan(prev_idx):
                    r = calculate_rotation(netw_coords[int(new_idx)] - netw_coords[node_idx],
                                           data_coords[data_idx] - netw_coords[node_idx])
                # else relative to the previous node
                else:
                    r = calculate_rotation(netw_coords[int(new_idx)] - netw_coords[node_idx],
                                           netw_coords[int(prev_idx)] - netw_coords[node_idx])
                if reversing:
                    r = 360 - r
                # if least angle, update
                if np.isnan(rotation) or r < rotation:
                    rotation = r
                    nb_idx = new_idx
            # allow backtracking if no neighbour is found - i.e. dead-ends
            if np.isnan(nb_idx):
                if np.isnan(pred_map[node_idx]):
                    # for isolated nodes: nb_idx == np.nan, pred_map[node_idx] == np.nan, and prev_idx == np.nan
                    if np.isnan(prev_idx):
                        break
                    # for isolated edges, the algorithm gets turned-around back to the starting node with nowhere to go
                    # nb_idx == np.nan, pred_map[node_idx] == np.nan
                    # in these cases, pass closest_intersections the prev idx so that it has a predecessor to follow
                    d, n, n_n = closest_intersections(data_coords[data_idx], pred_map, int(prev_idx))
                    if d < min_dist:
                        nearest = n
                        next_nearest = n_n
                    break
                # otherwise, go ahead and backtrack
                nb_idx = pred_map[node_idx]
            # if the distance is exceeded, reset and attempt in the other direction
            dist = np.hypot(netw_x_arr[int(nb_idx)] - data_x_arr[data_idx],
                            netw_y_arr[int(nb_idx)] - data_y_arr[data_idx])
            if dist > max_dist:
                pred_map[int(nb_idx)] = node_idx
                d, n, n_n = closest_intersections(data_coords[data_idx], pred_map, int(nb_idx))
                # if the distance to the street edge is less than the nearest node, or than the prior closest edge
                if d < min_dist:
                    min_dist = d
                    nearest = n
                    next_nearest = n_n
                # reverse and try in opposite direction
                if not reversing:
                    reversing = True
                    pred_map.fill(np.nan)
                    node_idx = int(min_idx)
                    prev_idx = np.nan
                    continue
                break
            # ignore the following conditions while backtracking
            # (if backtracking, the current node's predecessor will be equal to the new neighbour)
            if nb_idx != pred_map[node_idx]:
                # if the new nb node has already been visited then terminate, this prevent infinite loops
                # or, if the algorithm has circled the block back to the original starting node
                if not np.isnan(pred_map[int(nb_idx)]) or nb_idx == min_idx:
                    # set the final predecessor, BUT ONLY if re-encountered the original node
                    # this would otherwise occlude routes (e.g. backtracks) that have passed the same node twice
                    # (such routes are still able to recover the closest edge)
                    if nb_idx == min_idx:
                        pred_map[int(nb_idx)] = node_idx
                    d, n, n_n = closest_intersections(data_coords[data_idx], pred_map, int(nb_idx))
                    if d < min_dist:
                        nearest = n
                        next_nearest = n_n
                    break
                # set predecessor (only if not backtracking)
                pred_map[int(nb_idx)] = node_idx
            # otherwise, keep going
            prev_idx = node_idx
            node_idx = int(nb_idx)
        # print(f'[{data_idx}, {nearest}, {next_nearest}],')
        # set in the data map
        data_map[data_idx, 2] = nearest  # adj_idx
        # in some cases next nearest will be NaN
        # this is mostly in situations where it works to leave as NaN - e.g. access off dead-ends...
        data_map[data_idx, 3] = next_nearest  # next_adj_idx

    return data_map


@njit(cache=False, nogil=True)
def aggregate_to_src_idx(netw_src_idx: int,
                         node_data: np.ndarray,
                         edge_data: np.ndarray,
                         node_edge_map: Dict,
                         data_map: np.ndarray,
                         max_dist: float,
                         angular: bool = False):
    # this function is typically called iteratively, so do type checks from parent methods
    netw_x_arr = node_data[:, 0]
    netw_y_arr = node_data[:, 1]
    netw_src_x = netw_x_arr[netw_src_idx]
    netw_src_y = netw_y_arr[netw_src_idx]
    d_x_arr = data_map[:, 0]
    d_y_arr = data_map[:, 1]
    d_assign_nearest = data_map[:, 2]
    d_assign_next_nearest = data_map[:, 3]
    # run the shortest tree dijkstra
    # keep in mind that predecessor map is based on impedance heuristic - which can be different from metres
    # NOTE -> use np.inf for max distance so as to explore all paths
    # In some cases the predecessor nodes will be within reach even if the closest node is not
    # Total distance is checked later
    tree_map, tree_edges = centrality.shortest_path_tree(edge_data,
                                                         node_edge_map,
                                                         netw_src_idx,
                                                         max_dist=max_dist,
                                                         angular=angular)  # turn off checks! This is called iteratively...
    tree_preds = tree_map[:, 1]
    tree_dists = tree_map[:, 2]
    # filter the data by distance
    # in this case, the source x, y is the same as for the networks
    filtered_data = radial_filter(netw_src_x, netw_src_y, d_x_arr, d_y_arr, max_dist)
    # arrays for writing the reachable data points and their distances
    reachable_data = np.full(len(data_map), False)
    reachable_data_dist = np.full(len(data_map), np.inf)
    # iterate the distance trimmed data points
    reachable_idx = np.where(filtered_data)[0]
    for data_idx in reachable_idx:
        # find the primary assigned network index for the data point
        if np.isfinite(d_assign_nearest[data_idx]):
            netw_idx = int(d_assign_nearest[data_idx])
            # if the assigned network node is within the threshold
            if tree_dists[netw_idx] < max_dist:
                # get the distance from the data point to the network node
                d_d = np.hypot(d_x_arr[data_idx] - netw_x_arr[netw_idx],
                               d_y_arr[data_idx] - netw_y_arr[netw_idx])
                # add to the distance assigned for the network node
                dist = tree_dists[netw_idx] + d_d
                # only assign distance if within max distance
                if dist <= max_dist:
                    reachable_data[data_idx] = True
                    reachable_data_dist[data_idx] = dist
        # the next-nearest may offer a closer route depending on the direction the shortest path approaches from
        if np.isfinite(d_assign_next_nearest[data_idx]):
            netw_idx = int(d_assign_next_nearest[data_idx])
            # if the assigned network node is within the threshold
            if tree_dists[netw_idx] < max_dist:
                # get the distance from the data point to the network node
                d_d = np.hypot(d_x_arr[data_idx] - netw_x_arr[netw_idx],
                               d_y_arr[data_idx] - netw_y_arr[netw_idx])
                # add to the distance assigned for the network node
                dist = tree_dists[netw_idx] + d_d
                # only assign distance if within max distance
                # AND only if closer than other direction
                if dist <= max_dist and dist < reachable_data_dist[data_idx]:
                    reachable_data[data_idx] = True
                    reachable_data_dist[data_idx] = dist

    # note that some entries will be nan values if the max distance was exceeded
    return reachable_data, reachable_data_dist, tree_preds


@njit(cache=True, nogil=True)
def local_aggregator(node_data: np.ndarray,
                     edge_data: np.ndarray,
                     node_edge_map: Dict,
                     data_map: np.ndarray,
                     distances: np.ndarray,
                     betas: np.ndarray,
                     landuse_encodings: np.ndarray = np.array([]),
                     qs: np.ndarray = np.array([]),
                     mixed_use_hill_keys: np.ndarray = np.array([]),
                     mixed_use_other_keys: np.ndarray = np.array([]),
                     accessibility_keys: np.ndarray = np.array([]),
                     cl_disparity_wt_matrix: np.ndarray = np.array(np.full((0, 0), np.nan)),
                     numerical_arrays: np.ndarray = np.array(np.full((0, 0), np.nan)),
                     angular: bool = False,
                     suppress_progress: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                                               np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    5 - in bearing
    6 - out bearing
    DATA MAP:
    0 - x
    1 - y
    2 - assigned network index - nearest
    3 - assigned network index - next-nearest
    '''
    checks.check_network_maps(node_data, edge_data, node_edge_map)
    checks.check_data_map(data_map, check_assigned=True)  # raises ValueError data points are not assigned to a network
    checks.check_distances_and_betas(distances, betas)

    # check landuse encodings
    compute_landuses = False
    if len(landuse_encodings) == 0:
        if len(mixed_use_hill_keys) != 0 or len(mixed_use_other_keys) != 0 or len(accessibility_keys) != 0:
            raise ValueError('Mixed use metrics or land-use accessibilities require an array of landuse labels.')
    elif len(landuse_encodings) != len(data_map):
        raise ValueError('The number of landuse encodings does not match the number of data points.')
    else:
        checks.check_categorical_data(landuse_encodings)

    # catch completely missing metrics
    if len(mixed_use_hill_keys) == 0 and len(mixed_use_other_keys) == 0 and len(accessibility_keys) == 0:
        if len(numerical_arrays) == 0:
            raise ValueError(
                'No metrics specified, please specify at least one metric to compute.')
    else:
        compute_landuses = True

    # catch missing qs
    if len(mixed_use_hill_keys) != 0 and len(qs) == 0:
        raise ValueError('Hill diversity measures require that at least one value of q is specified.')

    # negative qs caught by hill diversity methods

    # check various problematic key combinations
    if len(mixed_use_hill_keys) != 0:
        if (mixed_use_hill_keys.min() < 0 or mixed_use_hill_keys.max() > 3):
            raise ValueError('Mixed-use "hill" keys out of range of 0:4.')

    if len(mixed_use_other_keys) != 0:
        if (mixed_use_other_keys.min() < 0 or mixed_use_other_keys.max() > 2):
            raise ValueError('Mixed-use "other" keys out of range of 0:3.')

    if len(accessibility_keys) != 0:
        max_ac_key = landuse_encodings.max()
        if (accessibility_keys.min() < 0 or accessibility_keys.max() > max_ac_key):
            raise ValueError('Negative or out of range accessibility key encountered. Keys must match class encodings.')

    for i in range(len(mixed_use_hill_keys)):
        for j in range(len(mixed_use_hill_keys)):
            if j > i:
                i_key = mixed_use_hill_keys[i]
                j_key = mixed_use_hill_keys[j]
                if i_key == j_key:
                    raise ValueError('Duplicate mixed-use "hill" key.')

    for i in range(len(mixed_use_other_keys)):
        for j in range(len(mixed_use_other_keys)):
            if j > i:
                i_key = mixed_use_other_keys[i]
                j_key = mixed_use_other_keys[j]
                if i_key == j_key:
                    raise ValueError('Duplicate mixed-use "other" key.')

    for i in range(len(accessibility_keys)):
        for j in range(len(accessibility_keys)):
            if j > i:
                i_key = accessibility_keys[i]
                j_key = accessibility_keys[j]
                if i_key == j_key:
                    raise ValueError('Duplicate accessibility key.')

    def disp_check(disp_matrix):
        # the length of the disparity matrix vis-a-vis unique landuses is tested in underlying diversity functions
        if disp_matrix.ndim != 2 or disp_matrix.shape[0] != disp_matrix.shape[1]:
            raise ValueError('The disparity matrix must be a square NxN matrix.')
        if len(disp_matrix) == 0:
            raise ValueError('Hill disparity and Rao pairwise measures requires a class disparity weights matrix.')

    # check that missing or malformed disparity weights matrices are caught
    for k in mixed_use_hill_keys:
        if k == 3:  # hill disparity
            disp_check(cl_disparity_wt_matrix)
    for k in mixed_use_other_keys:
        if k == 2:  # raos pairwise
            disp_check(cl_disparity_wt_matrix)

    compute_numerical = False
    # when passing an empty 2d array to numba, use: np.array(np.full((0, 0), np.nan))
    if len(numerical_arrays) != 0:
        compute_numerical = True
        if numerical_arrays.shape[1] != len(data_map):
            raise ValueError('The length of the numerical data arrays do not match the length of the data map.')
        checks.check_numerical_data(numerical_arrays)

    # establish variables
    netw_n = len(node_data)
    d_n = len(distances)
    q_n = len(qs)
    n_n = len(numerical_arrays)
    global_max_dist = distances.max()
    netw_nodes_live = node_data[:, 2]

    # setup data structures
    # hill mixed uses are structured separately to take values of q into account
    mixed_use_hill_data = np.full((4, q_n, d_n, netw_n), np.nan)  # 4 dim
    mixed_use_other_data = np.full((3, d_n, netw_n), np.nan)  # 3 dim

    accessibility_data = np.full((len(accessibility_keys), d_n, netw_n), 0.0)
    accessibility_data_wt = np.full((len(accessibility_keys), d_n, netw_n), 0.0)

    # stats
    stats_sum = np.full((n_n, d_n, netw_n), np.nan)
    stats_sum_wt = np.full((n_n, d_n, netw_n), np.nan)

    stats_mean = np.full((n_n, d_n, netw_n), np.nan)
    stats_mean_wt = np.full((n_n, d_n, netw_n), np.nan)

    stats_count = np.full((n_n, d_n, netw_n), np.nan)  # use np.nan instead of 0 to avoid division by zero issues
    stats_count_wt = np.full((n_n, d_n, netw_n), np.nan)

    stats_variance = np.full((n_n, d_n, netw_n), np.nan)
    stats_variance_wt = np.full((n_n, d_n, netw_n), np.nan)

    stats_max = np.full((n_n, d_n, netw_n), np.nan)
    stats_min = np.full((n_n, d_n, netw_n), np.nan)

    # iterate through each vert and aggregate
    steps = int(netw_n / 10000)
    for netw_src_idx in range(netw_n):
        if not suppress_progress:
            checks.progress_bar(netw_src_idx, netw_n, steps)
        # only compute for live nodes
        if not netw_nodes_live[netw_src_idx]:
            continue
        # generate the reachable classes and their respective distances
        # these are non-unique - i.e. simply the class of each data point within the maximum distance
        # the aggregate_to_src_idx method will choose the closer direction of approach to a data point
        # from the nearest or next-nearest network node (calculated once globally, prior to local_landuses method)
        reachable_data, reachable_data_dist, tree_preds = aggregate_to_src_idx(netw_src_idx,
                                                                               node_data,
                                                                               edge_data,
                                                                               node_edge_map,
                                                                               data_map,
                                                                               global_max_dist,
                                                                               angular)
        # LANDUSES
        if compute_landuses:
            mu_max_unique_cl = int(landuse_encodings.max() + 1)
            # counts of each class type (array length per max unique classes - not just those within max distance)
            classes_counts = np.full((d_n, mu_max_unique_cl), 0)
            # nearest of each class type (likewise)
            classes_nearest = np.full((d_n, mu_max_unique_cl), np.inf)
            # iterate the reachable indices and related distances
            for data_idx, (reachable, data_dist) in enumerate(zip(reachable_data, reachable_data_dist)):
                if not reachable:
                    continue
                # get the class category in integer form
                # all class codes were encoded to sequential integers - these correspond to the array indices
                cl_code = int(landuse_encodings[int(data_idx)])
                # iterate the distance dimensions
                for d_idx, (d, b) in enumerate(zip(distances, betas)):
                    # increment class counts at respective distances if the distance is less than current d
                    if data_dist <= d:
                        classes_counts[d_idx, cl_code] += 1
                        # if distance is nearer, update the nearest distance array too
                        if data_dist < classes_nearest[d_idx, cl_code]:
                            classes_nearest[d_idx, cl_code] = data_dist
                        # if within distance, and if in accessibility keys, then aggregate accessibility too
                        for ac_idx, ac_code in enumerate(accessibility_keys):
                            if ac_code == cl_code:
                                accessibility_data[ac_idx, d_idx, netw_src_idx] += 1
                                accessibility_data_wt[ac_idx, d_idx, netw_src_idx] += np.exp(-b * data_dist)
                                # if a match was found, then no need to check others
                                break
            # mixed uses can be calculated now that the local class counts are aggregated
            # iterate the distances and betas
            for d_idx, b in enumerate(betas):
                cl_counts = classes_counts[d_idx]
                cl_nearest = classes_nearest[d_idx]
                # mu keys determine which metrics to compute
                # don't confuse with indices
                # previously used dynamic indices in data structures - but obtuse if irregularly ordered keys
                for mu_hill_key in mixed_use_hill_keys:
                    for q_idx, q_key in enumerate(qs):
                        if mu_hill_key == 0:
                            mixed_use_hill_data[0, q_idx, d_idx, netw_src_idx] = \
                                diversity.hill_diversity(cl_counts, q_key)
                        elif mu_hill_key == 1:
                            mixed_use_hill_data[1, q_idx, d_idx, netw_src_idx] = \
                                diversity.hill_diversity_branch_distance_wt(cl_counts, cl_nearest, q=q_key, beta=b)
                        elif mu_hill_key == 2:
                            mixed_use_hill_data[2, q_idx, d_idx, netw_src_idx] = \
                                diversity.hill_diversity_pairwise_distance_wt(cl_counts, cl_nearest, q=q_key, beta=b)
                        # land-use classification disparity hill diversity
                        # the wt matrix can be used without mapping because cl_counts is based on all classes
                        # regardless of whether they are reachable
                        elif mu_hill_key == 3:
                            mixed_use_hill_data[3, q_idx, d_idx, netw_src_idx] = \
                                diversity.hill_diversity_pairwise_matrix_wt(cl_counts,
                                                                            wt_matrix=cl_disparity_wt_matrix,
                                                                            q=q_key)
                for mu_other_key in mixed_use_other_keys:
                    if mu_other_key == 0:
                        mixed_use_other_data[0, d_idx, netw_src_idx] = \
                            diversity.shannon_diversity(cl_counts)
                    elif mu_other_key == 1:
                        mixed_use_other_data[1, d_idx, netw_src_idx] = \
                            diversity.gini_simpson_diversity(cl_counts)
                    elif mu_other_key == 2:
                        mixed_use_other_data[2, d_idx, netw_src_idx] = \
                            diversity.raos_quadratic_diversity(cl_counts, wt_matrix=cl_disparity_wt_matrix)
        # IDW
        # the order of the loops matters because the nested aggregations happen per distance per numerical array
        if compute_numerical:
            # iterate the reachable indices and related distances
            for data_idx, (reachable, data_dist) in enumerate(zip(reachable_data, reachable_data_dist)):
                # some indices will be NaN if beyond max threshold distance - so check for infinity
                # this happens when within radial max distance, but beyond network max distance
                if not reachable:
                    continue
                # iterate the numerical arrays dimension
                for num_idx in range(n_n):
                    # some values will be NaN
                    num = numerical_arrays[num_idx, int(data_idx)]
                    if np.isnan(num):
                        continue
                    # iterate the distance dimensions
                    for d_idx, (d, b) in enumerate(zip(distances, betas)):
                        # increment mean aggregations at respective distances if the distance is less than current d
                        if data_dist <= d:
                            # aggregate
                            if np.isnan(stats_sum[num_idx, d_idx, netw_src_idx]):
                                stats_sum[num_idx, d_idx, netw_src_idx] = num
                                stats_count[num_idx, d_idx, netw_src_idx] = 1
                                stats_sum_wt[num_idx, d_idx, netw_src_idx] = num * np.exp(-b * data_dist)
                                stats_count_wt[num_idx, d_idx, netw_src_idx] = np.exp(-b * data_dist)
                            else:
                                stats_sum[num_idx, d_idx, netw_src_idx] += num
                                stats_count[num_idx, d_idx, netw_src_idx] += 1
                                stats_sum_wt[num_idx, d_idx, netw_src_idx] += num * np.exp(-b * data_dist)
                                stats_count_wt[num_idx, d_idx, netw_src_idx] += np.exp(-b * data_dist)

                            if np.isnan(stats_max[num_idx, d_idx, netw_src_idx]):
                                stats_max[num_idx, d_idx, netw_src_idx] = num
                            elif num > stats_max[num_idx, d_idx, netw_src_idx]:
                                stats_max[num_idx, d_idx, netw_src_idx] = num

                            if np.isnan(stats_min[num_idx, d_idx, netw_src_idx]):
                                stats_min[num_idx, d_idx, netw_src_idx] = num
                            elif num < stats_min[num_idx, d_idx, netw_src_idx]:
                                stats_min[num_idx, d_idx, netw_src_idx] = num
            # finalise mean calculations - this is happening for a single netw_src_idx, so fairly fast
            for num_idx in range(n_n):
                for d_idx in range(d_n):
                    stats_mean[num_idx, d_idx, netw_src_idx] = \
                        stats_sum[num_idx, d_idx, netw_src_idx] / stats_count[num_idx, d_idx, netw_src_idx]
                    stats_mean_wt[num_idx, d_idx, netw_src_idx] = \
                        stats_sum_wt[num_idx, d_idx, netw_src_idx] / stats_count_wt[num_idx, d_idx, netw_src_idx]
            # calculate variances - counts are already computed per above
            # weighted version is IDW by division through equivalently weighted counts above
            # iterate the reachable indices and related distances
            for data_idx, (reachable, data_dist) in enumerate(zip(reachable_data, reachable_data_dist)):
                # some indices will be NaN if beyond max threshold distance - so check for infinity
                # this happens when within radial max distance, but beyond network max distance
                if not reachable:
                    continue
                # iterate the numerical arrays dimension
                for num_idx in range(n_n):
                    # some values will be NaN
                    num = numerical_arrays[num_idx, int(data_idx)]
                    if np.isnan(num):
                        continue
                    # iterate the distance dimensions
                    for d_idx, (d, b) in enumerate(zip(distances, betas)):
                        # increment variance aggregations at respective distances if the distance is less than current d
                        if data_dist <= d:
                            # aggregate
                            if np.isnan(stats_variance[num_idx, d_idx, netw_src_idx]):
                                stats_variance[num_idx, d_idx, netw_src_idx] = \
                                    np.square(num - stats_mean[num_idx, d_idx, netw_src_idx])
                                stats_variance_wt[num_idx, d_idx, netw_src_idx] = \
                                    np.square(num - stats_mean_wt[num_idx, d_idx, netw_src_idx]) * np.exp(-b * data_dist)
                            else:
                                stats_variance[num_idx, d_idx, netw_src_idx] += \
                                    np.square(num - stats_mean[num_idx, d_idx, netw_src_idx])
                                stats_variance_wt[num_idx, d_idx, netw_src_idx] += \
                                    np.square(num - stats_mean_wt[num_idx, d_idx, netw_src_idx]) * np.exp(-b * data_dist)
            # finalise variance calculations
            for num_idx in range(n_n):
                for d_idx in range(d_n):
                    stats_variance[num_idx, d_idx, netw_src_idx] = \
                        stats_variance[num_idx, d_idx, netw_src_idx] / stats_count[num_idx, d_idx, netw_src_idx]
                    stats_variance_wt[num_idx, d_idx, netw_src_idx] = \
                        stats_variance_wt[num_idx, d_idx, netw_src_idx] / stats_count_wt[num_idx, d_idx, netw_src_idx]
    # send the data back in the same types and same order as the original keys - convert to int for indexing
    mu_hill_k_int = np.full(len(mixed_use_hill_keys), 0)
    for i, k in enumerate(mixed_use_hill_keys):
        mu_hill_k_int[i] = k
    mu_other_k_int = np.full(len(mixed_use_other_keys), 0)
    for i, k in enumerate(mixed_use_other_keys):
        mu_other_k_int[i] = k

    return mixed_use_hill_data[mu_hill_k_int], \
           mixed_use_other_data[mu_other_k_int], \
           accessibility_data, accessibility_data_wt, \
           stats_sum, stats_sum_wt, \
           stats_mean, stats_mean_wt, \
           stats_variance, stats_variance_wt, \
           stats_max, stats_min


@njit(cache=True, nogil=True)
def singly_constrained(node_data: np.ndarray,
                       edge_data: np.ndarray,
                       node_edge_map: Dict,
                       distances: np.ndarray,
                       betas: np.ndarray,
                       i_data_map: np.ndarray,
                       j_data_map: np.ndarray,
                       i_weights: np.ndarray,
                       j_weights: np.ndarray,
                       angular: bool = False,
                       suppress_progress: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    '''
    - Calculates trips from i to j and returns the assigned trips and network assigned flows for j nodes
    #TODO: consider enhanced numerical checks for single vs. multi dimensional numerical data

    - Keeping separate from local aggregator because singly-constrained origin / destination models computed separately
    - Requires two iters, one to gather all k-nodes to per j node, then another to get the ratio of j / k attractiveness
    - Assigns j -> k trips over the network as part of second iter
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
    DATA MAP:
    0 - x
    1 - y
    2 - assigned network index - nearest
    3 - assigned network index - next-nearest
    '''
    checks.check_network_maps(node_data, edge_data, node_edge_map)
    checks.check_distances_and_betas(distances, betas)
    checks.check_data_map(i_data_map, check_assigned=True)
    checks.check_data_map(j_data_map, check_assigned=True)

    if len(i_weights) != len(i_data_map):
        raise ValueError('The i_weights array must be the same length as the i_data_map.')

    if len(j_weights) != len(j_data_map):
        raise ValueError('The j_weights array must be the same length as the j_data_map.')

    # establish variables
    netw_n = len(node_data)
    d_n = len(distances)
    global_max_dist = np.max(distances)
    netw_flows = np.full((d_n, netw_n), 0.0)

    i_n = len(i_data_map)
    k_agg = np.full((d_n, i_n), 0.0)

    j_n = len(j_data_map)
    j_assigned = np.full((d_n, j_n), 0.0)

    # iterate all i nodes
    # filter all reachable nodes k and aggregate k attractiveness * negative exponential of distance
    steps = int(i_n / 10000)
    for i_idx in range(i_n):
        if not suppress_progress:
            checks.progress_bar(i_idx, i_n, steps)
        # get the nearest node
        i_assigned_netw_idx = int(i_data_map[i_idx, 2])
        # calculate the base distance from the data point to the nearest assigned node
        i_x, i_y = i_data_map[i_idx, :2]
        n_x, n_y = node_data[i_assigned_netw_idx, :2]
        i_door_dist = np.hypot(i_x - n_x, i_y - n_y)

        # find the reachable j data points and their respective points from the closest node
        reachable_j, reachable_j_dist, tree_preds = aggregate_to_src_idx(i_assigned_netw_idx,
                                                                         node_data,
                                                                         edge_data,
                                                                         node_edge_map,
                                                                         j_data_map,
                                                                         global_max_dist,
                                                                         angular)

        # aggregate the weighted j (all k) nodes
        # iterate the reachable indices and related distances
        for j_idx, (j_reachable, j_dist) in enumerate(zip(reachable_j, reachable_j_dist)):
            if not j_reachable:
                continue
            # iterate the distance dimensions
            for d_idx, (d, b) in enumerate(zip(distances, betas)):
                total_dist = j_dist + i_door_dist
                # increment weighted k aggregations at respective distances if the distance is less than current d
                if total_dist <= d:
                    k_agg[d_idx, i_idx] += j_weights[j_idx] * np.exp(-b * total_dist)

    # this is the second step
    # this time, filter all reachable j vertices and aggregate the proportion of flow from i to j
    # this is done by dividing i-j flow through i-k_agg flow from previous step
    steps = int(i_n / 10000)
    for i_idx in range(i_n):
        if not suppress_progress:
            checks.progress_bar(i_idx, i_n, steps)

        # get the nearest node
        i_assigned_netw_idx = int(i_data_map[i_idx, 2])
        # calculate the base distance from the data point to the nearest assigned node
        i_x, i_y = i_data_map[i_idx, :2]
        n_x, n_y = node_data[i_assigned_netw_idx, :2]
        i_door_dist = np.hypot(i_x - n_x, i_y - n_y)

        # find the reachable j data points and their respective points from the closest node
        reachable_j, reachable_j_dist, tree_preds = aggregate_to_src_idx(i_assigned_netw_idx,
                                                                         node_data,
                                                                         edge_data,
                                                                         node_edge_map,
                                                                         j_data_map,
                                                                         global_max_dist,
                                                                         angular)

        # aggregate j divided through all k nodes
        # iterate the reachable indices and related distances
        for j_idx, (j_reachable, j_dist) in enumerate(zip(reachable_j, reachable_j_dist)):
            if not j_reachable:
                continue
            # iterate the distance dimensions
            for d_idx, (d, b) in enumerate(zip(distances, betas)):
                total_dist = j_dist + i_door_dist
                # if the distance is less than current d
                if total_dist <= d:
                    # aggregate all flows from reachable j's to i_idx
                    # divide through respective i-k_agg sums
                    # catch division by zero:
                    if k_agg[d_idx, i_idx] == 0:
                        assigned = 0
                    else:
                        assigned = i_weights[i_idx] * j_weights[j_idx] * np.exp(-b * total_dist) / k_agg[d_idx,
                                                                                                        i_idx]
                    j_assigned[d_idx, j_idx] += assigned
                    # assign trips to network
                    if assigned != 0:
                        # get the j assigned node
                        j_assigned_netw_idx = int(j_data_map[j_idx, 2])
                        # in this case start and end nodes are counted...!
                        netw_flows[d_idx, j_assigned_netw_idx] += assigned
                        # skip if same start / end node
                        if j_assigned_netw_idx == i_assigned_netw_idx:
                            continue
                        # aggregate to the network
                        inter_idx = np.int(tree_preds[j_assigned_netw_idx])
                        while True:
                            # end nodes counted, so place above break
                            netw_flows[d_idx, inter_idx] += assigned
                            # break out of while loop if the intermediary has reached the source node
                            if inter_idx == i_assigned_netw_idx:
                                break
                            # follow the chain
                            inter_idx = np.int(tree_preds[inter_idx])

    return j_assigned, netw_flows
