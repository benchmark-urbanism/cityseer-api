from typing import Tuple

import numpy as np
from numba import njit

from cityseer.algos import centrality, checks, diversity


@njit(cache=True)
def _generate_trim_to_full_map(full_to_trim_map: np.ndarray, trim_count: int) -> np.ndarray:
    # prepare the trim to full map
    trim_to_full_idx_map = np.full(trim_count, np.nan)

    if trim_count == 0:
        return trim_to_full_idx_map

    for i in range(len(full_to_trim_map)):
        trim_idx = full_to_trim_map[i]
        # if the full map has a finite value, then respectively map from the trimmed index to the full index
        if not np.isnan(trim_idx):
            trim_to_full_idx_map[int(trim_idx)] = i

    return trim_to_full_idx_map


@njit(cache=True)
def radial_filter(src_x: float, src_y: float, x_arr: np.ndarray, y_arr: np.ndarray, max_dist: float):
    if len(x_arr) != len(y_arr):
        raise ValueError('Mismatching x and y array lengths.')

    # filter by distance
    total_count = len(x_arr)
    full_to_trim_idx_map = np.full(total_count, np.nan)

    trim_count = 0
    for i in range(total_count):
        dist = np.hypot(x_arr[i] - src_x, y_arr[i] - src_y)
        if dist <= max_dist:
            full_to_trim_idx_map[int(i)] = trim_count
            trim_count += 1

    trim_to_full_idx_map = _generate_trim_to_full_map(full_to_trim_idx_map, trim_count)

    return trim_to_full_idx_map, full_to_trim_idx_map


@njit(cache=True)
def find_nearest(src_x: float, src_y: float, x_arr: np.ndarray, y_arr: np.ndarray, max_dist: float):
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


@njit(cache=True)
def assign_to_network(data_map: np.ndarray,
                      node_map: np.ndarray,
                      edge_map: np.ndarray,
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
    2 - assigned network index - nearest
    3 - assigned network index - next-nearest
    '''

    checks.check_network_maps(node_map, edge_map)

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

        a_coords = node_map[netw_idx_a][:2]
        b_coords = node_map[netw_idx_b][:2]

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

    pred_map = np.full(len(node_map), np.nan)
    netw_coords = node_map[:, :2]
    netw_x_arr = node_map[:, 0]
    netw_y_arr = node_map[:, 1]
    data_coords = data_map[:, :2]
    data_x_arr = data_map[:, 0]
    data_y_arr = data_map[:, 1]

    for data_idx in range(len(data_map)):

        # numba no object mode can only handle basic printing
        if data_idx % 10000 == 0:
            print('...progress:', round(data_idx / len(data_map) * 100, 2), '%')

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
            # get the starting edge index
            # isolated nodes will have no edge index
            if np.isnan(node_map[node_idx][3]):
                break
            edge_idx = int(node_map[node_idx][3])
            while edge_idx < len(edge_map):
                # get the edge's start and end node indices
                start, end = edge_map[edge_idx][:2]
                # if the start index no longer matches it means the node's neighbours have all been visited
                if start != node_idx:
                    break
                # increment idx for next loop
                edge_idx += 1
                # check that this isn't the previous node (already visited as neighbour from other direction)
                if np.isfinite(prev_idx) and end == prev_idx:
                    continue
                # cast to int for indexing
                new_idx = int(end)
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
        data_map[data_idx][2] = nearest  # adj_idx
        # in some cases next nearest will be NaN
        # this is mostly in situations where it works to leave as NaN - e.g. access off dead-ends...
        data_map[data_idx][3] = next_nearest  # next_adj_idx

    print('...done')

    return data_map


@njit(cache=True)
def aggregate_to_src_idx(src_idx: int,
                         node_map: np.ndarray,
                         edge_map: np.ndarray,
                         data_map: np.ndarray,
                         max_dist: float,
                         angular: bool = False):
    # this function is typically called iteratively, so do type checks from parent methods

    netw_x_arr = node_map[:, 0]
    netw_y_arr = node_map[:, 1]
    src_x = netw_x_arr[src_idx]
    src_y = netw_y_arr[src_idx]

    d_x_arr = data_map[:, 0]
    d_y_arr = data_map[:, 1]
    d_assign_nearest = data_map[:, 2]
    d_assign_next_nearest = data_map[:, 3]

    # filter the network by distance
    netw_trim_to_full, netw_full_to_trim = radial_filter(src_x, src_y, netw_x_arr, netw_y_arr, max_dist)

    # run the shortest tree dijkstra
    # keep in mind that predecessor map is based on impedance heuristic - which can be different from metres
    # NOTE -> use np.inf for max distance so as to explore all paths
    # In some cases the predecessor nodes will be within reach even if the closest node is not
    # Total distance is check later
    _map_impedance_trim, map_distance_trim, _map_pred_trim, _cycles_trim = \
        centrality.shortest_path_tree(node_map,
                                      edge_map,
                                      src_idx,
                                      netw_trim_to_full,
                                      netw_full_to_trim,
                                      max_dist=max_dist,
                                      angular=angular)  # turn off checks! This is called iteratively...

    # filter the data by distance
    # in this case, the source x, y is the same as for the networks
    data_trim_to_full_idx_map, _data_full_to_trim_idx_map = radial_filter(src_x, src_y, d_x_arr, d_y_arr, max_dist)

    # prepare the new data arrays
    reachable_data_idx = np.full(len(data_trim_to_full_idx_map), np.nan)
    reachable_data_dist = np.full(len(data_trim_to_full_idx_map), np.inf)

    # iterate the distance trimmed data points
    for i, data_idx_full in enumerate(data_trim_to_full_idx_map):

        # cast to int
        data_idx = int(data_idx_full)

        # find the full and trim indices of the assigned network node
        if np.isfinite(d_assign_nearest[data_idx]):
            netw_full_idx = int(d_assign_nearest[data_idx])
            # if the assigned network node is within the threshold
            if np.isfinite(netw_full_to_trim[netw_full_idx]):
                # get the network distance
                netw_trim_idx = int(netw_full_to_trim[netw_full_idx])
                # find the distance from the nearest assigned network node to the actual location of the data point
                d_d = np.hypot(d_x_arr[data_idx] - netw_x_arr[netw_full_idx],
                               d_y_arr[data_idx] - netw_y_arr[netw_full_idx])
                # add to the distance assigned for the network node: use the distance map, not the impedance map
                dist = map_distance_trim[netw_trim_idx] + d_d
                # only assign distance if within max distance
                if dist <= max_dist:
                    reachable_data_idx[i] = data_idx
                    reachable_data_dist[i] = dist

        # the next-nearest may offer a closer route depending on the direction the shortest path approaches from
        if np.isfinite(d_assign_next_nearest[data_idx]):
            netw_full_idx = int(d_assign_next_nearest[data_idx])
            # if the assigned network node is within the threshold
            if np.isfinite(netw_full_to_trim[netw_full_idx]):
                # get the network distance
                netw_trim_idx = int(netw_full_to_trim[netw_full_idx])
                # find the distance from the nearest assigned network node to the actual location of the data point
                d_d = np.hypot(d_x_arr[data_idx] - netw_x_arr[netw_full_idx],
                               d_y_arr[data_idx] - netw_y_arr[netw_full_idx])
                # add to the distance assigned for the network node: use the distance map, not the impedance map
                dist = map_distance_trim[netw_trim_idx] + d_d
                # only assign distance if within max distance
                # AND only if closer than other direction
                if dist <= max_dist and dist < reachable_data_dist[i]:
                    reachable_data_idx[i] = data_idx
                    reachable_data_dist[i] = dist

    # note that some entries will be nan values if the max distance was exceeded
    return reachable_data_idx, reachable_data_dist, data_trim_to_full_idx_map


@njit(cache=True)
def local_aggregator(node_map: np.ndarray,
                     edge_map: np.ndarray,
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
                     angular: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                                     np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
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
    2 - assigned network index - nearest
    3 - assigned network index - next-nearest
    '''
    checks.check_network_maps(node_map, edge_map)
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
    netw_n = len(node_map)
    d_n = len(distances)
    q_n = len(qs)
    n_n = len(numerical_arrays)
    global_max_dist = distances.max()
    netw_nodes_live = node_map[:, 2]

    # setup data structures
    # hill mixed uses are structured separately to take values of q into account
    mixed_use_hill_data = np.full((4, q_n, d_n, netw_n), np.nan)  # 4 dim
    mixed_use_other_data = np.full((3, d_n, netw_n), np.nan)  # 3 dim

    accessibility_data = np.full((len(accessibility_keys), d_n, netw_n), 0.0)
    accessibility_data_wt = np.full((len(accessibility_keys), d_n, netw_n), 0.0)

    # stats
    stats_mean = np.full((n_n, d_n, netw_n), np.nan)
    stats_mean_wt = np.full((n_n, d_n, netw_n), np.nan)

    stats_count = np.full((n_n, d_n, netw_n), np.nan)  # use np.nan instead of 0 to avoid division by zero issues
    stats_count_wt = np.full((n_n, d_n, netw_n), np.nan)

    stats_variance = np.full((n_n, d_n, netw_n), np.nan)
    stats_variance_wt = np.full((n_n, d_n, netw_n), np.nan)

    stats_max = np.full((n_n, d_n, netw_n), np.nan)
    stats_min = np.full((n_n, d_n, netw_n), np.nan)

    # iterate through each vert and aggregate
    for src_idx in range(netw_n):

        # numba no object mode can only handle basic printing
        if src_idx % 10000 == 0:
            print('...progress:', round(src_idx / netw_n * 100, 2), '%')

        # only compute for live nodes
        if not netw_nodes_live[src_idx]:
            continue

        # generate the reachable classes and their respective distances
        # these are non-unique - i.e. simply the class of each data point within the maximum distance
        # the aggregate_to_src_idx method will choose the closer direction of approach to a data point
        # from the nearest or next-nearest network node (calculated once globally, prior to local_landuses method)
        reachable_data_idx, reachable_data_dist, _data_trim_to_full_idx_map = aggregate_to_src_idx(src_idx,
                                                                                                   node_map,
                                                                                                   edge_map,
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
            for i, (data_idx, data_dist) in enumerate(zip(reachable_data_idx, reachable_data_dist)):
                # some indices will be NaN if beyond max threshold distance - so check for infinity
                # this happens when within radial max distance, but beyond network max distance
                if np.isinf(data_dist):
                    continue
                # get the class category in integer form
                # all class codes were encoded to sequential integers - these correspond to the array indices
                cl_code = int(landuse_encodings[int(data_idx)])
                # iterate the distance dimensions
                for d_idx, (d, b) in enumerate(zip(distances, betas)):
                    # increment class counts at respective distances if the distance is less than current d
                    if data_dist <= d:
                        classes_counts[d_idx][cl_code] += 1
                        # if distance is nearer, update the nearest distance array too
                        if data_dist < classes_nearest[d_idx][cl_code]:
                            classes_nearest[d_idx][cl_code] = data_dist
                        # if within distance, and if in accessibility keys, then aggregate accessibility too
                        for ac_idx, ac_code in enumerate(accessibility_keys):
                            if ac_code == cl_code:
                                accessibility_data[ac_idx][d_idx][src_idx] += 1
                                accessibility_data_wt[ac_idx][d_idx][src_idx] += np.exp(b * data_dist)
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
                            mixed_use_hill_data[0][q_idx][d_idx][src_idx] = \
                                diversity.hill_diversity(cl_counts, q_key)

                        elif mu_hill_key == 1:
                            mixed_use_hill_data[1][q_idx][d_idx][src_idx] = \
                                diversity.hill_diversity_branch_distance_wt(cl_counts, cl_nearest, q=q_key, beta=b)

                        elif mu_hill_key == 2:
                            mixed_use_hill_data[2][q_idx][d_idx][src_idx] = \
                                diversity.hill_diversity_pairwise_distance_wt(cl_counts, cl_nearest, q=q_key, beta=b)

                        # land-use classification disparity hill diversity
                        # the wt matrix can be used without mapping because cl_counts is based on all classes
                        # regardless of whether they are reachable
                        elif mu_hill_key == 3:
                            mixed_use_hill_data[3][q_idx][d_idx][src_idx] = \
                                diversity.hill_diversity_pairwise_matrix_wt(cl_counts,
                                                                            wt_matrix=cl_disparity_wt_matrix,
                                                                            q=q_key)

                for mu_other_key in mixed_use_other_keys:

                    if mu_other_key == 0:
                        mixed_use_other_data[0][d_idx][src_idx] = \
                            diversity.shannon_diversity(cl_counts)

                    elif mu_other_key == 1:
                        mixed_use_other_data[1][d_idx][src_idx] = \
                            diversity.gini_simpson_diversity(cl_counts)

                    elif mu_other_key == 2:
                        mixed_use_other_data[2][d_idx][src_idx] = \
                            diversity.raos_quadratic_diversity(cl_counts, wt_matrix=cl_disparity_wt_matrix)

        # IDW
        # the order of the loops matters because the nested aggregations happen per distance per numerical array
        if compute_numerical:

            # iterate the reachable indices and related distances
            for i, (data_idx, data_dist) in enumerate(zip(reachable_data_idx, reachable_data_dist)):
                # some indices will be NaN if beyond max threshold distance - so check for infinity
                # this happens when within radial max distance, but beyond network max distance
                if np.isinf(data_dist):
                    continue

                # iterate the numerical arrays dimension
                for num_idx in range(n_n):

                    # some values will be NaN
                    num = numerical_arrays[num_idx][int(data_idx)]
                    if np.isnan(num):
                        continue

                    # iterate the distance dimensions
                    for d_idx, (d, b) in enumerate(zip(distances, betas)):

                        # increment mean aggregations at respective distances if the distance is less than current d
                        if data_dist <= d:

                            # aggregate
                            if np.isnan(stats_mean[num_idx][d_idx][src_idx]):
                                stats_mean[num_idx][d_idx][src_idx] = num
                                stats_count[num_idx][d_idx][src_idx] = 1
                                stats_mean_wt[num_idx][d_idx][src_idx] = num * np.exp(data_dist * b)
                                stats_count_wt[num_idx][d_idx][src_idx] = np.exp(data_dist * b)
                            else:
                                stats_mean[num_idx][d_idx][src_idx] += num
                                stats_count[num_idx][d_idx][src_idx] += 1
                                stats_mean_wt[num_idx][d_idx][src_idx] += num * np.exp(data_dist * b)
                                stats_count_wt[num_idx][d_idx][src_idx] += np.exp(data_dist * b)

                            if np.isnan(stats_max[num_idx][d_idx][src_idx]):
                                stats_max[num_idx][d_idx][src_idx] = num
                            elif num > stats_max[num_idx][d_idx][src_idx]:
                                stats_max[num_idx][d_idx][src_idx] = num

                            if np.isnan(stats_min[num_idx][d_idx][src_idx]):
                                stats_min[num_idx][d_idx][src_idx] = num
                            elif num < stats_min[num_idx][d_idx][src_idx]:
                                stats_min[num_idx][d_idx][src_idx] = num

            # finalise mean calculations - this is happening for a single src_idx, so fairly fast
            for num_idx in range(n_n):
                for d_idx in range(d_n):
                    stats_mean[num_idx][d_idx][src_idx] = \
                        stats_mean[num_idx][d_idx][src_idx] / stats_count[num_idx][d_idx][src_idx]
                    stats_mean_wt[num_idx][d_idx][src_idx] = \
                        stats_mean_wt[num_idx][d_idx][src_idx] / stats_count_wt[num_idx][d_idx][src_idx]

            # calculate variances - counts are already computed per above
            # weighted version is IDW by division through equivalently weighted counts above
            # iterate the reachable indices and related distances
            for i, (data_idx, data_dist) in enumerate(zip(reachable_data_idx, reachable_data_dist)):
                # some indices will be NaN if beyond max threshold distance - so check for infinity
                # this happens when within radial max distance, but beyond network max distance
                if np.isinf(data_dist):
                    continue

                # iterate the numerical arrays dimension
                for num_idx in range(n_n):

                    # some values will be NaN
                    num = numerical_arrays[num_idx][int(data_idx)]
                    if np.isnan(num):
                        continue

                    # iterate the distance dimensions
                    for d_idx, (d, b) in enumerate(zip(distances, betas)):

                        # increment variance aggregations at respective distances if the distance is less than current d
                        if data_dist <= d:

                            # aggregate
                            if np.isnan(stats_variance[num_idx][d_idx][src_idx]):
                                stats_variance[num_idx][d_idx][src_idx] = \
                                    np.square(num - stats_mean[num_idx][d_idx][src_idx])
                                stats_variance_wt[num_idx][d_idx][src_idx] = \
                                    np.square(num - stats_mean_wt[num_idx][d_idx][src_idx]) * np.exp(data_dist * b)
                            else:
                                stats_variance[num_idx][d_idx][src_idx] += \
                                    np.square(num - stats_mean[num_idx][d_idx][src_idx])
                                stats_variance_wt[num_idx][d_idx][src_idx] += \
                                    np.square(num - stats_mean_wt[num_idx][d_idx][src_idx]) * np.exp(data_dist * b)

            # finalise variance calculations
            for num_idx in range(n_n):
                for d_idx in range(d_n):
                    stats_variance[num_idx][d_idx][src_idx] = \
                        stats_variance[num_idx][d_idx][src_idx] / stats_count[num_idx][d_idx][src_idx]
                    stats_variance_wt[num_idx][d_idx][src_idx] = \
                        stats_variance_wt[num_idx][d_idx][src_idx] / stats_count_wt[num_idx][d_idx][src_idx]

    print('...done')

    # send the data back in the same types and same order as the original keys - convert to int for indexing
    mu_hill_k_int = np.full(len(mixed_use_hill_keys), 0)
    for i, k in enumerate(mixed_use_hill_keys):
        mu_hill_k_int[i] = k

    mu_other_k_int = np.full(len(mixed_use_other_keys), 0)
    for i, k in enumerate(mixed_use_other_keys):
        mu_other_k_int[i] = k

    return mixed_use_hill_data[mu_hill_k_int], \
           mixed_use_other_data[mu_other_k_int], \
           accessibility_data, \
           accessibility_data_wt, \
           stats_mean, \
           stats_mean_wt, \
           stats_variance, \
           stats_variance_wt, \
           stats_max, \
           stats_min
