import numpy as np
from numba import njit

from cityseer.algos import centrality, checks


# cc = CC('data')


# @cc.export('_generate_trim_to_full_map', '(float64[:], uint64)')
@njit
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


@njit
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


@njit
def nearest_idx(src_x: float, src_y: float, x_arr: np.ndarray, y_arr: np.ndarray, max_dist: float):
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


# @cc.export('assign_to_network', '(float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64)')
@njit
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
    2 - live
    3 - data class
    4 - assigned network index - nearest
    5 - assigned network index - next-nearest
    '''

    checks.check_data_map(data_map)

    checks.check_network_types(node_map, edge_map)

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

    def road_distance(data_coords, netw_idx_a, netw_idx_b):

        a_coords = node_map[netw_idx_a][:2]
        b_coords = node_map[netw_idx_b][:2]

        # get the angles from either intersection node to the data point
        ang_a = calculate_rotation_smallest(data_coords - a_coords, b_coords - a_coords)
        ang_b = calculate_rotation_smallest(data_coords - b_coords, a_coords - b_coords)

        # assume offset street segment if either is significantly greater than 90 (in which case offset from road)
        if ang_a > 110 or ang_b > 110:
            return np.inf, np.nan, np.nan

        # calculate height from two sides and included angle
        side_a = np.hypot(data_coords[0] - a_coords[0], data_coords[1] - a_coords[1])
        side_b = np.hypot(data_coords[0] - b_coords[0], data_coords[1] - b_coords[1])
        base = np.hypot(a_coords[0] - b_coords[0], a_coords[1] - b_coords[1])
        # forestall potential division by zero
        if base == 0:
            return np.inf, np.nan, np.nan
        # heron's formula
        s = (side_a + side_b + base) / 2  # perimeter / 2
        a = np.sqrt(s * (s - side_a) * (s - side_b) * (s - base))
        # area is 1/2 base * h, so h = area / (0.5 * base)
        h = a / (0.5 * base)
        # return indices in order of nearest then next nearest
        if side_a < side_b:
            return h, netw_idx_a, netw_idx_b
        else:
            return h, netw_idx_b, netw_idx_a

    def closest_intersections(data_coords, pred_map, end_node):

        if len(pred_map) == 1:
            return np.inf, end_node, np.nan

        current_idx = end_node
        next_idx = int(pred_map[int(end_node)])

        if len(pred_map) == 2:
            h, n, n_n = road_distance(data_coords, current_idx, next_idx)
            return h, n, n_n

        nearest = np.nan
        next_nearest = np.nan
        min_dist = np.inf
        first_pred = next_idx  # for finding end of loop
        while True:
            h, n, n_n = road_distance(data_coords, current_idx, next_idx)
            if h < min_dist:
                min_dist = h
                nearest = n
                next_nearest = n_n
            # if the next in the chain is nan, then break
            if np.isnan(pred_map[next_idx]):
                break
            current_idx = next_idx
            next_idx = int(pred_map[next_idx])
            if next_idx == first_pred:
                break

        return min_dist, nearest, next_nearest

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
        min_idx, min_dist = nearest_idx(data_x_arr[data_idx], data_y_arr[data_idx], netw_x_arr, netw_y_arr, max_dist)
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
            # if backtracking, the current node's predecessor will be the new neighbour
            if nb_idx != pred_map[node_idx]:

                # if the new nb node has already been visited
                # or if it is the original starting node
                if not np.isnan(pred_map[int(nb_idx)]) or nb_idx == min_idx:
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
        data_map[data_idx][4] = nearest  # adj_idx
        # in some cases next nearest will be NaN
        # this is mostly in situations where it works to leave as NaN - e.g. access off dead-ends...
        data_map[data_idx][5] = next_nearest  # next_adj_idx

    print('...done')

    return data_map


# @cc.export('aggregate_to_src_idx', '(uint64, float64, float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], boolean)')
@njit
def aggregate_to_src_idx(src_idx: int,
                         node_map: np.ndarray,
                         edge_map: np.ndarray,
                         data_map: np.ndarray,
                         max_dist: float,
                         angular: bool = False):
    netw_x_arr = node_map[:, 0]
    netw_y_arr = node_map[:, 1]
    src_x = netw_x_arr[src_idx]
    src_y = netw_y_arr[src_idx]

    d_x_arr = data_map[:, 0]
    d_y_arr = data_map[:, 1]
    d_classes = data_map[:, 3]
    d_assign_nearest = data_map[:, 4]
    d_assign_next_nearest = data_map[:, 5]

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
                                      angular=angular)

    # filter the data by distance
    # in this case, the source x, y is the same as for the networks
    data_trim_to_full_idx_map, _data_full_to_trim_idx_map = radial_filter(src_x, src_y, d_x_arr, d_y_arr, max_dist)

    # prepare the new data arrays
    reachable_classes_trim = np.full(len(data_trim_to_full_idx_map), np.nan)
    reachable_classes_dist_trim = np.full(len(data_trim_to_full_idx_map), np.inf)

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
                    reachable_classes_trim[i] = d_classes[data_idx]
                    reachable_classes_dist_trim[i] = dist

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
                if dist <= max_dist and dist < reachable_classes_dist_trim[i]:
                    reachable_classes_trim[i] = d_classes[data_idx]
                    reachable_classes_dist_trim[i] = dist

    # note that some entries will be nan values if the max distance was exceeded
    return reachable_classes_trim, reachable_classes_dist_trim, data_trim_to_full_idx_map
