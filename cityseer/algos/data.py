import numpy as np
from numba import njit

from cityseer.algos import centrality, checks


# cc = CC('data')


#@cc.export('_generate_trim_to_full_map', '(float64[:], uint64)')
@njit
def _generate_trim_to_full_map(full_to_trim_map: np.ndarray, trim_count: int) -> np.ndarray:
    # prepare the trim to full map
    trim_to_full_idx_map = np.full(trim_count, np.nan)

    if trim_count == 0:
        return trim_to_full_idx_map

    for idx in range(len(full_to_trim_map)):
        trim_idx = full_to_trim_map[idx]
        # if the full map has a finite value, then respectively map from the trimmed index to the full index
        if not np.isnan(trim_idx):
            trim_to_full_idx_map[int(trim_idx)] = idx

    return trim_to_full_idx_map


@njit
def radial_filter(src_x: float, src_y: float, x_arr: np.ndarray, y_arr: np.ndarray, max_dist: float):
    if len(x_arr) != len(y_arr):
        raise ValueError('Mismatching x and y array lengths.')

    # filter by distance
    total_count = len(x_arr)
    full_to_trim_idx_map = np.full(total_count, np.nan)

    trim_count = 0
    for idx in range(total_count):
        dist = np.hypot(x_arr[idx] - src_x, y_arr[idx] - src_y)
        if dist <= max_dist:
            full_to_trim_idx_map[int(idx)] = trim_count
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

    for idx in range(total_count):
        dist = np.hypot(x_arr[idx] - src_x, y_arr[idx] - src_y)
        if dist <= max_dist and dist < min_dist:
            min_idx = idx
            min_dist = dist

    return min_idx, min_dist


# @cc.export('assign_to_network', '(float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64)')
# @njit
def assign_to_network(data_map: np.ndarray, node_map: np.ndarray, edge_map: np.ndarray, max_dist: float) -> np.ndarray:
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
        ang_a = np.arctan2(*point_a[::-1])
        ang_b = np.arctan2(*point_b[::-1])
        return np.rad2deg((ang_a - ang_b) % (2 * np.pi))

    def calculate_rotation_smallest(point_a, point_b):
        # smallest difference angle
        ang_a = np.rad2deg(np.arctan2(*point_a[::-1]))
        ang_b = np.rad2deg(np.arctan2(*point_b[::-1]))
        return np.abs((ang_b - ang_a + 180) % 360 - 180)

    def road_distance(data_coords, netw_idx_a, netw_idx_b):

        a_coords = node_map[netw_idx_a][:2]
        b_coords = node_map[netw_idx_b][:2]

        # get the angles from either intersection node to the data point
        ang_a = calculate_rotation_smallest(data_coords - a_coords, b_coords - a_coords)
        ang_b = calculate_rotation_smallest(data_coords - b_coords, a_coords - b_coords)

        # assume offset street segment if either angle is greater than 135 (assuming outside bend of 90º corner...)
        if ang_a > 135 or ang_b > 135:
            return np.inf, np.nan, np.nan

        # calculate height from two sides and included angle
        side_a = np.hypot(data_coords[0] - a_coords[0], data_coords[1] - a_coords[1])
        side_b = np.hypot(data_coords[0] - b_coords[0], data_coords[1] - b_coords[1])
        base = np.hypot(a_coords[0] - b_coords[0], a_coords[1] - b_coords[1])
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
            return end_node, np.nan

        if len(pred_map) == 2:
            h, nearest_idx, next_nearest_idx = road_distance(data_coords, end_node, pred_map[end_node])
            return nearest_idx, next_nearest_idx

        nearest_idx = np.nan
        next_nearest_idx = np.nan
        min_dist = np.inf

        start_idx = end_node
        current_idx = end_node
        next_idx = int(pred_map[end_node])
        while True:
            h, nearest, next = road_distance(data_coords, current_idx, next_idx)
            if h < min_dist:
                min_dist = h
                nearest_idx = nearest
                next_nearest_idx = next
            # if the next in the chain is nan, then break
            if np.isnan(pred_map[next_idx]):
                break
            current_idx = next_idx
            next_idx = int(pred_map[next_idx])
            if next_idx == start_idx:
                break

        return nearest_idx, next_nearest_idx


    pred_map = np.full(len(node_map), np.nan)

    # iterate each data point
    for data_idx in range(len(data_map)):

        # numba no object mode can only handle basic printing
        if data_idx % 10000 == 0:
            print('...progress:', round(data_idx / len(data_map) * 100, 2), '%')

        # state
        enclosed = False
        reversing = False
        end_node = np.nan
        print('DATA INDEX', data_idx)
        # keep track of visited nodes
        pred_map.fill(np.nan)
        # the data point's coordinates don't change
        data_coords = data_map[data_idx][:2]
        # get the nearest point on the network
        netw_x_arr = node_map[:, 0]
        netw_y_arr = node_map[:, 1]
        min_idx, min_dist = nearest_idx(data_coords[0], data_coords[1], netw_x_arr, netw_y_arr, max_dist)
        # set start node to nearest network node
        node_idx = int(min_idx)
        print('start:', node_idx)
        # track total rotation - distinguishes between encircling and non-encircling cycles
        total_rot = 0
        # keep track of previous indices
        prev_idx = None
        # iterate neighbours
        while True:
            # update node coordinates
            node_coords = node_map[node_idx][:2]
            # reset neighbour rotation and index counters
            nb_rot = None
            nb_idx = None
            # get the starting edge index
            edge_idx = int(node_map[node_idx][3])
            while edge_idx < len(edge_map):
                # get the edge's start and end node indices
                start, end = edge_map[edge_idx][:2]
                # if the start index no longer matches it means the node's neighbours have all been visited
                if start != node_idx:
                    break
                # increment idx for next loop
                edge_idx += 1
                # check that this isn't the previous node
                if prev_idx is not None and end == prev_idx:
                    continue
                # cast to int for indexing
                new_idx = int(end)
                # look for the new neighbour with the smallest rightwards (anti-clockwise arctan2) angle
                new_coords = node_map[new_idx][:2]
                # measure the angle relative to the data point for the first node
                if prev_idx is None:
                    r = calculate_rotation(new_coords - node_coords, data_coords - node_coords)
                else:
                    prev_coords = node_map[prev_idx][:2]
                    r = calculate_rotation(new_coords - node_coords, prev_coords - node_coords)
                if reversing:
                    r = 360 - r
                # if least angle, update
                if nb_rot is None or r < nb_rot:
                    nb_rot = r
                    nb_idx = int(new_idx)

            # allow backtracking if no neighbour is found
            if nb_idx is None:
                nb_idx = int(pred_map[node_idx])
                print('BACKTRACKING')

            # aggregate total rotation - but in clockwise direction
            nb_coords = node_map[nb_idx][:2]
            total_rot += (calculate_rotation(node_coords - data_coords, nb_coords - data_coords) + 180) % 360 - 180

            # if the distance is exceeded, reset and attempt in the other direction
            dist = np.hypot(nb_coords[0] - data_coords[0], nb_coords[1] - data_coords[1])
            if dist > max_dist:
                print('DISTANCE EXCEEDED: ', round(dist), 'm')
                end_node = nb_idx
                pred_map[nb_idx] = node_idx
                if not reversing:
                    # reverse and try in opposite direction
                    print("REVERSING")
                    print(closest_intersections(data_coords, pred_map, nb_idx))
                    reversing = True
                    pred_map.fill(np.nan)
                    node_idx = int(min_idx)
                    total_rot = 0
                    prev_idx = None
                    continue
                print(closest_intersections(data_coords, pred_map, nb_idx))
                break

            # ignore the following conditions while backtracking
            if not nb_idx == pred_map[node_idx]:

                # if the new nb node has already been visited
                if not np.isnan(pred_map[nb_idx]):
                    end_node = nb_idx
                    pred_map[nb_idx] = node_idx
                    print('ARBITRARY CYCLE', nb_idx, round(total_rot))
                    print(closest_intersections(data_coords, pred_map, nb_idx))
                    break

                # break if the original node is re-encountered after circling a block
                if nb_idx == min_idx:
                    end_node = nb_idx
                    pred_map[nb_idx] = node_idx
                    # enclosing loops will give a rotation of 360
                    rot = round(abs(total_rot))
                    if rot >= 360:  # tot
                        print("ENCLOSED", nb_idx, round(total_rot))
                        print(closest_intersections(data_coords, pred_map, nb_idx))
                        enclosed = True
                    else:
                        print("NON-ENCLOSED", nb_idx, round(total_rot))
                        print(closest_intersections(data_coords, pred_map, nb_idx))
                        # other arbitrary spin-off loops may give other totals
                    break

                # set predecessor (only if not backtracking)
                pred_map[nb_idx] = node_idx

            # otherwise, keep going
            print('nb:', nb_idx)
            prev_idx = node_idx
            node_idx = nb_idx

        # TODO: add distance cutoff
        print('')

        # if clockwise fails, try counter-clockwise

        # if encircling succeeded: iterate visited nodes and select closest
        # NOTE -> this is not necessarily the starting node if the nearest node is not on the encircling block

        # select any neighbours on the encircling (visited) route and choose the one with the shallowest adjacent street

        # if encircling failed: simply select the closest and next closest neighbour

        # set in the data map
        data_map[data_idx][4] = 0  # adj_idx
        data_map[data_idx][5] = 0  # next_adj_idx

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
        ___distance_filter(netw_index, src_x, src_y, max_dist)

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
    data_trim_to_full_idx_map, _data_full_to_trim_idx_map = ___distance_filter(data_index, src_x, src_y, max_dist)

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
