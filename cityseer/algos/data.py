import numpy as np
from numba.pycc import CC
from numba import njit
from cityseer.algos import networks


cc = CC('data')


@cc.export('assign_to_network', '(float64[:,:], float64[:,:], float64)')
@njit
def assign_to_network(data_map:np.ndarray, node_map:np.ndarray, max_dist:float) -> np.ndarray:
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
            print('...progress')
            print(round(idx / len(data_map) * 100, 2))

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


@cc.export('aggregate_to_src_idx', '(float64[:,:], float64[:,:], float64[:,:], uint64, float64, boolean)')
@njit
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
    # distance map in metres still necessary for defining max distances and computing equivalent distance measures
    map_impedance_trim, _map_distance_trim, map_pred_trim, _cycles_trim = \
        networks.shortest_path_tree(node_map, edge_map, src_idx, netw_trim_to_full_idx_map, netw_full_to_trim_idx_map,
                                    max_dist, angular)

    # filter the data by distance
    # in this case, the source x, y is the same as for the networks
    data_trim_to_full_idx_map, data_full_to_trim_idx_map = \
        networks.crow_flies(src_x, src_y, netw_x_arr, netw_y_arr, max_dist)

    # iterate the distance trimmed data point
    reachable_classes_trim = np.full(len(data_trim_to_full_idx_map), np.nan)
    reachable_classes_dist_trim = np.full(len(data_trim_to_full_idx_map), np.inf)

    for i, original_data_idx_full in enumerate(data_trim_to_full_idx_map):
        # cast to int
        original_data_idx_full = int(original_data_idx_full)
        # find the network node that it was assigned to
        assigned_network_idx_full = int(data_assign_map[original_data_idx_full])
        # get the trim idx
        assigned_network_idx_trim = int(netw_full_to_trim_idx_map[assigned_network_idx_full])
        # fetch the corresponding distance over the network
        dist = map_impedance_trim[assigned_network_idx_trim]
        # check both current and previous nodes for valid distances before continuing
        # first calculate the distance to the current node
        # in many cases, dist_calc will be np.inf, though still works for the logic below
        dist_calc = dist + data_assign_dist[original_data_idx_full]
        # get the predecessor node so that distance to prior node can be compared
        # in some cases this is closer and therefore use the closer corner, especially for full networks
        prev_netw_node_idx_trim = map_pred_trim[assigned_network_idx_trim]
        # TODO: why would they be unreachable?
        # some will be unreachable causing dist = np.inf or prev_netw_node_trim_idx = np.nan
        if not np.isfinite(prev_netw_node_idx_trim):
            # in this cases, just check whether dist_calc is less than max and continue
            if dist_calc <= max_dist:
                reachable_classes_dist_trim[i] = dist_calc
                reachable_classes_trim[i] = data_classes[original_data_idx_full]
            continue

        # otherwise, go-ahead and calculate for the prior node
        prev_netw_node_idx_trim = int(prev_netw_node_idx_trim)
        prev_netw_node_full_idx = np.int(netw_trim_to_full_idx_map[prev_netw_node_idx_trim])
        prev_dist_calc = map_impedance_trim[prev_netw_node_idx_trim] + \
                np.sqrt((netw_x_arr[prev_netw_node_full_idx] - data_x_arr[original_data_idx_full]) ** 2
                    + (netw_y_arr[prev_netw_node_full_idx] - data_y_arr[original_data_idx_full]) ** 2)
        # use the shorter distance between the current and prior nodes
        # but only if less than the maximum distance
        if dist_calc < prev_dist_calc and dist_calc <= max_dist:
            reachable_classes_dist_trim[i] = dist_calc
            reachable_classes_trim[i] = data_classes[original_data_idx_full]
        elif prev_dist_calc < dist_calc and prev_dist_calc <= max_dist:
            reachable_classes_dist_trim[i] = prev_dist_calc
            reachable_classes_trim[i] = data_classes[original_data_idx_full]

    # note that some entries will be nan values if the max distance was exceeded
    return reachable_classes_trim, reachable_classes_dist_trim
