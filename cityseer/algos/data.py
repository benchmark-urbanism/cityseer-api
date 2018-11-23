import numpy as np
from numba.pycc import CC
from numba import njit


cc = CC('data')


@njit
def dist_filter(cl_unique_arr, cl_counts_arr, cl_nearest_arr, max_dist):
    # first figure out how many valid items there are
    c = 0
    for i, d in enumerate(cl_nearest_arr):
        if d <= max_dist:
            c += 1
    # create trimmed arrays
    cl_unique_arr_trim = np.full(c, np.nan)
    cl_counts_arr_trim = np.full(c, 0)
    # then copy over valid data
    # don't parallelise - would cause issues
    c = 0
    for i, d in enumerate(cl_nearest_arr):
        if d <= max_dist:
            cl_unique_arr_trim[c] = cl_unique_arr[i]
            cl_counts_arr_trim[c] = cl_counts_arr[i]
            c += 1

    return cl_unique_arr_trim, cl_counts_arr_trim


@cc.export('assign_data_to_network', '(float64[:,:], float64[:,:], float64)')
@njit
def assign_data_to_network(data_map:np.ndarray, node_map:np.ndarray, max_dist:float) -> np.ndarray:
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


@njit
def aggregate_data_to_node(netw_src_idx, max_dist, netw_dist_map_trim, netw_pred_map_trim, netw_idx_map_trim_to_full,
                           netw_x_arr, netw_y_arr, data_classes, data_x_arr, data_y_arr, data_assign_map, data_assign_dist):
    # window the data
    source_x = netw_x_arr[netw_src_idx]
    source_y = netw_y_arr[netw_src_idx]
    data_trim_count, data_trim_to_full_idx_map, data_full_to_trim_idx_map = \
        crow_flies(source_x, source_y, max_dist, data_x_arr, data_y_arr)

    # iterate the distance trimmed data point
    reachable_classes_trim = np.full(data_trim_count, np.nan)
    reachable_classes_dist_trim = np.full(data_trim_count, np.inf)
    for i, original_data_idx in enumerate(data_trim_to_full_idx_map):
        # find the network node that it was assigned to
        assigned_network_idx = data_assign_map[np.int(original_data_idx)]
        # now iterate the trimmed network distances
        for j, (original_network_idx, dist) in enumerate(zip(netw_idx_map_trim_to_full, netw_dist_map_trim)):
            # no need to continue if it doesn't match the data point's assigned network node idx
            if original_network_idx != assigned_network_idx:
                continue
            # check both current and previous nodes for valid distances before continuing
            # first calculate the distance to the current node
            # in many cases, dist_calc will be np.inf, though still works for the logic below
            dist_calc = dist + data_assign_dist[np.int(original_data_idx)]
            # get the predecessor node so that distance to prior node can be compared
            # in some cases this is closer and therefore use the closer corner, especially for full networks
            prev_netw_node_trim_idx = netw_pred_map_trim[j]
            # some will be unreachable causing dist = np.inf or prev_netw_node_trim_idx = np.nan
            if not np.isfinite(prev_netw_node_trim_idx):
                # in this cases, just check whether dist_calc is less than max and continue
                if dist_calc <= max_dist:
                    reachable_classes_dist_trim[i] = dist_calc
                    reachable_classes_trim[i] = data_classes[np.int(original_data_idx)]
                continue
            # otherwise, go-ahead and calculate for the prior node
            prev_netw_node_full_idx = np.int(netw_idx_map_trim_to_full[np.int(prev_netw_node_trim_idx)])
            prev_dist_calc = netw_dist_map_trim[np.int(prev_netw_node_trim_idx)] + \
                             np.sqrt((netw_x_arr[prev_netw_node_full_idx] - data_x_arr[np.int(original_data_idx)]) ** 2
                                     + (netw_y_arr[prev_netw_node_full_idx] - data_y_arr[
                                 np.int(original_data_idx)]) ** 2)
            # use the shorter distance between the current and prior nodes
            # but only if less than the maximum distance
            if dist_calc < prev_dist_calc and dist_calc <= max_dist:
                reachable_classes_dist_trim[i] = dist_calc
                reachable_classes_trim[i] = data_classes[np.int(original_data_idx)]
            elif prev_dist_calc < dist_calc and prev_dist_calc <= max_dist:
                reachable_classes_dist_trim[i] = prev_dist_calc
                reachable_classes_trim[i] = data_classes[np.int(original_data_idx)]

    # note that some entries will be nan values if the max distance was exceeded
    # return the trim to full idx map as well in case other forms of data also need to be processed
    return reachable_classes_trim, reachable_classes_dist_trim, data_trim_to_full_idx_map


# TODO: remove this
@njit
def accessibility_agg_angular(netw_src_idx, max_dist, netw_dist_map_a_m_trim, netw_pred_map_a_trim, netw_idx_map_trim_to_full, netw_x_arr, netw_y_arr, data_classes, data_x_arr, data_y_arr, data_assign_map, data_assign_dist):

    # window the data
    source_x = netw_x_arr[netw_src_idx]
    source_y = netw_y_arr[netw_src_idx]
    data_trim_count, data_trim_to_full_idx_map, data_full_to_trim_idx_map = \
        crow_flies(source_x, source_y, max_dist, data_x_arr, data_y_arr)

    # iterate the distance trimmed data point
    reachable_classes_trim = np.full(data_trim_count, np.nan)
    reachable_classes_dist_trim = np.full(data_trim_count, np.inf)
    for i, original_data_idx in enumerate(data_trim_to_full_idx_map):
        # find the network node that it was assigned to
        assigned_network_idx = data_assign_map[np.int(original_data_idx)]
        # now iterate the trimmed network distances
        # use the angular route (simplest paths) version of distances
        for j, (original_network_idx, dist) in enumerate(zip(netw_idx_map_trim_to_full, netw_dist_map_a_m_trim)):
            # no need to continue if it doesn't match the data point's assigned network node idx
            if original_network_idx != assigned_network_idx:
                continue
            # check both current and previous nodes for valid distances before continuing
            # first calculate the distance to the current node
            # in many cases, dist_calc will be np.inf, though still works for the logic below
            dist_calc = dist + data_assign_dist[np.int(original_data_idx)]
            # get the predecessor node so that distance to prior node can be compared
            # in some cases this is closer and therefore use the closer corner, especially for full networks
            prev_netw_node_trim_idx = netw_pred_map_a_trim[j]
            # some will be unreachable causing dist = np.inf or prev_netw_node_trim_idx = np.nan
            if not np.isfinite(prev_netw_node_trim_idx):
                # in this cases, just check whether dist_calc is less than max and continue
                if dist_calc <= max_dist:
                    reachable_classes_dist_trim[i] = dist_calc
                    reachable_classes_trim[i] = data_classes[np.int(original_data_idx)]
                continue
            # otherwise, go-ahead and calculate for the prior node
            prev_netw_node_full_idx = np.int(netw_idx_map_trim_to_full[np.int(prev_netw_node_trim_idx)])
            prev_dist_calc = netw_dist_map_a_m_trim[np.int(prev_netw_node_trim_idx)] + \
                             np.sqrt((netw_x_arr[prev_netw_node_full_idx] - data_x_arr[np.int(original_data_idx)]) ** 2
                                     + (netw_y_arr[prev_netw_node_full_idx] - data_y_arr[np.int(original_data_idx)]) ** 2)
            # use the shorter distance between the current and prior nodes
            # but only if less than the maximum distance
            if dist_calc < prev_dist_calc and dist_calc <= max_dist:
                reachable_classes_dist_trim[i] = dist_calc
                reachable_classes_trim[i] = data_classes[np.int(original_data_idx)]
            elif prev_dist_calc < dist_calc and prev_dist_calc <= max_dist:
                reachable_classes_dist_trim[i] = prev_dist_calc
                reachable_classes_trim[i] = data_classes[np.int(original_data_idx)]

    # note that some entries will be nan values if the max distance was exceeded
    return reachable_classes_trim, reachable_classes_dist_trim