import numpy as np
from numba.pycc import CC
from numba import njit


cc = CC('networks')

# below assumes running from the aot directory
# NOTE -> this is ignored if compiling from setup.py script instead
# cc.output_dir = os.path.abspath(os.path.join(os.pardir, 'compiled'))

# Uncomment the following line to print out the compilation steps
# cc.verbose = True

# NOTE -> this is ignored if compiling from setup.py script instead
# cc.compile()
# use numba.typeof to deduce signatures
# add @njit to help aot functions find each other, see https://stackoverflow.com/questions/49326937/error-when-compiling-a-numba-module-with-function-using-other-functions-inline


@cc.export('crow_flies', '(uint64, float64, float64[:], float64[:])')
@njit
def crow_flies(src_idx, max_dist, x_arr, y_arr):

    # source easting and northing
    src_x = x_arr[src_idx]
    src_y = y_arr[src_idx]

    # filter by distance
    total_count = len(x_arr)
    crow_flies = np.full(total_count, False)
    trim_count = 0
    for i in range(total_count):
        dist = np.sqrt((x_arr[i] - src_x) ** 2 + (y_arr[i] - src_y) ** 2)
        if dist <= max_dist:
            crow_flies[i] = True
            trim_count += 1

    # populate the trimmed to full index map, also populate a reverse index for mapping the neighbours
    trim_to_full_idx_map = np.full(trim_count, np.nan)
    full_to_trim_idx_map = np.full(total_count, np.nan)
    counter = 0
    for i in range(total_count):
        if crow_flies[i]:
            trim_to_full_idx_map[counter] = i
            full_to_trim_idx_map[i] = counter
            counter += 1

    return trim_to_full_idx_map, full_to_trim_idx_map


@cc.export('shortest_path_tree', '(float64[:,:], float64[:,:], uint64, float64[:], float64[:], float64, boolean)')
@njit
def shortest_path_tree(node_map, edge_map, src_idx, trim_to_full_idx_map, full_to_trim_idx_map, max_dist=np.inf, angular_wt=False):
    '''
    This is the no-frills all shortest paths to max dist from source nodes

    Returns shortest paths (distance_map_wt and pred_map) from a source node to all other nodes based on the weights
    Also returns a distance map (distance_map_m) based on actual distances

    NODE MAP:
    0 - x
    1 - y
    2 - live
    3 - edge indx

    EDGE MAP:
    0 - start node
    1 - end node
    2 - length in metres
    3 - weight
    '''

    # setup the arrays
    n_trim = len(trim_to_full_idx_map)
    active = np.full(n_trim, np.nan)  # whether the node is currently active or not
    dist_map_wt = np.full(n_trim, np.inf)  # the distance map based on the weights attribute - not necessarily metres
    dist_map_m = np.full(n_trim, np.inf)  # the distance map based on the metres distance attribute
    pred_map = np.full(n_trim, np.nan)  # predecessor map
    cycles = np.full(n_trim, False)  # graph cycles

    # set starting node
    # the active map is:
    # - NaN for unprocessed nodes
    # - set to idx of node once discovered
    # - set to Inf once processed
    src_idx_trim = np.int(full_to_trim_idx_map[src_idx])
    dist_map_wt[src_idx_trim] = 0
    dist_map_m[src_idx_trim] = 0
    active[src_idx_trim] = src_idx_trim

    # search to max distance threshold to determine reachable nodes
    while np.any(np.isfinite(active)):

        # get the index for the min of currently active node distances
        # note, this index corresponds only to currently active vertices

        # find the currently closest unprocessed node
        # manual iteration definitely faster than numpy methods
        min_idx = None
        min_dist = np.inf
        for i, d in enumerate(dist_map_wt):
            # find any closer nodes that have not yet been discovered
            if d < min_dist and np.isfinite(active[i]):
                min_dist = d
                min_idx = i
        # cast to int - do this step explicitly for numba type inference
        trim_idx = int(min_idx)
        # the node can now be set to visited
        active[trim_idx] = np.inf
        # convert the idx to the full node_map
        node_idx = int(trim_to_full_idx_map[trim_idx])
        # fetch the relevant edge_map index
        edge_idx = int(node_map[node_idx][3])
        # iterate the node's neighbours
        # manual iteration a tad faster than numpy methods
        # instead of while True use length of edge map to catch last node's termination
        while edge_idx < len(edge_map):
            # get the edge's start, end, length, weight
            start, end, nb_len, nb_wt = edge_map[edge_idx]
            # if the start index no longer matches it means all neighbours have been visited
            if start != node_idx:
                break
            # increment idx for next loop
            edge_idx += 1
            # cast to int for indexing
            nb_idx = np.int(end)
            # not all neighbours will be within crow-flies distance - if so, continue
            if np.isnan(full_to_trim_idx_map[nb_idx]):
                continue
            # fetch the neighbour's trim index
            nb_idx_trim = int(full_to_trim_idx_map[nb_idx])
            # if this neighbour has already been processed, continue
            if np.isinf(active[nb_idx_trim]):
                continue
            # distance is previous distance plus new distance
            d_w = dist_map_wt[trim_idx] + nb_wt
            d_m = dist_map_m[trim_idx] + nb_len
            # check that the distance doesn't exceed the max
            if d_m > max_dist:
                continue
            # it is necessary to check for angular sidestepping if using angular weights on a dual graph
            if angular_wt:
                prior_match = False
                # get the predecessor
                pred_idx = np.int(pred_map[node_idx])
                # check that the new neighbour was not directly accessible from the prior set of neighbours
                pred_edge_idx = int(node_map[pred_idx][3])
                while pred_edge_idx < len(edge_map):
                    # get the previous edge's start and end nodes
                    pred_start, pred_end = edge_map[pred_edge_idx][:2]
                    # if the prev start index no longer matches prev node, all previous neighbours have been visited
                    if pred_start != pred_idx:
                        break
                    # increment idx for next loop
                    edge_idx += 1
                    # check that the previous node's neighbour's node is not equal to the currently new neighbour node
                    if pred_end == nb_idx:
                        prior_match = True
                        break
                # continue if prior match was found
                if prior_match:
                    continue
            # if a neighbouring node has already been discovered, then it is a cycle
            # predecessor neighbour nodes are already filtered out above with: np.isinf(active[nb_idx_trim])
            # so np.isnan is adequate -> can only run into other active nodes - not completed nodes
            if not np.isnan(active[nb_idx_trim]):
                cycles[nb_idx_trim] = True
            # only pursue if weight distance is less than prior assigned distances
            if d_w < dist_map_wt[nb_idx_trim]:
                dist_map_wt[nb_idx_trim] = d_w
                dist_map_m[nb_idx_trim] = d_m
                # using actual node indices instead of boolean to simplify finding indices
                pred_map[nb_idx_trim] = trim_idx
                active[nb_idx_trim] = nb_idx_trim

    return dist_map_wt, dist_map_m, pred_map, cycles


# NOTE -> didn't work with boolean so using unsigned int...
@cc.export('compute_centrality', '(float64[:,:], float64[:,:], float64[:], float64[:])')
@njit
def compute_centrality(node_map, edge_map, distances, betas):
    '''
    NODE MAP:
    0 - x
    1 - y
    2 - live
    3 - edge indx

    EDGE MAP:
    0 - start node
    1 - end node
    2 - length in metres
    3 - weight
    '''

    # establish the number of nodes
    n = len(node_map)

    # the distances dimension
    d_n = len(distances)

    # maximum distance
    max_dist = distances.max()

    # disaggregate node_map
    x_arr = node_map[:,0]
    y_arr = node_map[:,1]
    nodes_live = node_map[:,2]

    # prepare data arrays
    node_density = np.full((d_n, n), 0.0)
    farness = np.full((d_n, n), 0.0)
    harmonic = np.full((d_n, n), 0.0)
    gravity = np.full((d_n, n), 0.0)
    betweenness = np.full((d_n, n), 0.0)
    betweenness_wt = np.full((d_n, n), 0.0)
    cycle_counts = np.full((d_n, n), 0.0)

    # iterate through each vert and calculate the shortest path tree
    for src_idx in range(n):

        if src_idx % 100000 == 0:
            print('...progress')
            print(round(src_idx / n * 100, 2))

        # only compute for nodes in current city
        if not nodes_live[src_idx]:
            continue

        # filter the graph by distance
        trim_to_full_idx_map, full_to_trim_idx_map = crow_flies(src_idx, max_dist, x_arr, y_arr)

        # run the shortest tree dijkstra
        # keep in mind that predecessor map is based on weights distance - which can be different from metres
        # distance map in metres still necessary for defining max distances
        dist_map_trim_wt, dist_map_trim_m, pred_map_trim, cycles = \
            shortest_path_tree(node_map, edge_map, src_idx, trim_to_full_idx_map, full_to_trim_idx_map, max_dist)

        # use corresponding indices for reachable verts
        ind = np.where(np.isfinite(dist_map_trim_m))[0]
        for to_idx_trim in ind:

            # skip self node
            if to_idx_trim == full_to_trim_idx_map[src_idx]:
                continue

            dist_m = dist_map_trim_m[to_idx_trim]

            # some crow-flies max distance nodes won't be reached within max distance threshold over the network
            if np.isinf(dist_m):
                continue

            # check here for distance - in case max distance in shortest_path_tree is set to infinity
            if dist_m > max_dist:
                continue

            # calculate centralities starting with closeness
            for i in range(len(distances)):
                d = distances[i]
                b = betas[i]

                # these arrays are oriented differently - first d index then src index
                if dist_m <= d:
                    node_density[i][src_idx] += 1
                    farness[i][src_idx] += dist_m
                    harmonic[i][src_idx] += 1/dist_m
                    gravity[i][src_idx] += np.exp(b * dist_m)
                    # check if the current to node is a cycle
                    if cycles[to_idx_trim]:
                        cycle_counts[i][src_idx] += np.exp(b * dist_m)

            # only process betweenness in one direction
            if to_idx_trim < full_to_trim_idx_map[src_idx]:
                continue

            # betweenness - only counting truly between vertices, not starting and ending verts
            intermediary_idx_trim = np.int(pred_map_trim[to_idx_trim])
            intermediary_idx_mapped = np.int(trim_to_full_idx_map[intermediary_idx_trim])  # cast to int
            # only counting betweenness in one 'direction' since the graph is symmetrical (non-directed)
            while True:
                # break out of while loop if the intermediary has reached the source node
                if intermediary_idx_trim == full_to_trim_idx_map[src_idx]:
                    break

                for i in range(len(distances)):
                    d = distances[i]
                    b = betas[i]

                    if dist_m <= d:

                        # weighted variants - summed at all distances
                        betweenness[i][intermediary_idx_mapped] += 1
                        betweenness_wt[i][intermediary_idx_mapped] += np.exp(b * dist_m)

                # follow the chain
                intermediary_idx_trim = np.int(pred_map_trim[intermediary_idx_trim])
                intermediary_idx_mapped = np.int(trim_to_full_idx_map[intermediary_idx_trim])  # cast to int

    return node_density, farness, harmonic, gravity, betweenness, betweenness_wt, cycle_counts


"""
@njit
def assign_accessibility_data(network_x_arr, network_y_arr, data_x_arr, data_y_arr, max_dist):
    '''
    assign data from an x, y array of data point (e.g. landuses)
    to the nearest corresponding point on an x, y array from a network

    This is done once for the whole graph because it only requires a one-dimensional array

    i.e. the similar crow-flies step for the network graph windowing has to happen inside the nested iteration
    because it would require an NxM matrix if done here - which is memory prohibitive
    '''

    verts_count = len(network_x_arr)
    data_count = len(data_x_arr)
    # prepare the arrays for tracking the respective nearest vertex
    data_assign_map = np.full(data_count, np.nan)
    # and the corresponding distance
    data_assign_dist = np.full(data_count, np.nan)
    # iterate each data point
    for data_idx in range(data_count):
        # iterate each network id
        for network_idx in range(verts_count):
            # get the distance
            dist = np.sqrt((network_x_arr[network_idx] - data_x_arr[data_idx]) ** 2 + (
                        network_y_arr[network_idx] - data_y_arr[data_idx]) ** 2)
            # only proceed if it is less than the max dist cutoff
            if dist > max_dist:
                continue
            # if within the cutoff distance
            # and if no adjacent network point has yet been assigned for this data point
            # then proceed to record this adjacency and the corresponding distance
            elif np.isnan(data_assign_dist[data_idx]):
                data_assign_dist[data_idx] = dist
                data_assign_map[data_idx] = network_idx
            # otherwise, only update if the new distance is less than any prior distances
            elif dist < data_assign_dist[data_idx]:
                data_assign_dist[data_idx] = dist
                data_assign_map[data_idx] = network_idx

    return data_assign_map, data_assign_dist


@njit
def accessibility_agg(netw_src_idx, max_dist, netw_dist_map_trim, netw_pred_map_trim, netw_idx_map_trim_to_full, netw_x_arr, netw_y_arr, data_classes, data_x_arr, data_y_arr, data_assign_map, data_assign_dist):

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
    # return the trim to full idx map as well in case other forms of data also need to be processed
    return reachable_classes_trim, reachable_classes_dist_trim, data_trim_to_full_idx_map


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
    
"""