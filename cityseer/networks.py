import numpy as np
from numba.pycc import CC
from numba import njit


cc = CC('networks')

# TODO: refactor and document

# below assumes running from the aot directory
# NOTE -> this is ignored if compiling from setup.py script instead
# cc.output_dir = os.path.abspath(os.path.join(os.pardir, 'compiled'))

# Uncomment the following line to print out the compilation steps
# cc.verbose = True

# NOTE -> this is ignored if compiling from setup.py script instead
# cc.compile()
# use numba.typeof to deduce signatures
# add @njit to help aot functions find each other, see https://stackoverflow.com/questions/49326937/error-when-compiling-a-numba-module-with-function-using-other-functions-inline

@cc.export('crow_flies',
           'Tuple((i8, Array(f8, 1, "C"), Array(f8, 1, "C")))'
           '(i8, i8, f8, Array(f8, 1, "C"), Array(f8, 1, "C"))')
@njit
def crow_flies(source_x, source_y, max_dist, x_arr, y_arr):

    # filter by distance
    total_count = len(x_arr)
    crow_flies = np.full(total_count, False)
    trim_count = 0
    for i in range(total_count):
        dist = np.sqrt((x_arr[i] - source_x) ** 2 + (y_arr[i] - source_y) ** 2)
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

    return trim_count, trim_to_full_idx_map, full_to_trim_idx_map


@cc.export('graph_window',
           'Tuple((i8, f8, Array(f8, 1, "C"), Array(f8, 2, "C"), Array(f8, 2, "C")))'
           '(i8, i8, Array(f8, 1, "C"), Array(f8, 1, "C"), Array(f8, 2, "C"), Array(f8, 2, "C"))')
@njit
def graph_window(source_idx, max_dist, x_arr, y_arr, nbs_arr, lens_arr):

    # filter by distance
    source_x = x_arr[source_idx]
    source_y = y_arr[source_idx]
    trim_count, trim_to_full_idx_map, full_to_trim_idx_map = crow_flies(source_x, source_y, max_dist, x_arr, y_arr)

    # trimmed versions of network
    nbs_trim = np.full((trim_count, nbs_arr.shape[1]), np.nan)
    lens_trim = np.full((trim_count, nbs_arr.shape[1]), np.nan)
    # populate
    for i, original_idx in enumerate(trim_to_full_idx_map):
        # using count instead of enumerate because some neighbours are nan
        # this can interfere with the shortest path algorithm which breaks the loop when encountering nans
        j = 0
        # don't confuse j and n!!!
        for n, nb in enumerate(nbs_arr[np.int(original_idx)]):
            # break once all neighbours processed
            if np.isnan(nb):
                break
            # map the original neighbour index to the trimmed version
            nb_trim_idx = full_to_trim_idx_map[np.int(nb)]
            # some of the neighbours will exceed crow flies distance, in which case they have no mapping
            if np.isnan(nb_trim_idx):
                continue
            nbs_trim[i][j] = nb_trim_idx
            # lens and angles can be transferred directly
            lens_trim[i][j] = lens_arr[np.int(original_idx)][n]
            j += 1

    # get trimmed version of source index
    trim_source_idx = np.int(full_to_trim_idx_map[source_idx])

    return trim_source_idx, trim_count, trim_to_full_idx_map, nbs_trim, lens_trim


# angular has to be separate, can't overload numba jitted function without causing typing issues
@cc.export('graph_window_angular',
           'Tuple((i8, i8, Array(f8, 1, "C"), Array(f8, 2, "C"), Array(f8, 2, "C"), Array(f8, 2, "C")))'
           '(i8, f8, Array(f8, 1, "C"), Array(f8, 1, "C"), Array(f8, 2, "C"), Array(f8, 2, "C"), Array(f8, 2, "C"))')
@njit
def graph_window_angular(source_idx, max_dist, x_arr, y_arr, nbs_arr, lens_arr, angles_arr):

    # filter by network
    source_x = x_arr[source_idx]
    source_y = y_arr[source_idx]
    trim_count, trim_to_full_idx_map, full_to_trim_idx_map = crow_flies(source_x, source_y, max_dist, x_arr, y_arr)

    # trimmed versions of data
    nbs_trim = np.full((trim_count, nbs_arr.shape[1]), np.nan)
    lens_trim = np.full((trim_count, nbs_arr.shape[1]), np.nan)
    angles_trim = np.full((trim_count, nbs_arr.shape[1]), np.nan)
    # populate
    for i, original_idx in enumerate(trim_to_full_idx_map):
        # using count instead of enumerate because some neighbours are nan
        # this can interfere with the shortest path algorithm which breaks the loop when encountering nans
        j = 0
        # don't confuse j and n!!!
        for n, nb in enumerate(nbs_arr[np.int(original_idx)]):
            # break once all neighbours processed
            if np.isnan(nb):
                break
            # map the original neighbour index to the trimmed version
            nb_trim_idx = full_to_trim_idx_map[np.int(nb)]
            # some of the neighbours will exceed crow flies distance, in which case they have no mapping
            if np.isnan(nb_trim_idx):
                continue
            nbs_trim[i][j] = nb_trim_idx
            # lens and angles can be transferred directly
            lens_trim[i][j] = lens_arr[np.int(original_idx)][n]
            angles_trim[i][j] = angles_arr[np.int(original_idx)][n]
            j += 1

    # get trimmed version of source index
    trim_source_idx = np.int(full_to_trim_idx_map[source_idx])

    return trim_source_idx, trim_count, trim_to_full_idx_map, nbs_trim, lens_trim, angles_trim


@cc.export('shortest_path_tree',
           'Tuple((Array(f8, 1, "C"), Array(f8, 1, "C")))'
           '(Array(f8, 2, "C"), Array(f8, 2, "C"), i8, i8, f8)')
@njit
def shortest_path_tree(nbs_arr, dist_arr, source_idx, trim_count, max_dist):
    '''
    This is the no-frills all shortest paths to max dist from source vertex
    '''

    # setup the arrays
    active = np.full(trim_count, np.nan)
    dist_map = np.full(trim_count, np.inf)
    pred_map = np.full(trim_count, np.nan)

    # set starting node
    dist_map[source_idx] = 0
    active[source_idx] = source_idx  # store actual index number instead of booleans, easier for iteration below:

    # search to max distance threshold to determine reachable verts
    while np.any(np.isfinite(active)):
        # get the index for the min of currently active vert distances
        # note, this index corresponds only to currently active vertices
        # min_idx = np.argmin(dist_map_m[np.isfinite(active)])
        # map the min index back to the vertices array to get the corresponding vert idx
        # v = active[np.isfinite(active)][min_idx]
        # v_idx = np.int(v)  # cast to int
        # manually iterating definitely faster
        min_idx = None
        min_dist = np.inf
        for i, d in enumerate(dist_map):
            if d < min_dist and np.isfinite(active[i]):
                min_dist = d
                min_idx = i
        v_idx = np.int(min_idx)  # cast to int
        # set current vertex to visited
        active[v_idx] = np.inf
        # visit neighbours
        # for n, dist in zip(nbs_arr[v_idx], dist_arr[v_idx]):
        # manually iterating a tad faster
        for i, n in enumerate(nbs_arr[v_idx]):
            # exit once all neighbours visited
            if np.isnan(n):
                break
            n_idx = np.int(n)  # cast to int for indexing
            # distance is previous distance plus new distance
            dist = dist_arr[v_idx][i]
            d = dist_map[v_idx] + dist
            # only pursue if less than max and less than prior assigned distances
            if d <= max_dist and d < dist_map[n_idx]:
                dist_map[n_idx] = d
                pred_map[n_idx] = v_idx
                active[n_idx] = n_idx  # using actual vert idx instead of boolean to simplify finding indices

    return dist_map, pred_map


# parallel and fastmath don't apply (fastmath causes issues...)
# if using np.inf for max-dist, then handle excessive distance issues from callee function
@cc.export('shortest_path_tree_angular',
           'Tuple((Array(f8, 1, "C"), Array(f8, 1, "C"), Array(f8, 1, "C"), Array(f8, 1, "C")))'
           '(Array(f8, 2, "C"), Array(f8, 2, "C"), Array(f8, 2, "C"), i8, i8, f8)')
@njit
def shortest_path_tree_angular(nbs_arr, ang_dist_arr, dist_arr, source_idx, trim_count, max_dist):
    '''
    This is the angular variant which has more complexity:
    - using source and target version of dijkstra because there are situations where angular routes exceed max dist
    - i.e. algo searches for and quits once target reached
    - returns both angular and corresponding euclidean distances
    - checks that shortest path algorithm doesn't back-step
    '''

    # setup shortest path arrays
    active = np.full(trim_count, np.nan)
    dist_map_m = np.full(trim_count, np.inf)
    dist_map_a = np.full(trim_count, np.inf)
    dist_map_a_m = np.full(trim_count, np.inf)
    pred_map_a = np.full(trim_count, np.nan)

    # set starting node
    dist_map_a[source_idx] = 0
    dist_map_a_m[source_idx] = 0
    active[source_idx] = source_idx  # store actual index number instead of booleans, easier for iteration below:

    # search to max distance threshold to determine reachable verts
    while np.any(np.isfinite(active)):
        # get the index for the min of currently active vert distances
        # note, this index corresponds only to currently active vertices
        # min_idx = np.argmin(dist_map_a[np.isfinite(active)])
        # map the min index back to the vertices array to get the corresponding vert idx
        # v = active[np.isfinite(active)][min_idx]
        # v_idx = np.int(v)  # cast to int
        # manually iterating definitely faster
        min_idx = None
        min_ang = np.inf
        for i, a in enumerate(dist_map_a):
            if a < min_ang and np.isfinite(active[i]):
                min_ang = a
                min_idx = i
        v_idx = np.int(min_idx)  # cast to int
        # set current vertex to visited
        active[v_idx] = np.inf
        # visit neighbours
        # for n, degrees, meters in zip(nbs_arr[v_idx], ang_dist_arr[v_idx], dist_arr[v_idx]):
        # manually iterating a tad faster
        for i, n in enumerate(nbs_arr[v_idx]):
            # exit once all neighbours visited
            if np.isnan(n):
                break
            n_idx = np.int(n)  # cast to int for indexing
            # check that the neighbour node does not exceed the euclidean distance threshold
            if dist_map_m[n_idx] > max_dist:
                continue
            # check that the neighbour was not directly accessible from the prior node
            # this prevents angular backtrack-shortcutting
            # first get the previous vert's id
            pred_idx = pred_map_a[v_idx]
            # need to check for nan in case of first vertex
            if np.isfinite(pred_idx):
                # could check that previous is not equal to current neighbour but would automatically die-out...
                # don't proceed with this index if it could've been followed from predecessor
                # if np.any(nbs_arr[np.int(prev_nb_idx)] == n_idx):
                if np.any(nbs_arr[np.int(pred_idx)] == n_idx):
                    continue
            # distance is previous distance plus new distance
            degrees = ang_dist_arr[v_idx][i]
            d_a = dist_map_a[v_idx] + degrees
            meters = dist_arr[v_idx][i]
            d_m = dist_map_a_m[v_idx] + meters
            # only pursue if angular distance is less than prior assigned distance
            if d_a < dist_map_a[n_idx]:
                dist_map_a[n_idx] = d_a
                dist_map_a_m[n_idx] = d_m
                pred_map_a[n_idx] = v_idx
                active[n_idx] = n_idx

    return dist_map_m, dist_map_a, dist_map_a_m, dist_map_a


@cc.export('assign_accessibility_data',
           'Tuple((Array(f8, 1, "C"), Array(f8, 1, "C")))'
           '(Array(f8, 1, "C"), Array(f8, 1, "C"), Array(f8, 1, "C"), Array(f8, 1, "C"), f8)')
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


@cc.export('accessibility_agg',
           'Tuple((Array(f8, 1, "C"), Array(f8, 1, "C"), Array(f8, 1, "C")))'
           '(i8, f8, Array(f8, 1, "C"), Array(f8, 1, "C"), Array(f8, 1, "C"), Array(f8, 1, "C"), Array(f8, 1, "C"), Array(f8, 1, "C"), Array(f8, 1, "C"), Array(f8, 1, "C"), Array(f8, 1, "C"), Array(f8, 1, "C"))')
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


@cc.export('accessibility_agg_angular',
           'Tuple((Array(f8, 1, "C"), Array(f8, 1, "C")))'
           '(i8, f8, Array(f8, 1, "C"), Array(f8, 1, "C"), Array(f8, 1, "C"), Array(f8, 1, "C"), Array(f8, 1, "C"), Array(f8, 1, "C"), Array(f8, 1, "C"), Array(f8, 1, "C"), Array(f8, 1, "C"), Array(f8, 1, "C"))')
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
