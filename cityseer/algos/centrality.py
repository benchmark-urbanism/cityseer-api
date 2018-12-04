from typing import Tuple

import numpy as np
from numba import njit

from cityseer.algos import data, checks


# cc = CC('centrality')


# @cc.export('shortest_path_tree', '(float64[:,:], float64[:,:], uint64, float64[:], float64[:], float64, boolean)')
@njit
def shortest_path_tree(
        node_map: np.ndarray,
        edge_map: np.ndarray,
        src_idx: int,
        trim_to_full_idx_map: np.ndarray,
        full_to_trim_idx_map: np.ndarray,
        max_dist: float = np.inf,
        angular: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    This is the no-frills all shortest paths to max dist from source nodes

    Returns shortest paths (map_impedance and map_pred) from a source node to all other nodes based on the impedances
    Also returns a distance map (map_distance) based on actual distances

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
    '''

    checks.check_network_types(node_map, edge_map)

    if not src_idx < len(node_map):
        raise ValueError('Source index is out of range.')

    if len(full_to_trim_idx_map) != len(node_map):
        raise ValueError('Mismatching lengths for node map and trim maps.')

    # setup the arrays
    n_trim = len(trim_to_full_idx_map)
    active = np.full(n_trim, np.nan)  # whether the node is currently active or not
    # in many cases the weight and distance maps will be the same, but not necessarily so
    map_impedance = np.full(n_trim, np.inf)  # the distance map based on the weights attribute - not necessarily metres
    map_distance = np.full(n_trim, np.inf)  # the distance map based on the metres distance attribute
    map_pred = np.full(n_trim, np.nan)  # predecessor map
    cycles = np.full(n_trim, False)  # graph cycles

    # set starting node
    # the active map is:
    # - NaN for unprocessed nodes
    # - set to idx of node once discovered
    # - set to Inf once processed
    src_idx_trim = np.int(full_to_trim_idx_map[src_idx])
    map_impedance[src_idx_trim] = 0
    map_distance[src_idx_trim] = 0
    active[src_idx_trim] = src_idx_trim

    # search to max distance threshold to determine reachable nodes
    while np.any(np.isfinite(active)):

        # get the index for the min of currently active node distances
        # note, this index corresponds only to currently active vertices

        # find the currently closest unprocessed node
        # manual iteration definitely faster than numpy methods
        min_idx = None
        min_dist = np.inf
        for i, d in enumerate(map_impedance):
            # find any closer nodes that have not yet been discovered
            if d < min_dist and np.isfinite(active[i]):
                min_dist = d
                min_idx = i
        # cast to int - do this step explicitly for numba type inference
        node_trim_idx = int(min_idx)
        # the node can now be set to visited
        active[node_trim_idx] = np.inf
        # convert the idx to the full node_map
        node_full_idx = int(trim_to_full_idx_map[node_trim_idx])
        # fetch the relevant edge_map index
        edge_idx = int(node_map[node_full_idx][3])
        # iterate the node's neighbours
        # manual iteration a tad faster than numpy methods
        # instead of while True use length of edge map to catch last node's termination
        while edge_idx < len(edge_map):
            # get the edge's start, end, length, weight
            start, end, nb_len, nb_imp = edge_map[edge_idx]
            # if the start index no longer matches it means all neighbours have been visited
            if start != node_full_idx:
                break
            # increment idx for next loop
            edge_idx += 1
            # cast to int for indexing
            nb_full_idx = np.int(end)
            # not all neighbours will be within crow-flies distance - if so, continue
            if np.isnan(full_to_trim_idx_map[nb_full_idx]):
                continue
            # fetch the neighbour's trim index
            nb_trim_idx = int(full_to_trim_idx_map[nb_full_idx])
            # if this neighbour has already been processed, continue
            # i.e. successive nodes would recheck predecessor (neighbour) nodes unnecessarily
            if np.isinf(active[nb_trim_idx]):
                continue
            # distance is previous distance plus new distance
            impedance = map_impedance[node_trim_idx] + nb_imp
            dist = map_distance[node_trim_idx] + nb_len
            # check that the distance doesn't exceed the max
            if dist > max_dist:
                continue
            # it is necessary to check for angular sidestepping if using angular weights on a dual graph
            # only do this for angular graphs, and only if predecessors exist
            if angular and not np.isnan(map_pred[node_trim_idx]):
                prior_match = False
                # get the predecessor
                pred_trim_idx = int(map_pred[node_trim_idx])
                # convert to full index
                pred_full_idx = int(trim_to_full_idx_map[pred_trim_idx])
                # check that the new neighbour was not directly accessible from the prior set of neighbours
                pred_edge_idx = int(node_map[pred_full_idx][3])
                while pred_edge_idx < len(edge_map):
                    # get the previous edge's start and end nodes
                    pred_start, pred_end = edge_map[pred_edge_idx][:2]
                    # if the prev start index no longer matches prev node, all previous neighbours have been visited
                    if pred_start != pred_full_idx:
                        break
                    # increment predecessor idx for next loop
                    pred_edge_idx += 1
                    # check that the previous node's neighbour's node is not equal to the currently new neighbour node
                    if pred_end == nb_full_idx:
                        prior_match = True
                        break
                # continue if prior match was found
                if prior_match:
                    continue
            # if a neighbouring node has already been discovered, then it is a cycle
            # predecessor neighbour nodes are already filtered out above with: np.isinf(active[nb_trim_idx])
            # so np.isnan is adequate -> can only run into other active nodes - not completed nodes
            if not np.isnan(active[nb_trim_idx]):
                cycles[nb_trim_idx] = True
            # only pursue if weight distance is less than prior assigned distances
            if impedance < map_impedance[nb_trim_idx]:
                map_impedance[nb_trim_idx] = impedance
                map_distance[nb_trim_idx] = dist
                # using actual node indices instead of boolean to simplify finding indices
                map_pred[nb_trim_idx] = node_trim_idx
                active[nb_trim_idx] = nb_trim_idx

    return map_impedance, map_distance, map_pred, cycles


# @cc.export('local_centrality', '(float64[:,:], float64[:,:], float64[:], float64[:], uint64[:], uint64[:], boolean)')
@njit
def local_centrality(node_map: np.ndarray,
                     edge_map: np.ndarray,
                     distances: np.ndarray,
                     betas: np.ndarray,
                     closeness_keys: np.ndarray,
                     betweenness_keys: np.ndarray,
                     angular: bool = False) -> Tuple[np.ndarray, np.ndarray]:
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
    '''

    if len(closeness_keys) == 0 and len(betweenness_keys) == 0:
        raise ValueError(
            'Neither closeness nor betweenness keys specified, please specify at least one metric to compute.')

    checks.check_network_types(node_map, edge_map)

    checks.check_distances_and_betas(distances, betas)

    # establish variables
    n = len(node_map)
    d_n = len(distances)
    max_dist = distances.max()
    x_arr = node_map[:, 0]
    y_arr = node_map[:, 1]
    nodes_live = node_map[:, 2]

    # prepare data arrays
    # indices correspond to different centrality formulations
    # the shortest path is based on impedances -> be cognisant of cases where impedances are not based on true distance:
    # in such cases, distances are equivalent to the impedance heuristic shortest path, not shortest distance in metres
    closeness_data = np.full((7, d_n, n), 0.0)
    betweenness_data = np.full((2, d_n, n), 0.0)

    # CLOSENESS MEASURES
    def closeness_metrics(idx, impedance, distance, weight, beta, is_cycle):
        # in the unweighted case, weight assumes 1
        # 0 - node_density
        # if using segment lengths per node -> converts from a node density measure to a segment length density measure
        if idx == 0:
            return weight
        # 1 - farness_impedance - aggregated impedances
        elif idx == 1:
            return impedance / weight
        # 2 - farness_distance - aggregated distances - not weighted
        elif idx == 2:
            return distance
        # 3 - harmonic closeness - sum of inverse weights
        elif idx == 3:
            if impedance == 0:
                return np.inf
            else:
                return weight / impedance
        # 4 - improved closeness - alternate closeness = node density**2 / farness aggregated weights (post calculated)
        # post-computed - so return 0
        elif idx == 4:
            return 0
        # 5 - gravity - sum of beta weighted distances
        elif idx == 5:
            return np.exp(beta * distance) * weight
        # 6 - cycles - sum of cycles weighted by equivalent distances
        elif idx == 6:
            if is_cycle:
                return np.exp(beta * distance)
            else:
                return 0
        if idx > 6:
            raise ValueError('Closeness key exceeds the available options.')

    # BETWEENNESS MEASURES
    def betweenness_metrics(idx, distance, weight, beta):
        # 0 - betweenness
        if idx == 0:
            return weight
        # 1 - betweenness_gravity - sum of gravity weighted betweenness
        elif idx == 1:
            return np.exp(beta * distance) * weight
        if idx > 1:
            raise ValueError('Betweenness key exceeds the available options.')

    # iterate through each vert and calculate the shortest path tree
    for src_idx in range(n):

        # numba no object mode can only handle basic printing
        if src_idx % 10000 == 0:
            print('...progress:', round(src_idx / n * 100, 2), '%')

        # only compute for live nodes
        if not nodes_live[src_idx]:
            continue

        # filter the graph by distance
        src_x = x_arr[src_idx]
        src_y = y_arr[src_idx]
        trim_to_full_idx_map, full_to_trim_idx_map = data.radial_filter(src_x, src_y, x_arr, y_arr, max_dist)

        # run the shortest tree dijkstra
        # keep in mind that predecessor map is based on impedance heuristic - which can be different from metres
        # distance map in metres still necessary for defining max distances and computing equivalent distance measures
        map_impedance_trim, map_distance_trim, map_pred_trim, cycles_trim = \
            shortest_path_tree(node_map,
                               edge_map,
                               src_idx,
                               trim_to_full_idx_map,
                               full_to_trim_idx_map,
                               max_dist,
                               angular)

        # use corresponding indices for reachable verts
        ind = np.where(np.isfinite(map_distance_trim))[0]
        for to_idx_trim in ind:

            # skip self node
            if to_idx_trim == full_to_trim_idx_map[src_idx]:
                continue

            impedance = map_impedance_trim[to_idx_trim]
            distance = map_distance_trim[to_idx_trim]

            # some crow-flies max distance nodes won't be reached within max distance threshold over the network
            if np.isinf(distance):
                continue

            # check here for distance - in case max distance in shortest_path_tree is set to infinity
            if distance > max_dist:
                continue

            # closeness weight is based on target index -> aggregated to source index
            cl_weight = node_map[int(trim_to_full_idx_map[to_idx_trim])][4]

            # calculate centralities starting with closeness
            for i in range(len(distances)):
                d = distances[i]
                b = betas[i]
                if distance <= d:
                    is_cycle = cycles_trim[to_idx_trim]
                    # closeness map indices determine which metrics to compute
                    for cl_idx in closeness_keys:
                        closeness_data[int(cl_idx)][i][src_idx] += \
                            closeness_metrics(int(cl_idx), impedance, distance, cl_weight, b, is_cycle)

            # only process betweenness if requested
            if len(betweenness_keys) == 0:
                continue

            # betweenness weight is based on source and target index - assigned to each between node
            src_weight = node_map[src_idx][4]
            to_weight = node_map[int(trim_to_full_idx_map[to_idx_trim])][4]
            bt_weight = (src_weight + to_weight) / 2  # the average of the two weights

            # only process betweenness in one direction
            if to_idx_trim < full_to_trim_idx_map[src_idx]:
                continue

            # betweenness - only counting truly between vertices, not starting and ending verts
            intermediary_idx_trim = np.int(map_pred_trim[to_idx_trim])
            intermediary_idx_mapped = np.int(trim_to_full_idx_map[intermediary_idx_trim])  # cast to int

            while True:
                # break out of while loop if the intermediary has reached the source node
                if intermediary_idx_trim == full_to_trim_idx_map[src_idx]:
                    break

                for i in range(len(distances)):
                    d = distances[i]
                    b = betas[i]
                    if distance <= d:
                        for bt_idx in betweenness_keys:
                            betweenness_data[int(bt_idx)][i][intermediary_idx_mapped] += \
                                betweenness_metrics(int(bt_idx), distance, bt_weight, b)

                # follow the chain
                intermediary_idx_trim = np.int(map_pred_trim[intermediary_idx_trim])
                intermediary_idx_mapped = np.int(trim_to_full_idx_map[intermediary_idx_trim])  # cast to int

    print('completed')

    # improved closeness is post-computed
    for cl_idx in closeness_keys:
        if cl_idx != 4:
            continue
        for d_idx in range(len(closeness_data[4])):
            for p_idx in range(len(closeness_data[4][d_idx])):
                # ignore 0 / 0 situations where no proximate nodes or zero impedance
                if closeness_data[2][d_idx][p_idx] != 0:
                    closeness_data[4][d_idx][p_idx] = closeness_data[0][d_idx][p_idx] ** 2 / closeness_data[2][d_idx][
                        p_idx]

    return closeness_data, betweenness_data
