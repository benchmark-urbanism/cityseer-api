from typing import Tuple

import numpy as np
from numba import njit

from cityseer.algos import data, checks


@njit(cache=True)
def shortest_path_tree(
        node_map: np.ndarray,
        edge_map: np.ndarray,
        src_idx: int,
        trim_to_full_idx_map: np.ndarray,
        full_to_trim_idx_map: np.ndarray,
        max_dist: float = np.inf,
        angular: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    All shortest paths to max network distance from source node

    Returns impedances and predecessors for shortest paths from a source node to all other nodes within max distance

    Also returns a distance map based on actual distances (as opposed to impedances)

    Angular flag triggers check for sidestepping / cheating with angular impedances (sharp turns)

    NODE MAP:
    0 - x
    1 - y
    2 - live
    3 - edge index
    4 - weight

    EDGE MAP:
    0 - start node
    1 - end node
    2 - length in metres
    3 - impedance
    '''

    # this function is typically called iteratively, so do type checks from parent methods

    if src_idx >= len(node_map):
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
    src_idx_trim = np.int(full_to_trim_idx_map[src_idx])
    map_impedance[src_idx_trim] = 0
    map_distance[src_idx_trim] = 0
    # the active map is:
    # - NaN for unprocessed nodes
    # - Finite for discovered nodes within max distance
    # - Inf for discovered nodes beyond max distance
    # - Inf for any processed node
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
        # isolated nodes will have no corresponding edges
        if np.isnan(node_map[node_full_idx][3]):
            continue
        edge_idx = int(node_map[node_full_idx][3])
        # iterate the node's neighbours
        # manual iteration a tad faster than numpy methods
        # instead of while True, use length of edge map to catch last node's termination
        while edge_idx < len(edge_map):
            # get the edge's start, end, length, weight
            start, end, nb_len, nb_imp = edge_map[edge_idx]
            # if the start index no longer matches it means all neighbours have been visited for current node
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
            # don't visit predecessor nodes - otherwise successive nodes revisit out-edges to previous (neighbour) nodes
            if nb_trim_idx == map_pred[node_trim_idx]:
                continue
            # DO ANGULAR BEFORE CYCLES, AND CYCLES BEFORE DISTANCE THRESHOLD
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
            # do before distance cutoff because this node and the neighbour can respectively be within max distance
            # in some cases e.g. local_centrality(), all distances are run at once, so keep behaviour consistent by
            # designating the farthest node (but via the shortest distance) as the cycle node
            if not np.isnan(active[nb_trim_idx]):
                # set the farthest location to True
                if map_distance[nb_trim_idx] > map_distance[node_trim_idx]:
                    cycles[nb_trim_idx] = True
                else:
                    cycles[node_trim_idx] = True
            # impedance and distance is previous plus new
            impedance = map_impedance[node_trim_idx] + nb_imp
            dist = map_distance[node_trim_idx] + nb_len
            # check that the distance doesn't exceed the max
            # remember that a closer route may be found, and that this may well be within max...
            if dist > max_dist:
                continue
            # only pursue if weight distance is less than prior assigned distances
            if impedance < map_impedance[nb_trim_idx]:
                map_impedance[nb_trim_idx] = impedance
                map_distance[nb_trim_idx] = dist
                map_pred[nb_trim_idx] = node_trim_idx
                active[nb_trim_idx] = nb_trim_idx

    return map_impedance, map_distance, map_pred, cycles


@njit(cache=True)
def local_centrality(node_map: np.ndarray,
                     edge_map: np.ndarray,
                     distances: np.ndarray,
                     betas: np.ndarray,
                     closeness_keys: np.ndarray = np.array([]),
                     betweenness_keys: np.ndarray = np.array([]),
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

    checks.check_network_maps(node_map, edge_map)

    checks.check_distances_and_betas(distances, betas)

    if len(closeness_keys) == 0 and len(betweenness_keys) == 0:
        raise ValueError(
            'Neither closeness nor betweenness keys specified, please specify at least one metric to compute.')

    if len(closeness_keys) != 0 and (closeness_keys.min() < 0 or closeness_keys.max() > 6):
        raise ValueError('Closeness keys out of range of 0:6.')

    if len(betweenness_keys) != 0 and (betweenness_keys.min() < 0 or betweenness_keys.max() > 1):
        raise ValueError('Betweenness keys out of range of 0:1.')

    for i in range(len(closeness_keys)):
        for j in range(len(closeness_keys)):
            if j > i:
                i_key = closeness_keys[i]
                j_key = closeness_keys[j]
                if i_key == j_key:
                    raise ValueError('Duplicate closeness key.')

    for i in range(len(betweenness_keys)):
        for j in range(len(betweenness_keys)):
            if j > i:
                i_key = betweenness_keys[i]
                j_key = betweenness_keys[j]
                if i_key == j_key:
                    raise ValueError('Duplicate betweenness key.')

    # establish variables
    n = len(node_map)
    d_n = len(distances)
    global_max_dist = distances.max()
    x_arr = node_map[:, 0]
    y_arr = node_map[:, 1]
    nodes_live = node_map[:, 2]
    nodes_wt = node_map[:, 4]

    # prepare data arrays
    # indices correspond to different centrality formulations
    # the shortest path is based on impedances -> be cognisant of cases where impedances are not based on true distance:
    # in such cases, distances are equivalent to the impedance heuristic shortest path, not shortest distance in metres
    closeness_data = np.full((7, d_n, n), 0.0)
    betweenness_data = np.full((2, d_n, n), 0.0)

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
        trim_to_full_idx_map, full_to_trim_idx_map = data.radial_filter(src_x,
                                                                        src_y,
                                                                        x_arr,
                                                                        y_arr,
                                                                        global_max_dist)

        # run the shortest tree dijkstra
        # keep in mind that predecessor map is based on impedance heuristic - which can be different from metres
        # distance map in metres still necessary for defining max distances and computing equivalent distance measures
        map_impedance_trim, map_distance_trim, map_pred_trim, cycles_trim = \
            shortest_path_tree(node_map,
                               edge_map,
                               src_idx,
                               trim_to_full_idx_map,
                               full_to_trim_idx_map,
                               global_max_dist,
                               angular)

        # use corresponding indices for reachable verts
        ind = np.where(np.isfinite(map_distance_trim))[0]
        for to_idx_trim in ind:

            # skip self node
            if to_idx_trim == full_to_trim_idx_map[src_idx]:
                continue

            impedance = map_impedance_trim[to_idx_trim]
            dist = map_distance_trim[to_idx_trim]

            # closeness weight is based on target index -> aggregated to source index
            cl_weight = nodes_wt[int(trim_to_full_idx_map[to_idx_trim])]

            # calculate centralities starting with closeness
            if len(closeness_keys) != 0:
                for d_idx, dist_cutoff in enumerate(distances):
                    beta = betas[d_idx]
                    if dist <= dist_cutoff:
                        # closeness keys determine which metrics to compute
                        # don't confuse with indices
                        # previously used dynamic indices in data structures - but obtuse if irregularly ordered keys

                        # compute node density and farness distance regardless -> required for improved closeness
                        # in the unweighted case, weights assume 1
                        # 0 - node_density
                        # if using segment lengths weighted node then weight has the effect of converting
                        # from a node density measure to a segment length density measure
                        closeness_data[0][d_idx][src_idx] += cl_weight
                        # 2 - farness_distance - aggregated distances - not weighted
                        closeness_data[2][d_idx][src_idx] += dist
                        for cl_key in closeness_keys:
                            # 1 - farness_impedance - aggregated impedances
                            if cl_key == 1:
                                if cl_weight == 0:
                                    closeness_data[1][d_idx][src_idx] += np.inf
                                else:
                                    closeness_data[1][d_idx][src_idx] += impedance / cl_weight
                            # 3 - harmonic closeness - sum of inverse weights
                            elif cl_key == 3:
                                if impedance == 0:
                                    closeness_data[3][d_idx][src_idx] += np.inf
                                else:
                                    closeness_data[3][d_idx][src_idx] += cl_weight / impedance
                            # 4 - improved closeness - alternate closeness = node density**2 / farness aggregated weights
                            # post-computed - so ignore here
                            # 5 - gravity - sum of beta weighted distances
                            elif cl_key == 5:
                                closeness_data[5][d_idx][src_idx] += np.exp(beta * dist) * cl_weight
                            # 6 - cycles - sum of cycles weighted by equivalent distances
                            elif cl_key == 6:
                                # if there is a cycle
                                if cycles_trim[to_idx_trim]:
                                    closeness_data[6][d_idx][src_idx] += np.exp(beta * dist)

            # only process betweenness if requested
            if len(betweenness_keys) == 0:
                continue

            # only process betweenness in one direction
            if to_idx_trim < full_to_trim_idx_map[src_idx]:
                continue

            # betweenness weight is based on source and target index - assigned to each between node
            src_weight = node_map[src_idx][4]
            to_weight = node_map[int(trim_to_full_idx_map[to_idx_trim])][4]
            bt_weight = (src_weight + to_weight) / 2  # the average of the two weights

            # betweenness - only counting truly between vertices, not starting and ending verts
            inter_idx_trim = np.int(map_pred_trim[to_idx_trim])
            inter_idx_full = np.int(trim_to_full_idx_map[inter_idx_trim])  # cast to int

            while True:
                # break out of while loop if the intermediary has reached the source node
                if inter_idx_trim == full_to_trim_idx_map[src_idx]:
                    break

                for d_idx, dist_cutoff in enumerate(distances):
                    beta = betas[d_idx]
                    if dist <= dist_cutoff:
                        # betweenness map indices determine which metrics to compute
                        for bt_key in betweenness_keys:
                            # 0 - betweenness
                            if bt_key == 0:
                                betweenness_data[0][d_idx][inter_idx_full] += bt_weight
                            # 1 - betweenness_gravity - sum of gravity weighted betweenness
                            # distance is based on distance between from and to vertices
                            # thus potential gravity via between vertex
                            elif bt_key == 1:
                                betweenness_data[1][d_idx][inter_idx_full] += np.exp(beta * dist) * bt_weight

                # follow the chain
                inter_idx_trim = np.int(map_pred_trim[inter_idx_trim])
                inter_idx_full = np.int(trim_to_full_idx_map[inter_idx_trim])  # cast to int

    print('...done')

    # if improved closeness is required, then compute
    for cl_k in closeness_keys:
        if cl_k == 4:
            for d_idx in range(len(distances)):
                for src_idx in range(len(node_map)):
                    # ignore 0 / 0 situations where no proximate nodes or zero impedance
                    if closeness_data[2][d_idx][src_idx] != 0:
                        # node density squared / farness
                        closeness_data[4][d_idx][src_idx] = \
                            closeness_data[0][d_idx][src_idx] ** 2 / closeness_data[2][d_idx][src_idx]

    # send the data back in the same types and same order as the original keys - convert to int for indexing
    cl_k_int = np.full(len(closeness_keys), 0)
    for i, k in enumerate(closeness_keys):
        cl_k_int[i] = k

    bt_k_int = np.full(len(betweenness_keys), 0)
    for i, k in enumerate(betweenness_keys):
        bt_k_int[i] = k

    return closeness_data[cl_k_int], betweenness_data[bt_k_int]
