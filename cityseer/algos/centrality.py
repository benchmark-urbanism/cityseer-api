from typing import Tuple

import numpy as np
from numba import njit

from cityseer.algos import data, checks


@njit(cache=True)
def netw_radial_filter(src_idx: int, node_map: np.ndarray, max_dist: float):
    '''
    This version is customised for the network and will ignore ghosted nodes (unless source node)

    NODE MAP:
    0 - x
    1 - y
    2 - live
    3 - edge index
    4 - ghosted
    '''

    x_arr = node_map[:, 0]
    y_arr = node_map[:, 1]
    ghosted = node_map[:, 4]
    src_x = x_arr[src_idx]
    src_y = y_arr[src_idx]

    # filter by distance
    total_count = len(x_arr)
    full_to_trim_idx_map = np.full(total_count, np.nan)
    trim_count = 0
    for i in range(total_count):
        # ignore ghosted (except source)
        if ghosted[i] and not i == src_idx:
            continue
        # all other nodes filtered by distance
        dist = np.hypot(x_arr[i] - src_x, y_arr[i] - src_y)
        if dist <= max_dist:
            full_to_trim_idx_map[int(i)] = trim_count
            trim_count += 1
    trim_to_full_map = data._generate_trim_to_full_map(full_to_trim_idx_map, trim_count)

    return trim_to_full_map, full_to_trim_idx_map


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
    4 - ghosted

    EDGE MAP:
    0 - start node
    1 - end node
    2 - length in metres
    3 - sum of angular travel along length
    4 - impedance factor
    5 - entry bearing
    6 - exit bearing
    '''

    # this function is typically called iteratively, so do type checks from parent methods
    if len(full_to_trim_idx_map) != len(node_map):
        raise ValueError('Mismatching lengths for node map and trim maps.')

    # setup the arrays
    n_trim = len(trim_to_full_idx_map)
    active = np.full(n_trim, np.nan)  # whether the node is currently active or not
    # in many cases the impedance and distance maps will be the same, but not necessarily so
    map_impedance = np.full(n_trim, np.inf)  # the distance map based on the impedances attribute - not necessarily metres
    map_distance = np.full(n_trim, np.inf)  # the distance map based on the metres distance attribute
    map_pred = np.full(n_trim, np.nan)  # predecessor map - have to use trimmed index to follow predecessors
    cycles = np.full(n_trim, False)  # graph cycles

    # set starting node
    src_idx_trim = int(full_to_trim_idx_map[src_idx])
    map_impedance[src_idx_trim] = 0
    map_distance[src_idx_trim] = 0
    '''
    The active map is:
    - np.nan for inactive (and unprocessed nodes)
    - 1 for activated but unprocessed nodes
    - np.inf for processed nodes
    '''
    active[src_idx_trim] = 1

    # this loops continues until all nodes within the max distance have been discovered and processed
    while np.any(np.isfinite(active)):

        # get the index for the min of currently active node distances
        # note, this index corresponds only to currently active vertices

        # find the currently closest unprocessed node
        # manual iteration definitely faster than numpy methods
        min_idx = None
        min_dist = np.inf
        for i, d in enumerate(map_impedance):
            # find any closer nodes that have not yet been processed
            if d < min_dist and np.isfinite(active[i]):
                min_dist = d
                min_idx = i
        # cast to int - do this step explicitly for numba type inference
        active_nd_idx_trim = int(min_idx)
        active_nd_idx_full = int(trim_to_full_idx_map[active_nd_idx_trim])
        # the node can now be set to visited
        active[active_nd_idx_trim] = np.inf
        # fetch the relevant edge_map index
        # isolated nodes will have no corresponding edges
        if np.isnan(node_map[active_nd_idx_full][3]):
            continue
        edge_idx = int(node_map[active_nd_idx_full][3])
        # iterate the node's neighbours
        # manual iteration a tad faster than numpy methods
        # instead of while True, use length of edge map to catch last node's termination
        while edge_idx < len(edge_map):
            # get the edge's properties
            start_nd_full, end_nd_full, seg_len, seg_ang, seg_imp_fact, seg_en_bear, seg_ex_bear = edge_map[edge_idx]
            # if the start index no longer matches it means all neighbours have been visited for current node
            if start_nd_full != active_nd_idx_full:
                break
            # increment idx for next loop
            edge_idx += 1
            #TODO: clip edges - how to add virtual clipped node to fixed length array?
            #TODO: clipping is just leftover until max distance...
            # not all neighbours within crow-flies distance
            # cast nb node to int for indexing
            if np.isnan(full_to_trim_idx_map[int(end_nd_full)]):
                continue
            nb_idx_full = int(end_nd_full)
            nb_idx_trim = int(full_to_trim_idx_map[nb_idx_full])
            # don't visit predecessor nodes - otherwise successive nodes revisit out-edges to previous (neighbour) nodes
            if nb_idx_trim == map_pred[active_nd_idx_trim]:
                continue
            # DO ANGULAR BEFORE CYCLES, AND CYCLES BEFORE DISTANCE THRESHOLD
            # it is necessary to check for angular sidestepping if using angular impedances on a dual graph
            # only do this for angular graphs, and only if predecessors exist
            if angular and not np.isnan(map_pred[active_nd_idx_trim]):
                prior_match = False
                # get the predecessor
                pred_idx_trim = int(map_pred[active_nd_idx_trim])
                pred_idx_full = int(trim_to_full_idx_map[pred_idx_trim])
                # check that the new neighbour was not directly accessible from the prior set of neighbours
                pred_edge_idx = int(node_map[pred_idx_full][3])
                while pred_edge_idx < len(edge_map):
                    # get the previous edge's start and end nodes
                    pred_start, pred_end = edge_map[pred_edge_idx][:2]
                    # if the prev start index no longer matches prev node, all previous neighbours have been visited
                    if pred_start != pred_idx_full:
                        break
                    # check that the previous node's neighbour's node is not equal to the currently new neighbour node
                    if pred_end == nb_idx_full:
                        prior_match = True
                        break
                    # increment predecessor idx for next loop
                    pred_edge_idx += 1
                # continue if prior match was found
                if prior_match:
                    continue
            # if a neighbouring node has already been discovered, then it is a cycle
            # do before distance cutoff because this node and the neighbour can respectively be within max distance
            # in some cases e.g. local_centrality(), all distances are run at once, so keep behaviour consistent by
            # designating the farthest node (but via the shortest distance) as the cycle node
            if not np.isnan(active[nb_idx_trim]):
                # set the farthest location to True - nb node vs active node
                if map_distance[nb_idx_trim] > map_distance[active_nd_idx_trim]:
                    cycles[nb_idx_trim] = True
                else:
                    cycles[active_nd_idx_trim] = True
            # impedance and distance is previous plus new
            if not angular:
                impedance = map_impedance[active_nd_idx_trim] + seg_len * seg_imp_fact
            else:
                impedance = map_impedance[active_nd_idx_trim] + (1 + seg_ang / 180) * seg_imp_fact
            dist = map_distance[active_nd_idx_trim] + seg_len
            # check that the distance doesn't exceed the max
            # remember that a closer route may be found, and that this may well be within max...
            if dist > max_dist:
                continue
            # only pursue if impedance is less than prior assigned
            if impedance < map_impedance[nb_idx_trim]:
                map_impedance[nb_idx_trim] = impedance
                map_distance[nb_idx_trim] = dist
                map_pred[nb_idx_trim] = active_nd_idx_trim
                active[nb_idx_trim] = 1  # set node to discovered

    return map_impedance, map_distance, map_pred, cycles


# cache has to be set to false per Numba issue:
# https://github.com/numba/numba/issues/3555
# which prevents nested print function from working as intended
# TODO: set to True once resolved - likely 2020
@njit(cache=False)
def local_centrality(node_map: np.ndarray,
                     edge_map: np.ndarray,
                     distances: np.ndarray,
                     betas: np.ndarray,
                     measure_keys: tuple,
                     angular: bool = False,
                     suppress_progress: bool = False) -> np.ndarray:
    '''
    Call from "compute_centrality", which handles high level checks on keys and heuristic flag

    NODE MAP:
    0 - x
    1 - y
    2 - live
    3 - edge index
    4 - ghosted

    EDGE MAP:
    0 - start node
    1 - end node
    2 - length in metres
    3 - sum of angular travel along length
    4 - impedance factor
    5 - entry bearing
    6 - exit bearing
    '''

    checks.check_network_maps(node_map, edge_map)

    checks.check_distances_and_betas(distances, betas)

    # establish variables
    n = len(node_map)
    d_n = len(distances)
    k_n = len(measure_keys)
    global_max_dist = np.nanmax(distances)
    x_arr = node_map[:, 0]
    y_arr = node_map[:, 1]
    nodes_live = node_map[:, 2]

    # string comparisons will substantially slow down nested loops
    # hence the out-of-loop strategy to map strings to indices corresponding to respective measures
    # keep name and index relationships explicit
    agg_keys = []
    agg_targets = []
    betw_keys = []
    betw_targets = []
    if not angular:
        for m_idx, measure_name in enumerate(measure_keys):
            # aggregating keys
            if measure_name == 'node_density':
                agg_keys.append(0)
                agg_targets.append(m_idx)
            elif measure_name == 'segment_density':
                agg_keys.append(1)
                agg_targets.append(m_idx)
            elif measure_name == 'farness':
                agg_keys.append(2)
                agg_targets.append(m_idx)
            elif measure_name == 'cycles':
                agg_keys.append(3)
                agg_targets.append(m_idx)
            elif measure_name == 'harmonic_node':
                agg_keys.append(4)
                agg_targets.append(m_idx)
            elif measure_name == 'harmonic_segment':
                agg_keys.append(5)
                agg_targets.append(m_idx)
            elif measure_name == 'beta_node':
                agg_keys.append(6)
                agg_targets.append(m_idx)
            elif measure_name == 'beta_segment':
                agg_keys.append(7)
                agg_targets.append(m_idx)
            # betweenness keys
            elif measure_name == 'betweenness_node':
                betw_keys.append(0)
                betw_targets.append(m_idx)
            elif measure_name == 'betweenness_node_wt':
                betw_keys.append(1)
                betw_targets.append(m_idx)
            elif measure_name == 'betweenness_segment':
                betw_keys.append(2)
                betw_targets.append(m_idx)
            else:
                raise ValueError('''
                    Unable to match requested centrality measure key against available options.
                    Shortest-path measures can't be mixed with simplest-path measures.
                    Set angular=True if using simplest-path measures. 
                ''')
    else:
        for m_idx, measure_name in enumerate(measure_keys):
            # aggregating keys
            if measure_name == 'harmonic_node_angle':
                agg_keys.append(8)
                agg_targets.append(m_idx)
            elif measure_name == 'harmonic_segment_hybrid':
                agg_keys.append(9)
                agg_targets.append(m_idx)
            # betweenness keys
            elif measure_name == 'betweenness_node_angle':
                betw_keys.append(3)
                betw_targets.append(m_idx)
            elif measure_name == 'betweenness_segment_hybrid':
                betw_keys.append(4)
                betw_targets.append(m_idx)
            else:
                raise ValueError('''
                    Unable to match requested centrality measure key against available options.
                    Shortest-path measures can't be mixed with simplest-path measures.
                    Set angular=False if using shortest-path measures. 
                ''')
    # prepare data arrays
    # the shortest path is based on impedances -> be cognisant of cases where impedances are not based on true distance:
    # in such cases, distances are equivalent to the impedance heuristic shortest path, not shortest distance in metres
    measures_data = np.full((k_n, d_n, n), 0.0)

    # iterate through each vert and calculate the shortest path tree
    progress_chunks = int(n / 2000)
    for src_idx in range(n):

        # numba no object mode can only handle basic printing
        if not suppress_progress:
            checks.progress_bar(src_idx, n, progress_chunks)

        # only compute for live nodes
        if not nodes_live[src_idx]:
            continue

        # filter the graph by distance
        # note that if global_max_dist == np.inf, then the radial_filter function basically returns np.arange(0.0, n)
        trim_to_full_idx_map, full_to_trim_idx_map = netw_radial_filter(src_idx, node_map, global_max_dist)

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
        for to_idx_trim in range(len(map_distance_trim)):

            if not np.isfinite(map_distance_trim[to_idx_trim]):
                continue

            # skip self node
            if to_idx_trim == full_to_trim_idx_map[src_idx]:
                continue

            impedance = map_impedance_trim[to_idx_trim]
            dist = map_distance_trim[to_idx_trim]

            # node weights removed since v0.10
            # switched to edge impedance factors

            # calculate centralities
            for d_idx in range(len(distances)):
                dist_cutoff = distances[d_idx]
                beta = betas[d_idx]
                if dist <= dist_cutoff:
                    # iterate aggregation functions
                    for agg_idx, agg_key in enumerate(agg_keys):
                        # fetch target index for writing data
                        # stored at equivalent index in agg_targets
                        m_idx = agg_targets[agg_idx]
                        # go through keys and write data
                        # 0 - simple node counts
                        if agg_key == 0:
                            measures_data[m_idx][d_idx][src_idx] += 1
                        # 1 - network density
                        # elif agg_key == 1:

                        # 2 - farness
                        elif agg_key == 2:
                            measures_data[m_idx][d_idx][src_idx] += dist
                        # 3 - cycles
                        elif agg_key == 3:
                            # if there is a cycle
                            if cycles_trim[to_idx_trim]:
                                measures_data[m_idx][d_idx][src_idx] += 1
                        # 4 - harmonic node
                        elif agg_key == 4:
                            measures_data[m_idx][d_idx][src_idx] += 1 / impedance
                        # 5 - harmonic segments
                        # elif agg_key == 5:
                        #
                        # 6 - beta weighted node
                        elif agg_key == 6:
                            measures_data[m_idx][d_idx][src_idx] += np.exp(beta * dist)
                        # 7 - beta weighted segments
                        # elif agg_key == 7:
                        #
                        # 8 - harmonic angle based
                        # elif agg_key == 8:
                        #
                        # 9 - harmonic segments hybrid based
                        # elif agg_key == 9:
                        #

            # check whether betweenness keys actually present prior to proceeding
            if len(betw_keys) == 0:
                continue

            # only process betweenness in one direction
            if to_idx_trim < full_to_trim_idx_map[src_idx]:
                continue

            # weights removed since v0.10
            # switched to impedance factor

            # betweenness - only counting truly between vertices, not starting and ending verts
            inter_idx_trim = int(map_pred_trim[to_idx_trim])
            inter_idx_full = int(trim_to_full_idx_map[inter_idx_trim])  # cast to int

            while True:
                # break out of while loop if the intermediary has reached the source node
                if inter_idx_trim == full_to_trim_idx_map[src_idx]:
                    break

                for d_idx in range(len(distances)):
                    dist_cutoff = distances[d_idx]
                    beta = betas[d_idx]
                    if dist <= dist_cutoff:
                        # iterate betweenness functions
                        for betw_idx, betw_key in enumerate(betw_keys):
                            # fetch target index for writing data
                            # stored at equivalent index in betw_targets
                            m_idx = betw_targets[betw_idx]
                            # go through keys and write data
                            # node count
                            if betw_key == 0:
                                measures_data[m_idx][d_idx][inter_idx_full] += 1
                            # 1 - distance weighted node count
                            # distance is based on distance between from and to vertices
                            # thus potential spatial impedance via between vertex
                            elif betw_key == 1:
                                measures_data[m_idx][d_idx][inter_idx_full] += np.exp(beta * dist)
                            # 2 - network density

                            # 3 - angle based node count

                            # 4 - hybrid segments


                # follow the chain
                inter_idx_trim = int(map_pred_trim[inter_idx_trim])
                inter_idx_full = int(trim_to_full_idx_map[inter_idx_trim])  # cast to int

    return measures_data
