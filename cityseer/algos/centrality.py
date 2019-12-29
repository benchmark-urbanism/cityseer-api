from typing import Tuple
from numba.typed import List

import numpy as np
from numba import njit, int64

from cityseer.algos import checks


@njit(cache=True)
def shortest_path_tree(
        node_map: np.ndarray,
        edge_map: np.ndarray,
        src_idx: int,
        max_dist: float = np.inf,
        angular: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    All shortest paths to max network distance from source node
    Returns impedances and predecessors for shortest paths from a source node to all other nodes within max distance
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

    # prepare the arrays
    n = len(node_map)
    map_impedance = np.full(n, np.inf)
    map_distance = np.full(n, np.inf)
    map_pred = np.full(n, np.nan)
    cycles = np.full(n, np.nan)

    # the starting node's impedance and distance will be zero
    map_impedance[src_idx] = 0
    map_distance[src_idx] = 0

    # prep the active list and add the source index
    active = List.empty_list(int64)
    active.append(src_idx)
    # this loops continues until all nodes within the max distance have been discovered and processed
    while len(active):
        # iterate the currently active indices and find the one with the smallest distance
        min_nd_idx = None
        min_imp = np.inf
        for idx, nd_idx in enumerate(active):
            imp = map_impedance[nd_idx]
            if imp < min_imp:
                min_imp = imp
                min_nd_idx = nd_idx
        # needs this step with explicit cast to int for numpy type inference
        active_nd_idx = int(min_nd_idx)
        # the currently processed node can now be removed from the active list and added to the processed list
        active.remove(active_nd_idx)
        # fetch the associated edge_map index
        # isolated nodes will have no corresponding edges
        if np.isnan(node_map[active_nd_idx][3]):
            continue
        edge_idx = int(node_map[active_nd_idx][3])
        # iterate the node's neighbours
        # instead of while True, use length of edge map to catch last node's termination
        while edge_idx < len(edge_map):
            # get the edge's properties
            start_nd, end_nd, seg_len, seg_ang, seg_imp_fact, seg_en_bear, seg_ex_bear = edge_map[edge_idx]
            # if the start index no longer matches it means all neighbours have been visited for current node
            if start_nd != active_nd_idx:
                break
            # increment idx for next loop
            edge_idx += 1
            # cast to int for indexing
            nb_nd_idx = int(end_nd)
            # don't visit predecessor nodes - otherwise successive nodes revisit out-edges to previous (neighbour) nodes
            if nb_nd_idx == map_pred[active_nd_idx]:
                continue
            # DO ANGULAR BEFORE CYCLES, AND CYCLES BEFORE DISTANCE THRESHOLD
            # it is necessary to check for angular sidestepping if using angular impedances on a dual graph
            # only do this for angular graphs, and if the nb node has already been discovered
            if angular and not np.isnan(map_pred[nb_nd_idx]):
                prior_match = False
                # get the active node's predecessor
                pred_nd_idx = int(map_pred[active_nd_idx])
                # check that the new neighbour was not directly accessible from the predecessor's set of neighbours
                pred_edge_idx = int(node_map[pred_nd_idx][3])
                while pred_edge_idx < len(edge_map):
                    # iterate start and end nodes corresponding to edges accessible from the predecessor node
                    pred_start, pred_end = edge_map[pred_edge_idx][:2]
                    # if the predecessor start index no longer matches, all have been visited
                    if pred_start != pred_nd_idx:
                        break
                    # check that the previous node's neighbour's node is not equal to the currently new neighbour node
                    # if so, the new neighbour was previously accessible
                    if pred_end == nb_nd_idx:
                        prior_match = True
                        break
                    # increment predecessor idx for next loop
                    pred_edge_idx += 1
                # continue if prior match was found
                if prior_match:
                    continue
            # if a neighbouring node has already been discovered, then it is a cycle
            # do before distance cutoff because this node and the neighbour can respectively be within max distance
            # in some cases all distances are run at once, so keep behaviour consistent by
            # designating the farthest node (but via the shortest distance) as the cycle node
            if not np.isnan(map_pred[nb_nd_idx]):
                # set the farthest location to True - nb node vs active node
                if map_distance[nb_nd_idx] > map_distance[active_nd_idx]:
                    cycles[nb_nd_idx] = True
                else:
                    cycles[active_nd_idx] = True
            # impedance and distance is previous plus new
            if not angular:
                impedance = map_impedance[active_nd_idx] + seg_len * seg_imp_fact
            else:
                impedance = map_impedance[active_nd_idx] + (1 + seg_ang / 180) * seg_imp_fact
            dist = map_distance[active_nd_idx] + seg_len
            # check that the distance doesn't exceed the max
            # remember that a closer route may be found, and that this may well be within max...
            if dist > max_dist:
                continue
            # add to active if undiscovered
            if np.isnan(map_pred[nb_nd_idx]):
                active.append(nb_nd_idx)
            # if impedance less than prior, update
            if impedance < map_impedance[nb_nd_idx]:
                map_impedance[nb_nd_idx] = impedance
                map_distance[nb_nd_idx] = dist
                map_pred[nb_nd_idx] = active_nd_idx

    return map_pred, map_impedance, map_distance, cycles


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
    # prepare data array
    # the shortest path is based on impedances -> be cognisant of cases where impedances are not based on true distance:
    # in such cases, distances are equivalent to the impedance heuristic shortest path, not shortest distance in metres
    measures_data = np.full((k_n, d_n, n), 0.0)
    # iterate through each vert and calculate the shortest path tree
    progress_chunks = int(n / 5000)
    for src_idx in range(n):
        # numba no object mode can only handle basic printing
        if not suppress_progress:
            checks.progress_bar(src_idx, n, progress_chunks)
        # only compute for live nodes
        if not nodes_live[src_idx]:
            continue
        # run the shortest tree dijkstra
        # keep in mind that predecessor map is based on impedance heuristic - which can be different from metres
        # distance map in metres still necessary for defining max distances and computing equivalent distance measures
        map_pred, map_impedance, map_distance, cycles = shortest_path_tree(node_map,
                                                                           edge_map,
                                                                           src_idx,
                                                                           global_max_dist,
                                                                           angular)
        # use corresponding indices for reachable verts
        for to_idx in np.where(~np.isinf(map_impedance))[0]:
            # skip self node
            if to_idx == src_idx:
                continue
            impedance = map_impedance[to_idx]
            dist = map_distance[to_idx]
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
                            if cycles[to_idx]:
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
            if to_idx < src_idx:
                continue
            # weights removed since v0.10
            # switched to impedance factor
            # betweenness - only counting truly between vertices, not starting and ending verts
            inter_idx = int(map_pred[to_idx])
            while True:
                # break out of while loop if the intermediary has reached the source node
                if inter_idx == src_idx:
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
                                measures_data[m_idx][d_idx][inter_idx] += 1
                            # 1 - distance weighted node count
                            # distance is based on distance between from and to vertices
                            # thus potential spatial impedance via between vertex
                            elif betw_key == 1:
                                measures_data[m_idx][d_idx][inter_idx] += np.exp(beta * dist)
                            # 2 - network density

                            # 3 - angle based node count

                            # 4 - hybrid segments
                # follow the chain
                inter_idx = int(map_pred[inter_idx])
    return measures_data
