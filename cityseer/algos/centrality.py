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
        angular: bool = False) -> Tuple[np.ndarray, np.ndarray]:
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

    RETURNS A SHORTEST PATH TREE MAP:
    0 - active
    1 - pred
    2 - distance
    3 - impedance
    4 - cycles
    '''
    # prepare the arrays
    n = len(node_map)
    tree_map = np.full((n, 5), np.inf)
    tree_map[:, 0] = 0
    tree_map[:, 1] = np.nan
    tree_map[:, 4] = 0
    # prepare proxies
    tree_nodes = tree_map[:, 0]
    tree_preds = tree_map[:, 1]
    tree_dists = tree_map[:, 2]
    tree_imps = tree_map[:, 3]
    tree_cycles = tree_map[:, 4]
    # when doing angular, need to keep track of exit bearings
    exit_bearings = np.full(n, np.nan)
    # keep track of visited edges
    tree_edges = np.full(len(edge_map), False)
    # the starting node's impedance and distance will be zero
    tree_imps[src_idx] = 0
    tree_dists[src_idx] = 0
    # prep the active list and add the source index
    active = List.empty_list(int64)
    active.append(src_idx)
    # this loops continues until all nodes within the max distance have been discovered and processed
    while len(active):
        # iterate the currently active indices and find the one with the smallest distance
        min_nd_idx = None
        min_imp = np.inf
        for idx, nd_idx in enumerate(active):
            imp = tree_imps[nd_idx]
            if imp < min_imp:
                min_imp = imp
                min_nd_idx = nd_idx
        # needs this step with explicit cast to int for numpy type inference
        active_nd_idx = int(min_nd_idx)
        # the currently processed node can now be removed from the active list and added to the processed list
        active.remove(active_nd_idx)
        # add to active node
        tree_nodes[active_nd_idx] = True
        # fetch the associated edge_map index
        # isolated nodes will have no corresponding edges
        if np.isnan(node_map[active_nd_idx, 3]):
            continue
        edge_idx = int(node_map[active_nd_idx, 3])
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
            if nb_nd_idx == tree_preds[active_nd_idx]:
                continue
            # DO ANGULAR BEFORE CYCLES, AND CYCLES BEFORE DISTANCE THRESHOLD
            # it is necessary to check for angular sidestepping if using angular impedances on a dual graph
            # only do this for angular graphs, and if the nb node has already been discovered
            if angular and not np.isnan(tree_preds[nb_nd_idx]):
                prior_match = False
                # get the active node's predecessor
                pred_nd_idx = int(tree_preds[active_nd_idx])
                # check that the new neighbour was not directly accessible from the predecessor's set of neighbours
                pred_edge_idx = int(node_map[pred_nd_idx, 3])
                while pred_edge_idx < len(edge_map):
                    # iterate start and end nodes corresponding to edges accessible from the predecessor node
                    pred_start, pred_end = edge_map[pred_edge_idx, :2]
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
            if not np.isnan(tree_preds[nb_nd_idx]):
                # set the farthest location to True - nb node vs active node
                if tree_dists[nb_nd_idx] > tree_dists[active_nd_idx]:
                    tree_cycles[nb_nd_idx] = 1
                else:
                    tree_cycles[active_nd_idx] = 1
            # impedance and distance is previous plus new
            if not angular:
                impedance = tree_imps[active_nd_idx] + seg_len * seg_imp_fact
            else:
                # angular impedance include two parts:
                # A - turn from prior simplest-path route segment
                # B - angular change across current segment
                if active_nd_idx == src_idx:
                    turn = 0
                else:
                    turn = np.abs((seg_en_bear - exit_bearings[active_nd_idx] + 180) % 360 - 180)
                impedance = tree_imps[active_nd_idx] + (turn + seg_ang) * seg_imp_fact
            dist = tree_dists[active_nd_idx] + seg_len
            # add the neighbour to active if undiscovered but only if less than max threshold
            if np.isnan(tree_preds[nb_nd_idx]) and dist <= max_dist:
                active.append(nb_nd_idx)
            # only add edge to active if the neighbour node has not been processed previously
            if not tree_nodes[nb_nd_idx]:
                # minus one because index has already been incremented
                tree_edges[edge_idx - 1] = True
            # if impedance less than prior, update
            if impedance < tree_imps[nb_nd_idx]:
                tree_imps[nb_nd_idx] = impedance
                tree_dists[nb_nd_idx] = dist
                tree_preds[nb_nd_idx] = active_nd_idx
                exit_bearings[nb_nd_idx] = seg_ex_bear
    # the returned active edges contain activated edges, but only in direction of shortest-path discovery
    return tree_map, tree_edges


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
    seg_keys = []
    seg_targets = []
    betw_keys = []
    betw_targets = []
    if not angular:
        for m_idx, measure_name in enumerate(measure_keys):
            # aggregating keys
            if measure_name == 'node_density':
                agg_keys.append(0)
                agg_targets.append(m_idx)
            elif measure_name == 'farness':
                agg_keys.append(1)
                agg_targets.append(m_idx)
            elif measure_name == 'cycles':
                agg_keys.append(2)
                agg_targets.append(m_idx)
            elif measure_name == 'harmonic_node':
                agg_keys.append(3)
                agg_targets.append(m_idx)
            elif measure_name == 'beta_node':
                agg_keys.append(4)
                agg_targets.append(m_idx)
            # segment keys (betweenness segments can be built during betweenness iters)
            elif measure_name == 'segment_density':
                seg_keys.append(0)
                seg_targets.append(m_idx)
            elif measure_name == 'harmonic_segment':
                seg_keys.append(1)
                seg_targets.append(m_idx)
            elif measure_name == 'beta_segment':
                seg_keys.append(2)
                seg_targets.append(m_idx)
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
                agg_keys.append(5)
                agg_targets.append(m_idx)
            # segment keys
            elif measure_name == 'harmonic_segment_hybrid':
                seg_keys.append(3)
                seg_targets.append(m_idx)
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
        '''
        run the shortest tree dijkstra
        keep in mind that predecessor map is based on impedance heuristic - which can be different from metres
        distance map in metres still necessary for defining max distances and computing equivalent distance measures
        RETURNS A SHORTEST PATH TREE MAP:
        0 - active
        1 - pred
        2 - distance
        3 - impedance
        4 - cycles
        '''
        tree_map, tree_edges = shortest_path_tree(node_map,
                                                      edge_map,
                                                      src_idx,
                                                      global_max_dist,
                                                      angular)
        tree_nodes = np.where(tree_map[:, 0])[0]
        tree_preds = tree_map[:, 1]
        tree_dists = tree_map[:, 2]
        tree_imps = tree_map[:, 3]
        tree_cycles = tree_map[:, 4]

        # only build edge data if necessary
        if len(seg_keys) > 0:
            # visit all processed nodes
            for edge_idx in np.where(tree_edges)[0]:
                # unpack
                start_nd, end_nd, seg_len, seg_ang, seg_imp_fact, seg_en_bear, seg_ex_bear = edge_map[edge_idx]
                # get the start and end distances and impedances
                start_idx = int(start_nd)
                end_idx = int(end_nd)
                start_dist = tree_dists[start_idx]
                end_dist = tree_dists[end_idx]
                start_imp = tree_imps[start_idx]
                end_imp = tree_imps[end_idx]
                # sort where a < b
                if start_dist < end_dist:
                    a = start_dist
                    a_imp = start_imp
                    b = end_dist
                    b_imp = end_imp
                else:
                    a = end_dist
                    a_imp = end_imp
                    b = start_dist
                    b_imp = start_imp
                # iterate the distance and beta thresholds
                for d_idx in range(len(distances)):
                    dist_cutoff = distances[d_idx]
                    beta = betas[d_idx]
                    # only proceed if the smaller a distance is within the current cutoff
                    if a < dist_cutoff:
                        # in some cases the segment calcs need to be split
                        split_segs = False
                        # if the larger b distance exceeds the distance threshold then snip
                        if b > dist_cutoff:
                            b = dist_cutoff
                            b_imp = a_imp + (dist_cutoff - a) * seg_imp_fact
                        # else if the segment is not on the shortest-path, split the segment and compute from either end
                        elif a + seg_len != end_dist:
                            split_segs = True
                            # get the max distance along the segment
                            # seg_len = (m - start_len) + (m - end_len)
                            m = (seg_len + a + b) / 2
                            # check that max is not exceeded
                            if m > dist_cutoff:
                                m = dist_cutoff
                            # determine c and d
                            c = m
                            c_imp = a_imp + (m - a) * seg_imp_fact
                            d = m
                            d_imp = b_imp + (m - b) * seg_imp_fact
                        # iterate the segment function keys
                        for seg_idx, seg_key in enumerate(seg_keys):
                            # fetch target index for writing data
                            # stored at equivalent index in seg_targets
                            m_idx = seg_targets[seg_idx]
                            # 0 - segment density - uses plain distances
                            if seg_key == 0:
                                measures_data[m_idx, d_idx, src_idx] += b - a
                            # 1 - harmonic segments - uses impedances in case of impedance multiplier
                            elif seg_key == 1:
                                if not split_segs:
                                    measures_data[m_idx, d_idx, src_idx] += np.log(b_imp) - np.log(a_imp)
                                else:
                                    measures_data[m_idx, d_idx, src_idx] += np.log(c_imp) - np.log(a_imp)
                                    measures_data[m_idx, d_idx, src_idx] += np.log(d_imp) - np.log(b_imp)
                            # 2 - beta weighted segments
                            elif seg_key == 2:
                                if not split_segs:
                                    measures_data[m_idx, d_idx, src_idx] += (np.exp(beta * b_imp) -
                                                                         np.exp(beta * a_imp)) / beta
                                else:
                                    measures_data[m_idx, d_idx, src_idx] += (np.exp(beta * c_imp) -
                                                                             np.exp(beta * a_imp)) / beta
                                    measures_data[m_idx, d_idx, src_idx] += (np.exp(beta * d_imp) -
                                                                             np.exp(beta * b_imp)) / beta
                            # 3 - harmonic segments hybrid
                            # Uses integral of segment distances as a base - then weighted by angular
                            elif seg_key == 3:
                                # average from the end impedance minus half of the segment's angular change
                                a = end_imp - seg_ang / 2
                                # transform - prevents division by zero
                                a = 1 + (a / 180)
                                if not split_segs:
                                    measures_data[m_idx, d_idx, src_idx] += (np.log(b) - np.log(a)) / a
                                else:
                                    measures_data[m_idx, d_idx, src_idx] += (np.log(c) - np.log(a)) / a
                                    measures_data[m_idx, d_idx, src_idx] += (np.log(d) - np.log(b)) / a
        # aggregative and betweenness keys can be computed per to_idx
        for to_idx in tree_nodes:
            # skip self node
            if to_idx == src_idx:
                continue
            to_imp = tree_imps[to_idx]
            to_dist = tree_dists[to_idx]
            # node weights removed since v0.10
            # switched to edge impedance factors
            # calculate centralities
            for d_idx in range(len(distances)):
                dist_cutoff = distances[d_idx]
                beta = betas[d_idx]
                if to_dist <= dist_cutoff:
                    # iterate aggregation functions
                    for agg_idx, agg_key in enumerate(agg_keys):
                        # fetch target index for writing data
                        # stored at equivalent index in agg_targets
                        m_idx = agg_targets[agg_idx]
                        # go through keys and write data
                        # 0 - simple node counts
                        if agg_key == 0:
                            measures_data[m_idx, d_idx, src_idx] += 1
                        # 1 - farness
                        elif agg_key == 1:
                            measures_data[m_idx, d_idx, src_idx] += to_dist
                        # 2 - cycles
                        elif agg_key == 2:
                            if tree_cycles[to_idx]:
                                measures_data[m_idx, d_idx, src_idx] += 1
                        # 3 - harmonic node
                        elif agg_key == 3:
                            measures_data[m_idx, d_idx, src_idx] += 1 / to_imp
                        # 4 - beta weighted node
                        elif agg_key == 4:
                            measures_data[m_idx, d_idx, src_idx] += np.exp(beta * to_dist)
                        # 5 - harmonic node - angular
                        elif agg_key == 5:
                            a = 1 + (to_imp / 180)  # transform angles
                            measures_data[m_idx, d_idx, src_idx] += 1 / a
            # check whether betweenness keys are present prior to proceeding
            if len(betw_keys) == 0:
                continue
            # only process betweenness in one direction
            if to_idx < src_idx:
                continue
            # weights removed since v0.10
            # switched to impedance factor
            # betweenness - only counting truly between vertices, not starting and ending verts
            inter_idx = int(tree_preds[to_idx])
            while True:
                # break out of while loop if the intermediary has reached the source node
                if inter_idx == src_idx:
                    break
                for d_idx in range(len(distances)):
                    dist_cutoff = distances[d_idx]
                    beta = betas[d_idx]
                    # node based betweenness measures only count
                    if to_dist <= dist_cutoff:
                        # iterate betweenness functions
                        for betw_idx, betw_key in enumerate(betw_keys):
                            # fetch target index for writing data
                            # stored at equivalent index in betw_targets
                            m_idx = betw_targets[betw_idx]
                            # go through keys and write data
                            # simple count of nodes for betweenness
                            if betw_key == 0:
                                measures_data[m_idx, d_idx, inter_idx] += 1
                            # 1 - beta weighted betweenness
                            # distance is based on distance between from and to vertices
                            # thus potential spatial impedance via between vertex
                            elif betw_key == 1:
                                measures_data[m_idx, d_idx, inter_idx] += np.exp(beta * to_dist)
                            # 2 - segment version of betweenness
                            # elif betw_key == 2:
                            # 3 - betweenness node count - angular heuristic version
                            elif betw_key == 3:
                                measures_data[m_idx, d_idx, inter_idx] += 1
                            # 4 - betweeenness segment hybrid version
                            # elif betw_key == 4:
                # follow the chain
                inter_idx = int(tree_preds[inter_idx])
    return measures_data
