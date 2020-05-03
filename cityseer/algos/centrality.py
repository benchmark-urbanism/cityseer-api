from typing import Tuple
from numba.typed import List, Dict

import numpy as np
from numba import njit, uint8, int64, types

from cityseer.algos import checks



# don't use 'nnan' fastmath flag
@njit(cache=True, fastmath={'ninf', 'nsz', 'arcp', 'contract', 'afn', 'reassoc'})
def shortest_path_tree(
        edge_data: np.ndarray,
        node_edge_map: Dict,
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
    3 - ghosted

    EDGE MAP:
    0 - start node
    1 - end node
    2 - length in metres
    3 - sum of angular travel along length
    4 - impedance factor
    5 - in bearing
    6 - out bearing

    RETURNS A SHORTEST PATH TREE MAP:
    0 - processed nodes
    1 - predecessors
    2 - distances
    3 - impedances
    4 - cycles
    5 - origin segments - for any to_idx, the origin segment of the shortest path
    6 - last segments - for any to_idx, the last segment of the shortest path
    '''
    # prepare the arrays
    n = len(node_edge_map)
    tree_map = np.full((n, 7), np.nan, dtype=np.float32)
    tree_map[:, 0] = 0
    tree_map[:, 2] = np.inf
    tree_map[:, 3] = np.inf
    tree_map[:, 4] = 0
    # prepare proxies
    tree_nodes = tree_map[:, 0]
    tree_preds = tree_map[:, 1]
    tree_dists = tree_map[:, 2]
    tree_imps = tree_map[:, 3]
    tree_cycles = tree_map[:, 4]
    tree_origin_seg = tree_map[:, 5]
    tree_last_seg = tree_map[:, 6]
    # when doing angular, need to keep track of out bearings
    out_bearings = np.full(n, np.nan, dtype=np.float32)
    # keep track of visited edges
    tree_edges = np.full(len(edge_data), False)
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
        # iterate the node's neighbours
        for edge_idx in node_edge_map[active_nd_idx]:
            # get the edge's properties
            start_nd, end_nd, seg_len, seg_ang, seg_imp_fact, seg_in_bear, seg_out_bear = edge_data[edge_idx]
            # cast to int for indexing
            nb_nd_idx = int(end_nd)
            # don't follow self-loops
            if nb_nd_idx == active_nd_idx:
                # add edge to active (used for segment methods)
                tree_edges[edge_idx] = True
                continue
            # don't visit predecessor nodes - otherwise successive nodes revisit out-edges to previous (neighbour) nodes
            if nb_nd_idx == tree_preds[active_nd_idx]:
                continue
            # DO ANGULAR BEFORE CYCLES, AND CYCLES BEFORE DISTANCE THRESHOLD
            # it is necessary to check for angular sidestepping if using angular impedances on a dual graph
            # only do this for angular graphs, and if the nb node has already been discovered
            if angular and active_nd_idx != src_idx and not np.isnan(tree_preds[nb_nd_idx]):
                prior_match = False
                # get the active node's predecessor
                pred_nd_idx = int(tree_preds[active_nd_idx])
                # check that the new neighbour was not directly accessible from the predecessor's set of neighbours
                for pred_edge_idx in node_edge_map[pred_nd_idx]:
                    # iterate start and end nodes corresponding to edges accessible from the predecessor node
                    pred_start, pred_end = edge_data[pred_edge_idx, :2]
                    # check that the previous node's neighbour's node is not equal to the currently new neighbour node
                    # if so, the new neighbour was previously accessible
                    if pred_end == nb_nd_idx:
                        prior_match = True
                        break
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
                    turn = np.abs((seg_in_bear - out_bearings[active_nd_idx] + 180) % 360 - 180)
                impedance = tree_imps[active_nd_idx] + (turn + seg_ang) * seg_imp_fact
            dist = tree_dists[active_nd_idx] + seg_len
            # add the neighbour to active if undiscovered but only if less than max threshold
            if np.isnan(tree_preds[nb_nd_idx]) and dist <= max_dist:
                active.append(nb_nd_idx)
            # only add edge to active if the neighbour node has not been processed previously (i.e. single direction only)
            if not tree_nodes[nb_nd_idx]:
                tree_edges[edge_idx] = True
            # if impedance less than prior, update
            # this will also happen for the first nodes that overshoot the boundary
            # they will not be explored further because they have not been added to active
            if impedance < tree_imps[nb_nd_idx]:
                tree_imps[nb_nd_idx] = impedance
                tree_dists[nb_nd_idx] = dist
                tree_preds[nb_nd_idx] = active_nd_idx
                out_bearings[nb_nd_idx] = seg_out_bear
                # chain through origin segs - identifies which segment a particular shortest path originated from
                if active_nd_idx == src_idx:
                    tree_origin_seg[nb_nd_idx] = edge_idx
                else:
                    tree_origin_seg[nb_nd_idx] = tree_origin_seg[active_nd_idx]
                # keep track of last seg
                tree_last_seg[nb_nd_idx] = edge_idx
    # the returned active edges contain activated edges, but only in direction of shortest-path discovery
    return tree_map, tree_edges

@njit(cache=True, fastmath=True)
def _find_edge_idx(node_edge_map: Dict, edge_data: np.ndarray, start_nd_idx: int, end_nd_idx: int) -> int:
    '''
    Finds an edge from start and end nodes
    '''
    # iterate the start node's edges
    for edge_idx in node_edge_map[start_nd_idx]:
        # check whether the edge's out node matches the target node
        if edge_data[edge_idx, 1] == end_nd_idx:
            return edge_idx


close_func_proto = types.FunctionType(types.float64(types.float64, types.float64, types.float64, types.float64))


# node density
@njit(cache=True, fastmath=True)
def node_density(to_dist, to_imp, beta, cycles):
    return 1.0  # return float explicitly


# node farness
@njit(cache=True, fastmath=True)
def node_farness(to_dist, to_imp, beta, cycles):
    return to_dist


# node cycles
@njit(cache=True, fastmath=True)
def node_cycles(to_dist, to_imp, beta, cycles):
    return cycles


# node harmonic
@njit(cache=True, fastmath=True)
def node_harmonic(to_dist, to_imp, beta, cycles):
    return 1.0 / to_imp


# node beta weighted
@njit(cache=True, fastmath=True)
def node_beta(to_dist, to_imp, beta, cycles):
    return np.exp(beta * to_dist)


# node harmonic angular
@njit(cache=True, fastmath=True)
def node_harmonic_angular(to_dist, to_imp, beta, cycles):
    a = 1 + (to_imp / 180)
    return 1.0 / a


betw_func_proto = types.FunctionType(types.float64(types.float64, types.float64))


# node betweenness
@njit(cache=True, fastmath=True)
def node_betweenness(to_dist, beta):
    return 1.0  # return float explicitly


# node betweenness beta weighted
@njit(cache=True, fastmath=True)
def node_betweenness_beta(to_dist, beta):
    '''
    distance is based on distance between from and to vertices
    thus potential spatial impedance via between vertex
    '''
    return np.exp(beta * to_dist)


'''
NODE MAP:
0 - x
1 - y
2 - live
3 - ghosted
EDGE MAP:
0 - start node
1 - end node
2 - length in metres
3 - sum of angular travel along length
4 - impedance factor
5 - in bearing
6 - out bearing
'''


@njit(cache=False, fastmath=True)
def local_node_centrality(node_data: np.ndarray,
                          edge_data: np.ndarray,
                          node_edge_map: Dict,
                          distances: np.ndarray,
                          betas: np.ndarray,
                          measure_keys: tuple,
                          angular=False,
                          suppress_progress: bool = False) -> np.ndarray:
    # integrity checks
    checks.check_distances_and_betas(distances, betas)
    checks.check_network_maps(node_data, edge_data, node_edge_map)
    # gather functions
    close_funcs = List.empty_list(close_func_proto)
    close_idxs = []
    betw_funcs = List.empty_list(betw_func_proto)
    betw_idxs = []
    if not angular:
        for m_idx, m_key in enumerate(measure_keys):
            # closeness keys
            if m_key == 'node_density':
                close_funcs.append(node_density)
                close_idxs.append(m_idx)
            elif m_key == 'node_farness':
                close_funcs.append(node_farness)
                close_idxs.append(m_idx)
            elif m_key == 'node_cycles':
                close_funcs.append(node_cycles)
                close_idxs.append(m_idx)
            elif m_key == 'node_harmonic':
                close_funcs.append(node_harmonic)
                close_idxs.append(m_idx)
            elif m_key == 'node_beta':
                close_funcs.append(node_beta)
                close_idxs.append(m_idx)
            # betweenness keys
            elif m_key == 'node_betweenness':
                betw_funcs.append(node_betweenness)
                betw_idxs.append(m_idx)
            elif m_key == 'node_betweenness_beta':
                betw_funcs.append(node_betweenness_beta)
                betw_idxs.append(m_idx)
            else:
                raise ValueError('''
                Unable to match requested centrality measure key against available options.
                Shortest-path measures can't be mixed with simplest-path measures.
                Set angular=True if using simplest-path measures.''')
    else:
        for m_idx, m_key in enumerate(measure_keys):
            # aggregative keys
            if m_key == 'node_harmonic_angular':
                close_funcs.append(node_harmonic_angular)
                close_idxs.append(m_idx)
            # betweenness keys
            elif m_key == 'node_betweenness_angular':
                betw_funcs.append(node_betweenness)
                betw_idxs.append(m_idx)
            else:
                raise ValueError('''
                Unable to match requested centrality measure key against available options.
                Shortest-path measures can't be mixed with simplest-path measures.
                Set angular=True if using simplest-path measures.''')
    # prepare variables
    n = len(node_data)
    d_n = len(distances)
    k_n = len(measure_keys)
    measures_data = np.full((k_n, d_n, n), 0.0, dtype=np.float32)
    global_max_dist = float(np.nanmax(distances))
    nodes_live = node_data[:, 2]
    # progress steps
    steps = int(n / 10000)
    # iterate through each vert and calculate the shortest path tree
    for src_idx in range(n):
        # numba no object mode can only handle basic printing
        # note that progress bar adds a performance penalty
        if not suppress_progress:
            checks.progress_bar(src_idx, n, steps)
        # only compute for live nodes
        if not nodes_live[src_idx]:
            continue
        '''
        Shortest tree dijkstra        
        Predecessor map is based on impedance heuristic - which can be different from metres
        Distance map in metres still necessary for defining max distances and computing equivalent distance measures
        RETURNS A SHORTEST PATH TREE MAP:
        '''
        tree_map, tree_edges = shortest_path_tree(edge_data,
                                                  node_edge_map,
                                                  src_idx,
                                                  max_dist=global_max_dist,
                                                  angular=False)
        '''
        0 - processed nodes
        1 - predecessors
        2 - distances
        3 - impedances
        4 - cycles
        5 - origin segments
        6 - last segments
        '''
        tree_nodes = np.where(tree_map[:, 0])[0]
        tree_preds = tree_map[:, 1]
        tree_dists = tree_map[:, 2]
        tree_imps = tree_map[:, 3]
        tree_cycles = tree_map[:, 4]
        # process each reachable node
        for to_idx in tree_nodes:
            # skip self node
            if to_idx == src_idx:
                continue
            # unpack impedance and distance for to index
            to_imp = tree_imps[to_idx]
            to_dist = tree_dists[to_idx]
            cycles = tree_cycles[to_idx]  # bool in form of 1 or 0
            # do not proceed if no route available
            if np.isinf(to_dist):
                continue
            # calculate closeness centralities
            if close_funcs:
                for d_idx in range(len(distances)):
                    dist_cutoff = distances[d_idx]
                    beta = betas[d_idx]
                    if to_dist <= dist_cutoff:
                        for m_idx, close_func in zip(close_idxs, close_funcs):
                            measures_data[m_idx, d_idx, src_idx] += close_func(to_dist, to_imp, beta, cycles)
            # only process in one direction
            if to_idx < src_idx:
                continue
            # calculate betweenness centralities
            if betw_funcs:
                # only counting truly between vertices, not starting and ending verts
                inter_idx = int(tree_preds[to_idx])
                while True:
                    # break out of while loop if the intermediary has reached the source node
                    if inter_idx == src_idx:
                        break
                    # iterate the distance thresholds
                    for d_idx in range(len(distances)):
                        dist_cutoff = distances[d_idx]
                        beta = betas[d_idx]
                        # check threshold
                        if tree_dists[to_idx] <= dist_cutoff:
                            # iterate betweenness functions
                            for m_idx, betw_func in zip(betw_idxs, betw_funcs):
                                measures_data[m_idx, d_idx, inter_idx] += betw_func(to_dist, beta)
                    # follow the chain
                    inter_idx = int(tree_preds[inter_idx])
    return measures_data


"""
@njit(cache=False, fastmath=True)
def local_segment_centrality(node_data: np.ndarray,
                          edge_data: np.ndarray,
                          node_edge_map: Dict,
                          distances: np.ndarray,
                          betas: np.ndarray,
                          measure_keys: tuple,
                          angular: bool = False,
                          suppress_progress: bool = False) -> np.ndarray:
    checks.check_distances_and_betas(distances, betas)
    checks.check_network_maps(node_data, edge_data, node_edge_map)
    node_closeness_funcs = List.empty_list(func_proto)
    node_closeness_targets = List()
    node_betweenness_funcs = List.empty_list(func_proto)
    node_betweenness_targets = List()
    for m_idx, measure_name in enumerate(measure_keys):
        if not angular:
    # segment keys (betweenness segments can be built during betweenness iters)
    if measure_name == 'segment_density':
    seg_keys.append(0)
    seg_targets.append(m_idx)
    elif measure_name == 'segment_harmonic':
    seg_keys.append(1)
    seg_targets.append(m_idx)
    elif measure_name == 'segment_beta':
    seg_keys.append(2)
    seg_targets.append(m_idx)
    elif measure_name == 'segment_betweenness':
    betw_keys.append(2)
    betw_targets.append(m_idx)

    tree_map, tree_edges = shortest_path_tree(edge_data,
                                              node_edge_map,
                                              src_idx,
                                              max_dist=global_max_dist,
                                              angular=False)
    tree_nodes = np.where(tree_map[:, 0])[0]
    tree_preds = tree_map[:, 1]
    tree_dists = tree_map[:, 2]
    tree_imps = tree_map[:, 3]
    tree_cycles = tree_map[:, 4]
    tree_origin_seg = tree_map[:, 5]
    tree_last_seg = tree_map[:, 6]

# only build edge data if necessary
        if len(seg_keys) > 0:
            # can't do edge processing as part of shortest tree because all shortest paths have to be resolved first
            # visit all processed edges
            for edge_idx in np.where(tree_edges)[0]:
                # unpack
                seg_in_nd, seg_out_nd, seg_len, seg_ang, seg_imp_fact, seg_in_bear, seg_out_bear = edge_data[edge_idx]
                in_nd_idx = int(seg_in_nd)
                out_nd_idx = int(seg_out_nd)
                in_imp = tree_imps[in_nd_idx]
                out_imp = tree_imps[out_nd_idx]
                in_dist = tree_dists[in_nd_idx]
                out_dist = tree_dists[out_nd_idx]
                # don't process unreachable segments
                if np.isinf(in_dist) and np.isinf(out_dist):
                    continue
                # for conceptual simplicity, separate angular and non-angular workflows
                # non angular uses a split segment workflow
                # if the segment is on the shortest path then the second segment will squash down to naught
                if not angular:
                    # sort where a < b
                    if in_imp <= out_imp:
                        a = tree_dists[in_nd_idx]
                        a_imp = tree_imps[in_nd_idx]
                        b = tree_dists[out_nd_idx]
                        b_imp = tree_imps[out_nd_idx]
                    else:
                        a = tree_dists[out_nd_idx]
                        a_imp = tree_imps[out_nd_idx]
                        b = tree_dists[in_nd_idx]
                        b_imp = tree_imps[in_nd_idx]
                    # get the max distance along the segment: seg_len = (m - start_len) + (m - end_len)
                    # c and d variables can diverge per beneath
                    c = d = (seg_len + a + b) / 2
                    # c / d impedance should technically be the same if computed from either side
                    c_imp = d_imp = a_imp + (c - a) * seg_imp_fact
                    # iterate the distance and beta thresholds - from large to small for threshold snipping
                    for d_idx in range(len(distances) - 1, -1, -1):
                        dist_cutoff = distances[d_idx]
                        beta = betas[d_idx]
                        # a-c segment
                        if a <= dist_cutoff:
                            if c > dist_cutoff:
                                c = dist_cutoff
                                c_imp = a_imp + (dist_cutoff - a) * seg_imp_fact
                            for seg_idx, seg_key in enumerate(seg_keys):
                                m_idx = seg_targets[seg_idx]
                                if seg_key == 0:
                                    measures_data[m_idx, d_idx, src_idx] += c - a
                                elif seg_key == 1:
                                    if a_imp < 1:
                                        measures_data[m_idx, d_idx, src_idx] += np.log(c_imp)
                                    else:
                                        measures_data[m_idx, d_idx, src_idx] += np.log(c_imp) - np.log(a_imp)
                                elif seg_key == 2:
                                    if beta == -0.0:
                                        auc = c_imp - a_imp
                                    else:
                                        auc = (np.exp(beta * c_imp) -
                                               np.exp(beta * a_imp)) / beta
                                    measures_data[m_idx, d_idx, src_idx] += auc
                        # a-b segment - if on the shortest path then d == b - in which case, continue
                        if b == d:
                            continue
                        if b <= dist_cutoff:
                            if d > dist_cutoff:
                                d = dist_cutoff
                                d_imp = b_imp + (dist_cutoff - b) * seg_imp_fact
                            for seg_idx, seg_key in enumerate(seg_keys):
                                m_idx = seg_targets[seg_idx]
                                if seg_key == 0:
                                    measures_data[m_idx, d_idx, src_idx] += d - b
                                elif seg_key == 1:
                                    if b_imp < 1:
                                        measures_data[m_idx, d_idx, src_idx] += np.log(d_imp)
                                    else:
                                        measures_data[m_idx, d_idx, src_idx] += np.log(d_imp) - np.log(b_imp)
                                elif seg_key == 2:
                                    # catch division by zero
                                    # as beta approaches 0 the distance is weighted by 1 instead of < 1
                                    if beta == -0.0:
                                        auc = d_imp - b_imp
                                    else:
                                        auc = (np.exp(beta * d_imp) -
                                               np.exp(beta * b_imp)) / beta
                                    measures_data[m_idx, d_idx, src_idx] += auc
                # different workflow for angular - uses single segment
                # otherwise many assumptions if splitting segments re: angular vs. distance shortest-paths...
                else:
                    # get the approach angles for either side
                    # this involves impedance up to that point plus the turn onto the segment
                    # also add half of the segment's length-wise angular impedance
                    in_ang = in_imp + seg_ang / 2
                    # the source node won't have a predecessor
                    if in_nd_idx != src_idx:
                        # get the out bearing from the predecessor and calculate the turn onto current seg's in bearing
                        in_pred_idx = int(tree_preds[in_nd_idx])
                        e_i = _find_edge_idx(node_edge_map, edge_data, in_pred_idx, in_nd_idx)
                        in_pred_out_bear = edge_data[int(e_i), 6]
                        in_ang += np.abs((seg_in_bear - in_pred_out_bear + 180) % 360 - 180)
                    # same for other side
                    out_ang = out_imp + seg_ang / 2
                    if out_nd_idx != src_idx:
                        out_pred_idx = int(tree_preds[out_nd_idx])
                        e_i = _find_edge_idx(node_edge_map, edge_data, out_pred_idx, out_nd_idx)
                        out_pred_out_bear = edge_data[int(e_i), 6]
                        out_ang += np.abs((seg_out_bear - out_pred_out_bear + 180) % 360 - 180)
                    # the distance and angle are based on the smallest angular impedance onto the segment
                    # shortest-path segments will have exit bearings equal to the entry bearings
                    # in this case, select the closest by shortest distance
                    if in_ang == out_ang:
                        if in_dist < out_dist:
                            e = tree_dists[in_nd_idx]
                            ang = in_ang
                        else:
                            e = tree_dists[out_nd_idx]
                            ang = out_ang
                    elif in_ang < out_ang:
                        e = tree_dists[in_nd_idx]
                        ang = in_ang
                    else:
                        e = tree_dists[out_nd_idx]
                        ang = out_ang
                    # f is the entry distance plus segment length
                    f = e + seg_len
                    # iterate the distance thresholds - from large to small for threshold snipping
                    for d_idx in range(len(distances) - 1, -1, -1):
                        dist_cutoff = distances[d_idx]
                        if e <= dist_cutoff:
                            if f > dist_cutoff:
                                f = dist_cutoff
                            # 3 - harmonic segments hybrid
                            # Uses integral of segment distances as a base - then weighted by angular
                            for seg_idx, seg_key in enumerate(seg_keys):
                                if seg_key == 3:
                                    m_idx = seg_targets[seg_idx]
                                    # transform - prevents division by zero
                                    agg_ang = 1 + (ang / 180)
                                    # then aggregate - angular uses distances explicitly
                                    measures_data[m_idx, d_idx, src_idx] += (f - e) / agg_ang
# segment versions only agg first and last segments - intervening bits are processed from other to nodes
                o_seg_len = edge_data[int(tree_origin_seg[to_idx])][2]
                l_seg_len = edge_data[int(tree_last_seg[to_idx])][2]
                min_seg_span = tree_dists[to_idx] - o_seg_len - l_seg_len
                o_1 = min_seg_span
                o_2 = min_seg_span + o_seg_len
                l_1 = min_seg_span
                l_2 = min_seg_span + l_seg_len
                # betweenness - only counting truly between vertices, not starting and ending verts
                inter_idx = int(tree_preds[to_idx])
                while True:
                    # break out of while loop if the intermediary has reached the source node
                    if inter_idx == src_idx:
                        break
                    # iterate the distance thresholds - from large to small for threshold snipping
                    for d_idx in range(len(distances) - 1, -1, -1):
                        dist_cutoff = distances[d_idx]
                        beta = betas[d_idx]
                        if min_seg_span <= dist_cutoff:
                            # prune if necessary
                            if o_2 > dist_cutoff:
                                o_2 = dist_cutoff
                            if l_2 > dist_cutoff:
                                l_2 = dist_cutoff
                            for betw_idx, betw_key in enumerate(betw_keys):
                                m_idx = betw_targets[betw_idx]
                                # 2 - segment version of betweenness
                                if betw_key == 2:
                                    # catch division by zero
                                    if beta == -0.0:
                                        auc = o_2 - o_1 + l_2 - l_1
                                    else:
                                        auc = (np.exp(beta * o_2) -
                                               np.exp(beta * o_1)) / beta + \
                                              (np.exp(beta * l_2) -
                                               np.exp(beta * l_1)) / beta
                                    measures_data[m_idx, d_idx, inter_idx] += auc
                                # 4 - betweeenness segment hybrid version
                                elif betw_key == 4:
                                    bt_ang = 1 + tree_imps[to_idx] / 180
                                    pt_a = o_2 - o_1
                                    pt_b = l_2 - l_1
                                    measures_data[m_idx, d_idx, inter_idx] += (pt_a + pt_b) / bt_ang
                    # follow the chain
                    inter_idx = int(tree_preds[inter_idx])

@njit(cache=False, fastmath=True)
def local_segment_centrality_angular(node_data: np.ndarray,
                             edge_data: np.ndarray,
                             node_edge_map: Dict,
                             distances: np.ndarray,
                             betas: np.ndarray,
                             measure_keys: tuple,
                             angular: bool = False,
                             suppress_progress: bool = False) -> np.ndarray:

        # segment keys
        elif measure_name == 'segment_harmonic_hybrid':
            seg_keys.append(3)
            seg_targets.append(m_idx)

        elif measure_name == 'segment_betweeness_hybrid':
            betw_keys.append(4)
            betw_targets.append(m_idx)
"""
