from __future__ import annotations

from typing import Any
import numpy as np
import numpy.typing as npt
from numba import njit, prange, types  # type: ignore
from numba.experimental import jitclass  # type: ignore
from numba.typed import Dict, List

from cityseer import config
from cityseer.algos import checks, structures

# Note: Tempting to wrap these into a numba jitclass, but parallel and fastmath njit flags are not yet supported.


@njit(cache=True, fastmath=config.FASTMATH, nogil=True)
def shortest_path_tree(
    edge_data: npt.NDArray[np.float32],
    node_edge_map: Dict,  # type: ignore
    src_idx: int,
    max_dist: float = np.inf,
    jitter_scale: float = 0.0,
    angular: bool = False,
) -> structures.TreeMap:
    """
    All shortest paths to max network distance from source node.

    Returns impedances and predecessors for shortest paths from a source node to all other nodes within max distance.
    Angular flag triggers check for sidestepping / cheating with angular impedances (sharp turns).

    Prepares a shortest path tree map - loosely based on dijkstra's shortest path algo.
    Predecessor map is based on impedance heuristic - which can be different from metres.
    Distance map in metres is used for defining max distances and computing equivalent distance measures.
    """
    if angular and not np.all(edge_data[:, 4] == 1):  # type: ignore
        raise ValueError("The distance impedance factor parameter must be set to 1 when using angular centralities.")
    # prepare the arrays
    nodes_n: int = len(node_edge_map)  # type: ignore
    edges_n: int = len(edge_data)  # type: ignore
    tree_map: structures.TreeMap = structures.TreeMap(nodes_n, edges_n)
    # the starting node's impedance and distance will be zero
    tree_map.simpl_dist[src_idx] = 0
    tree_map.short_dist[src_idx] = 0
    # prep the active list and add the source index
    active: list[int] = List.empty_list(types.int64)
    active.append(src_idx)  # type: ignore
    # this loops continues until all nodes within the max distance have been discovered and processed
    while len(active):  # type: ignore
        # iterate the currently active indices and find the one with the smallest distance
        min_nd_idx: int = -1  # uses -1 as placeholder instead of None for type-checking
        min_imp = np.inf
        for nd_idx in active:  # type: ignore
            if angular:
                imp = tree_map.simpl_dist[nd_idx]
            else:
                imp = tree_map.short_dist[nd_idx]
            if imp < min_imp:
                min_imp = imp
                min_nd_idx = nd_idx
        # needs this step with explicit cast to int for numpy type inference
        active_nd_idx = int(min_nd_idx)
        # the currently processed node can now be removed from the active list and added to the processed list
        active.remove(active_nd_idx)  # type: ignore
        # add to processed nodes
        tree_map.visited_nodes[active_nd_idx] = True
        # iterate the node's neighbours
        for edge_idx in node_edge_map[active_nd_idx]:  # type: ignore
            # get the edge's properties
            (_start_nd, end_nd, seg_len, seg_ang, seg_imp_fact, seg_in_bear, seg_out_bear,) = edge_data[
                edge_idx
            ]  # type: ignore
            # cast to int for indexing
            nb_nd_idx: int = int(end_nd)
            # don't follow self-loops
            if nb_nd_idx == active_nd_idx:
                # add edge to active (used for segment methods)
                tree_map.visited_edges[edge_idx] = True  # type: ignore
                continue
            # don't visit predecessor nodes - otherwise successive nodes revisit out-edges to previous (neighbour) nodes
            if nb_nd_idx == tree_map.preds[active_nd_idx]:
                continue
            # only add edge to active if the neighbour node has not been processed previously
            # i.e. single direction only
            # i.e. if neighbour node has been processed all out edges have already been explored
            if not tree_map.visited_nodes[nb_nd_idx]:
                tree_map.visited_edges[edge_idx] = True  # type: ignore
            if not angular:
                # if edge has not been claimed AND the neighbouring node has already been discovered, then it is a cycle
                # do before distance cutoff because this node and the neighbour can respectively be within max distance
                # even if cumulative distance across this edge (via non-shortest path) exceeds distance
                # in some cases all distances are run at once, so keep behaviour consistent by
                # designating the farthest node (but via the shortest distance) as the cycle node
                if not tree_map.preds[nb_nd_idx] == -1:
                    # bump farther location
                    # prevents mismatching if cycle exceeds threshold in one direction or another
                    if tree_map.short_dist[active_nd_idx] <= tree_map.short_dist[nb_nd_idx]:
                        tree_map.cycles[nb_nd_idx] += 0.5
                    else:
                        tree_map.cycles[active_nd_idx] += 0.5
            # impedance and distance is previous plus new
            short_dist = tree_map.short_dist[active_nd_idx] + seg_len * seg_imp_fact
            # angular impedance include two parts:
            # A - turn from prior simplest-path route segment
            # B - angular change across current segment
            if active_nd_idx == src_idx:
                turn = 0
            else:
                turn = np.abs((seg_in_bear - tree_map.out_bearings[active_nd_idx] + 180) % 360 - 180)
            simpl_dist = tree_map.simpl_dist[active_nd_idx] + turn + seg_ang
            # add the neighbour to active if undiscovered but only if less than max shortest path threshold
            if tree_map.preds[nb_nd_idx] == -1 and short_dist <= max_dist:
                active.append(nb_nd_idx)  # type: ignore
            # if impedance less than prior, update
            # this will also happen for the first nodes that overshoot the boundary
            # they will not be explored further because they have not been added to active
            # jitter injects a small amount of stochasticity for rectlinear grids
            jitter = np.random.normal(loc=0, scale=jitter_scale)
            # shortest path heuristic differs for angular vs. not
            if (angular and simpl_dist + jitter < tree_map.simpl_dist[nb_nd_idx]) or (
                not angular and short_dist + jitter < tree_map.short_dist[nb_nd_idx]
            ):
                tree_map.simpl_dist[nb_nd_idx] = simpl_dist
                tree_map.short_dist[nb_nd_idx] = short_dist
                tree_map.preds[nb_nd_idx] = active_nd_idx
                tree_map.out_bearings[nb_nd_idx] = seg_out_bear
                # chain through origin segs - identifies which segment a particular shortest path originated from
                if active_nd_idx == src_idx:
                    tree_map.origin_seg[nb_nd_idx] = edge_idx  # type: ignore
                else:
                    tree_map.origin_seg[nb_nd_idx] = tree_map.origin_seg[active_nd_idx]
                # keep track of last seg
                tree_map.last_seg[nb_nd_idx] = edge_idx  # type: ignore

    # the returned active edges contain activated edges, but only in direction of shortest-path discovery
    return tree_map


@njit(cache=True, fastmath=config.FASTMATH, nogil=True)
def _find_edge_idx(node_edge_map: Dict, edge_data: npt.NDArray[np.float32], start_nd_idx: int, end_nd_idx: int) -> int:  # type: ignore
    """
    Find the edge spanning the specified start / end node pair.
    """
    # iterate the start node's edges
    for edge_idx in node_edge_map[start_nd_idx]:  # type: ignore
        # find the edge which has an out node matching the target node
        if edge_data[edge_idx, 1] == end_nd_idx:
            return int(edge_idx)  # type: ignore
    return -1


node_close_func_proto = types.FunctionType(types.float32(types.float64, types.float64, types.float64, types.float64))  # type: ignore


# node density
@njit("float32(float64, float64, float64, float64)", cache=True, fastmath=config.FASTMATH, nogil=True)
def _node_density(
    to_short_dist: float, to_simpl_dist: float, beta: float, cycles: float  # pylint: disable=unused-argument
) -> np.float32:
    return np.float32(1)


# node farness
@njit("float32(float64, float64, float64, float64)", cache=True, fastmath=config.FASTMATH, nogil=True)
def _node_farness(
    to_short_dist: float, to_simpl_dist: float, beta: float, cycles: float  # pylint: disable=unused-argument
) -> np.float32:
    return np.float32(to_short_dist)


# node cycles
@njit("float32(float64, float64, float64, float64)", cache=True, fastmath=config.FASTMATH, nogil=True)
def _node_cycles(
    to_short_dist: float, to_simpl_dist: float, beta: float, cycles: float  # pylint: disable=unused-argument
) -> np.float32:
    return np.float32(cycles)


# node harmonic
@njit("float32(float64, float64, float64, float64)", cache=True, fastmath=config.FASTMATH, nogil=True)
def _node_harmonic(
    to_short_dist: float, to_simpl_dist: float, beta: float, cycles: float  # pylint: disable=unused-argument
) -> np.float32:
    return np.float32(1 / to_short_dist)


# node beta weighted
@njit("float32(float64, float64, float64, float64)", cache=True, fastmath=config.FASTMATH, nogil=True)
def _node_beta(
    to_short_dist: float, to_simpl_dist: float, beta: float, cycles: float  # pylint: disable=unused-argument
) -> np.float32:
    return np.float32(np.exp(-beta * to_short_dist))


# node harmonic angular
@njit("float32(float64, float64, float64, float64)", cache=True, fastmath=config.FASTMATH, nogil=True)
def _node_harmonic_angular(
    to_short_dist: float, to_simpl_dist: float, beta: float, cycles: float  # pylint: disable=unused-argument
) -> np.float32:
    ang = 1 + (to_simpl_dist / 180)
    return np.float32(1 / ang)


node_betw_func_proto = types.FunctionType(types.float32(types.float64, types.float64))  # type: ignore


# node betweenness
@njit("float32(float64, float64)", cache=True, fastmath=config.FASTMATH, nogil=True)
def _node_betweenness(to_short_dist: float, beta: float) -> np.float32:  # pylint: disable=unused-argument
    return np.float32(1)


# node betweenness beta weighted
@njit("float32(float64, float64)", cache=True, fastmath=config.FASTMATH, nogil=True)
def _node_betweenness_beta(to_short_dist: float, beta: float) -> np.float32:  # pylint: disable=unused-argument
    """
    Distance is based on distance between from and to vertices.

    Thus potential spatial impedance via between vertex.
    """
    return np.float32(np.exp(-beta * to_short_dist))


@njit(cache=True, fastmath=config.FASTMATH, nogil=True, parallel=True)
def local_node_centrality(
    node_data: npt.NDArray[np.float32],
    edge_data: npt.NDArray[np.float32],
    node_edge_map: Dict,  # type: ignore
    distances: npt.NDArray[np.float32],
    betas: npt.NDArray[np.float32],
    measure_keys: tuple[str],
    jitter_scale: float = 0.0,
    angular: bool = False,
    progress_proxy=None,  # type: ignore
) -> npt.NDArray[np.float32]:
    """
    Localised node centrality.
    """
    # integrity checks
    checks.check_distances_and_betas(distances, betas)
    checks.check_network_maps(node_data, edge_data, node_edge_map)
    # gather functions
    close_funcs = List.empty_list(node_close_func_proto)  # type: ignore
    close_idxs: list[int] = []
    betw_funcs = List.empty_list(node_betw_func_proto)  # type: ignore
    betw_idxs: list[int] = []
    for m_idx, m_key in enumerate(measure_keys):
        if not angular:
            # closeness keys
            if m_key == "node_density":
                close_funcs.append(_node_density)
                close_idxs.append(m_idx)
            elif m_key == "node_farness":
                close_funcs.append(_node_farness)
                close_idxs.append(m_idx)
            elif m_key == "node_cycles":
                close_funcs.append(_node_cycles)
                close_idxs.append(m_idx)
            elif m_key == "node_harmonic":
                close_funcs.append(_node_harmonic)
                close_idxs.append(m_idx)
            elif m_key == "node_beta":
                close_funcs.append(_node_beta)
                close_idxs.append(m_idx)
            # betweenness keys
            elif m_key == "node_betweenness":
                betw_funcs.append(_node_betweenness)
                betw_idxs.append(m_idx)
            elif m_key == "node_betweenness_beta":
                betw_funcs.append(_node_betweenness_beta)
                betw_idxs.append(m_idx)
            else:
                raise ValueError(
                    """
                Unable to match requested centrality measure key against available options.
                Shortest-path measures can't be mixed with simplest-path measures.
                Set angular=True if using simplest-path measures."""
                )
        else:
            # aggregative keys
            if m_key == "node_harmonic_angular":
                close_funcs.append(_node_harmonic_angular)  # type: ignore
                close_idxs.append(m_idx)
            # betweenness keys
            elif m_key == "node_betweenness_angular":
                betw_funcs.append(_node_betweenness)  # type: ignore
                betw_idxs.append(m_idx)
            else:
                raise ValueError(
                    """
                Unable to match requested centrality measure key against available options.
                Shortest-path measures can't be mixed with simplest-path measures.
                Set angular=False if using shortest-path measures."""
                )
    # prepare variables
    n = len(node_data)
    d_n = len(distances)
    k_n = len(measure_keys)
    measures_data: npt.NDArray[np.float_] = np.full((k_n, d_n, n), 0.0, dtype=np.float32)
    global_max_dist: float = float(np.nanmax(distances))  # type: ignore
    nodes_live = node_data[:, 2]
    # iterate through each vert and calculate the shortest path tree
    for src_idx in prange(n):  # pylint: disable=not-an-iterable
        shadow_arr: npt.NDArray[np.float_] = np.full((k_n, d_n, n), 0.0, dtype=np.float32)  # type: ignore
        # numba no object mode can only handle basic printing
        # note that progress bar adds a performance penalty
        if progress_proxy is not None:
            progress_proxy.update(1)  # type: ignore
        # only compute for live nodes
        if not nodes_live[src_idx]:
            continue
        tree_map = shortest_path_tree(
            edge_data,
            node_edge_map,  # type: ignore
            src_idx,
            max_dist=global_max_dist,
            jitter_scale=jitter_scale,
            angular=angular,
        )
        visited_nodes: list[int] = np.where(tree_map.visited_nodes)[0]  # type: ignore
        # process each reachable node
        for to_idx in visited_nodes:
            # skip self node
            if to_idx == src_idx:
                continue
            # unpack impedance and distance for to index
            to_short_dist = tree_map.short_dist[to_idx]
            to_simpl_dist = tree_map.simpl_dist[to_idx]
            cycles = tree_map.cycles[to_idx]
            # do not proceed if no route available
            if np.isinf(to_short_dist):
                continue
            # calculate closeness centralities
            if close_funcs:
                for d_idx, dist_cutoff in enumerate(distances):
                    beta = betas[d_idx]
                    if to_short_dist <= dist_cutoff:
                        for m_idx, close_func in zip(close_idxs, close_funcs):  # type: ignore
                            shadow_arr[m_idx, d_idx, src_idx] += close_func(  # type: ignore
                                to_short_dist, to_simpl_dist, beta, cycles
                            )
            # only process in one direction
            if to_idx < src_idx:
                continue
            # calculate betweenness centralities
            if betw_funcs:
                # only counting truly between vertices, not starting and ending verts
                inter_idx = tree_map.preds[to_idx]
                while True:
                    # break out of while loop if the intermediary has reached the source node
                    if inter_idx == src_idx:
                        break
                    # iterate the distance thresholds
                    for d_idx, dist_cutoff in enumerate(distances):
                        beta = betas[d_idx]
                        # check threshold
                        if tree_map.short_dist[to_idx] <= dist_cutoff:
                            # iterate betweenness functions
                            for m_idx, betw_func in zip(betw_idxs, betw_funcs):  # type: ignore
                                shadow_arr[m_idx, d_idx, inter_idx] += betw_func(to_short_dist, beta)  # type: ignore
                    # follow the chain
                    inter_idx = tree_map.preds[inter_idx]
        # reduce
        measures_data += shadow_arr

    return measures_data


segment_func_proto = types.FunctionType(
    types.float32(types.float64, types.float64, types.float64, types.float64, types.float64)  # type: ignore
)


# segment density
@njit("float32(float64, float64, float64, float64, float64)", cache=True, fastmath=config.FASTMATH, nogil=True)
def _segment_density(
    n_val: float, m_val: float, n_imp: float, m_imp: float, beta: float  # type: ignore pylint: disable=unused-argument
) -> np.float32:
    return np.float32(m_val - n_val)


# segment harmonic
@njit("float32(float64, float64, float64, float64, float64)", cache=True, fastmath=config.FASTMATH, nogil=True)
def _segment_harmonic(
    n_val: float, m_val: float, n_imp: float, m_imp: float, beta: float  # type: ignore pylint: disable=unused-argument
) -> np.float32:
    if n_imp < 1:
        return np.float32(np.log(m_imp))
    return np.float32(np.log(m_imp) - np.log(n_imp))


# segment beta
@njit("float32(float64, float64, float64, float64, float64)", cache=True, fastmath=config.FASTMATH, nogil=True)
def _segment_beta(
    n_val: float, m_val: float, n_imp: float, m_imp: float, beta: float  # type: ignore pylint: disable=unused-argument
) -> np.float32:
    if beta == 0.0:
        return np.float32(m_imp - n_imp)
    return np.float32((np.exp(-beta * m_imp) - np.exp(-beta * n_imp)) / -beta)


@njit(cache=True, fastmath=config.FASTMATH, nogil=True, parallel=True)
def local_segment_centrality(
    node_data: npt.NDArray[np.float32],
    edge_data: npt.NDArray[np.float32],
    node_edge_map: Dict,  # type: ignore
    distances: npt.NDArray[np.float32],
    betas: npt.NDArray[np.float32],
    measure_keys: tuple[str],
    jitter_scale: float = 0.0,
    angular: bool = False,
    progress_proxy=None,  # type: ignore
) -> npt.NDArray[np.float32]:
    """
    Localised segment centrality.
    """
    # integrity checks
    checks.check_distances_and_betas(distances, betas)
    checks.check_network_maps(node_data, edge_data, node_edge_map)  # type: ignore
    # gather functions
    close_funcs = List.empty_list(segment_func_proto)  # type: ignore
    close_idxs: list[int] = []
    betw_idxs: list[int] = []
    for m_idx, m_key in enumerate(measure_keys):
        if not angular:
            # segment keys
            if m_key == "segment_density":
                close_funcs.append(_segment_density)  # type: ignore
                close_idxs.append(m_idx)
            elif m_key == "segment_harmonic":
                close_funcs.append(_segment_harmonic)  # type: ignore
                close_idxs.append(m_idx)
            elif m_key == "segment_beta":
                close_funcs.append(_segment_beta)  # type: ignore
                close_idxs.append(m_idx)
            elif m_key == "segment_betweenness":
                # only one version of shortest path betweenness - no need for func
                betw_idxs.append(m_idx)
            else:
                raise ValueError(
                    """
                    Unable to match requested centrality measure key against available options.
                    Shortest-path measures can't be mixed with simplest-path measures.
                    Set angular=True if using simplest-path measures.
                """
                )
        else:
            # segment keys
            if m_key == "segment_harmonic_hybrid":
                # only one version of simplest path closeness - no need for func
                close_idxs.append(m_idx)
            elif m_key == "segment_betweeness_hybrid":
                # only one version of simplest path betweenness - no need for func
                betw_idxs.append(m_idx)
            else:
                raise ValueError(
                    """
                    Unable to match requested centrality measure key against available options.
                    Shortest-path measures can't be mixed with simplest-path measures.
                    Set angular=False if using shortest-path measures.
                """
                )
    # prepare variables
    n = len(node_data)
    d_n = len(distances)
    k_n = len(measure_keys)
    measures_data: npt.NDArray[np.float_] = np.full((k_n, d_n, n), 0.0, dtype=np.float32)  # type: ignore
    global_max_dist: float = np.nanmax(distances)  # type: ignore
    nodes_live = node_data[:, 2]
    # iterate through each vert and calculate the shortest path tree
    for src_idx in prange(n):  # pylint: disable=not-an-iterable
        shadow_arr: npt.NDArray[np.float_] = np.full((k_n, d_n, n), 0.0, dtype=np.float32)  # type: ignore
        # numba no object mode can only handle basic printing
        # note that progress bar adds a performance penalty
        if progress_proxy is not None:
            progress_proxy.update(1)  # type: ignore
        # only compute for live nodes
        if not nodes_live[src_idx]:
            continue
        """
        Shortest tree dijkstra
        Predecessor map is based on impedance heuristic - i.e. angular vs not
        Shortest path distances in metres used for defining max distances regardless
        RETURNS A SHORTEST PATH TREE MAP:
        0 - processed nodes
        1 - predecessors
        2 - shortest path distance
        3 - simplest path angular distance
        4 - cycles
        5 - origin segments
        6 - last segments
        """
        tree_map = shortest_path_tree(
            edge_data,
            node_edge_map,  # type: ignore
            src_idx,
            max_dist=global_max_dist,
            jitter_scale=jitter_scale,
            angular=angular,
        )
        visited_nodes: list[int] = np.where(tree_map.visited_nodes[:, 0])[0]  # type: ignore
        tree_preds = tree_map[:, 1]
        tree_short_dists = tree_map[:, 2]
        tree_simpl_dists = tree_map[:, 3]
        tree_origin_seg = tree_map[:, 5]
        tree_last_seg = tree_map[:, 6]
        """
        can't do edge processing as part of shortest tree because all shortest paths have to be resolved first
        hence visiting all processed edges and extrapolating information
        NOTES:
        1. the above shortest tree algorithm only tracks edges in one direction - i.e. no duplication
        2. dijkstra sorts all active nodes by distance: explores from near to far: edges discovered accordingly
        """
        # only build edge data if necessary
        if close_idxs:
            for edge_idx in np.where(tree_edges)[0]:  # type: ignore
                # unpack the edge data
                (seg_n_nd, seg_m_nd, seg_len, seg_ang, seg_imp_fact, seg_in_bear, _seg_out_bear,) = edge_data[
                    edge_idx
                ]  # type: ignore
                n_nd_idx = int(seg_n_nd)
                m_nd_idx = int(seg_m_nd)
                n_simpl_dist = tree_simpl_dists[n_nd_idx]
                m_simpl_dist = tree_simpl_dists[m_nd_idx]
                n_short_dist = tree_short_dists[n_nd_idx]
                m_short_dist = tree_short_dists[m_nd_idx]
                # don't process unreachable segments
                if np.isinf(n_short_dist) and np.isinf(m_short_dist):
                    continue
                """
                shortest path (non-angular) uses a split segment workflow
                the split workflow allows for non-shortest-path edges to be approached from either direction
                i.e. the shortest path to node "b" isn't necessarily via node "a"
                the edge is then split at the farthest point from either direction and apportioned either way
                if the segment is on the shortest path then the second segment will squash down to naught
                """
                if not angular:
                    """
                    dijkstra discovers edges from near to far (sorts before popping next node)
                    i.e. this sort may be unnecessary?
                    """
                    # sort where a < b
                    if n_short_dist <= m_short_dist:
                        a = tree_short_dists[n_nd_idx]
                        a_imp = tree_short_dists[n_nd_idx]
                        b = tree_short_dists[m_nd_idx]
                        b_imp = tree_short_dists[m_nd_idx]
                    else:
                        a = tree_short_dists[m_nd_idx]
                        a_imp = tree_short_dists[m_nd_idx]
                        b = tree_short_dists[n_nd_idx]
                        b_imp = tree_short_dists[n_nd_idx]
                    # get the max distance along the segment: seg_len = (m - start_len) + (m - end_len)
                    # c and d variables can diverge per beneath
                    c = d = (seg_len + a + b) / 2
                    # c | d impedance should technically be the same if computed from either side
                    c_imp = d_imp = a_imp + (c - a) * seg_imp_fact
                    # iterate the distance and beta thresholds - from large to small for threshold snipping
                    for d_idx in range(len(distances) - 1, -1, -1):
                        dist_cutoff = distances[d_idx]
                        beta = betas[d_idx]
                        """
                        if c or d are greater than the distance threshold, then the segments are "snipped"
                        """
                        # a to c segment
                        if a <= dist_cutoff:
                            if c > dist_cutoff:
                                c = dist_cutoff
                                c_imp = a_imp + (dist_cutoff - a) * seg_imp_fact
                            for m_idx, close_func in zip(close_idxs, close_funcs):  # type: ignore
                                shadow_arr[m_idx, d_idx, src_idx] += close_func(a, c, a_imp, c_imp, beta)  # type: ignore
                        # a to b segment - if on the shortest path then b == d, in which case, continue
                        if b == d:
                            continue
                        if b <= dist_cutoff:
                            if d > dist_cutoff:
                                d = dist_cutoff
                                d_imp = b_imp + (dist_cutoff - b) * seg_imp_fact
                            for m_idx, close_func in zip(close_idxs, close_funcs):  # type: ignore
                                shadow_arr[m_idx, d_idx, src_idx] += close_func(b, d, b_imp, d_imp, beta)  # type: ignore
                else:
                    """
                    there is a different workflow for angular - uses single segment (no segment splitting)
                    this is because the simplest path onto the entire length of segment is from the lower impedance end
                    this assumes segments are relatively straight, overly complex to subdivide segments for spliting...
                    """
                    # only a single case existing for angular version so no need for abstracted functions
                    # there are three scenarios:
                    # 1) e is the predecessor for f
                    if n_nd_idx == src_idx or tree_preds[m_nd_idx] == n_nd_idx:  # pylint: disable=consider-using-in
                        e = tree_short_dists[n_nd_idx]
                        f = tree_short_dists[m_nd_idx]
                        # if travelling via n, then m = n_imp + seg_ang
                        # calculations are based on segment length / angle
                        # i.e. need to decide whether to base angular change on entry vs exit impedance
                        # else take midpoint of segment as ballpark for average, which is the course taken here
                        # i.e. exit impedance minus half segment impedance
                        ang = m_simpl_dist - seg_ang / 2
                    # 2) f is the predecessor for e
                    elif m_nd_idx == src_idx or tree_preds[n_nd_idx] == m_nd_idx:  # pylint: disable=consider-using-in
                        e = tree_short_dists[m_nd_idx]
                        f = tree_short_dists[n_nd_idx]
                        ang = n_simpl_dist - seg_ang / 2  # per above
                    # 3) neither of the above
                    # get the approach angles for either side and compare to find the least inwards impedance
                    # this involves impedance up to entrypoint either side plus respective turns onto the segment
                    else:
                        # get the out bearing from the predecessor and calculate the turn onto current seg's in bearing
                        # find n's predecessor
                        n_pred_idx = int(tree_preds[n_nd_idx])
                        # find the edge from n's predecessor to n
                        e_i = _find_edge_idx(node_edge_map, edge_data, n_pred_idx, n_nd_idx)  # type: ignore
                        # get the predecessor edge's outwards bearing at index 6
                        n_pred_out_bear = edge_data[int(e_i), 6]
                        # calculating the turn into this segment from the predecessor's out bearing
                        n_turn_in = np.abs((seg_in_bear - n_pred_out_bear + 180) % 360 - 180)
                        # then add the turn-in to the aggregated impedance at n
                        # i.e. total angular impedance onto this segment
                        # as above two scenarios, adding half of angular impedance for segment as avg between in / out
                        n_ang = n_simpl_dist + n_turn_in + seg_ang / 2
                        # repeat for the other side other side
                        # per original n -> m edge destructuring: m is the node in the outwards bound direction
                        # i.e. need to first find the corresponding edge in the opposite m -> n direction of travel
                        # this gives the correct inwards bearing as if m were the entry point
                        opp_i = _find_edge_idx(node_edge_map, edge_data, m_nd_idx, n_nd_idx)  # type: ignore
                        # now that the opposing edge is known, we can fetch the inwards bearing at index 5 (not 6)
                        opp_in_bear = edge_data[int(opp_i), 5]
                        # find m's predecessor
                        m_pred_idx = int(tree_preds[m_nd_idx])
                        # we can now go ahead and find m's predecessor edge
                        e_i = _find_edge_idx(node_edge_map, edge_data, m_pred_idx, m_nd_idx)  # type: ignore
                        # get the predecessor edge's outwards bearing at index 6
                        m_pred_out_bear = edge_data[int(e_i), 6]
                        # and calculate the turn-in from m's predecessor onto the m inwards bearing
                        m_turn_in = np.abs((opp_in_bear - m_pred_out_bear + 180) % 360 - 180)
                        # then add to aggregated impedance at m
                        m_ang = m_simpl_dist + m_turn_in + seg_ang / 2
                        # the distance and angle are based on the smallest angular impedance onto the segment
                        # select by shortest distance in event angular impedances are identical from either direction
                        if n_ang == m_ang:
                            if n_short_dist <= m_short_dist:
                                e = tree_short_dists[n_nd_idx]
                                ang = n_ang
                            else:
                                e = tree_short_dists[m_nd_idx]
                                ang = m_ang
                        elif n_ang < m_ang:
                            e = tree_short_dists[n_nd_idx]
                            ang = n_ang
                        else:
                            e = tree_short_dists[m_nd_idx]
                            ang = m_ang
                        # f is the entry distance plus segment length
                        f = e + seg_len
                    # iterate the distance thresholds - from large to small for threshold snipping
                    for d_idx in range(len(distances) - 1, -1, -1):
                        dist_cutoff = distances[d_idx]
                        if e <= dist_cutoff:
                            f = min(f, dist_cutoff)
                            # uses segment length as base (in this sense hybrid)
                            # intentionally not using integral because conflates harmonic shortest-path w. simplest
                            # there is only one case for angular - no need to abstract to func
                            for m_idx in close_idxs:
                                # transform - prevents division by zero
                                agg_ang = 1 + (ang / 180)
                                # then aggregate - angular uses distances explicitly
                                shadow_arr[m_idx, d_idx, src_idx] += (f - e) / agg_ang
        if betw_idxs:
            # prepare a list of neighbouring nodes
            nb_nodes = List.empty_list(types.int64)  # type: ignore
            for edge_idx in node_edge_map[src_idx]:  # type: ignore
                out_nd_idx = int(edge_data[edge_idx][1])  # type: ignore # to node is index 1
                nb_nodes.append(out_nd_idx)  # type: ignore
            # betweenness keys computed per to_idx
            for to_idx in visited_nodes:
                # only process in one direction
                if to_idx < src_idx:
                    continue
                # skip self node
                if to_idx == src_idx:
                    continue
                # skip direct neighbours (no nodes between)
                if to_idx in nb_nodes:
                    continue
                # distance - do not proceed if no route available
                to_dist = tree_short_dists[to_idx]
                if np.isinf(to_dist):
                    continue
                """
                BETWEENNESS
                segment versions only agg first and last segments
                the distance decay is based on the distance between the src segment and to segment
                i.e. willingness of people to walk between src and to segments

                betweenness is aggregated to intervening nodes based on above distances and decays
                other sections (in between current first and last) are respectively processed from other to nodes

                distance thresholds are computed using the innner as opposed to outer edges of the segments
                """
                o_seg_len = edge_data[int(tree_origin_seg[to_idx])][2]
                l_seg_len = edge_data[int(tree_last_seg[to_idx])][2]
                min_span = to_dist - o_seg_len - l_seg_len
                # calculate traversal distances from opposing segments
                o_1 = min_span
                o_2 = min_span + o_seg_len
                l_1 = min_span
                l_2 = min_span + l_seg_len
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
                        if min_span <= dist_cutoff:
                            # prune if necessary
                            o_2 = min(o_2, dist_cutoff)
                            l_2 = min(l_2, dist_cutoff)
                            # only one version for betweenness for respective angular / non angular
                            # i.e. no need to abstract to function
                            for m_idx in betw_idxs:
                                if not angular:
                                    # catch division by zero
                                    if beta == 0.0:
                                        auc = o_2 - o_1 + l_2 - l_1
                                    else:
                                        auc = (np.exp(-beta * o_2) - np.exp(-beta * o_1)) / -beta + (
                                            np.exp(-beta * l_2) - np.exp(-beta * l_1)
                                        ) / -beta
                                    shadow_arr[m_idx, d_idx, inter_idx] += auc
                                else:
                                    bt_ang = 1 + tree_simpl_dists[to_idx] / 180
                                    pt_a = o_2 - o_1
                                    pt_b = l_2 - l_1
                                    shadow_arr[m_idx, d_idx, inter_idx] += (pt_a + pt_b) / bt_ang
                    # follow the chain
                    inter_idx = int(tree_preds[inter_idx])

        # reduction
        measures_data += shadow_arr

    return measures_data
