from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numba import njit, prange, types  # type: ignore
from numba.typed import List

from cityseer import config
from cityseer.algos import common


@njit(cache=True, fastmath=config.FASTMATH, nogil=True)
def _find_edge_idx(
    node_edge_map: dict[int, list[int]], edges_end_arr: npt.NDArray[np.int_], start_nd_idx: int, end_nd_idx: int
) -> int:
    """
    Find the edge spanning the specified start / end node pair.
    """
    # iterate the start node's edges
    for edge_idx in node_edge_map[start_nd_idx]:
        # find the edge which has an out node matching the target node
        if edges_end_arr[edge_idx] == end_nd_idx:
            return int(edge_idx)
    return -1


segment_func_proto = types.FunctionType(
    types.float32(types.float32, types.float32, types.float32, types.float32, types.float32)  # type: ignore
)


@njit(cache=True, fastmath=config.FASTMATH, nogil=True, parallel=True)
def local_segment_centrality(
    distances: npt.NDArray[np.int_],
    betas: npt.NDArray[np.float32],
    measure_keys: tuple[str],
    nodes_live_arr: npt.NDArray[np.bool_],
    edges_start_arr: npt.NDArray[np.int_],
    edges_end_arr: npt.NDArray[np.int_],
    edges_length_arr: npt.NDArray[np.float32],
    edges_angle_sum_arr: npt.NDArray[np.float32],
    edges_imp_factor_arr: npt.NDArray[np.float32],
    edges_in_bearing_arr: npt.NDArray[np.float32],
    edges_out_bearing_arr: npt.NDArray[np.float32],
    node_edge_map: dict[int, list[int]],
    jitter_scale: np.float32 = np.float32(0.0),
    angular: bool = False,
    progress_proxy=None,  # type: ignore
) -> npt.NDArray[np.float32]:
    """
    Localised segment centrality.
    """
    # integrity checks
    common.check_distances_and_betas(distances, betas)
    # gather keys - classes with only a single variant don't need list of keys
    close_short_keys: list[str] = []
    close_short_idxs: list[int] = []
    close_simpl_idxs: list[int] = []
    betw_short_idxs: list[int] = []
    betw_simpl_idxs: list[int] = []
    for m_idx, m_key in enumerate(measure_keys):
        if not angular:
            # segment keys
            if m_key == "segment_density":
                close_short_keys.append("segment_density")
                close_short_idxs.append(m_idx)
            elif m_key == "segment_harmonic":
                close_short_keys.append("segment_harmonic")
                close_short_idxs.append(m_idx)
            elif m_key == "segment_beta":
                close_short_keys.append("segment_beta")
                close_short_idxs.append(m_idx)
            elif m_key == "segment_betweenness":
                betw_short_idxs.append(m_idx)
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
                close_simpl_idxs.append(m_idx)
            elif m_key == "segment_betweeness_hybrid":
                betw_simpl_idxs.append(m_idx)
            else:
                raise ValueError(
                    """
                    Unable to match requested centrality measure key against available options.
                    Shortest-path measures can't be mixed with simplest-path measures.
                    Set angular=False if using shortest-path measures.
                """
                )
    # prepare variables
    n_n = nodes_live_arr.shape[0]
    d_n = len(distances)
    k_n = len(measure_keys)
    measures_data: npt.NDArray[np.float32] = np.full((k_n, d_n, n_n), 0.0, dtype=np.float32)
    global_max_dist: np.float32 = np.float32(np.nanmax(distances))
    # iterate through each vert and calculate the shortest path tree
    for src_idx in prange(n_n):  # pylint: disable=not-an-iterable
        shadow_arr: npt.NDArray[np.float32] = np.full((k_n, d_n, n_n), 0.0, dtype=np.float32)
        # numba no object mode can only handle basic printing
        # note that progress bar adds a performance penalty
        if progress_proxy is not None:
            progress_proxy.update(1)
        # only compute for live nodes
        if not nodes_live_arr[src_idx]:
            continue
        """
        Shortest tree dijkstra
        Predecessor map is based on impedance heuristic - i.e. angular vs not
        Shortest path distances in metres used for defining max distances regardless
        """
        (
            visited_nodes,
            preds,
            short_dist,
            simpl_dist,
            _cycles,
            origin_seg,
            last_seg,
            _out_bearings,
            visited_edges,
        ) = shortest_path_tree(
            edges_start_arr,
            edges_end_arr,
            edges_length_arr,
            edges_angle_sum_arr,
            edges_imp_factor_arr,
            edges_in_bearing_arr,
            edges_out_bearing_arr,
            node_edge_map,
            int(src_idx),
            max_dist=global_max_dist,
            jitter_scale=jitter_scale,
            angular=angular,
        )
        """
        can't do edge processing as part of shortest tree because all shortest paths have to be resolved first
        hence visiting all processed edges and extrapolating information
        NOTES:
        1. the above shortest tree algorithm only tracks edges in one direction - i.e. no duplication
        2. dijkstra sorts all active nodes by distance: explores from near to far: edges discovered accordingly
        """
        # only build edge data if necessary
        if close_short_idxs or close_simpl_idxs:
            for edge_idx in np.where(visited_edges)[0]:  # type: ignore
                # unpack the edge data
                n_nd_idx = edges_start_arr[edge_idx]
                m_nd_idx = edges_end_arr[edge_idx]
                seg_len = edges_length_arr[edge_idx]
                seg_ang = edges_angle_sum_arr[edge_idx]
                seg_imp_fact = edges_imp_factor_arr[edge_idx]
                seg_in_bear = edges_in_bearing_arr[edge_idx]
                # go
                n_simpl_dist = simpl_dist[n_nd_idx]
                m_simpl_dist = simpl_dist[m_nd_idx]
                n_short_dist = simpl_dist[n_nd_idx]
                m_short_dist = simpl_dist[m_nd_idx]
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
                if close_short_idxs:
                    """
                    dijkstra discovers edges from near to far (sorts before popping next node)
                    i.e. this sort may be unnecessary?
                    """
                    # sort where a < b
                    if n_short_dist <= m_short_dist:
                        a = short_dist[n_nd_idx]
                        a_imp = short_dist[n_nd_idx]
                        b = short_dist[m_nd_idx]
                        b_imp = short_dist[m_nd_idx]
                    else:
                        a = short_dist[m_nd_idx]
                        a_imp = short_dist[m_nd_idx]
                        b = short_dist[n_nd_idx]
                        b_imp = short_dist[n_nd_idx]
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
                            for m_idx, m_key in zip(close_short_idxs, close_short_keys):
                                if m_key == "segment_density":
                                    shadow_arr[m_idx, d_idx, src_idx] += c - a
                                elif m_key == "segment_harmonic":
                                    if a_imp < 1:
                                        shadow_arr[m_idx, d_idx, src_idx] += np.log(c_imp)
                                    else:
                                        shadow_arr[m_idx, d_idx, src_idx] += np.log(c_imp) - np.log(a_imp)
                                elif m_key == "segment_beta":
                                    if beta == 0.0:
                                        shadow_arr[m_idx, d_idx, src_idx] += c_imp - a_imp
                                    else:
                                        shadow_arr[m_idx, d_idx, src_idx] += (
                                            np.exp(-beta * c_imp) - np.exp(-beta * a_imp)
                                        ) / -beta
                        # a to b segment - if on the shortest path then b == d, in which case, continue
                        if b == d:
                            continue
                        if b <= dist_cutoff:
                            if d > dist_cutoff:
                                d = dist_cutoff
                                d_imp = b_imp + (dist_cutoff - b) * seg_imp_fact
                            for m_idx, m_key in zip(close_short_idxs, close_short_keys):
                                if m_key == "segment_density":
                                    shadow_arr[m_idx, d_idx, src_idx] += d - b
                                elif m_key == "segment_harmonic":
                                    if a_imp < 1:
                                        shadow_arr[m_idx, d_idx, src_idx] += np.log(d_imp)
                                    else:
                                        shadow_arr[m_idx, d_idx, src_idx] += np.log(d_imp) - np.log(b_imp)
                                elif m_key == "segment_beta":
                                    if beta == 0.0:
                                        shadow_arr[m_idx, d_idx, src_idx] += d_imp - b_imp
                                    else:
                                        shadow_arr[m_idx, d_idx, src_idx] += (
                                            np.exp(-beta * d_imp) - np.exp(-beta * b_imp)
                                        ) / -beta
                elif close_simpl_idxs:
                    """
                    there is a different workflow for angular - uses single segment (no segment splitting)
                    this is because the simplest path onto the entire length of segment is from the lower impedance end
                    this assumes segments are relatively straight, overly complex to subdivide segments for spliting...
                    """
                    # only a single case existing for angular version so no need for abstracted functions
                    # there are three scenarios:
                    # 1) e is the predecessor for f
                    if n_nd_idx == src_idx or preds[m_nd_idx] == n_nd_idx:  # pylint: disable=consider-using-in
                        e = short_dist[n_nd_idx]
                        f = short_dist[m_nd_idx]
                        # if travelling via n, then m = n_imp + seg_ang
                        # calculations are based on segment length / angle
                        # i.e. need to decide whether to base angular change on entry vs exit impedance
                        # else take midpoint of segment as ballpark for average, which is the course taken here
                        # i.e. exit impedance minus half segment impedance
                        ang = m_simpl_dist - seg_ang / 2
                    # 2) f is the predecessor for e
                    elif m_nd_idx == src_idx or preds[n_nd_idx] == m_nd_idx:  # pylint: disable=consider-using-in
                        e = short_dist[m_nd_idx]
                        f = short_dist[n_nd_idx]
                        ang = n_simpl_dist - seg_ang / 2  # per above
                    # 3) neither of the above
                    # get the approach angles for either side and compare to find the least inwards impedance
                    # this involves impedance up to entrypoint either side plus respective turns onto the segment
                    else:
                        # get the out bearing from the predecessor and calculate the turn onto current seg's in bearing
                        # find n's predecessor
                        n_pred_idx = int(preds[n_nd_idx])
                        # find the edge from n's predecessor to n
                        e_i = _find_edge_idx(node_edge_map, edges_end_arr, n_pred_idx, n_nd_idx)
                        # get the predecessor edge's outwards bearing at index 6
                        n_pred_out_bear = edges_out_bearing_arr[e_i]
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
                        opp_i = _find_edge_idx(node_edge_map, edges_end_arr, m_nd_idx, n_nd_idx)
                        # now that the opposing edge is known, we can fetch the inwards bearing at index 5 (not 6)
                        opp_in_bear = edges_in_bearing_arr[opp_i]
                        # find m's predecessor
                        m_pred_idx = int(preds[m_nd_idx])
                        # we can now go ahead and find m's predecessor edge
                        e_i = _find_edge_idx(node_edge_map, edges_end_arr, m_pred_idx, m_nd_idx)
                        # get the predecessor edge's outwards bearing at index 6
                        m_pred_out_bear = edges_out_bearing_arr[e_i]
                        # and calculate the turn-in from m's predecessor onto the m inwards bearing
                        m_turn_in = np.abs((opp_in_bear - m_pred_out_bear + 180) % 360 - 180)
                        # then add to aggregated impedance at m
                        m_ang = m_simpl_dist + m_turn_in + seg_ang / 2
                        # the distance and angle are based on the smallest angular impedance onto the segment
                        # select by shortest distance in event angular impedances are identical from either direction
                        if n_ang == m_ang:
                            if n_short_dist <= m_short_dist:
                                e = short_dist[n_nd_idx]
                                ang = n_ang
                            else:
                                e = short_dist[m_nd_idx]
                                ang = m_ang
                        elif n_ang < m_ang:
                            e = short_dist[n_nd_idx]
                            ang = n_ang
                        else:
                            e = short_dist[m_nd_idx]
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
                            for m_idx in close_simpl_idxs:
                                # transform - prevents division by zero
                                agg_ang = 1 + (ang / 180)
                                # then aggregate - angular uses distances explicitly
                                shadow_arr[m_idx, d_idx, src_idx] += (f - e) / agg_ang
        if betw_short_idxs or betw_simpl_idxs:
            # prepare a list of neighbouring nodes
            nb_nodes: list[int] = List.empty_list(types.int64)
            for edge_idx in node_edge_map[int(src_idx)]:
                out_nd_idx = edges_end_arr[edge_idx]
                nb_nodes.append(out_nd_idx)
            # betweenness keys computed per to_idx
            for to_idx in np.where(visited_nodes)[0]:  # type: ignore
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
                to_dist = short_dist[to_idx]
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
                origin_seg_idx = origin_seg[to_idx]
                o_seg_len = edges_length_arr[origin_seg_idx]
                last_seg_idx = last_seg[to_idx]
                l_seg_len = edges_length_arr[last_seg_idx]
                min_span = to_dist - o_seg_len - l_seg_len
                # calculate traversal distances from opposing segments
                o_1 = min_span
                o_2 = min_span + o_seg_len
                l_1 = min_span
                l_2 = min_span + l_seg_len
                # betweenness - only counting truly between vertices, not starting and ending verts
                inter_idx = int(preds[to_idx])
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
                            for m_idx in betw_short_idxs:
                                # catch division by zero
                                if beta == 0.0:
                                    auc = o_2 - o_1 + l_2 - l_1
                                else:
                                    auc = (np.exp(-beta * o_2) - np.exp(-beta * o_1)) / -beta + (
                                        np.exp(-beta * l_2) - np.exp(-beta * l_1)
                                    ) / -beta
                                shadow_arr[m_idx, d_idx, inter_idx] += auc
                            for m_idx in betw_simpl_idxs:
                                bt_ang = 1 + simpl_dist[to_idx] / 180
                                pt_a = o_2 - o_1
                                pt_b = l_2 - l_1
                                shadow_arr[m_idx, d_idx, inter_idx] += (pt_a + pt_b) / bt_ang
                    # follow the chain
                    inter_idx = int(preds[inter_idx])

        # reduction
        measures_data += shadow_arr

    return measures_data
