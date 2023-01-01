from __future__ import annotations

from typing import Any, Optional

import numpy as np
import numpy.typing as npt
from numba import njit, prange  # type: ignore

from cityseer import config
from cityseer.algos import centrality, common, diversity


@njit(cache=True, fastmath=config.FASTMATH, nogil=True)
def find_nearest(
    src_x: np.float32,
    src_y: np.float32,
    nodes_x_arr: npt.NDArray[np.float32],
    nodes_y_arr: npt.NDArray[np.float32],
    max_dist: np.float32,
) -> tuple[int, np.float32, int]:
    """Find nearest index and distance from a given point."""
    min_idx = -1
    min_dist = np.float32(np.inf)
    next_min_idx = -1
    next_min_dist = np.float32(np.inf)
    # filter by distance
    for i in range(nodes_x_arr.shape[0]):
        dist = np.hypot(nodes_x_arr[i] - src_x, nodes_y_arr[i] - src_y)
        if dist <= max_dist and dist < min_dist:
            next_min_idx = min_idx
            next_min_dist = min_dist
            min_idx = i
            min_dist = dist
        elif dist <= max_dist and dist < next_min_dist:
            next_min_idx = i
            next_min_dist = dist

    return min_idx, min_dist, next_min_idx


# https://stackoverflow.com/questions/37459121/calculating-angle-between-three-points-but-only-anticlockwise-in-python
# these two points / angles are relative to the origin so pass in difference between the points and origin as vectors
@njit(cache=True, fastmath=config.FASTMATH, nogil=True)
def _calculate_rotation(point_a: npt.NDArray[np.float32], point_b: npt.NDArray[np.float32]) -> np.float32:
    ang_a = np.arctan2(point_a[1], point_a[0])  # arctan is in y/x order
    ang_b = np.arctan2(point_b[1], point_b[0])
    return np.rad2deg((ang_a - ang_b) % (2 * np.pi))


@njit(cache=True, fastmath=config.FASTMATH, nogil=True)
def _calculate_rotation_smallest(point_a: npt.NDArray[np.float32], point_b: npt.NDArray[np.float32]) -> np.float32:
    # smallest difference angle
    ang_a = np.rad2deg(np.arctan2(point_a[1], point_a[0]))
    ang_b = np.rad2deg(np.arctan2(point_b[1], point_b[0]))
    return np.abs((ang_b - ang_a + 180) % 360 - 180)


@njit(cache=True, fastmath=config.FASTMATH, nogil=True)
def _road_distance(
    nodes_x_arr: npt.NDArray[np.float32],
    nodes_y_arr: npt.NDArray[np.float32],
    d_coords: npt.NDArray[np.float32],
    netw_idx_a: int,
    netw_idx_b: int,
) -> tuple[np.float32, int, int]:
    a_coords: npt.NDArray[np.float32] = np.array([nodes_x_arr[netw_idx_a], nodes_y_arr[netw_idx_a]])
    b_coords: npt.NDArray[np.float32] = np.array([nodes_x_arr[netw_idx_b], nodes_y_arr[netw_idx_b]])
    # get the angles from either intersection node to the data point
    ang_a = _calculate_rotation_smallest(d_coords - a_coords, b_coords - a_coords)
    ang_b = _calculate_rotation_smallest(d_coords - b_coords, a_coords - b_coords)
    # assume offset street segment if either is significantly greater than 90
    # (in which case sideways offset from road)
    if ang_a > 110 or ang_b > 110:
        return np.float32(np.inf), -1, -1
    # calculate height from two sides and included angle
    side_a = np.hypot(d_coords[0] - a_coords[0], d_coords[1] - a_coords[1])
    side_b = np.hypot(d_coords[0] - b_coords[0], d_coords[1] - b_coords[1])
    base = np.hypot(a_coords[0] - b_coords[0], a_coords[1] - b_coords[1])
    # forestall potential division by zero
    if base == 0:
        return np.float32(np.inf), -1, -1
    # heron's formula
    half_perim = (side_a + side_b + base) / 2  # perimeter / 2
    area = np.sqrt(half_perim * (half_perim - side_a) * (half_perim - side_b) * (half_perim - base))
    # area is 1/2 base * height, so height = area / (0.5 * base)
    height = area / (0.5 * base)
    # NOTE - the height of the triangle may be less than the distance to the nodes
    # happens due to offset segments: can cause wrong assignment where adjacent segments have same triangle height
    # in this case, set to length of closest node so that height (minimum distance) is still meaningful
    # return indices in order of nearest then next nearest
    if side_a < side_b:
        if ang_a > 90:
            height = side_a
        return height, netw_idx_a, netw_idx_b
    if ang_b > 90:
        height = side_b
    return height, netw_idx_b, netw_idx_a


@njit(cache=True, fastmath=config.FASTMATH, nogil=True)
def _closest_intersections(
    nodes_x_arr: npt.NDArray[np.float32],
    nodes_y_arr: npt.NDArray[np.float32],
    d_coords: npt.NDArray[np.float32],
    predecessor_map: npt.NDArray[np.int_],
    end_node_idx: int,
) -> tuple[np.float32, int, int]:
    """Find the closest and next closest nodes."""
    if len(predecessor_map) == 1:
        return np.float32(np.inf), end_node_idx, -1
    current_idx = end_node_idx
    next_idx = predecessor_map[end_node_idx]
    if len(predecessor_map) == 2:
        return _road_distance(nodes_x_arr, nodes_y_arr, d_coords, current_idx, next_idx)
    nearest_idx = -1
    next_nearest_idx = -1
    min_d = np.float32(np.inf)
    first_pred = next_idx  # for finding end of loop
    while True:
        height, n_idx, n_n_idx = _road_distance(nodes_x_arr, nodes_y_arr, d_coords, current_idx, next_idx)
        if height < min_d:
            min_d = height
            nearest_idx = n_idx
            next_nearest_idx = n_n_idx
        # if the next in the chain is -1, then break
        if predecessor_map[next_idx] == -1:
            break
        current_idx = next_idx
        next_idx = predecessor_map[next_idx]
        if next_idx == first_pred:
            break
    return min_d, nearest_idx, next_nearest_idx


@njit(cache=True, fastmath=config.FASTMATH, nogil=True, parallel=True)
def assign_to_network(
    data_map_x_arr: npt.NDArray[np.float32],
    data_map_y_arr: npt.NDArray[np.float32],
    data_map_nearest_arr: npt.NDArray[np.int_],
    data_map_next_nearest_arr: npt.NDArray[np.int_],
    nodes_x_arr: npt.NDArray[np.float32],
    nodes_y_arr: npt.NDArray[np.float32],
    edges_end_arr: npt.NDArray[np.int_],
    node_edge_map: dict[int, list[int]],
    max_dist: np.float32,
    progress_proxy: Any = None,
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
    """
    To save unnecessary computation - this is done once and written to the data map.

    1 - find the closest network node from each data point
    2A - wind clockwise along the network to preferably find a block cycle surrounding the node
    2B - in event of topological traps, try anti-clockwise as well
    3A - select the closest block cycle node
    3B - if no enclosing cycle - simply use the closest node
    4 - find the neighbouring node that minimises the distance between the data point on "street-front"
    """
    for data_idx in prange(data_map_x_arr.shape[0]):  # pylint: disable=not-an-iterable
        if progress_proxy is not None:
            progress_proxy.update(1)
        # find the nearest and next nearest network nodes
        src_x = data_map_x_arr[data_idx]
        src_y = data_map_y_arr[data_idx]
        min_idx, min_dist, next_min_idx = find_nearest(src_x, src_y, nodes_x_arr, nodes_y_arr, max_dist)
        # in some cases no network node will be within max_dist...
        if min_idx == -1:
            continue
        connected = False
        # check if min and next min are connected
        for edge_idx in node_edge_map[min_idx]:
            nb_idx = edges_end_arr[edge_idx]
            if nb_idx == next_min_idx:
                connected = True
                break
        # if connected, then no need to circle the block
        if connected:
            data_map_nearest_arr[data_idx] = min_idx
            data_map_next_nearest_arr[data_idx] = next_min_idx
            continue
        # if not connected, find the nearest adjacent by edges
        # nearest is initially set for this nearest node, but if a nearer street-edge is found, it will be overriden
        nearest_idx = min_idx
        next_nearest_idx = -1
        # set start node to nearest network node
        node_idx: int = min_idx
        # keep track of visited nodes
        pred_map: npt.NDArray[np.int_] = np.full(len(node_edge_map), -1, dtype=np.int_)
        # state
        reversing = False
        # keep track of previous indices
        prev_idx: int = -1
        # iterate neighbours
        while True:
            # reset neighbour rotation and index counters
            rotation = np.nan
            nb_idx = -1
            # iterate the edges
            for edge_idx in node_edge_map[node_idx]:
                # new idx from the edge's end node index
                new_idx = edges_end_arr[edge_idx]
                # don't follow self-loops
                if new_idx == node_idx:
                    continue
                # check that this isn't the previous node (already visited as neighbour from other direction)
                if new_idx == prev_idx:
                    continue
                # look for the new neighbour with the smallest rightwards (anti-clockwise arctan2) angle
                # measure the angle relative to the data point for the first node
                coords_new: npt.NDArray[np.float32] = np.array([nodes_x_arr[new_idx], nodes_y_arr[new_idx]])
                coords_node: npt.NDArray[np.float32] = np.array([nodes_x_arr[node_idx], nodes_y_arr[node_idx]])
                coords_prev: npt.NDArray[np.float32] = np.array([nodes_x_arr[prev_idx], nodes_y_arr[prev_idx]])
                coords_data: npt.NDArray[np.float32] = np.array([data_map_x_arr[data_idx], data_map_y_arr[data_idx]])
                if prev_idx == -1:
                    rot = _calculate_rotation(
                        coords_new - coords_node,
                        coords_data - coords_node,
                    )
                # else relative to the previous node
                else:
                    rot = _calculate_rotation(
                        coords_new - coords_node,
                        coords_prev - coords_node,
                    )
                if reversing:
                    rot = 360 - rot
                # if least angle, update
                if np.isnan(rotation) or rot < rotation:
                    rotation = rot
                    nb_idx = new_idx
            # allow backtracking if no neighbour is found - i.e. dead-ends
            if nb_idx == -1:
                if pred_map[node_idx] == -1:
                    # for isolated nodes: nb_idx == -1, pred_map[node_idx] == -1, and prev_idx == -1
                    if prev_idx == -1:
                        break
                    # for isolated edges, the algorithm gets turned-around back to the starting node with nowhere to go
                    # nb_idx == -1, pred_map[node_idx] == -1
                    # in these cases, pass _closest_intersections the prev idx so that it has a predecessor to follow
                    coords_data: npt.NDArray[np.float32] = np.array(
                        [data_map_x_arr[data_idx], data_map_y_arr[data_idx]]
                    )
                    dist, n, n_n = _closest_intersections(nodes_x_arr, nodes_y_arr, coords_data, pred_map, prev_idx)
                    if dist < min_dist:
                        nearest_idx = n
                        next_nearest_idx = n_n
                    break
                # otherwise, go ahead and backtrack
                nb_idx = pred_map[node_idx]
            # if the distance is exceeded, reset and attempt in the other direction
            dist = np.hypot(
                nodes_x_arr[nb_idx] - data_map_x_arr[data_idx],
                nodes_y_arr[nb_idx] - data_map_y_arr[data_idx],
            )
            if dist > max_dist:
                pred_map[nb_idx] = node_idx
                coords_data: npt.NDArray[np.float32] = np.array([data_map_x_arr[data_idx], data_map_y_arr[data_idx]])
                dist, n, n_n = _closest_intersections(nodes_x_arr, nodes_y_arr, coords_data, pred_map, nb_idx)
                # if the distance to the street edge is less than the nearest node, or than the prior closest edge
                if dist < min_dist:
                    min_dist = dist
                    nearest_idx = n
                    next_nearest_idx = n_n
                # reverse and try in opposite direction
                if not reversing:
                    reversing = True
                    pred_map.fill(-1)
                    node_idx = min_idx
                    prev_idx = -1
                    continue
                break
            # ignore the following conditions while backtracking
            # (if backtracking, the current node's predecessor will be equal to the new neighbour)
            if nb_idx != pred_map[node_idx]:
                # if the new nb node has already been visited then terminate, this prevents infinite loops
                # or, if the algorithm has circled the block back to the original starting node
                if not pred_map[nb_idx] == -1 or nb_idx == min_idx:
                    # set the final predecessor, BUT ONLY if re-encountered the original node
                    # this would otherwise occlude routes (e.g. backtracks) that have passed the same node twice
                    # (such routes are still able to recover the closest edge)
                    if nb_idx == min_idx:
                        pred_map[nb_idx] = node_idx
                    coords_data: npt.NDArray[np.float32] = np.array(
                        [data_map_x_arr[data_idx], data_map_y_arr[data_idx]]
                    )
                    dist, n, n_n = _closest_intersections(nodes_x_arr, nodes_y_arr, coords_data, pred_map, nb_idx)
                    if dist < min_dist:
                        nearest_idx = n
                        next_nearest_idx = n_n
                    break
                # set predecessor (only if not backtracking)
                pred_map[nb_idx] = node_idx
            # otherwise, keep going
            prev_idx = node_idx
            node_idx = nb_idx
        # print(f'[{data_idx}, {nearest}, {next_nearest}],')
        # set in the data map
        # no race condition in spite of direct indexing because each is set only once?
        data_map_nearest_arr[data_idx] = nearest_idx
        # in some cases next nearest will be NaN
        # this is mostly in situations where it works to leave as NaN
        # e.g. access off dead-ends...
        data_map_next_nearest_arr[data_idx] = next_nearest_idx

    return data_map_nearest_arr, data_map_next_nearest_arr


@njit(cache=True, fastmath=config.FASTMATH, nogil=True)
def aggregate_to_src_idx(
    netw_src_idx: int,
    nodes_x_arr: npt.NDArray[np.float32],
    nodes_y_arr: npt.NDArray[np.float32],
    edges_start_arr: npt.NDArray[np.int_],
    edges_end_arr: npt.NDArray[np.int_],
    edges_length_arr: npt.NDArray[np.float32],
    edges_angle_sum_arr: npt.NDArray[np.float32],
    edges_imp_factor_arr: npt.NDArray[np.float32],
    edges_in_bearing_arr: npt.NDArray[np.float32],
    edges_out_bearing_arr: npt.NDArray[np.float32],
    node_edge_map: dict[int, list[int]],
    data_map_x_arr: npt.NDArray[np.float32],
    data_map_y_arr: npt.NDArray[np.float32],
    data_map_nearest_arr: npt.NDArray[np.int_],
    data_map_next_nearest_arr: npt.NDArray[np.int_],
    max_dist: np.float32,
    jitter_scale: np.float32 = np.float32(0.0),
    angular: bool = False,
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.float32]]:
    """
    Aggregate data points relative to a src index.

    Shortest tree dijkstra returns predecessor map is based on impedance heuristic - i.e. angular vs not angular.
    Shortest path distances are in metres and are used for defining max distances regardless.
    """
    # this function is typically called iteratively, so do type checks from parent methods
    # run the shortest tree dijkstra
    # keep in mind that predecessor map is based on impedance heuristic - which can be different from metres
    # NOTE -> use np.inf for max distance so as to explore all paths
    # In some cases the predecessor nodes will be within reach even if the closest node is not
    # Total distance is checked later
    (
        _visited_nodes,
        _preds,
        short_dist,
        _simpl_dist,
        _cycles,
        _origin_seg,
        _last_seg,
        _out_bearings,
        _visited_edges,
    ) = centrality.shortest_path_tree(
        edges_start_arr,
        edges_end_arr,
        edges_length_arr,
        edges_angle_sum_arr,
        edges_imp_factor_arr,
        edges_in_bearing_arr,
        edges_out_bearing_arr,
        node_edge_map,
        netw_src_idx,
        max_dist=max_dist,
        jitter_scale=jitter_scale,
        angular=angular,
    )
    # arrays for writing the reachable data points and their distances
    d_n = len(data_map_x_arr)
    reachable_data: npt.NDArray[np.bool_] = np.full(d_n, False)
    reachable_data_dist: npt.NDArray[np.float32] = np.full(d_n, np.inf)
    # iterate the data points
    for data_idx in range(d_n):
        # find the primary assigned network index for the data point
        nearest_netw_idx = data_map_nearest_arr[data_idx]
        if nearest_netw_idx != -1:
            # if the assigned network node is within the threshold
            dist = short_dist[nearest_netw_idx]
            if dist < max_dist:
                # get the distance from the data point to the network node
                d_d = np.hypot(
                    data_map_x_arr[data_idx] - nodes_x_arr[nearest_netw_idx],
                    data_map_y_arr[data_idx] - nodes_y_arr[nearest_netw_idx],
                )
                # only assign distance if within max distance
                total_dist = dist + d_d
                if total_dist <= max_dist:
                    reachable_data[data_idx] = True
                    reachable_data_dist[data_idx] = total_dist
        # the next-nearest may offer a closer route depending on the direction the shortest path approaches from
        next_nearest_netw_idx = data_map_next_nearest_arr[data_idx]
        if next_nearest_netw_idx != -1:
            # if the assigned network node is within the threshold
            dist = short_dist[next_nearest_netw_idx]
            if dist < max_dist:
                # get the distance from the data point to the network node
                d_d = np.hypot(
                    data_map_x_arr[data_idx] - nodes_x_arr[next_nearest_netw_idx],
                    data_map_y_arr[data_idx] - nodes_y_arr[next_nearest_netw_idx],
                )
                # only assign distance if within max distance
                # AND only if closer than other direction
                total_dist = dist + d_d
                if total_dist <= max_dist and total_dist < reachable_data_dist[data_idx]:
                    reachable_data[data_idx] = True
                    reachable_data_dist[data_idx] = total_dist

    # note that some entries will be nan values if the max distance was exceeded
    return reachable_data, reachable_data_dist


@njit(cache=True, fastmath=config.FASTMATH, nogil=True, parallel=True)
def accessibility(
    nodes_x_arr: npt.NDArray[np.float32],
    nodes_y_arr: npt.NDArray[np.float32],
    nodes_live_arr: npt.NDArray[np.bool_],
    edges_start_arr: npt.NDArray[np.int_],
    edges_end_arr: npt.NDArray[np.int_],
    edges_length_arr: npt.NDArray[np.float32],
    edges_angle_sum_arr: npt.NDArray[np.float32],
    edges_imp_factor_arr: npt.NDArray[np.float32],
    edges_in_bearing_arr: npt.NDArray[np.float32],
    edges_out_bearing_arr: npt.NDArray[np.float32],
    node_edge_map: dict[int, list[int]],
    data_map_x_arr: npt.NDArray[np.float32],
    data_map_y_arr: npt.NDArray[np.float32],
    data_map_nearest_arr: npt.NDArray[np.int_],
    data_map_next_nearest_arr: npt.NDArray[np.int_],
    data_id_arr: npt.NDArray[np.int_],
    distances: npt.NDArray[np.int_],
    betas: npt.NDArray[np.float32],
    max_curve_wts: npt.NDArray[np.float32],
    landuse_encodings: npt.NDArray[np.int_],
    accessibility_keys: npt.NDArray[np.int_],
    jitter_scale: np.float32 = np.float32(0.0),
    angular: bool = False,
    progress_proxy: Optional[Any] = None,
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.float32]]:
    """
    Aggregate landuses.
    """
    common.check_distances_and_betas(distances, betas)
    # check landuse encodings
    if len(landuse_encodings) == 0:
        raise ValueError("Mixed use metrics require an array of landuse labels.")
    if len(landuse_encodings) != data_map_x_arr.shape[0]:
        raise ValueError("The number of landuse encodings does not match the number of data points.")
    if len(accessibility_keys) == 0:
        raise ValueError("Please specify at least one landuse key.")
    common.check_categorical_data(landuse_encodings)
    max_ac_key: np.int_ = np.nanmax(landuse_encodings)
    if np.nanmin(accessibility_keys) < 0 or np.max(accessibility_keys) > max_ac_key:
        raise ValueError("Negative or out of range accessibility key encountered. Keys must match class encodings.")
    for i_idx, i_key in enumerate(accessibility_keys):
        for j_idx, j_key in enumerate(accessibility_keys):
            if j_idx > i_idx:
                if i_key == j_key:
                    raise ValueError("Duplicate accessibility key.")
    # check that beta_wt_clip is between 0 and 1 if provided
    if np.any(max_curve_wts < 0) or np.any(max_curve_wts > 1) or max_curve_wts.shape != distances.shape:
        raise ValueError(
            "max_curve_wts should be between 0 and 1 and should be of the same length as the distances / betas params."
        )
    # establish variables
    netw_n = len(node_edge_map)
    d_n = len(distances)
    global_max_dist: np.float32 = np.float32(np.nanmax(distances))
    # setup data structures
    accessibility_data: npt.NDArray[np.int_] = np.full((len(accessibility_keys), d_n, netw_n), 0, dtype=np.int_)  # int
    accessibility_data_wt: npt.NDArray[np.float32] = np.full(
        (len(accessibility_keys), d_n, netw_n), 0.0, dtype=np.float32
    )
    # iterate through each vert and aggregate
    # parallelise over n nodes:
    # each distance or stat array index is therefore only touched by one thread at a time
    # i.e. no need to use inner array deductions as with centralities
    for netw_src_idx in prange(netw_n):  # pylint: disable=not-an-iterable
        if progress_proxy is not None:
            progress_proxy.update(1)
        # only compute for live nodes
        if not nodes_live_arr[netw_src_idx]:
            continue
        # generate the reachable classes and their respective distances
        # these are non-unique - i.e. simply the class of each data point within the maximum distance
        # the aggregate_to_src_idx method will choose the closer direction of approach to a data point
        # from the nearest or next-nearest network node (calculated once globally, prior to local_landuses method)
        reachable_data, reachable_data_dist = aggregate_to_src_idx(
            netw_src_idx,
            nodes_x_arr,
            nodes_y_arr,
            edges_start_arr,
            edges_end_arr,
            edges_length_arr,
            edges_angle_sum_arr,
            edges_imp_factor_arr,
            edges_in_bearing_arr,
            edges_out_bearing_arr,
            node_edge_map,
            data_map_x_arr,
            data_map_y_arr,
            data_map_nearest_arr,
            data_map_next_nearest_arr,
            global_max_dist,
            jitter_scale=jitter_scale,
            angular=angular,
        )
        # iterate the reachable indices and related distances
        # sort by increasing distance re: deduplication via data keys
        # because these are sorted, no need to deduplicate by respective distance thresholds
        dist_inc_idx: npt.NDArray[np.int_] = np.argsort(reachable_data_dist)
        data_id_dupes: set[int] = set()
        for data_idx in dist_inc_idx:
            reachable = reachable_data[data_idx]
            data_dist = reachable_data_dist[data_idx]
            # some indices will be NaN if beyond max threshold distance - so check for infinity
            # this happens when within radial max distance, but beyond network max distance
            if not reachable:
                continue
            # check for duplicate instances
            data_id = data_id_arr[data_idx]
            if data_id != -1:
                if data_id in data_id_dupes:
                    continue
            # get the class category in integer form
            # all class codes were encoded to sequential integers - these correspond to the array indices
            cl_code = int(landuse_encodings[int(data_idx)])
            # iterate the distance dimensions
            for d_idx, (dist, beta, max_wt) in enumerate(zip(distances, betas, max_curve_wts)):
                # increment class counts at respective distances if the distance is less than current dist
                if data_dist <= dist:
                    # if within distance, and if in accessibility keys, then aggregate accessibility too
                    for ac_idx, ac_code in enumerate(accessibility_keys):
                        if ac_code == cl_code:
                            data_id_dupes.add(data_id)
                            accessibility_data[ac_idx, d_idx, netw_src_idx] += 1
                            accessibility_data_wt[ac_idx, d_idx, netw_src_idx] += common.clipped_beta_wt(
                                beta, max_wt, data_dist
                            )
                            # if a match was found, then no need to check others
                            break

    return accessibility_data, accessibility_data_wt


@njit(cache=True, fastmath=config.FASTMATH, nogil=True, parallel=True)
def mixed_uses(
    nodes_x_arr: npt.NDArray[np.float32],
    nodes_y_arr: npt.NDArray[np.float32],
    nodes_live_arr: npt.NDArray[np.bool_],
    edges_start_arr: npt.NDArray[np.int_],
    edges_end_arr: npt.NDArray[np.int_],
    edges_length_arr: npt.NDArray[np.float32],
    edges_angle_sum_arr: npt.NDArray[np.float32],
    edges_imp_factor_arr: npt.NDArray[np.float32],
    edges_in_bearing_arr: npt.NDArray[np.float32],
    edges_out_bearing_arr: npt.NDArray[np.float32],
    node_edge_map: dict[int, list[int]],
    data_map_x_arr: npt.NDArray[np.float32],
    data_map_y_arr: npt.NDArray[np.float32],
    data_map_nearest_arr: npt.NDArray[np.int_],
    data_map_next_nearest_arr: npt.NDArray[np.int_],
    distances: npt.NDArray[np.int_],
    betas: npt.NDArray[np.float32],
    max_curve_wts: npt.NDArray[np.float32],
    landuse_encodings: npt.NDArray[np.int_] = np.array([], dtype=np.int_),
    qs: npt.NDArray[np.float32] = np.array([], dtype=np.float32),
    mixed_use_hill_keys: npt.NDArray[np.int_] = np.array([], dtype=np.int_),
    mixed_use_other_keys: npt.NDArray[np.int_] = np.array([], dtype=np.int_),
    cl_disparity_wt_matrix: npt.NDArray[np.float32] = np.array(
        np.full((0, 0), np.nan), dtype=np.float32
    ),  # pylint: disable=line-too-long
    jitter_scale: np.float32 = np.float32(0.0),
    angular: bool = False,
    progress_proxy: Optional[Any] = None,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """
    Aggregate landuses.
    """
    common.check_distances_and_betas(distances, betas)
    # check landuse encodings
    if len(landuse_encodings) == 0:
        raise ValueError("Mixed use metrics or land-use accessibilities require an array of landuse labels.")
    if len(landuse_encodings) != data_map_x_arr.shape[0]:
        raise ValueError("The number of landuse encodings does not match the number of data points.")
    common.check_categorical_data(landuse_encodings)
    # catch completely missing metrics
    if len(mixed_use_hill_keys) == 0 and len(mixed_use_other_keys) == 0:
        raise ValueError("Please specify at least one mixed use metric to compute.")
    # catch missing qs
    if len(mixed_use_hill_keys) != 0 and len(qs) == 0:
        raise ValueError("Hill diversity measures require that at least one value of q is specified.")
    # negative qs caught by hill diversity methods
    # check various problematic key combinations
    if len(mixed_use_hill_keys) != 0:
        if np.nanmin(mixed_use_hill_keys) < 0 or np.max(mixed_use_hill_keys) > 3:
            raise ValueError('Mixed-use "hill" keys out of range of 0:4.')
    if len(mixed_use_other_keys) != 0:
        if np.nanmin(mixed_use_other_keys) < 0 or np.max(mixed_use_other_keys) > 2:
            raise ValueError('Mixed-use "other" keys out of range of 0:3.')
    for i_idx, i_key in enumerate(mixed_use_hill_keys):
        for j_idx, j_key in enumerate(mixed_use_hill_keys):
            if j_idx > i_idx:
                if i_key == j_key:
                    raise ValueError('Duplicate mixed-use "hill" key.')
    for i_idx, i_key in enumerate(mixed_use_other_keys):
        for j_idx, j_key in enumerate(mixed_use_other_keys):
            if j_idx > i_idx:
                if i_key == j_key:
                    raise ValueError('Duplicate mixed-use "other" key.')

    def disp_check(disp_matrix):  # type: ignore # numba can't handle nested function with type annotation
        # the length of the disparity matrix vis-a-vis unique landuses is tested in underlying diversity functions
        if disp_matrix.ndim != 2 or disp_matrix.shape[0] != disp_matrix.shape[1]:
            raise ValueError("The disparity matrix must be a square NxN matrix.")
        if disp_matrix.shape[0] == 0:
            raise ValueError("Hill disparity and Rao pairwise measures requires a class disparity weights matrix.")

    # check that missing or malformed disparity weights matrices are caught
    for key_idx in mixed_use_hill_keys:
        if key_idx == 3:  # hill disparity
            disp_check(cl_disparity_wt_matrix)
    for key_idx in mixed_use_other_keys:
        if key_idx == 2:  # raos pairwise
            disp_check(cl_disparity_wt_matrix)
    # check that beta_wt_clip is between 0 and 1 if provided
    if np.any(max_curve_wts < 0) or np.any(max_curve_wts > 1) or max_curve_wts.shape != distances.shape:
        raise ValueError(
            "max_curve_wts should be between 0 and 1 and should be of the same length as the distances / betas params."
        )
    # establish variables
    netw_n = len(node_edge_map)
    d_n = len(distances)
    q_n = len(qs)
    global_max_dist: np.float32 = np.float32(np.nanmax(distances))
    # setup data structures
    # hill mixed uses are structured separately to take values of q into account
    mixed_use_hill_data: npt.NDArray[np.float32] = np.full((4, q_n, d_n, netw_n), 0.0, dtype=np.float32)  # 4 dim
    mixed_use_other_data: npt.NDArray[np.float32] = np.full((3, d_n, netw_n), 0.0, dtype=np.float32)  # 3 dim
    # iterate through each vert and aggregate
    # parallelise over n nodes:
    # each distance or stat array index is therefore only touched by one thread at a time
    # i.e. no need to use inner array deductions as with centralities
    for netw_src_idx in prange(netw_n):  # pylint: disable=not-an-iterable
        if progress_proxy is not None:
            progress_proxy.update(1)
        # only compute for live nodes
        if not nodes_live_arr[netw_src_idx]:
            continue
        # generate the reachable classes and their respective distances
        # these are non-unique - i.e. simply the class of each data point within the maximum distance
        # the aggregate_to_src_idx method will choose the closer direction of approach to a data point
        # from the nearest or next-nearest network node (calculated once globally, prior to local_landuses method)
        reachable_data, reachable_data_dist = aggregate_to_src_idx(
            netw_src_idx,
            nodes_x_arr,
            nodes_y_arr,
            edges_start_arr,
            edges_end_arr,
            edges_length_arr,
            edges_angle_sum_arr,
            edges_imp_factor_arr,
            edges_in_bearing_arr,
            edges_out_bearing_arr,
            node_edge_map,
            data_map_x_arr,
            data_map_y_arr,
            data_map_nearest_arr,
            data_map_next_nearest_arr,
            global_max_dist,
            jitter_scale=jitter_scale,
            angular=angular,
        )
        # LANDUSES
        mu_max_unique_cl = int(landuse_encodings.max() + 1)
        # counts of each class type (array length per max unique classes - not just those within max distance)
        classes_counts: npt.NDArray[np.int_] = np.full((d_n, mu_max_unique_cl), 0)
        # nearest of each class type (likewise)
        classes_nearest: npt.NDArray[np.float32] = np.full((d_n, mu_max_unique_cl), np.inf)
        # iterate the reachable indices and related distances
        for data_idx, (reachable, data_dist) in enumerate(zip(reachable_data, reachable_data_dist)):
            if not reachable:
                continue
            # get the class category in integer form
            # all class codes were encoded to sequential integers - these correspond to the array indices
            cl_code = int(landuse_encodings[int(data_idx)])
            # iterate the distance dimensions
            for d_idx, (dist, beta, max_wt) in enumerate(zip(distances, betas, max_curve_wts)):
                # increment class counts at respective distances if the distance is less than current dist
                if data_dist <= dist:
                    classes_counts[d_idx, cl_code] += 1
                    # if distance is nearer, update the nearest distance array too
                    if data_dist < classes_nearest[d_idx, cl_code]:
                        classes_nearest[d_idx, cl_code] = data_dist
        # mixed uses can be calculated now that the local class counts are aggregated
        # iterate the distances and betas
        for d_idx, (beta, max_wt) in enumerate(zip(betas, max_curve_wts)):
            cl_counts = classes_counts[d_idx]
            cl_nearest = classes_nearest[d_idx]
            # mu keys determine which metrics to compute
            # don't confuse with indices
            # previously used dynamic indices in data structures - but obtuse if irregularly ordered keys
            for mu_hill_key in mixed_use_hill_keys:
                for q_idx, q_key in enumerate(qs):
                    if mu_hill_key == 0:
                        mixed_use_hill_data[0, q_idx, d_idx, netw_src_idx] = diversity.hill_diversity(cl_counts, q_key)
                    elif mu_hill_key == 1:
                        mixed_use_hill_data[
                            1, q_idx, d_idx, netw_src_idx
                        ] = diversity.hill_diversity_branch_distance_wt(cl_counts, cl_nearest, q_key, beta, max_wt)
                    elif mu_hill_key == 2:
                        mixed_use_hill_data[
                            2, q_idx, d_idx, netw_src_idx
                        ] = diversity.hill_diversity_pairwise_distance_wt(cl_counts, cl_nearest, q_key, beta, max_wt)
                    # land-use classification disparity hill diversity
                    # the wt matrix can be used without mapping because cl_counts is based on all classes
                    # regardless of whether they are reachable
                    elif mu_hill_key == 3:
                        mixed_use_hill_data[
                            3, q_idx, d_idx, netw_src_idx
                        ] = diversity.hill_diversity_pairwise_matrix_wt(
                            cl_counts, wt_matrix=cl_disparity_wt_matrix, q=q_key
                        )
            for mu_other_key in mixed_use_other_keys:
                if mu_other_key == 0:
                    mixed_use_other_data[0, d_idx, netw_src_idx] = diversity.shannon_diversity(cl_counts)
                elif mu_other_key == 1:
                    mixed_use_other_data[1, d_idx, netw_src_idx] = diversity.gini_simpson_diversity(cl_counts)
                elif mu_other_key == 2:
                    mixed_use_other_data[2, d_idx, netw_src_idx] = diversity.raos_quadratic_diversity(
                        cl_counts, wt_matrix=cl_disparity_wt_matrix
                    )
    # send the data back in the same types and same order as the original keys - convert to int for indexing
    mu_hill_k_int: npt.NDArray[np.int_] = np.full(len(mixed_use_hill_keys), 0, dtype=np.int_)
    for idx, mu_hill_key in enumerate(mixed_use_hill_keys):
        mu_hill_k_int[idx] = mu_hill_key
    mu_other_k_int: npt.NDArray[np.int_] = np.full(len(mixed_use_other_keys), 0, dtype=np.int_)
    for idx, mu_oth_key in enumerate(mixed_use_other_keys):
        mu_other_k_int[idx] = mu_oth_key

    return mixed_use_hill_data[mu_hill_k_int], mixed_use_other_data[mu_other_k_int]


@njit(cache=True, fastmath=config.FASTMATH, nogil=True, parallel=True)
def aggregate_stats(
    nodes_x_arr: npt.NDArray[np.float32],
    nodes_y_arr: npt.NDArray[np.float32],
    nodes_live_arr: npt.NDArray[np.bool_],
    edges_start_arr: npt.NDArray[np.int_],
    edges_end_arr: npt.NDArray[np.int_],
    edges_length_arr: npt.NDArray[np.float32],
    edges_angle_sum_arr: npt.NDArray[np.float32],
    edges_imp_factor_arr: npt.NDArray[np.float32],
    edges_in_bearing_arr: npt.NDArray[np.float32],
    edges_out_bearing_arr: npt.NDArray[np.float32],
    node_edge_map: dict[int, list[int]],
    data_map_x_arr: npt.NDArray[np.float32],
    data_map_y_arr: npt.NDArray[np.float32],
    data_map_nearest_arr: npt.NDArray[np.int_],
    data_map_next_nearest_arr: npt.NDArray[np.int_],
    data_id_arr: npt.NDArray[np.int_],
    distances: npt.NDArray[np.int_],
    betas: npt.NDArray[np.float32],
    numerical_arrays: npt.NDArray[np.float32] = np.array(np.full((0, 0), np.nan, dtype=np.float32)),
    jitter_scale: np.float32 = np.float32(0.0),
    angular: bool = False,
    progress_proxy: Any = None,
) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
]:
    """
    Aggregate stats.

    NODE MAP:
    0 - x
    1 - y
    2 - live
    EDGE MAP:
    0 - start node
    1 - end node
    2 - length in metres
    3 - sum of angular travel along length
    4 - impedance factor
    5 - in bearing
    6 - out bearing
    DATA MAP:
    0 - x
    1 - y
    2 - assigned network index - nearest
    3 - assigned network index - next-nearest
    """
    common.check_distances_and_betas(distances, betas)
    # when passing an empty 2d array to numba, use: np.array(np.full((0, 0), np.nan, dtype=np.float32))
    if numerical_arrays.shape[1] != data_map_x_arr.shape[0]:
        raise ValueError("The length of the numerical data arrays do not match the length of the data map.")
    common.check_numerical_data(numerical_arrays)
    # establish variables
    netw_n = len(node_edge_map)
    d_n = len(distances)
    n_n = len(numerical_arrays)
    global_max_dist: np.float32 = np.float32(np.nanmax(distances))
    # setup data structures
    stats_sum: npt.NDArray[np.float32] = np.full((n_n, d_n, netw_n), 0.0)
    stats_sum_wt: npt.NDArray[np.float32] = np.full((n_n, d_n, netw_n), 0.0)
    stats_mean: npt.NDArray[np.float32] = np.full((n_n, d_n, netw_n), np.nan)
    stats_mean_wt: npt.NDArray[np.float32] = np.full((n_n, d_n, netw_n), np.nan)
    stats_count: npt.NDArray[np.float32] = np.full((n_n, d_n, netw_n), 0.0)
    stats_count_wt: npt.NDArray[np.float32] = np.full((n_n, d_n, netw_n), 0.0)
    stats_variance: npt.NDArray[np.float32] = np.full((n_n, d_n, netw_n), np.nan)
    stats_variance_wt: npt.NDArray[np.float32] = np.full((n_n, d_n, netw_n), np.nan)
    stats_max: npt.NDArray[np.float32] = np.full((n_n, d_n, netw_n), np.nan)
    stats_min: npt.NDArray[np.float32] = np.full((n_n, d_n, netw_n), np.nan)
    # iterate through each vert and aggregate
    # parallelise over n nodes:
    # each distance or stat array index is therefore only touched by one thread at a time
    # i.e. no need to use inner array deductions as with centralities
    for netw_src_idx in prange(netw_n):  # pylint: disable=not-an-iterable
        if progress_proxy is not None:
            progress_proxy.update(1)
        # only compute for live nodes
        if not nodes_live_arr[netw_src_idx]:
            continue
        # generate the reachable classes and their respective distances
        # these are non-unique - i.e. simply the class of each data point within the maximum distance
        # the aggregate_to_src_idx method will choose the closer direction of approach to a data point
        # from the nearest or next-nearest network node (calculated once globally, prior to local_landuses method)
        reachable_data, reachable_data_dist = aggregate_to_src_idx(
            netw_src_idx,
            nodes_x_arr,
            nodes_y_arr,
            edges_start_arr,
            edges_end_arr,
            edges_length_arr,
            edges_angle_sum_arr,
            edges_imp_factor_arr,
            edges_in_bearing_arr,
            edges_out_bearing_arr,
            node_edge_map,
            data_map_x_arr,
            data_map_y_arr,
            data_map_nearest_arr,
            data_map_next_nearest_arr,
            global_max_dist,
            jitter_scale=jitter_scale,
            angular=angular,
        )
        # IDW
        # the order of the loops matters because the nested aggregations happen per distance per numerical array
        # iterate the reachable indices and related distances
        # sort by increasing distance re: deduplication via data keys
        # because these are sorted, no need to deduplicate by respective distance thresholds
        dist_inc_idx: npt.NDArray[np.int_] = np.argsort(reachable_data_dist)
        data_id_dupes: set[int] = set()
        for data_idx in dist_inc_idx:
            reachable = reachable_data[data_idx]
            data_dist = reachable_data_dist[data_idx]
            # some indices will be NaN if beyond max threshold distance - so check for infinity
            # this happens when within radial max distance, but beyond network max distance
            if not reachable:
                continue
            # check for duplicate instances
            data_id = data_id_arr[data_idx]
            if data_id != -1:
                if data_id in data_id_dupes:
                    continue
            # iterate the numerical arrays dimension
            for num_idx in range(n_n):
                # some values will be NaN
                num = numerical_arrays[num_idx, int(data_idx)]
                if np.isnan(num):
                    continue
                # iterate the distance dimensions
                for d_idx, (dist, beta) in enumerate(zip(distances, betas)):
                    # increment mean aggregations at respective distances if the distance is less than current dist
                    if data_dist <= dist:
                        data_id_dupes.add(data_id)
                        # aggregate
                        stats_sum[num_idx, d_idx, netw_src_idx] += num
                        stats_count[num_idx, d_idx, netw_src_idx] += 1
                        stats_sum_wt[num_idx, d_idx, netw_src_idx] += num * np.exp(-beta * data_dist)
                        stats_count_wt[num_idx, d_idx, netw_src_idx] += np.exp(-beta * data_dist)
                        # max
                        if np.isnan(stats_max[num_idx, d_idx, netw_src_idx]):
                            stats_max[num_idx, d_idx, netw_src_idx] = num
                        elif num > stats_max[num_idx, d_idx, netw_src_idx]:
                            stats_max[num_idx, d_idx, netw_src_idx] = num
                        # min
                        if np.isnan(stats_min[num_idx, d_idx, netw_src_idx]):
                            stats_min[num_idx, d_idx, netw_src_idx] = num
                        elif num < stats_min[num_idx, d_idx, netw_src_idx]:
                            stats_min[num_idx, d_idx, netw_src_idx] = num
        # finalise mean calculations - this is happening for a single netw_src_idx, so fairly fast
        for num_idx in range(n_n):
            for d_idx in range(d_n):
                # use divide so that division through zero doesn't trigger
                stats_mean[num_idx, d_idx, netw_src_idx] = np.divide(
                    stats_sum[num_idx, d_idx, netw_src_idx],
                    stats_count[num_idx, d_idx, netw_src_idx],
                )
                stats_mean_wt[num_idx, d_idx, netw_src_idx] = np.divide(
                    stats_sum_wt[num_idx, d_idx, netw_src_idx],
                    stats_count_wt[num_idx, d_idx, netw_src_idx],
                )
        # calculate variances - counts are already computed per above
        # weighted version is IDW by division through equivalently weighted counts above
        # iterate the reachable indices and related distances
        # sort by increasing distance re: deduplication via data keys
        dist_inc_idx: npt.NDArray[np.int_] = np.argsort(reachable_data_dist)
        data_id_dupes: set[int] = set()
        for data_idx in dist_inc_idx:
            reachable = reachable_data[data_idx]
            data_dist = reachable_data_dist[data_idx]
            # some indices will be NaN if beyond max threshold distance - so check for infinity
            # this happens when within radial max distance, but beyond network max distance
            if not reachable:
                continue
            # check for duplicate instances
            data_id = data_id_arr[data_idx]
            if data_id != -1:
                if data_id in data_id_dupes:
                    continue
            # iterate the numerical arrays dimension
            for num_idx in range(n_n):
                # some values will be NaN
                num = numerical_arrays[num_idx, int(data_idx)]
                if np.isnan(num):
                    continue
                # iterate the distance dimensions
                for d_idx, (dist, beta) in enumerate(zip(distances, betas)):
                    # increment variance aggregations at respective distances if the distance is less than current dist
                    if data_dist <= dist:
                        data_id_dupes.add(data_id)
                        # aggregate
                        if np.isnan(stats_variance[num_idx, d_idx, netw_src_idx]):
                            stats_variance[num_idx, d_idx, netw_src_idx] = np.square(
                                num - stats_mean[num_idx, d_idx, netw_src_idx]
                            )
                            stats_variance_wt[num_idx, d_idx, netw_src_idx] = np.square(
                                num - stats_mean_wt[num_idx, d_idx, netw_src_idx]
                            ) * np.exp(-beta * data_dist)
                        else:
                            stats_variance[num_idx, d_idx, netw_src_idx] += np.square(
                                num - stats_mean[num_idx, d_idx, netw_src_idx]
                            )
                            stats_variance_wt[num_idx, d_idx, netw_src_idx] += np.square(
                                num - stats_mean_wt[num_idx, d_idx, netw_src_idx]
                            ) * np.exp(-beta * data_dist)
        # finalise variance calculations
        for num_idx in range(n_n):
            for d_idx in range(d_n):
                # use divide so that division through zero doesn't trigger
                stats_variance[num_idx, d_idx, netw_src_idx] = np.divide(
                    stats_variance[num_idx, d_idx, netw_src_idx],
                    stats_count[num_idx, d_idx, netw_src_idx],
                )
                stats_variance_wt[num_idx, d_idx, netw_src_idx] = np.divide(
                    stats_variance_wt[num_idx, d_idx, netw_src_idx],
                    stats_count_wt[num_idx, d_idx, netw_src_idx],
                )

    # pylint: disable=duplicate-code
    return (
        stats_sum,
        stats_sum_wt,
        stats_mean,
        stats_mean_wt,
        stats_variance,
        stats_variance_wt,
        stats_max,
        stats_min,
    )
