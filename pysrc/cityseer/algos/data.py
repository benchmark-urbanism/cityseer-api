from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


# @njit(cache=True, fastmath=config.FASTMATH, nogil=True, parallel=True)
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
