# pyright: basic
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from cityseer import config, rustalgos
from cityseer.metrics import layers, networks
from cityseer.tools import graphs, io, mock


def test_aggregate_to_src_idx(primal_graph):
    for max_dist in [400, 750]:
        for deduplicate in [False, True]:
            # generate data
            _nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(primal_graph, 3395)
            data_gdf = mock.mock_data_gdf(primal_graph)
            if deduplicate is False:
                data_map, data_gdf = layers.assign_gdf_to_network(
                    data_gdf, network_structure, max_dist, data_id_col=None
                )
            else:
                data_map, data_gdf = layers.assign_gdf_to_network(
                    data_gdf, network_structure, max_dist, data_id_col="data_id"
                )
            # in this case, use same assignment max dist as search max dist
            # for debugging
            # from cityseer.tools import plot
            # plot.plot_network_structure(network_structure, data_map)
            for angular in [True, False]:
                for netw_src_idx in network_structure.node_indices():
                    # aggregate to src...
                    reachable_entries = data_map.aggregate_to_src_idx(
                        netw_src_idx, network_structure, max_dist, angular=angular
                    )
                    # compare to manual checks on distances:
                    # get the network distances
                    if angular is False:
                        _nodes, tree_map = network_structure.dijkstra_tree_shortest(netw_src_idx, max_dist)
                    else:
                        _nodes, tree_map = network_structure.dijkstra_tree_simplest(netw_src_idx, max_dist)
                    # verify distances vs. the max
                    for data_key, data_entry in data_map.entries.items():
                        # nearest
                        if data_entry.nearest_assign is not None:
                            nearest_netw_node = network_structure.get_node_payload(data_entry.nearest_assign)
                            nearest_assign_dist = tree_map[data_entry.nearest_assign].short_dist
                            # add tail
                            if not np.isposinf(nearest_assign_dist):
                                nearest_assign_dist += nearest_netw_node.coord.hypot(data_entry.coord)
                        else:
                            nearest_assign_dist = np.inf
                        # next nearest
                        if data_entry.next_nearest_assign is not None:
                            next_nearest_netw_node = network_structure.get_node_payload(data_entry.next_nearest_assign)
                            next_nearest_assign_dist = tree_map[data_entry.next_nearest_assign].short_dist
                            # add tail
                            if not np.isposinf(next_nearest_assign_dist):
                                next_nearest_assign_dist += next_nearest_netw_node.coord.hypot(data_entry.coord)
                        else:
                            next_nearest_assign_dist = np.inf
                        # check deduplication - 49 is the closest, so others should not make it through
                        # checks
                        if nearest_assign_dist > max_dist and next_nearest_assign_dist > max_dist:
                            assert data_key not in reachable_entries
                        elif deduplicate and data_key in ["45", "46", "47", "48"]:
                            assert data_key not in reachable_entries and "49" in reachable_entries
                        elif np.isposinf(nearest_assign_dist) and next_nearest_assign_dist < max_dist:
                            assert reachable_entries[data_key] - next_nearest_assign_dist < config.ATOL
                        elif np.isposinf(next_nearest_assign_dist) and nearest_assign_dist < max_dist:
                            assert reachable_entries[data_key] - nearest_assign_dist < config.ATOL
                        else:
                            assert (
                                reachable_entries[data_key] - min(nearest_assign_dist, next_nearest_assign_dist)
                                < config.ATOL
                            )
    # reuse the last instance of data_gdf and check that recomputing is not happening if already assigned
    assert "nearest_assign" in data_gdf.columns
    assert "next_nearest_assign" in data_gdf.columns
    # override with nonsense value
    data_gdf["nearest_assign"] = 0
    data_gdf["next_nearest_assign"] = 0
    # check that these have not been replaced
    data_map, data_gdf = layers.assign_gdf_to_network(data_gdf, network_structure, max_dist, data_id_col=None)
    assert np.all(data_gdf["nearest_assign"].values == 0)
    assert np.all(data_gdf["next_nearest_assign"].values == 0)


def test_accessibility(primal_graph):
    # generate node and edge maps
    _nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(primal_graph, 3395)
    data_gdf = mock.mock_landuse_categorical_data(primal_graph, random_seed=13)
    distances = [200, 400, 800, 1600]
    max_dist = max(distances)
    data_map, data_gdf = layers.assign_gdf_to_network(data_gdf, network_structure, max_dist, data_id_col="data_id")
    landuses_map = data_gdf["categorical_landuses"].to_dict()
    # all datapoints and types are completely unique except for the last five - which all point to the same source
    accessibility_keys = ["a", "b", "c", "z"]  # the duplicate keys are per landuse 'z'
    # generate
    accessibilities = data_map.accessibility(
        network_structure,
        landuses_map,
        accessibility_keys,
        distances,
    )
    # test manual metrics against all nodes
    betas = rustalgos.betas_from_distances(distances)
    for dist, beta in zip(distances, betas):
        for src_idx in network_structure.node_indices():  # type: ignore
            # aggregate
            a_nw = 0
            b_nw = 0
            c_nw = 0
            z_nw = 0
            a_wt = 0
            b_wt = 0
            c_wt = 0
            z_wt = 0
            a_dist = np.nan
            b_dist = np.nan
            c_dist = np.nan
            z_dist = np.nan
            # iterate reachable
            reachable_entries = data_map.aggregate_to_src_idx(src_idx, network_structure, max_dist)
            for data_key, data_dist in reachable_entries.items():
                # double check distance is within threshold
                assert data_dist <= max_dist
                if data_dist <= dist:
                    data_class = landuses_map[data_key]
                    # aggregate accessibility codes
                    if data_class == "a":
                        a_nw += 1
                        a_wt += np.exp(-beta * data_dist)
                        if np.isnan(a_dist) or data_dist < a_dist:
                            a_dist = data_dist
                    elif data_class == "b":
                        b_nw += 1
                        b_wt += np.exp(-beta * data_dist)
                        if np.isnan(b_dist) or data_dist < b_dist:
                            b_dist = data_dist
                    elif data_class == "c":
                        c_nw += 1
                        c_wt += np.exp(-beta * data_dist)
                        if np.isnan(c_dist) or data_dist < c_dist:
                            c_dist = data_dist
                    elif data_class == "z":
                        z_nw += 1
                        z_wt += np.exp(-beta * data_dist)
                        if np.isnan(z_dist) or data_dist < z_dist:
                            z_dist = data_dist
            # assertions
            assert accessibilities["a"].unweighted[dist][src_idx] - a_nw < config.ATOL
            assert accessibilities["b"].unweighted[dist][src_idx] - b_nw < config.ATOL
            assert accessibilities["c"].unweighted[dist][src_idx] - c_nw < config.ATOL
            assert accessibilities["z"].unweighted[dist][src_idx] - z_nw < config.ATOL
            assert accessibilities["a"].weighted[dist][src_idx] - a_wt < config.ATOL
            assert accessibilities["b"].weighted[dist][src_idx] - b_wt < config.ATOL
            assert accessibilities["c"].weighted[dist][src_idx] - c_wt < config.ATOL
            assert accessibilities["z"].weighted[dist][src_idx] - z_wt < config.ATOL
            if dist == max(distances):
                if np.isfinite(a_dist):
                    assert accessibilities["a"].distance[dist][src_idx] - a_dist < config.ATOL
                else:
                    assert np.isnan(a_dist) and np.isnan(accessibilities["a"].distance[dist][src_idx])
                if np.isfinite(b_dist):
                    assert accessibilities["b"].distance[dist][src_idx] - b_dist < config.ATOL
                else:
                    assert np.isnan(b_dist) and np.isnan(accessibilities["b"].distance[dist][src_idx])
                if np.isfinite(c_dist):
                    assert accessibilities["c"].distance[dist][src_idx] - c_dist < config.ATOL
                else:
                    assert np.isnan(c_dist) and np.isnan(accessibilities["c"].distance[dist][src_idx])
                if np.isfinite(z_dist):
                    assert accessibilities["z"].distance[dist][src_idx] - z_dist < config.ATOL
                else:
                    assert np.isnan(z_dist) and np.isnan(accessibilities["z"].distance[dist][src_idx])
            else:
                assert dist not in accessibilities["a"].distance
                assert dist not in accessibilities["b"].distance
                assert dist not in accessibilities["c"].distance
                assert dist not in accessibilities["z"].distance
            # check for deduplication
            assert z_nw in [0, 1]
            assert z_wt <= 1
    # setup dual data
    accessibilities_ang = data_map.accessibility(
        network_structure,
        landuses_map,
        accessibility_keys,
        distances,
        angular=True,
    )
    # angular should deviate from non angular
    some_false = False
    for acc_key in accessibility_keys:
        for dist_key in distances:
            if not np.allclose(
                accessibilities[acc_key].weighted[dist_key],
                accessibilities_ang[acc_key].weighted[dist_key],
                rtol=config.RTOL,
                atol=config.ATOL,
            ):
                some_false = True
            if not np.allclose(
                accessibilities[acc_key].unweighted[dist_key],
                accessibilities_ang[acc_key].unweighted[dist_key],
                rtol=config.RTOL,
                atol=config.ATOL,
            ):
                some_false = True
    assert some_false is True


def test_mixed_uses(primal_graph):
    # generate node and edge maps
    _nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(primal_graph, 3395)
    data_gdf = mock.mock_landuse_categorical_data(primal_graph, random_seed=13)
    distances = [200, 400, 800, 1600]
    max_dist = max(distances)
    data_map, data_gdf = layers.assign_gdf_to_network(data_gdf, network_structure, max_dist, data_id_col="data_id")
    landuses_map = data_gdf["categorical_landuses"].to_dict()
    # test against various distances
    betas = rustalgos.betas_from_distances(distances)
    for angular in [False, True]:
        # generate
        mixed_uses_data = data_map.mixed_uses(
            network_structure,
            landuses_map,
            distances=distances,
            compute_hill=True,
            compute_hill_weighted=True,
            compute_shannon=True,
            compute_gini=True,
            angular=angular,
        )
        for netw_src_idx in network_structure.node_indices():
            reachable_entries = data_map.aggregate_to_src_idx(
                netw_src_idx, network_structure, max_dist, angular=angular
            )
            for dist_cutoff, beta in zip(distances, betas):
                class_agg = dict()
                # iterate reachable
                for data_key, data_dist in reachable_entries.items():
                    # double check distance is within threshold
                    if data_dist > dist_cutoff:
                        continue
                    cl = landuses_map[data_key]
                    if cl not in class_agg:
                        class_agg[cl] = {"count": 0, "nearest": np.inf}
                    # update the class counts
                    class_agg[cl]["count"] += 1
                    # if distance is nearer, update the nearest distance array too
                    if data_dist < class_agg[cl]["nearest"]:
                        class_agg[cl]["nearest"] = data_dist
                # summarise
                cl_counts = [v["count"] for v in class_agg.values()]
                cl_nearest = [v["nearest"] for v in class_agg.values()]
                # assertions
                assert np.isclose(
                    mixed_uses_data.hill[0][dist_cutoff][netw_src_idx],
                    rustalgos.hill_diversity(cl_counts, 0.0),
                    rtol=config.RTOL,
                    atol=config.ATOL,
                )
                assert np.isclose(
                    mixed_uses_data.hill[1][dist_cutoff][netw_src_idx],
                    rustalgos.hill_diversity(cl_counts, 1),
                    rtol=config.RTOL,
                    atol=config.ATOL,
                )
                assert np.isclose(
                    mixed_uses_data.hill[2][dist_cutoff][netw_src_idx],
                    rustalgos.hill_diversity(cl_counts, 2),
                    rtol=config.RTOL,
                    atol=config.ATOL,
                )
                assert np.isclose(
                    mixed_uses_data.hill_weighted[0][dist_cutoff][netw_src_idx],
                    rustalgos.hill_diversity_branch_distance_wt(cl_counts, cl_nearest, 0, beta, 1.0),
                    rtol=config.RTOL,
                    atol=config.ATOL,
                )
                assert np.isclose(
                    mixed_uses_data.hill_weighted[1][dist_cutoff][netw_src_idx],
                    rustalgos.hill_diversity_branch_distance_wt(cl_counts, cl_nearest, 1, beta, 1.0),
                    rtol=config.RTOL,
                    atol=config.ATOL,
                )
                assert np.isclose(
                    mixed_uses_data.hill_weighted[2][dist_cutoff][netw_src_idx],
                    rustalgos.hill_diversity_branch_distance_wt(cl_counts, cl_nearest, 2, beta, 1.0),
                    rtol=config.RTOL,
                    atol=config.ATOL,
                )
                assert np.isclose(
                    mixed_uses_data.shannon[dist_cutoff][netw_src_idx],
                    rustalgos.shannon_diversity(cl_counts),
                    rtol=config.RTOL,
                    atol=config.ATOL,
                )
                assert np.isclose(
                    mixed_uses_data.gini[dist_cutoff][netw_src_idx],
                    rustalgos.gini_simpson_diversity(cl_counts),
                    rtol=config.RTOL,
                    atol=config.ATOL,
                )


def test_stats(primal_graph):
    # generate node and edge maps
    # generate node and edge maps
    _nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(primal_graph, 3395)
    data_gdf = mock.mock_numerical_data(primal_graph, num_arrs=1, random_seed=13)
    # use a large enough distance such that simple non-weighted checks can be run for max, mean, variance
    max_assign_dist = 3200
    # don't deduplicate with data_id column otherwise below tallys won't work
    data_map, data_gdf = layers.assign_gdf_to_network(data_gdf, network_structure, max_assign_dist)
    numerical_map = data_gdf["mock_numerical_1"].to_dict()
    # for debugging
    # from cityseer.tools import plot
    # plot.plot_network_structure(network_structure, data_gdf)
    # non connected portions of the graph will have different stats
    # used manual data plots from test_assign_to_network() to see which nodes the data points are assigned to
    # connected graph is from 0 to 48 -> assigned data points are all except per below
    connected_nodes_idx = list(range(49))
    # and the respective data assigned to connected portion of the graph
    connected_data_idx = [i for i in range(len(data_gdf)) if i not in [1, 16, 24, 31, 36, 37, 33, 44]]
    # isolated node = 49 -> assigned no data points
    # isolated nodes = 50 & 51 -> assigned data points = 33, 44
    # isolated loop = 52, 53, 54, 55 -> assigned data points = 1, 16, 24, 31, 36, 37
    isolated_nodes_idx = [52, 53, 54, 55]
    isolated_data_idx = [1, 16, 24, 31, 36, 37]
    # numeric precision - keep fairly relaxed
    mock_num_arr = data_gdf["mock_numerical_1"].values
    # compute - first do with no deduplication so that direct comparisons can be made to numpy methods
    # have to use a single large distance, otherwise distance cutoffs will result in limited agg
    distances = [10000]
    stats_result = data_map.stats(
        network_structure,
        numerical_map=numerical_map,
        distances=distances,
    )
    for dist_key in distances:
        # i.e. this scenarios considers all datapoints as unique (no two datapoints point to the same source)
        # max
        assert np.isnan(stats_result.max[dist_key][49])
        assert np.allclose(
            stats_result.max[dist_key][[50, 51]],
            mock_num_arr[[33, 44]].max(),
            atol=config.ATOL,
            rtol=config.RTOL,
        )
        assert np.allclose(
            stats_result.max[dist_key][isolated_nodes_idx],
            mock_num_arr[isolated_data_idx].max(),
            atol=config.ATOL,
            rtol=config.RTOL,
        )
        assert np.allclose(
            stats_result.max[dist_key][connected_nodes_idx],
            mock_num_arr[connected_data_idx].max(),
            atol=config.ATOL,
            rtol=config.RTOL,
        )
        # min
        assert np.isnan(stats_result.max[dist_key][49])
        assert np.allclose(
            stats_result.min[dist_key][[50, 51]],
            mock_num_arr[[33, 44]].min(),
            atol=config.ATOL,
            rtol=config.RTOL,
        )
        assert np.allclose(
            stats_result.min[dist_key][isolated_nodes_idx],
            mock_num_arr[isolated_data_idx].min(),
            atol=config.ATOL,
            rtol=config.RTOL,
        )
        assert np.allclose(
            stats_result.min[dist_key][connected_nodes_idx],
            mock_num_arr[connected_data_idx].min(),
            atol=config.ATOL,
            rtol=config.RTOL,
        )
        # sum
        assert np.isnan(stats_result.max[dist_key][49])
        assert np.allclose(
            stats_result.sum[dist_key][[50, 51]],
            mock_num_arr[[33, 44]].sum(),
            atol=config.ATOL,
            rtol=config.RTOL,
        )
        assert np.allclose(
            stats_result.sum[dist_key][isolated_nodes_idx],
            mock_num_arr[isolated_data_idx].sum(),
            atol=config.ATOL,
            rtol=config.RTOL,
        )
        assert np.allclose(
            stats_result.sum[dist_key][connected_nodes_idx],
            mock_num_arr[connected_data_idx].sum(),
            atol=config.ATOL,
            rtol=config.RTOL,
        )
        # mean
        assert np.isnan(stats_result.max[dist_key][49])
        assert np.allclose(
            stats_result.mean[dist_key][[50, 51]],
            mock_num_arr[[33, 44]].mean(),
            atol=config.ATOL,
            rtol=config.RTOL,
        )
        assert np.allclose(
            stats_result.mean[dist_key][isolated_nodes_idx],
            mock_num_arr[isolated_data_idx].mean(),
            atol=config.ATOL,
            rtol=config.RTOL,
        )
        assert np.allclose(
            stats_result.mean[dist_key][connected_nodes_idx],
            mock_num_arr[connected_data_idx].mean(),
            atol=config.ATOL,
            rtol=config.RTOL,
        )
        # variance
        assert np.isnan(stats_result.max[dist_key][49])
        assert np.allclose(
            stats_result.variance[dist_key][[50, 51]],
            mock_num_arr[[33, 44]].var(),
            atol=config.ATOL,
            rtol=config.RTOL,
        )
        assert np.allclose(
            stats_result.variance[dist_key][isolated_nodes_idx],
            mock_num_arr[isolated_data_idx].var(),
            atol=config.ATOL,
            rtol=config.RTOL,
        )
        assert np.allclose(
            stats_result.variance[dist_key][connected_nodes_idx],
            mock_num_arr[connected_data_idx].var(),
            atol=config.ATOL,
            rtol=config.RTOL,
        )
    # do deduplication - the stats should now be lower on average
    # the last five datapoints are pointing to the same source
    data_map_dedupe, data_gdf_dedupe = layers.assign_gdf_to_network(
        data_gdf, network_structure, max_assign_dist, data_id_col="data_id"
    )
    stats_result_dedupe = data_map_dedupe.stats(
        network_structure,
        numerical_map=numerical_map,
        distances=distances,
    )
    for dist_key in distances:
        # min and max are be the same
        assert np.allclose(
            stats_result.min[dist_key],
            stats_result_dedupe.min[dist_key],
            rtol=config.RTOL,
            atol=config.ATOL,
            equal_nan=True,
        )
        assert np.allclose(
            stats_result.max[dist_key],
            stats_result_dedupe.max[dist_key],
            rtol=config.RTOL,
            atol=config.ATOL,
            equal_nan=True,
        )
        # sum should be lower when deduplicated
        assert np.all(
            stats_result.sum[dist_key][connected_nodes_idx] >= stats_result_dedupe.sum[dist_key][connected_nodes_idx]
        )
        assert np.all(
            stats_result.sum_wt[dist_key][connected_nodes_idx]
            >= stats_result_dedupe.sum_wt[dist_key][connected_nodes_idx]
        )
        # mean and variance should also be diminished
        assert np.all(
            stats_result.mean[dist_key][connected_nodes_idx] >= stats_result_dedupe.mean[dist_key][connected_nodes_idx]
        )
        assert np.all(
            stats_result.mean_wt[dist_key][connected_nodes_idx]
            >= stats_result_dedupe.mean_wt[dist_key][connected_nodes_idx]
        )
        assert np.all(
            stats_result.variance[dist_key][connected_nodes_idx]
            >= stats_result_dedupe.variance[dist_key][connected_nodes_idx]
        )
        assert np.all(
            stats_result.variance_wt[dist_key][connected_nodes_idx]
            >= stats_result_dedupe.variance_wt[dist_key][connected_nodes_idx]
        )
