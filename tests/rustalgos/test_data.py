# pyright: basic
from __future__ import annotations


import networkx as nx
import numpy as np
import numpy.typing as npt
import pytest

from sklearn.preprocessing import LabelEncoder  # type: ignore

from cityseer import config, rustalgos
from cityseer.metrics import layers, networks
from cityseer.tools import graphs, mock


def test_aggregate_to_src_idx(primal_graph):
    for max_dist in [400, 750]:
        for deduplicate in [False, True]:
            # generate data
            _nodes_gdf, _edges_gdf, network_structure = graphs.network_structure_from_nx(primal_graph, 3395)
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
                    _nodes, _edges, tree_map, _edge_map = network_structure.shortest_path_tree(
                        netw_src_idx, max_dist, angular
                    )
                    # verify distances vs. the max
                    for data_key, data_entry in data_map.entries.items():
                        # nearest
                        if data_entry.nearest_assign is not None:
                            nearest_netw_node = network_structure.get_node_payload(data_entry.nearest_assign)
                            nearest_assign_dist = tree_map[data_entry.nearest_assign].short_dist
                            # add tail
                            if not np.isinf(nearest_assign_dist):
                                nearest_assign_dist += nearest_netw_node.coord.hypot(data_entry.coord)
                        else:
                            nearest_assign_dist = np.inf
                        # next nearest
                        if data_entry.next_nearest_assign is not None:
                            next_nearest_netw_node = network_structure.get_node_payload(data_entry.next_nearest_assign)
                            next_nearest_assign_dist = tree_map[data_entry.next_nearest_assign].short_dist
                            # add tail
                            if not np.isinf(next_nearest_assign_dist):
                                next_nearest_assign_dist += next_nearest_netw_node.coord.hypot(data_entry.coord)
                        else:
                            next_nearest_assign_dist = np.inf
                        # check deduplication - 49 is the closest, so others should not make it through
                        # checks
                        if nearest_assign_dist > max_dist and next_nearest_assign_dist > max_dist:
                            assert data_key not in reachable_entries
                        elif deduplicate and data_key in ["45", "46", "47", "48"]:
                            assert data_key not in reachable_entries and "49" in reachable_entries
                        elif np.isinf(nearest_assign_dist) and next_nearest_assign_dist < max_dist:
                            assert reachable_entries[data_key] - next_nearest_assign_dist < config.ATOL
                        elif np.isinf(next_nearest_assign_dist) and nearest_assign_dist < max_dist:
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
    _nodes_gdf, _edges_gdf, network_structure = graphs.network_structure_from_nx(primal_graph, 3395)
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
                    elif data_class == "b":
                        b_nw += 1
                        b_wt += np.exp(-beta * data_dist)
                    elif data_class == "c":
                        c_nw += 1
                        c_wt += np.exp(-beta * data_dist)
                    elif data_class == "z":
                        z_nw += 1
                        z_wt += np.exp(-beta * data_dist)
            # assertions
            assert accessibilities["a"].unweighted[dist][src_idx] - a_nw < config.ATOL
            assert accessibilities["b"].unweighted[dist][src_idx] - b_nw < config.ATOL
            assert accessibilities["c"].unweighted[dist][src_idx] - c_nw < config.ATOL
            assert accessibilities["z"].unweighted[dist][src_idx] - z_nw < config.ATOL
            assert accessibilities["a"].weighted[dist][src_idx] - a_wt < config.ATOL
            assert accessibilities["b"].weighted[dist][src_idx] - b_wt < config.ATOL
            assert accessibilities["c"].weighted[dist][src_idx] - c_wt < config.ATOL
            assert accessibilities["z"].weighted[dist][src_idx] - z_wt < config.ATOL
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
    _nodes_gdf, _edges_gdf, network_structure = graphs.network_structure_from_nx(primal_graph, 3395)
    data_gdf = mock.mock_landuse_categorical_data(primal_graph, random_seed=13)
    distances = [200, 400, 800, 1600]
    max_dist = max(distances)
    data_map, data_gdf = layers.assign_gdf_to_network(data_gdf, network_structure, max_dist, data_id_col="data_id")
    landuses_map = data_gdf["categorical_landuses"].to_dict()
    # test against various distances
    betas = rustalgos.betas_from_distances(distances)
    for angular in [False, True]:
        # generate
        mu_data_hill, mu_data_other = data_map.mixed_uses(
            network_structure,
            landuses_map,
            distances=distances,
            mixed_uses_hill=True,
            mixed_uses_other=True,
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
                    mu_data_hill.hill[0][dist_cutoff][netw_src_idx],
                    rustalgos.hill_diversity(cl_counts, 0.0),
                    rtol=config.RTOL,
                    atol=config.ATOL,
                )
                assert np.isclose(
                    mu_data_hill.hill[1][dist_cutoff][netw_src_idx],
                    rustalgos.hill_diversity(cl_counts, 1),
                    rtol=config.RTOL,
                    atol=config.ATOL,
                )
                assert np.isclose(
                    mu_data_hill.hill[2][dist_cutoff][netw_src_idx],
                    rustalgos.hill_diversity(cl_counts, 2),
                    rtol=config.RTOL,
                    atol=config.ATOL,
                )
                assert np.isclose(
                    mu_data_hill.hill_weighted[0][dist_cutoff][netw_src_idx],
                    rustalgos.hill_diversity_branch_distance_wt(cl_counts, cl_nearest, 0, beta, 1.0),
                    rtol=config.RTOL,
                    atol=config.ATOL,
                )
                assert np.isclose(
                    mu_data_hill.hill_weighted[1][dist_cutoff][netw_src_idx],
                    rustalgos.hill_diversity_branch_distance_wt(cl_counts, cl_nearest, 1, beta, 1.0),
                    rtol=config.RTOL,
                    atol=config.ATOL,
                )
                assert np.isclose(
                    mu_data_hill.hill_weighted[2][dist_cutoff][netw_src_idx],
                    rustalgos.hill_diversity_branch_distance_wt(cl_counts, cl_nearest, 2, beta, 1.0),
                    rtol=config.RTOL,
                    atol=config.ATOL,
                )
                assert np.isclose(
                    mu_data_other.shannon[dist_cutoff][netw_src_idx],
                    rustalgos.shannon_diversity(cl_counts),
                    rtol=config.RTOL,
                    atol=config.ATOL,
                )
                assert np.isclose(
                    mu_data_other.gini[dist_cutoff][netw_src_idx],
                    rustalgos.gini_simpson_diversity(cl_counts),
                    rtol=config.RTOL,
                    atol=config.ATOL,
                )


def test_aggregate_stats(primal_graph):
    # generate node and edge maps
    _nodes_gpd, network_structure = graphs.network_structure_from_nx(primal_graph, 3395)
    data_gdf = mock.mock_numerical_data(primal_graph, num_arrs=2, random_seed=13)
    data_map, data_gdf = layers.assign_gdf_to_network(data_gdf, network_structure, 500, data_id_col="data_id")
    # for debugging
    # from cityseer.tools import plot
    # plot.plot_network_structure(network_structure, data_gdf)
    # set parameters - use a large enough distance such that simple non-weighted checks can be run for max, mean, variance
    betas: npt.NDArray[np.float32] = np.array([0.00125])
    distances = networks.distance_from_beta(betas)
    mock_num_arr = data_gdf[["mock_numerical_1", "mock_numerical_2"]].values.T
    # compute - first do with no deduplication so that direct comparisons can be made to numpy methods
    # i.e. this scenarios considers all datapoints as unique (no two datapoints point to the same source)
    (
        stats_sum,
        stats_sum_wt,
        stats_mean,
        stats_mean_wt,
        stats_variance,
        stats_variance_wt,
        stats_max,
        stats_min,
    ) = data.aggregate_stats(
        network_structure.nodes.xs,
        network_structure.nodes.ys,
        network_structure.nodes.live,
        network_structure.edges.start,
        network_structure.edges.end,
        network_structure.edges.length,
        network_structure.edges.angle_sum,
        network_structure.edges.imp_factor,
        network_structure.edges.in_bearing,
        network_structure.edges.out_bearing,
        network_structure.node_edge_map,
        data_map.xs,
        data_map.ys,
        data_map.nearest_assign,
        data_map.next_nearest_assign,
        # replace datakey with -1 array - i.e. no unique datapoint keys
        np.full(data_map.data_id.shape[0], -1, dtype=np.int_),
        distances,
        betas,
        numerical_arrays=mock_num_arr,
        angular=False,
    )
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
    for stats_idx in range(len(mock_num_arr)):
        for d_idx in range(len(distances)):
            # max
            assert np.isnan(stats_max[stats_idx, d_idx, 49])
            assert np.allclose(
                stats_max[stats_idx, d_idx, [50, 51]],
                mock_num_arr[stats_idx, [33, 44]].max(),
                atol=config.ATOL,
                rtol=config.RTOL,
            )
            assert np.allclose(
                stats_max[stats_idx, d_idx, isolated_nodes_idx],
                mock_num_arr[stats_idx, isolated_data_idx].max(),
                atol=config.ATOL,
                rtol=config.RTOL,
            )
            assert np.allclose(
                stats_max[stats_idx, d_idx, connected_nodes_idx],
                mock_num_arr[stats_idx, connected_data_idx].max(),
                atol=config.ATOL,
                rtol=config.RTOL,
            )
            # min
            assert np.isnan(stats_min[stats_idx, d_idx, 49])
            assert np.allclose(
                stats_min[stats_idx, d_idx, [50, 51]],
                mock_num_arr[stats_idx, [33, 44]].min(),
                atol=config.ATOL,
                rtol=config.RTOL,
            )
            assert np.allclose(
                stats_min[stats_idx, d_idx, isolated_nodes_idx],
                mock_num_arr[stats_idx, isolated_data_idx].min(),
                atol=config.ATOL,
                rtol=config.RTOL,
            )
            assert np.allclose(
                stats_min[stats_idx, d_idx, connected_nodes_idx],
                mock_num_arr[stats_idx, connected_data_idx].min(),
                atol=config.ATOL,
                rtol=config.RTOL,
            )
            # sum
            assert stats_sum[stats_idx, d_idx, 49] == 0
            assert np.allclose(
                stats_sum[stats_idx, d_idx, [50, 51]],
                mock_num_arr[stats_idx, [33, 44]].sum(),
                atol=config.ATOL,
                rtol=config.RTOL,
            )
            assert np.allclose(
                stats_sum[stats_idx, d_idx, isolated_nodes_idx],
                mock_num_arr[stats_idx, isolated_data_idx].sum(),
                atol=config.ATOL,
                rtol=config.RTOL,
            )
            assert np.allclose(
                stats_sum[stats_idx, d_idx, connected_nodes_idx],
                mock_num_arr[stats_idx, connected_data_idx].sum(),
                atol=config.ATOL,
                rtol=config.RTOL,
            )
            # mean
            assert np.isnan(stats_mean[stats_idx, d_idx, 49])
            assert np.allclose(
                stats_mean[stats_idx, d_idx, [50, 51]],
                mock_num_arr[stats_idx, [33, 44]].mean(),
                atol=config.ATOL,
                rtol=config.RTOL,
            )
            assert np.allclose(
                stats_mean[stats_idx, d_idx, isolated_nodes_idx],
                mock_num_arr[stats_idx, isolated_data_idx].mean(),
                atol=config.ATOL,
                rtol=config.RTOL,
            )
            assert np.allclose(
                stats_mean[stats_idx, d_idx, connected_nodes_idx],
                mock_num_arr[stats_idx, connected_data_idx].mean(),
                atol=config.ATOL,
                rtol=config.RTOL,
            )
            # variance
            assert np.isnan(stats_variance[stats_idx, d_idx, 49])
            assert np.allclose(
                stats_variance[stats_idx, d_idx, [50, 51]],
                mock_num_arr[stats_idx, [33, 44]].var(),
                atol=config.ATOL,
                rtol=config.RTOL,
            )
            assert np.allclose(
                stats_variance[stats_idx, d_idx, isolated_nodes_idx],
                mock_num_arr[stats_idx, isolated_data_idx].var(),
                atol=config.ATOL,
                rtol=config.RTOL,
            )
            assert np.allclose(
                stats_variance[stats_idx, d_idx, connected_nodes_idx],
                mock_num_arr[stats_idx, connected_data_idx].var(),
                atol=config.ATOL,
                rtol=config.RTOL,
            )
    # do deduplication - the stats should now be lower on average
    # the last five datapoints are pointing to the same source
    (
        stats_sum_no_dupe,
        stats_sum_wt_no_dupe,
        stats_mean_no_dupe,
        stats_mean_wt_no_dupe,
        stats_variance_no_dupe,
        stats_variance_wt_no_dupe,
        stats_max_no_dupe,
        stats_min_no_dupe,
    ) = data.aggregate_stats(
        network_structure.nodes.xs,
        network_structure.nodes.ys,
        network_structure.nodes.live,
        network_structure.edges.start,
        network_structure.edges.end,
        network_structure.edges.length,
        network_structure.edges.angle_sum,
        network_structure.edges.imp_factor,
        network_structure.edges.in_bearing,
        network_structure.edges.out_bearing,
        network_structure.node_edge_map,
        data_map.xs,
        data_map.ys,
        data_map.nearest_assign,
        data_map.next_nearest_assign,
        data_map.data_id,
        distances,
        betas,
        numerical_arrays=mock_num_arr,
        angular=False,
    )
    # min and max should be the same
    assert np.allclose(stats_max, stats_max_no_dupe, rtol=config.RTOL, atol=config.ATOL, equal_nan=True)
    assert np.allclose(stats_min, stats_min_no_dupe, rtol=config.RTOL, atol=config.ATOL, equal_nan=True)
    # sum should be lower when deduplicated
    assert np.all(stats_sum >= stats_sum_no_dupe)
    assert np.all(stats_sum_wt >= stats_sum_wt_no_dupe)
    # mean and variance should also be diminished
    assert np.all(stats_mean[~np.isnan(stats_mean)] >= stats_mean_no_dupe[~np.isnan(stats_mean_no_dupe)])
    assert np.all(stats_mean_wt[~np.isnan(stats_mean_wt)] >= stats_mean_wt_no_dupe[~np.isnan(stats_mean_wt_no_dupe)])
    assert np.all(
        stats_variance[~np.isnan(stats_variance)] >= stats_variance_no_dupe[~np.isnan(stats_variance_no_dupe)]
    )
    assert np.all(
        stats_variance_wt[~np.isnan(stats_variance_wt)]
        >= stats_variance_wt_no_dupe[~np.isnan(stats_variance_wt_no_dupe)]
    )
