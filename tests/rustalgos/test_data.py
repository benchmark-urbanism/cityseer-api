# pyright: basic
from __future__ import annotations

import geopandas as gpd
import networkx as nx
import numpy as np
from cityseer import config, rustalgos
from cityseer.metrics import layers
from cityseer.tools import graphs, io, mock
from shapely import geometry


def override_coords(nx_multigraph: nx.MultiGraph) -> gpd.GeoDataFrame:
    """Some tweaks for visual checks."""
    data_gdf = mock.mock_data_gdf(nx_multigraph, random_seed=25)
    data_gdf.loc[18, "geometry"] = geometry.Point(701200, 5719400)
    data_gdf.loc[39, "geometry"] = geometry.Point(700750, 5720025)
    data_gdf.loc[26, "geometry"] = geometry.Point(700400, 5719525)

    return data_gdf


def test_assign_to_network(primal_graph):
    # create additional dead-end scenario
    primal_graph.remove_edge("14", "15")
    primal_graph.remove_edge("15", "28")
    # G = graphs.nx_auto_edge_params(G)
    G = graphs.nx_decompose(primal_graph, 50)
    # visually confirmed in plots
    targets = {
        "0": [257, 256],
        "1": [17, 131],
        "10": [32, 207],
        "11": [118, 119],
        "12": [67, 4],
        "13": [250, 251],
        "14": [116, 11],
        "15": [204, 31],
        "16": [272, 271],
        "17": [142, 20],
        "18": [182, 183],
        "19": [184, 183],
        "2": [43, 115],
        "20": [238, 44],
        "21": [226, 225],
        "22": [63, 64],
        "23": [199, 198],
        "24": [264, 263],
        "25": [17, 131],
        "26": [274, 275],
        "27": [149, 148],
        "28": [207, 208],
        "29": [202, 203],
        "3": [110, 109],
        "30": [42, 221],
        "31": [169, 168],
        "32": [129, 130],
        "33": [66, 67],
        "34": [43, 244],
        "35": [125, 124],
        "36": [234, 233],
        "37": [141, 24],
        "38": [187, 186],
        "39": [264, 263],
        "4": [66, 67],
        "40": [111, 112],
        "41": [132, 131],
        "42": [244, 43],
        "43": [265, 264],
        "44": [174, 173],
        "45": [114, 113],
        "46": [114, 113],
        "47": [114, 113],
        "48": [113, 114],
        "49": [113, 114],
        "5": [105, 106],
        "6": [18, 136],
        "7": [58, 1],
        "8": [126, 17],
        "9": [53, 271],
    }
    # generate data
    _nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(G)
    data_gdf = override_coords(G)
    data_map = mock.mock_data_map(data_gdf)
    data_map.assign_to_network(network_structure, max_dist=1600)
    # from cityseer.tools import plot
    # plot.plot_network_structure(network_structure, data_map)
    # plot.plot_assignment(network_structure, G, data_map)
    collect = {}
    for target_idx, (data_key, data_entry) in enumerate(data_map.entries.items()):
        collect[data_key] = [data_entry.node_matches.nearest.idx, data_entry.node_matches.next_nearest.idx]
        assert targets[data_entry.data_key][0] == data_entry.node_matches.nearest.idx
        assert targets[data_entry.data_key][1] == data_entry.node_matches.next_nearest.idx
    # should be None if distance is 0m
    data_map.assign_to_network(network_structure, max_dist=0)
    for target_idx, data_entry in enumerate(data_map.entries.values()):
        assert data_entry.node_matches.nearest is None
        assert data_entry.node_matches.next_nearest is None


def test_aggregate_to_src_idx(primal_graph):
    for max_dist in [750]:
        max_seconds = max_dist / config.SPEED_M_S
        for deduplicate in [False]:
            # generate data
            _nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(primal_graph)
            data_gdf = mock.mock_data_gdf(primal_graph)
            data_map = mock.mock_data_map(data_gdf)
            # nearest assigned distance is different to overall distance above
            data_map.assign_to_network(network_structure, 400)
            # in this case, use same assignment max dist as search max dist
            # for debugging
            # from cityseer.tools import plot
            # plot.plot_network_structure(network_structure, data_gdf)
            for angular in [False]:
                for netw_src_idx in network_structure.node_indices():
                    # aggregate to src...
                    reachable_entries = data_map.aggregate_to_src_idx(
                        netw_src_idx, network_structure, int(max_seconds), config.SPEED_M_S, angular=angular
                    )
                    # compare to manual checks on distances:
                    # get the network distances
                    if angular is False:
                        _nodes, tree_map = network_structure.dijkstra_tree_shortest(
                            netw_src_idx, int(max_seconds), config.SPEED_M_S
                        )
                    else:
                        _nodes, tree_map = network_structure.dijkstra_tree_simplest(
                            netw_src_idx, int(max_seconds), config.SPEED_M_S
                        )
                    # verify distances vs. the max
                    for data_key, data_entry in data_map.entries.items():
                        # Use node_matches for clarity and consistency with Rust struct
                        nearest_assign = (
                            data_entry.node_matches.nearest.idx
                            if data_entry.node_matches and data_entry.node_matches.nearest is not None
                            else None
                        )
                        next_nearest_assign = (
                            data_entry.node_matches.next_nearest.idx
                            if data_entry.node_matches and data_entry.node_matches.next_nearest is not None
                            else None
                        )
                        # nearest
                        nearest_assign_sec = np.inf
                        if nearest_assign is not None:
                            nearest_netw_node = network_structure.get_node_payload(nearest_assign)
                            # add tail
                            if not np.isposinf(tree_map[nearest_assign].agg_seconds):
                                nearest_assign_sec = (
                                    tree_map[nearest_assign].agg_seconds
                                    + nearest_netw_node.coord.hypot(data_entry.coord) / config.SPEED_M_S
                                )
                        # next nearest
                        next_nearest_assign_sec = np.inf
                        if next_nearest_assign is not None:
                            next_nearest_netw_node = network_structure.get_node_payload(next_nearest_assign)
                            # add tail
                            if not np.isposinf(tree_map[next_nearest_assign].agg_seconds):
                                next_nearest_assign_sec = (
                                    tree_map[next_nearest_assign].agg_seconds
                                    + next_nearest_netw_node.coord.hypot(data_entry.coord) / config.SPEED_M_S
                                )
                        # check deduplication - 49 is the closest, so others should not make it through
                        # checks
                        if nearest_assign_sec > max_seconds and next_nearest_assign_sec > max_seconds:
                            assert data_key not in reachable_entries
                        elif deduplicate and data_key in ["45", "46", "47", "48"]:
                            assert data_key not in reachable_entries and "49" in reachable_entries
                        # due to rounding errors with f32 conversion, skip where within 0.5 seconds of max_seconds
                        elif nearest_assign_sec > max_seconds - 0.5 and next_nearest_assign_sec > max_seconds - 0.5:
                            print(f"Skipping {data_key} due to potential rounding errors affecting max seconds cutoff")
                            continue
                        elif nearest_assign_sec > max_seconds and next_nearest_assign_sec <= max_seconds:
                            assert (
                                reachable_entries[data_key] / config.SPEED_M_S - next_nearest_assign_sec < config.ATOL
                            )
                        elif next_nearest_assign_sec > max_seconds and nearest_assign_sec <= max_seconds:
                            assert reachable_entries[data_key] / config.SPEED_M_S - nearest_assign_sec < config.ATOL
                        else:
                            # If either assign is within max_seconds, must be present
                            assert data_key in reachable_entries
                            assert (
                                reachable_entries[data_key] / config.SPEED_M_S
                                - min(nearest_assign_sec, next_nearest_assign_sec)
                                < config.ATOL
                            )


def test_accessibility(primal_graph):
    # generate node and edge maps
    _nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(primal_graph)
    data_gdf = mock.mock_landuse_categorical_data(primal_graph, random_seed=13)
    data_map = mock.mock_data_map(data_gdf)
    distances = [200, 400, 800, 1600]
    max_dist = max(distances)
    max_seconds = max_dist / config.SPEED_M_S
    # max assign dist is different from max search dist
    data_map.assign_to_network(network_structure, max_dist=400)
    data_keys: list[str] = data_gdf["datamap_key"]  # type: ignore
    landuses: list[str] = data_gdf["categorical_landuses"]  # type: ignore
    landuses_map: dict[str, str] = dict(zip(data_keys, landuses, strict=True))
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
    for dist, beta in zip(distances, betas, strict=True):
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
            reachable_entries = data_map.aggregate_to_src_idx(
                src_idx, network_structure, int(max_seconds), config.SPEED_M_S
            )
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
    _nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(primal_graph)
    data_gdf = mock.mock_landuse_categorical_data(primal_graph, random_seed=13)
    distances = [200, 400, 800, 1600]
    max_dist = max(distances)
    max_seconds = max_dist / config.SPEED_M_S
    data_map, data_gdf = layers.assign_gdf_to_network(data_gdf, network_structure, max_dist, data_id_col="data_id")
    data_keys: list[str] = data_gdf["datamap_key"]  # type: ignore
    landuses: list[str] = data_gdf["categorical_landuses"]  # type: ignore
    landuses_map: dict[str, str] = dict(zip(data_keys, landuses, strict=True))
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
                netw_src_idx, network_structure, int(max_seconds), config.SPEED_M_S, angular=angular
            )
            for dist_cutoff, beta in zip(distances, betas, strict=True):
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
                    rustalgos.diversity.hill_diversity(cl_counts, 0.0),
                    rtol=config.RTOL,
                    atol=config.ATOL,
                )
                assert np.isclose(
                    mixed_uses_data.hill[1][dist_cutoff][netw_src_idx],
                    rustalgos.diversity.hill_diversity(cl_counts, 1),
                    rtol=config.RTOL,
                    atol=config.ATOL,
                )
                assert np.isclose(
                    mixed_uses_data.hill[2][dist_cutoff][netw_src_idx],
                    rustalgos.diversity.hill_diversity(cl_counts, 2),
                    rtol=config.RTOL,
                    atol=config.ATOL,
                )
                assert np.isclose(
                    mixed_uses_data.hill_weighted[0][dist_cutoff][netw_src_idx],
                    rustalgos.diversity.hill_diversity_branch_distance_wt(cl_counts, cl_nearest, 0, beta, 1.0),
                    rtol=config.RTOL,
                    atol=config.ATOL,
                )
                assert np.isclose(
                    mixed_uses_data.hill_weighted[1][dist_cutoff][netw_src_idx],
                    rustalgos.diversity.hill_diversity_branch_distance_wt(cl_counts, cl_nearest, 1, beta, 1.0),
                    rtol=config.RTOL,
                    atol=config.ATOL,
                )
                assert np.isclose(
                    mixed_uses_data.hill_weighted[2][dist_cutoff][netw_src_idx],
                    rustalgos.diversity.hill_diversity_branch_distance_wt(cl_counts, cl_nearest, 2, beta, 1.0),
                    rtol=config.RTOL,
                    atol=config.ATOL,
                )
                assert np.isclose(
                    mixed_uses_data.shannon[dist_cutoff][netw_src_idx],
                    rustalgos.diversity.shannon_diversity(cl_counts),
                    rtol=config.RTOL,
                    atol=config.ATOL,
                )
                assert np.isclose(
                    mixed_uses_data.gini[dist_cutoff][netw_src_idx],
                    rustalgos.diversity.gini_simpson_diversity(cl_counts),
                    rtol=config.RTOL,
                    atol=config.ATOL,
                )


def test_stats(primal_graph):
    # generate node and edge maps
    # generate node and edge maps
    _nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(primal_graph)
    data_gdf = mock.mock_numerical_data(primal_graph, num_arrs=2, random_seed=13)
    # use a large enough distance such that simple non-weighted checks can be run for max, mean, variance
    max_assign_dist = 3200
    # don't deduplicate with data_id column otherwise below tallys won't work
    data_map, data_gdf = layers.assign_gdf_to_network(data_gdf, network_structure, max_assign_dist)
    data_keys: list[str] = data_gdf["datamap_key"]  # type: ignore
    numerical_maps = []
    for stats_col in ["mock_numerical_1", "mock_numerical_2"]:
        stats: list[str] = data_gdf[stats_col]  # type: ignore
        stats_map: dict[str, str] = dict(zip(data_keys, stats, strict=True))
        numerical_maps.append(stats_map)
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
    # compute - first do with no deduplication so that direct comparisons can be made to numpy methods
    # have to use a single large distance, otherwise distance cutoffs will result in limited agg
    distances = [10000]
    stats_results = data_map.stats(
        network_structure,
        numerical_maps=numerical_maps,
        distances=distances,
    )
    stats_result = stats_results[0]
    for stats_result, mock_num_arr in zip(
        stats_results, [data_gdf["mock_numerical_1"].values, data_gdf["mock_numerical_2"].values], strict=True
    ):
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
    stats_results_dedupe = data_map_dedupe.stats(
        network_structure,
        numerical_maps=numerical_maps,
        distances=distances,
    )
    for stats_result, stats_result_dedupe in zip(stats_results, stats_results_dedupe, strict=True):
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
                stats_result.sum[dist_key][connected_nodes_idx]
                >= stats_result_dedupe.sum[dist_key][connected_nodes_idx]
            )
            assert np.all(
                stats_result.sum_wt[dist_key][connected_nodes_idx]
                >= stats_result_dedupe.sum_wt[dist_key][connected_nodes_idx]
            )
            # mean and variance should also be diminished
            assert np.all(
                stats_result.mean[dist_key][connected_nodes_idx]
                >= stats_result_dedupe.mean[dist_key][connected_nodes_idx]
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
