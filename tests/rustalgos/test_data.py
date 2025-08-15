# pyright: basic
from __future__ import annotations

import builtins

import geopandas as gpd
import networkx as nx
import numpy as np
from cityseer import config, rustalgos
from cityseer.tools import graphs, io, mock
from shapely import geometry, wkt

primal_targets = {
    1: ["int:33", "int:22", "int:7", "int:12", "int:4"],
    54: ["int:9", "int:26", "int:16"],
    29: ["int:44", "int:31"],
    37: ["int:21"],
    32: ["int:28", "int:29", "int:10"],
    4: ["int:33", "int:12", "int:4"],
    42: ["int:30"],
    18: ["int:6", "int:32"],
    17: ["int:1", "int:25", "int:41", "int:35", "int:8", "int:32"],
    11: ["int:14", "int:11"],
    16: ["int:35", "int:8"],
    10: ["int:5", "int:2", "int:40", "int:45", "int:46", "int:47", "int:3", "int:48", "int:49"],
    35: ["int:30"],
    33: ["int:15"],
    21: ["int:6", "int:27"],
    51: ["int:24", "int:39", "int:43"],
    31: ["int:15", "int:29"],
    43: ["int:5", "int:2", "int:42", "int:40", "int:45", "int:46", "int:47", "int:3", "int:48", "int:34", "int:49"],
    14: ["int:11"],
    53: ["int:9", "int:16"],
    45: ["int:0", "int:38", "int:19", "int:18", "int:13"],
    2: ["int:22"],
    30: ["int:44", "int:38", "int:19", "int:18", "int:23"],
    44: ["int:42", "int:13", "int:34", "int:20"],
    12: ["int:14"],
    24: ["int:37", "int:27"],
    28: ["int:17", "int:31"],
    34: ["int:28", "int:10"],
    40: ["int:36", "int:20"],
    20: ["int:1", "int:17", "int:25", "int:41", "int:37"],
    39: ["int:21", "int:36"],
    0: ["int:7"],
    55: ["int:26"],
    56: ["int:0", "int:23"],
    50: ["int:24", "int:39", "int:43"],
}
decomp_targets = {
    109: ["int:3"],
    221: ["int:30"],
    118: ["int:11"],
    106: ["int:5"],
    119: ["int:11"],
    132: ["int:41"],
    265: ["int:43"],
    112: ["int:40"],
    124: ["int:35"],
    126: ["int:8"],
    116: ["int:14"],
    1: ["int:7"],
    238: ["int:20"],
    131: ["int:1", "int:41", "int:25"],
    115: ["int:2"],
    43: ["int:42", "int:2", "int:34"],
    182: ["int:18"],
    148: ["int:27"],
    24: ["int:37"],
    136: ["int:6"],
    234: ["int:36"],
    198: ["int:23"],
    174: ["int:44"],
    272: ["int:16"],
    170: ["int:31"],
    173: ["int:44"],
    274: ["int:26"],
    105: ["int:5"],
    67: ["int:12", "int:4", "int:33"],
    11: ["int:14"],
    199: ["int:23"],
    187: ["int:38"],
    256: ["int:0"],
    130: ["int:32"],
    149: ["int:27"],
    257: ["int:0"],
    110: ["int:3"],
    142: ["int:17"],
    244: ["int:42", "int:34"],
    31: ["int:15"],
    203: ["int:29"],
    263: ["int:24", "int:39"],
    129: ["int:32"],
    58: ["int:7"],
    114: ["int:47", "int:45", "int:46", "int:48", "int:49"],
    17: ["int:1", "int:8", "int:25"],
    202: ["int:29"],
    66: ["int:4", "int:33"],
    4: ["int:12"],
    233: ["int:36"],
    271: ["int:16", "int:9"],
    250: ["int:13"],
    42: ["int:30"],
    141: ["int:37"],
    63: ["int:22"],
    53: ["int:9"],
    18: ["int:6"],
    113: ["int:47", "int:45", "int:46", "int:48", "int:49"],
    169: ["int:31"],
    111: ["int:40"],
    225: ["int:21"],
    207: ["int:28", "int:10"],
    186: ["int:38"],
    275: ["int:26"],
    204: ["int:15"],
    20: ["int:17"],
    125: ["int:35"],
    44: ["int:20"],
    32: ["int:10"],
    226: ["int:21"],
    251: ["int:13"],
    183: ["int:18", "int:19"],
    208: ["int:28"],
    264: ["int:24", "int:39", "int:43"],
    184: ["int:19"],
    64: ["int:22"],
}
decomp_targets_with_barriers = {
    18: ["int:6"],
    169: ["int:31"],
    129: ["int:32"],
    141: ["int:37"],
    257: ["int:0"],
    202: ["int:29"],
    149: ["int:27"],
    208: ["int:28"],
    126: ["int:8"],
    174: ["int:44"],
    44: ["int:20"],
    53: ["int:9"],
    225: ["int:21"],
    199: ["int:23"],
    42: ["int:30"],
    204: ["int:15"],
    274: ["int:26"],
    118: ["int:11"],
    244: ["int:34", "int:42"],
    17: ["int:1", "int:8", "int:25"],
    170: ["int:31"],
    130: ["int:32"],
    196: ["int:18"],
    1: ["int:7"],
    255: ["int:38"],
    132: ["int:41"],
    58: ["int:7"],
    116: ["int:14"],
    110: ["int:3"],
    233: ["int:13", "int:36"],
    265: ["int:43"],
    105: ["int:5"],
    207: ["int:28", "int:10"],
    271: ["int:16", "int:9"],
    31: ["int:15"],
    119: ["int:11"],
    198: ["int:23"],
    32: ["int:10"],
    234: ["int:13", "int:36"],
    148: ["int:27"],
    24: ["int:37"],
    109: ["int:3"],
    64: ["int:22"],
    125: ["int:35"],
    203: ["int:29"],
    197: ["int:18"],
    238: ["int:20"],
    66: ["int:4", "int:33"],
    63: ["int:22"],
    256: ["int:0", "int:38"],
    272: ["int:16"],
    124: ["int:35"],
    275: ["int:26"],
    4: ["int:12"],
    106: ["int:5"],
    136: ["int:6"],
    221: ["int:30"],
    183: ["int:19"],
    67: ["int:4", "int:33", "int:12"],
    131: ["int:1", "int:25", "int:41"],
    173: ["int:44"],
    184: ["int:19"],
    11: ["int:14"],
    43: ["int:46", "int:45", "int:47", "int:48", "int:34", "int:2", "int:49", "int:42"],
    115: ["int:2"],
    264: ["int:39", "int:43", "int:24"],
    142: ["int:17"],
    226: ["int:21"],
    263: ["int:39", "int:24"],
    20: ["int:17"],
}
targets_poly = {
    37: ["int:28", "int:21"],
    4: ["int:4", "int:35", "int:12", "int:33"],
    43: ["int:42", "int:40", "int:2", "int:30", "int:5", "int:3", "int:34"],
    51: ["int:1", "int:43", "int:25", "int:35", "int:8", "int:39", "int:24", "int:12", "int:33"],
    53: ["int:26", "int:9", "int:29", "int:16"],
    28: ["int:31", "int:17", "int:39", "int:37"],
    52: ["int:26", "int:9", "int:5", "int:29", "int:16"],
    29: ["int:31", "int:44", "int:17"],
    27: ["int:44"],
    39: ["int:13", "int:20", "int:36", "int:21"],
    24: ["int:17", "int:41", "int:27", "int:32", "int:37"],
    45: ["int:38", "int:13", "int:18", "int:23", "int:0", "int:20", "int:19", "int:36"],
    11: ["int:14", "int:11"],
    26: ["int:44"],
    56: ["int:38", "int:18", "int:23", "int:0"],
    21: ["int:41", "int:27", "int:32", "int:6"],
    20: ["int:1", "int:43", "int:17", "int:25", "int:8", "int:41", "int:27", "int:32", "int:39", "int:24", "int:37"],
    40: ["int:13", "int:20", "int:2", "int:36", "int:21"],
    16: ["int:1", "int:43", "int:25", "int:35", "int:8", "int:39", "int:24"],
    31: ["int:9", "int:15", "int:7", "int:29", "int:10", "int:16"],
    1: ["int:4", "int:35", "int:22", "int:7", "int:12", "int:33"],
    13: ["int:14", "int:11"],
    38: ["int:13", "int:36"],
    54: ["int:26", "int:9", "int:3", "int:16"],
    33: ["int:15"],
    3: ["int:4"],
    10: ["int:40", "int:2", "int:30", "int:5", "int:3", "int:11", "int:34"],
    5: ["int:5"],
    9: ["int:12", "int:33"],
    17: ["int:1", "int:43", "int:25", "int:35", "int:8", "int:41", "int:27", "int:32", "int:39", "int:24"],
    12: ["int:14"],
    55: ["int:26", "int:5", "int:3"],
    0: ["int:15", "int:35", "int:8", "int:22", "int:7", "int:29", "int:12", "int:33"],
    32: ["int:9", "int:15", "int:29", "int:28", "int:10", "int:16"],
    50: ["int:1", "int:43", "int:25", "int:35", "int:8", "int:39", "int:24", "int:12", "int:33"],
    18: ["int:41", "int:27", "int:32", "int:6"],
    34: ["int:15", "int:28", "int:10"],
    19: ["int:6"],
    23: ["int:6"],
    36: ["int:28", "int:10", "int:21"],
    41: ["int:2", "int:21"],
    25: ["int:44", "int:17", "int:37"],
    6: ["int:11"],
    30: ["int:38", "int:44", "int:18", "int:23", "int:0", "int:19"],
    14: ["int:14", "int:11"],
    35: ["int:9", "int:30", "int:3", "int:28", "int:10", "int:16"],
    2: ["int:4", "int:22", "int:5", "int:7"],
    42: ["int:2", "int:30", "int:3", "int:16"],
    44: ["int:42", "int:13", "int:20", "int:2", "int:34", "int:36"],
}
targets_poly_0 = {
    56: ["int:23"],
    52: ["int:9"],
    11: ["int:11"],
    32: ["int:10"],
    54: ["int:16", "int:9"],
    31: ["int:15"],
    43: ["int:2"],
    34: ["int:10"],
    20: ["int:17"],
    1: ["int:7"],
    30: ["int:23"],
    33: ["int:15"],
    14: ["int:11"],
    28: ["int:17"],
    53: ["int:16", "int:9"],
    42: ["int:2"],
    0: ["int:7"],
    10: ["int:2"],
}


def override_coords(nx_multigraph: nx.MultiGraph) -> gpd.GeoDataFrame:
    """Some tweaks for visual checks."""
    data_gdf = mock.mock_data_gdf(nx_multigraph, random_seed=25)
    data_gdf.loc[18, "geometry"] = geometry.Point(701200, 5719400)  # type: ignore
    data_gdf.loc[39, "geometry"] = geometry.Point(700750, 5720025)  # type: ignore
    data_gdf.loc[26, "geometry"] = geometry.Point(700400, 5719525)  # type: ignore

    return data_gdf


def test_assign_to_network(primal_graph):
    # create additional dead-end scenario
    primal_graph.remove_edge("14", "15")
    primal_graph.remove_edge("15", "28")
    # G = graphs.nx_auto_edge_params(G)
    G_decomp = graphs.nx_decompose(primal_graph, 50)
    # targets visually confirmed in plots
    for G, targets in [(primal_graph, primal_targets), (G_decomp, decomp_targets)]:
        # generate data
        _nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(G)
        data_gdf = override_coords(G)
        data_map = mock.mock_data_map(data_gdf)
        # use small enough distance to ensure closest edge used regrardless of whether both nodes within
        data_map.assign_data_to_network(network_structure, max_assignment_dist=400, n_nearest_candidates=6)
        # from cityseer.tools import plot
        # plot.plot_network_structure(network_structure, data_map)
        # plot.plot_assignment(network_structure, G, data_map)
        for node_idx, data_assignments in data_map.node_data_map.items():
            matches = []
            for data_idx, data_dist in data_assignments:
                matches.append(data_idx)
                # get the data point
                data_entry = data_map.entries[data_idx]
                # get the data point geometry
                data_geom = wkt.loads(data_entry.geom_wkt)
                # get the node data
                node_data = network_structure.get_node_payload_py(node_idx)
                # get the node geometry
                node_geom = geometry.Point(node_data.coord)
                # check the assigned distance
                assert data_geom.distance(node_geom) - data_dist < config.ATOL
            # check the assignments
            assert sorted(matches) == sorted(targets[node_idx])
        # should be None if distance is 0m
        data_map.assign_data_to_network(network_structure, max_assignment_dist=0, n_nearest_candidates=6)
        assert data_map.node_data_map == {}

    # test with polygons
    _nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(primal_graph)
    data_gdf = override_coords(primal_graph)
    data_gdf_buff = data_gdf.copy()
    data_gdf_buff["geometry"] = data_gdf["geometry"].buffer(10)
    data_map = mock.mock_data_map(data_gdf_buff)
    data_map.assign_data_to_network(network_structure, max_assignment_dist=200, n_nearest_candidates=6)
    # from cityseer.tools import plot
    # plot.plot_network_structure(network_structure, data_map)
    # plot.plot_assignment(network_structure, G, data_map)
    for node_idx, data_assignments in data_map.node_data_map.items():
        matches = []
        for data_idx, data_dist in data_assignments:
            matches.append(data_idx)
            # get the data point
            data_entry = data_map.entries[data_idx]
            # get the data point geometry
            data_geom = wkt.loads(data_entry.geom_wkt)
            # get the node data
            node_data = network_structure.get_node_payload_py(node_idx)
            # get the node geometry
            node_geom = geometry.Point(node_data.coord)
            # check the assigned distance
            assert data_geom.distance(node_geom) - data_dist < config.ATOL
        # check the assignments
        assert sorted(matches) == sorted(targets_poly[node_idx])

    # Some polygons will overlap edges if distance is 0m
    data_map.assign_data_to_network(network_structure, max_assignment_dist=0, n_nearest_candidates=6)
    for node_idx, data_assignments in data_map.node_data_map.items():
        matches = []
        for data_idx, data_dist in data_assignments:
            matches.append(data_idx)
            # get the data point
            data_entry = data_map.entries[data_idx]
            # get the data point geometry
            data_geom = wkt.loads(data_entry.geom_wkt)
            # get the node data
            node_data = network_structure.get_node_payload_py(node_idx)
            # get the node geometry
            node_geom = geometry.Point(node_data.coord)
            # check the assigned distance
            assert data_geom.distance(node_geom) - data_dist < config.ATOL
        # check the assignments
        assert sorted(matches) == sorted(targets_poly_0[node_idx])

    # test with barriers
    _nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(G)
    data_gdf = override_coords(G_decomp)
    barriers_gdf, barriers_wkt = mock.mock_barriers()
    # ax = barriers_gdf.plot()
    # ax = data_gdf.plot(ax=ax, color="red", markersize=5)
    # ax = _edges_gdf.plot(ax=ax, color="blue", linewidth=0.5)
    # plt.show()
    # data_map = mock.mock_data_map(data_gdf)
    # data_map.assign_data_to_network(network_structure, max_assignment_dist=400, n_nearest_candidates=6)
    # plot.plot_network_structure(network_structure, data_map)
    data_map = mock.mock_data_map(data_gdf)
    network_structure.set_barriers(barriers_wkt)
    data_map.assign_data_to_network(network_structure, max_assignment_dist=400, n_nearest_candidates=6)
    # plot.plot_network_structure(network_structure, data_map)
    # plot.plot_assignment(network_structure, G_decomp, data_map)

    for node_idx, data_assignments in data_map.node_data_map.items():
        matches = []
        for data_idx, data_dist in data_assignments:
            matches.append(data_idx)
            # get the data point
            data_entry = data_map.entries[data_idx]
            # get the data point geometry
            data_geom = wkt.loads(data_entry.geom_wkt)
            # get the node data
            node_data = network_structure.get_node_payload_py(node_idx)
            # get the node geometry
            node_geom = geometry.Point(node_data.coord)
            # check the assigned distance
            assert data_geom.distance(node_geom) - data_dist < config.ATOL
        # check the assignments
        assert sorted(matches) == sorted(decomp_targets_with_barriers[node_idx])


def test_aggregate_to_src_idx(primal_graph):
    for deduplicate in [False, True]:
        for max_dist in [250, 750]:
            max_seconds = max_dist / config.SPEED_M_S
            # generate data
            _nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(primal_graph)
            data_gdf = mock.mock_data_gdf(primal_graph)
            if deduplicate is False:
                data_map = mock.mock_data_map(data_gdf, dedupe_key_col=None)
            else:
                data_map = mock.mock_data_map(data_gdf, dedupe_key_col="data_id")
            # nearest assigned distance is different to overall distance above
            data_map.assign_data_to_network(network_structure, 400, n_nearest_candidates=6)
            # in this case, use same assignment max dist as search max dist
            # for debugging
            # from cityseer.tools import plot
            # plot.plot_network_structure(network_structure, data_map)
            for angular in [False, True]:
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
                    # check that reachable entries and respective distances are correct
                    # should match against the network structure plus data_map.node_data_map distances
                    manual_reachable = {}
                    manual_not_reachable = {}
                    for node_idx, node_visit in enumerate(tree_map):  # type: ignore
                        if np.isfinite(node_visit.agg_seconds):
                            if node_idx not in data_map.node_data_map:
                                continue
                            for assigned_data_idx, assigned_data_dist in data_map.node_data_map[node_idx]:
                                dist = node_visit.agg_seconds * config.SPEED_M_S + assigned_data_dist
                                if dist <= max_dist:
                                    manual_reachable[assigned_data_idx] = dist
                                else:
                                    manual_not_reachable[assigned_data_idx] = dist
                    # all reachable entries should be in manual reachable and distances should be the same
                    for reachable_key, reachable_dist in reachable_entries.items():
                        assert reachable_key in manual_reachable
                        assert reachable_dist - manual_reachable[reachable_key] < config.ATOL
                    # manual reachable shouldn't contain keys not in reachable entries
                    # allow 1m tolerance for floating point errors
                    # handle shadowed dupe id nodes in deduplication case
                    shadowed_dupes = ["int:45", "int:46", "int:47", "int:48"]
                    for reachable_key in manual_reachable:
                        if deduplicate is True and reachable_key in shadowed_dupes:
                            # if shadowed by closest dedupe node, then it should not be reachable
                            # but if within max dist, then "int:49" should be reachable
                            assert "int:49" in manual_reachable
                            assert "int:49" in reachable_entries
                            assert reachable_key not in reachable_entries
                            continue
                        try:
                            assert reachable_key in reachable_entries
                        except AssertionError:
                            # If the key is not found, assert the distance condition
                            assert max_dist - manual_reachable[reachable_key] < 1


def test_accessibility(primal_graph):
    # generate node and edge maps
    _nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(primal_graph)
    data_gdf = mock.mock_landuse_categorical_data(primal_graph, random_seed=13)
    for deduplicate in [False, True]:
        if deduplicate is False:
            data_map = mock.mock_data_map(data_gdf, dedupe_key_col=None)
        else:
            data_map = mock.mock_data_map(data_gdf, dedupe_key_col="data_id")
        distances = [200, 400, 800, 1600]
        max_dist = max(distances)
        max_seconds = max_dist / config.SPEED_M_S
        # max assign dist is different from max search dist
        data_map.assign_data_to_network(network_structure, max_assignment_dist=400, n_nearest_candidates=6)
        landuses_map = dict(data_gdf["categorical_landuses"])  # type: ignore
        # all datapoints and types are completely unique except for the last five - which all point to the same source
        accessibility_keys = ["a", "b", "c", "z"]  # the duplicate keys are per landuse 'z'
        # generate
        accessibilities = data_map.accessibility(
            network_structure,
            landuses_map,  # type: ignore
            accessibility_keys,
            distances,
        )
        # test manual metrics against all nodes
        betas = rustalgos.betas_from_distances(distances)
        for dist, beta in zip(distances, betas, strict=True):
            z_nws = []
            z_wts = []
            for src_idx in network_structure.street_node_indices():  # type: ignore
                # aggregate
                a_nw = 0
                b_nw = 0
                c_nw = 0
                z_nw = 0
                a_wt = 0
                b_wt = 0
                c_wt = 0
                z_wt = 0  # for deduplication checks on data items 45-49 - which are only z items
                a_dist = np.nan
                b_dist = np.nan
                c_dist = np.nan
                z_dist = np.nan
                # iterate reachable
                reachable_entries = data_map.aggregate_to_src_idx(
                    src_idx, network_structure, int(max_seconds), config.SPEED_M_S
                )
                for data_key, data_dist in reachable_entries.items():
                    py_type, py_key = data_key.split(":")
                    py_cast = getattr(builtins, py_type)
                    data_key_py = py_cast(py_key)
                    # double check distance is within threshold
                    assert data_dist <= max_dist
                    if data_dist <= dist:
                        data_class = landuses_map[data_key_py]
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
                assert accessibilities.result["a"].unweighted[dist][src_idx] - a_nw < config.ATOL
                assert accessibilities.result["b"].unweighted[dist][src_idx] - b_nw < config.ATOL
                assert accessibilities.result["c"].unweighted[dist][src_idx] - c_nw < config.ATOL
                assert accessibilities.result["z"].unweighted[dist][src_idx] - z_nw < config.ATOL
                assert accessibilities.result["a"].weighted[dist][src_idx] - a_wt < config.ATOL
                assert accessibilities.result["b"].weighted[dist][src_idx] - b_wt < config.ATOL
                assert accessibilities.result["c"].weighted[dist][src_idx] - c_wt < config.ATOL
                assert accessibilities.result["z"].weighted[dist][src_idx] - z_wt < config.ATOL
                if dist == max(distances):
                    if np.isfinite(a_dist):
                        assert accessibilities.result["a"].distance[dist][src_idx] - a_dist < config.ATOL
                    else:
                        assert np.isnan(a_dist) and np.isnan(accessibilities.result["a"].distance[dist][src_idx])
                    if np.isfinite(b_dist):
                        assert accessibilities.result["b"].distance[dist][src_idx] - b_dist < config.ATOL
                    else:
                        assert np.isnan(b_dist) and np.isnan(accessibilities.result["b"].distance[dist][src_idx])
                    if np.isfinite(c_dist):
                        assert accessibilities.result["c"].distance[dist][src_idx] - c_dist < config.ATOL
                    else:
                        assert np.isnan(c_dist) and np.isnan(accessibilities.result["c"].distance[dist][src_idx])
                    if np.isfinite(z_dist):
                        assert accessibilities.result["z"].distance[dist][src_idx] - z_dist < config.ATOL
                    else:
                        assert np.isnan(z_dist) and np.isnan(accessibilities.result["z"].distance[dist][src_idx])
                else:
                    assert dist not in accessibilities.result["a"].distance
                    assert dist not in accessibilities.result["b"].distance
                    assert dist not in accessibilities.result["c"].distance
                    assert dist not in accessibilities.result["z"].distance
                # check for deduplication
                z_nws.append(z_nw)
                z_wts.append(z_wt)
            if deduplicate is True:
                assert np.all(z_nws) <= 1
                assert np.all(z_wts) <= 1
            else:
                if dist >= 400:
                    assert max(z_nws) == 5
                if dist >= 800:
                    assert max(z_wts) >= 1
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
                accessibilities.result[acc_key].weighted[dist_key],
                accessibilities_ang.result[acc_key].weighted[dist_key],
                rtol=config.RTOL,
                atol=config.ATOL,
            ):
                some_false = True
            if not np.allclose(
                accessibilities.result[acc_key].unweighted[dist_key],
                accessibilities_ang.result[acc_key].unweighted[dist_key],
                rtol=config.RTOL,
                atol=config.ATOL,
            ):
                some_false = True
    assert some_false is True


def test_mixed_uses(primal_graph):
    # generate node and edge maps
    _nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(primal_graph)
    data_gdf = mock.mock_landuse_categorical_data(primal_graph, random_seed=13)
    data_map = mock.mock_data_map(data_gdf, dedupe_key_col="data_id")
    data_map.assign_data_to_network(network_structure, max_assignment_dist=400, n_nearest_candidates=6)
    distances = [200, 400, 800, 1600]
    max_dist = max(distances)
    max_seconds = max_dist / config.SPEED_M_S
    landuses_map = dict(data_gdf["categorical_landuses"])  # type: ignore
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
        for netw_src_idx in network_structure.street_node_indices():
            reachable_entries = data_map.aggregate_to_src_idx(
                netw_src_idx, network_structure, int(max_seconds), config.SPEED_M_S, angular=angular
            )
            for dist_cutoff, beta in zip(distances, betas, strict=True):
                class_agg = dict()
                # iterate reachable
                for data_key, data_dist in reachable_entries.items():
                    py_type, py_key = data_key.split(":")
                    py_cast = getattr(builtins, py_type)
                    data_key_py = py_cast(py_key)
                    # double check distance is within threshold
                    if data_dist > dist_cutoff:
                        continue
                    cl = landuses_map[data_key_py]
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
    data_map = mock.mock_data_map(data_gdf)
    data_map.assign_data_to_network(network_structure, max_assignment_dist=max_assign_dist, n_nearest_candidates=6)

    numerical_maps = []
    for stats_col in ["mock_numerical_1", "mock_numerical_2"]:
        numerical_maps.append(dict(data_gdf[stats_col]))  # type: ignore)
    # for debugging
    # from cityseer.tools import plot
    # plot.plot_network_structure(network_structure, data_map)
    # non connected portions of the graph will have different stats
    # used manual data plots from test_assign_to_network() to see which nodes the data points are assigned to
    # connected graph is from 0 to 48 -> assigned data points are all except per below
    connected_nodes_idx = list(range(49))
    # and the respective data assigned to connected portion of the graph
    connected_data_idx = [i for i in range(len(data_gdf)) if i not in [16, 37, 33]]
    # isolated node = 49 -> assigned no data points
    # isolated nodes = 50 & 51 -> assigned data points = 33
    # isolated loop = 52, 53, 54, 55 -> assigned data points = 16, 37
    isolated_nodes_idx = [52, 53, 54, 55]
    isolated_data_idx = [16, 37]
    # compute - first do with no deduplication so that direct comparisons can be made to numpy methods
    # have to use a single large distance, otherwise distance cutoffs will result in limited agg
    distances = [10000]
    stats_results = data_map.stats(
        network_structure,
        numerical_maps=numerical_maps,
        distances=distances,
    )
    for stats_result, mock_num_arr in zip(
        stats_results.result, [data_gdf["mock_numerical_1"].values, data_gdf["mock_numerical_2"].values], strict=True
    ):
        for dist in distances:
            # i.e. this scenarios considers all datapoints as unique (no two datapoints point to the same source)
            # max
            assert np.isnan(stats_result.max[dist][49])
            assert np.allclose(
                stats_result.max[dist][[50, 51]],
                mock_num_arr[[33]].max(),
                atol=config.ATOL,
                rtol=config.RTOL,
            )
            assert np.allclose(
                stats_result.max[dist][isolated_nodes_idx],
                mock_num_arr[isolated_data_idx].max(),
                atol=config.ATOL,
                rtol=config.RTOL,
            )
            assert np.allclose(
                stats_result.max[dist][connected_nodes_idx],
                mock_num_arr[connected_data_idx].max(),
                atol=config.ATOL,
                rtol=config.RTOL,
            )
            # min
            assert np.isnan(stats_result.min[dist][49])
            assert np.allclose(
                stats_result.min[dist][[50, 51]],
                mock_num_arr[[33]].min(),
                atol=config.ATOL,
                rtol=config.RTOL,
            )
            assert np.allclose(
                stats_result.min[dist][isolated_nodes_idx],
                mock_num_arr[isolated_data_idx].min(),
                atol=config.ATOL,
                rtol=config.RTOL,
            )
            assert np.allclose(
                stats_result.min[dist][connected_nodes_idx],
                mock_num_arr[connected_data_idx].min(),
                atol=config.ATOL,
                rtol=config.RTOL,
            )
            # sum
            assert np.isnan(stats_result.max[dist][49])
            assert np.allclose(
                stats_result.sum[dist][[50, 51]],
                mock_num_arr[[33]].sum(),
                atol=config.ATOL,
                rtol=config.RTOL,
            )
            assert np.allclose(
                stats_result.sum[dist][isolated_nodes_idx],
                mock_num_arr[isolated_data_idx].sum(),
                atol=config.ATOL,
                rtol=config.RTOL,
            )
            assert np.allclose(
                stats_result.sum[dist][connected_nodes_idx],
                mock_num_arr[connected_data_idx].sum(),
                atol=config.ATOL,
                rtol=config.RTOL,
            )
            # mean
            assert np.isnan(stats_result.max[dist][49])
            assert np.allclose(
                stats_result.mean[dist][[50, 51]],
                mock_num_arr[[33]].mean(),
                atol=config.ATOL,
                rtol=config.RTOL,
            )
            assert np.allclose(
                stats_result.mean[dist][isolated_nodes_idx],
                mock_num_arr[isolated_data_idx].mean(),
                atol=config.ATOL,
                rtol=config.RTOL,
            )
            assert np.allclose(
                stats_result.mean[dist][connected_nodes_idx],
                mock_num_arr[connected_data_idx].mean(),
                atol=config.ATOL,
                rtol=config.RTOL,
            )
            # variance
            assert np.isnan(stats_result.max[dist][49])
            assert np.allclose(
                stats_result.variance[dist][[50, 51]],
                mock_num_arr[[33]].var(),
                atol=config.ATOL,
                rtol=config.RTOL,
            )
            assert np.allclose(
                stats_result.variance[dist][isolated_nodes_idx],
                mock_num_arr[isolated_data_idx].var(),
                atol=config.ATOL,
                rtol=config.RTOL,
            )
            assert np.allclose(
                stats_result.variance[dist][connected_nodes_idx],
                mock_num_arr[connected_data_idx].var(),
                atol=config.ATOL,
                rtol=config.RTOL,
            )
            # median
            assert np.isnan(stats_result.max[dist][49])
            assert np.allclose(
                stats_result.median[dist][[50, 51]],
                np.median(mock_num_arr[[33]]),
                atol=config.ATOL,
                rtol=config.RTOL,
            )
            assert np.allclose(
                stats_result.median[dist][isolated_nodes_idx],
                np.median(mock_num_arr[isolated_data_idx]),
                atol=config.ATOL,
                rtol=config.RTOL,
            )
            assert np.allclose(
                stats_result.median[dist][connected_nodes_idx],
                np.median(mock_num_arr[connected_data_idx]),
                atol=config.ATOL,
                rtol=config.RTOL,
            )
    # do deduplication - the stats should now be lower on average
    # the last five datapoints are pointing to the same source
    data_map_dedupe = mock.mock_data_map(data_gdf, dedupe_key_col="data_id")
    data_map_dedupe.assign_data_to_network(
        network_structure, max_assignment_dist=max_assign_dist, n_nearest_candidates=6
    )
    # plot.plot_network_structure(network_structure, data_map_dedupe)
    stats_results_dedupe = data_map_dedupe.stats(
        network_structure,
        numerical_maps=numerical_maps,
        distances=distances,
    )
    for data_entry in data_map.entries.values():
        for data_dedupe_entry in data_map_dedupe.entries.values():
            if data_entry.data_key_py == data_dedupe_entry.data_key_py:
                assert data_entry.data_key == data_dedupe_entry.data_key
    for stats_result, stats_result_dedupe in zip(stats_results.result, stats_results_dedupe.result, strict=True):
        for dist in distances:
            # min and max are be the same
            assert np.allclose(
                stats_result.min[dist],
                stats_result_dedupe.min[dist],
                rtol=config.RTOL,
                atol=config.ATOL,
                equal_nan=True,
            )
            assert np.allclose(
                stats_result.max[dist],
                stats_result_dedupe.max[dist],
                rtol=config.RTOL,
                atol=config.ATOL,
                equal_nan=True,
            )
            # sum should be lower when deduplicated
            assert np.all(
                stats_result.sum[dist][connected_nodes_idx] >= stats_result_dedupe.sum[dist][connected_nodes_idx]
            )
            assert np.all(
                stats_result.sum_wt[dist][connected_nodes_idx] >= stats_result_dedupe.sum_wt[dist][connected_nodes_idx]
            )
            # mean and variance should also be diminished
            assert np.all(
                stats_result.mean[dist][connected_nodes_idx] >= stats_result_dedupe.mean[dist][connected_nodes_idx]
            )
            assert np.all(
                stats_result.mean_wt[dist][connected_nodes_idx]
                >= stats_result_dedupe.mean_wt[dist][connected_nodes_idx]
            )
            assert np.all(
                stats_result.variance[dist][connected_nodes_idx]
                >= stats_result_dedupe.variance[dist][connected_nodes_idx]
            )
            assert np.all(
                stats_result.variance_wt[dist][connected_nodes_idx]
                >= stats_result_dedupe.variance_wt[dist][connected_nodes_idx]
            )


def weighted_median(data, weights):
    """
    Compute the weighted median.
    """
    if not data or not weights:
        return np.nan
    s_data, s_weights = map(np.array, zip(*sorted(zip(data, weights, strict=True)), strict=True))
    midpoint = 0.5 * sum(s_weights)
    if any(s_weights > midpoint):
        return (s_data[s_weights > midpoint])[0]
    cs_weights = np.cumsum(s_weights)
    idx = np.where(cs_weights <= midpoint)[0][-1]
    if cs_weights[idx] == midpoint:
        return np.mean(s_data[idx : idx + 2])
    return s_data[idx + 1]


def test_stats_weighted(primal_graph):
    # generate node and edge maps
    _nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(primal_graph)
    data_gdf = mock.mock_numerical_data(primal_graph, num_arrs=1, random_seed=13)
    data_map = mock.mock_data_map(data_gdf)
    max_assign_dist = 3200
    data_map.assign_data_to_network(network_structure, max_assignment_dist=max_assign_dist, n_nearest_candidates=6)

    numerical_maps = [dict(data_gdf["mock_numerical_1"])]
    distances = [400, 800]
    betas = rustalgos.betas_from_distances(distances)
    max_seconds = max(distances) / config.SPEED_M_S

    stats_results = data_map.stats(
        network_structure,
        numerical_maps=numerical_maps,
        distances=distances,
    )

    stats_result = stats_results.result[0]
    num_map = numerical_maps[0]

    max_curve_wts = rustalgos.clip_wts_curve(distances, betas, 0)

    for netw_src_idx in network_structure.street_node_indices():
        if not network_structure.is_node_live(netw_src_idx):
            continue

        reachable_entries = data_map.aggregate_to_src_idx(
            netw_src_idx, network_structure, int(max_seconds), config.SPEED_M_S, angular=False
        )

        for i, dist_key in enumerate(distances):
            beta = betas[i]
            max_curve_wt = max_curve_wts[i]

            vals = []
            wts = []
            for data_key, data_dist in reachable_entries.items():
                if data_dist <= dist_key:
                    py_key = int(data_key.split(":")[1])
                    num = num_map[py_key]
                    if not np.isnan(num):
                        vals.append(num)
                        wt = rustalgos.clipped_beta_wt(beta, max_curve_wt, data_dist)
                        wts.append(wt)

            if not vals:
                assert np.isnan(stats_result.mean_wt[dist_key][netw_src_idx])
                assert np.isnan(stats_result.variance_wt[dist_key][netw_src_idx])
                assert np.isnan(stats_result.median_wt[dist_key][netw_src_idx])
                continue

            # weighted mean
            mean_wt_py = np.average(vals, weights=wts)
            assert np.isclose(
                stats_result.mean_wt[dist_key][netw_src_idx],
                mean_wt_py,
                rtol=config.RTOL,
                atol=config.ATOL,
            )

            # weighted variance
            variance_wt_py = np.average((np.array(vals) - mean_wt_py) ** 2, weights=wts)
            assert np.isclose(
                stats_result.variance_wt[dist_key][netw_src_idx],
                variance_wt_py,
                rtol=config.RTOL,
                atol=config.ATOL,
            )

            # weighted median
            median_wt_py = weighted_median(vals, wts)
            median_wt_rust = stats_result.median_wt[dist_key][netw_src_idx]
            print(f"Rust median: {median_wt_rust}, Python median: {median_wt_py}")
