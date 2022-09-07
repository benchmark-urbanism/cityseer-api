# pyright: basic
from __future__ import annotations

import geopandas as gpd
import networkx as nx
import numpy as np
import numpy.typing as npt
import pytest
from shapely import geometry
from sklearn.preprocessing import LabelEncoder  # type: ignore

from cityseer import config, structures
from cityseer.algos import centrality, data, diversity
from cityseer.metrics import layers, networks
from cityseer.tools import graphs, mock


def test_find_nearest(primal_graph):
    _nodes_gdf, network_structure = graphs.network_structure_from_nx(primal_graph, 3395)
    data_gdf = mock.mock_data_gdf(primal_graph)
    data_map, data_gdf = layers.assign_gdf_to_network(data_gdf, network_structure, max_netw_assign_dist=400)
    # test the filter - iterating each point in data map
    for d_idx in range(data_map.count):
        # find the closest point on the network
        d_x, d_y = data_map.x_y(d_idx)
        min_idx, min_dist, _next_min_idx = data.find_nearest(
            d_x, d_y, network_structure.nodes.xs, network_structure.nodes.ys, max_dist=np.float32(500)
        )
        # check that no other indices are nearer
        for n_idx in range(network_structure.nodes.count):
            n_x = network_structure.nodes.xs[n_idx]
            n_y = network_structure.nodes.ys[n_idx]
            dist = np.sqrt((d_x - n_x) ** 2 + (d_y - n_y) ** 2)
            if n_idx == min_idx:
                assert np.isclose(dist, min_dist, rtol=config.RTOL, atol=config.ATOL)
            else:
                assert dist > min_dist


def override_coords(
    nx_multigraph: nx.MultiGraph, max_dist: int
) -> tuple[structures.DataMap, gpd.GeoDataFrame, structures.NetworkStructure]:
    """Some tweaks for visual checks."""
    _nodes_gdf, network_structure = graphs.network_structure_from_nx(nx_multigraph, 3395)
    data_gdf = mock.mock_data_gdf(nx_multigraph, random_seed=25)
    data_gdf.geometry[18] = geometry.Point(701200, 5719400)
    data_gdf.geometry[39] = geometry.Point(700750, 5720025)
    data_gdf.geometry[26] = geometry.Point(700400, 5719525)
    data_map, data_gdf = layers.assign_gdf_to_network(data_gdf, network_structure, max_dist)

    return data_map, data_gdf, network_structure


def test_assign_to_network(primal_graph):
    # create additional dead-end scenario
    primal_graph.remove_edge(14, 15)
    primal_graph.remove_edge(15, 28)
    # G = graphs.nx_auto_edge_params(G)
    G = graphs.nx_decompose(primal_graph, 50)
    # visually confirmed in plots
    targets = np.array(
        [
            [0, 257, 256],
            [1, 17, 131],
            [2, 43, 243],
            [3, 110, 109],
            [4, 66, 67],
            [5, 105, 106],
            [6, 18, 136],
            [7, 58, 1],
            [8, 126, 17],
            [9, 32, 209],
            [10, 32, 207],
            [11, 118, 119],
            [12, 67, 4],
            [13, 250, 251],
            [14, 116, 11],
            [15, 204, 31],
            [16, 270, 53],
            [17, 142, 20],
            [18, 182, 183],
            [19, 184, 183],
            [20, 238, 44],
            [21, 226, 225],
            [22, 63, 64],
            [23, 199, 198],
            [24, 264, 263],
            [25, 17, 131],
            [26, 49, -1],
            [27, 149, 148],
            [28, 207, 208],
            [29, 202, 203],
            [30, 42, 221],
            [31, 169, 170],
            [32, 129, 130],
            [33, 66, 67],
            [34, 43, 244],
            [35, 125, 124],
            [36, 234, 233],
            [37, 141, 24],
            [38, 187, 186],
            [39, 263, 264],
            [40, 111, 112],
            [41, 132, 131],
            [42, 244, 43],
            [43, 265, 264],
            [44, 174, 173],
            [45, 114, 113],
            [46, 114, 113],
            [47, 114, 113],
            [48, 113, 114],
            [49, 113, 114],
        ]
    )
    # generate data
    data_map_1600, _data_gdf_1600, _network_structure_1600 = override_coords(G, 1600)
    # from cityseer.tools import plot
    # plot.plot_network_structure(_network_structure_1600, _data_gdf_1600)
    # plot.plot_assignment(_network_structure_1600, G, _data_gdf_1600)
    # for idx in range(data_map_1600.count):
    #     print(idx, data_map_1600.nearest_assign[idx], data_map_1600.next_nearest_assign[idx])
    # assignment map includes data x, data y, nearest assigned, next nearest assigned
    assert np.allclose(data_map_1600.nearest_assign, targets[:, 1], atol=0, rtol=0)
    assert np.allclose(data_map_1600.next_nearest_assign, targets[:, 2], atol=0, rtol=0)
    # max distance of 0 should return all NaN
    data_map_0, _, _ = override_coords(G, 0)
    assert np.all(data_map_0.nearest_assign == -1)
    assert np.all(data_map_0.next_nearest_assign == -1)
    # with enough distance, all should be matched except location 26
    data_map_2000, _, _ = override_coords(G, 2000)
    assert not np.any(data_map_2000.nearest_assign == -1)
    assert np.where(data_map_2000.next_nearest_assign == -1)[0][0] == 26


def test_aggregate_to_src_idx(primal_graph):
    for max_dist in [400, 750]:
        # generate data
        _nodes_gpd, network_structure = graphs.network_structure_from_nx(primal_graph, 3395)
        data_gdf = mock.mock_data_gdf(primal_graph, random_seed=13)
        data_map, data_gdf = layers.assign_gdf_to_network(data_gdf, network_structure, 400)
        # in this case, use same assignment max dist as search max dist
        for angular in [True, False]:
            for netw_src_idx in range(network_structure.nodes.count):
                # aggregate to src...
                reachable_data, reachable_data_dist = data.aggregate_to_src_idx(
                    netw_src_idx,
                    network_structure.nodes.xs,
                    network_structure.nodes.ys,
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
                    np.float32(max_dist),
                    angular=angular,
                )
                # compare to manual checks on distances:
                # get the network distances
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
                    network_structure.edges.start,
                    network_structure.edges.end,
                    network_structure.edges.length,
                    network_structure.edges.angle_sum,
                    network_structure.edges.imp_factor,
                    network_structure.edges.in_bearing,
                    network_structure.edges.out_bearing,
                    network_structure.node_edge_map,
                    netw_src_idx,
                    max_dist=np.float32(max_dist),
                    angular=angular,
                )
                # for debugging
                # from cityseer.tools import plot
                # plot.plot_network_structure(network_structure, data_map)
                # verify distances vs. the max
                for d_idx in range(data_map.count):
                    # check the integrity of the distances and classes
                    reachable = reachable_data[d_idx]
                    reachable_dist = reachable_data_dist[d_idx]
                    # get the distance via the nearest assigned index
                    nearest_dist = np.inf
                    # if a nearest node has been assigned
                    if data_map.nearest_assign[d_idx] != -1:
                        # get the index for the assigned network node
                        netw_idx = data_map.nearest_assign[d_idx]
                        # if this node is within the cutoff distance:
                        if short_dist[netw_idx] < max_dist:
                            # get the distances from the data point to the assigned network node
                            d_d = np.hypot(
                                data_map.xs[d_idx] - network_structure.nodes.xs[netw_idx],
                                data_map.ys[d_idx] - network_structure.nodes.ys[netw_idx],
                            )
                            # and add it to the network distance path from the source to the assigned node
                            n_d = short_dist[netw_idx]
                            nearest_dist = d_d + n_d
                    # also get the distance via the next nearest assigned index
                    next_nearest_dist = np.inf
                    # if a nearest node has been assigned
                    if data_map.next_nearest_assign[d_idx] != -1:
                        # get the index for the assigned network node
                        netw_idx = data_map.next_nearest_assign[d_idx]
                        # if this node is within the radial cutoff distance:
                        if short_dist[netw_idx] < max_dist:
                            # get the distances from the data point to the assigned network node
                            d_d = np.hypot(
                                data_map.xs[d_idx] - network_structure.nodes.xs[netw_idx],
                                data_map.ys[d_idx] - network_structure.nodes.ys[netw_idx],
                            )
                            # and add it to the network distance path from the source to the assigned node
                            n_d = short_dist[netw_idx]
                            next_nearest_dist = d_d + n_d
                    # now check distance integrity
                    if np.isinf(reachable_dist):
                        assert not reachable
                        assert nearest_dist > max_dist and next_nearest_dist > max_dist
                    else:
                        assert reachable
                        assert reachable_dist <= max_dist
                        if nearest_dist < next_nearest_dist:
                            assert reachable_dist == nearest_dist
                        else:
                            assert reachable_dist == next_nearest_dist


def test_accessibility(primal_graph):
    # generate node and edge maps
    _nodes_gpd, network_structure = graphs.network_structure_from_nx(primal_graph, 3395)
    data_gdf = mock.mock_landuse_categorical_data(primal_graph, random_seed=13)
    data_map, data_gdf = layers.assign_gdf_to_network(data_gdf, network_structure, 500, data_id_col="data_id")
    lab_enc = LabelEncoder()
    encoded_labels: npt.NDArray[np.int_] = lab_enc.fit_transform(data_gdf["categorical_landuses"])  # type: ignore
    # set parameters
    betas: npt.NDArray[np.float32] = np.array([0.02, 0.01, 0.005, 0.0025], dtype=np.float32)
    distances = networks.distance_from_beta(betas)
    max_curve_wts = networks.clip_weights_curve(distances, betas, 0)
    # all datapoints and types are completely unique except for the last five - which all point to the same source
    ac_dupe_check_key: int = lab_enc.transform(["z"])[0]  # type: ignore
    # set the keys - add shuffling to be sure various orders work
    ac_keys = np.array([1, 2, 5, ac_dupe_check_key])
    np.random.shuffle(ac_keys)
    # check that problematic keys are caught
    for ac_key in [[-1], [max(encoded_labels) + 1], [1, 1]]:
        with pytest.raises(ValueError):
            data.accessibility(
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
                max_curve_wts,
                encoded_labels,
                accessibility_keys=np.array(ac_key),
            )
    # generate
    ac_data, ac_data_wt = data.accessibility(
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
        max_curve_wts,
        landuse_encodings=encoded_labels,
        accessibility_keys=ac_keys,
        angular=False,
    )
    # access non-weighted
    ac_1_nw = ac_data[np.where(ac_keys == 1)][0]
    ac_2_nw = ac_data[np.where(ac_keys == 2)][0]
    ac_5_nw = ac_data[np.where(ac_keys == 5)][0]
    ac_dupe_nw = ac_data[np.where(ac_keys == ac_dupe_check_key)][0]
    # access weighted
    ac_1_w = ac_data_wt[np.where(ac_keys == 1)][0]
    ac_2_w = ac_data_wt[np.where(ac_keys == 2)][0]
    ac_5_w = ac_data_wt[np.where(ac_keys == 5)][0]
    ac_dupe_w = ac_data_wt[np.where(ac_keys == ac_dupe_check_key)][0]
    # test manual metrics against all nodes
    mu_max_unique = len(lab_enc.classes_)  # type: ignore
    # test against various distances
    for d_idx, (dist_cutoff, beta) in enumerate(zip(distances, betas)):
        for src_idx in range(len(primal_graph)):  # type: ignore
            reachable_data, reachable_data_dist = data.aggregate_to_src_idx(
                src_idx,
                network_structure.nodes.xs,
                network_structure.nodes.ys,
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
                dist_cutoff,
            )
            # counts of each class type (array length per max unique classes - not just those within max distance)
            cl_counts: npt.NDArray[np.int_] = np.full(mu_max_unique, 0)
            # nearest of each class type (likewise)
            cl_nearest: npt.NDArray[np.float32] = np.full(mu_max_unique, np.inf)
            # aggregate
            a_1_nw = 0
            a_2_nw = 0
            a_5_nw = 0
            a_1_w = 0
            a_2_w = 0
            a_5_w = 0
            # iterate reachable
            for data_idx, (reachable, data_dist) in enumerate(zip(reachable_data, reachable_data_dist)):
                if not reachable:
                    continue
                cl = encoded_labels[data_idx]
                # double check distance is within threshold
                assert data_dist <= dist_cutoff
                # update the class counts
                cl_counts[cl] += 1
                # if distance is nearer, update the nearest distance array too
                if data_dist < cl_nearest[cl]:
                    cl_nearest[cl] = data_dist
                # aggregate accessibility codes
                if cl == 1:
                    a_1_nw += 1
                    a_1_w += np.exp(-beta * data_dist)
                elif cl == 2:
                    a_2_nw += 1
                    a_2_w += np.exp(-beta * data_dist)
                elif cl == 5:
                    a_5_nw += 1
                    a_5_w += np.exp(-beta * data_dist)
            # assertions
            assert np.isclose(ac_1_nw[d_idx, src_idx], a_1_nw, rtol=config.RTOL, atol=config.ATOL)
            assert np.isclose(ac_2_nw[d_idx, src_idx], a_2_nw, rtol=config.RTOL, atol=config.ATOL)
            assert np.isclose(ac_5_nw[d_idx, src_idx], a_5_nw, rtol=config.RTOL, atol=config.ATOL)
            assert np.isclose(ac_1_w[d_idx, src_idx], a_1_w, rtol=config.RTOL, atol=config.ATOL)
            assert np.isclose(ac_2_w[d_idx, src_idx], a_2_w, rtol=config.RTOL, atol=config.ATOL)
            assert np.isclose(ac_5_w[d_idx, src_idx], a_5_w, rtol=config.RTOL, atol=config.ATOL)
    # there should be five duplicates in source encoded labels
    assert len(np.where(encoded_labels == ac_dupe_check_key)[0]) == 5
    # deduplication means dedupes are max 1
    assert np.max(ac_dupe_nw) == 1
    assert np.min(ac_dupe_nw) == 0
    # weighted
    assert np.all(ac_dupe_w <= 1)

    # check that angular is passed-through
    # actual angular tests happen in test_shortest_path_tree()
    # here the emphasis is simply on checking that the angular instruction gets chained through

    # setup dual data
    G_dual = graphs.nx_to_dual(primal_graph)

    _nodes_gpd_dual, network_structure_dual = graphs.network_structure_from_nx(G_dual, 3395)
    data_gdf_dual = mock.mock_landuse_categorical_data(primal_graph, random_seed=13)
    data_map_dual, data_gdf_dual = layers.assign_gdf_to_network(data_gdf_dual, network_structure_dual, 500)
    lab_enc_dual = LabelEncoder()
    encoded_labels_dual: npt.NDArray[np.int_] = lab_enc_dual.fit_transform(data_gdf_dual["categorical_landuses"])
    # checks
    ac_dual, ac_wt_dual = data.accessibility(
        network_structure_dual.nodes.xs,
        network_structure_dual.nodes.ys,
        network_structure_dual.nodes.live,
        network_structure_dual.edges.start,
        network_structure_dual.edges.end,
        network_structure_dual.edges.length,
        network_structure_dual.edges.angle_sum,
        network_structure_dual.edges.imp_factor,
        network_structure_dual.edges.in_bearing,
        network_structure_dual.edges.out_bearing,
        network_structure_dual.node_edge_map,
        data_map_dual.xs,
        data_map_dual.ys,
        data_map_dual.nearest_assign,
        data_map_dual.next_nearest_assign,
        data_map_dual.data_id,
        distances,
        betas,
        max_curve_wts,
        encoded_labels_dual,
        accessibility_keys=ac_keys,
        angular=True,
    )

    ac_dual_sidestep, ac_wt_dual_sidestep = data.accessibility(
        network_structure_dual.nodes.xs,
        network_structure_dual.nodes.ys,
        network_structure_dual.nodes.live,
        network_structure_dual.edges.start,
        network_structure_dual.edges.end,
        network_structure_dual.edges.length,
        network_structure_dual.edges.angle_sum,
        network_structure_dual.edges.imp_factor,
        network_structure_dual.edges.in_bearing,
        network_structure_dual.edges.out_bearing,
        network_structure_dual.node_edge_map,
        data_map_dual.xs,
        data_map_dual.ys,
        data_map_dual.nearest_assign,
        data_map_dual.next_nearest_assign,
        data_map_dual.data_id,
        distances,
        betas,
        max_curve_wts,
        encoded_labels_dual,
        accessibility_keys=ac_keys,
        angular=False,
    )

    assert not np.allclose(ac_dual, ac_dual_sidestep, atol=config.ATOL, rtol=config.RTOL)
    assert not np.allclose(ac_wt_dual, ac_wt_dual_sidestep, atol=config.ATOL, rtol=config.RTOL)


def test_mixed_uses_signatures(primal_graph):
    # generate node and edge maps
    _nodes_gpd, network_structure = graphs.network_structure_from_nx(primal_graph, 3395)
    data_gdf = mock.mock_landuse_categorical_data(primal_graph, random_seed=13)
    data_map, data_gdf = layers.assign_gdf_to_network(data_gdf, network_structure, 500)
    lab_enc = LabelEncoder()
    encoded_labels: npt.NDArray[np.int_] = lab_enc.fit_transform(data_gdf["categorical_landuses"])  # type: ignore
    # set parameters
    betas: npt.NDArray[np.float32] = np.array([0.02, 0.01, 0.005, 0.0025])
    distances = networks.distance_from_beta(betas)
    max_curve_wts = networks.clip_weights_curve(distances, betas, 0)
    qs: npt.NDArray[np.float32] = np.array([0, 1, 2])
    # check that empty land_use encodings are caught
    with pytest.raises(ValueError):
        data.mixed_uses(
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
            distances,
            betas,
            max_curve_wts,
            mixed_use_hill_keys=np.array([0], dtype=np.int_),
        )
    # check that unequal land_use encodings vs data map lengths are caught
    with pytest.raises(ValueError):
        data.mixed_uses(
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
            distances,
            betas,
            max_curve_wts,
            landuse_encodings=encoded_labels[:-1],
            mixed_use_other_keys=np.array([0]),
        )
    # check that no provided metrics flags
    with pytest.raises(ValueError):
        data.mixed_uses(
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
            distances,
            betas,
            max_curve_wts,
            landuse_encodings=encoded_labels,
        )
    # check that missing qs flags
    with pytest.raises(ValueError):
        data.mixed_uses(
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
            distances,
            betas,
            max_curve_wts,
            mixed_use_hill_keys=np.array([0]),
            landuse_encodings=encoded_labels,
        )
    # check that problematic mixed use and accessibility keys are caught
    for mu_h_key, mu_o_key in [
        # negatives
        ([-1], [1]),
        ([1], [-1]),
        # out of range
        ([4], [1]),
        ([1], [3]),
        # duplicates
        ([1, 1], [1]),
        ([1], [1, 1]),
    ]:
        with pytest.raises(ValueError):
            data.mixed_uses(
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
                distances,
                betas,
                max_curve_wts,
                encoded_labels,
                qs=qs,
                mixed_use_hill_keys=np.array(mu_h_key),
                mixed_use_other_keys=np.array(mu_o_key),
            )
    for h_key, o_key in (([3], []), ([], [2])):
        # check that missing matrix is caught for disparity weighted indices
        with pytest.raises(ValueError):
            data.mixed_uses(
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
                distances,
                betas,
                max_curve_wts,
                landuse_encodings=encoded_labels,
                qs=qs,
                mixed_use_hill_keys=np.array(h_key),
                mixed_use_other_keys=np.array(o_key),
            )
        # check that non-square disparity matrix is caught
        mock_matrix = np.full((len(lab_enc.classes_), len(lab_enc.classes_)), 1)
        with pytest.raises(ValueError):
            data.mixed_uses(
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
                distances,
                betas,
                max_curve_wts,
                landuse_encodings=encoded_labels,
                qs=qs,
                mixed_use_hill_keys=np.array(h_key),
                mixed_use_other_keys=np.array(o_key),
                cl_disparity_wt_matrix=mock_matrix[:-1],
            )


def test_mixed_uses(primal_graph):
    # generate node and edge maps
    _nodes_gpd, network_structure = graphs.network_structure_from_nx(primal_graph, 3395)
    data_gdf = mock.mock_landuse_categorical_data(primal_graph, random_seed=13)
    data_map, data_gdf = layers.assign_gdf_to_network(data_gdf, network_structure, 500)
    lab_enc = LabelEncoder()
    encoded_labels: npt.NDArray[np.int_] = lab_enc.fit_transform(data_gdf["categorical_landuses"])  # type: ignore
    # set parameters
    betas: npt.NDArray[np.float32] = np.array([0.02, 0.01, 0.005, 0.0025], dtype=np.float32)
    distances = networks.distance_from_beta(betas)
    max_curve_wts = networks.clip_weights_curve(distances, betas, 0)
    qs: npt.NDArray[np.float32] = np.array([0, 1, 2])
    mock_matrix: npt.NDArray[np.float32] = np.full((len(lab_enc.classes_), len(lab_enc.classes_)), 1)
    # set the keys - add shuffling to be sure various orders work
    hill_keys: npt.NDArray[np.int_] = np.arange(4)
    np.random.shuffle(hill_keys)
    non_hill_keys: npt.NDArray[np.int_] = np.arange(3)
    np.random.shuffle(non_hill_keys)
    ac_keys: npt.NDArray[np.int_] = np.array([1, 2, 5])
    np.random.shuffle(ac_keys)
    # generate
    mu_data_hill, mu_data_other = data.mixed_uses(
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
        distances,
        betas,
        max_curve_wts,
        landuse_encodings=encoded_labels,
        qs=qs,
        mixed_use_hill_keys=hill_keys,
        mixed_use_other_keys=non_hill_keys,
        cl_disparity_wt_matrix=mock_matrix,
        angular=False,
    )
    # hill
    hill = mu_data_hill[np.where(hill_keys == 0)][0]
    hill_branch_wt = mu_data_hill[np.where(hill_keys == 1)][0]
    hill_pw_wt = mu_data_hill[np.where(hill_keys == 2)][0]
    hill_disp_wt = mu_data_hill[np.where(hill_keys == 3)][0]
    # non hill
    shannon = mu_data_other[np.where(non_hill_keys == 0)][0]
    gini = mu_data_other[np.where(non_hill_keys == 1)][0]
    raos = mu_data_other[np.where(non_hill_keys == 2)][0]
    # test manual metrics against all nodes
    mu_max_unique = len(lab_enc.classes_)
    # test against various distances
    for d_idx, (dist_cutoff, beta) in enumerate(zip(distances, betas)):
        for src_idx in range(len(primal_graph)):
            reachable_data, reachable_data_dist = data.aggregate_to_src_idx(
                src_idx,
                network_structure.nodes.xs,
                network_structure.nodes.ys,
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
                dist_cutoff,
            )
            # counts of each class type (array length per max unique classes - not just those within max distance)
            cl_counts: npt.NDArray[np.int_] = np.full(mu_max_unique, 0)
            # nearest of each class type (likewise)
            cl_nearest: npt.NDArray[np.float32] = np.full(mu_max_unique, np.inf)
            # aggregate
            a_1_nw = 0
            a_2_nw = 0
            a_5_nw = 0
            a_1_w = 0
            a_2_w = 0
            a_5_w = 0
            # iterate reachable
            for data_idx, (reachable, data_dist) in enumerate(zip(reachable_data, reachable_data_dist)):
                if not reachable:
                    continue
                cl = encoded_labels[data_idx]
                # double check distance is within threshold
                assert data_dist <= dist_cutoff
                # update the class counts
                cl_counts[cl] += 1
                # if distance is nearer, update the nearest distance array too
                if data_dist < cl_nearest[cl]:
                    cl_nearest[cl] = data_dist
                # aggregate accessibility codes
                if cl == 1:
                    a_1_nw += 1
                    a_1_w += np.exp(-beta * data_dist)
                elif cl == 2:
                    a_2_nw += 1
                    a_2_w += np.exp(-beta * data_dist)
                elif cl == 5:
                    a_5_nw += 1
                    a_5_w += np.exp(-beta * data_dist)
            # assertions
            assert np.isclose(
                hill[0, d_idx, src_idx],
                diversity.hill_diversity(cl_counts, np.float32(0)),
                rtol=config.RTOL,
                atol=config.ATOL,
            )
            assert np.isclose(
                hill[1, d_idx, src_idx],
                diversity.hill_diversity(cl_counts, np.float32(1)),
                rtol=config.RTOL,
                atol=config.ATOL,
            )
            assert np.isclose(
                hill[2, d_idx, src_idx],
                diversity.hill_diversity(cl_counts, np.float32(2)),
                rtol=config.RTOL,
                atol=config.ATOL,
            )
            assert np.isclose(
                hill_branch_wt[0, d_idx, src_idx],
                diversity.hill_diversity_branch_distance_wt(cl_counts, cl_nearest, np.float32(0), beta),
                rtol=config.RTOL,
                atol=config.ATOL,
            )
            assert np.isclose(
                hill_branch_wt[1, d_idx, src_idx],
                diversity.hill_diversity_branch_distance_wt(cl_counts, cl_nearest, np.float32(1), beta),
                rtol=config.RTOL,
                atol=config.ATOL,
            )
            assert np.isclose(
                hill_branch_wt[2, d_idx, src_idx],
                diversity.hill_diversity_branch_distance_wt(cl_counts, cl_nearest, np.float32(2), beta),
                rtol=config.RTOL,
                atol=config.ATOL,
            )
            assert np.isclose(
                hill_pw_wt[0, d_idx, src_idx],
                diversity.hill_diversity_pairwise_distance_wt(cl_counts, cl_nearest, np.float32(0), beta),
                rtol=config.RTOL,
                atol=config.ATOL,
            )
            assert np.isclose(
                hill_pw_wt[1, d_idx, src_idx],
                diversity.hill_diversity_pairwise_distance_wt(cl_counts, cl_nearest, np.float32(1), beta),
                rtol=config.RTOL,
                atol=config.ATOL,
            )
            assert np.isclose(
                hill_pw_wt[2, d_idx, src_idx],
                diversity.hill_diversity_pairwise_distance_wt(cl_counts, cl_nearest, np.float32(2), beta),
                rtol=config.RTOL,
                atol=config.ATOL,
            )
            assert np.isclose(
                hill_disp_wt[0, d_idx, src_idx],
                diversity.hill_diversity_pairwise_matrix_wt(cl_counts, mock_matrix, np.float32(0)),
                rtol=config.RTOL,
                atol=config.ATOL,
            )
            assert np.isclose(
                hill_disp_wt[1, d_idx, src_idx],
                diversity.hill_diversity_pairwise_matrix_wt(cl_counts, mock_matrix, np.float32(1)),
                rtol=config.RTOL,
                atol=config.ATOL,
            )
            assert np.isclose(
                hill_disp_wt[2, d_idx, src_idx],
                diversity.hill_diversity_pairwise_matrix_wt(cl_counts, mock_matrix, np.float32(2)),
                rtol=config.RTOL,
                atol=config.ATOL,
            )
            assert np.isclose(
                shannon[d_idx, src_idx], diversity.shannon_diversity(cl_counts), rtol=config.RTOL, atol=config.ATOL
            )
            assert np.isclose(
                gini[d_idx, src_idx], diversity.gini_simpson_diversity(cl_counts), rtol=config.RTOL, atol=config.ATOL
            )
            assert np.isclose(
                raos[d_idx, src_idx],
                diversity.raos_quadratic_diversity(cl_counts, mock_matrix),
                rtol=config.RTOL,
                atol=config.ATOL,
            )
    # check that angular is passed-through
    # actual angular tests happen in test_shortest_path_tree()
    # here the emphasis is simply on checking that the angular instruction gets chained through
    # setup dual data
    G_dual = graphs.nx_to_dual(primal_graph)
    _nodes_gpd_dual, network_structure_dual = graphs.network_structure_from_nx(G_dual, 3395)
    data_gdf_dual = mock.mock_landuse_categorical_data(primal_graph, random_seed=13)
    data_map_dual, data_gdf_dual = layers.assign_gdf_to_network(data_gdf_dual, network_structure_dual, 500)
    lab_enc_dual = LabelEncoder()
    encoded_labels_dual: npt.NDArray[np.int_] = lab_enc_dual.fit_transform(data_gdf_dual["categorical_landuses"])
    mock_matrix: npt.NDArray[np.float32] = np.full((len(lab_enc_dual.classes_), len(lab_enc_dual.classes_)), 1)
    # checks
    mu_hill_dual, mu_other_dual = data.mixed_uses(
        network_structure_dual.nodes.xs,
        network_structure_dual.nodes.ys,
        network_structure_dual.nodes.live,
        network_structure_dual.edges.start,
        network_structure_dual.edges.end,
        network_structure_dual.edges.length,
        network_structure_dual.edges.angle_sum,
        network_structure_dual.edges.imp_factor,
        network_structure_dual.edges.in_bearing,
        network_structure_dual.edges.out_bearing,
        network_structure_dual.node_edge_map,
        data_map_dual.xs,
        data_map_dual.ys,
        data_map_dual.nearest_assign,
        data_map_dual.next_nearest_assign,
        distances,
        betas,
        max_curve_wts,
        encoded_labels_dual,
        qs=qs,
        mixed_use_hill_keys=hill_keys,
        mixed_use_other_keys=non_hill_keys,
        cl_disparity_wt_matrix=mock_matrix,
        angular=True,
    )
    mu_hill_dual_sidestep, mu_other_dual_sidestep = data.mixed_uses(
        network_structure_dual.nodes.xs,
        network_structure_dual.nodes.ys,
        network_structure_dual.nodes.live,
        network_structure_dual.edges.start,
        network_structure_dual.edges.end,
        network_structure_dual.edges.length,
        network_structure_dual.edges.angle_sum,
        network_structure_dual.edges.imp_factor,
        network_structure_dual.edges.in_bearing,
        network_structure_dual.edges.out_bearing,
        network_structure_dual.node_edge_map,
        data_map_dual.xs,
        data_map_dual.ys,
        data_map_dual.nearest_assign,
        data_map_dual.next_nearest_assign,
        distances,
        betas,
        max_curve_wts,
        encoded_labels_dual,
        qs=qs,
        mixed_use_hill_keys=hill_keys,
        mixed_use_other_keys=non_hill_keys,
        cl_disparity_wt_matrix=mock_matrix,
        angular=False,
    )
    assert not np.allclose(mu_hill_dual, mu_hill_dual_sidestep, atol=config.ATOL, rtol=config.RTOL)
    assert not np.allclose(mu_other_dual, mu_other_dual_sidestep, atol=config.ATOL, rtol=config.RTOL)


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
