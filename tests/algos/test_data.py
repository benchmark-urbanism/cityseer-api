# pyright: basic
from __future__ import annotations

import networkx as nx
import numpy as np
import numpy.typing as npt
import pytest

from cityseer import config, structures
from cityseer.algos import centrality, data, diversity
from cityseer.metrics import layers, networks
from cityseer.tools import graphs, mock


def test_find_nearest(primal_graph):
    cc_netw = networks.NetworkLayerFromNX(primal_graph, distances=[100])
    # generate some data
    data_dict = mock.mock_data_dict(primal_graph)
    cc_data = layers.DataLayerFromDict(data_dict)
    # test the filter - iterating each point in data map
    for d_idx in range(cc_data.data_map.count):
        # find the closest point on the network
        d_x, d_y = cc_data.data_map.x_y(d_idx)
        min_idx, min_dist, _next_min_idx = data.find_nearest(
            d_x, d_y, cc_netw.network_structure.nodes, max_dist=np.float32(500)
        )
        # check that no other indices are nearer
        for n_idx in range(cc_netw.network_structure.nodes.count):
            n_x = cc_netw.network_structure.nodes.xs[n_idx]
            n_y = cc_netw.network_structure.nodes.ys[n_idx]
            dist = np.sqrt((d_x - n_x) ** 2 + (d_y - n_y) ** 2)
            if n_idx == min_idx:
                assert np.isclose(dist, min_dist, rtol=config.RTOL, atol=config.ATOL)
            else:
                assert dist > min_dist


def override_coords(graph: nx.MultiGraph, max_dist: int) -> structures.DataMap:
    data_dict = mock.mock_data_dict(graph, random_seed=25)
    _data_keys, data_map = layers.data_map_from_dict(data_dict)
    _node_keys, network_structure = graphs.network_structure_from_nx(graph)
    data_map.xs[18] = 701200
    data_map.ys[18] = 5719400
    data_map.xs[39] = 700750
    data_map.ys[39] = 5720025
    data_map.xs[26] = 700400
    data_map.ys[26] = 5719525
    data_map = data.assign_to_network(data_map, network_structure, max_dist=np.float32(max_dist))
    return data_map


def test_assign_to_network(primal_graph):
    # create additional dead-end scenario
    primal_graph.remove_edge(14, 15)
    primal_graph.remove_edge(15, 28)
    # G = graphs.nx_auto_edge_params(G)
    G = graphs.nx_decompose(primal_graph, 50)
    # visually confirmed in plots
    targets = np.array(
        [
            [0, 164, 163],
            [1, 42, 241],
            [2, 236, 235],
            [3, 48, 262],
            [4, 211, 212],
            [5, 236, 235],
            [6, 58, 57],
            [7, 72, 5],
            [8, 75, 76],
            [9, 92, 9],
            [10, 61, 62],
            [11, 96, 13],
            [12, 0, 59],
            [13, 98, 99],
            [14, 203, 202],
            [15, 121, 120],
            [16, 48, 262],
            [17, 2, 70],
            [18, 182, 183],
            [19, 158, 157],
            [20, 83, 84],
            [21, 2, 65],
            [22, 171, 170],
            [23, 266, 52],
            [24, 83, 84],
            [25, 88, 11],
            [26, 49, -1],
            [27, 19, 138],
            [28, 134, 135],
            [29, 262, 46],
            [30, 78, 9],
            [31, 188, 189],
            [32, 180, 181],
            [33, 95, 94],
            [34, 226, 225],
            [35, 110, 111],
            [36, 39, 228],
            [37, 158, 25],
            [38, 88, 87],
            [39, 263, 264],
            [40, 120, 121],
            [41, 146, 21],
            [42, 10, 97],
            [43, 119, 118],
            [44, 82, 5],
            [45, 116, 11],
            [46, 100, 99],
            [47, 138, 19],
            [48, 14, 119],
            [49, 106, 105],
        ]
    )
    # generate data
    data_map_1600 = override_coords(G, 1600)
    # from cityseer.tools import plot
    # _node_keys, network_structure = graphs.network_structure_from_nx(G)
    # plot.plot_network_structure(network_structure, data_map_1600)
    # plot.plot_assignment(network_structure, G, data_map_1600)
    # assignment map includes data x, data y, nearest assigned, next nearest assigned
    assert np.allclose(data_map_1600.nearest_assign, targets[:, 1], atol=0, rtol=0)
    assert np.allclose(data_map_1600.next_nearest_assign, targets[:, 2], atol=0, rtol=0)
    # max distance of 0 should return all NaN
    data_map_0 = override_coords(G, 0)
    # plot.plot_network_structure(network_structure, data_map_0)
    # plot.plot_assignment(network_structure, G, data_map_0)
    assert np.all(data_map_0.nearest_assign == -1)
    assert np.all(data_map_0.next_nearest_assign == -1)
    # with enough distance, all should be matched except location 26
    data_map_2000 = override_coords(G, 2000)
    # plot.plot_network_structure(network_structure, data_map_2000)
    # plot.plot_assignment(network_structure, G, data_map_2000)
    assert not np.any(data_map_2000.nearest_assign == -1)
    assert np.where(data_map_2000.next_nearest_assign == -1)[0][0] == 26


def test_aggregate_to_src_idx(primal_graph):
    for max_dist in [400, 750]:
        # generate data
        _node_keys, network_structure = graphs.network_structure_from_nx(primal_graph)
        data_dict = mock.mock_data_dict(primal_graph, random_seed=13)
        _data_keys, data_map = layers.data_map_from_dict(data_dict)
        # in this case, use same assignment max dist as search max dist
        data_map = data.assign_to_network(data_map, network_structure, max_dist=np.float32(max_dist))
        for angular in [True, False]:
            for netw_src_idx in range(network_structure.nodes.count):
                # aggregate to src...
                (reachable_data, reachable_data_dist, _tree_map) = data.aggregate_to_src_idx(
                    netw_src_idx,
                    network_structure,
                    data_map,
                    np.float32(max_dist),
                    angular=angular,
                )
                # compare to manual checks on distances:
                # get the network distances
                tree_map = centrality.shortest_path_tree(
                    network_structure,
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
                        if tree_map.short_dist[netw_idx] < max_dist:
                            # get the distances from the data point to the assigned network node
                            d_d = np.hypot(
                                data_map.xs[d_idx] - network_structure.nodes.xs[netw_idx],
                                data_map.ys[d_idx] - network_structure.nodes.ys[netw_idx],
                            )
                            # and add it to the network distance path from the source to the assigned node
                            n_d = tree_map.short_dist[netw_idx]
                            nearest_dist = d_d + n_d
                    # also get the distance via the next nearest assigned index
                    next_nearest_dist = np.inf
                    # if a nearest node has been assigned
                    if data_map.next_nearest_assign[d_idx] != -1:
                        # get the index for the assigned network node
                        netw_idx = data_map.next_nearest_assign[d_idx]
                        # if this node is within the radial cutoff distance:
                        if tree_map.short_dist[netw_idx] < max_dist:
                            # get the distances from the data point to the assigned network node
                            d_d = np.hypot(
                                data_map.xs[d_idx] - network_structure.nodes.xs[netw_idx],
                                data_map.ys[d_idx] - network_structure.nodes.ys[netw_idx],
                            )
                            # and add it to the network distance path from the source to the assigned node
                            n_d = tree_map.short_dist[netw_idx]
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


# TODO: pickling error re: cache and NodeMap
def test_aggregate_landuses_signatures(primal_graph):
    # generate node and edge maps
    _node_keys, network_structure = graphs.network_structure_from_nx(primal_graph)
    # setup data
    data_dict = mock.mock_data_dict(primal_graph, random_seed=13)
    _data_keys, data_map = layers.data_map_from_dict(data_dict)
    data_map = data.assign_to_network(data_map, network_structure, np.float32(500))
    # set parameters
    betas: npt.NDArray[np.float32] = np.array([0.02, 0.01, 0.005, 0.0025])
    distances = networks.distance_from_beta(betas)
    qs: npt.NDArray[np.float32] = np.array([0, 1, 2])
    mock_categorical = mock.mock_categorical_data(data_map.count)
    landuse_classes, landuse_encodings = layers.encode_categorical(mock_categorical)
    # check that empty land_use encodings are caught
    with pytest.raises(ValueError):
        data.aggregate_landuses(
            network_structure,
            data_map,
            distances,
            betas,
            mixed_use_hill_keys=np.array([0], dtype=np.int_),
        )
    # check that unequal land_use encodings vs data map lengths are caught
    with pytest.raises(ValueError):
        data.aggregate_landuses(
            network_structure,
            data_map,
            distances,
            betas,
            landuse_encodings=landuse_encodings[:-1],
            mixed_use_other_keys=np.array([0]),
        )
    # check that no provided metrics flags
    with pytest.raises(ValueError):
        data.aggregate_landuses(
            network_structure,
            data_map,
            distances,
            betas,
            landuse_encodings=landuse_encodings,
        )
    # check that missing qs flags
    with pytest.raises(ValueError):
        data.aggregate_landuses(
            network_structure,
            data_map,
            distances,
            betas,
            mixed_use_hill_keys=np.array([0]),
            landuse_encodings=landuse_encodings,
        )
    # check that problematic mixed use and accessibility keys are caught
    for mu_h_key, mu_o_key, ac_key in [
        # negatives
        ([-1], [1], [1]),
        ([1], [-1], [1]),
        ([1], [1], [-1]),
        # out of range
        ([4], [1], [1]),
        ([1], [3], [1]),
        ([1], [1], [max(landuse_encodings) + 1]),
        # duplicates
        ([1, 1], [1], [1]),
        ([1], [1, 1], [1]),
        ([1], [1], [1, 1]),
    ]:
        with pytest.raises(ValueError):
            data.aggregate_landuses(
                network_structure,
                data_map,
                distances,
                betas,
                landuse_encodings,
                qs=qs,
                mixed_use_hill_keys=np.array(mu_h_key),
                mixed_use_other_keys=np.array(mu_o_key),
                accessibility_keys=np.array(ac_key),
            )
    for h_key, o_key in (([3], []), ([], [2])):
        # check that missing matrix is caught for disparity weighted indices
        with pytest.raises(ValueError):
            data.aggregate_landuses(
                network_structure,
                data_map,
                distances,
                betas,
                landuse_encodings=landuse_encodings,
                qs=qs,
                mixed_use_hill_keys=np.array(h_key),
                mixed_use_other_keys=np.array(o_key),
            )
        # check that non-square disparity matrix is caught
        mock_matrix = np.full((len(landuse_classes), len(landuse_classes)), 1)
        with pytest.raises(ValueError):
            data.aggregate_landuses(
                network_structure,
                data_map,
                distances,
                betas,
                landuse_encodings=landuse_encodings,
                qs=qs,
                mixed_use_hill_keys=np.array(h_key),
                mixed_use_other_keys=np.array(o_key),
                cl_disparity_wt_matrix=mock_matrix[:-1],
            )


def test_aggregate_landuses_categorical_components(primal_graph):
    # generate node and edge maps
    _node_keys, network_structure = graphs.network_structure_from_nx(primal_graph)
    # setup data
    data_dict = mock.mock_data_dict(primal_graph, random_seed=13)
    _data_keys, data_map = layers.data_map_from_dict(data_dict)
    data_map = data.assign_to_network(data_map, network_structure, np.float32(500))
    # set parameters
    betas: npt.NDArray[np.float32] = np.array([0.02, 0.01, 0.005, 0.0025], dtype=np.float32)
    distances = networks.distance_from_beta(betas)
    qs: npt.NDArray[np.float32] = np.array([0, 1, 2])
    mock_categorical = mock.mock_categorical_data(data_map.count)
    landuse_classes, landuse_encodings = layers.encode_categorical(mock_categorical)
    mock_matrix: npt.NDArray[np.float32] = np.full((len(landuse_classes), len(landuse_classes)), 1)
    # set the keys - add shuffling to be sure various orders work
    hill_keys: npt.NDArray[np.int_] = np.arange(4)
    np.random.shuffle(hill_keys)
    non_hill_keys: npt.NDArray[np.int_] = np.arange(3)
    np.random.shuffle(non_hill_keys)
    ac_keys: npt.NDArray[np.int_] = np.array([1, 2, 5])
    np.random.shuffle(ac_keys)
    # generate
    mu_data_hill, mu_data_other, ac_data, ac_data_wt = data.aggregate_landuses(
        network_structure,
        data_map,
        distances,
        betas,
        landuse_encodings=landuse_encodings,
        qs=qs,
        mixed_use_hill_keys=hill_keys,
        mixed_use_other_keys=non_hill_keys,
        accessibility_keys=ac_keys,
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
    # access non-weighted
    ac_1_nw = ac_data[np.where(ac_keys == 1)][0]
    ac_2_nw = ac_data[np.where(ac_keys == 2)][0]
    ac_5_nw = ac_data[np.where(ac_keys == 5)][0]
    # access weighted
    ac_1_w = ac_data_wt[np.where(ac_keys == 1)][0]
    ac_2_w = ac_data_wt[np.where(ac_keys == 2)][0]
    ac_5_w = ac_data_wt[np.where(ac_keys == 5)][0]
    # test manual metrics against all nodes
    mu_max_unique = len(landuse_classes)
    # test against various distances
    for d_idx, (dist_cutoff, beta) in enumerate(zip(distances, betas)):
        for src_idx in range(len(primal_graph)):
            reachable_data, reachable_data_dist, _tree_map = data.aggregate_to_src_idx(
                src_idx, network_structure, data_map, dist_cutoff
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
                cl = landuse_encodings[data_idx]
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
    _node_keys_dual, network_structure_dual = graphs.network_structure_from_nx(G_dual)
    data_dict_dual = mock.mock_data_dict(G_dual, random_seed=13)
    _data_uids_dual, data_map_dual = layers.data_map_from_dict(data_dict_dual)
    data_map_dual = data.assign_to_network(data_map_dual, network_structure_dual, np.float32(500))
    mock_categorical = mock.mock_categorical_data(data_map_dual.count)
    landuse_classes_dual, landuse_encodings_dual = layers.encode_categorical(mock_categorical)
    mock_matrix = np.full((len(landuse_classes_dual), len(landuse_classes_dual)), 1)

    mu_hill_dual, mu_other_dual, ac_dual, ac_wt_dual = data.aggregate_landuses(
        network_structure_dual,
        data_map_dual,
        distances,
        betas,
        landuse_encodings_dual,
        qs=qs,
        mixed_use_hill_keys=hill_keys,
        mixed_use_other_keys=non_hill_keys,
        accessibility_keys=ac_keys,
        cl_disparity_wt_matrix=mock_matrix,
        angular=True,
    )

    (mu_hill_dual_sidestep, mu_other_dual_sidestep, ac_dual_sidestep, ac_wt_dual_sidestep,) = data.aggregate_landuses(
        network_structure_dual,
        data_map_dual,
        distances,
        betas,
        landuse_encodings_dual,
        qs=qs,
        mixed_use_hill_keys=hill_keys,
        mixed_use_other_keys=non_hill_keys,
        accessibility_keys=ac_keys,
        cl_disparity_wt_matrix=mock_matrix,
        angular=False,
    )

    assert not np.allclose(mu_hill_dual, mu_hill_dual_sidestep, atol=config.ATOL, rtol=config.RTOL)
    assert not np.allclose(mu_other_dual, mu_other_dual_sidestep, atol=config.ATOL, rtol=config.RTOL)
    assert not np.allclose(ac_dual, ac_dual_sidestep, atol=config.ATOL, rtol=config.RTOL)
    assert not np.allclose(ac_wt_dual, ac_wt_dual_sidestep, atol=config.ATOL, rtol=config.RTOL)


def test_local_aggregator_numerical_components(primal_graph):
    # generate node and edge maps
    _node_keys, network_structure = graphs.network_structure_from_nx(primal_graph)
    # setup data
    data_dict = mock.mock_data_dict(primal_graph, random_seed=13)
    _data_keys, data_map = layers.data_map_from_dict(data_dict)
    data_map = data.assign_to_network(data_map, network_structure, np.float32(500))
    # for debugging
    # from cityseer.tools import plot
    # plot.plot_network_structure(_node_keys, node_data, edge_data, data_map)
    # set parameters - use a large enough distance such that simple non-weighted checks can be run for max, mean, variance
    betas: npt.NDArray[np.float32] = np.array([0.00125])
    distances = networks.distance_from_beta(betas)
    mock_numerical = mock.mock_numerical_data(len(data_dict), num_arrs=2, random_seed=0)
    # compute
    (
        stats_sum,
        _stats_sum_wt,
        stats_mean,
        _stats_mean_wt,
        stats_variance,
        _stats_variance_wt,
        stats_max,
        stats_min,
    ) = data.aggregate_stats(
        network_structure,
        data_map,
        distances,
        betas,
        numerical_arrays=mock_numerical,
        angular=False,
    )
    # non connected portions of the graph will have different stats
    # used manual data plots from test_assign_to_network() to see which nodes the data points are assigned to
    # connected graph is from 0 to 48 -> assigned data points are all except 5, 8, 17, 33, 48
    connected_nodes_idx = list(range(49))
    # and the respective data assigned to connected portion of the graph
    connected_data_idx = [i for i in range(len(data_dict)) if i not in [5, 8, 9, 17, 18, 29, 33, 38, 48]]
    # isolated node = 49 -> assigned no data points
    # isolated nodes = 50 & 51 -> assigned data points = 17, 33
    # isolated loop = 52, 53, 54, 55 -> assigned data points = 5, 8, 9, 18, 29, 38, 48
    isolated_nodes_idx = [52, 53, 54, 55]
    isolated_data_idx = [5, 8, 9, 18, 29, 38, 48]
    # numeric precision - keep fairly relaxed
    for stats_idx in range(len(mock_numerical)):
        for d_idx in range(len(distances)):
            # max
            assert np.isnan(stats_max[stats_idx, d_idx, 49])
            assert np.allclose(
                stats_max[stats_idx, d_idx, [50, 51]],
                mock_numerical[stats_idx, [17, 33]].max(),
                atol=config.ATOL,
                rtol=config.RTOL,
            )
            assert np.allclose(
                stats_max[stats_idx, d_idx, isolated_nodes_idx],
                mock_numerical[stats_idx, isolated_data_idx].max(),
                atol=config.ATOL,
                rtol=config.RTOL,
            )
            assert np.allclose(
                stats_max[stats_idx, d_idx, connected_nodes_idx],
                mock_numerical[stats_idx, connected_data_idx].max(),
                atol=config.ATOL,
                rtol=config.RTOL,
            )
            # min
            assert np.isnan(stats_min[stats_idx, d_idx, 49])
            assert np.allclose(
                stats_min[stats_idx, d_idx, [50, 51]],
                mock_numerical[stats_idx, [17, 33]].min(),
                atol=config.ATOL,
                rtol=config.RTOL,
            )
            assert np.allclose(
                stats_min[stats_idx, d_idx, isolated_nodes_idx],
                mock_numerical[stats_idx, isolated_data_idx].min(),
                atol=config.ATOL,
                rtol=config.RTOL,
            )
            assert np.allclose(
                stats_min[stats_idx, d_idx, connected_nodes_idx],
                mock_numerical[stats_idx, connected_data_idx].min(),
                atol=config.ATOL,
                rtol=config.RTOL,
            )
            # sum
            assert stats_sum[stats_idx, d_idx, 49] == 0
            assert np.allclose(
                stats_sum[stats_idx, d_idx, [50, 51]],
                mock_numerical[stats_idx, [17, 33]].sum(),
                atol=config.ATOL,
                rtol=config.RTOL,
            )
            assert np.allclose(
                stats_sum[stats_idx, d_idx, isolated_nodes_idx],
                mock_numerical[stats_idx, isolated_data_idx].sum(),
                atol=config.ATOL,
                rtol=config.RTOL,
            )
            assert np.allclose(
                stats_sum[stats_idx, d_idx, connected_nodes_idx],
                mock_numerical[stats_idx, connected_data_idx].sum(),
                atol=config.ATOL,
                rtol=config.RTOL,
            )
            # mean
            assert np.isnan(stats_mean[stats_idx, d_idx, 49])
            assert np.allclose(
                stats_mean[stats_idx, d_idx, [50, 51]],
                mock_numerical[stats_idx, [17, 33]].mean(),
                atol=config.ATOL,
                rtol=config.RTOL,
            )
            assert np.allclose(
                stats_mean[stats_idx, d_idx, isolated_nodes_idx],
                mock_numerical[stats_idx, isolated_data_idx].mean(),
                atol=config.ATOL,
                rtol=config.RTOL,
            )
            assert np.allclose(
                stats_mean[stats_idx, d_idx, connected_nodes_idx],
                mock_numerical[stats_idx, connected_data_idx].mean(),
                atol=config.ATOL,
                rtol=config.RTOL,
            )
            # variance
            assert np.isnan(stats_variance[stats_idx, d_idx, 49])
            assert np.allclose(
                stats_variance[stats_idx, d_idx, [50, 51]],
                mock_numerical[stats_idx, [17, 33]].var(),
                atol=config.ATOL,
                rtol=config.RTOL,
            )
            assert np.allclose(
                stats_variance[stats_idx, d_idx, isolated_nodes_idx],
                mock_numerical[stats_idx, isolated_data_idx].var(),
                atol=config.ATOL,
                rtol=config.RTOL,
            )
            assert np.allclose(
                stats_variance[stats_idx, d_idx, connected_nodes_idx],
                mock_numerical[stats_idx, connected_data_idx].var(),
                atol=config.ATOL,
                rtol=config.RTOL,
            )
