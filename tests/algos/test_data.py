import numpy as np
import pytest

from cityseer.algos import data, centrality, checks, diversity
from cityseer.metrics import networks, layers
from cityseer.util import graphs, mock


def test_radial_filter():
    G = mock.mock_graph()
    G = graphs.nX_simple_geoms(G)
    # generate some data
    data_dict = mock.mock_data_dict(G)
    D = layers.Data_Layer_From_Dict(data_dict)
    # generate some random source nodes
    rd_src_idxs = np.random.randint(0, len(data_dict), 10)
    for src_idx in rd_src_idxs:
        # test the filter
        src_x = G.nodes[src_idx]['x']
        src_y = G.nodes[src_idx]['y']
        for max_dist in [0, 200, 500, 750, np.inf]:
            data_filter = data.radial_filter(src_x, src_y, D.x_arr, D.y_arr, max_dist)
            # check that the full_to_trim map is the correct number of elements
            assert len(data_filter) == len(D._data)
            # test that all reachable indices are, in fact, within the max distance
            for i, reachable in enumerate(data_filter):
                dist = np.sqrt((D.x_arr[i] - src_x) ** 2 + (D.y_arr[i] - src_y) ** 2)
                if reachable:
                    assert dist <= max_dist
                else:
                    assert dist > max_dist


def test_nearest_idx():
    G = mock.mock_graph()
    G = graphs.nX_simple_geoms(G)
    N = networks.Network_Layer_From_nX(G, distances=[100])
    # generate some data
    data_dict = mock.mock_data_dict(G)
    D = layers.Data_Layer_From_Dict(data_dict)
    # test the filter - iterating each point in data map
    for d in D._data:
        d_x = d[0]
        d_y = d[1]
        # find the closest point on the network
        min_idx, min_dist = data.find_nearest(d_x, d_y, N.x_arr, N.y_arr, max_dist=500)
        # check that no other indices are nearer
        for i, n in enumerate(N._node_data):
            n_x = n[0]
            n_y = n[1]
            dist = np.sqrt((d_x - n_x) ** 2 + (d_y - n_y) ** 2)
            if i == min_idx:
                assert round(dist, 8) == round(min_dist, 8)
            else:
                assert dist > min_dist


def test_assign_to_network():
    # generate network
    G = mock.mock_graph()
    G = graphs.nX_simple_geoms(G)
    # create additional dead-end scenario
    G.remove_edge(14, 15)
    G.remove_edge(15, 28)
    # G = graphs.nX_auto_edge_params(G)
    G = graphs.nX_decompose(G, 50)
    node_uids, node_data, edge_data, node_edge_map = graphs.graph_maps_from_nX(G)
    # generate data
    data_dict = mock.mock_data_dict(G, random_seed=25)
    data_uids, data_map = layers.data_map_from_dict(data_dict)
    # override data point locations for test cases vis-a-vis isolated nodes and isolated edges
    data_map[18, :2] = [701200, 5719400]
    data_map[39, :2] = [700750, 5720025]
    data_map[26, :2] = [700400, 5719525]
    # 500m visually confirmed in plots
    data_map_test_1600 = data_map.copy()
    data_map_test_1600 = data.assign_to_network(data_map_test_1600,
                                                node_data,
                                                edge_data,
                                                node_edge_map,
                                                max_dist=1600)
    targets = [
        [0, 163, 162],
        [1, 42, 228],
        [2, 223, 222],
        [3, 48, 242],
        [4, 198, 199],
        [5, 223, 222],
        [6, 57, 56],
        [7, 71, 5],
        [8, 74, 75],
        [9, 91, 9],
        [10, 60, 61],
        [11, 95, 13],
        [12, 0, 58],
        [13, 97, 98],
        [14, 190, 189],
        [15, 120, 119],
        [16, 48, 242],
        [17, 2, 69],
        [18, 181, 182],
        [19, 157, 156],
        [20, 82, 83],
        [21, 2.0, np.nan],
        [22, 170, 169],
        [23, 246, 52],
        [24, 82, 83],
        [25, 87, 11],
        [26, 49.0, np.nan],
        [27, 19, 137],
        [28, 133, 134],
        [29, 242, 46],
        [30, 77, 9],
        [31, 187, 188],
        [32, 179, 180],
        [33, 94, 93],
        [34, 213, 212],
        [35, 109, 110],
        [36, 39, 215],
        [37, 157, 25],
        [38, 87, 86],
        [39, 243.0, np.nan],
        [40, 119, 120],
        [41, 145, 21],
        [42, 10, 96],
        [43, 118, 117],
        [44, 81, 5],
        [45, 11, 87],
        [46, 99, 98],
        [47, 137, 19],
        [48, 14.0, np.nan],
        [49, 105, 104]
    ]
    for i in range(len(data_map_test_1600)):
        assert data_map_test_1600[i, 2] == targets[i][1]
        np.testing.assert_array_almost_equal(data_map_test_1600[i, 3], targets[i][2], decimal=3)
    # for debugging
    # from cityseer.util import plot
    # plot.plot_graph_maps(node_uids, node_data, edge_data, data_map_test_1600)
    # max distance of 0 should return all NaN
    data_map_test_0 = data_map.copy()
    data_map_test_0 = data.assign_to_network(data_map_test_0, node_data, edge_data, node_edge_map, max_dist=0)
    assert np.all(np.isnan(data_map_test_0[:, 2]))
    assert np.all(np.isnan(data_map_test_0[:, 3]))
    # max distance of 2000 should return no NaN for nearest
    # there will be some NaN for next nearest
    data_map_test_2000 = data_map.copy()
    data_map_test_2000 = data.assign_to_network(data_map_test_2000, node_data, edge_data, node_edge_map, max_dist=2000)
    assert not np.any(np.isnan(data_map_test_2000[:, 2]))


def test_aggregate_to_src_idx():
    # generate network
    G = mock.mock_graph()
    G = graphs.nX_simple_geoms(G)
    node_uids, node_data, edge_data, node_edge_map = graphs.graph_maps_from_nX(G)
    # generate data
    data_dict = mock.mock_data_dict(G, random_seed=13)
    data_uids, data_map = layers.data_map_from_dict(data_dict)
    for max_dist in [400, 750]:
        # in this case, use same assignment max dist as search max dist
        data_map_temp = data_map.copy()
        data_map_temp = data.assign_to_network(data_map_temp,
                                               node_data,
                                               edge_data,
                                               node_edge_map,
                                               max_dist=max_dist)
        for angular in [True, False]:
            for netw_src_idx in range(len(node_data)):
                # aggregate to src...
                reachable_data, reachable_data_dist, tree_preds = data.aggregate_to_src_idx(netw_src_idx,
                                                                                            node_data,
                                                                                            edge_data,
                                                                                            node_edge_map,
                                                                                            data_map_temp,
                                                                                            max_dist,
                                                                                            angular=angular)
                # for debugging
                # from cityseer.util import plot
                # plot.plot_graph_maps(node_uids, node_data, edge_data, data_map)
                # compare to manual checks on distances:
                netw_x_arr = node_data[:, 0]
                netw_y_arr = node_data[:, 1]
                netw_src_x = netw_x_arr[netw_src_idx]
                netw_src_y = netw_y_arr[netw_src_idx]
                data_x_arr = data_map_temp[:, 0]
                data_y_arr = data_map_temp[:, 1]
                # get the network distances
                tree_map, tree_edges = centrality.shortest_path_tree(node_data,
                                                                     edge_data,
                                                                     node_edge_map,
                                                                     netw_src_idx,
                                                                     max_dist=max_dist,
                                                                     angular=angular)
                tree_dists = tree_map[:, 2]
                # generate a data filter
                data_filter = data.radial_filter(netw_src_x, netw_src_y, data_x_arr, data_y_arr, max_dist)
                # verify distances vs. the max
                for d_idx in range(len(data_map_temp)):
                    # check that all filtered items are beyond the radial cutoff distance
                    if not data_filter[d_idx]:
                        r_dist = np.hypot(netw_src_x - data_x_arr[d_idx], netw_src_y - data_y_arr[d_idx])
                        assert r_dist > max_dist
                    # otherwise they should be within the cutoff distance
                    else:
                        r_dist = np.hypot(netw_src_x - data_x_arr[d_idx], netw_src_y - data_y_arr[d_idx])
                        assert r_dist <= max_dist
                    # check the integrity of the distances and classes
                    reachable = reachable_data[d_idx]
                    reachable_dist = reachable_data_dist[d_idx]
                    # get the distance via the nearest assigned index
                    nearest_dist = np.inf
                    # if a nearest node has been assigned
                    if np.isfinite(data_map_temp[d_idx, 2]):
                        # get the index for the assigned network node
                        netw_idx = int(data_map_temp[d_idx, 2])
                        # if this node is within the cutoff distance:
                        if tree_dists[netw_idx] < max_dist:
                            # get the distances from the data point to the assigned network node
                            d_d = np.hypot(data_x_arr[d_idx] - netw_x_arr[netw_idx],
                                           data_y_arr[d_idx] - netw_y_arr[netw_idx])
                            # and add it to the network distance path from the source to the assigned node
                            n_d = tree_dists[netw_idx]
                            nearest_dist = d_d + n_d
                    # also get the distance via the next nearest assigned index
                    next_nearest_dist = np.inf
                    # if a nearest node has been assigned
                    if np.isfinite(data_map_temp[d_idx, 3]):
                        # get the index for the assigned network node
                        netw_idx = int(data_map_temp[d_idx, 3])
                        # if this node is within the radial cutoff distance:
                        if tree_dists[netw_idx] < max_dist:
                            # get the distances from the data point to the assigned network node
                            d_d = np.hypot(data_x_arr[d_idx] - netw_x_arr[netw_idx],
                                           data_y_arr[d_idx] - netw_y_arr[netw_idx])
                            # and add it to the network distance path from the source to the assigned node
                            n_d = tree_dists[netw_idx]
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


def test_local_aggregator_signatures():
    # load the test graph
    G = mock.mock_graph()
    G = graphs.nX_simple_geoms(G)
    node_uids, node_data, edge_data, node_edge_map = graphs.graph_maps_from_nX(G)  # generate node and edge maps
    # setup data
    data_dict = mock.mock_data_dict(G, random_seed=13)
    data_uids, data_map = layers.data_map_from_dict(data_dict)
    data_map = data.assign_to_network(data_map, node_data, edge_data, node_edge_map, 500)
    # set parameters
    betas = np.array([-0.02, -0.01, -0.005, -0.0025])
    distances = networks.distance_from_beta(betas)
    qs = np.array([0, 1, 2])
    mock_categorical = mock.mock_categorical_data(len(data_map))
    landuse_classes, landuse_encodings = layers.encode_categorical(mock_categorical)
    # check that empty land_use encodings are caught
    with pytest.raises(ValueError):
        data.local_aggregator(node_data,
                              edge_data,
                              node_edge_map,
                              data_map,
                              distances,
                              betas,
                              mixed_use_hill_keys=np.array([0]))
    # check that unequal land_use encodings vs data map lengths are caught
    with pytest.raises(ValueError):
        data.local_aggregator(node_data,
                              edge_data,
                              node_edge_map,
                              data_map,
                              distances,
                              betas,
                              landuse_encodings=landuse_encodings[:-1],
                              mixed_use_other_keys=np.array([0]))
    # check that no provided metrics flags
    with pytest.raises(ValueError):
        data.local_aggregator(node_data,
                              edge_data,
                              node_edge_map,
                              data_map,
                              distances,
                              betas,
                              landuse_encodings=landuse_encodings)
    # check that missing qs flags
    with pytest.raises(ValueError):
        data.local_aggregator(node_data,
                              edge_data,
                              node_edge_map,
                              data_map,
                              distances,
                              betas,
                              mixed_use_hill_keys=np.array([0]),
                              landuse_encodings=landuse_encodings)
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
        ([1], [1], [1, 1])]:
        with pytest.raises(ValueError):
            data.local_aggregator(node_data,
                                  edge_data,
                                  node_edge_map,
                                  data_map,
                                  distances,
                                  betas,
                                  landuse_encodings,
                                  qs=qs,
                                  mixed_use_hill_keys=np.array(mu_h_key),
                                  mixed_use_other_keys=np.array(mu_o_key),
                                  accessibility_keys=np.array(ac_key))
    for h_key, o_key in (([3], []), ([], [2])):
        # check that missing matrix is caught for disparity weighted indices
        with pytest.raises(ValueError):
            data.local_aggregator(node_data,
                                  edge_data,
                                  node_edge_map,
                                  data_map,
                                  distances,
                                  betas,
                                  landuse_encodings=landuse_encodings,
                                  qs=qs,
                                  mixed_use_hill_keys=np.array(h_key),
                                  mixed_use_other_keys=np.array(o_key))
        # check that non-square disparity matrix is caught
        mock_matrix = np.full((len(landuse_classes), len(landuse_classes)), 1)
        with pytest.raises(ValueError):
            data.local_aggregator(node_data,
                                  edge_data,
                                  node_edge_map,
                                  data_map,
                                  distances,
                                  betas,
                                  landuse_encodings=landuse_encodings,
                                  qs=qs,
                                  mixed_use_hill_keys=np.array(h_key),
                                  mixed_use_other_keys=np.array(o_key),
                                  cl_disparity_wt_matrix=mock_matrix[:-1])


def test_local_aggregator_categorical_components():
    # load the test graph
    G = mock.mock_graph()
    G = graphs.nX_simple_geoms(G)
    node_uids, node_data, edge_data, node_edge_map, = graphs.graph_maps_from_nX(G)  # generate node and edge maps
    # setup data
    data_dict = mock.mock_data_dict(G, random_seed=13)
    data_uids, data_map = layers.data_map_from_dict(data_dict)
    data_map = data.assign_to_network(data_map, node_data, edge_data, node_edge_map, 500)
    # set parameters
    betas = np.array([-0.02, -0.01, -0.005, -0.0025])
    distances = networks.distance_from_beta(betas)
    qs = np.array([0, 1, 2])
    mock_categorical = mock.mock_categorical_data(len(data_map))
    landuse_classes, landuse_encodings = layers.encode_categorical(mock_categorical)
    mock_matrix = np.full((len(landuse_classes), len(landuse_classes)), 1)
    # set the keys - add shuffling to be sure various orders work
    hill_keys = np.arange(4)
    np.random.shuffle(hill_keys)
    non_hill_keys = np.arange(3)
    np.random.shuffle(non_hill_keys)
    ac_keys = np.array([1, 2, 5])
    np.random.shuffle(ac_keys)
    # generate
    mu_data_hill, mu_data_other, ac_data, ac_data_wt, stats_sum, stats_sum_wt, stats_mean, stats_mean_wt, \
    stats_variance, stats_variance_wt, stats_max, stats_min = \
        data.local_aggregator(node_data,
                              edge_data,
                              node_edge_map,
                              data_map,
                              distances,
                              betas,
                              landuse_encodings=landuse_encodings,
                              qs=qs,
                              mixed_use_hill_keys=hill_keys,
                              mixed_use_other_keys=non_hill_keys,
                              accessibility_keys=ac_keys,
                              cl_disparity_wt_matrix=mock_matrix,
                              angular=False)
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
    for d_idx in range(len(distances)):
        dist_cutoff = distances[d_idx]
        beta = betas[d_idx]
        for src_idx in range(len(G)):
            reachable_data, reachable_data_dist, tree_preds = data.aggregate_to_src_idx(src_idx,
                                                                                        node_data,
                                                                                        edge_data,
                                                                                        node_edge_map,
                                                                                        data_map,
                                                                                        dist_cutoff)
            # counts of each class type (array length per max unique classes - not just those within max distance)
            cl_counts = np.full(mu_max_unique, 0)
            # nearest of each class type (likewise)
            cl_nearest = np.full(mu_max_unique, np.inf)
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
                    a_1_w += np.exp(beta * data_dist)
                elif cl == 2:
                    a_2_nw += 1
                    a_2_w += np.exp(beta * data_dist)
                elif cl == 5:
                    a_5_nw += 1
                    a_5_w += np.exp(beta * data_dist)
            # assertions
            assert ac_1_nw[d_idx, src_idx] == a_1_nw
            assert ac_2_nw[d_idx, src_idx] == a_2_nw
            assert ac_5_nw[d_idx, src_idx] == a_5_nw

            assert ac_1_w[d_idx, src_idx] == a_1_w
            assert ac_2_w[d_idx, src_idx] == a_2_w
            assert ac_5_w[d_idx, src_idx] == a_5_w

            assert hill[0, d_idx, src_idx] == diversity.hill_diversity(cl_counts, 0)
            assert hill[1, d_idx, src_idx] == diversity.hill_diversity(cl_counts, 1)
            assert hill[2, d_idx, src_idx] == diversity.hill_diversity(cl_counts, 2)

            assert hill_branch_wt[0, d_idx, src_idx] == \
                   diversity.hill_diversity_branch_distance_wt(cl_counts, cl_nearest, 0, beta)
            assert hill_branch_wt[1, d_idx, src_idx] == \
                   diversity.hill_diversity_branch_distance_wt(cl_counts, cl_nearest, 1, beta)
            assert hill_branch_wt[2, d_idx, src_idx] == \
                   diversity.hill_diversity_branch_distance_wt(cl_counts, cl_nearest, 2, beta)

            assert hill_pw_wt[0, d_idx, src_idx] == \
                   diversity.hill_diversity_pairwise_distance_wt(cl_counts, cl_nearest, 0, beta)
            assert hill_pw_wt[1, d_idx, src_idx] == \
                   diversity.hill_diversity_pairwise_distance_wt(cl_counts, cl_nearest, 1, beta)
            assert hill_pw_wt[2, d_idx, src_idx] == \
                   diversity.hill_diversity_pairwise_distance_wt(cl_counts, cl_nearest, 2, beta)

            assert hill_disp_wt[0, d_idx, src_idx] == \
                   diversity.hill_diversity_pairwise_matrix_wt(cl_counts, mock_matrix, 0)
            assert hill_disp_wt[1, d_idx, src_idx] == \
                   diversity.hill_diversity_pairwise_matrix_wt(cl_counts, mock_matrix, 1)
            assert hill_disp_wt[2, d_idx, src_idx] == \
                   diversity.hill_diversity_pairwise_matrix_wt(cl_counts, mock_matrix, 2)

            assert shannon[d_idx, src_idx] == diversity.shannon_diversity(cl_counts)
            assert gini[d_idx, src_idx] == diversity.gini_simpson_diversity(cl_counts)
            assert raos[d_idx, src_idx] == diversity.raos_quadratic_diversity(cl_counts, mock_matrix)

    # check that angular is passed-through
    # actual angular tests happen in test_shortest_path_tree()
    # here the emphasis is simply on checking that the angular instruction gets chained through

    # setup dual data
    G_dual = graphs.nX_to_dual(G)
    node_labels_dual, node_data_dual, edge_data_dual, node_edge_map_dual = graphs.graph_maps_from_nX(G_dual)
    data_dict_dual = mock.mock_data_dict(G_dual, random_seed=13)
    data_uids_dual, data_map_dual = layers.data_map_from_dict(data_dict_dual)
    data_map_dual = data.assign_to_network(data_map_dual, node_data_dual, edge_data_dual, node_edge_map_dual, 500)
    mock_categorical = mock.mock_categorical_data(len(data_map_dual))
    landuse_classes_dual, landuse_encodings_dual = layers.encode_categorical(mock_categorical)
    mock_matrix = np.full((len(landuse_classes_dual), len(landuse_classes_dual)), 1)

    mu_hill_dual, mu_other_dual, ac_dual, ac_wt_dual, stats_sum, stats_sum_wt, stats_mean, stats_mean_wt, \
    stats_variance, stats_variance_wt, stats_max, stats_min = \
        data.local_aggregator(node_data_dual,
                              edge_data_dual,
                              node_edge_map_dual,
                              data_map_dual,
                              distances,
                              betas,
                              landuse_encodings_dual,
                              qs=qs,
                              mixed_use_hill_keys=hill_keys,
                              mixed_use_other_keys=non_hill_keys,
                              accessibility_keys=ac_keys,
                              cl_disparity_wt_matrix=mock_matrix,
                              angular=True)

    mu_hill_dual_sidestep, mu_other_dual_sidestep, ac_dual_sidestep, ac_wt_dual_sidestep, \
    stats_sum, stats_sum_wt, stats_mean, stats_mean_wt, stats_variance, stats_variance_wt, stats_max, stats_min = \
        data.local_aggregator(node_data_dual,
                              edge_data_dual,
                              node_edge_map_dual,
                              data_map_dual,
                              distances,
                              betas,
                              landuse_encodings_dual,
                              qs=qs,
                              mixed_use_hill_keys=hill_keys,
                              mixed_use_other_keys=non_hill_keys,
                              accessibility_keys=ac_keys,
                              cl_disparity_wt_matrix=mock_matrix,
                              angular=False)

    assert not np.allclose(mu_hill_dual, mu_hill_dual_sidestep)
    assert not np.allclose(mu_other_dual, mu_other_dual_sidestep)
    assert not np.allclose(ac_dual, ac_dual_sidestep)
    assert not np.allclose(ac_wt_dual, ac_wt_dual_sidestep)


def test_local_aggregator_numerical_components():
    # load the test graph
    G = mock.mock_graph()
    G = graphs.nX_simple_geoms(G)
    node_uids, node_data, edge_data, node_edge_map = graphs.graph_maps_from_nX(G)  # generate node and edge maps

    # setup data
    data_dict = mock.mock_data_dict(G, random_seed=13)
    data_uids, data_map = layers.data_map_from_dict(data_dict)
    data_map = data.assign_to_network(data_map, node_data, edge_data, node_edge_map, 500)
    # for debugging
    # from cityseer.util import plot
    # plot.plot_graph_maps(node_uids, node_data, edge_data, data_map)

    # set parameters - use a large enough distance such that simple non-weighted checks can be run for max, mean, variance
    betas = np.array([-0.00125])
    distances = networks.distance_from_beta(betas)
    mock_numerical = mock.mock_numerical_data(len(data_dict), num_arrs=2, random_seed=0)

    mu_data_hill, mu_data_other, ac_data, ac_data_wt, stats_sum, stats_sum_wt, stats_mean, stats_mean_wt, \
    stats_variance, stats_variance_wt, stats_max, stats_min = \
        data.local_aggregator(node_data,
                              edge_data,
                              node_edge_map,
                              data_map,
                              distances,
                              betas,
                              numerical_arrays=mock_numerical,
                              angular=False)

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
    for stats_idx in range(len(mock_numerical)):
        for d_idx in range(len(distances)):
            # max
            assert np.isnan(stats_max[stats_idx, d_idx, 49])
            np.testing.assert_array_almost_equal(stats_max[stats_idx, d_idx, [50, 51]],
                                                 mock_numerical[stats_idx, [17, 33]].max(), decimal=3)
            np.testing.assert_array_almost_equal(stats_max[stats_idx, d_idx, isolated_nodes_idx],
                                                 mock_numerical[stats_idx, isolated_data_idx].max(), decimal=3)
            np.testing.assert_array_almost_equal(stats_max[stats_idx, d_idx, connected_nodes_idx],
                                                 mock_numerical[stats_idx, connected_data_idx].max(), decimal=3)
            # min
            assert np.isnan(stats_min[stats_idx, d_idx, 49])
            np.testing.assert_array_almost_equal(stats_min[stats_idx, d_idx, [50, 51]],
                                                 mock_numerical[stats_idx, [17, 33]].min(), decimal=3)
            np.testing.assert_array_almost_equal(stats_min[stats_idx, d_idx, isolated_nodes_idx],
                                                 mock_numerical[stats_idx, isolated_data_idx].min(), decimal=3)
            np.testing.assert_array_almost_equal(stats_min[stats_idx, d_idx, connected_nodes_idx],
                                                 mock_numerical[stats_idx, connected_data_idx].min(), decimal=3)
            # sum
            assert np.isnan(stats_sum[stats_idx, d_idx, 49])
            np.testing.assert_array_almost_equal(stats_sum[stats_idx, d_idx, [50, 51]],
                                                 mock_numerical[stats_idx, [17, 33]].sum(), decimal=3)
            np.testing.assert_array_almost_equal(stats_sum[stats_idx, d_idx, isolated_nodes_idx],
                                                 mock_numerical[stats_idx, isolated_data_idx].sum(), decimal=3)
            np.testing.assert_array_almost_equal(stats_sum[stats_idx, d_idx, connected_nodes_idx],
                                                 mock_numerical[stats_idx, connected_data_idx].sum(), decimal=3)
            # mean
            assert np.isnan(stats_mean[stats_idx, d_idx, 49])
            np.testing.assert_array_almost_equal(stats_mean[stats_idx, d_idx, [50, 51]],
                                                 mock_numerical[stats_idx, [17, 33]].mean(), decimal=3)
            np.testing.assert_array_almost_equal(stats_mean[stats_idx, d_idx, isolated_nodes_idx],
                                                 mock_numerical[stats_idx, isolated_data_idx].mean(), decimal=3)
            np.testing.assert_array_almost_equal(stats_mean[stats_idx, d_idx, connected_nodes_idx],
                                                 mock_numerical[stats_idx, connected_data_idx].mean(), decimal=3)
            # variance
            assert np.isnan(stats_variance[stats_idx, d_idx, 49])
            np.testing.assert_array_almost_equal(stats_variance[stats_idx, d_idx, [50, 51]],
                                                 mock_numerical[stats_idx, [17, 33]].var(), decimal=3)
            np.testing.assert_array_almost_equal(stats_variance[stats_idx, d_idx, isolated_nodes_idx],
                                                 mock_numerical[stats_idx, isolated_data_idx].var(), decimal=3)
            np.testing.assert_array_almost_equal(stats_variance[stats_idx, d_idx, connected_nodes_idx],
                                                 mock_numerical[stats_idx, connected_data_idx].var(), decimal=3)


# TODO: add more extensive tests if / when publishing this function
def test_model_singly_constrained():
    import networkx as nx

    G = nx.Graph()
    G.add_node(0, x=0, y=0)
    G.add_node(1, x=100, y=0)
    G.add_node(2, x=200, y=0)
    G.add_node(3, x=300, y=0)
    G.add_node(4, x=400, y=0)
    G.add_edge(0, 1)
    G.add_edge(1, 2)
    G.add_edge(2, 3)
    G.add_edge(3, 4)
    G = graphs.nX_simple_geoms(G)
    node_uids, node_data, edge_data, node_edge_map = graphs.graph_maps_from_nX(G)  # generate node and edge maps

    landuses = {}
    pop = {}
    counter = 0
    for n, d in G.nodes(data=True):
        landuses[counter] = {
            'x': d['x'],
            'y': d['y']
        }
        pop[counter] = {
            'x': d['x'],
            'y': d['y']
        }
        counter += 1

    landuse_uids, landuse_map = layers.data_map_from_dict(landuses)
    landuse_map = data.assign_to_network(landuse_map, node_data, edge_data, node_edge_map, 500)

    pop_uids, pop_map = layers.data_map_from_dict(pop)
    pop_map = data.assign_to_network(pop_map, node_data, edge_data, node_edge_map, 500)

    betas = np.array([-0.00125])
    distances = networks.distance_from_beta(betas)

    pop = np.array([3, 3, 3, 3, 3])
    lu = np.array([0, 0, 0, 0, 1])

    j_assigned, netw_flows = data.singly_constrained(node_data,
                                                     edge_data,
                                                     node_edge_map,
                                                     distances,
                                                     betas,
                                                     pop_map,
                                                     landuse_map,
                                                     pop,
                                                     lu)

    assert np.sum(j_assigned) == np.sum(pop)
    np.testing.assert_array_almost_equal(j_assigned, [[0, 0, 0, 0, 15]], decimal=3)
    np.testing.assert_array_almost_equal(netw_flows, [[3, 6, 9, 12, 15]], decimal=3)
