import numpy as np

from cityseer.algos import data, centrality, checks
from cityseer.metrics import networks, layers
from cityseer.util import graphs, mock


def test_radial_filter():
    G, pos = mock.mock_graph()
    G = graphs.networkX_simple_geoms(G)
    G = graphs.networkX_edge_defaults(G)

    # generate some data
    data_dict = mock.mock_landuse_data(G)
    D = layers.Landuse_Layer_From_Dict(data_dict)

    # test the filter
    src_x = G.nodes[0]['x']
    src_y = G.nodes[0]['y']
    for max_dist in [0, 200, 500, 750]:
        trim_to_full_map, full_to_trim_map = \
            data.radial_filter(src_x, src_y, D.x_arr, D.y_arr, max_dist)

        checks.check_trim_maps(trim_to_full_map, full_to_trim_map)

        # plots for debugging
        # override the d_map's data class with the results of the filtering
        # NOTE -> if all are on, then matplotlib will plot all the same dark color
        # d_map = data.assign_to_network(d_map, n_map, 2000)
        # d_map[:,3] = 0
        # on_idx = np.where(np.isfinite(full_to_trim_map))
        # d_map[on_idx, 3] = 1
        # geom = None
        # if max_dist:
        #    geom = geometry.Point(src_x, src_y).buffer(max_dist)
        # plot.plot_graph_maps(n_labels, n_map, e_map, d_map=d_map, poly=geom)

        # check that the full_to_trim map is the correct number of elements
        assert len(full_to_trim_map) == len(D._data)
        # check that all non NaN indices are reflected in the either direction
        c = 0
        for i, n in enumerate(full_to_trim_map):
            if not np.isnan(n):
                c += 1
                assert trim_to_full_map[int(n)] == i
        assert c == len(trim_to_full_map)

        # test that all reachable indices are, in fact, within the max distance
        for i, val in enumerate(full_to_trim_map):
            dist = np.sqrt((D.x_arr[i] - src_x) ** 2 + (D.y_arr[i] - src_y) ** 2)
            if np.isfinite(val):
                assert dist <= max_dist
            else:
                assert dist > max_dist


def test_nearest_idx():
    G, pos = mock.mock_graph()
    G = graphs.networkX_simple_geoms(G)
    G = graphs.networkX_edge_defaults(G)
    N = networks.Network_Layer_From_NetworkX(G, distances=[100])

    # generate some data
    data_dict = mock.mock_landuse_data(G)
    D = layers.Landuse_Layer_From_Dict(data_dict)

    # test the filter - iterating each point in data map
    for d in D._data:
        d_x = d[0]
        d_y = d[1]

        # find the closest point on the network
        min_idx, min_dist = data.find_nearest(d_x, d_y, N.x_arr, N.y_arr, max_dist=500)

        # check that no other indices are nearer
        for i, n in enumerate(N._nodes):
            n_x = n[0]
            n_y = n[1]
            dist = np.sqrt((d_x - n_x) ** 2 + (d_y - n_y) ** 2)
            if i == min_idx:
                assert round(dist, 8) == round(min_dist, 8)
            else:
                assert dist > min_dist


def test_assign_to_network():
    # generate network
    G, pos = mock.mock_graph()
    G = graphs.networkX_simple_geoms(G)

    # create additional dead-end scenario
    G.remove_edge(14, 15)
    G.remove_edge(15, 28)

    G = graphs.networkX_edge_defaults(G)
    node_uids, node_map, edge_map = graphs.graph_maps_from_networkX(G)

    # generate data
    data_dict = mock.mock_landuse_data(G, random_seed=13)
    data_uids, data_map, class_labels = layers.categorical_data_map_from_dict(data_dict)

    # override data point 0 and 1's locations for test cases
    data_map[0][:2] = [6001000, 600350]
    data_map[1][:2] = [6001170, 600445]

    # 500m visually confirmed in plots
    data_map_test_500 = data_map.copy()
    data_map_test_500 = data.assign_to_network(data_map_test_500, node_map, edge_map, 500)
    targets = [[0, 45, 30],
               [1, 30, 45],
               [2, 23, 21],
               [3, 32, 35],
               [4, 25, 26],
               [5, 3, 4],
               [6, 50, np.nan],
               [7, 13, 9],
               [8, 30, 45],
               [9, 43, 10],
               [10, 43, 44],
               [11, 24, 20],
               [12, 19, 22],
               [13, 51, np.nan],
               [14, 14, 11],
               [15, 16, 19],
               [16, 49, np.nan],
               [17, 10, 43],
               [18, 43, 42],
               [19, 19, 16],
               [20, 43, 10],
               [21, 15, np.nan],
               [22, 16, 0],
               [23, 49, np.nan],
               [24, 31, 0],
               [25, 29, 28],
               [26, 45, 30],
               [27, 36, 41],
               [28, 41, 40],
               [29, 16, 19],
               [30, 10, 43],
               [31, 49, np.nan],
               [32, 47, np.nan],
               [33, 36, 37],
               [34, 14, np.nan],
               [35, 14, np.nan],
               [36, 10, 43],
               [37, 30, 45],
               [38, 29, 25],
               [39, 43, 10],
               [40, 45, 30],
               [41, 5, 2],
               [42, 43, 10],
               [43, 10, 43],
               [44, 20, 17],
               [45, 30, 45],
               [46, 43, 10],
               [47, 27, 22],
               [48, 2, 5],
               [49, 29, 30]]
    for i in range(len(data_map_test_500)):
        assert data_map_test_500[i][4] == targets[i][1]
        assert np.allclose(data_map_test_500[i][5], targets[i][2], equal_nan=True)

    # for debugging
    # from cityseer.util import plot
    # plot.plot_graph_maps(node_uids, node_map, edge_map, data_map_test_500)

    # max distance of 0 should return all NaN
    data_map_test_0 = data_map.copy()
    data_map_test_0 = data.assign_to_network(data_map_test_0, node_map, edge_map, max_dist=0)
    assert np.all(np.isnan(data_map_test_0[:, 4]))
    assert np.all(np.isnan(data_map_test_0[:, 5]))

    # max distance of 2000 should return no NaN for nearest
    # there will be some NaN for next nearest
    data_map_test_2000 = data_map.copy()
    data_map_test_2000 = data.assign_to_network(data_map_test_2000, node_map, edge_map, max_dist=2000)
    assert not np.any(np.isnan(data_map_test_2000[:, 4]))


def test_aggregate_to_src_idx():
    # generate network
    G, pos = mock.mock_graph()
    G = graphs.networkX_simple_geoms(G)
    G = graphs.networkX_edge_defaults(G)
    node_uids, node_map, edge_map = graphs.graph_maps_from_networkX(G)

    # generate data
    data_dict = mock.mock_landuse_data(G, random_seed=13)
    data_uids, data_map, class_labels = layers.categorical_data_map_from_dict(data_dict)

    for max_dist in [400, 750]:

        # in this case, use same assignment max dist as search max dist
        data_map_temp = data_map.copy()
        data_map_temp = data.assign_to_network(data_map_temp, node_map, edge_map, max_dist=max_dist)

        for angular in [True, False]:
            for src_idx in range(len(node_map)):

                # aggregate to src...
                reachable_classes_trim, reachable_classes_dist_trim, data_trim_to_full = \
                    data.aggregate_to_src_idx(src_idx, node_map, edge_map, data_map_temp, max_dist, angular=angular)

                # now compare to manual checks on distances:

                # for debugging
                # plot.plot_graph_maps(node_uids, node_map, edge_map, data_map)

                # generate data for testing against
                netw_x_arr = node_map[:, 0]
                netw_y_arr = node_map[:, 1]
                src_x = netw_x_arr[src_idx]
                src_y = netw_y_arr[src_idx]
                data_x_arr = data_map_temp[:, 0]
                data_y_arr = data_map_temp[:, 1]

                # setup a full to trim data map
                data_full_to_trim = np.full(len(data_map), np.nan)
                for d_trim_idx in range(len(data_trim_to_full)):
                    if not np.isnan(data_trim_to_full[d_trim_idx]):
                        full_idx = int(data_trim_to_full[d_trim_idx])
                        data_full_to_trim[full_idx] = d_trim_idx

                checks.check_trim_maps(data_trim_to_full, data_full_to_trim)

                # get the network trim maps
                netw_trim_to_full, netw_full_to_trim = data.radial_filter(src_x,
                                                                          src_y,
                                                                          netw_x_arr,
                                                                          netw_y_arr,
                                                                          max_dist)

                # get the network distances
                map_impedance_trim, map_distance_trim, map_pred_trim, _cycles_trim = \
                    centrality.shortest_path_tree(node_map,
                                                  edge_map,
                                                  src_idx,
                                                  netw_trim_to_full,
                                                  netw_full_to_trim,
                                                  max_dist=max_dist,
                                                  angular=angular)

                # verify distances vs. the max
                for d_full_idx in range(len(data_map_temp)):

                    # all elements within the radial cutoff should be in the trim map
                    if np.isnan(data_full_to_trim[d_full_idx]):
                        r_dist = np.hypot(src_x - data_x_arr[d_full_idx], src_y - data_y_arr[d_full_idx])
                        assert r_dist > max_dist

                    else:

                        # all element in the trim map, must be within the radial cutoff distance
                        r_dist = np.hypot(src_x - data_x_arr[d_full_idx], src_y - data_y_arr[d_full_idx])
                        assert r_dist <= max_dist

                        # get the trim index, and check the integrity of the distances and classes
                        d_trim_idx = int(data_full_to_trim[d_full_idx])
                        cl = reachable_classes_trim[d_trim_idx]
                        dist = reachable_classes_dist_trim[d_trim_idx]

                        # get the distance via the nearest assigned index
                        nearest_dist = np.inf
                        # if a nearest node has been assigned
                        if np.isfinite(data_map_temp[d_full_idx][4]):
                            # get the full index for the assigned network node
                            netw_full_idx = int(data_map_temp[d_full_idx][4])
                            # if this node is within the radial cutoff distance:
                            if np.isfinite(netw_full_to_trim[netw_full_idx]):
                                # get the network node's trim index
                                netw_trim_idx = int(netw_full_to_trim[netw_full_idx])
                                # get the distances from the data point to the assigned network node
                                d_d = np.hypot(data_x_arr[d_full_idx] - netw_x_arr[netw_full_idx],
                                               data_y_arr[d_full_idx] - netw_y_arr[netw_full_idx])
                                # and add it to the network distance path from the source to the assigned node
                                n_d = map_distance_trim[netw_trim_idx]
                                nearest_dist = d_d + n_d

                        # also get the distance via the next nearest assigned index
                        next_nearest_dist = np.inf
                        # if a nearest node has been assigned
                        if np.isfinite(data_map_temp[d_full_idx][5]):
                            # get the full index for the assigned network node
                            netw_full_idx = int(data_map_temp[d_full_idx][5])
                            # if this node is within the radial cutoff distance:
                            if np.isfinite(netw_full_to_trim[netw_full_idx]):
                                # get the network node's trim index
                                netw_trim_idx = int(netw_full_to_trim[netw_full_idx])
                                # get the distances from the data point to the assigned network node
                                d_d = np.hypot(data_x_arr[d_full_idx] - netw_x_arr[netw_full_idx],
                                               data_y_arr[d_full_idx] - netw_y_arr[netw_full_idx])
                                # and add it to the network distance path from the source to the assigned node
                                n_d = map_distance_trim[netw_trim_idx]
                                next_nearest_dist = d_d + n_d

                        # now check distance integrity
                        if np.isinf(dist):
                            assert nearest_dist > max_dist and next_nearest_dist > max_dist
                        else:
                            assert dist <= max_dist
                            if nearest_dist < next_nearest_dist:
                                assert dist == nearest_dist
                            else:
                                assert dist == next_nearest_dist

                        # check the class integrity
                        if np.isfinite(dist):
                            assert cl == data_map_temp[d_full_idx][3]
                        else:
                            assert np.isnan(cl)
