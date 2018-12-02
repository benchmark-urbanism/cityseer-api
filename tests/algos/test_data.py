import numpy as np
from shapely import geometry

from cityseer.algos import data
from cityseer.metrics import networks, layers
from cityseer.util import graphs, mock, plot


def test_radial_filter():
    G, pos = mock.mock_graph()
    G = graphs.networkX_simple_geoms(G)
    G = graphs.networkX_edge_defaults(G)

    # generate some data
    data_dict = mock.mock_data(G)
    D = layers.Data_Layer_From_Dict(data_dict)

    # test the filter
    src_x = G.nodes[0]['x']
    src_y = G.nodes[0]['y']
    for max_dist in [0, 200, 500, 750]:
        trim_to_full_map, full_to_trim_map = \
            data.radial_filter(src_x, src_y, D.x_arr, D.y_arr, max_dist)

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
        assert len(full_to_trim_map) == len(D.data)
        # check that all non NaN indices are reflected in the either direction
        c = 0
        for idx, n in enumerate(full_to_trim_map):
            if not np.isnan(n):
                c += 1
                assert trim_to_full_map[int(n)] == idx
        assert c == len(trim_to_full_map)

        # test that all reachable indices are, in fact, within the max distance
        for idx, val in enumerate(full_to_trim_map):
            dist = np.sqrt((D.x_arr[idx] - src_x) ** 2 + (D.y_arr[idx] - src_y) ** 2)
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
    data_dict = mock.mock_data(G)
    D = layers.Data_Layer_From_Dict(data_dict)

    # test the filter - iterating each point in data map
    for d in D.data:
        d_x = d[0]
        d_y = d[1]

        # find the closest point on the network
        min_idx, min_dist = data.nearest_idx(d_x, d_y, N.x_arr, N.y_arr, max_dist=500)

        # check that no other indices are nearer
        for idx, n in enumerate(N._nodes):
            n_x = n[0]
            n_y = n[1]
            dist = np.sqrt((d_x - n_x) ** 2 + (d_y - n_y) ** 2)
            if idx == min_idx:
                assert round(dist, 8) == round(min_dist, 8)
            else:
                assert dist > min_dist


def test_assign_to_network():
    # generate network
    G, pos = mock.mock_graph()
    G = graphs.networkX_simple_geoms(G)

    # create some dead-end scenarios
    G.remove_edge(14, 15)
    G.remove_edge(15, 28)
    G.remove_edge(16, 17)
    G.remove_edge(18, 19)

    # add a dead-end condition near the first data point
    coord_46 = (6001015, 600535)
    G.add_node(46, x=coord_46[0], y=coord_46[1])
    line_geom_a = geometry.LineString([(G.nodes[22]['x'], G.nodes[22]['y']), coord_46])
    G.add_edge(22, 46, geom=line_geom_a)

    coord_47 = (6001100, 600480)
    G.add_node(47, x=coord_47[0], y=coord_47[1])
    line_geom_b = geometry.LineString([coord_46, coord_47])
    G.add_edge(46, 47, geom=line_geom_b)

    coord_48 = (6000917, 600517)
    G.add_node(48, x=coord_48[0], y=coord_48[1])
    line_geom_c = geometry.LineString([coord_46, coord_48])
    G.add_edge(46, 48, geom=line_geom_c)

    G = graphs.networkX_edge_defaults(G)

    node_uids, node_map, edge_map = graphs.graph_maps_from_networkX(G)

    # generate data
    data_dict = mock.mock_data(G, random_seed=13)
    data_uids, data_map, class_labels = layers.data_map_from_dict(data_dict)

    # override data point 0 and 1's locations for test cases
    data_map[0][:2] = [6001000, 600350]
    data_map[1][:2] = [6001170, 600445]

    # for debugging
    plot.plot_graph_maps(node_uids, node_map, edge_map, data_map)

    data.assign_to_network(data_map, node_map, edge_map, 500)


def test_aggregate_to_src_idx():
    # generate network
    G, pos = mock.mock_graph()
    G = graphs.networkX_simple_geoms(G)
    G = graphs.networkX_edge_defaults(G)
    N = networks.Network_Layer_From_NetworkX(G)

    # generate data
    data_dict = mock.mock_data(G, random_seed=5)
    d_labels, d_map, d_classes = layers.data_map_from_dict(data_dict)
    d_map = data.assign_to_network(Layer, Network, 500)
    data_index = data.generate_index(d_map[:, 0], d_map[:, 1])

    reachable_classes_trim, reachable_classes_dist_trim, data_trim_to_full_idx_map = \
        data.aggregate_to_src_idx(0, 400, node_map, edge_map, netw_index, d_map, data_index, angular=False)

    # for debugging
    plot.plot_graph_maps(node_labels, node_map, edge_map, d_map[data_trim_to_full_idx_map])
