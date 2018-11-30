import numpy as np
import pytest

from cityseer.algos import data
from cityseer.metrics import networks, layers
from cityseer.util import graphs, mock, plot


def test_merge_sort():
    # test 2d - this version uses a second dimension storing the indices
    for n in range(1, 20):
        # create random data - stack with indices
        random_data = np.random.uniform(0, 500, n)
        indices = np.arange(len(random_data))
        stacked_data = np.vstack((random_data, indices)).T

        # check that merge sort behaves as anticipated
        sorted_data = data.tiered_sort(stacked_data, tier=0)
        for idx in range(len(sorted_data) - 1):
            # check that values increase
            assert sorted_data[idx][0] <= sorted_data[idx + 1][0]
            # check that the order of the data and the indices has remained consistent
            r = sorted_data[idx][0]
            i = sorted_data[idx][1]
            assert list(random_data).index(r) == list(indices).index(i)
        # check that all of the data has been retained
        assert len(sorted_data[:, 0]) == len(random_data)
        for d in random_data:
            assert d in sorted_data[:, 0]

        # cast back to the original order, this time using the indices to sort, i.e. tier 2
        sorted_data = data.tiered_sort(sorted_data, tier=1)
        # check that the arrays now match their original versions
        assert np.array_equal(sorted_data[:, 0], random_data)
        assert np.array_equal(sorted_data[:, 1], indices)

        # test malformed signatures
        with pytest.raises(ValueError):
            data.tiered_sort(stacked_data, tier=2)


def test_binary_search():
    for n in range(1, 10):
        # create index sorted data
        max_val = 500
        random_data = np.random.uniform(0, max_val, n)
        indices = np.arange(len(random_data))
        stacked_data = np.vstack((random_data, indices)).T
        sorted_data = data.tiered_sort(stacked_data, tier=0)
        # check some permutations
        mid_val = random_data[int(np.floor(len(random_data) / 2))]
        left_thresholds = [0, 0, 111.11, 200, mid_val]
        right_thresholds = [0, max_val, 333.3, 300, mid_val]
        test_data = sorted_data[:, 0]
        for left_min, right_max in zip(left_thresholds, right_thresholds):
            l_idx, r_idx = data.binary_search(test_data, left_min, right_max)
            # check the indices
            for idx, d in enumerate(test_data):
                if idx < l_idx:
                    assert d < left_min
                else:
                    assert d >= left_min
                if idx >= r_idx:
                    assert d > right_max
                else:
                    assert d <= right_max


def test_generate_index():
    for n in range(1, 20):
        # create some random x and y data
        random_x = np.random.uniform(0, 1000, n)
        random_y = np.random.uniform(2000, 3000, n)
        index_map = data.generate_index(random_x, random_y)

        # test arrangement of index data against independently sorted data
        x_sort = data.tiered_sort(np.vstack((random_x, np.arange(len(random_x)))).T, tier=0)
        y_sort = data.tiered_sort(np.vstack((random_y, np.arange(len(random_y)))).T, tier=0)
        for idx in range(len(index_map)):
            assert np.array_equal(x_sort[idx][:2], index_map[idx][:2])
            assert np.array_equal(y_sort[idx][:2], index_map[idx][2:])

        # test the integrity of the x and y data against the indices
        for idx, (x, p_x, y, p_y) in enumerate(index_map):
            assert random_x[int(p_x)] == x
            assert random_y[int(p_y)] == y

        # test malformed signatures
        with pytest.raises(ValueError):
            data.generate_index(random_x[:-1], random_y)
        with pytest.raises(ValueError):
            data.generate_index(random_x, random_y[:-1])


def test_distance_filter():
    G, pos = mock.mock_graph()
    G = graphs.networkX_simple_geoms(G)
    G = graphs.networkX_edge_defaults(G)
    N = networks.Network_Layer_From_NetworkX(G)

    # generate some data
    data_dict = mock.mock_data(G)
    D = layers.Data_Layer_From_Dict(data_dict)

    # test the filter
    src_x = N.x_arr[0]
    src_y = N.y_arr[1]
    for max_dist in [0, 200, 500, 750]:
        trim_to_full_map, full_to_trim_map = \
            data.distance_filter(D.index, src_x, src_y, max_dist, radial=True)

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

        # test the non radial version
        trim_to_full_map, full_to_trim_map = \
            data.distance_filter(D.index, src_x, src_y, max_dist, radial=False)
        for idx, val in enumerate(full_to_trim_map):
            if abs(D.x_arr[idx] - src_x) <= max_dist and abs(D.y_arr[idx] - src_y) <= max_dist:
                assert np.isfinite(val)
            else:
                assert np.isnan(val)


def test_nearest_idx():
    G, pos = mock.mock_graph()
    G = graphs.networkX_simple_geoms(G)
    G = graphs.networkX_edge_defaults(G)
    N = networks.Network_Layer_From_NetworkX(G)

    # generate some data
    data_dict = mock.mock_data(G)
    D = layers.Data_Layer_From_Dict(data_dict)

    # test the filter - iterating each point in data map
    for d in D.data:
        d_x = d[0]
        d_y = d[1]

        # find the closest point on the network
        min_idx, min_dist = data.nearest_idx(N.index, d_x, d_y, max_dist=500)

        # check that no other indices are nearer
        for idx, n in enumerate(N.nodes):
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
    G = graphs.networkX_edge_defaults(G)
    node_uids, node_map, edge_map = graphs.graph_maps_from_networkX(G)
    x_arr = node_map[:, 0]
    y_arr = node_map[:, 1]
    netw_index = data.generate_index(x_arr, y_arr)

    # generate data
    data_dict = mock.mock_data(G, random_seed=13)
    data_uids, data_map, class_labels = layers.data_map_from_dict(data_dict)

    # for debugging
    plot.plot_graph_maps(node_uids, node_map, edge_map, data_map)

    data.assign_to_network(data_map, node_map, edge_map, netw_index, 500)




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
