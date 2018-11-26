import pytest
import numpy as np
from cityseer.util import graphs, layers, mock, plot
from cityseer.algos import data


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
        assert len(sorted_data[:,0]) == len(random_data)
        for d in random_data:
            assert d in sorted_data[:,0]

        # cast back to the original order, this time using the indices to sort, i.e. tier 2
        sorted_data = data.tiered_sort(sorted_data, tier=1)
        # check that the arrays now match their original versions
        assert np.array_equal(sorted_data[:,0], random_data)
        assert np.array_equal(sorted_data[:,1], indices)

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
        mid_val = random_data[int(np.floor(len(random_data)/2))]
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


def test_crow_flies():

    G, pos = mock.mock_graph()
    G = graphs.networkX_simple_geoms(G)
    G = graphs.networkX_edge_defaults(G)
    n_labels, n_map, e_map = graphs.graph_maps_from_networkX(G)

    max_dist = 200
    x_arr = n_map[:,0]
    y_arr = n_map[:,1]

    # generate trim and full index maps
    src_x = x_arr[0]
    src_y = y_arr[0]
    trim_to_full_idx_map, full_to_trim_idx_map = data.crow_flies(src_x, src_y, x_arr, y_arr, max_dist)

    # debugging
    # plot.plot_networkX_graphs(primal=G)

    # manually confirmed for 200m max distance:
    assert np.array_equal(trim_to_full_idx_map, np.array([0, 1, 16, 31]))
    # check that the full to trim is still the same length
    assert len(full_to_trim_idx_map) == G.number_of_nodes()
    # check that all non NaN full_to_trim_idx_map indices are reflected in the either direction
    c = 0
    for idx, n in enumerate(full_to_trim_idx_map):
        if not np.isnan(n):
            c += 1
            assert trim_to_full_idx_map[int(n)] == idx
    assert c == len(trim_to_full_idx_map)

    # test for malformed data
    with pytest.raises(ValueError):
        data.crow_flies(src_x, src_y, x_arr[:-1], y_arr, max_dist)
    with pytest.raises(ValueError):
        data.crow_flies(src_x, src_y, x_arr, y_arr[:-1], max_dist)


def test_spatial_filter():

    # create some random x and y data
    random_x = np.random.uniform(0, 1000, 20)
    random_y = np.random.uniform(2000, 3000, 20)
    index_map = data.generate_index(random_x, random_y)
    src_x = 500
    src_y = 2500
    max_dist = 250
    # generate the filtered maps
    trim_to_full_idx_map, full_to_trim_idx_map = data.spatial_filter(index_map, src_x, src_y, max_dist, radial=True)

    # test radial version against crow-flies
    crow_trim_to_full, crow_full_to_trim = data.crow_flies(src_x, src_y, random_x, random_y, max_dist)
    assert np.array_equal(crow_trim_to_full, trim_to_full_idx_map)
    assert np.allclose(crow_full_to_trim, full_to_trim_idx_map, equal_nan=True)

    # test that all reachable indices are, in fact, within the max distance
    for idx, val in enumerate(full_to_trim_idx_map):
        dist = np.sqrt((random_x[idx] - src_x) ** 2 + (random_y[idx] - src_y) ** 2)
        if dist > max_dist:
            assert np.isnan(val)
        else:
            assert np.isfinite(val)

    # test the non radial version
    trim_to_full_idx_map, full_to_trim_idx_map = data.spatial_filter(index_map, src_x, src_y, max_dist, radial=False)
    for idx, val in enumerate(full_to_trim_idx_map):
        if abs(random_x[idx] - src_x) <= max_dist and abs(random_y[idx] - src_y) <= max_dist:
            assert np.isfinite(val)
        else:
            assert np.isnan(val)


def test_assign_to_network():

    # generate network
    G, pos = mock.mock_graph()
    G = graphs.networkX_simple_geoms(G)
    G = graphs.networkX_edge_defaults(G)
    node_labels, node_map, edge_map = graphs.graph_maps_from_networkX(G)

    # generate data
    data_dict = mock.mock_data(G, random_seed=5)
    data_labels, data_map, data_classes = layers.dict_to_data_map(data_dict)

    data_map = data.assign_to_network(data_map, node_map, 500)

    # for debugging
    # plot.plot_graph_maps(node_labels, node_map, edge_map, data_map)

    for data_point in data_map:
        x = data_point[0]
        y = data_point[1]
        assigned_idx = int(data_point[4])
        assigned_dist = data_point[5]

        node_x = node_map[assigned_idx][0]
        node_y = node_map[assigned_idx][1]

        # check the assigned distance
        assert abs(assigned_dist - np.sqrt((node_x - x) ** 2 + (node_y - y) ** 2)) < 0.00000001

        # check that no other nodes are closer
        for idx, node in enumerate(node_map):
            if idx != assigned_idx:
                test_x = node[0]
                test_y = node[1]
                assert assigned_dist <= np.sqrt((test_x - x) ** 2 + (test_y - y) ** 2)

    # check that malformed node and data maps throw errors
    with pytest.raises(AttributeError):
        data.assign_to_network(data_map[:, :-1], node_map, 500)

    with pytest.raises(AttributeError):
        data.assign_to_network(data_map, node_map[:, :-1], 500)


def test_aggregate_to_src_idx():

    # generate network
    G, pos = mock.mock_graph()
    G = graphs.networkX_simple_geoms(G)
    G = graphs.networkX_edge_defaults(G)
    node_labels, node_map, edge_map = graphs.graph_maps_from_networkX(G)
    netw_index = data.generate_index(node_map[:, 0], node_map[:, 1])

    # generate data
    data_dict = mock.mock_data(G, random_seed=5)
    data_labels, data_map, data_classes = layers.dict_to_data_map(data_dict)
    data_map = data.assign_to_network(data_map, node_map, 500)
    data_index = data.generate_index(data_map[:, 0], data_map[:, 1])

    reachable_classes_trim, reachable_classes_dist_trim, data_trim_to_full_idx_map = \
        data.aggregate_to_src_idx(0, 400, node_map, edge_map, netw_index, data_map, data_index, angular=False)

    # for debugging
    plot.plot_graph_maps(node_labels, node_map, edge_map, data_map[data_trim_to_full_idx_map])
