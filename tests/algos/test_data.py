import pytest
import numpy as np
from cityseer.util import graphs, layers, mock, plot
from cityseer.algos import data



def test_merge_sort_1d():

    for n in range(1, 20):
        # create random data - stack with indices
        random_data = np.random.uniform(0, 500, n)
        # check that merge sort behaves as anticipated
        sorted_data = data.merge_sort_1d(random_data)
        for idx in range(len(sorted_data) - 1):
            # check that values increase
            assert sorted_data[idx] <= sorted_data[idx + 1]
        # check that all of the data has been retained
        assert len(sorted_data) == len(random_data)
        for d in random_data:
            assert d in sorted_data


def test_merge_sort_2d():

    # test 2d - this version uses a second dimension storing the indices
    for n in range(1, 20):
        # create random data - stack with indices
        random_data = np.random.uniform(0, 500, n)
        indices = np.arange(len(random_data))
        stacked_data = np.vstack((random_data, indices)).T
        # check that merge sort behaves as anticipated
        sorted_data = data.merge_sort_2d(stacked_data)
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


def test_generate_index():

    for n in range(1, 20):
        # create some random x and y data
        random_x = np.random.uniform(0, 1000, 20)
        random_y = np.random.uniform(2000, 3000, 20)
        index_map = data.generate_index(random_x, random_y)
        # test arrangement of index data against independently sorted data
        x_sort = data.merge_sort_2d(np.vstack((random_x, np.arange(len(random_x)))).T)
        y_sort = data.merge_sort_2d(np.vstack((random_y, np.arange(len(random_y)))).T)
        for idx in range(len(index_map)):
            assert np.array_equal(x_sort[idx][:2], index_map[idx][:2])
            assert np.array_equal(y_sort[idx][:2], index_map[idx][2:])
        # test malformed signature
        with pytest.raises(ValueError):
            data.generate_index(random_x[:-1], random_y)
        with pytest.raises(ValueError):
            data.generate_index(random_x, random_y[:-1])


def test_binary_search():

    for n in range(1, 10):
        # create index sorted data
        max_val = 500
        random_data = np.random.uniform(0, max_val, n)
        indices = np.arange(len(random_data))
        stacked_data = np.vstack((random_data, indices)).T
        sorted_data = data.merge_sort_2d(stacked_data)
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
                elif l_idx:
                    assert d <= right_max


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

    # generate data
    data_dict = mock.mock_data(G, random_seed=5)
    data_labels, data_map, data_classes = layers.dict_to_data_map(data_dict)

    data_map = data.assign_to_network(data_map, node_map, 500)

    reachable_classes_trim, reachable_classes_dist_trim, data_trim_to_full_idx_map = \
        data.aggregate_to_src_idx(node_map, edge_map, data_map, 0, max_dist=400, angular=False)

    # for debugging
    plot.plot_graph_maps(node_labels, node_map, edge_map, data_map[data_trim_to_full_idx_map])

