import pytest
import numpy as np
from cityseer.util import graphs, layers, mock
from cityseer.algos import data


def test_assign_data_to_network():

    # generate network
    G, pos = mock.mock_graph()
    G = graphs.networkX_simple_geoms(G)
    G = graphs.networkX_edge_defaults(G)
    node_labels, node_map, edge_map = graphs.graph_maps_from_networkX(G)

    # generate data
    data_dict = mock.mock_data(G)
    data_labels, data_map, data_classes = layers.dict_to_data_map(data_dict)

    data_map = data.assign_data_to_network(data_map, node_map, 500)

    for data_point in data_map:
        x = data_point[0]
        y = data_point[1]
        assigned_idx = int(data_point[4])
        assigned_dist = data_point[5]

        node_x = node_map[assigned_idx][0]
        node_y = node_map[assigned_idx][1]

        # check the assigned distance
        assert assigned_dist == np.sqrt((node_x - x) ** 2 + (node_y - y) ** 2)

        # check that no other nodes are closer
        for idx, node in enumerate(node_map):
            if idx != assigned_idx:
                test_x = node[0]
                test_y = node[1]
                assert assigned_dist <= np.sqrt((test_x - x) ** 2 + (test_y - y) ** 2)

    # check that malformed node and data maps throw errors
    with pytest.raises(AttributeError):
        data.assign_data_to_network(data_map[:, :-1], node_map, 500)

    with pytest.raises(AttributeError):
        data.assign_data_to_network(data_map[:, :-1], node_map, 500)
