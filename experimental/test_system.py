import pytest

from cityseer.metrics import layers
from cityseer.util import mock, graphs
from experimental import system


def test_System():
    # load the test graph
    G, pos = mock.mock_graph()
    G = graphs.networkX_simple_geoms(G)
    G = graphs.networkX_edge_defaults(G)  # set default edge attributes
    node_uids, node_map, edge_map = graphs.graph_maps_from_networkX(G)  # generate node and edge maps

    data_dict = mock.mock_data_dict(G)
    data_uids, data_map = layers.data_map_from_dict(data_dict)

    S = system.System(node_map, edge_map)
    S.load_data_map('test_1', data_map)
    S.load_data_map('test_2', data_map)
    S.load_data_map('test_3', data_map)
    S.load_data_map('test_4', data_map)
    S.load_data_map('test_5', data_map)
    # catch exceeded number of data maps
    with pytest.raises(ValueError):
        S.load_data_map('test_6', data_map)
    # catch duplicate data maps
    with pytest.raises(ValueError):
        S.load_data_map('test_1', data_map)

    S.remove_data_map('test_1')
    S.load_data_map('test_7', data_map)
    assert S.data_map_0_name == 'test_7'
