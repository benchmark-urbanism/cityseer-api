import copy
import pytest
import utm
import numpy as np
from cityseer import data, util, graphs


def test_dict_wgs_to_utm():

    # check that node coordinates are correctly converted
    G_utm, pos = util.tutte_graph()
    data_dict = util.mock_data(G_utm)

    data_dict_wgs = copy.deepcopy(data_dict)
    for k, v in data_dict_wgs.items():
        x = v['x']
        y = v['y']
        y, x = utm.to_latlon(y, x, 30, 'U')
        data_dict_wgs[k]['x'] = x
        data_dict_wgs[k]['y'] = y

    data_dict_converted = data.dict_wgs_to_utm(data_dict_wgs)

    for k in data_dict.keys():
        # rounding can be tricky
        assert data_dict[k]['x'] - data_dict_converted[k]['x'] < 0.001
        assert data_dict[k]['y'] - data_dict_converted[k]['y'] < 0.001

    # check that missing node attributes throw an error
    for attr in ['x', 'y']:
        G_wgs, pos_wgs = util.tutte_graph(wgs84_coords=True)
        data_dict_wgs = util.mock_data(G_wgs)
        for k in data_dict_wgs.keys():
            del data_dict_wgs[k][attr]
            break
        # check that missing attribute throws an error
        with pytest.raises(AttributeError):
            data.dict_wgs_to_utm(data_dict_wgs)

    # check that non WGS coordinates throw error
    G_utm, pos = util.tutte_graph()
    data_dict = util.mock_data(G_utm)
    with pytest.raises(AttributeError):
        data.dict_wgs_to_utm(data_dict)


def test_dict_to_map():

    # generate mock data
    G, pos = util.tutte_graph()
    data_dict = util.mock_data(G)
    data_labels, data_map = data.dict_to_data_map(data_dict)

    assert len(data_labels) == len(data_map) == len(data_dict)

    for d_label, d in zip(data_labels, data_map):
        assert d[0] == data_dict[d_label]['x']
        assert d[1] == data_dict[d_label]['y']
        assert d[2] == data_dict[d_label]['live']
        assert d[3] == data_dict[d_label]['class']
        assert np.isnan(d[4])
        assert np.isnan(d[5])

    # check that missing attributes throw errors
    for attr in ['x', 'y']:
        for k in data_dict.keys():
            del data_dict[k][attr]
        with pytest.raises(AttributeError):
            data.dict_to_data_map(data_dict)


def test_assign_data_to_network():

    # generate network
    G, pos = util.tutte_graph()
    G = graphs.networkX_simple_geoms(G)
    G = graphs.networkX_edge_defaults(G)
    node_labels, node_map, edge_map = graphs.graph_maps_from_networkX(G)

    # generate data
    data_dict = util.mock_data(G)
    data_labels, data_map = data.dict_to_data_map(data_dict)

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
