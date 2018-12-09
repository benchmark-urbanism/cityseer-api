import copy

import numpy as np
import pytest
import utm

from cityseer.metrics import layers, networks
from cityseer.util import mock, graphs


def test_dict_wgs_to_utm():
    # check that node coordinates are correctly converted
    G_utm, pos = mock.mock_graph()
    data_dict = mock.mock_data(G_utm)

    data_dict_wgs = copy.deepcopy(data_dict)
    for k, v in data_dict_wgs.items():
        x = v['x']
        y = v['y']
        y, x = utm.to_latlon(y, x, 30, 'U')
        data_dict_wgs[k]['x'] = x
        data_dict_wgs[k]['y'] = y

    data_dict_converted = layers.dict_wgs_to_utm(data_dict_wgs)

    for k in data_dict.keys():
        # rounding can be tricky
        assert abs(data_dict[k]['x'] - data_dict_converted[k]['x']) < 0.01
        assert abs(data_dict[k]['y'] - data_dict_converted[k]['y']) < 0.01

    # check that missing node attributes throw an error
    for attr in ['x', 'y']:
        G_wgs, pos_wgs = mock.mock_graph(wgs84_coords=True)
        data_dict_wgs = mock.mock_data(G_wgs)
        for k in data_dict_wgs.keys():
            del data_dict_wgs[k][attr]
            break
        # check that missing attribute throws an error
        with pytest.raises(AttributeError):
            layers.dict_wgs_to_utm(data_dict_wgs)

    # check that non WGS coordinates throw error
    G_utm, pos = mock.mock_graph()
    data_dict = mock.mock_data(G_utm)
    with pytest.raises(AttributeError):
        layers.dict_wgs_to_utm(data_dict)


def test_data_map_from_dict():
    # generate mock data
    G, pos = mock.mock_graph()
    data_dict = mock.mock_data(G)
    data_uids, data_map, class_labels = layers.data_map_from_dict(data_dict)

    assert len(data_uids) == len(data_map) == len(data_dict)

    for d_label, d in zip(data_uids, data_map):
        assert d[0] == data_dict[d_label]['x']
        assert d[1] == data_dict[d_label]['y']
        assert d[2] == data_dict[d_label]['live']
        # check that the class encoding maps back to the class
        class_encoding = int(d[3])
        assert class_labels[class_encoding] == data_dict[d_label]['class']
        assert np.isnan(d[4])
        assert np.isnan(d[5])

    # check that missing attributes throw errors
    for attr in ['x', 'y']:
        for k in data_dict.keys():
            del data_dict[k][attr]
        with pytest.raises(AttributeError):
            layers.data_map_from_dict(data_dict)


def test_Data_Layer():
    G, pos = mock.mock_graph()
    data_dict = mock.mock_data(G)
    data_uids, data_map, class_labels = layers.data_map_from_dict(data_dict)
    x_arr = data_map[:, 0]
    y_arr = data_map[:, 1]
    live = data_map[:, 2]
    class_codes = data_map[:, 3]

    # test against Data_Layer internal process
    D = layers.Data_Layer(data_uids, data_map, class_labels)
    assert D.uids == data_uids
    assert np.allclose(D._data, data_map, equal_nan=True)
    assert D.class_labels == class_labels
    assert np.array_equal(D.x_arr, x_arr)
    assert np.array_equal(D.y_arr, y_arr)
    assert np.array_equal(D.live, live)
    assert np.array_equal(D.class_codes, class_codes)


def test_Data_Layer_From_Dict():
    G, pos = mock.mock_graph()
    data_dict = mock.mock_data(G)
    data_uids, data_map, class_labels = layers.data_map_from_dict(data_dict)
    x_arr = data_map[:, 0]
    y_arr = data_map[:, 1]
    live = data_map[:, 2]
    class_codes = data_map[:, 3]

    # test against Data_Layer_From_Dict's internal process
    D = layers.Data_Layer_From_Dict(data_dict)
    assert D.uids == data_uids
    assert np.allclose(D._data, data_map, equal_nan=True)
    assert D.class_labels == class_labels
    assert np.array_equal(D.x_arr, x_arr)
    assert np.array_equal(D.y_arr, y_arr)
    assert np.array_equal(D.live, live)
    assert np.array_equal(D.class_codes, class_codes)


def test_compute_landuses():
    G, pos = mock.mock_graph()
    G = graphs.networkX_simple_geoms(G)
    G = graphs.networkX_edge_defaults(G)

    distances = [100, 200, 400]
    N = networks.Network_Layer_From_NetworkX(G, distances)
    data_dict = mock.mock_data(G)
    D = layers.Data_Layer_From_Dict(data_dict, qs=[0, 1, 2])
    # hacking this for testing
    D._cl_disparity_wt_matrix = np.full((len(D.class_labels), len(D.class_labels)), 1)
    D.assign_to_network(N, max_dist=400)

    div_keys = ['hill',
                'hill_branch_wt',
                'hill_pairwise_wt',
                'hill_pairwise_disparity',
                'shannon',
                'gini_simpson',
                'raos_pairwise_disparity']
    ac_codes = [2, 4, 6]

    D.compute_landuses(div_keys, ac_codes)

    data = N.metrics_to_dict()
