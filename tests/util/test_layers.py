import copy
import pytest
import utm
import numpy as np
from cityseer.util import mock, layers


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


def test_dict_to_data_map():

    # generate mock data
    G, pos = mock.mock_graph()
    data_dict = mock.mock_data(G)
    data_labels, data_map, data_classes = layers.dict_to_data_map(data_dict)

    assert len(data_labels) == len(data_map) == len(data_dict)

    for d_label, d in zip(data_labels, data_map):
        assert d[0] == data_dict[d_label]['x']
        assert d[1] == data_dict[d_label]['y']
        assert d[2] == data_dict[d_label]['live']
        # check that the class encoding maps back to the class
        class_encoding = int(d[3])
        assert data_classes[class_encoding] == data_dict[d_label]['class']
        assert np.isnan(d[4])
        assert np.isnan(d[5])

    # check that missing attributes throw errors
    for attr in ['x', 'y']:
        for k in data_dict.keys():
            del data_dict[k][attr]
        with pytest.raises(AttributeError):
            layers.dict_to_data_map(data_dict)
