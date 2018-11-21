import pytest
import random
import numpy as np
from cityseer import util
import networkx as nx
import matplotlib.pyplot as plt


def test_data_dict_to_map():

    # generate mock data
    data_dict = {}
    for i in range(100):
        data_dict[i] = {
            'x': random.uniform(0, 200000),
            'y': random.uniform(0, 200000),
            'live': bool(random.getrandbits(1)),
            'class': random.uniform(0, 10)
        }

    data_labels, data_map = util.data_dict_to_map(data_dict)

    assert len(data_labels) == len(data_map) == len(data_dict)

    for d_label, d in zip(data_labels, data_map):
        assert d[0] == data_dict[d_label]['x']
        assert d[1] == data_dict[d_label]['y']
        assert d[2] == data_dict[d_label]['live']
        assert np.isnan(d[3])
        assert d[4] == data_dict[d_label]['class']

    # check that missing attributes throw errors
    for attr in ['x', 'y']:
        for k in data_dict.keys():
            del data_dict[k][attr]
        with pytest.raises(AttributeError):
            util.data_dict_to_map(data_dict)


def test_tutte_graph():

    G, pos = util.tutte_graph()
    G_wgs, pos_wgs = util.tutte_graph(wgs84_coords=True)

    #nx.draw(G, pos=pos, with_labels=True)
    #plt.show()

    for g in [G, G_wgs]:

        assert g.number_of_nodes() == 46
        assert g.number_of_edges() == 69

        assert nx.average_degree_connectivity(g) == { 3: 3.0 }

        assert nx.average_shortest_path_length(g) == 4.356521739130435