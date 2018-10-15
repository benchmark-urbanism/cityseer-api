import pytest
import numpy as np
import networkx as nx
from shapely import geometry
from cityseer import centrality
from . import generate_graph

def test_custom_decay_betas():

    assert centrality.distance_from_beta(0.04) == (np.array([100.]), 0.01831563888873418)

    assert centrality.distance_from_beta(0.0025) == (np.array([1600.]), 0.01831563888873418)

    arr, t_w = centrality.distance_from_beta(0.04, min_threshold_wt=0.001)
    assert np.array_equal(arr.round(8), np.array([172.69388197]).round(8))
    assert t_w == 0.001

    arr, t_w = centrality.distance_from_beta([0.04, 0.0025])
    assert np.array_equal(arr, np.array([100, 1600]))
    assert t_w == 0.01831563888873418

    with pytest.raises(ValueError):
        centrality.distance_from_beta(-0.04)


def test_generate_graph():

    G, pos = generate_graph.tutte_graph()

    # import matplotlib.pyplot as plt
    # nx.draw(G, pos=pos, with_labels=True)
    # plt.show()

    assert G.number_of_nodes() == 46

    assert G.number_of_edges() == 69

    assert nx.average_degree_connectivity(G) == { 3: 3.0 }

    assert nx.average_shortest_path_length(G) == 4.356521739130435


def test_graph_from_networkx():

    G, pos = generate_graph.tutte_graph()

    poly = geometry.Polygon([[300, 300], [900, 300], [900, 900], [300, 900]])

    node_map, link_map = centrality.graph_from_networkx(G, wgs84_coords=False, decompose=False, pre_geom=None)

    assert np.array_equal(node_map[0], np.array([700, 700, 1, 0]))

    assert np.array_equal(node_map[21], np.array([1000, 870, 1, 63]))

    assert np.array_equal(link_map[0].round(8), np.array([0, 1, 120.41594579]).round(8))

    assert np.array_equal(link_map[40].round(8), np.array([13, 12, 116.6190379]).round(8))
