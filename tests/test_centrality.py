import pytest
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from shapely import geometry
from cityseer import centrality
from cityseer.util import graph_util

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

    G, pos = graph_util.tutte_graph()

    # nx.draw(G, pos=pos, with_labels=True)
    # plt.show()

    assert G.number_of_nodes() == 46
    assert G.number_of_edges() == 69

    assert nx.average_degree_connectivity(G) == { 3: 3.0 }

    assert nx.average_shortest_path_length(G) == 4.356521739130435


def test_graph_from_networkx():

    # TODO: add geom variant and test three-wise

    # fetch graphs for testing
    G, pos = graph_util.tutte_graph()
    G_wgs, pos_wgs = graph_util.tutte_graph(wgs84_coords=True)

    # test non-decomposed versions
    node_map, edge_map = centrality.graph_from_networkx(G, wgs84_coords=False, decompose=False, geom=None)
    node_map_wgs, edge_map_wgs= centrality.graph_from_networkx(G_wgs, wgs84_coords=True, decompose=False, geom=None)

    # check basic graph lengths
    assert len(node_map) == len(node_map_wgs) == G.number_of_nodes()
    assert len(edge_map) == len(edge_map_wgs) == G.number_of_edges() * 2

    # check wgs vs. non
    assert np.array_equal(node_map, node_map_wgs)
    assert np.array_equal(edge_map, edge_map_wgs)

    # check attributes, just use node_map version since wgs and non have already been asserted as equal
    assert np.array_equal(node_map[0], np.array([6000700, 600700, 1, 0]))
    assert np.array_equal(node_map[21], np.array([6001000, 600870, 1, 63]))

    assert np.array_equal(edge_map[0], np.array([0, 1, 120]))
    assert np.array_equal(edge_map[40], np.array([13, 12, 116]))

    # plots for debugging
    # graph_util.plot_graph_maps(node_map, edge_map)
    # graph_util.plot_graph_maps(node_map_wgs, edge_map_wgs)

    # test decomposed versions
    node_map, edge_map = centrality.graph_from_networkx(G, wgs84_coords=False, decompose=20, geom=None)
    node_map_wgs, edge_map_wgs = centrality.graph_from_networkx(G_wgs, wgs84_coords=True, decompose=20, geom=None)

    # check basic graph lengths
    assert len(node_map) == len(node_map_wgs) == 602
    assert len(edge_map) == len(edge_map_wgs) == 1250

    # check wgs vs. non
    assert np.array_equal(node_map, node_map_wgs)
    assert np.array_equal(edge_map, edge_map_wgs)

    # check attributes, just use node_map version since wgs and non have already been asserted as equal
    assert np.array_equal(node_map[0], np.array([6000700, 600700, 1, 0]))
    assert np.array_equal(node_map[21], np.array([6001000, 600870, 1, 63]))

    assert np.array_equal(edge_map[0], np.array([0, 46, 17]))
    assert np.array_equal(edge_map[40], np.array([13, 221, 19]))

    # plots for debugging
    # graph_util.plot_graph_maps(node_map, edge_map)
    # graph_util.plot_graph_maps(node_map_wgs, edge_map_wgs)

    # poly = geometry.Polygon([[300, 300], [900, 300], [900, 900], [300, 900]])
