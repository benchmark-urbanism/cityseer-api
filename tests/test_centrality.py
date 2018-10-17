import pytest
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from shapely import geometry
from cityseer import centrality
from . import graph_util

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

    nx.draw(G, pos=pos, with_labels=True)
    plt.show()

    assert G.number_of_nodes() == 46
    assert G.number_of_edges() == 69

    assert nx.average_degree_connectivity(G) == { 3: 3.0 }

    assert nx.average_shortest_path_length(G) == 4.356521739130435


def test_temp():

    # for d in distances:

    # fetch graphs for testing
    G, pos = graph_util.tutte_graph()
    node_map, edge_map = centrality.graph_from_networkx(G, wgs84_coords=False, decompose=False, geom=None)

    G_wgs, pos_wgs = graph_util.tutte_graph(wgs84_coords=True)
    node_map_wgs, edge_map_wgs= centrality.graph_from_networkx(G_wgs, wgs84_coords=True, decompose=False, geom=None)

    non = edge_map[:,2].round(4).tolist()
    wgs = edge_map_wgs[:,2].round(4).tolist()
    for p in zip(non, wgs):
        #print(p, round(p[0] - p[1], 4))
        print( np.ceil(p[0] / 20) == np.ceil(p[1] / 20))

    #assert len(node_map) == len(node_map_wgs)


def test_graph_from_networkx():

    # fetch graphs for testing
    G, pos = graph_util.tutte_graph()
    G_wgs, pos_wgs = graph_util.tutte_graph(wgs84_coords=True)

    # test WGS 84 conversion
    for g, wgs in zip((G, G_wgs), (False, True)):

        # check non decomposed
        node_map, edge_map = centrality.graph_from_networkx(g, wgs84_coords=wgs, decompose=False, geom=None)

        # visual checks help quickly debug issues
        # graph_util.plot_graph_maps(node_map, edge_map)

        # check basic graph lengths
        assert len(node_map) == g.number_of_nodes()
        assert len(edge_map) == g.number_of_edges() * 2

        # relax accuracy because resolution lost going to and from WGS 84
        assert np.array_equal(node_map[0].round(1), np.array([6000700, 600700, 1, 0]).round(1))
        assert np.array_equal(node_map[21].round(1), np.array([6001000, 600870, 1, 63]).round(1))

        # nearest cm
        assert np.array_equal(edge_map[0].round(2), np.array([0, 1, 120.41594579]).round(2))
        assert np.array_equal(edge_map[40].round(2), np.array([13, 12, 116.6190379]).round(2))

        # check decomposed
        node_map, edge_map = centrality.graph_from_networkx(g, wgs84_coords=wgs, decompose=100, geom=None)

        # visual checks help quickly debug issues
        graph_util.plot_graph_maps(node_map, edge_map)

        # check basic graph lengths
        #assert len(node_map) == 246
        #assert len(edge_map) == 538

        # relax accuracy because resolution lost going to and from WGS 84
        assert np.array_equal(node_map[0].round(1), np.array([6000700, 600700, 1, 0]).round(1))
        assert np.array_equal(node_map[21].round(1), np.array([6001000, 600870, 1, 63]).round(1))

        # nearest cm
        #assert np.array_equal(edge_map[0].round(2), np.array([0, 46, 40.13864859597432]).round(2))
        #assert np.array_equal(edge_map[40].round(2), np.array([13, 110, 38.873012632302]).round(2))



    # poly = geometry.Polygon([[300, 300], [900, 300], [900, 900], [300, 900]])
