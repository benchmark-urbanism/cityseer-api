import pytest
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from shapely import geometry
from cityseer import centrality
from cityseer.util import graph_util



def test_generate_graph():

    G, pos = graph_util.tutte_graph()

    nx.draw(G, pos=pos, with_labels=True)
    plt.show()

    assert G.number_of_nodes() == 46
    assert G.number_of_edges() == 69

    assert nx.average_degree_connectivity(G) == { 3: 3.0 }

    assert nx.average_shortest_path_length(G) == 4.356521739130435


def test_distance_from_beta():

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


def test_graph_from_networkx():

    # polygon for testing geom method
    geom = geometry.Polygon([[6000500, 600500], [6000900, 600500], [6000900, 600900], [6000500, 600900]])

    # fetch graphs for testing
    G, pos = graph_util.tutte_graph()
    G_wgs, pos_wgs = graph_util.tutte_graph(wgs84_coords=True)

    # test non-decomposed versions
    # ============================
    n_map, e_map = centrality.graph_from_networkx(G, wgs84_coords=False, decompose=False, geom=None)
    n_map_geom, e_map_geom = centrality.graph_from_networkx(G, wgs84_coords=False, decompose=False, geom=geom)
    n_map_wgs, e_map_wgs = centrality.graph_from_networkx(G_wgs, wgs84_coords=True, decompose=False, geom=None)
    n_map_wgs_geom, e_map_wgs_geom = centrality.graph_from_networkx(G_wgs, wgs84_coords=True, decompose=False, geom=geom)

    # check basic graph lengths
    assert len(n_map) == len(n_map_geom) == len(n_map_wgs) == len(n_map_wgs_geom) == G.number_of_nodes()
    assert len(e_map) == len(e_map_geom) == len(e_map_wgs) == len(e_map_wgs_geom) == G.number_of_edges() * 2

    # pairwise permutations - since n_map = n_map_wgs just need to repeat for n_map and n_map_geom
    assert np.array_equal(n_map[:,[0, 1, 3]], n_map_geom[:,[0, 1, 3]])  # live designations won't match
    assert np.array_equal(n_map, n_map_wgs)
    assert np.array_equal(n_map[:,[0, 1, 3]], n_map_wgs_geom[:,[0, 1, 3]])  # live designations won't match

    assert np.array_equal(e_map, e_map_geom)
    assert np.array_equal(e_map, e_map_wgs)
    assert np.array_equal(e_map, e_map_wgs_geom)

    # check attributes, just use n_map version since other arrays already asserted as equal
    assert np.array_equal(n_map[0], np.array([6000700, 600700, 1, 0]))
    assert np.array_equal(n_map[21], np.array([6001000, 600870, 1, 63]))

    assert np.array_equal(e_map[0], np.array([0, 1, 120.41594578792295, 120.41594578792295]))
    assert np.array_equal(e_map[40], np.array([13, 12, 116.61903789690601, 116.61903789690601]))

    # check live designations
    assert n_map[:, 2].sum() == n_map_wgs[:,2].sum() == G.number_of_nodes()
    # small differences due to rounding in conversions, so split over two lines
    assert n_map_geom[:,2].sum() == 6
    assert n_map_wgs_geom[:,2].sum() == 7

    # plots for debugging
    # graph_util.plot_graph_maps(n_map, e_map, geom=geom)
    # graph_util.plot_graph_maps(n_map_wgs, e_map_wgs, geom=geom)

    # test decomposed versions
    # ========================
    n_map, e_map = centrality.graph_from_networkx(G, wgs84_coords=False, decompose=20, geom=None)
    n_map_geom, e_map_geom = centrality.graph_from_networkx(G, wgs84_coords=False, decompose=20, geom=geom)
    n_map_wgs, e_map_wgs = centrality.graph_from_networkx(G_wgs, wgs84_coords=True, decompose=20, geom=None)
    n_map_wgs_geom, e_map_wgs_geom = centrality.graph_from_networkx(G_wgs, wgs84_coords=True, decompose=20, geom=geom)

    # check basic graph lengths
    # conversion and rounding differences means slight decomposition differences
    assert len(n_map) == len(n_map_geom) == len(n_map_wgs) == len(n_map_wgs_geom) == 602
    assert len(e_map) == len(e_map_geom) == len(e_map_wgs) == len(e_map_wgs_geom) == 1250

    # pairwise permutations - since n_map = n_map_wgs just need to repeat for n_map and n_map_geom
    assert np.array_equal(n_map[:,[0, 1, 3]], n_map_geom[:,[0, 1, 3]])  # live designations won't match
    assert np.array_equal(n_map, n_map_wgs)
    assert np.array_equal(n_map[:,[0, 1, 3]], n_map_wgs_geom[:,[0, 1, 3]])  # live designations won't match

    assert np.array_equal(e_map, e_map_geom)
    assert np.array_equal(e_map, e_map_wgs)
    assert np.array_equal(e_map, e_map_wgs_geom)

    # check attributes, just use n_map version since other arrays already asserted as equal
    assert np.array_equal(n_map[0], np.array([6000700, 600700, 1, 0]))
    assert np.array_equal(n_map[21], np.array([6001000, 600870, 1, 63]))

    assert np.array_equal(e_map[0], np.array([0, 46, 17.202277969703278, 17.202277969703278]))
    assert np.array_equal(e_map[40], np.array([13, 221, 19.436506316151, 19.436506316151]))

    # check live designations
    assert n_map[:, 2].sum() == n_map_wgs[:,2].sum() == 602
    # small differences due to rounding in conversions, so split over two lines
    assert n_map_geom[:,2].sum() == 81
    assert n_map_wgs_geom[:,2].sum() == 82

    # plots for debugging
    # graph_util.plot_graph_maps(n_map, e_map, geom=geom)
    # graph_util.plot_graph_maps(n_map_wgs, e_map_wgs, geom=geom)

    # SOME OTHER CHECKS
    # =================

    # check that passing lng, lat without WGS flag raises error
    with pytest.raises(ValueError):
        n, e = centrality.graph_from_networkx(G_wgs, wgs84_coords=False, decompose=False, geom=None)

    # check that custom lengths are processed
    # weights should automatically be set to the same value (instead of the geom length)
    for s, e in G.edges():
        s_geom = geometry.Point(G.node[s]['x'], G.node[s]['y'])
        e_geom = geometry.Point(G.node[e]['x'], G.node[e]['y'])
        G[s][e]['length'] = s_geom.distance(e_geom) * 2

    n_map, e_map = centrality.graph_from_networkx(G, wgs84_coords=False, decompose=None, geom=None)

    assert np.array_equal(n_map[0], np.array([6000700, 600700, 1, 0]))
    assert np.array_equal(n_map[21], np.array([6001000, 600870, 1, 63]))

    assert np.array_equal(e_map[0], np.array([0, 1, 240.8318915758459, 240.8318915758459]))
    assert np.array_equal(e_map[40], np.array([13, 12, 233.23807579381202, 233.23807579381202]))

    n_map, e_map = centrality.graph_from_networkx(G, wgs84_coords=False, decompose=20, geom=None)

    assert np.array_equal(n_map[0], np.array([6000700, 600700, 1, 0]))
    assert np.array_equal(n_map[21], np.array([6001000, 600870, 1, 63]))

    assert np.array_equal(e_map[0], np.array([0, 46, 18.525530121218914, 18.525530121218914]))
    assert np.array_equal(e_map[40], np.array([13, 411, 19.436506316151, 19.436506316151]))

    # check that custom weights are processed
    for s, e in G.edges():
        s_geom = geometry.Point(G.node[s]['x'], G.node[s]['y'])
        e_geom = geometry.Point(G.node[e]['x'], G.node[e]['y'])
        G[s][e]['weight'] = s_geom.distance(e_geom)

    n_map, e_map = centrality.graph_from_networkx(G, wgs84_coords=False, decompose=None, geom=None)

    assert np.array_equal(n_map[0], np.array([6000700, 600700, 1, 0]))
    assert np.array_equal(n_map[21], np.array([6001000, 600870, 1, 63]))

    assert np.array_equal(e_map[0], np.array([0, 1, 240.8318915758459, 120.41594578792295]))
    assert np.array_equal(e_map[40], np.array([13, 12, 233.23807579381202, 116.61903789690601]))

    n_map, e_map = centrality.graph_from_networkx(G, wgs84_coords=False, decompose=20, geom=None)

    assert np.array_equal(n_map[0], np.array([6000700, 600700, 1, 0]))
    assert np.array_equal(n_map[21], np.array([6001000, 600870, 1, 63]))

    assert np.array_equal(e_map[0], np.array([0, 46, 18.525530121218914, 9.262765060609457]))
    assert np.array_equal(e_map[40], np.array([13, 411, 19.436506316151, 9.7182531580755]))

    # check that passing negative lenghts or weights throw errors
    G[0][1]['length'] = -1
    with pytest.raises(ValueError):
        n, e = centrality.graph_from_networkx(G, wgs84_coords=False, decompose=False, geom=None)

    G[0][1]['length'] = 240
    G[0][1]['weight'] = -1
    with pytest.raises(ValueError):
        n, e = centrality.graph_from_networkx(G, wgs84_coords=False, decompose=False, geom=None)


def test_centrality():

    G, pos = graph_util.tutte_graph()

    n_map, e_map = centrality.graph_from_networkx(G, wgs84_coords=False, decompose=False, geom=None)

    node_density, imp_closeness, gravity, betweenness, betweenness_wt = centrality.centrality(n_map, e_map, [100, 1000])

    print(node_density)
