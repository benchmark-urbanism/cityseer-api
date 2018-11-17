import pytest
import numpy as np
import networkx as nx
from itertools import permutations
from shapely import geometry
from cityseer import centrality, graphs, util
import matplotlib.pyplot as plt


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


def test_compute_centrality():
    '''
    Underlying method also tested via test_networks.test_network_centralities
    '''

    # load the test graph
    G, pos = util.tutte_graph()
    G = graphs.networkX_simple_geoms(G)
    G = graphs.networkX_edge_defaults(G)  # set default edge attributes
    n_labels, n_map, e_map = graphs.graph_maps_from_networkX(G)  # generate node and edge maps

    dist = [100]

    # check that malformed signatures trigger errors
    with pytest.raises(ValueError):
        centrality.compute_centrality(n_map[:, :4], e_map, dist, close_metrics=['node_density'])

    with pytest.raises(ValueError):
        centrality.compute_centrality(n_map, e_map[:, :3], dist, close_metrics=['node_density'])

    with pytest.raises(ValueError):
        centrality.compute_centrality(n_map, e_map, dist)

    with pytest.raises(ValueError):
        centrality.compute_centrality(n_map, e_map, dist, close_metrics=['node_density_spelling_typo'])

    # check the number of returned types
    closeness_types = ['node_density', 'farness_impedance', 'farness_distance', 'harmonic', 'improved', 'gravity', 'cycles']
    betweenness_types = ['betweenness', 'betweenness_gravity']

    centrality.compute_centrality(n_map, e_map, dist, close_metrics=['node_density'])

    for cl_types in permutations(closeness_types):
        for bt_types in permutations(betweenness_types):
            args_ret = centrality.compute_centrality(n_map, e_map, dist,
                                                 close_metrics=list(cl_types), between_metrics=list(bt_types))
            assert len(args_ret) == len(cl_types) + len(bt_types) + 1  # beta list is also added

    # test centrality methods being passed through correctly
    # networkx doesn't have a maximum distance cutoff, so have to run on the whole graph
    dist = [2000]
    node_density, harmonic, betweenness, betas = \
        centrality.compute_centrality(n_map, e_map, dist,
                                      close_metrics=['node_density', 'harmonic'], between_metrics=['betweenness'])

    # check betas
    for b, d in zip(betas, dist):
        assert np.exp(b * d) == 0.01831563888873418

    # test node density
    # node density count doesn't include self-node
    for n in node_density[0]:
        assert n + 1 == len(G)

    # NOTE: modified improved closeness is not comparable to networkx version
    # test harmonic closeness
    nx_harm_cl = nx.harmonic_centrality(G, distance='impedance')
    nx_harm_cl = np.array([v for v in nx_harm_cl.values()])
    assert np.array_equal(nx_harm_cl.round(8), harmonic[0].round(8))

    # test betweenness
    # set endpoint counting to false and do not normalise
    nx_betw = nx.betweenness_centrality(G, weight='impedance', endpoints=False, normalized=False)
    nx_betw = np.array([v for v in nx_betw.values()])
    assert np.array_equal(nx_betw, betweenness[0])


def test_compute_harmonic_closeness():

    # load the test graph
    G, pos = util.tutte_graph()
    G = graphs.networkX_simple_geoms(G)
    G = graphs.networkX_edge_defaults(G)  # set default edge attributes
    n_labels, n_map, e_map = graphs.graph_maps_from_networkX(G)  # generate node and edge maps

    dist = [2000]

    nx_harm_cl = nx.harmonic_centrality(G, distance='impedance')
    nx_harm_cl = np.array([v for v in nx_harm_cl.values()])

    # test easy wrapper version of harmonic closeness
    harmonic_easy, betas = centrality.compute_harmonic_closeness(n_map, e_map, dist)
    assert np.array_equal(nx_harm_cl.round(8), harmonic_easy[0].round(8))


def test_compute_betweenness():

    # load the test graph
    G, pos = util.tutte_graph()
    G = graphs.networkX_simple_geoms(G)
    G = graphs.networkX_edge_defaults(G)  # set default edge attributes
    n_labels, n_map, e_map = graphs.graph_maps_from_networkX(G)  # generate node and edge maps

    dist = [2000]

    nx_betw = nx.betweenness_centrality(G, weight='impedance', endpoints=False, normalized=False)
    nx_betw = np.array([v for v in nx_betw.values()])

    # test easy wrapper version of betweenness
    betweenness_easy, betas = centrality.compute_betweenness(n_map, e_map, dist)
    assert np.array_equal(nx_betw, betweenness_easy[0])

# TODO: is there a way to test compute_angular_betweenness?

# TODO: is there a way to test  compute_angular_harmonic_closeness?
