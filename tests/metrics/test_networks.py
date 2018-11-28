import pytest
import numpy as np
import networkx as nx
from itertools import permutations
from cityseer.algos import data
from cityseer.util import mock, graphs
from cityseer.metrics import networks


def test_Network_Layer():

    G, pos = mock.mock_graph()
    G = graphs.networkX_simple_geoms(G)
    G = graphs.networkX_edge_defaults(G)
    node_uids, node_map, edge_map = graphs.graph_maps_from_networkX(G)
    x_arr = node_map[:, 0]
    y_arr = node_map[:, 1]
    live = node_map[:, 2]
    index = data.generate_index(x_arr, y_arr)

    # test against Network_Layer's internal process
    N = networks.Network_Layer(node_uids, node_map, edge_map)
    assert N.uids == node_uids
    assert np.array_equal(N.nodes, node_map)
    assert np.array_equal(N.edges, edge_map)
    assert np.array_equal(N.index, index)
    assert np.array_equal(N.x_arr, x_arr)
    assert np.array_equal(N.y_arr, y_arr)
    assert np.array_equal(N.live, live)

    G_round_trip = N.to_networkX()
    # graph_maps_from_networkX generates implicit live (all True) and weight (all 1) attributes if missing
    for n, d in G.nodes(data=True):
        assert n in G_round_trip
        assert G.nodes[n]['x'] == d['x']
        assert G.nodes[n]['y'] == d['y']
    assert G_round_trip.edges == G.edges


def test_Network_Layer_From_NetworkX():

    G, pos = mock.mock_graph()
    G = graphs.networkX_simple_geoms(G)
    G = graphs.networkX_edge_defaults(G)
    node_uids, node_map, edge_map = graphs.graph_maps_from_networkX(G)
    x_arr = node_map[:, 0]
    y_arr = node_map[:, 1]
    live = node_map[:, 2]
    index = data.generate_index(x_arr, y_arr)

    # test against Network_Layer_From_NetworkX's internal process
    N = networks.Network_Layer_From_NetworkX(G)
    assert N.uids == node_uids
    assert np.array_equal(N.nodes, node_map)
    assert np.array_equal(N.edges, edge_map)
    assert np.array_equal(N.index, index)
    assert np.array_equal(N.x_arr, x_arr)
    assert np.array_equal(N.y_arr, y_arr)
    assert np.array_equal(N.live, live)


def test_distance_from_beta():

    assert networks.distance_from_beta(-0.04) == (np.array([100.]), 0.01831563888873418)
    assert networks.distance_from_beta(-0.0025) == (np.array([1600.]), 0.01831563888873418)

    arr, t_w = networks.distance_from_beta(-0.04, min_threshold_wt=0.001)
    assert np.array_equal(arr.round(8), np.array([172.69388197]).round(8))
    assert t_w == 0.001

    arr, t_w = networks.distance_from_beta([-0.04, -0.0025])
    assert np.array_equal(arr, np.array([100, 1600]))
    assert t_w == 0.01831563888873418

    with pytest.raises(ValueError):
        networks.distance_from_beta(0.04)


def test_compute_centrality():
    '''
    Underlying method also tested via test_networks.test_network_centralities
    '''

    G, pos = mock.mock_graph()
    G = graphs.networkX_simple_geoms(G)
    G = graphs.networkX_edge_defaults(G)
    N = networks.Network_Layer_From_NetworkX(G)

    dist = [100]

    # check that malformed signatures trigger errors
    with pytest.raises(ValueError):
        networks.compute_centrality(N, dist)
    with pytest.raises(ValueError):
        networks.compute_centrality(N, dist, close_metrics=['node_density_spelling_typo'])

    # check the number of returned types
    closeness_types = ['node_density', 'farness_impedance', 'farness_distance', 'harmonic', 'improved', 'gravity', 'cycles']
    betweenness_types = ['betweenness', 'betweenness_gravity']

    networks.compute_centrality(N, dist, close_metrics=['node_density'])

    for cl_types in permutations(closeness_types):
        for bt_types in permutations(betweenness_types):
            args_ret = networks.compute_centrality(N, dist, close_metrics=list(cl_types), between_metrics=list(bt_types))
            assert len(args_ret) == len(cl_types) + len(bt_types) + 1  # beta list is also added

    # test centrality methods being passed through correctly
    # networkx doesn't have a maximum distance cutoff, so have to run on the whole graph
    dist = [2000]
    node_density, harmonic, betweenness, betas = \
        networks.compute_centrality(N, dist, close_metrics=['node_density', 'harmonic'], between_metrics=['betweenness'])

    # check betas
    for b, d in zip(betas, dist):
        assert np.exp(b * d) == 0.01831563888873418

    # test node density
    # node density count doesn't include self-node
    for n in node_density:
        assert n + 1 == len(G)

    # NOTE: modified improved closeness is not comparable to networkx version
    # test harmonic closeness
    nx_harm_cl = nx.harmonic_centrality(G, distance='impedance')
    nx_harm_cl = np.array([v for v in nx_harm_cl.values()])
    assert np.array_equal(nx_harm_cl.round(8), harmonic.round(8))

    # test betweenness
    # set endpoint counting to false and do not normalise
    nx_betw = nx.betweenness_centrality(G, weight='impedance', endpoints=False, normalized=False)
    nx_betw = np.array([v for v in nx_betw.values()])
    assert np.array_equal(nx_betw, betweenness)


def wrapper_for_centralities(func, close_key, betw_key, angular):

    # load the test graph
    G, pos = mock.mock_graph()
    G = graphs.networkX_simple_geoms(G)
    G = graphs.networkX_to_dual(G)
    N = networks.Network_Layer_From_NetworkX(G)

    # test easy wrapper version of angular harmonic closeness against underlying function
    # (networkX doesn't check against sidestepping...)

    # test first with single distance
    dist = [500]
    easy = func(N, dist)
    full, betas = networks.compute_centrality(N, dist, close_metrics=close_key, between_metrics=betw_key, angular=angular)
    # compute_angular_harmonic_closeness unpacks single dimension arrays
    assert np.array_equal(easy, full)

    # test with multiple distances
    dist = [500, 2000]
    easy = func(N, dist)
    full, betas = networks.compute_centrality(N, dist, close_metrics=close_key, between_metrics=betw_key, angular=angular)
    assert np.array_equal(easy, full)


# this function is for running the same tests on gravity and betweenness_gravity
def wrapper_for_gravity_centralities(func, close_key, betw_key):

    # load the test graph
    G, pos = mock.mock_graph()
    G = graphs.networkX_simple_geoms(G)
    G = graphs.networkX_edge_defaults(G)  # set default edge attributes
    N = networks.Network_Layer_From_NetworkX(G)

    # test gravity centrality easy wrapper versions against underlying function
    dist = [500]
    easy_compute = func(N, distances=dist)
    full_compute, betas = networks.compute_centrality(N, dist, close_metrics=close_key, between_metrics=betw_key)
    assert np.array_equal(easy_compute, full_compute)

    # check that betas return same results
    full_compute_betas = func(N, betas=betas)
    assert np.array_equal(easy_compute, full_compute_betas)

    # check on multiple distances
    dist = [500, 2000]
    easy_compute = func(N, distances=dist)
    full_compute, betas = networks.compute_centrality(N, dist, close_metrics=close_key, between_metrics=betw_key)
    assert np.array_equal(easy_compute, full_compute)

    # check that betas return same results
    betweenness_gravity_easy_betas = func(N, betas=betas)
    assert np.array_equal(easy_compute, betweenness_gravity_easy_betas)

    # check that providing both betas and distances returns error
    with pytest.raises(ValueError):
        func(N, betas=betas, distances=dist)

    # check that providing neither betas nor distances returns error
    with pytest.raises(ValueError):
        func(N)

    # check custom min_threshold_wt
    betas = [-0.0025, -0.005]
    dist, threshold_wt = networks.distance_from_beta(betas, min_threshold_wt=0.1)
    easy_dist = func(N, distances=dist, min_threshold_wt=threshold_wt)
    easy_betas = func(N, betas=betas, min_threshold_wt=threshold_wt)
    full_compute, betas = networks.compute_centrality(N, dist, close_metrics=close_key, between_metrics=betw_key, min_threshold_wt=threshold_wt)
    assert np.array_equal(easy_dist, full_compute)
    assert np.array_equal(easy_betas, full_compute)


def test_harmonic_closeness():

    wrapper_for_centralities(networks.harmonic_closeness, ['harmonic'], None, angular=False)

    # test against networkX
    G, pos = mock.mock_graph()
    G = graphs.networkX_simple_geoms(G)
    G = graphs.networkX_edge_defaults(G)  # set default edge attributes
    N = networks.Network_Layer_From_NetworkX(G)
    # test first with single distance - use large distance because networkX does not have maximum distance parameter
    dist = [2000]
    harmonic_easy = networks.harmonic_closeness(N, dist)
    nx_harm_cl = nx.harmonic_centrality(G, distance='impedance')
    nx_harm_cl = np.array([v for v in nx_harm_cl.values()])
    assert np.array_equal(nx_harm_cl.round(8), harmonic_easy.round(8))


def test_gravity():
    wrapper_for_gravity_centralities(networks.gravity, ['gravity'], None)


def test_angular_harmonic_closeness():
    wrapper_for_centralities(networks.angular_harmonic_closeness, ['harmonic'], None, angular=True)


def test_betweenness():

    wrapper_for_centralities(networks.betweenness, None, ['betweenness'], angular=False)

    # test against networkX
    G, pos = mock.mock_graph()
    G = graphs.networkX_simple_geoms(G)
    G = graphs.networkX_edge_defaults(G)
    N = networks.Network_Layer_From_NetworkX(G)
    # test first with single distance - use large distance because networkX does not have maximum distance parameter
    dist = [2000]
    betweenness_easy = networks.betweenness(N, dist)
    nx_betw = nx.betweenness_centrality(G, weight='impedance', endpoints=False, normalized=False)
    nx_betw = np.array([v for v in nx_betw.values()])
    assert np.array_equal(nx_betw, betweenness_easy)


def test_betweenness_gravity():
    wrapper_for_gravity_centralities(networks.betweenness_gravity, None, ['betweenness_gravity'])


def test_angular_betweenness():
    wrapper_for_centralities(networks.angular_betweenness, None, ['betweenness'], angular=True)
