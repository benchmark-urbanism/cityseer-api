from itertools import permutations

import networkx as nx
import numpy as np
import pytest

from cityseer.algos import data
from cityseer.metrics import networks
from cityseer.util import mock, graphs


def test_distance_from_beta():
    assert networks.distance_from_beta(-0.04) == np.array([100.])
    assert networks.distance_from_beta(-0.0025) == np.array([1600.])

    arr = networks.distance_from_beta(-0.04, min_threshold_wt=0.001)
    assert np.array_equal(arr, np.array([172.69388197455342]))

    arr = networks.distance_from_beta([-0.04, -0.0025])
    assert np.array_equal(arr, np.array([100, 1600]))

    with pytest.raises(ValueError):
        networks.distance_from_beta(0.04)


def test_Network_Layer():
    G, pos = mock.mock_graph()
    G = graphs.networkX_simple_geoms(G)
    G = graphs.networkX_edge_defaults(G)
    node_uids, node_map, edge_map = graphs.graph_maps_from_networkX(G)
    x_arr = node_map[:, 0]
    y_arr = node_map[:, 1]
    index = data.generate_index(x_arr, y_arr)
    distances = [100, 200]
    min_thresh_wt = 0.018
    angular = False

    # test against Network_Layer's internal process
    N = networks.Network_Layer(node_uids, node_map, edge_map, distances, betas=None, min_threshold_wt=min_thresh_wt,
                               angular=angular)
    assert N.uids == node_uids
    assert np.array_equal(N.nodes, node_map)
    assert np.array_equal(N.edges, edge_map)
    assert np.array_equal(N.index, index)
    assert np.array_equal(N.x_arr, x_arr)
    assert np.array_equal(N.y_arr, y_arr)
    assert np.array_equal(N.live, node_map[:, 2])
    assert np.array_equal(N.edge_lengths, edge_map[:, 2])
    assert np.array_equal(N.edge_impedances, edge_map[:, 3])
    assert N.min_threshold_wt == min_thresh_wt
    assert N.betas == [np.log(min_thresh_wt) / d for d in distances]
    assert N.angular == angular

    G_round_trip = N.to_networkX()
    # graph_maps_from_networkX generates implicit live (all True) and weight (all 1) attributes if missing
    for n, d in G.nodes(data=True):
        assert n in G_round_trip
        assert G.nodes[n]['x'] == d['x']
        assert G.nodes[n]['y'] == d['y']
    assert G_round_trip.edges == G.edges

    # check for malformed signatures
    with pytest.raises(ValueError):
        networks.Network_Layer(node_uids[:-1], node_map, edge_map, distances)
    with pytest.raises(ValueError):
        networks.Network_Layer(node_uids, node_map[:, :-1], edge_map, distances)
    with pytest.raises(ValueError):
        networks.Network_Layer(node_uids, node_map, edge_map[:, :-1], distances)
    with pytest.raises(ValueError):
        networks.Network_Layer(node_uids, node_map, edge_map, [])
    with pytest.raises(ValueError):
        networks.Network_Layer(node_uids, node_map, edge_map, None, None)


def test_Network_Layer_From_NetworkX():
    G, pos = mock.mock_graph()
    G = graphs.networkX_simple_geoms(G)
    G = graphs.networkX_edge_defaults(G)
    node_uids, node_map, edge_map = graphs.graph_maps_from_networkX(G)
    x_arr = node_map[:, 0]
    y_arr = node_map[:, 1]
    live = node_map[:, 2]
    index = data.generate_index(x_arr, y_arr)
    distances = [100, 200]
    min_thresh_wt = 0.018
    angular = False

    # test against Network_Layer_From_NetworkX's internal process
    N = networks.Network_Layer_From_NetworkX(G, distances, betas=None, min_threshold_wt=min_thresh_wt, angular=angular)
    assert N.uids == node_uids
    assert np.array_equal(N.nodes, node_map)
    assert np.array_equal(N.edges, edge_map)
    assert np.array_equal(N.index, index)
    assert np.array_equal(N.x_arr, x_arr)
    assert np.array_equal(N.y_arr, y_arr)
    assert np.array_equal(N.live, live)
    assert N.min_threshold_wt == min_thresh_wt
    assert N.betas == [np.log(min_thresh_wt) / d for d in distances]
    assert N.angular == angular

    # check for malformed signatures
    with pytest.raises(ValueError):
        networks.Network_Layer_From_NetworkX(G, [])
    with pytest.raises(ValueError):
        networks.Network_Layer_From_NetworkX(G, None, None)


def test_to_networkX():
    # also see test_graphs.test_networkX_from_graph_maps for plain graph maps version

    # check round trip to and from graph maps results in same graph
    G, pos = mock.mock_graph()
    G = graphs.networkX_simple_geoms(G)
    G = graphs.networkX_edge_defaults(G)
    # explicitly set live and weight params for equality checks
    # graph_maps_from_networkX generates these implicitly if missing
    for n in G.nodes():
        G.nodes[n]['live'] = bool(np.random.randint(0, 1))
        G.nodes[n]['weight'] = np.random.random() * 2000

    # add random data to check persistence at other end
    G.nodes[0]['boo'] = 'baa'
    G[0][1]['baa'] = 'boo'

    # test with metrics
    N = networks.Network_Layer_From_NetworkX(G, distances=[500])
    N.harmonic_closeness()
    N.betweenness()
    G_round_trip = N.to_networkX()
    for n, d in G.nodes(data=True):
        assert d['x'] == G_round_trip.nodes[n]['x']
        assert d['y'] == G_round_trip.nodes[n]['y']
        assert d['live'] == G_round_trip.nodes[n]['live']
        assert d['weight'] == G_round_trip.nodes[n]['weight']
    for s, e, d in G.edges(data=True):
        assert d['geom'] == G_round_trip[s][e]['geom']
        assert d['length'] == G_round_trip[s][e]['length']
        assert d['impedance'] == G_round_trip[s][e]['impedance']

    for metric_key in N.metrics.keys():
        for measure_key in N.metrics[metric_key].keys():
            for dist in N.metrics[metric_key][measure_key].keys():
                for idx, uid in enumerate(N.uids):
                    assert N.metrics[metric_key][measure_key][dist][idx] == \
                           G_round_trip.nodes[uid]['metrics'][metric_key][measure_key][dist]

    assert G_round_trip.nodes[0]['boo'] == 'baa'
    assert G_round_trip[0][1]['baa'] == 'boo'


def test_metrics_to_dict():
    G, pos = mock.mock_graph()
    G = graphs.networkX_simple_geoms(G)
    G = graphs.networkX_edge_defaults(G)
    N = networks.Network_Layer_From_NetworkX(G, distances=[500])
    N.harmonic_closeness()
    N.betweenness()
    metrics_dict = N.metrics_to_dict()

    for idx, uid in enumerate(N.uids):
        assert metrics_dict[uid]['x'] == N.nodes[idx][0]
        assert metrics_dict[uid]['y'] == N.nodes[idx][1]
        assert metrics_dict[uid]['live'] == (N.nodes[idx][2] == 1)
        assert metrics_dict[uid]['weight'] == N.nodes[idx][4]
        for metric_key in N.metrics.keys():
            for measure_key in N.metrics[metric_key].keys():
                print(measure_key)
                for dist in N.metrics[metric_key][measure_key].keys():
                    print(N.metrics[metric_key][measure_key][dist][idx],
                          metrics_dict[uid][metric_key][measure_key][dist])
                    assert N.metrics[metric_key][measure_key][dist][idx] == \
                           metrics_dict[uid][metric_key][measure_key][dist]


def test_compute_centrality():
    '''
    Underlying method also tested via test_networks.test_network_centralities
    '''

    G, pos = mock.mock_graph()
    G = graphs.networkX_simple_geoms(G)
    G = graphs.networkX_edge_defaults(G)

    distances = [100]
    N = networks.Network_Layer_From_NetworkX(G, distances)

    # check that malformed signatures trigger errors
    with pytest.raises(ValueError):
        N.compute_centrality()
    with pytest.raises(ValueError):
        N.compute_centrality(close_metrics=[])
    with pytest.raises(ValueError):
        N.compute_centrality(close_metrics=['node_density_spelling_typo'])

    # check single closeness and betweenness independently
    N.compute_centrality(close_metrics=['node_density'])
    assert len(N.metrics['centrality']) == 1
    # this is a reset hack for testing - less computationally intensive
    N.metrics['centrality'] = {}
    N.compute_centrality(between_metrics=['betweenness'])
    assert len(N.metrics['centrality']) == 1

    # also check the number of returned types for all permutations
    closeness_types = ['node_density', 'farness_impedance', 'farness_distance', 'harmonic', 'improved', 'gravity',
                       'cycles']
    betweenness_types = ['betweenness', 'betweenness_gravity']
    N = networks.Network_Layer_From_NetworkX(G, distances)
    for cl_types in permutations(closeness_types):
        for bt_types in permutations(betweenness_types):
            # this is a reset hack for testing - less computationally intensive
            N.metrics['centrality'] = {}
            N.compute_centrality(close_metrics=list(cl_types), between_metrics=list(bt_types))
            # the number of items in the dictionary should match the sum of closeness and betweenness metrics
            assert len(N.metrics['centrality']) == len(cl_types) + len(bt_types)

    # test against networkX
    distances = [2000]
    N = networks.Network_Layer_From_NetworkX(G, distances)
    N.compute_centrality(close_metrics=['node_density', 'harmonic'], between_metrics=['betweenness'])

    # test node density
    # node density count doesn't include self-node
    node_density_2000 = N.metrics['centrality']['node_density'][2000]
    for n in node_density_2000:
        assert n + 1 == len(G)

    # NOTE: modified improved closeness is not comparable to networkx version

    # test harmonic closeness vs networkX version
    nx_harm_cl = nx.harmonic_centrality(G, distance='impedance')
    nx_harm_cl = np.array([v for v in nx_harm_cl.values()])
    # compare
    harmonic_2000 = N.metrics['centrality']['harmonic'][2000]
    assert np.allclose(nx_harm_cl, harmonic_2000)

    # test betweenness vs networkX version
    # set endpoint counting to false and do not normalise
    nx_betw = nx.betweenness_centrality(G, weight='impedance', endpoints=False, normalized=False)
    nx_betw = np.array([v for v in nx_betw.values()])
    # compare
    betweenness_2000 = N.metrics['centrality']['betweenness'][2000]
    assert np.allclose(nx_betw, betweenness_2000)


def network_generator():
    for betas in [[-0.008], [-0.008, -0.002]]:
        distances = networks.distance_from_beta(betas)
        for angular in [False, True]:
            G, pos = mock.mock_graph()
            G = graphs.networkX_simple_geoms(G)
            G = graphs.networkX_edge_defaults(G)
            yield G, distances, betas, angular


def test_harmonic_closeness():
    for G, distances, betas, angular in network_generator():

        # easy version
        N_easy = networks.Network_Layer_From_NetworkX(G, distances, angular=angular)
        N_easy.harmonic_closeness()
        # custom version
        N_full = networks.Network_Layer_From_NetworkX(G, distances, angular=angular)
        N_full.compute_centrality(close_metrics=['harmonic'])

        # compare
        for d in distances:
            harmonic_easy = N_easy.metrics['centrality']['harmonic'][d]
            harmonic_full = N_full.metrics['centrality']['harmonic'][d]
            assert np.allclose(harmonic_easy, harmonic_full)


def test_gravity():
    for G, distances, betas, angular in network_generator():

        # DISTANCES
        # easy version
        N_easy = networks.Network_Layer_From_NetworkX(G, distances=distances, angular=angular)
        N_easy.gravity()
        # easy version via betas
        N_easy_betas = networks.Network_Layer_From_NetworkX(G, betas=betas, angular=angular)
        N_easy_betas.gravity()
        # custom version
        N_full = networks.Network_Layer_From_NetworkX(G, distances=distances, angular=angular)
        N_full.compute_centrality(close_metrics=['gravity'])

        # compare
        for d in distances:
            gravity_easy = N_easy.metrics['centrality']['gravity'][d]
            gravity_easy_betas = N_easy_betas.metrics['centrality']['gravity'][d]
            gravity_full = N_full.metrics['centrality']['gravity'][d]
            assert np.allclose(gravity_easy, gravity_full)
            assert np.allclose(gravity_easy_betas, gravity_full)


def test_betweenness():
    for G, distances, betas, angular in network_generator():

        # easy version
        N_easy = networks.Network_Layer_From_NetworkX(G, distances, angular=angular)
        N_easy.betweenness()
        # custom version
        N_full = networks.Network_Layer_From_NetworkX(G, distances, angular=angular)
        N_full.compute_centrality(between_metrics=['betweenness'])

        # compare
        for d in distances:
            between_easy = N_easy.metrics['centrality']['betweenness'][d]
            between_full = N_full.metrics['centrality']['betweenness'][d]
            assert np.allclose(between_easy, between_full)


def test_betweenness_gravity():
    for G, distances, betas, angular in network_generator():

        # DISTANCES
        # easy version
        N_easy = networks.Network_Layer_From_NetworkX(G, distances=distances, angular=angular)
        N_easy.betweenness_gravity()
        # easy version via betas
        N_easy_betas = networks.Network_Layer_From_NetworkX(G, betas=betas, angular=angular)
        N_easy_betas.betweenness_gravity()
        # custom version
        N_full = networks.Network_Layer_From_NetworkX(G, distances=distances, angular=angular)
        N_full.compute_centrality(between_metrics=['betweenness_gravity'])

        # compare
        for d in distances:
            between_gravity_easy = N_easy.metrics['centrality']['betweenness_gravity'][d]
            between_gravity_easy_betas = N_easy_betas.metrics['centrality']['betweenness_gravity'][d]
            between_gravity_full = N_full.metrics['centrality']['betweenness_gravity'][d]
            assert np.allclose(between_gravity_easy, between_gravity_full)
            assert np.allclose(between_gravity_easy_betas, between_gravity_full)
