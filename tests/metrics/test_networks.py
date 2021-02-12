import numpy as np
import pytest

from cityseer.algos import checks, centrality
from cityseer.metrics import networks, layers
from cityseer.util import mock, graphs
from cityseer.util.mock import primal_graph


def test_distance_from_beta():
    # some basic checks - using float form
    for b, d in zip([-0.04, -0.0025, -0.0], [100, 1600, np.inf]):
        # simple straight check against corresponding distance
        assert networks.distance_from_beta(b) == np.array([d])
        # circular check
        assert networks.beta_from_distance(networks.distance_from_beta(b)) == b
        # array form check
        assert networks.distance_from_beta(np.array([b])) == np.array([d])
    # check that custom min_threshold_wt works
    arr = networks.distance_from_beta(-0.04, min_threshold_wt=0.001)
    assert np.allclose(arr, np.array([172.69388197455342]), atol=0.001, rtol=0)
    # check on array form
    arr = networks.distance_from_beta([-0.04, -0.0025, -0.0])
    assert np.allclose(arr, np.array([100, 1600, np.inf]), atol=0.001, rtol=0)
    # check for type error
    with pytest.raises(TypeError):
        networks.distance_from_beta('boo')
    # check that invalid beta values raise an error
    for b in [0.04, 0, -0]:
        with pytest.raises(ValueError):
            networks.distance_from_beta(b)


def test_beta_from_distance():
    # some basic checks
    for d, b in zip([100, 1600, np.inf], [-0.04, -0.0025, -0.0]):
        # simple straight check against corresponding distance
        assert networks.beta_from_distance(d) == np.array([b])
        # circular check
        assert networks.distance_from_beta(networks.beta_from_distance(d)) == d
        # array form check
        assert networks.beta_from_distance(np.array([d])) == np.array([b])
    # check that custom min_threshold_wt works
    arr = networks.beta_from_distance(172.69388197455342, min_threshold_wt=0.001)
    assert np.allclose(arr, np.array([-0.04]), atol=0.001, rtol=0)
    # check on array form
    arr = networks.beta_from_distance([100, 1600, np.inf])
    assert np.allclose(arr, np.array([-0.04, -0.0025, -0.0]), atol=0.001, rtol=0)
    # check for type error
    with pytest.raises(TypeError):
        networks.beta_from_distance('boo')
    # check that invalid beta values raise an error
    for d in [-100, 0]:
        with pytest.raises(ValueError):
            networks.beta_from_distance(d)


def test_Network_Layer(primal_graph):
    # manual graph maps for comparison
    node_uids, node_data, edge_data, node_edge_map = graphs.graph_maps_from_nX(primal_graph)
    x_arr = node_data[:, 0]
    y_arr = node_data[:, 1]
    betas = [-0.02, -0.005]
    distances = networks.distance_from_beta(betas)

    # test Network_Layer's class
    for d, b in zip([distances, None], [None, betas]):
        for angular in [True, False]:
            N = networks.Network_Layer(node_uids,
                                       node_data,
                                       edge_data,
                                       node_edge_map,
                                       distances=d,
                                       betas=b)
            assert np.allclose(N.uids, node_uids, atol=0.001, rtol=0)
            assert np.allclose(N._node_data, node_data, atol=0.001, rtol=0)
            assert np.allclose(N._edge_data, edge_data, atol=0.001, rtol=0)
            assert np.allclose(N.distances, distances, atol=0.001,
                               rtol=0)  # inferred automatically when only betas provided
            assert np.allclose(N.betas, betas, atol=0.001,
                               rtol=0)  # inferred automatically when only distances provided
            assert N.min_threshold_wt == checks.def_min_thresh_wt
            assert np.allclose(N.x_arr, x_arr, atol=0.001, rtol=0)
            assert np.allclose(N.y_arr, y_arr, atol=0.001, rtol=0)
            assert np.allclose(N.live, node_data[:, 2], atol=0.001, rtol=0)
            assert np.allclose(N.edge_lengths, edge_data[:, 2], atol=0.001, rtol=0)
            assert np.allclose(N.edge_angles, edge_data[:, 3], atol=0.001, rtol=0)
            assert np.allclose(N.edge_impedance_factor, edge_data[:, 4], atol=0.001, rtol=0)
            assert np.allclose(N.edge_in_bearing, edge_data[:, 5], atol=0.001, rtol=0)
            assert np.allclose(N.edge_out_bearing, edge_data[:, 6], atol=0.001, rtol=0)

    # test round-trip graph to and from Network_Layer
    N = networks.Network_Layer(node_uids,
                               node_data,
                               edge_data,
                               node_edge_map,
                               distances=distances)
    G_round_trip = N.to_networkX()
    # graph_maps_from_networkX generates implicit live (all True) and weight (all 1) attributes if missing
    # i.e. can't simply check that all nodes equal, so check properties manually
    for n, d in primal_graph.nodes(data=True):
        assert n in G_round_trip
        assert G_round_trip.nodes[n]['x'] == d['x']
        assert G_round_trip.nodes[n]['y'] == d['y']
    # edges can be checked en masse
    assert G_round_trip.edges == primal_graph.edges
    # check alternate min_threshold_wt gets passed through successfully
    alt_min = 0.02
    alt_distances = networks.distance_from_beta(betas, min_threshold_wt=alt_min)
    N = networks.Network_Layer(node_uids,
                               node_data,
                               edge_data,
                               node_edge_map,
                               betas=betas,
                               min_threshold_wt=alt_min)
    assert np.allclose(N.distances, alt_distances, atol=0.001, rtol=0)
    # check for malformed signatures
    with pytest.raises(ValueError):
        networks.Network_Layer(node_uids[:-1],
                               node_data,
                               edge_data,
                               node_edge_map,
                               distances)
    with pytest.raises(ValueError):
        networks.Network_Layer(node_uids,
                               node_data[:, :-1],
                               edge_data,
                               node_edge_map,
                               distances)
    with pytest.raises(ValueError):
        networks.Network_Layer(node_uids,
                               node_data,
                               edge_data[:, :-1],
                               node_edge_map,
                               distances)
    with pytest.raises(ValueError):
        networks.Network_Layer(node_uids,
                               node_data,
                               edge_data[:, :-1],
                               node_edge_map,
                               distances)
    with pytest.raises(ValueError):
        networks.Network_Layer(node_uids,
                               node_data,
                               edge_data,
                               node_edge_map)  # no betas or distances
    with pytest.raises(ValueError):
        networks.Network_Layer(node_uids,
                               node_data,
                               edge_data,
                               node_edge_map,
                               distances=None,
                               betas=None)
    with pytest.raises(ValueError):
        networks.Network_Layer(node_uids,
                               node_data,
                               edge_data,
                               node_edge_map,
                               distances=[])
    with pytest.raises(ValueError):
        networks.Network_Layer(node_uids,
                               node_data,
                               edge_data,
                               node_edge_map,
                               betas=[])


def test_Network_Layer_From_nX(primal_graph):
    node_uids, node_data, edge_data, node_edge_map = graphs.graph_maps_from_nX(primal_graph)
    x_arr = node_data[:, 0]
    y_arr = node_data[:, 1]
    betas = np.array([-0.04, -0.02])
    distances = networks.distance_from_beta(betas)

    # test Network_Layer_From_NetworkX's class
    for d, b in zip([distances, None], [None, betas]):
        for angular in [True, False]:
            N = networks.Network_Layer_From_nX(primal_graph, distances=d, betas=b)
            assert np.allclose(N.uids, node_uids, atol=0.001, rtol=0)
            assert np.allclose(N._node_data, node_data, atol=0.001, rtol=0)
            assert np.allclose(N._edge_data, edge_data, atol=0.001, rtol=0)
            assert np.allclose(N.distances, distances, atol=0.001,
                               rtol=0)  # inferred automatically when only betas provided
            assert np.allclose(N.betas, betas, atol=0.001,
                               rtol=0)  # inferred automatically when only distances provided
            assert N.min_threshold_wt == checks.def_min_thresh_wt
            assert np.allclose(N.x_arr, x_arr, atol=0.001, rtol=0)
            assert np.allclose(N.y_arr, y_arr, atol=0.001, rtol=0)
            assert np.allclose(N.live, node_data[:, 2], atol=0.001, rtol=0)
            assert np.allclose(N.edge_lengths, edge_data[:, 2], atol=0.001, rtol=0)
            assert np.allclose(N.edge_angles, edge_data[:, 3], atol=0.001, rtol=0)
            assert np.allclose(N.edge_impedance_factor, edge_data[:, 4], atol=0.001, rtol=0)
            assert np.allclose(N.edge_in_bearing, edge_data[:, 5], atol=0.001, rtol=0)
            assert np.allclose(N.edge_out_bearing, edge_data[:, 6], atol=0.001, rtol=0)

    # check alternate min_threshold_wt gets passed through successfully
    alt_min = 0.02
    alt_distances = networks.distance_from_beta(betas, min_threshold_wt=alt_min)
    N = networks.Network_Layer_From_nX(primal_graph, betas=betas, min_threshold_wt=alt_min)
    assert np.allclose(N.distances, alt_distances, atol=0.001, rtol=0)

    # check for malformed signatures
    with pytest.raises(TypeError):
        networks.Network_Layer_From_nX('boo', distances=distances)
    with pytest.raises(ValueError):
        networks.Network_Layer_From_nX(primal_graph)  # no betas or distances
    with pytest.raises(ValueError):
        networks.Network_Layer_From_nX(primal_graph, distances=None, betas=None)
    with pytest.raises(ValueError):
        networks.Network_Layer_From_nX(primal_graph, distances=[])
    with pytest.raises(ValueError):
        networks.Network_Layer_From_nX(primal_graph, betas=[])


def dict_check(m_dict, Network):
    for i, uid in enumerate(Network.uids):
        assert m_dict[uid]['x'] == Network._node_data[i][0]
        assert m_dict[uid]['y'] == Network._node_data[i][1]
        assert m_dict[uid]['live'] == Network._node_data[i][2]

        for c_key, c_val in Network.metrics['centrality'].items():
            for d_key, d_val in c_val.items():
                assert d_val[i] == m_dict[uid]['centrality'][c_key][d_key]

        for mu_key, mu_val in Network.metrics['mixed_uses'].items():
            if 'hill' in mu_key:
                for q_key, q_val in mu_val.items():
                    for d_key, d_val in q_val.items():
                        assert d_val[i] == m_dict[uid]['mixed_uses'][mu_key][q_key][d_key]
            else:
                for d_key, d_val in mu_val.items():
                    assert d_val[i] == m_dict[uid]['mixed_uses'][mu_key][d_key]

        for cat in ['non_weighted', 'weighted']:
            if cat not in Network.metrics['accessibility']:
                continue
            for cl_key, cl_val in Network.metrics['accessibility'][cat].items():
                for d_key, d_val in cl_val.items():
                    assert d_val[i] == m_dict[uid]['accessibility'][cat][cl_key][d_key]

        for th_key, th_val in Network.metrics['stats'].items():
            for stat_key, stat_val in th_val.items():
                for d_key, d_val in stat_val.items():
                    # some NaN so use np.allclose
                    assert np.allclose(d_val[i], m_dict[uid]['stats'][th_key][stat_key][d_key], equal_nan=True,
                                       atol=0.001, rtol=0)


def test_metrics_to_dict(primal_graph):
    # create a network layer and run some metrics
    N = networks.Network_Layer_From_nX(primal_graph, distances=[500, 1000])

    # check with no metrics
    metrics_dict = N.metrics_to_dict()
    dict_check(metrics_dict, N)

    # check with centrality metrics
    N.compute_node_centrality(measures=['node_harmonic'])
    metrics_dict = N.metrics_to_dict()
    dict_check(metrics_dict, N)

    # check with data metrics
    data_dict = mock.mock_data_dict(primal_graph)
    landuse_labels = mock.mock_categorical_data(len(data_dict))
    numerical_data = mock.mock_numerical_data(len(data_dict))
    # TODO:
    '''
    D = layers.Data_Layer_From_Dict(data_dict)
    D.assign_to_network(N, max_dist=400)
    D.compute_aggregated(landuse_labels,
                         mixed_use_keys=['hill', 'shannon'],
                         accessibility_keys=['a', 'c'],
                         qs=[0, 1],
                         stats_keys=['boo'],
                         stats_data_arrs=numerical_data)
    '''
    metrics_dict = N.metrics_to_dict()
    dict_check(metrics_dict, N)


def test_to_networkX(primal_graph):
    # also see test_graphs.test_networkX_from_graph_maps for underlying graph maps version

    # check round trip to and from graph maps results in same graph
    # explicitly set live and weight params for equality checks
    # graph_maps_from_networkX generates these implicitly if missing
    G = graphs.nX_decompose(primal_graph, decompose_max=20)
    for n in G.nodes():
        G.nodes[n]['live'] = bool(np.random.randint(0, 1))
    for s, e, k in G.edges(keys=True):
        G[s][e][k]['imp_factor'] = np.random.randint(0, 2)

    # add random data to check persistence at other end
    baa_node = None
    for n in G.nodes():
        baa_node = n
        G.nodes[n]['boo'] = 'baa'
        break
    boo_edge = None
    for s, e, k in G.edges(keys=True):
        boo_edge = (s, e)
        G[s][e][k]['baa'] = 'boo'
        break

    # test with metrics
    N = networks.Network_Layer_From_nX(G, distances=[500])
    N.compute_node_centrality(measures=['node_harmonic'])
    metrics_dict = N.metrics_to_dict()
    G_round_trip = N.to_networkX()
    for n, d in G.nodes(data=True):
        assert G_round_trip.nodes[n]['x'] == d['x']
        assert G_round_trip.nodes[n]['y'] == d['y']
        assert G_round_trip.nodes[n]['live'] == d['live']
    for s, e, k, d in G.edges(keys=True, data=True):
        assert G_round_trip[s][e][k]['geom'] == d['geom']
        assert G_round_trip[s][e][k]['imp_factor'] == d['imp_factor']
    # check that metrics came through
    for uid, metrics in metrics_dict.items():
        assert G_round_trip.nodes[uid]['metrics'] == metrics
    # check data persistence
    assert G_round_trip.nodes[baa_node]['boo'] == 'baa'
    assert G_round_trip[boo_edge[0]][boo_edge[1]][0]['baa'] == 'boo'


def test_compute_centrality(primal_graph):
    '''
    Underlying methods also tested via test_networks.test_network_centralities
    '''
    betas = np.array([-0.01, -0.005])
    distances = networks.distance_from_beta(betas)
    # generate data structures
    N = networks.Network_Layer_From_nX(primal_graph, distances=distances)
    node_data = N._node_data
    edge_data = N._edge_data
    node_edge_map = N._node_edge_map

    # CHECK NODE BASED
    node_measures = ['node_density',
                     'node_farness',
                     'node_cycles',
                     'node_harmonic',
                     'node_beta',
                     'node_betweenness',
                     'node_betweenness_beta']
    node_measures_ang = ['node_harmonic_angular',
                         'node_betweenness_angular']

    # check measures against underlying method
    N = networks.Network_Layer_From_nX(primal_graph, distances=distances)
    N.compute_node_centrality(measures=['node_density'])
    # test against underlying method
    measures_data = centrality.local_node_centrality(node_data,
                                                     edge_data,
                                                     node_edge_map,
                                                     distances,
                                                     betas,
                                                     measure_keys=('node_density',))
    for d_idx, d_key in enumerate(distances):
        assert np.allclose(N.metrics['centrality']['node_density'][d_key], measures_data[0][d_idx])
    # also check the number of returned types for a few assortments of metrics
    np.random.shuffle(node_measures)  # in place
    # not necessary to do all labels, first few should do
    for min_idx in range(3):
        measure_keys = np.array(node_measures[min_idx:])
        N = networks.Network_Layer_From_nX(primal_graph, distances=distances)
        N.compute_node_centrality(measures=node_measures)
        # test against underlying method
        measures_data = centrality.local_node_centrality(node_data,
                                                         edge_data,
                                                         node_edge_map,
                                                         distances,
                                                         betas,
                                                         measure_keys=tuple(measure_keys))
        for m_idx, measure_name in enumerate(measure_keys):
            for d_idx, d_key in enumerate(distances):
                assert np.allclose(N.metrics['centrality'][measure_name][d_key],
                                   measures_data[m_idx][d_idx], atol=0.001, rtol=0)
    # check that angular gets passed through
    N_ang = networks.Network_Layer_From_nX(primal_graph, distances=[2000])
    N_ang.compute_node_centrality(measures=['node_harmonic_angular'], angular=True)
    N = networks.Network_Layer_From_nX(primal_graph, distances=[2000])
    N.compute_node_centrality(measures=['node_harmonic'], angular=False)
    assert not np.allclose(N_ang.metrics['centrality']['node_harmonic_angular'][2000],
                           N.metrics['centrality']['node_harmonic'][2000], atol=0.001, rtol=0)
    assert not np.allclose(N_ang.metrics['centrality']['node_harmonic_angular'][2000],
                           N.metrics['centrality']['node_harmonic'][2000], atol=0.001, rtol=0)
    # check that typos, duplicates, and mixed angular / non-angular are caught
    with pytest.raises(ValueError):
        N.compute_node_centrality(measures=['spelling_typo'])
    with pytest.raises(ValueError):
        N.compute_node_centrality(measures=['node_density', 'node_density'])
    with pytest.raises(ValueError):
        N.compute_node_centrality(measures=['node_density', 'node_harmonic_angular'])

    # CHECK SEGMENTISED
    segment_measures = ['segment_density',
                        'segment_harmonic',
                        'segment_beta',
                        'segment_betweenness']
    segment_measures_ang = ['segment_harmonic_hybrid',
                            'segment_betweeness_hybrid']

    # check measures against underlying method
    N = networks.Network_Layer_From_nX(primal_graph, distances=distances)
    N.compute_segment_centrality(measures=['segment_density'])
    # test against underlying method
    measures_data = centrality.local_segment_centrality(node_data,
                                                        edge_data,
                                                        node_edge_map,
                                                        distances,
                                                        betas,
                                                        measure_keys=('segment_density',))
    for d_idx, d_key in enumerate(distances):
        assert np.allclose(N.metrics['centrality']['segment_density'][d_key], measures_data[0][d_idx])
    # also check the number of returned types for a few assortments of metrics
    np.random.shuffle(segment_measures)  # in place
    # not necessary to do all labels, first few should do
    for min_idx in range(3):
        measure_keys = np.array(segment_measures[min_idx:])
        N = networks.Network_Layer_From_nX(primal_graph, distances=distances)
        N.compute_segment_centrality(measures=segment_measures)
        # test against underlying method
        measures_data = centrality.local_segment_centrality(node_data,
                                                            edge_data,
                                                            node_edge_map,
                                                            distances,
                                                            betas,
                                                            measure_keys=tuple(measure_keys))
        for m_idx, measure_name in enumerate(measure_keys):
            for d_idx, d_key in enumerate(distances):
                assert np.allclose(N.metrics['centrality'][measure_name][d_key],
                                   measures_data[m_idx][d_idx], atol=0.001, rtol=0)
    # check that angular gets passed through
    N_ang = networks.Network_Layer_From_nX(primal_graph, distances=[2000])
    N_ang.compute_segment_centrality(measures=['segment_harmonic_hybrid'], angular=True)
    N = networks.Network_Layer_From_nX(primal_graph, distances=[2000])
    N.compute_segment_centrality(measures=['segment_harmonic'], angular=False)
    assert not np.allclose(N_ang.metrics['centrality']['segment_harmonic_hybrid'][2000],
                           N.metrics['centrality']['segment_harmonic'][2000], atol=0.001, rtol=0)
    assert not np.allclose(N_ang.metrics['centrality']['segment_harmonic_hybrid'][2000],
                           N.metrics['centrality']['segment_harmonic'][2000], atol=0.001, rtol=0)
    # check that typos, duplicates, and mixed angular / non-angular are caught
    with pytest.raises(ValueError):
        N.compute_segment_centrality(measures=['spelling_typo'])
    with pytest.raises(ValueError):
        N.compute_segment_centrality(measures=['segment_density', 'segment_density'])
    with pytest.raises(ValueError):
        N.compute_segment_centrality(measures=['segment_density', 'segment_harmonic_hybrid'])

    # check that the deprecated method raises:
    with pytest.raises(DeprecationWarning):
        N.compute_centrality()


def network_generator(primal_graph):
    for betas in [[-0.008], [-0.008, -0.002, -0.0]]:
        distances = networks.distance_from_beta(betas)
        for angular in [False, True]:
            yield primal_graph, distances, betas, angular
