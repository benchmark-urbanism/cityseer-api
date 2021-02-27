import copy

import numpy as np
import pytest
import utm

from cityseer.algos import data
from cityseer.metrics import layers, networks
from cityseer.tools import mock, graphs
from cityseer.tools.mock import primal_graph


def test_dict_wgs_to_utm(primal_graph):
    # check that node coordinates are correctly converted
    G_utm = mock.mock_graph()
    data_dict_utm = mock.mock_data_dict(G_utm)

    # create a test dictionary
    test_dict = copy.deepcopy(data_dict_utm)
    # cast to lat, lon
    for k, v in test_dict.items():
        easting = v['x']
        northing = v['y']
        # be cognisant of parameter and return order
        # returns in lat lng order
        lat, lng = utm.to_latlon(easting, northing, 30, 'U')
        test_dict[k]['x'] = lng
        test_dict[k]['y'] = lat

    # convert back
    dict_converted = layers.dict_wgs_to_utm(test_dict)

    # check that round-trip converted match with reasonable proximity given rounding errors
    for k in data_dict_utm.keys():
        # rounding can be tricky
        assert np.allclose(data_dict_utm[k]['x'], dict_converted[k]['x'], atol=0.1, rtol=0)  # relax precision
        assert np.allclose(data_dict_utm[k]['y'], dict_converted[k]['y'], atol=0.1, rtol=0)  # relax precision

    # check that missing node attributes throw an error
    for attr in ['x', 'y']:
        G_wgs = mock.mock_graph(wgs84_coords=True)
        data_dict_wgs = mock.mock_data_dict(G_wgs)
        for k in data_dict_wgs.keys():
            del data_dict_wgs[k][attr]
            break
        # check that missing attribute throws an error
        with pytest.raises(AttributeError):
            layers.dict_wgs_to_utm(data_dict_wgs)

    # check that non WGS coordinates throw error
    with pytest.raises(AttributeError):
        layers.dict_wgs_to_utm(data_dict_utm)


def test_encode_categorical():
    # generate mock data
    mock_categorical = mock.mock_categorical_data(50)

    classes, class_encodings = layers.encode_categorical(mock_categorical)

    for cl in classes:
        assert cl in mock_categorical

    for idx, label in enumerate(mock_categorical):
        assert label in classes
        assert classes.index(label) == class_encodings[idx]


def test_data_map_from_dict(primal_graph):
    # generate mock data
    data_dict = mock.mock_data_dict(primal_graph)
    data_uids, data_map = layers.data_map_from_dict(data_dict)

    assert len(data_uids) == len(data_map) == len(data_dict)

    for d_label, d in zip(data_uids, data_map):
        assert d[0] == data_dict[d_label]['x']
        assert d[1] == data_dict[d_label]['y']
        assert np.isnan(d[2])
        assert np.isnan(d[3])

    # check that missing attributes throw errors
    for attr in ['x', 'y']:
        for k in data_dict.keys():
            del data_dict[k][attr]
        with pytest.raises(AttributeError):
            layers.data_map_from_dict(data_dict)


def test_Data_Layer(primal_graph):
    data_dict = mock.mock_data_dict(primal_graph)
    data_uids, data_map = layers.data_map_from_dict(data_dict)
    x_arr = data_map[:, 0]
    y_arr = data_map[:, 1]

    # test against DataLayer internal process
    D = layers.DataLayer(data_uids, data_map)
    assert D.uids == data_uids
    assert np.allclose(D._data, data_map, equal_nan=True, atol=0.001, rtol=0)
    assert np.allclose(D.data_x_arr, x_arr, atol=0.001, rtol=0)
    assert np.allclose(D.data_y_arr, y_arr, atol=0.001, rtol=0)


def test_Data_Layer_From_Dict(primal_graph):
    data_dict = mock.mock_data_dict(primal_graph)
    data_uids, data_map = layers.data_map_from_dict(data_dict)
    x_arr = data_map[:, 0]
    y_arr = data_map[:, 1]

    # test against DataLayerFromDict's internal process
    D = layers.DataLayerFromDict(data_dict)
    assert D.uids == data_uids
    assert np.allclose(D._data, data_map, equal_nan=True)
    assert np.allclose(D.data_x_arr, x_arr, atol=0.001, rtol=0)
    assert np.allclose(D.data_y_arr, y_arr, atol=0.001, rtol=0)


def test_compute_aggregated_A(primal_graph):
    betas = np.array([-0.01, -0.005])
    distances = networks.distance_from_beta(betas)
    # network layer
    N = networks.NetworkLayerFromNX(primal_graph, distances=distances)
    node_map = N._node_data
    edge_map = N._edge_data
    node_edge_map = N._node_edge_map
    # data layer
    data_dict = mock.mock_data_dict(primal_graph)
    qs = np.array([0, 1, 2])
    D = layers.DataLayerFromDict(data_dict)
    # check single metrics independently against underlying for some use-cases, e.g. hill, non-hill, accessibility...
    D.assign_to_network(N, max_dist=500)
    # generate some mock landuse data
    landuse_labels = mock.mock_categorical_data(len(data_dict))
    landuse_classes, landuse_encodings = layers.encode_categorical(landuse_labels)
    # compute hill mixed uses
    D.compute_aggregated(landuse_labels, mixed_use_keys=['hill_branch_wt'], qs=qs)
    # test against underlying method
    data_map = D._data
    mu_data_hill, mu_data_other, ac_data, ac_data_wt, \
    stats_sum, stats_sum_wt, stats_mean, stats_mean_wt, stats_variance, stats_variance_wt, stats_max, stats_min = \
        data.local_aggregator(node_map,
                              edge_map,
                              node_edge_map,
                              data_map,
                              distances,
                              betas,
                              landuse_encodings,
                              qs=qs,
                              mixed_use_hill_keys=np.array([1]))
    for q_idx, q_key in enumerate(qs):
        for d_idx, d_key in enumerate(distances):
            assert np.allclose(N.metrics['mixed_uses']['hill_branch_wt'][q_key][d_key],
                               mu_data_hill[0][q_idx][d_idx], atol=0.001, rtol=0)
    # gini simpson
    D.compute_aggregated(landuse_labels, mixed_use_keys=['gini_simpson'])
    # test against underlying method
    data_map = D._data
    mu_data_hill, mu_data_other, ac_data, ac_data_wt, \
    stats_sum, stats_sum_wt, stats_mean, stats_mean_wt, stats_variance, stats_variance_wt, stats_max, stats_min = \
        data.local_aggregator(node_map,
                              edge_map,
                              node_edge_map,
                              data_map,
                              distances,
                              betas,
                              landuse_encodings,
                              mixed_use_other_keys=np.array([1]))
    for d_idx, d_key in enumerate(distances):
        assert np.allclose(N.metrics['mixed_uses']['gini_simpson'][d_key], mu_data_other[0][d_idx], atol=0.001, rtol=0)
    # accessibilities
    D.compute_aggregated(landuse_labels, accessibility_keys=['c'])
    # test against underlying method
    data_map = D._data
    mu_data_hill, mu_data_other, ac_data, ac_data_wt, \
    stats_sum, stats_sum_wt, stats_mean, stats_mean_wt, stats_variance, stats_variance_wt, stats_max, stats_min = \
        data.local_aggregator(node_map,
                              edge_map,
                              node_edge_map,
                              data_map,
                              distances,
                              betas,
                              landuse_encodings,
                              accessibility_keys=np.array([landuse_classes.index('c')]))
    for d_idx, d_key in enumerate(distances):
        assert np.allclose(N.metrics['accessibility']['non_weighted']['c'][d_key], ac_data[0][d_idx], atol=0.001,
                           rtol=0)
        assert np.allclose(N.metrics['accessibility']['weighted']['c'][d_key], ac_data_wt[0][d_idx], atol=0.001, rtol=0)
    # also check the number of returned types for a few assortments of metrics
    mixed_uses_hill_types = np.array(['hill',
                                      'hill_branch_wt',
                                      'hill_pairwise_wt',
                                      'hill_pairwise_disparity'])
    mixed_use_other_types = np.array(['shannon',
                                      'gini_simpson',
                                      'raos_pairwise_disparity'])
    ac_codes = np.array(landuse_classes)

    mu_hill_random = np.arange(len(mixed_uses_hill_types))
    np.random.shuffle(mu_hill_random)

    mu_other_random = np.arange(len(mixed_use_other_types))
    np.random.shuffle(mu_other_random)

    ac_random = np.arange(len(landuse_classes))
    np.random.shuffle(ac_random)

    # mock disparity matrix
    mock_disparity_wt_matrix = np.full((len(landuse_classes), len(landuse_classes)), 1)

    # not necessary to do all labels, first few should do
    for mu_h_min in range(3):
        mu_h_keys = np.array(mu_hill_random[mu_h_min:])

        for mu_o_min in range(3):
            mu_o_keys = np.array(mu_other_random[mu_o_min:])

            for ac_min in range(3):
                ac_keys = np.array(ac_random[ac_min:])

                # in the final case, set accessibility to a single code otherwise an error would be raised
                if len(mu_h_keys) == 0 and len(mu_o_keys) == 0 and len(ac_keys) == 0:
                    ac_keys = np.array([0])

                # randomise order of keys and metrics
                mu_h_metrics = mixed_uses_hill_types[mu_h_keys]
                mu_o_metrics = mixed_use_other_types[mu_o_keys]
                ac_metrics = ac_codes[ac_keys]

                N_temp = networks.NetworkLayerFromNX(primal_graph, distances=distances)
                D_temp = layers.DataLayerFromDict(data_dict)
                D_temp.assign_to_network(N_temp, max_dist=500)
                D_temp.compute_aggregated(landuse_labels,
                                          mixed_use_keys=list(mu_h_metrics) + list(mu_o_metrics),
                                          accessibility_keys=ac_metrics,
                                          cl_disparity_wt_matrix=mock_disparity_wt_matrix,
                                          qs=qs)

                # test against underlying method
                mu_data_hill, mu_data_other, ac_data, ac_data_wt, stats_sum, stats_sum_wt, \
                stats_mean, stats_mean_wt, stats_variance, stats_variance_wt, stats_max, stats_min = \
                    data.local_aggregator(node_map,
                                          edge_map,
                                          node_edge_map,
                                          data_map,
                                          distances,
                                          betas,
                                          landuse_encodings,
                                          qs=qs,
                                          mixed_use_hill_keys=mu_h_keys,
                                          mixed_use_other_keys=mu_o_keys,
                                          accessibility_keys=ac_keys,
                                          cl_disparity_wt_matrix=mock_disparity_wt_matrix)

                for mu_h_idx, mu_h_met in enumerate(mu_h_metrics):
                    for q_idx, q_key in enumerate(qs):
                        for d_idx, d_key in enumerate(distances):
                            assert np.allclose(N_temp.metrics['mixed_uses'][mu_h_met][q_key][d_key],
                                               mu_data_hill[mu_h_idx][q_idx][d_idx], atol=0.001, rtol=0)

                for mu_o_idx, mu_o_met in enumerate(mu_o_metrics):
                    for d_idx, d_key in enumerate(distances):
                        assert np.allclose(N_temp.metrics['mixed_uses'][mu_o_met][d_key],
                                           mu_data_other[mu_o_idx][d_idx], atol=0.001, rtol=0)

                for ac_idx, ac_met in enumerate(ac_metrics):
                    for d_idx, d_key in enumerate(distances):
                        assert np.allclose(N_temp.metrics['accessibility']['non_weighted'][ac_met][d_key],
                                           ac_data[ac_idx][d_idx], atol=0.001, rtol=0)
                        assert np.allclose(N_temp.metrics['accessibility']['weighted'][ac_met][d_key],
                                           ac_data_wt[ac_idx][d_idx], atol=0.001, rtol=0)

    # most integrity checks happen in underlying method, though check here for mismatching labels length and typos
    with pytest.raises(ValueError):
        D.compute_aggregated(landuse_labels[-1], mixed_use_keys=['shannon'])
    with pytest.raises(ValueError):
        D.compute_aggregated(landuse_labels, mixed_use_keys=['spelling_typo'])
    # don't check accessibility_labels for typos - because only warning is triggered (not all labels will be in all data)
    # check that unassigned data layer flags
    with pytest.raises(ValueError):
        D_new = layers.DataLayerFromDict(data_dict)
        D_new.compute_aggregated(landuse_labels, mixed_use_keys=['shannon'])


def test_compute_aggregated_B(primal_graph):
    '''
    Test stats component
    '''
    betas = np.array([-0.01, -0.005])
    distances = networks.distance_from_beta(betas)
    # network layer
    N = networks.NetworkLayerFromNX(primal_graph, distances=distances)
    node_map = N._node_data
    edge_map = N._edge_data
    node_edge_map = N._node_edge_map
    # data layer
    data_dict = mock.mock_data_dict(primal_graph)
    qs = np.array([0, 1, 2])
    D = layers.DataLayerFromDict(data_dict)
    # check single metrics independently against underlying for some use-cases, e.g. hill, non-hill, accessibility...
    D.assign_to_network(N, max_dist=500)

    # generate some mock landuse data
    mock_numeric = mock.mock_numerical_data(len(data_dict), num_arrs=2)

    # generate stats
    D.compute_aggregated(stats_keys=['boo', 'baa'], stats_data_arrs=mock_numeric)

    # test against underlying method
    data_map = D._data
    mu_data_hill, mu_data_other, ac_data, ac_data_wt, \
    stats_sum, stats_sum_wt, stats_mean, stats_mean_wt, stats_variance, stats_variance_wt, stats_max, stats_min = \
        data.local_aggregator(node_map,
                              edge_map,
                              node_edge_map,
                              data_map,
                              distances,
                              betas,
                              numerical_arrays=mock_numeric)

    stats_keys = ['max', 'min', 'sum', 'sum_weighted', 'mean', 'mean_weighted', 'variance', 'variance_weighted']
    stats_data = [stats_max, stats_min, stats_sum, stats_sum_wt, stats_mean, stats_mean_wt, stats_variance,
                  stats_variance_wt]

    for num_idx, num_label in enumerate(['boo', 'baa']):
        for s_key, stats in zip(stats_keys, stats_data):
            for d_idx, d_key in enumerate(distances):
                assert np.allclose(N.metrics['stats'][num_label][s_key][d_key], stats[num_idx][d_idx], atol=0.001,
                                   rtol=0)

    # check that mismatching label and array lengths are caught
    for labels, arrs in ((['a'], mock_numeric),  # mismatching lengths
                         (['a', 'b'], None),  # missing arrays
                         (None, mock_numeric)):  # missing labels
        with pytest.raises(ValueError):
            D.compute_aggregated(stats_keys=labels, stats_data_arrs=arrs)


def network_generator():
    for betas in [[-0.008], [-0.008, -0.002]]:
        distances = networks.distance_from_beta(betas)
        yield distances, betas


def test_hill_diversity(primal_graph):
    for distances, betas in network_generator():
        G = primal_graph.copy()
        data_dict = mock.mock_data_dict(G)
        landuse_labels = mock.mock_categorical_data(len(data_dict))
        # easy version
        N_easy = networks.NetworkLayerFromNX(G, distances=distances)
        D_easy = layers.DataLayerFromDict(data_dict)
        D_easy.assign_to_network(N_easy, max_dist=500)
        D_easy.hill_diversity(landuse_labels, qs=[0, 1, 2])
        # custom version
        N_full = networks.NetworkLayerFromNX(G, distances=distances)
        D_full = layers.DataLayerFromDict(data_dict)
        D_full.assign_to_network(N_full, max_dist=500)
        D_full.compute_aggregated(landuse_labels, mixed_use_keys=['hill'], qs=[0, 1, 2])
        # compare
        for d in distances:
            for q in [0, 1, 2]:
                assert np.allclose(N_easy.metrics['mixed_uses']['hill'][q][d],
                                   N_full.metrics['mixed_uses']['hill'][q][d], atol=0.001, rtol=0)


def test_hill_branch_wt_diversity(primal_graph):
    for distances, betas in network_generator():
        G = primal_graph.copy()
        data_dict = mock.mock_data_dict(G)
        landuse_labels = mock.mock_categorical_data(len(data_dict))
        # easy version
        N_easy = networks.NetworkLayerFromNX(G, distances=distances)
        D_easy = layers.DataLayerFromDict(data_dict)
        D_easy.assign_to_network(N_easy, max_dist=500)
        D_easy.hill_branch_wt_diversity(landuse_labels, qs=[0, 1, 2])
        # custom version
        N_full = networks.NetworkLayerFromNX(G, distances=distances)
        D_full = layers.DataLayerFromDict(data_dict)
        D_full.assign_to_network(N_full, max_dist=500)
        D_full.compute_aggregated(landuse_labels, mixed_use_keys=['hill_branch_wt'], qs=[0, 1, 2])
        # compare
        for d in distances:
            for q in [0, 1, 2]:
                assert np.allclose(N_easy.metrics['mixed_uses']['hill_branch_wt'][q][d],
                                   N_full.metrics['mixed_uses']['hill_branch_wt'][q][d], atol=0.001, rtol=0)


def test_compute_accessibilities(primal_graph):
    for distances, betas in network_generator():
        G = primal_graph.copy()
        data_dict = mock.mock_data_dict(G)
        landuse_labels = mock.mock_categorical_data(len(data_dict))
        # easy version
        N_easy = networks.NetworkLayerFromNX(G, distances=distances)
        D_easy = layers.DataLayerFromDict(data_dict)
        D_easy.assign_to_network(N_easy, max_dist=500)
        D_easy.compute_accessibilities(landuse_labels, ['c'])
        # custom version
        N_full = networks.NetworkLayerFromNX(G, distances=distances)
        D_full = layers.DataLayerFromDict(data_dict)
        D_full.assign_to_network(N_full, max_dist=500)
        D_full.compute_aggregated(landuse_labels, accessibility_keys=['c'])
        # compare
        for d in distances:
            for wt in ['weighted', 'non_weighted']:
                assert np.allclose(N_easy.metrics['accessibility'][wt]['c'][d],
                                   N_full.metrics['accessibility'][wt]['c'][d], atol=0.001, rtol=0)


def test_compute_stats_single(primal_graph):
    for distances, betas in network_generator():
        G = primal_graph.copy()
        data_dict = mock.mock_data_dict(G)
        numeric_data = mock.mock_numerical_data(len(data_dict), num_arrs=1)
        # easy version
        N_easy = networks.NetworkLayerFromNX(G, distances=distances)
        D_easy = layers.DataLayerFromDict(data_dict)
        D_easy.assign_to_network(N_easy, max_dist=500)
        D_easy.compute_stats_single('boo', numeric_data[0])
        # custom version
        N_full = networks.NetworkLayerFromNX(G, distances=distances)
        D_full = layers.DataLayerFromDict(data_dict)
        D_full.assign_to_network(N_full, max_dist=500)
        D_full.compute_aggregated(stats_keys=['boo'], stats_data_arrs=numeric_data)
        # compare
        for n_label in ['boo']:
            for s_label in ['max', 'min', 'mean', 'mean_weighted', 'variance', 'variance_weighted']:
                for dist in distances:
                    assert np.allclose(N_easy.metrics['stats'][n_label][s_label][dist],
                                       N_full.metrics['stats'][n_label][s_label][dist], equal_nan=True, atol=0.001,
                                       rtol=0)
        # check that non-single dimension arrays are caught
        with pytest.raises(ValueError):
            D_easy.compute_stats_single('boo', numeric_data)


def test_compute_stats_multiple(primal_graph):
    for distances, betas in network_generator():
        G = primal_graph.copy()
        data_dict = mock.mock_data_dict(G)
        numeric_data = mock.mock_numerical_data(len(data_dict), num_arrs=2)
        # easy version
        N_easy = networks.NetworkLayerFromNX(G, distances=distances)
        D_easy = layers.DataLayerFromDict(data_dict)
        D_easy.assign_to_network(N_easy, max_dist=500)
        D_easy.compute_stats_multiple(['boo', 'baa'], numeric_data)
        # custom version
        N_full = networks.NetworkLayerFromNX(G, distances=distances)
        D_full = layers.DataLayerFromDict(data_dict)
        D_full.assign_to_network(N_full, max_dist=500)
        D_full.compute_aggregated(stats_keys=['boo', 'baa'], stats_data_arrs=numeric_data)
        # compare
        for n_label in ['boo', 'baa']:
            for s_label in ['max', 'min', 'mean', 'mean_weighted', 'variance', 'variance_weighted']:
                for dist in distances:
                    assert np.allclose(N_easy.metrics['stats'][n_label][s_label][dist],
                                       N_full.metrics['stats'][n_label][s_label][dist], equal_nan=True, atol=0.001,
                                       rtol=0)
