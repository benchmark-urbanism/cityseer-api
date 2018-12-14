import copy

import numpy as np
import pytest
import utm

from cityseer.algos import diversity
from cityseer.metrics import layers, networks
from cityseer.util import mock, graphs


def test_dict_wgs_to_utm():
    # check that node coordinates are correctly converted
    G_utm, pos = mock.mock_graph()
    data_dict = mock.mock_data(G_utm)

    data_dict_wgs = copy.deepcopy(data_dict)
    for k, v in data_dict_wgs.items():
        x = v['x']
        y = v['y']
        y, x = utm.to_latlon(y, x, 30, 'U')
        data_dict_wgs[k]['x'] = x
        data_dict_wgs[k]['y'] = y

    data_dict_converted = layers.dict_wgs_to_utm(data_dict_wgs)

    for k in data_dict.keys():
        # rounding can be tricky
        assert abs(data_dict[k]['x'] - data_dict_converted[k]['x']) < 0.01
        assert abs(data_dict[k]['y'] - data_dict_converted[k]['y']) < 0.01

    # check that missing node attributes throw an error
    for attr in ['x', 'y']:
        G_wgs, pos_wgs = mock.mock_graph(wgs84_coords=True)
        data_dict_wgs = mock.mock_data(G_wgs)
        for k in data_dict_wgs.keys():
            del data_dict_wgs[k][attr]
            break
        # check that missing attribute throws an error
        with pytest.raises(AttributeError):
            layers.dict_wgs_to_utm(data_dict_wgs)

    # check that non WGS coordinates throw error
    G_utm, pos = mock.mock_graph()
    data_dict = mock.mock_data(G_utm)
    with pytest.raises(AttributeError):
        layers.dict_wgs_to_utm(data_dict)


def test_data_map_from_dict():
    # generate mock data
    G, pos = mock.mock_graph()
    data_dict = mock.mock_data(G)
    data_uids, data_map, class_labels = layers.data_map_from_dict(data_dict)

    assert len(data_uids) == len(data_map) == len(data_dict)

    for d_label, d in zip(data_uids, data_map):
        assert d[0] == data_dict[d_label]['x']
        assert d[1] == data_dict[d_label]['y']
        assert d[2] == data_dict[d_label]['live']
        # check that the class encoding maps back to the class
        class_encoding = int(d[3])
        assert class_labels[class_encoding] == data_dict[d_label]['class']
        assert np.isnan(d[4])
        assert np.isnan(d[5])

    # check that missing attributes throw errors
    for attr in ['x', 'y']:
        for k in data_dict.keys():
            del data_dict[k][attr]
        with pytest.raises(AttributeError):
            layers.data_map_from_dict(data_dict)


def test_Data_Layer():
    G, pos = mock.mock_graph()
    data_dict = mock.mock_data(G)
    data_uids, data_map, class_labels = layers.data_map_from_dict(data_dict)
    x_arr = data_map[:, 0]
    y_arr = data_map[:, 1]
    live = data_map[:, 2]
    class_codes = data_map[:, 3]

    # test against Data_Layer internal process
    D = layers.Data_Layer(data_uids, data_map, class_labels)
    assert D.uids == data_uids
    assert np.allclose(D._data, data_map, equal_nan=True)
    assert D.class_labels == class_labels
    assert np.array_equal(D.x_arr, x_arr)
    assert np.array_equal(D.y_arr, y_arr)
    assert np.array_equal(D.live, live)
    assert np.array_equal(D.class_codes, class_codes)


def test_Data_Layer_From_Dict():
    G, pos = mock.mock_graph()
    data_dict = mock.mock_data(G)
    data_uids, data_map, class_labels = layers.data_map_from_dict(data_dict)
    x_arr = data_map[:, 0]
    y_arr = data_map[:, 1]
    live = data_map[:, 2]
    class_codes = data_map[:, 3]

    # test against Data_Layer_From_Dict's internal process
    D = layers.Data_Layer_From_Dict(data_dict)
    assert D.uids == data_uids
    assert np.allclose(D._data, data_map, equal_nan=True)
    assert D.class_labels == class_labels
    assert np.array_equal(D.x_arr, x_arr)
    assert np.array_equal(D.y_arr, y_arr)
    assert np.array_equal(D.live, live)
    assert np.array_equal(D.class_codes, class_codes)


def test_compute_landuses():
    '''
    Underlying method also tested via diversity.test_local_landuses()
    '''

    G, pos = mock.mock_graph()
    G = graphs.networkX_simple_geoms(G)
    G = graphs.networkX_edge_defaults(G)

    betas = np.array([-0.01, -0.005])
    distances = networks.distance_from_beta(betas)
    N = networks.Network_Layer_From_NetworkX(G, distances)
    node_map = N._nodes
    edge_map = N._edges

    data_dict = mock.mock_data(G)
    qs = np.array([0, 1, 2])
    D = layers.Data_Layer_From_Dict(data_dict, qs=qs)

    # check single metrics independently against underlying
    N_temp = copy.deepcopy(N)
    D.assign_to_network(N_temp, max_dist=500)
    D.compute_landuses(mixed_use_metrics=['hill_branch_wt'])
    # test against underlying method
    data_map = D._data
    mu_data_hill, mu_data_other, ac_data, ac_data_wt = \
        diversity.local_landuses(node_map,
                                 edge_map,
                                 data_map,
                                 distances,
                                 betas,
                                 qs=qs,
                                 mixed_use_hill_keys=np.array([1]))

    for q_idx, q_key in enumerate(qs):
        for d_idx, d_key in enumerate(distances):
            assert np.array_equal(N_temp.metrics['mixed_uses']['hill_branch_wt'][q_key][d_key],
                                  mu_data_hill[0][q_idx][d_idx])

    N_temp = copy.deepcopy(N)
    D.assign_to_network(N_temp, max_dist=500)
    D.compute_landuses(mixed_use_metrics=['gini_simpson'])
    # test against underlying method
    data_map = D._data
    mu_data_hill, mu_data_other, ac_data, ac_data_wt = \
        diversity.local_landuses(node_map,
                                 edge_map,
                                 data_map,
                                 distances,
                                 betas,
                                 mixed_use_other_keys=np.array([1]))

    for d_idx, d_key in enumerate(distances):
        assert np.array_equal(N_temp.metrics['mixed_uses']['gini_simpson'][d_key], mu_data_other[0][d_idx])

    N_temp = copy.deepcopy(N)
    D.assign_to_network(N_temp, max_dist=500)
    D.compute_landuses(accessibility_labels=['c'])
    # test against underlying method
    data_map = D._data
    mu_data_hill, mu_data_other, ac_data, ac_data_wt = \
        diversity.local_landuses(node_map,
                                 edge_map,
                                 data_map,
                                 distances,
                                 betas,
                                 accessibility_keys=np.array([D.class_labels.index('c')]))

    for d_idx, d_key in enumerate(distances):
        assert np.array_equal(N_temp.metrics['accessibility']['non_weighted']['c'][d_key], ac_data[0][d_idx])
        assert np.array_equal(N_temp.metrics['accessibility']['weighted']['c'][d_key], ac_data_wt[0][d_idx])

    # also check the number of returned types for a few assortments of metrics
    mixed_uses_hill_types = np.array(['hill',
                                      'hill_branch_wt',
                                      'hill_pairwise_wt',
                                      'hill_pairwise_disparity'])
    mixed_use_other_types = np.array(['shannon',
                                      'gini_simpson',
                                      'raos_pairwise_disparity'])
    ac_codes = np.array(D.class_labels)

    mu_hill_random = np.arange(len(mixed_uses_hill_types))
    np.random.shuffle(mu_hill_random)

    mu_other_random = np.arange(len(mixed_use_other_types))
    np.random.shuffle(mu_other_random)

    ac_random = np.arange(len(ac_codes))
    np.random.shuffle(ac_random)

    # mock disparity matrix
    mock_disparity_wt_matrix = np.full((len(ac_codes), len(ac_codes)), 1)

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

                N_temp = copy.deepcopy(N)
                D_temp = layers.Data_Layer_From_Dict(data_dict, mock_disparity_wt_matrix, qs)
                D_temp.assign_to_network(N_temp, max_dist=500)
                D_temp.compute_landuses(mixed_use_metrics=list(mu_h_metrics) + list(mu_o_metrics),
                                        accessibility_labels=ac_metrics)

                # test against underlying method
                mu_data_hill, mu_data_other, ac_data, ac_data_wt = \
                    diversity.local_landuses(node_map,
                                             edge_map,
                                             data_map,
                                             distances,
                                             betas,
                                             qs=qs,
                                             mixed_use_hill_keys=mu_h_keys,
                                             mixed_use_other_keys=mu_o_keys,
                                             accessibility_keys=ac_keys,
                                             cl_disparity_wt_matrix=mock_disparity_wt_matrix)

                for mu_h_idx, mu_h_met in enumerate(mu_h_metrics):
                    for q_idx, q_key in enumerate(qs):
                        for d_idx, d_key in enumerate(distances):
                            assert np.array_equal(N_temp.metrics['mixed_uses'][mu_h_met][q_key][d_key],
                                                  mu_data_hill[mu_h_idx][q_idx][d_idx])

                for mu_o_idx, mu_o_met in enumerate(mu_o_metrics):
                    for d_idx, d_key in enumerate(distances):
                        assert np.array_equal(N_temp.metrics['mixed_uses'][mu_o_met][d_key],
                                              mu_data_other[mu_o_idx][d_idx])

                for ac_idx, ac_met in enumerate(ac_metrics):
                    for d_idx, d_key in enumerate(distances):
                        assert np.array_equal(N_temp.metrics['accessibility']['non_weighted'][ac_met][d_key],
                                              ac_data[ac_idx][d_idx])
                        assert np.array_equal(N_temp.metrics['accessibility']['weighted'][ac_met][d_key],
                                              ac_data_wt[ac_idx][d_idx])

    # check that angular gets passed through
    G_dual = graphs.networkX_to_dual(G)

    N_dual = networks.Network_Layer_From_NetworkX(G_dual, distances=[2000], angular=True)
    D_dual = layers.Data_Layer_From_Dict(data_dict)
    D_dual.assign_to_network(N_dual, max_dist=500)
    D_dual.compute_landuses(mixed_use_metrics=['shannon'], accessibility_labels=['c'])

    N_dual_sidestep = networks.Network_Layer_From_NetworkX(G_dual, distances=[2000], angular=False)
    D_dual = layers.Data_Layer_From_Dict(data_dict)
    D_dual.assign_to_network(N_dual_sidestep, max_dist=500)
    D_dual.compute_landuses(mixed_use_metrics=['shannon'], accessibility_labels=['c'])

    assert not np.array_equal(N_dual.metrics['mixed_uses']['shannon'][2000],
                              N_dual_sidestep.metrics['mixed_uses']['shannon'][2000])
    assert not np.array_equal(N_dual.metrics['accessibility']['non_weighted']['c'][2000],
                              N_dual_sidestep.metrics['accessibility']['non_weighted']['c'][2000])
    assert not np.array_equal(N_dual.metrics['accessibility']['weighted']['c'][2000],
                              N_dual_sidestep.metrics['accessibility']['weighted']['c'][2000])

    # most integrity checks happen in underlying method, though check here for typos
    with pytest.raises(ValueError):
        D.compute_landuses(mixed_use_metrics=['spelling_typo'])
    with pytest.raises(ValueError):
        D.compute_landuses(accessibility_labels=['spelling_typo'])
    # check that unassigned data layer flags
    with pytest.raises(ValueError):
        D_new = layers.Data_Layer_From_Dict(data_dict)
        D_new.compute_landuses(mixed_use_metrics=['shannon'])


def network_generator():
    for betas in [[-0.008], [-0.008, -0.002]]:
        distances = networks.distance_from_beta(betas)
        for angular in [False, True]:
            G, pos = mock.mock_graph()
            G = graphs.networkX_simple_geoms(G)
            G = graphs.networkX_edge_defaults(G)
            yield G, distances, betas, angular


def test_hill_diversity():
    for G, distances, betas, angular in network_generator():

        data_dict = mock.mock_data(G)

        # easy version
        N_easy = networks.Network_Layer_From_NetworkX(G, distances, angular=angular)
        D_easy = layers.Data_Layer_From_Dict(data_dict, qs=[0, 1, 2])
        D_easy.assign_to_network(N_easy, max_dist=500)
        D_easy.hill_diversity()
        # custom version
        N_full = networks.Network_Layer_From_NetworkX(G, distances, angular=angular)
        D_full = layers.Data_Layer_From_Dict(data_dict, qs=[0, 1, 2])
        D_full.assign_to_network(N_full, max_dist=500)
        D_full.compute_landuses(mixed_use_metrics=['hill'])

        # compare
        for d in distances:
            for q in [0, 1, 2]:
                assert np.array_equal(N_easy.metrics['mixed_uses']['hill'][q][d],
                                      N_full.metrics['mixed_uses']['hill'][q][d])


def test_hill_branch_wt_diversity():
    for G, distances, betas, angular in network_generator():

        data_dict = mock.mock_data(G)

        # easy version
        N_easy = networks.Network_Layer_From_NetworkX(G, distances, angular=angular)
        D_easy = layers.Data_Layer_From_Dict(data_dict, qs=[0, 1, 2])
        D_easy.assign_to_network(N_easy, max_dist=500)
        D_easy.hill_branch_wt_diversity()
        # custom version
        N_full = networks.Network_Layer_From_NetworkX(G, distances, angular=angular)
        D_full = layers.Data_Layer_From_Dict(data_dict, qs=[0, 1, 2])
        D_full.assign_to_network(N_full, max_dist=500)
        D_full.compute_landuses(mixed_use_metrics=['hill_branch_wt'])

        # compare
        for d in distances:
            for q in [0, 1, 2]:
                assert np.array_equal(N_easy.metrics['mixed_uses']['hill_branch_wt'][q][d],
                                      N_full.metrics['mixed_uses']['hill_branch_wt'][q][d])


def test_hill_pairwise_wt_diversity():
    for G, distances, betas, angular in network_generator():

        data_dict = mock.mock_data(G)

        # easy version
        N_easy = networks.Network_Layer_From_NetworkX(G, distances, angular=angular)
        D_easy = layers.Data_Layer_From_Dict(data_dict, qs=[0, 1, 2])
        D_easy.assign_to_network(N_easy, max_dist=500)
        D_easy.hill_pairwise_wt_diversity()
        # custom version
        N_full = networks.Network_Layer_From_NetworkX(G, distances, angular=angular)
        D_full = layers.Data_Layer_From_Dict(data_dict, qs=[0, 1, 2])
        D_full.assign_to_network(N_full, max_dist=500)
        D_full.compute_landuses(mixed_use_metrics=['hill_pairwise_wt'])

        # compare
        for d in distances:
            for q in [0, 1, 2]:
                assert np.array_equal(N_easy.metrics['mixed_uses']['hill_pairwise_wt'][q][d],
                                      N_full.metrics['mixed_uses']['hill_pairwise_wt'][q][d])


def test_compute_accessibilities():
    for G, distances, betas, angular in network_generator():

        data_dict = mock.mock_data(G)

        # easy version
        N_easy = networks.Network_Layer_From_NetworkX(G, distances, angular=angular)
        D_easy = layers.Data_Layer_From_Dict(data_dict, qs=[0, 1, 2])
        D_easy.assign_to_network(N_easy, max_dist=500)
        D_easy.compute_accessibilities(['c'])
        # custom version
        N_full = networks.Network_Layer_From_NetworkX(G, distances, angular=angular)
        D_full = layers.Data_Layer_From_Dict(data_dict, qs=[0, 1, 2])
        D_full.assign_to_network(N_full, max_dist=500)
        D_full.compute_landuses(accessibility_labels=['c'])

        # compare
        for d in distances:
            for wt in ['weighted', 'non_weighted']:
                assert np.array_equal(N_easy.metrics['accessibility'][wt]['c'][d],
                                      N_full.metrics['accessibility'][wt]['c'][d])
