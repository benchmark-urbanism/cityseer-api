# pyright: basic


import copy

import numpy as np
import numpy.typing as npt
import pytest
import utm

from cityseer import config
from cityseer.algos import data
from cityseer.metrics import layers, networks
from cityseer.tools import mock


def test_dict_wgs_to_utm(primal_graph):
    # check that node coordinates are correctly converted
    G_utm = mock.mock_graph()
    data_dict_utm = mock.mock_data_dict(G_utm)
    # create a test dictionary
    test_dict = copy.deepcopy(data_dict_utm)
    # cast to lat, lon
    for k, v in test_dict.items():
        easting = v["x"]
        northing = v["y"]
        # be cognisant of parameter and return order
        # returns in lat lng order
        lat, lng = utm.to_latlon(easting, northing, 30, "U")
        test_dict[k]["x"] = lng
        test_dict[k]["y"] = lat
    # convert back
    dict_converted = layers.dict_wgs_to_utm(test_dict)
    # check that round-trip converted match with reasonable proximity given rounding errors
    for k in data_dict_utm.keys():
        # rounding can be tricky
        assert np.allclose(data_dict_utm[k]["x"], dict_converted[k]["x"], atol=0.1, rtol=config.RTOL)  # relax precision
        assert np.allclose(data_dict_utm[k]["y"], dict_converted[k]["y"], atol=config.ATOL, rtol=config.RTOL)
    # check that missing node attributes throw an error
    for attr in ["x", "y"]:
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
    data_keys, data_map = layers.data_map_from_dict(data_dict)
    assert len(data_keys) == data_map.count == len(data_dict)
    for d_idx, data_key in enumerate(data_keys):
        assert np.isclose(data_map.xs[d_idx], data_dict[data_key]["x"], rtol=config.RTOL, atol=config.ATOL)
        assert np.isclose(data_map.ys[d_idx], data_dict[data_key]["y"], rtol=config.RTOL, atol=config.ATOL)
        assert data_map.nearest_assign[d_idx] == -1
        assert data_map.next_nearest_assign[d_idx] == -1
    # check that missing attributes throw errors
    for attr in ["x", "y"]:
        for k in data_dict.keys():
            del data_dict[k][attr]
        with pytest.raises(AttributeError):
            layers.data_map_from_dict(data_dict)


def test_Data_Layer(primal_graph):
    data_dict = mock.mock_data_dict(primal_graph)
    data_keys, data_map = layers.data_map_from_dict(data_dict)
    # test against DataLayer internal process
    cc_data = layers.DataLayer(data_keys, data_map)
    assert cc_data.data_keys == data_keys
    assert np.allclose(cc_data.data_map.xs, data_map.xs, atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(cc_data.data_map.ys, data_map.ys, atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(cc_data.data_map.nearest_assign, data_map.nearest_assign, atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(
        cc_data.data_map.next_nearest_assign, data_map.next_nearest_assign, atol=config.ATOL, rtol=config.RTOL
    )


def test_Data_Layer_From_Dict(primal_graph):
    data_dict = mock.mock_data_dict(primal_graph)
    data_keys, data_map = layers.data_map_from_dict(data_dict)
    # test against DataLayerFromDict's internal process
    cc_data = layers.DataLayerFromDict(data_dict)
    assert cc_data.data_keys == data_keys
    assert np.allclose(cc_data.data_map.xs, data_map.xs, atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(cc_data.data_map.ys, data_map.ys, atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(cc_data.data_map.nearest_assign, data_map.nearest_assign, atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(
        cc_data.data_map.next_nearest_assign, data_map.next_nearest_assign, atol=config.ATOL, rtol=config.RTOL
    )


def test_compute_landuses(primal_graph):
    betas: npt.NDArray[np.float32] = np.array([0.01, 0.005], dtype=np.float32)
    distances = networks.distance_from_beta(betas)
    # network layer
    cc_netw = networks.NetworkLayerFromNX(primal_graph, distances=distances)
    # data layer
    data_dict = mock.mock_data_dict(primal_graph)
    qs: npt.NDArray[np.float32] = np.array([0, 1, 2], dtype=np.float32)
    cc_data = layers.DataLayerFromDict(data_dict)
    # check single metrics independently against underlying for some use-cases, e.g. hill, non-hill, accessibility...
    cc_data.assign_to_network(cc_netw, max_dist=500)
    # generate some mock landuse data
    landuse_labels = mock.mock_categorical_data(len(data_dict))
    landuse_classes, landuse_encodings = layers.encode_categorical(landuse_labels)
    # compute hill mixed uses
    cc_data.compute_landuses(landuse_labels, mixed_use_keys=["hill_branch_wt"], qs=qs)
    # test against underlying method
    mu_data_hill, mu_data_other, ac_data, ac_data_wt = data.aggregate_landuses(
        cc_netw.network_structure,
        cc_data.data_map,
        distances,
        betas,
        landuse_encodings,
        qs=qs,
        mixed_use_hill_keys=np.array([1]),
    )
    for q_idx, q_key in enumerate(qs):
        for d_idx, d_key in enumerate(distances):
            assert np.allclose(
                cc_netw.metrics_state.mixed_uses["hill_branch_wt"][q_key][d_key],
                mu_data_hill[0][q_idx][d_idx],
                atol=config.ATOL,
                rtol=config.RTOL,
            )
    # gini simpson
    cc_data.compute_landuses(landuse_labels, mixed_use_keys=["gini_simpson"])
    # test against underlying method
    mu_data_hill, mu_data_other, ac_data, ac_data_wt = data.aggregate_landuses(
        cc_netw.network_structure,
        cc_data.data_map,
        distances,
        betas,
        landuse_encodings,
        mixed_use_other_keys=np.array([1]),
    )
    for d_idx, d_key in enumerate(distances):
        assert np.allclose(
            cc_netw.metrics_state.mixed_uses["gini_simpson"][d_key],
            mu_data_other[0][d_idx],
            atol=config.ATOL,
            rtol=config.RTOL,
        )
    # accessibilities
    cc_data.compute_landuses(landuse_labels, accessibility_keys=["c"])
    # test against underlying method
    mu_data_hill, mu_data_other, ac_data, ac_data_wt = data.aggregate_landuses(
        cc_netw.network_structure,
        cc_data.data_map,
        distances,
        betas,
        landuse_encodings,
        accessibility_keys=np.array([landuse_classes.index("c")]),
    )
    for d_idx, d_key in enumerate(distances):
        assert np.allclose(
            cc_netw.metrics_state.accessibility.non_weighted["c"][d_key],
            ac_data[0][d_idx],
            atol=config.ATOL,
            rtol=config.RTOL,
        )
        assert np.allclose(
            cc_netw.metrics_state.accessibility.weighted["c"][d_key],
            ac_data_wt[0][d_idx],
            atol=config.ATOL,
            rtol=config.RTOL,
        )
    # also check the number of returned types for a few assortments of metrics
    mixed_uses_hill_types = np.array(["hill", "hill_branch_wt", "hill_pairwise_wt", "hill_pairwise_disparity"])
    mixed_use_other_types = np.array(["shannon", "gini_simpson", "raos_pairwise_disparity"])
    ac_codes = np.array(landuse_classes)
    # mixed uses hill
    mu_hill_random = np.arange(len(mixed_uses_hill_types))
    np.random.shuffle(mu_hill_random)
    # mixed uses other
    mu_other_random = np.arange(len(mixed_use_other_types))
    np.random.shuffle(mu_other_random)
    # accessibility
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
                # prepare network and compute
                N_temp = networks.NetworkLayerFromNX(primal_graph, distances=distances)
                D_temp = layers.DataLayerFromDict(data_dict)
                D_temp.assign_to_network(N_temp, max_dist=500)
                D_temp.compute_landuses(
                    landuse_labels,
                    mixed_use_keys=list(mu_h_metrics) + list(mu_o_metrics),
                    accessibility_keys=ac_metrics,
                    cl_disparity_wt_matrix=mock_disparity_wt_matrix,
                    qs=qs,
                )
                # test against underlying method
                (mu_data_hill, mu_data_other, ac_data, ac_data_wt,) = data.aggregate_landuses(
                    cc_netw.network_structure,
                    cc_data.data_map,
                    distances,
                    betas,
                    landuse_encodings,
                    qs=qs,
                    mixed_use_hill_keys=mu_h_keys,
                    mixed_use_other_keys=mu_o_keys,
                    accessibility_keys=ac_keys,
                    cl_disparity_wt_matrix=mock_disparity_wt_matrix,
                )
                for mu_h_idx, mu_h_met in enumerate(mu_h_metrics):
                    for q_idx, q_key in enumerate(qs):
                        for d_idx, d_key in enumerate(distances):
                            assert np.allclose(
                                N_temp.metrics_state.mixed_uses[mu_h_met][q_key][d_key],
                                mu_data_hill[mu_h_idx][q_idx][d_idx],
                                atol=config.ATOL,
                                rtol=config.RTOL,
                            )
                for mu_o_idx, mu_o_met in enumerate(mu_o_metrics):
                    for d_idx, d_key in enumerate(distances):
                        assert np.allclose(
                            N_temp.metrics_state.mixed_uses[mu_o_met][d_key],
                            mu_data_other[mu_o_idx][d_idx],
                            atol=config.ATOL,
                            rtol=config.RTOL,
                        )
                for ac_idx, ac_met in enumerate(ac_metrics):
                    for d_idx, d_key in enumerate(distances):
                        assert np.allclose(
                            N_temp.metrics_state.accessibility.non_weighted[ac_met][d_key],
                            ac_data[ac_idx][d_idx],
                            atol=config.ATOL,
                            rtol=config.RTOL,
                        )
                        assert np.allclose(
                            N_temp.metrics_state.accessibility.weighted[ac_met][d_key],
                            ac_data_wt[ac_idx][d_idx],
                            atol=config.ATOL,
                            rtol=config.RTOL,
                        )
    # most integrity checks happen in underlying method, though check here for mismatching labels length and typos
    with pytest.raises(ValueError):
        cc_data.compute_landuses(landuse_labels[-1], mixed_use_keys=["shannon"])
    with pytest.raises(ValueError):
        cc_data.compute_landuses(landuse_labels, mixed_use_keys=["spelling_typo"])
    # don't check accessibility_labels for typos - because only warning is triggered (not all labels will be in all data)
    # check that unassigned data layer flags
    with pytest.raises(ValueError):
        D_new = layers.DataLayerFromDict(data_dict)
        D_new.compute_landuses(landuse_labels, mixed_use_keys=["shannon"])


def network_generator():
    for betas in [[0.008], [0.008, 0.002]]:
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
        D_full.compute_landuses(landuse_labels, mixed_use_keys=["hill"], qs=[0, 1, 2])
        # compare
        for d in distances:
            for q in [0, 1, 2]:
                assert np.allclose(
                    N_easy.metrics_state.mixed_uses["hill"][q][d],
                    N_full.metrics_state.mixed_uses["hill"][q][d],
                    atol=config.ATOL,
                    rtol=config.RTOL,
                )


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
        D_full.compute_landuses(landuse_labels, mixed_use_keys=["hill_branch_wt"], qs=[0, 1, 2])
        # compare
        for d in distances:
            for q in [0, 1, 2]:
                assert np.allclose(
                    N_easy.metrics_state.mixed_uses["hill_branch_wt"][q][d],
                    N_full.metrics_state.mixed_uses["hill_branch_wt"][q][d],
                    atol=config.ATOL,
                    rtol=config.RTOL,
                )


def test_compute_accessibilities(primal_graph):
    for distances, betas in network_generator():
        G = primal_graph.copy()
        data_dict = mock.mock_data_dict(G)
        landuse_labels = mock.mock_categorical_data(len(data_dict))
        # easy version
        N_easy = networks.NetworkLayerFromNX(G, distances=distances)
        D_easy = layers.DataLayerFromDict(data_dict)
        D_easy.assign_to_network(N_easy, max_dist=500)
        D_easy.compute_accessibilities(landuse_labels, ["c"])
        # custom version
        N_full = networks.NetworkLayerFromNX(G, distances=distances)
        D_full = layers.DataLayerFromDict(data_dict)
        D_full.assign_to_network(N_full, max_dist=500)
        D_full.compute_landuses(landuse_labels, accessibility_keys=["c"])
        # compare
        for d in distances:
            assert np.allclose(
                N_easy.metrics_state.accessibility.weighted["c"][d],
                N_full.metrics_state.accessibility.weighted["c"][d],
                atol=config.ATOL,
                rtol=config.RTOL,
            )
            assert np.allclose(
                N_easy.metrics_state.accessibility.non_weighted["c"][d],
                N_full.metrics_state.accessibility.non_weighted["c"][d],
                atol=config.ATOL,
                rtol=config.RTOL,
            )


def test_compute_stats(primal_graph):
    """
    Test stats component
    """
    betas: npt.NDArray[np.float32] = np.array([0.01, 0.005])
    distances = networks.distance_from_beta(betas)
    # network layer
    N_single = networks.NetworkLayerFromNX(primal_graph, distances=distances)
    N_multi = networks.NetworkLayerFromNX(primal_graph, distances=distances)
    # data layer
    data_dict = mock.mock_data_dict(primal_graph)
    D_single = layers.DataLayerFromDict(data_dict)
    D_multi = layers.DataLayerFromDict(data_dict)
    # check single metrics independently against underlying for some use-cases, e.g. hill, non-hill, accessibility...
    D_single.assign_to_network(N_single, max_dist=500)
    D_multi.assign_to_network(N_multi, max_dist=500)
    # generate some mock landuse data
    mock_numeric = mock.mock_numerical_data(len(data_dict), num_arrs=2)
    # generate stats
    D_single.compute_stats(stats_keys="boo", stats_data=mock_numeric[0])
    D_single.compute_stats(stats_keys="baa", stats_data=mock_numeric[1])
    D_multi.compute_stats(stats_keys=["boo", "baa"], stats_data=mock_numeric)
    # test against data computed directly from underlying methods
    (
        stats_sum,
        stats_sum_wt,
        stats_mean,
        stats_mean_wt,
        stats_variance,
        stats_variance_wt,
        stats_max,
        stats_min,
    ) = data.aggregate_stats(
        N_single.network_structure,
        D_single.data_map,
        distances,
        betas,
        numerical_arrays=mock_numeric,
    )
    stats_keys = [
        "max",
        "min",
        "sum",
        "sum_weighted",
        "mean",
        "mean_weighted",
        "variance",
        "variance_weighted",
    ]
    stats_data = [
        stats_max,
        stats_min,
        stats_sum,
        stats_sum_wt,
        stats_mean,
        stats_mean_wt,
        stats_variance,
        stats_variance_wt,
    ]
    for num_idx, num_label in enumerate(["boo", "baa"]):
        for s_key, stats in zip(stats_keys, stats_data):
            for d_idx, d_key in enumerate(distances):
                # check one-at-a-time computed vs multiply computed
                assert np.allclose(
                    N_single.metrics_state.stats[num_label][s_key][d_key],
                    N_multi.metrics_state.stats[num_label][s_key][d_key],
                    atol=config.ATOL,
                    rtol=config.RTOL,
                    equal_nan=True,
                )
                # check one-at-a-time against manual
                assert np.allclose(
                    N_single.metrics_state.stats[num_label][s_key][d_key],
                    stats[num_idx][d_idx],
                    atol=config.ATOL,
                    rtol=config.RTOL,
                    equal_nan=True,
                )
                # check multiply computed against manual
                assert np.allclose(
                    N_multi.metrics_state.stats[num_label][s_key][d_key],
                    stats[num_idx][d_idx],
                    atol=config.ATOL,
                    rtol=config.RTOL,
                    equal_nan=True,
                )
    # check that problematic keys and data arrays are caught
    for labels, arrs, err in (
        (["a"], mock_numeric, ValueError),  # mismatching lengths
        (["a", "b"], None, TypeError),  # missing arrays
        (["a", "b"], [], ValueError),  # missing arrays
        (None, mock_numeric, TypeError),  # missing labels
        ([], mock_numeric, ValueError),
    ):  # missing labels
        with pytest.raises(err):
            D_multi.compute_stats(stats_keys=labels, stats_data=arrs)
