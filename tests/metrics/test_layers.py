# pyright: basic
from __future__ import annotations

from typing import Generator

import numpy as np
import numpy.typing as npt
import pytest
from sklearn.preprocessing import LabelEncoder  # type: ignore

from cityseer import config, structures
from cityseer.algos import data
from cityseer.metrics import layers, networks
from cityseer.tools import graphs, mock


def test_assign_gdf_to_network(primal_graph):
    _nodes_gdf, network_structure = graphs.network_structure_from_nx(primal_graph, 3395)
    data_gdf = mock.mock_data_gdf(primal_graph)
    _data_map, data_gdf = layers.assign_gdf_to_network(data_gdf, network_structure, 400)
    # generate manual data map
    data_map = structures.DataMap(len(data_gdf))
    data_map.xs = data_gdf.geometry.x.values.astype(np.float32)
    data_map.ys = data_gdf.geometry.y.values.astype(np.float32)
    data.assign_to_network(
        data_map,
        network_structure,
        np.float32(400),
    )
    # cross check
    assert np.allclose(
        data_map.nearest_assign,
        data_gdf.nearest_assign,
        atol=config.ATOL,
        rtol=config.RTOL,
    )
    assert np.allclose(
        data_map.next_nearest_assign,
        data_gdf.next_nearest_assign,
        atol=config.ATOL,
        rtol=config.RTOL,
    )
    # repeat
    # this time, already populated GDF will be passed through
    # data map will be populated from GDF
    new_data_map, data_gdf = layers.assign_gdf_to_network(data_gdf, network_structure, 400)
    assert np.allclose(
        data_map.nearest_assign,
        data_gdf.nearest_assign,
        atol=config.ATOL,
        rtol=config.RTOL,
    )
    assert np.allclose(
        data_map.next_nearest_assign,
        data_gdf.next_nearest_assign,
        atol=config.ATOL,
        rtol=config.RTOL,
    )
    assert np.allclose(
        data_map.nearest_assign,
        new_data_map.nearest_assign,
        atol=config.ATOL,
        rtol=config.RTOL,
    )
    assert np.allclose(
        data_map.next_nearest_assign,
        new_data_map.next_nearest_assign,
        atol=config.ATOL,
        rtol=config.RTOL,
    )


def test_compute_landuses(primal_graph):
    betas: npt.NDArray[np.float32] = np.array([0.01, 0.005], dtype=np.float32)
    # network
    nodes_gdf, network_structure = graphs.network_structure_from_nx(primal_graph, 3395)
    # data
    data_gdf = mock.mock_landuse_categorical_data(primal_graph)
    # compute hill mixed uses
    raw_betas = [0.01, 0.005]
    qs: npt.NDArray[np.float32] = np.array([0, 1, 2], dtype=np.float32)
    nodes_gdf, data_gdf = layers.compute_landuses(
        data_gdf,
        "categorical_landuses",
        nodes_gdf,
        network_structure,
        betas=raw_betas,
        mixed_use_keys=["hill_branch_wt", "gini_simpson"],
        accessibility_keys=["c"],
        qs=qs,
    )
    # test against manual implementation over underlying method
    data_map, data_gdf = layers.assign_gdf_to_network(data_gdf, network_structure, 400)
    distances, betas = networks.pair_distances_betas(betas=raw_betas)
    lab_enc = LabelEncoder()
    encoded_lu_labels = lab_enc.fit_transform(data_gdf["categorical_landuses"])  # type: ignore
    mu_data_hill, mu_data_other, ac_data, ac_data_wt = data.aggregate_landuses(
        network_structure,
        data_map,
        distances,
        betas,
        encoded_lu_labels,  # type: ignore
        qs=qs,
        mixed_use_hill_keys=np.array([1]),
    )
    for q_idx, q_key in enumerate(qs):
        for d_idx, d_key in enumerate(distances):
            mu_hill_data_key = config.prep_gdf_key(f"hill_branch_wt_q{q_key}_{d_key}")
            assert np.allclose(
                nodes_gdf[mu_hill_data_key],
                mu_data_hill[0][q_idx][d_idx],
                atol=config.ATOL,
                rtol=config.RTOL,
            )
    # gini simpson
    mu_data_hill, mu_data_other, ac_data, ac_data_wt = data.aggregate_landuses(
        network_structure,
        data_map,
        distances,
        betas,
        encoded_lu_labels,  # type: ignore
        qs=qs,
        mixed_use_other_keys=np.array([1]),
    )
    for d_idx, d_key in enumerate(distances):
        mu_oth_data_key = config.prep_gdf_key(f"gini_simpson_{d_key}")
        assert np.allclose(
            nodes_gdf[mu_oth_data_key],
            mu_data_other[0][d_idx],
            atol=config.ATOL,
            rtol=config.RTOL,
        )
    # accessibilities
    mu_data_hill, mu_data_other, ac_data, ac_data_wt = data.aggregate_landuses(
        network_structure,
        data_map,
        distances,
        betas,
        encoded_lu_labels,  # type: ignore
        qs=qs,
        accessibility_keys=np.array([lab_enc.classes_.tolist().index("c")]),
    )
    for d_idx, d_key in enumerate(distances):
        acc_data_key_nw = config.prep_gdf_key(f"c_{d_key}_non_weighted")
        assert np.allclose(
            nodes_gdf[acc_data_key_nw],
            ac_data[0][d_idx],
            atol=config.ATOL,
            rtol=config.RTOL,
        )
        acc_data_key_wt = config.prep_gdf_key(f"c_{d_key}_weighted")
        assert np.allclose(
            nodes_gdf[acc_data_key_wt],
            ac_data_wt[0][d_idx],
            atol=config.ATOL,
            rtol=config.RTOL,
        )
    # most integrity checks happen in underlying method, though check here for mismatching labels length and typos
    with pytest.raises(ValueError):
        layers.compute_landuses(
            data_gdf,
            "lu_col_label_typo",
            nodes_gdf,
            network_structure,
            distances=distances,
        )
    # both distances and betas
    with pytest.raises(ValueError):
        layers.compute_landuses(
            data_gdf,
            "categorical_landuses",
            nodes_gdf,
            network_structure,
            distances=distances,
            betas=betas,
            mixed_use_keys=["spelling_typo"],
        )
    # mu typo
    with pytest.raises(ValueError):
        layers.compute_landuses(
            data_gdf,
            "categorical_landuses",
            nodes_gdf,
            network_structure,
            betas=betas,
            mixed_use_keys=["spelling_typo"],
        )
    # also check the number of returned types for a few assortments of metrics
    mixed_uses_hill_types = np.array(["hill", "hill_branch_wt", "hill_pairwise_wt", "hill_pairwise_disparity"])
    mixed_use_other_types = np.array(["shannon", "gini_simpson", "raos_pairwise_disparity"])
    # mixed uses hill
    mu_hill_random = np.arange(len(mixed_uses_hill_types))
    np.random.shuffle(mu_hill_random)
    # mixed uses other
    mu_other_random = np.arange(len(mixed_use_other_types))
    np.random.shuffle(mu_other_random)
    # accessibility
    ac_random = np.arange(len(lab_enc.classes_))
    np.random.shuffle(ac_random)
    # mock disparity matrix
    mock_disparity_wt_matrix = np.full((len(lab_enc.classes_), len(lab_enc.classes_)), 1)
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
                ac_metrics: list[str] = lab_enc.inverse_transform(ac_keys).tolist()
                # prepare network and compute
                nodes_gdf, network_structure = graphs.network_structure_from_nx(primal_graph, 3395)
                data_map, data_gdf = layers.assign_gdf_to_network(data_gdf, network_structure, 400)
                distances, betas = networks.pair_distances_betas(betas=raw_betas)
                # landuse encodings
                lab_enc = LabelEncoder()
                encoded_lu_labels = lab_enc.fit_transform(data_gdf["categorical_landuses"])  # type: ignore
                nodes_gdf, data_gdf = layers.compute_landuses(
                    data_gdf,
                    "categorical_landuses",
                    nodes_gdf,
                    network_structure,
                    betas=raw_betas,
                    mixed_use_keys=list(mu_h_metrics) + list(mu_o_metrics),
                    accessibility_keys=ac_metrics,
                    cl_disparity_wt_matrix=mock_disparity_wt_matrix,
                    qs=qs,
                )
                # test against underlying method
                (mu_data_hill, mu_data_other, ac_data, ac_data_wt,) = data.aggregate_landuses(
                    network_structure,
                    data_map,
                    distances,
                    betas,
                    encoded_lu_labels,  # type: ignore
                    qs=qs,
                    mixed_use_hill_keys=mu_h_keys,
                    mixed_use_other_keys=mu_o_keys,
                    accessibility_keys=ac_keys,
                    cl_disparity_wt_matrix=mock_disparity_wt_matrix,
                )
                for mu_h_idx, mu_h_met in enumerate(mu_h_metrics):
                    for q_idx, q_key in enumerate(qs):
                        for d_idx, d_key in enumerate(distances):
                            mu_hill_data_key = config.prep_gdf_key(f"{mu_h_met}_q{q_key}_{d_key}")
                            assert np.allclose(
                                nodes_gdf[mu_hill_data_key],
                                mu_data_hill[mu_h_idx][q_idx][d_idx],
                                atol=config.ATOL,
                                rtol=config.RTOL,
                            )
                for mu_o_idx, mu_o_met in enumerate(mu_o_metrics):
                    for d_idx, d_key in enumerate(distances):
                        mu_oth_data_key = config.prep_gdf_key(f"{mu_o_met}_{d_key}")
                        assert np.allclose(
                            nodes_gdf[mu_oth_data_key],
                            mu_data_other[mu_o_idx][d_idx],
                            atol=config.ATOL,
                            rtol=config.RTOL,
                        )
                for ac_idx, ac_met in enumerate(ac_metrics):
                    for d_idx, d_key in enumerate(distances):
                        acc_data_key_nw = config.prep_gdf_key(f"{ac_met}_{d_key}_non_weighted")
                        assert np.allclose(
                            nodes_gdf[acc_data_key_nw],
                            ac_data[ac_idx][d_idx],
                            atol=config.ATOL,
                            rtol=config.RTOL,
                        )
                        acc_data_key_wt = config.prep_gdf_key(f"{ac_met}_{d_key}_weighted")
                        assert np.allclose(
                            nodes_gdf[acc_data_key_wt],
                            ac_data_wt[ac_idx][d_idx],
                            atol=config.ATOL,
                            rtol=config.RTOL,
                        )


def distance_generator() -> Generator[npt.NDArray[np.int_]]:  # type: ignore
    for distances in [[500], [500, 2000]]:
        yield distances


def test_hill_diversity(primal_graph):
    for distances in distance_generator():
        # prepare network and compute
        nodes_gdf, network_structure = graphs.network_structure_from_nx(primal_graph, 3395)
        data_gdf = mock.mock_landuse_categorical_data(primal_graph)
        _data_map, data_gdf = layers.assign_gdf_to_network(data_gdf, network_structure, 400)
        # easy version
        nodes_gdf_easy, _data_gdf_easy = layers.hill_diversity(
            data_gdf,
            "categorical_landuses",
            nodes_gdf,
            network_structure,
            max_netw_assign_dist=400,
            distances=distances,
            qs=[0, 1, 2],
        )
        # custom version
        nodes_gdf_full, _data_gdf_full = layers.compute_landuses(
            data_gdf,
            "categorical_landuses",
            nodes_gdf,
            network_structure,
            max_netw_assign_dist=400,
            mixed_use_keys=["hill"],
            distances=distances,
            qs=[0, 1, 2],
        )
        # compare
        for d_key in distances:
            for q_key in [0, 1, 2]:
                mu_hill_data_key = config.prep_gdf_key(f"hill_q{q_key}_{d_key}")
                assert np.allclose(
                    nodes_gdf_easy[mu_hill_data_key],  # type: ignore
                    nodes_gdf_full[mu_hill_data_key],  # type: ignore
                    atol=config.ATOL,
                    rtol=config.RTOL,
                )


def test_hill_branch_wt_diversity(primal_graph):
    for distances in distance_generator():
        # prepare network and compute
        nodes_gdf, network_structure = graphs.network_structure_from_nx(primal_graph, 3395)
        data_gdf = mock.mock_landuse_categorical_data(primal_graph)
        _data_map, data_gdf = layers.assign_gdf_to_network(data_gdf, network_structure, 400)
        # easy version
        nodes_gdf_easy, _data_gdf_easy = layers.hill_branch_wt_diversity(
            data_gdf,
            "categorical_landuses",
            nodes_gdf,
            network_structure,
            max_netw_assign_dist=400,
            distances=distances,
            qs=[0, 1, 2],
        )
        # custom version
        nodes_gdf_full, _data_gdf_full = layers.compute_landuses(
            data_gdf,
            "categorical_landuses",
            nodes_gdf,
            network_structure,
            max_netw_assign_dist=400,
            mixed_use_keys=["hill_branch_wt"],
            distances=distances,
            qs=[0, 1, 2],
        )
        # compare
        for d_key in distances:
            for q_key in [0, 1, 2]:
                mu_hill_data_key = config.prep_gdf_key(f"hill_branch_wt_q{q_key}_{d_key}")
                assert np.allclose(
                    nodes_gdf_easy[mu_hill_data_key],  # type: ignore
                    nodes_gdf_full[mu_hill_data_key],  # type: ignore
                    atol=config.ATOL,
                    rtol=config.RTOL,
                )


def test_compute_accessibilities(primal_graph):
    for distances in distance_generator():
        # prepare network and compute
        nodes_gdf, network_structure = graphs.network_structure_from_nx(primal_graph, 3395)
        data_gdf = mock.mock_landuse_categorical_data(primal_graph)
        _data_map, data_gdf = layers.assign_gdf_to_network(data_gdf, network_structure, 400)
        # easy version
        nodes_gdf_easy, _data_gdf_easy = layers.compute_accessibilities(
            data_gdf,
            "categorical_landuses",
            ["c"],
            nodes_gdf,
            network_structure,
            distances=distances,
        )
        # custom version
        nodes_gdf_full, _data_gdf_full = layers.compute_landuses(
            data_gdf,
            "categorical_landuses",
            nodes_gdf,
            network_structure,
            max_netw_assign_dist=400,
            accessibility_keys=["c"],
            distances=distances,
        )
        # compare
        for d_key in distances:
            acc_data_key_nw = config.prep_gdf_key(f"c_{d_key}_non_weighted")
            assert np.allclose(
                nodes_gdf_easy[acc_data_key_nw],  # type: ignore
                nodes_gdf_full[acc_data_key_nw],  # type: ignore
                atol=config.ATOL,
                rtol=config.RTOL,
            )
            acc_data_key_wt = config.prep_gdf_key(f"c_{d_key}_weighted")
            assert np.allclose(
                nodes_gdf_easy[acc_data_key_wt],  # type: ignore
                nodes_gdf_full[acc_data_key_wt],  # type: ignore
                atol=config.ATOL,
                rtol=config.RTOL,
            )


def test_compute_stats(primal_graph):
    """
    Test stats component
    """
    for distances in distance_generator():
        # prepare network and compute
        nodes_gdf, network_structure = graphs.network_structure_from_nx(primal_graph, 3395)
        data_gdf = mock.mock_numerical_data(primal_graph, num_arrs=2)
        data_map, data_gdf = layers.assign_gdf_to_network(data_gdf, network_structure, 400)
        # generate stats
        nodes_gdf, data_gdf = layers.compute_stats(
            data_gdf,
            ["mock_numerical_1", "mock_numerical_2"],
            nodes_gdf,
            network_structure,
            distances=distances,
        )
        # test against data computed directly from underlying methods
        stats_arrays = data_gdf[["mock_numerical_1", "mock_numerical_2"]].values.T
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
            network_structure,
            data_map,
            distances,
            networks.beta_from_distance(distances),
            numerical_arrays=stats_arrays,
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
        for stats_idx, stats_key in enumerate(["mock_numerical_1", "mock_numerical_2"]):
            for stats_type_key, stats in zip(stats_keys, stats_data):
                for d_idx, d_key in enumerate(distances):
                    stats_data_key = config.prep_gdf_key(f"{stats_key}_{stats_type_key}_{d_key}")
                    # check one-at-a-time computed vs multiply computed
                    assert np.allclose(
                        nodes_gdf[stats_data_key],
                        stats[stats_idx][d_idx],
                        atol=config.ATOL,
                        rtol=config.RTOL,
                        equal_nan=True,
                    )
    # check that problematic column labels are raised
    with pytest.raises(ValueError):
        layers.compute_stats(
            data_gdf,
            ["typo"],
            nodes_gdf,
            network_structure,
            distances=distances,
        )
