# pyright: basic
from __future__ import annotations

from typing import Generator

import numpy as np
import numpy.typing as npt
import pytest
from sklearn.preprocessing import LabelEncoder  # type: ignore

from cityseer import config, rustalgos
from cityseer.metrics import layers, networks
from cityseer.tools import graphs, mock


def test_assign_gdf_to_network(primal_graph):
    _nodes_gdf, _edges_gdf, network_structure = graphs.network_structure_from_nx(primal_graph, 3395)
    data_gdf = mock.mock_data_gdf(primal_graph)
    data_map, data_gdf = layers.assign_gdf_to_network(data_gdf, network_structure, 400, data_id_col="data_id")
    # check assignments
    for data_key in data_map.entry_keys():
        data_entry = data_map.get_entry(data_key)
        # compute manually
        nearest_idx, next_nearest_idx = network_structure.assign_to_network(data_entry.coord, 400)
        assert nearest_idx == data_entry.nearest_assign
        assert next_nearest_idx == data_entry.next_nearest_assign
        assert data_gdf.at[data_key, "nearest_assign"] == data_entry.nearest_assign
        assert data_gdf.at[data_key, "next_nearest_assign"] == data_entry.next_nearest_assign
        assert str(data_gdf.at[data_key, "data_id"]) == data_entry.data_id
    assert data_map.all_assigned()


def test_compute_accessibilities(primal_graph):
    nodes_gdf, _edges_gdf, network_structure = graphs.network_structure_from_nx(primal_graph, 3395)
    data_gdf = mock.mock_landuse_categorical_data(primal_graph)
    distances = [400, 800]
    max_assign_dist = 400
    for angular in [False, True]:
        for data_id_col in [None, "data_id"]:
            for key_set in (["a"], ["b"], ["a", "b"]):
                nodes_gdf, data_gdf = layers.compute_accessibilities(
                    data_gdf,
                    "categorical_landuses",
                    key_set,
                    nodes_gdf,
                    network_structure,
                    max_netw_assign_dist=max_assign_dist,
                    distances=distances,
                    data_id_col=data_id_col,
                    angular=angular,
                )
                # test against manual implementation over underlying method
                landuses_map = data_gdf["categorical_landuses"].to_dict()
                data_map, data_gdf = layers.assign_gdf_to_network(
                    data_gdf,
                    network_structure,
                    max_assign_dist,
                    data_id_col=data_id_col,
                )
                # accessibilities
                accessibility_data = data_map.accessibility(
                    network_structure,
                    landuses_map,
                    key_set,
                    distances,
                    angular=angular,
                )
                for acc_key in key_set:
                    for dist_key in distances:
                        acc_data_key_nw = config.prep_gdf_key(f"{acc_key}_{dist_key}_non_weighted")
                        assert np.allclose(
                            nodes_gdf[acc_data_key_nw].values,
                            accessibility_data[acc_key].unweighted[dist_key],
                            atol=config.ATOL,
                            rtol=config.RTOL,
                        )
                        acc_data_key_wt = config.prep_gdf_key(f"{acc_key}_{dist_key}_weighted")
                        assert np.allclose(
                            nodes_gdf[acc_data_key_wt].values,
                            accessibility_data[acc_key].weighted[dist_key],
                            atol=config.ATOL,
                            rtol=config.RTOL,
                        )
                # most integrity checks happen in underlying method
                with pytest.raises(ValueError):
                    nodes_gdf, data_gdf = layers.compute_accessibilities(
                        data_gdf,
                        "categorical_landuses-TYPO",
                        ["c"],
                        nodes_gdf,
                        network_structure,
                        max_netw_assign_dist=max_assign_dist,
                        distances=distances,
                    )


def test_compute_mixed_uses(primal_graph):
    nodes_gdf, _edges_gdf, network_structure = graphs.network_structure_from_nx(primal_graph, 3395)
    data_gdf = mock.mock_landuse_categorical_data(primal_graph)
    distances = [400, 800]
    max_assign_dist = 400
    # test against manual implementation over underlying method
    max_dist = max(distances)
    landuses_map = data_gdf["categorical_landuses"].to_dict()
    for data_id_col in [None, "data_id"]:
        for angular in [False, True]:
            nodes_gdf, data_gdf = layers.compute_mixed_uses(
                data_gdf,
                "categorical_landuses",
                nodes_gdf,
                network_structure,
                max_netw_assign_dist=max_assign_dist,
                distances=distances,
                compute_hill=True,
                compute_hill_weighted=True,
                compute_shannon=True,
                compute_gini=True,
                data_id_col=data_id_col,
                angular=angular,
            )
            # generate manually
            data_map, data_gdf = layers.assign_gdf_to_network(
                data_gdf, network_structure, max_dist, data_id_col=data_id_col
            )
            mu_data = data_map.mixed_uses(
                network_structure,
                landuses_map,
                compute_hill=True,
                compute_hill_weighted=True,
                compute_shannon=True,
                compute_gini=True,
                distances=distances,
                angular=angular,
            )
            for dist_key in distances:
                for q_key in [0, 1, 2]:
                    hill_nw_data_key = config.prep_gdf_key(f"q{q_key}_{dist_key}_hill")
                    assert np.allclose(
                        nodes_gdf[hill_nw_data_key].values,
                        mu_data.hill[q_key][dist_key],
                        atol=config.ATOL,
                        rtol=config.RTOL,
                    )
                    hill_wt_data_key = config.prep_gdf_key(f"q{q_key}_{dist_key}_hill_weighted")
                    assert np.allclose(
                        nodes_gdf[hill_wt_data_key].values,
                        mu_data.hill_weighted[q_key][dist_key],
                        atol=config.ATOL,
                        rtol=config.RTOL,
                    )
                shannon_data_key = config.prep_gdf_key(f"{dist_key}_shannon")
                assert np.allclose(
                    nodes_gdf[shannon_data_key].values,
                    mu_data.shannon[dist_key],
                    atol=config.ATOL,
                    rtol=config.RTOL,
                )
                gini_data_key = config.prep_gdf_key(f"{dist_key}_gini")
                assert np.allclose(
                    nodes_gdf[gini_data_key].values,
                    mu_data.gini[dist_key],
                    atol=config.ATOL,
                    rtol=config.RTOL,
                )


def test_compute_stats(primal_graph):
    """
    Test stats component
    """
    nodes_gdf, _edges_gdf, network_structure = graphs.network_structure_from_nx(primal_graph, 3395)
    data_gdf = mock.mock_numerical_data(primal_graph, num_arrs=1)
    max_assign_dist = 400
    data_map, data_gdf = layers.assign_gdf_to_network(data_gdf, network_structure, max_assign_dist)
    # test against manual implementation over underlying method
    distances = [400, 800]
    for data_id_col in [None, "data_id"]:
        for angular in [False, True]:
            nodes_gdf, data_gdf = layers.compute_stats(
                data_gdf,
                "mock_numerical_1",
                nodes_gdf,
                network_structure,
                max_assign_dist,
                distances=distances,
            )
            # generate stats
            # compare to manual
            numerical_map = data_gdf["mock_numerical_1"].to_dict()
            stats_result = data_map.stats(
                network_structure,
                numerical_map=numerical_map,
                distances=distances,
            )
            for dist_key in distances:
                assert np.allclose(
                    nodes_gdf[config.prep_gdf_key(f"sum_{dist_key}")],
                    stats_result.sum[dist_key],
                    atol=config.ATOL,
                    rtol=config.RTOL,
                    equal_nan=True,
                )
                assert np.allclose(
                    nodes_gdf[config.prep_gdf_key(f"sum_wt_{dist_key}")],
                    stats_result.sum_wt[dist_key],
                    atol=config.ATOL,
                    rtol=config.RTOL,
                    equal_nan=True,
                )
                assert np.allclose(
                    nodes_gdf[config.prep_gdf_key(f"mean_{dist_key}")],
                    stats_result.mean[dist_key],
                    atol=config.ATOL,
                    rtol=config.RTOL,
                    equal_nan=True,
                )
                assert np.allclose(
                    nodes_gdf[config.prep_gdf_key(f"mean_wt_{dist_key}")],
                    stats_result.mean_wt[dist_key],
                    atol=config.ATOL,
                    rtol=config.RTOL,
                    equal_nan=True,
                )
                assert np.allclose(
                    nodes_gdf[config.prep_gdf_key(f"count_{dist_key}")],
                    stats_result.count[dist_key],
                    atol=config.ATOL,
                    rtol=config.RTOL,
                    equal_nan=True,
                )
                assert np.allclose(
                    nodes_gdf[config.prep_gdf_key(f"count_wt_{dist_key}")],
                    stats_result.count_wt[dist_key],
                    atol=config.ATOL,
                    rtol=config.RTOL,
                    equal_nan=True,
                )
                assert np.allclose(
                    nodes_gdf[config.prep_gdf_key(f"variance_{dist_key}")],
                    stats_result.variance[dist_key],
                    atol=config.ATOL,
                    rtol=config.RTOL,
                    equal_nan=True,
                )
                assert np.allclose(
                    nodes_gdf[config.prep_gdf_key(f"variance_wt_{dist_key}")],
                    stats_result.variance_wt[dist_key],
                    atol=config.ATOL,
                    rtol=config.RTOL,
                    equal_nan=True,
                )
                assert np.allclose(
                    nodes_gdf[config.prep_gdf_key(f"max_{dist_key}")],
                    stats_result.max[dist_key],
                    atol=config.ATOL,
                    rtol=config.RTOL,
                    equal_nan=True,
                )
                assert np.allclose(
                    nodes_gdf[config.prep_gdf_key(f"min_{dist_key}")],
                    stats_result.min[dist_key],
                    atol=config.ATOL,
                    rtol=config.RTOL,
                    equal_nan=True,
                )
    # check that problematic column labels are raised
    with pytest.raises(ValueError):
        layers.compute_stats(
            data_gdf,
            "typo",
            nodes_gdf,
            network_structure,
            distances=distances,
        )
