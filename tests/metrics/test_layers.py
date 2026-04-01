# pyright: basic
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from cityseer import config
from cityseer.metrics import layers
from cityseer.tools import io, mock
from shapely import geometry


def test_build_data_map(primal_graph):
    _nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(primal_graph)
    for typ in [int, float, str]:
        for data_id_col in [None, "data_id"]:
            data_gdf = mock.mock_data_gdf(primal_graph)
            data_gdf.index = data_gdf.index.astype(typ)
            for to_poly in [False, True]:
                # handle both points and polys
                if to_poly is True:
                    data_gdf.geometry = data_gdf.geometry.buffer(10)
                #
                data_map = layers.build_data_map(
                    data_gdf, network_structure, max_netw_assign_dist=400, data_id_col=data_id_col
                )
                for _data_key, data_entry in data_map.entries.items():
                    data_gdf_row = data_gdf.loc[data_entry.data_key_py]  # type: ignore
                    assert data_entry.geom_wkt == data_gdf_row.geometry.wkt
                    if data_id_col is not None:
                        assert data_entry.dedupe_key_py == data_gdf_row[data_id_col]
                    else:
                        assert data_entry.dedupe_key_py == data_entry.data_key_py
    # check with different geometry column name
    data_gdf = mock.mock_data_gdf(primal_graph)
    data_gdf.rename(columns={"geometry": "geom"}, inplace=True)
    data_gdf.set_geometry("geom", inplace=True)
    data_map = layers.build_data_map(data_gdf, network_structure, max_netw_assign_dist=400, data_id_col="data_id")
    # catch non unique indices
    data_gdf = gpd.GeoDataFrame(
        {
            "data_idx": [1, 2, 2],
            "geometry": [
                geometry.Point(0, 0),
                geometry.Point(1, 1),
                geometry.Point(2, 2),
            ],
        },
        crs="EPSG:3857",
    )
    data_gdf.set_index("data_idx", inplace=True)
    with pytest.raises(ValueError):
        data_map = layers.build_data_map(
            data_gdf,
            network_structure,
            max_netw_assign_dist=400,
        )


def test_compute_accessibilities(primal_graph, dual_graph):
    nodes_gdf_primal, _edges_gdf_primal, network_structure_primal = io.network_structure_from_nx(primal_graph)
    nodes_gdf_dual, _edges_gdf_dual, network_structure_dual = io.network_structure_from_nx(dual_graph)
    data_gdf = mock.mock_landuse_categorical_data(primal_graph)
    distances = [400, 800]
    max_assign_dist = 400
    for angular in [False, True]:
        nodes_gdf = nodes_gdf_dual.copy() if angular else nodes_gdf_primal.copy()
        network_structure = network_structure_dual if angular else network_structure_primal
        for data_id_col in [None, "data_id"]:
            for key_set in (["a"], ["b"], ["a", "b"]):
                nodes_gdf, data_gdf = layers.compute_accessibilities(
                    data_gdf,  # type: ignore
                    "categorical_landuses",
                    key_set,
                    nodes_gdf,  # type: ignore
                    network_structure,
                    max_netw_assign_dist=max_assign_dist,
                    distances=distances,
                    data_id_col=data_id_col,
                    angular=angular,
                )
                # test against manual implementation over underlying method
                landuses_map = dict(data_gdf["categorical_landuses"])  # type: ignore
                data_map = layers.build_data_map(
                    data_gdf,
                    network_structure,
                    max_netw_assign_dist=max_assign_dist,
                    data_id_col=data_id_col,
                )
                accessibility_data = data_map.accessibility(
                    network_structure,
                    landuses_map,  # type: ignore
                    key_set,
                    distances=distances,
                    angular=angular,
                )
                for acc_key in key_set:
                    for dist_key in distances:
                        acc_data_key = config.prep_gdf_key(acc_key, dist_key, angular)
                        assert np.allclose(
                            nodes_gdf[acc_data_key].values,  # type: ignore
                            accessibility_data.result[acc_key].count[dist_key],
                            atol=config.ATOL,
                            rtol=config.RTOL,
                            equal_nan=True,
                        )
                        acc_data_key_dist = config.prep_gdf_key(f"{acc_key}_nearest_max", dist_key, angular)
                        if dist_key == max(distances):
                            assert np.allclose(
                                nodes_gdf[acc_data_key_dist].values,  # type: ignore
                                accessibility_data.result[acc_key].distance[dist_key],
                                atol=config.ATOL,
                                rtol=config.RTOL,
                                equal_nan=True,
                            )
                        else:
                            assert acc_data_key_dist not in nodes_gdf.columns  # type: ignore
                # most integrity checks happen in underlying method
                with pytest.raises(ValueError):
                    nodes_gdf = layers.compute_accessibilities(
                        data_gdf,  # type: ignore
                        "categorical_landuses-TYPO",
                        ["c"],
                        nodes_gdf,  # type: ignore
                        network_structure,
                        max_netw_assign_dist=max_assign_dist,
                        distances=distances,
                    )


def test_compute_mixed_uses(primal_graph, dual_graph):
    nodes_gdf_primal, _edges_gdf_primal, network_structure_primal = io.network_structure_from_nx(primal_graph)
    nodes_gdf_dual, _edges_gdf_dual, network_structure_dual = io.network_structure_from_nx(dual_graph)
    data_gdf = mock.mock_landuse_categorical_data(primal_graph)
    distances = [400, 800]
    max_assign_dist = 400
    # test against manual implementation over underlying method
    for data_id_col in [None, "data_id"]:
        for angular in [False, True]:
            nodes_gdf = nodes_gdf_dual.copy() if angular else nodes_gdf_primal.copy()
            network_structure = network_structure_dual if angular else network_structure_primal
            nodes_gdf, data_gdf = layers.compute_mixed_uses(
                data_gdf,
                "categorical_landuses",
                nodes_gdf,
                network_structure,
                distances=distances,
                compute_hill=True,
                compute_shannon=True,
                compute_gini=True,
                data_id_col=data_id_col,
                angular=angular,
                max_netw_assign_dist=max_assign_dist,
            )
            # generate manually
            data_map = layers.build_data_map(
                data_gdf,
                network_structure,
                max_netw_assign_dist=max_assign_dist,
                data_id_col=data_id_col,
            )
            landuses_map = dict(data_gdf["categorical_landuses"])
            mu_data = data_map.mixed_uses(
                network_structure,
                landuses_map,
                compute_hill=True,
                compute_shannon=True,
                compute_gini=True,
                distances=distances,
                angular=angular,
            )
            for dist_key in distances:
                for q_key in [0, 1, 2]:
                    hill_data_key = config.prep_gdf_key(f"hill_q{q_key}", dist_key, angular=angular)
                    assert np.allclose(
                        nodes_gdf[hill_data_key].values,
                        mu_data.hill[q_key][dist_key],
                        atol=config.ATOL,
                        rtol=config.RTOL,
                    )
                shannon_data_key = config.prep_gdf_key("shannon", dist_key, angular=angular)
                assert np.allclose(
                    nodes_gdf[shannon_data_key].values,
                    mu_data.shannon[dist_key],
                    atol=config.ATOL,
                    rtol=config.RTOL,
                )
                gini_data_key = config.prep_gdf_key("gini", dist_key, angular=angular)
                assert np.allclose(
                    nodes_gdf[gini_data_key].values,
                    mu_data.gini[dist_key],
                    atol=config.ATOL,
                    rtol=config.RTOL,
                )


def test_compute_stats(primal_graph, dual_graph):
    """
    Test stats component
    """
    nodes_gdf_primal, _edges_gdf_primal, network_structure_primal = io.network_structure_from_nx(primal_graph)
    nodes_gdf_dual, _edges_gdf_dual, network_structure_dual = io.network_structure_from_nx(dual_graph)
    data_gdf = mock.mock_numerical_data(primal_graph, num_arrs=2)
    max_assign_dist = 400
    distances = [400, 800]
    for _data_id_col in [None, "data_id"]:
        for angular in [False, True]:
            nodes_gdf = nodes_gdf_dual.copy() if angular else nodes_gdf_primal.copy()
            network_structure = network_structure_dual if angular else network_structure_primal
            data_map = layers.build_data_map(
                data_gdf,
                network_structure,
                max_netw_assign_dist=max_assign_dist,
                data_id_col=None,
            )
            nodes_gdf, data_gdf = layers.compute_stats(
                data_gdf,
                ["mock_numerical_1", "mock_numerical_2"],
                nodes_gdf,
                network_structure,
                distances=distances,
                angular=angular,
                max_netw_assign_dist=max_assign_dist,
            )
            # compare to manual
            for stats_key in ["mock_numerical_1", "mock_numerical_2"]:
                stats_map = dict(data_gdf[stats_key])  # type: ignore
                # generate stats
                stats_results = data_map.stats(
                    network_structure,
                    numerical_maps=[stats_map],
                    distances=distances,
                    angular=angular,
                )
                stats_result = stats_results.result[0]
                for dist_key in distances:
                    assert np.allclose(
                        nodes_gdf[config.prep_gdf_key(f"{stats_key}_sum", dist_key, angular=angular)],
                        stats_result.sum[dist_key],
                        atol=config.ATOL,
                        rtol=config.RTOL,
                        equal_nan=True,
                    )
                    assert np.allclose(
                        nodes_gdf[config.prep_gdf_key(f"{stats_key}_mean", dist_key, angular=angular)],
                        stats_result.mean[dist_key],
                        atol=config.ATOL,
                        rtol=config.RTOL,
                        equal_nan=True,
                    )
                    assert np.allclose(
                        nodes_gdf[config.prep_gdf_key(f"{stats_key}_count", dist_key, angular=angular)],
                        stats_result.count[dist_key],
                        atol=config.ATOL,
                        rtol=config.RTOL,
                        equal_nan=True,
                    )
                    assert np.allclose(
                        nodes_gdf[config.prep_gdf_key(f"{stats_key}_var", dist_key, angular=angular)],
                        stats_result.variance[dist_key],
                        atol=config.ATOL,
                        rtol=config.RTOL,
                        equal_nan=True,
                    )
                    assert np.allclose(
                        nodes_gdf[config.prep_gdf_key(f"{stats_key}_max", dist_key, angular=angular)],
                        stats_result.max[dist_key],
                        atol=config.ATOL,
                        rtol=config.RTOL,
                        equal_nan=True,
                    )
                    assert np.allclose(
                        nodes_gdf[config.prep_gdf_key(f"{stats_key}_min", dist_key, angular=angular)],
                        stats_result.min[dist_key],
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


def test_angular_layer_wrappers_require_dual_graph(primal_graph):
    nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(primal_graph)
    landuse_gdf = mock.mock_landuse_categorical_data(primal_graph)
    numerical_gdf = mock.mock_numerical_data(primal_graph, num_arrs=1)
    with pytest.raises(ValueError, match="dual graph"):
        layers.compute_accessibilities(
            landuse_gdf,
            "categorical_landuses",
            ["a"],
            nodes_gdf.copy(),
            network_structure,
            distances=[400],
            angular=True,
        )
    with pytest.raises(ValueError, match="dual graph"):
        layers.compute_mixed_uses(
            landuse_gdf,
            "categorical_landuses",
            nodes_gdf.copy(),
            network_structure,
            distances=[400],
            angular=True,
        )
    with pytest.raises(ValueError, match="dual graph"):
        layers.compute_stats(
            numerical_gdf,
            ["mock_numerical_1"],
            nodes_gdf.copy(),
            network_structure,
            distances=[400],
            angular=True,
        )


def test_custom_decay_fn(primal_graph):
    """Test that custom decay_fn expressions produce different results from default."""
    from cityseer import decay

    nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(primal_graph)
    numerical_gdf = mock.mock_numerical_data(primal_graph, num_arrs=1)
    landuse_gdf = mock.mock_landuse_categorical_data(primal_graph)
    distances = [800]
    col_mean = config.prep_gdf_key("mock_numerical_1_mean", 800)
    col_acc = config.prep_gdf_key("a", 800)
    # --- compute_stats ---
    n_default, _ = layers.compute_stats(
        numerical_gdf, ["mock_numerical_1"], nodes_gdf.copy(), network_structure, distances=distances
    )
    n_exp, _ = layers.compute_stats(
        numerical_gdf,
        ["mock_numerical_1"],
        nodes_gdf.copy(),
        network_structure,
        distances=distances,
        decay_fn=decay.exponential(),
    )
    n_linear, _ = layers.compute_stats(
        numerical_gdf,
        ["mock_numerical_1"],
        nodes_gdf.copy(),
        network_structure,
        distances=distances,
        decay_fn=decay.linear(),
    )
    # default is flat ("1"), so default and exponential should differ
    assert not np.allclose(n_default[col_mean].dropna(), n_exp[col_mean].dropna(), atol=0.1)
    # default and linear should differ
    assert not np.allclose(n_default[col_mean].dropna(), n_linear[col_mean].dropna(), atol=0.1)
    # exponential and linear should also differ from each other
    assert not np.allclose(n_exp[col_mean].dropna(), n_linear[col_mean].dropna(), atol=0.1)
    # --- compute_accessibilities ---
    a_default, _ = layers.compute_accessibilities(
        landuse_gdf, "categorical_landuses", ["a"], nodes_gdf.copy(), network_structure, distances=distances
    )
    a_linear, _ = layers.compute_accessibilities(
        landuse_gdf,
        "categorical_landuses",
        ["a"],
        nodes_gdf.copy(),
        network_structure,
        distances=distances,
        decay_fn=decay.linear(),
    )
    assert not np.allclose(a_default[col_acc].dropna(), a_linear[col_acc].dropna(), atol=0.1)
    # --- helper-generated expressions ---
    n_gauss, _ = layers.compute_stats(
        numerical_gdf,
        ["mock_numerical_1"],
        nodes_gdf.copy(),
        network_structure,
        distances=[1200],
        decay_fn=decay.gaussian(peak=400, cutoff=1200, std=150),
    )
    col_gauss = config.prep_gdf_key("mock_numerical_1_mean", 1200)
    assert not n_gauss[col_gauss].dropna().empty
    # --- invalid expression ---
    with pytest.raises(ValueError, match="parse"):
        layers.compute_stats(
            numerical_gdf,
            ["mock_numerical_1"],
            nodes_gdf.copy(),
            network_structure,
            distances=distances,
            decay_fn="invalid !! expression",
        )
