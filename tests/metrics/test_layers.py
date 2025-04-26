# pyright: basic
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from cityseer import config
from cityseer.metrics import layers
from cityseer.tools import io, mock
from shapely import geometry


def test_decompose_gdf(primal_graph):
    # Create a diverse GeoDataFrame
    gdf = gpd.GeoDataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "value": ["a", "b", "c", "d", "e", "f", "g", "h", "i"],
            "geometry": [
                geometry.Point(0, 0),  # Point
                geometry.LineString([(10, 10), (25, 10)]),  # LineString (length 15)
                geometry.Polygon([(30, 30), (40, 30), (40, 40), (30, 40)]),  # Polygon (perimeter 40)
                geometry.MultiPoint([(50, 50), (51, 51)]),  # MultiPoint
                geometry.LineString([(60, 60), (60, 60)]),  # Zero-length LineString
                geometry.MultiPolygon([geometry.Polygon([(70, 70), (80, 70), (80, 80), (70, 80)])]),  # MultiPolygon
                geometry.GeometryCollection(
                    [geometry.Point(90, 90), geometry.LineString([(95, 95), (105, 95)])]
                ),  # GeometryCollection
                None,  # None geometry
                geometry.LineString(),  # Empty geometry
            ],
        },
        crs="EPSG:3857",
        index=pd.Index([10, 20, 30, 40, 50, 60, 70, 80, 90], name="original_index"),  # Use a named index
    )

    sampling_distance = 5.0
    decomposed_gdf = layers.decompose_gdf(gdf, distance=sampling_distance)

    # --- Assertions on the output ---
    assert isinstance(decomposed_gdf, gpd.GeoDataFrame)
    assert decomposed_gdf.crs == gdf.crs
    assert not decomposed_gdf.empty

    # Check columns
    assert "src_fid" in decomposed_gdf.columns
    assert "id" in decomposed_gdf.columns
    assert "value" in decomposed_gdf.columns
    assert decomposed_gdf.geometry.name in decomposed_gdf.columns
    assert len(decomposed_gdf.columns) == 4  # id, value, geometry, src_fid

    # Check geometry types
    assert all(decomposed_gdf.geometry.geom_type == "Point")

    # Check index is reset
    assert decomposed_gdf.index.is_unique
    assert list(decomposed_gdf.index) == list(range(len(decomposed_gdf)))

    # Check src_fid and attribute transfer
    original_indices = gdf.index.tolist()
    for src_fid in original_indices:
        if src_fid == 80 or src_fid == 90:  # Skip None/Empty geometry tests here
            assert src_fid not in decomposed_gdf["src_fid"].values
            continue
        original_row = gdf.loc[src_fid]
        subset_gdf = decomposed_gdf[decomposed_gdf["src_fid"] == src_fid]
        assert not subset_gdf.empty
        assert all(subset_gdf["id"] == original_row["id"])
        assert all(subset_gdf["value"] == original_row["value"])

    # Check number of points (approximate for sampled geoms)
    # Point (src_fid=10): 1 point
    assert len(decomposed_gdf[decomposed_gdf["src_fid"] == 10]) == 1
    # LineString (src_fid=20, length 15, dist 5): ~ 15/5 + 1 = 4 points (linspace includes ends)
    assert len(decomposed_gdf[decomposed_gdf["src_fid"] == 20]) == int(np.ceil(15 / sampling_distance)) + 1
    # Polygon (src_fid=30, perimeter 40, dist 5): ~ 40/5 + 1 = 9 points for exterior
    assert len(decomposed_gdf[decomposed_gdf["src_fid"] == 30]) == int(np.ceil(40 / sampling_distance)) + 1
    # MultiPoint (src_fid=40): 2 points
    assert len(decomposed_gdf[decomposed_gdf["src_fid"] == 40]) == 2
    # Zero-length LineString (src_fid=50): 1 point
    assert len(decomposed_gdf[decomposed_gdf["src_fid"] == 50]) == 1
    # MultiPolygon (src_fid=60, 1 poly, perimeter 40, dist 5): ~ 40/5 + 1 = 9 points
    assert len(decomposed_gdf[decomposed_gdf["src_fid"] == 60]) == int(np.ceil(40 / sampling_distance)) + 1
    # GeometryCollection (src_fid=70, 1 point + 1 line length 10): 1 + (10/5 + 1) = 4 points
    assert len(decomposed_gdf[decomposed_gdf["src_fid"] == 70]) == 1 + (int(np.ceil(10 / sampling_distance)) + 1)
    # None geometry (src_fid=80): 0 points
    assert len(decomposed_gdf[decomposed_gdf["src_fid"] == 80]) == 0
    # Empty geometry (src_fid=90): 0 points
    assert len(decomposed_gdf[decomposed_gdf["src_fid"] == 90]) == 0

    # --- Test Edge Cases ---
    # Empty GeoDataFrame
    empty_gdf = gpd.GeoDataFrame([], columns=["id", "geometry"], crs="EPSG:3857")
    decomposed_empty = layers.decompose_gdf(empty_gdf)
    assert isinstance(decomposed_empty, gpd.GeoDataFrame)
    assert decomposed_empty.empty
    assert "src_fid" in decomposed_empty.columns
    assert "id" in decomposed_empty.columns
    assert decomposed_empty.geometry.name in decomposed_empty.columns

    # GeoDataFrame with only Points
    points_gdf = gdf[gdf.geometry.geom_type == "Point"].copy()
    decomposed_points = layers.decompose_gdf(points_gdf)
    assert len(decomposed_points) == len(points_gdf)
    assert all(decomposed_points.geometry.geom_type == "Point")
    assert all(decomposed_points["src_fid"] == points_gdf.index)

    # --- Test Error Handling ---
    # Non-positive distance
    with pytest.raises(ValueError, match="Sampling distance must be greater than 5."):
        layers.decompose_gdf(gdf, distance=0)
    with pytest.raises(ValueError, match="Sampling distance must be greater than 5."):
        layers.decompose_gdf(gdf, distance=-5)
    with pytest.raises(ValueError, match="Sampling distance must be greater than 5."):
        layers.decompose_gdf(gdf, distance=2)


def test_assign_gdf_to_network(primal_graph):
    _nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(primal_graph)
    for typ in [int, float, str]:
        data_gdf = mock.mock_data_gdf(primal_graph)
        data_gdf.index = data_gdf.index.astype(typ)
        for to_poly in [False, True]:
            # handle both points and polys
            if to_poly is True:
                data_gdf.geometry = data_gdf.geometry.buffer(10)
            #
            data_map = layers.assign_gdf_to_network(data_gdf, network_structure, 400, data_id_col="data_id")
            assert data_map.assigned_to_network is True
            for _data_key, data_entry in data_map.entries.items():
                assert data_gdf.loc[data_entry.data_key_py].geometry.centroid.x - data_entry.coord.x < 1
                assert data_gdf.loc[data_entry.data_key_py].geometry.centroid.y - data_entry.coord.y < 1
                assert data_entry.node_matches is not None
                assert data_entry.node_matches.nearest is not None
    # test with barriers against manual - already tested in test_assign_to_network
    data_gdf = mock.mock_data_gdf(primal_graph)
    data_map_manual = mock.mock_data_map(data_gdf, with_barriers=True)
    data_map_manual.assign_to_network(network_structure, max_dist=400)
    # auto via layers
    barriers_gdf = mock.mock_barriers_gdf()
    data_map_auto = layers.assign_gdf_to_network(data_gdf, network_structure, 400, barriers_gdf=barriers_gdf)
    for data_key, data_entry_auto in data_map_auto.entries.items():
        data_entry_manual = data_map_manual.entries[data_key]
        # check if the data entry is the same
        if data_entry_auto.node_matches.nearest is not None:
            assert data_entry_auto.node_matches.nearest.idx == data_entry_manual.node_matches.nearest.idx
        else:
            assert data_entry_manual.node_matches.nearest is None
        #
        if data_entry_auto.node_matches.next_nearest is not None:
            assert data_entry_auto.node_matches.next_nearest.idx == data_entry_manual.node_matches.next_nearest.idx
        else:
            assert data_entry_manual.node_matches.next_nearest is None
    # check with different geometry column name
    data_gdf = mock.mock_data_gdf(primal_graph)
    data_gdf.rename(columns={"geometry": "geom"}, inplace=True)
    data_gdf.set_geometry("geom", inplace=True)
    data_map = layers.assign_gdf_to_network(data_gdf, network_structure, 400, data_id_col="data_id")
    # catch non unique indices
    data_gdf = gpd.GeoDataFrame(
        {
            "data_id": [1, 2, 2],
            "geometry": [
                geometry.Point(0, 0),
                geometry.Point(1, 1),
                geometry.Point(2, 2),
            ],
        },
        crs="EPSG:3857",
    )
    data_gdf.set_index("data_id", inplace=True)
    with pytest.raises(ValueError):
        data_map = layers.assign_gdf_to_network(data_gdf, network_structure, 400)
    # catch duplicate geom types
    data_gdf = mock.mock_data_gdf(primal_graph)
    data_gdf.geometry[0] = data_gdf.geometry[0].buffer(10)
    with pytest.raises(ValueError):
        data_map = layers.assign_gdf_to_network(data_gdf, network_structure, 400)


def test_compute_accessibilities(primal_graph):
    nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(primal_graph)
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
                data_map = layers.assign_gdf_to_network(
                    data_gdf,
                    network_structure,
                    max_assign_dist,
                    data_id_col=data_id_col,
                )
                # accessibilities
                landuses_map = dict(data_gdf["categorical_landuses"])
                accessibility_data = data_map.accessibility(
                    network_structure,
                    landuses_map,
                    key_set,
                    distances,
                    angular=angular,
                )
                for acc_key in key_set:
                    for dist_key in distances:
                        acc_data_key_nw = config.prep_gdf_key(acc_key, dist_key, angular, weighted=False)
                        assert np.allclose(
                            nodes_gdf[acc_data_key_nw].values,
                            accessibility_data[acc_key].unweighted[dist_key],
                            atol=config.ATOL,
                            rtol=config.RTOL,
                            equal_nan=True,
                        )
                        acc_data_key_wt = config.prep_gdf_key(acc_key, dist_key, angular, weighted=True)
                        assert np.allclose(
                            nodes_gdf[acc_data_key_wt].values,
                            accessibility_data[acc_key].weighted[dist_key],
                            atol=config.ATOL,
                            rtol=config.RTOL,
                            equal_nan=True,
                        )
                        acc_data_key_dist = config.prep_gdf_key(f"{acc_key}_nearest_max", dist_key, angular)
                        if dist_key == max(distances):
                            assert np.allclose(
                                nodes_gdf[acc_data_key_dist].values,
                                accessibility_data[acc_key].distance[dist_key],
                                atol=config.ATOL,
                                rtol=config.RTOL,
                                equal_nan=True,
                            )
                        else:
                            assert acc_data_key_dist not in nodes_gdf.columns
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
    nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(primal_graph)
    data_gdf = mock.mock_landuse_categorical_data(primal_graph)
    distances = [400, 800]
    max_assign_dist = 400
    # test against manual implementation over underlying method
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
            data_map = layers.assign_gdf_to_network(
                data_gdf, network_structure, max_assign_dist, data_id_col=data_id_col
            )
            landuses_map = dict(data_gdf["categorical_landuses"])
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
                    hill_nw_data_key = config.prep_gdf_key(f"hill_q{q_key}", dist_key, angular=angular, weighted=False)
                    assert np.allclose(
                        nodes_gdf[hill_nw_data_key].values,
                        mu_data.hill[q_key][dist_key],
                        atol=config.ATOL,
                        rtol=config.RTOL,
                    )
                    hill_wt_data_key = config.prep_gdf_key(f"hill_q{q_key}", dist_key, angular=angular, weighted=True)
                    assert np.allclose(
                        nodes_gdf[hill_wt_data_key].values,
                        mu_data.hill_weighted[q_key][dist_key],
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


def test_compute_stats(primal_graph):
    """
    Test stats component
    """
    nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(primal_graph)
    data_gdf = mock.mock_numerical_data(primal_graph, num_arrs=2)
    max_assign_dist = 400
    data_map = layers.assign_gdf_to_network(data_gdf, network_structure, max_assign_dist)
    # test against manual implementation over underlying method
    distances = [400, 800]
    for _data_id_col in [None, "data_id"]:
        for angular in [False, True]:
            nodes_gdf, data_gdf = layers.compute_stats(
                data_gdf,
                ["mock_numerical_1", "mock_numerical_2"],
                nodes_gdf,
                network_structure,
                max_assign_dist,
                distances=distances,
                angular=angular,
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
                stats_result = stats_results[0]
                for dist_key in distances:
                    assert np.allclose(
                        nodes_gdf[config.prep_gdf_key(f"{stats_key}_sum", dist_key, angular=angular, weighted=False)],
                        stats_result.sum[dist_key],
                        atol=config.ATOL,
                        rtol=config.RTOL,
                        equal_nan=True,
                    )
                    assert np.allclose(
                        nodes_gdf[config.prep_gdf_key(f"{stats_key}_sum", dist_key, angular=angular, weighted=True)],
                        stats_result.sum_wt[dist_key],
                        atol=config.ATOL,
                        rtol=config.RTOL,
                        equal_nan=True,
                    )
                    assert np.allclose(
                        nodes_gdf[config.prep_gdf_key(f"{stats_key}_mean", dist_key, angular=angular, weighted=False)],
                        stats_result.mean[dist_key],
                        atol=config.ATOL,
                        rtol=config.RTOL,
                        equal_nan=True,
                    )
                    assert np.allclose(
                        nodes_gdf[config.prep_gdf_key(f"{stats_key}_mean", dist_key, angular=angular, weighted=True)],
                        stats_result.mean_wt[dist_key],
                        atol=config.ATOL,
                        rtol=config.RTOL,
                        equal_nan=True,
                    )
                    assert np.allclose(
                        nodes_gdf[config.prep_gdf_key(f"{stats_key}_count", dist_key, angular=angular, weighted=False)],
                        stats_result.count[dist_key],
                        atol=config.ATOL,
                        rtol=config.RTOL,
                        equal_nan=True,
                    )
                    assert np.allclose(
                        nodes_gdf[config.prep_gdf_key(f"{stats_key}_count", dist_key, angular=angular, weighted=True)],
                        stats_result.count_wt[dist_key],
                        atol=config.ATOL,
                        rtol=config.RTOL,
                        equal_nan=True,
                    )
                    assert np.allclose(
                        nodes_gdf[config.prep_gdf_key(f"{stats_key}_var", dist_key, angular=angular, weighted=False)],
                        stats_result.variance[dist_key],
                        atol=config.ATOL,
                        rtol=config.RTOL,
                        equal_nan=True,
                    )
                    assert np.allclose(
                        nodes_gdf[config.prep_gdf_key(f"{stats_key}_var", dist_key, angular=angular, weighted=True)],
                        stats_result.variance_wt[dist_key],
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
