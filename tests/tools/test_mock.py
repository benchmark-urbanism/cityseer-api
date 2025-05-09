# pyright: basic
from __future__ import annotations

import string

import networkx as nx
import numpy as np
import pytest
from cityseer import config
from cityseer.tools import mock


def test_mock_graph(primal_graph):
    G = mock.mock_graph()
    G_wgs = mock.mock_graph(wgs84_coords=True)
    for graph in [G, G_wgs, primal_graph]:  # type: ignore
        assert graph.number_of_nodes() == 57
        assert graph.number_of_edges() == 79
        assert nx.average_degree_connectivity(graph) == {
            4: 3.0,
            3: 3.0303030303030303,
            2: 2.4,
            1: 2.0,
            0: 0,
        }
        for _node_key, node_data in graph.nodes(data=True):  # type: ignore
            assert "x" in node_data and isinstance(node_data["y"], int | float)
            assert "y" in node_data and isinstance(node_data["y"], int | float)
    # from cityseer.tools import plot
    # plot.plot_nx(G)


def test_mock_data_gdf(primal_graph):
    min_x = np.inf
    max_x = -np.inf
    min_y = np.inf
    max_y = -np.inf
    for _node_key, node_data in primal_graph.nodes(data=True):  # type: ignore
        if node_data["x"] < min_x:
            min_x = node_data["x"]
        if node_data["x"] > max_x:
            max_x = node_data["x"]
        if node_data["y"] < min_y:
            min_y = node_data["y"]
        if node_data["y"] > max_y:
            max_y = node_data["y"]
    data_gdf = mock.mock_data_gdf(primal_graph)
    assert "geometry" in data_gdf.columns
    assert np.all(data_gdf.geometry.x >= min_x)
    assert np.all(data_gdf.geometry.x <= max_x)
    assert np.all(data_gdf.geometry.y >= min_y)
    assert np.all(data_gdf.geometry.y <= max_y)


def test_mock_landuse_categorical_data(primal_graph):
    categorical_gdf = mock.mock_landuse_categorical_data(primal_graph, length=50, num_classes=10)
    assert len(categorical_gdf) == 50
    # classes are generated randomly from max number of classes
    # i.e. situations do exist where the number of classes will be less than the max permitted
    # use large enough max to reduce likelihood of this triggering issue for test
    assert len(set(categorical_gdf.categorical_landuses)) == 10
    for cat in categorical_gdf.categorical_landuses:
        assert isinstance(cat, str)
        assert cat in string.ascii_lowercase or cat == "z"
    categorical_gdf = mock.mock_landuse_categorical_data(primal_graph, length=50, num_classes=3)
    assert len(set(categorical_gdf.categorical_landuses)) == 3
    # test that an error is raised when requesting more than available max classes per asii_lowercase
    with pytest.raises(ValueError):
        mock.mock_landuse_categorical_data(primal_graph, length=50, num_classes=len(string.ascii_lowercase) + 1)


def test_mock_numerical_data(primal_graph):
    for length in [50, 100]:
        for num_arrs in range(1, 3):
            numerical_gdf = mock.mock_numerical_data(primal_graph, length=length, num_arrs=num_arrs)
            assert len(numerical_gdf.columns) == num_arrs + 2
            for col_label in numerical_gdf.columns:
                if col_label in ["uid", "geometry"]:
                    continue
                assert np.all(numerical_gdf[col_label] >= 0)
                assert np.all(numerical_gdf[col_label] <= 100000)


def test_mock_species_data():
    for counts, probs in mock.mock_species_data():
        counts = np.array(counts)
        probs = np.array(probs)
        assert np.allclose(counts / counts.sum(), probs, atol=config.ATOL, rtol=config.RTOL)
        assert round(probs.sum(), 8) == 1


def test_mock_osm_data():
    pass


def test_mock_data_map(primal_graph):
    """
    Test the mock_data_map function to ensure it creates a DataMap from a GeoDataFrame.
    """
    for length in [10, 20]:
        for random_seed in [0, 42]:
            data_gdf = mock.mock_data_gdf(primal_graph, length=length, random_seed=random_seed)
            data_map = mock.mock_data_map(data_gdf)
            assert data_map.count() == length
            for uid, row in data_gdf.iterrows():
                data_key = f"{uid.__class__.__name__}:{uid}"
                entry = data_map.get_entry(data_key)
                assert entry is not None
                assert entry.data_key == data_key
                assert entry.data_key_py == uid
                assert entry.geom_wkt == row.geometry.wkt
                # should match fallback
                assert entry.dedupe_key_py == uid
                assert entry.dedupe_key == data_key


def test_mock_barriers():
    """
    Test the mock_barriers function to ensure it creates a GeoDataFrame of barriers.
    """
    barriers_gdf, barriers_wkt = mock.mock_barriers()
    assert len(barriers_gdf) == 4
    assert "geometry" in barriers_gdf.columns
    assert barriers_gdf.geometry.iloc[0].geom_type == "Point"
    assert barriers_gdf.geometry.iloc[1].geom_type == "LineString"
    assert barriers_gdf.geometry.iloc[2].geom_type == "MultiLineString"
    assert barriers_gdf.geometry.iloc[3].geom_type == "Polygon"
    for idx, geom in enumerate(barriers_gdf.geometry):
        assert geom.wkt == barriers_wkt[idx]
