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
            assert "x" in node_data and isinstance(node_data["y"], (int, float))
            assert "y" in node_data and isinstance(node_data["y"], (int, float))
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
