# pyright: basic
from __future__ import annotations

import networkx as nx
import pytest

from cityseer.tools import graphs
from cityseer.tools.mock import mock_graph


@pytest.fixture
def primal_graph() -> nx.MultiGraph:
    """
    Prepare a primal graph for testing.

    Returns
    -------
    nx.MultiGraph
        A primal `NetworkX` `MultiGraph` for `pytest` tests.

    """
    G_primal = mock_graph()
    G_primal = graphs.nx_simple_geoms(G_primal)
    return G_primal


@pytest.fixture
def dual_graph() -> nx.MultiGraph:
    """
    Prepare a dual graph for testing.

    Returns
    -------
    nx.MultiGraph
        A dual `NetworkX` `MultiGraph` for `pytest` tests.

    """
    G_dual = mock_graph()
    G_dual = graphs.nx_simple_geoms(G_dual)
    G_dual = graphs.nx_to_dual(G_dual)
    return G_dual


@pytest.fixture
def diamond_graph() -> nx.MultiGraph:
    r"""
    Generate a diamond shaped `NetworkX` `MultiGraph` for testing or experimentation purposes.

    For manual checks of all node and segmentised methods.

    Returns
    -------
    nx.MultiGraph
        A `NetworkX` `MultiGraph` with `x` and `y` node attributes.

    Notes
    -----
    ```python
    #     3
    #    / \
    #   /   \
    #  /  a  \
    # 1-------2
    #  \  |  /
    #   \ |b/ c
    #    \|/
    #     0
    # a = 100m = 2 * 50m
    # b = 86.60254m
    # c = 100m
    # all inner angles = 60º
    ```

    """
    G_diamond = nx.MultiGraph()
    G_diamond.add_nodes_from(
        [
            ("0", {"x": 0, "y": -86.60254}),
            ("1", {"x": -50, "y": 0}),
            ("2", {"x": 50, "y": 0}),
            ("3", {"x": 0, "y": 86.60254}),
        ]
    )
    G_diamond.add_edges_from([("0", "1"), ("0", "2"), ("1", "2"), ("1", "3"), ("2", "3")])
    G_diamond = graphs.nx_simple_geoms(G_diamond)
    return G_diamond


@pytest.fixture
def box_graph() -> nx.MultiGraph:
    G_box = nx.MultiGraph()
    G_box.add_nodes_from(
        [
            ("0", {"x": 0, "y": 0}),
            ("1", {"x": 5, "y": 0}),
            ("2", {"x": 5, "y": 5}),
            ("3", {"x": 0, "y": 5}),
        ]
    )
    G_box.add_edges_from([("0", "1"), ("1", "2"), ("2", "3")])
    G_box = graphs.nx_simple_geoms(G_box)
    return G_box


@pytest.fixture
def parallel_segments_graph() -> nx.MultiGraph:
    """ """
    nodes = [
        (0, {"x": 620, "y": 720}),
        (1, {"x": 620, "y": 700}),
        (2, {"x": 660, "y": 700}),
        (3, {"x": 660, "y": 660}),
        (4, {"x": 700, "y": 800}),
        (5, {"x": 720, "y": 800}),
        (6, {"x": 700, "y": 720}),
        (7, {"x": 720, "y": 720}),
        (8, {"x": 700, "y": 700}),
        (9, {"x": 700, "y": 620}),
        (10, {"x": 720, "y": 620}),
        (11, {"x": 760, "y": 760}),
        (12, {"x": 800, "y": 760}),
        (13, {"x": 780, "y": 720}),
        (14, {"x": 840, "y": 720}),
        (15, {"x": 840, "y": 700}),
    ]
    edges = [
        (0, 6),
        (1, 2),
        (2, 3),
        (2, 8),
        (4, 6),
        (5, 7),
        (6, 7),
        (6, 8),
        (7, 10),
        (7, 13),
        (8, 9),
        (8, 15),
        (11, 12),
        (11, 13),
        (12, 13),
        (13, 14),
    ]
    G_parallel = nx.MultiGraph()
    G_parallel.add_nodes_from(nodes)
    G_parallel.add_edges_from(edges)
    G_parallel = graphs.nx_simple_geoms(G_parallel)
    return G_parallel
