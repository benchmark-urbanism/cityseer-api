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
            (0, {"x": 0, "y": -86.60254}),
            (1, {"x": -50, "y": 0}),
            (2, {"x": 50, "y": 0}),
            (3, {"x": 0, "y": 86.60254}),
        ]
    )
    G_diamond.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)])
    G_diamond = graphs.nx_simple_geoms(G_diamond)
    return G_diamond
