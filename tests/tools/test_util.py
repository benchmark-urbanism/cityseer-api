# pyright: basic
from __future__ import annotations

import pytest

from cityseer.tools import util


def test_add_node(diamond_graph):
    new_name, is_dupe = util.add_node(diamond_graph, ["0", "1"], 50, 50)
    assert is_dupe is False
    assert new_name == "0±1"
    assert list(diamond_graph.nodes) == ["0", "1", "2", "3", "0±1"]
    assert diamond_graph.nodes["0±1"] == {"x": 50, "y": 50}

    # same name and coordinates should return None
    response, is_dupe = util.add_node(diamond_graph, ["0", "1"], 50, 50)
    assert is_dupe is True

    # same name and different coordinates should return v2
    new_name, is_dupe = util.add_node(diamond_graph, ["0", "1"], 40, 50)
    assert is_dupe is False
    assert new_name == "0±1§v2"
    assert list(diamond_graph.nodes) == ["0", "1", "2", "3", "0±1", "0±1§v2"]
    assert diamond_graph.nodes["0±1§v2"] == {"x": 40, "y": 50}

    # likewise v3
    new_name, is_dupe = util.add_node(diamond_graph, ["0", "1"], 30, 50)
    assert is_dupe is False
    assert new_name == "0±1§v3"
    assert list(diamond_graph.nodes) == ["0", "1", "2", "3", "0±1", "0±1§v2", "0±1§v3"]
    assert diamond_graph.nodes["0±1§v3"] == {"x": 30, "y": 50}

    # and names should concatenate over old merges
    new_name, is_dupe = util.add_node(diamond_graph, ["0", "0±1"], 60, 30)
    assert is_dupe is False
    assert new_name == "0±0±1"
    assert list(diamond_graph.nodes) == ["0", "1", "2", "3", "0±1", "0±1§v2", "0±1§v3", "0±0±1"]
    assert diamond_graph.nodes["0±0±1"] == {"x": 60, "y": 30}
