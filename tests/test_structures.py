# pyright: basic
from __future__ import annotations

import numpy as np
import pytest

from cityseer import rustalgos
from cityseer.metrics import layers
from cityseer.tools import graphs, mock


def test_check_network_structure(primal_graph):
    # TODO: raise exceptions from rust
    _node_keys, network_structure = graphs.network_structure_from_nx(primal_graph, 3395)
    network_structure.validate()


def test_check_data_map(primal_graph):
    data_gdf = mock.mock_data_gdf(primal_graph)
    _nodes_gdf, network_structure = graphs.network_structure_from_nx(primal_graph, 3395)
    data_map, data_gdf = layers.assign_gdf_to_network(data_gdf, network_structure, max_netw_assign_dist=400)
    # should work if flag set to False
    data_map.validate(False)
    # should throw error if not assigned
    data_map.nearest_assign = np.full(data_map.count, -1, np.int_)
    data_map.next_nearest_assign = np.full(data_map.count, -1, np.int_)
    with pytest.raises(ValueError):
        data_map.validate(True)
    # assign then check that it runs as intended
    data_map, data_gdf = layers.assign_gdf_to_network(data_gdf, network_structure, max_netw_assign_dist=400)
    data_map.validate(True)
    # catch zero length data arrays
    data_map, data_gdf = layers.assign_gdf_to_network(data_gdf, network_structure, max_netw_assign_dist=400)
    data_map.xs = np.array([], np.float32)
    with pytest.raises(ValueError):
        data_map.validate()
    # equal array lengths
    data_map, data_gdf = layers.assign_gdf_to_network(data_gdf, network_structure, max_netw_assign_dist=400)
    data_map.xs = np.array([1.0, 2.0], dtype=np.float32)
    with pytest.raises(ValueError):
        data_map.validate()
    # catch problematic x or y values
    for bad_val in [np.nan, -1.0]:
        data_map, data_gdf = layers.assign_gdf_to_network(data_gdf, network_structure, max_netw_assign_dist=400)
        data_map.xs[0] = bad_val
        with pytest.raises(ValueError):
            data_map.validate()
        data_map, data_gdf = layers.assign_gdf_to_network(data_gdf, network_structure, max_netw_assign_dist=400)
        data_map.ys[0] = bad_val
        with pytest.raises(ValueError):
            data_map.validate()
