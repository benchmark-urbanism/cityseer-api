# pyright: basic


import numpy as np
import numpy.typing as npt
import pytest

from cityseer import structures
from cityseer.algos import checks, data
from cityseer.metrics import layers, networks
from cityseer.tools import graphs, mock


def test_validate_node_map(primal_graph):
    _node_keys, network_structure = graphs.network_structure_from_nx(primal_graph)
    network_structure.nodes.validate()
    # equal array lengths
    network_structure.nodes.xs = np.array([1.0, 2.0], np.float32)
    with pytest.raises(ValueError):
        network_structure.nodes.validate()
    for bad_val in [np.nan, -1]:
        # xs
        _node_keys, network_structure = graphs.network_structure_from_nx(primal_graph)
        network_structure.nodes.xs[0] = bad_val
        with pytest.raises(ValueError):
            network_structure.nodes.validate()
        # ys
        _node_keys, network_structure = graphs.network_structure_from_nx(primal_graph)
        network_structure.nodes.xs[0] = bad_val
        with pytest.raises(ValueError):
            network_structure.nodes.validate()
    # all dead
    _node_keys, network_structure = graphs.network_structure_from_nx(primal_graph)
    network_structure.nodes.live[:] = False
    with pytest.raises(ValueError):
        network_structure.nodes.validate()


def test_validate_edge_map(primal_graph):
    _node_keys, network_structure = graphs.network_structure_from_nx(primal_graph)
    network_structure.edges.validate()
    # equal array lengths
    network_structure.edges.start = np.array([1, 1], dtype=np.int_)
    with pytest.raises(ValueError):
        network_structure.edges.validate()
    # catch problematic edge map values
    for bad_val in [-1]:
        # missing start node
        _node_keys, network_structure = graphs.network_structure_from_nx(primal_graph)
        network_structure.edges.start[0] = bad_val
        with pytest.raises(ValueError):
            network_structure.validate()
        # missing end node
        _node_keys, network_structure = graphs.network_structure_from_nx(primal_graph)
        network_structure.edges.end[0] = bad_val
        with pytest.raises(ValueError):
            network_structure.validate()
        # invalid length
        _node_keys, network_structure = graphs.network_structure_from_nx(primal_graph)
        network_structure.edges.length[0] = bad_val
        with pytest.raises(ValueError):
            network_structure.validate()
        # invalid angle_sum
        _node_keys, network_structure = graphs.network_structure_from_nx(primal_graph)
        network_structure.edges.angle_sum[0] = bad_val
        with pytest.raises(ValueError):
            network_structure.validate()
        # invalid imp_factor
        _node_keys, network_structure = graphs.network_structure_from_nx(primal_graph)
        network_structure.edges.imp_factor[0] = bad_val
        with pytest.raises(ValueError):
            network_structure.validate()


def test_check_network_structure(primal_graph):
    # corrupted node to edge maps
    _node_keys, network_structure = graphs.network_structure_from_nx(primal_graph)
    network_structure.validate()
    # corrupt
    network_structure.edges.start[0] = network_structure.edges.start[0] + 1
    with pytest.raises(ValueError):
        network_structure.validate()


def test_check_data_map(primal_graph):
    cc_netw = networks.NetworkLayerFromNX(primal_graph, distances=[500])
    data_dict = mock.mock_data_dict(primal_graph)
    _data_keys, data_map = layers.data_map_from_dict(data_dict)
    # should work if flag set to False
    data_map.validate(False)
    # should throw error if not assigned
    with pytest.raises(ValueError):
        data_map.validate(True)
    # assign then check that it runs as intended
    data_map = data.assign_to_network(data_map, cc_netw.network_structure, max_dist=np.float32(400))
    data_map.validate(True)
    # catch zero length data arrays
    _data_keys, data_map = layers.data_map_from_dict(data_dict)
    data_map.xs = np.array([], np.float32)
    with pytest.raises(ValueError):
        data_map.validate()
    # equal array lengths
    _data_keys, data_map = layers.data_map_from_dict(data_dict)
    data_map.xs = np.array([1.0, 2.0], dtype=np.float32)
    with pytest.raises(ValueError):
        data_map.validate()
    # catch problematic x or y values
    for bad_val in [np.nan, -1.0]:
        _data_keys, data_map = layers.data_map_from_dict(data_dict)
        data_map.xs[0] = bad_val
        with pytest.raises(ValueError):
            data_map.validate()
        _data_keys, data_map = layers.data_map_from_dict(data_dict)
        data_map.ys[0] = bad_val
        with pytest.raises(ValueError):
            data_map.validate()
