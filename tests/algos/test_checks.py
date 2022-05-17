import numpy as np
import pytest

from cityseer.algos import checks, data
from cityseer.metrics import layers, networks
from cityseer.tools import graphs, mock


def test_check_numerical_data():
    mock_numerical = mock.mock_numerical_data(50)

    # check for malformed data
    # difficult to catch int arrays without running into numba type checking errors
    # single dimension
    with pytest.raises(ValueError):
        corrupt_numerical = mock_numerical[0]
        assert corrupt_numerical.ndim == 1
        checks.check_numerical_data(corrupt_numerical)
    # catch infinites
    with pytest.raises(ValueError):
        mock_numerical[0][0] = np.inf
        checks.check_numerical_data(mock_numerical)


def test_check_categorical_data():
    mock_categorical = mock.mock_categorical_data(50)
    _data_classes, data_encoding = layers.encode_categorical(mock_categorical)

    # check for malformed data
    # negatives
    with pytest.raises(ValueError):
        data_encoding[0] = -1
        checks.check_categorical_data(data_encoding)
    # NaN
    with pytest.raises(ValueError):
        data_encoding[0] = np.nan
        checks.check_categorical_data(data_encoding)
    # floats
    with pytest.raises(ValueError):
        data_encoding_float = np.full(data_encoding.shape[0], np.nan)
        data_encoding_float[:] = data_encoding[:].astype(float)
        data_encoding_float[0] = 1.2345
        checks.check_categorical_data(data_encoding_float)


def test_check_data_map(primal_graph):
    N = networks.NetworkLayerFromNX(primal_graph, distances=[500])
    data_dict = mock.mock_data_dict(primal_graph)
    data_uids, data_map = layers.data_map_from_dict(data_dict)

    # should throw error if not assigned
    with pytest.raises(ValueError):
        checks.check_data_map(data_map)

    # should work if flag set to False
    checks.check_data_map(data_map, check_assigned=False)

    # assign then check that it runs as intended
    data_map = data.assign_to_network(data_map, N._node_data, N._edge_data, N._node_edge_map, max_dist=400)
    checks.check_data_map(data_map)

    # catch zero length data arrays
    empty_2d_arr = np.full((0, 4), np.nan)
    with pytest.raises(ValueError):
        checks.check_data_map(empty_2d_arr)

    # catch invalid dimensionality
    with pytest.raises(ValueError):
        checks.check_data_map(data_map[:, :-1])


def test_check_network_maps(primal_graph):
    # network maps
    node_uids, node_data, edge_data, node_edge_map = graphs.network_structure_from_nx(primal_graph)
    # catch zero length node and edge arrays
    empty_node_arr = np.full((0, 5), np.nan)
    with pytest.raises(ValueError):
        checks.check_network_maps(empty_node_arr, edge_data, node_edge_map)
    empty_edge_arr = np.full((0, 4), np.nan)
    with pytest.raises(ValueError):
        checks.check_network_maps(node_data, empty_edge_arr, node_edge_map)
    # check that malformed node and data maps throw errors
    with pytest.raises(ValueError):
        checks.check_network_maps(node_data[:, :-1], edge_data, node_edge_map)
    with pytest.raises(ValueError):
        checks.check_network_maps(node_data, edge_data[:, :-1], node_edge_map)
    # catch problematic edge map values
    for x in [np.nan, -1]:
        # missing start node
        corrupted_edges = edge_data.copy()
        corrupted_edges[0, 0] = x
        with pytest.raises(ValueError):
            checks.check_network_maps(node_data, corrupted_edges, node_edge_map)
        # missing end node
        corrupted_edges = edge_data.copy()
        corrupted_edges[0, 1] = x
        with pytest.raises((ValueError, KeyError)):
            checks.check_network_maps(node_data, corrupted_edges, node_edge_map)
        # invalid length
        corrupted_edges = edge_data.copy()
        corrupted_edges[0, 2] = x
        with pytest.raises(ValueError):
            checks.check_network_maps(node_data, corrupted_edges, node_edge_map)
        # invalid angle_sum
        corrupted_edges = edge_data.copy()
        corrupted_edges[0, 3] = x
        with pytest.raises(ValueError):
            checks.check_network_maps(node_data, corrupted_edges, node_edge_map)
        # invalid imp_factor
        corrupted_edges = edge_data.copy()
        corrupted_edges[0, 4] = x
        with pytest.raises(ValueError):
            checks.check_network_maps(node_data, corrupted_edges, node_edge_map)


def test_check_distances_and_betas():
    betas = np.array([0.02, 0.01, 0.005, 0.0025, 0.0])
    distances = np.array(networks.distance_from_beta(betas))

    # zero length arrays
    with pytest.raises(ValueError):
        checks.check_distances_and_betas(np.array([]), betas)
    with pytest.raises(ValueError):
        checks.check_distances_and_betas(distances, np.array([]))
    # mismatching array lengths
    with pytest.raises(ValueError):
        checks.check_distances_and_betas(np.array(distances[:-1]), betas)
    with pytest.raises(ValueError):
        checks.check_distances_and_betas(distances, betas[:-1])
    # check that duplicates are caught
    dup_betas = np.array([0.02, 0.02])
    dup_distances = np.array(networks.distance_from_beta(dup_betas))
    with pytest.raises(ValueError):
        checks.check_distances_and_betas(dup_distances, dup_betas)
    # negative values of beta
    betas_pos = betas.copy()
    betas_pos[0] = -4
    with pytest.raises(ValueError):
        checks.check_distances_and_betas(distances, betas_pos)
    # negative values of distance
    distances_neg = distances.copy()
    distances_neg[0] = -100
    with pytest.raises(ValueError):
        checks.check_distances_and_betas(distances_neg, betas)
    # inconsistent distances <-> betas
    betas[1] = 0.03
    with pytest.raises(ValueError):
        checks.check_distances_and_betas(distances, betas)
