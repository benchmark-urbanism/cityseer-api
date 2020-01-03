import numpy as np
import pytest

from cityseer.algos import data, checks
from cityseer.metrics import networks, layers
from cityseer.util import graphs, mock


def test_progress_bar():
    for n, step_size in zip([1, 10, 100], [1, 3, 10]):
        for i in range(n):
            checks.progress_bar(i, n, step_size)
    # check that chunks > total doesn't raise
    checks.progress_bar(10, 10, 20)


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
    data_classes, data_encoding = layers.encode_categorical(mock_categorical)

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
        data_encoding_float = np.full(len(data_encoding), np.nan)
        data_encoding_float[:] = data_encoding[:].astype(np.float)
        data_encoding_float[0] = 1.2345
        checks.check_categorical_data(data_encoding_float)


def test_check_data_map():
    G = mock.mock_graph()
    G = graphs.nX_simple_geoms(G)
    N = networks.Network_Layer_From_nX(G, distances=[500])
    data_dict = mock.mock_data_dict(G)
    data_uids, data_map = layers.data_map_from_dict(data_dict)

    # should throw error if not assigned
    with pytest.raises(ValueError):
        checks.check_data_map(data_map)

    # should work if flag set to False
    checks.check_data_map(data_map, check_assigned=False)

    # assign then check that it runs as intended
    data_map = data.assign_to_network(data_map,
                                      N._node_data,
                                      N._edge_data,
                                      N._node_edge_map,
                                      max_dist=400)
    checks.check_data_map(data_map)

    # catch zero length data arrays
    empty_2d_arr = np.full((0, 4), np.nan)
    with pytest.raises(ValueError):
        checks.check_data_map(empty_2d_arr)

    # catch invalid dimensionality
    with pytest.raises(ValueError):
        checks.check_data_map(data_map[:, :-1])


def test_check_network_maps():
    # network maps
    G = mock.mock_graph()
    G = graphs.nX_simple_geoms(G)
    N = networks.Network_Layer_From_nX(G, distances=[500])
    # from cityseer.util import plot
    # plot.plot_networkX_primal_or_dual(primal=G)
    # plot.plot_graph_maps(N.uids, N._node_data, N._edge_data)
    # catch zero length node and edge arrays
    empty_node_arr = np.full((0, 5), np.nan)
    with pytest.raises(ValueError):
        checks.check_network_maps(empty_node_arr, N._edge_data, N._node_edge_map)
    empty_edge_arr = np.full((0, 4), np.nan)
    with pytest.raises(ValueError):
        checks.check_network_maps(N._node_data, empty_edge_arr, N._node_edge_map)
    # check that malformed node and data maps throw errors
    with pytest.raises(ValueError):
        checks.check_network_maps(N._node_data[:, :-1], N._edge_data, N._node_edge_map)
    with pytest.raises(ValueError):
        checks.check_network_maps(N._node_data, N._edge_data[:, :-1], N._node_edge_map)
    # catch problematic edge map values
    for x in [np.nan, -1]:
        # missing start node
        corrupted_edges = N._edge_data.copy()
        corrupted_edges[0, 0] = x
        with pytest.raises(AssertionError):
            checks.check_network_maps(N._node_data, corrupted_edges, N._node_edge_map)
        # missing end node
        corrupted_edges = N._edge_data.copy()
        corrupted_edges[0, 1] = x
        with pytest.raises(KeyError):
            checks.check_network_maps(N._node_data, corrupted_edges, N._node_edge_map)
        # invalid length
        corrupted_edges = N._edge_data.copy()
        corrupted_edges[0, 2] = x
        with pytest.raises(ValueError):
            checks.check_network_maps(N._node_data, corrupted_edges, N._node_edge_map)
        # invalid angle_sum
        corrupted_edges = N._edge_data.copy()
        corrupted_edges[0, 3] = x
        with pytest.raises(ValueError):
            checks.check_network_maps(N._node_data, corrupted_edges, N._node_edge_map)
        # invalid imp_factor
        corrupted_edges = N._edge_data.copy()
        corrupted_edges[0, 4] = x
        with pytest.raises(ValueError):
            checks.check_network_maps(N._node_data, corrupted_edges, N._node_edge_map)


def test_check_distances_and_betas():
    betas = np.array([-0.02, -0.01, -0.005, -0.0025, -0.0])
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
    dup_betas = np.array([-0.02, -0.02])
    dup_distances = np.array(networks.distance_from_beta(dup_betas))
    with pytest.raises(ValueError):
        checks.check_distances_and_betas(dup_distances, dup_betas)
    # positive values of beta
    betas_pos = betas.copy()
    betas_pos[0] = 4
    with pytest.raises(ValueError):
        checks.check_distances_and_betas(distances, betas_pos)
    # negative values of distance
    distances_neg = distances.copy()
    distances_neg[0] = -100
    with pytest.raises(ValueError):
        checks.check_distances_and_betas(distances_neg, betas)
    # inconsistent distances <-> betas
    betas[1] = -0.03
    with pytest.raises(ValueError):
        checks.check_distances_and_betas(distances, betas)
