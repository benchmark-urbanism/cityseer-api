import numpy as np
import pytest

from cityseer.algos import data, checks
from cityseer.metrics import networks, layers
from cityseer.util import graphs, mock


def test_check_data_map():
    G, pos = mock.mock_graph()
    G = graphs.networkX_simple_geoms(G)
    G = graphs.networkX_edge_defaults(G)
    N = networks.Network_Layer_From_NetworkX(G, distances=[500])
    data_dict = mock.mock_data(G)
    D = layers.Data_Layer_From_Dict(data_dict)

    # should throw error if not assigned
    with pytest.raises(ValueError):
        checks.check_data_map(D._data)

    # should work if flag set to False
    checks.check_data_map(D._data, check_assigned=False)

    # assign then check that it runs as intended
    D.assign_to_network(N, 400)
    checks.check_data_map(D._data)

    # catch invalid dimensionality
    with pytest.raises(ValueError):
        checks.check_data_map(D._data[:, :-1])

    # catch missing data classes
    with pytest.raises(ValueError):
        D._data[:, 3] = np.nan
        checks.check_data_map(D._data)


def test_check_trim_maps():
    G, pos = mock.mock_graph()
    G = graphs.networkX_simple_geoms(G)
    G = graphs.networkX_edge_defaults(G)
    N = networks.Network_Layer_From_NetworkX(G, distances=[500])
    trim_to_full_idx_map, full_to_trim_idx_map = \
        data.radial_filter(N.x_arr[0], N.y_arr[0], N.x_arr, N.y_arr, 500)

    # incorrect order
    with pytest.raises(ValueError):
        checks.check_trim_maps(full_to_trim_idx_map, trim_to_full_idx_map)
    # non-sequential trim indices in full map
    with pytest.raises(ValueError):
        non_seq_full_to_trim = full_to_trim_idx_map.copy()
        non_seq_full_to_trim[1] = 2
        non_seq_full_to_trim[2] = 1
        checks.check_trim_maps(trim_to_full_idx_map, non_seq_full_to_trim)
    # wrong length of trim_to_full map
    with pytest.raises(ValueError):
        checks.check_trim_maps(trim_to_full_idx_map[:-1], full_to_trim_idx_map)
    # corrupt trim_to_full
    corrupt_trim_to_full = trim_to_full_idx_map.copy()
    corrupt_trim_to_full[0] = np.nan
    with pytest.raises(ValueError):
        checks.check_trim_maps(corrupt_trim_to_full, full_to_trim_idx_map)
    # corrupt full_to_trim
    corrupt_full_to_trim = full_to_trim_idx_map.copy()
    corrupt_full_to_trim[41] = 100
    with pytest.raises(ValueError):
        checks.check_trim_maps(trim_to_full_idx_map, corrupt_full_to_trim)


def test_check_network_types():
    # network maps
    G, pos = mock.mock_graph()
    G = graphs.networkX_simple_geoms(G)
    G = graphs.networkX_edge_defaults(G)
    N = networks.Network_Layer_From_NetworkX(G, distances=[500])

    # from cityseer.util import plot
    # plot.plot_networkX_primal_or_dual(primal=G)
    # plot.plot_graph_maps(N.uids, N._nodes, N._edges)

    # check that malformed node and data maps throw errors
    with pytest.raises(ValueError):
        checks.check_network_types(N._nodes[:, :-1], N._edges)
    with pytest.raises(ValueError):
        checks.check_network_types(N._nodes, N._edges[:, :-1])
    # catch corrupted edge references from node map
    corrupted_nodes = N._nodes.copy()
    corrupted_nodes[1][3] = 4
    checks.check_network_types(corrupted_nodes, N._edges, check_integrity=False)  # shouldn't raise
    with pytest.raises(ValueError):
        checks.check_network_types(corrupted_nodes, N._edges)
    # catch corrupted node references from edge map
    # first out of order
    corrupted_edges = N._edges.copy()
    corrupted_edges[1][0] = 1
    checks.check_network_types(N._nodes, corrupted_edges, check_integrity=False)  # shouldn't raise
    with pytest.raises(ValueError):
        checks.check_network_types(N._nodes, corrupted_edges)
    # greater than sequential step
    corrupted_edges = N._edges.copy()
    corrupted_edges[3][0] = 2
    checks.check_network_types(N._nodes, corrupted_edges, check_integrity=False)  # shouldn't raise
    with pytest.raises(ValueError):
        checks.check_network_types(N._nodes, corrupted_edges)
    # catch NaN or negative values
    for x in [np.nan, -1]:
        # invalid weights
        corrupted_nodes = N._nodes.copy()
        corrupted_nodes[0][4] *= x
        with pytest.raises(ValueError):
            checks.check_network_types(corrupted_nodes, N._edges)
        # invalid weights
        corrupted_edges = N._edges.copy()
        corrupted_edges[0][2] *= x
        with pytest.raises(ValueError):
            checks.check_network_types(N._nodes, corrupted_edges)
        # invalid weights
        corrupted_edges = N._edges.copy()
        corrupted_edges[0][3] *= x
        with pytest.raises(ValueError):
            checks.check_network_types(N._nodes, corrupted_edges)


def test_check_distances_and_betas():
    betas = np.array([-0.02, -0.01, -0.005, -0.0025])
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
    with pytest.raises(ValueError):
        dup_betas = np.array([-0.02, -0.02])
        dup_distances = np.array(networks.distance_from_beta(dup_betas))
        checks.check_distances_and_betas(dup_distances, dup_betas)
    # positive values of beta
    with pytest.raises(ValueError):
        betas[0] = 4
        checks.check_distances_and_betas(distances, betas)
    # inconsistent distances <-> betas
    with pytest.raises(ValueError):
        betas[0] = -0.04
        checks.check_distances_and_betas(distances, betas)
