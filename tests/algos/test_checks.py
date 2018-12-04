import numpy as np
import pytest

from cityseer.algos import data, checks
from cityseer.metrics import networks, layers
from cityseer.util import graphs, mock


def test_check_data_map():
    G, pos = mock.mock_graph()
    G = graphs.networkX_simple_geoms(G)
    G = graphs.networkX_edge_defaults(G)
    data_dict = mock.mock_data(G)
    D = layers.Data_Layer_From_Dict(data_dict)

    with pytest.raises(ValueError):
        checks.check_data_map(D._data[:, :-1])


def test_check_trim_maps():
    G, pos = mock.mock_graph()
    G = graphs.networkX_simple_geoms(G)
    G = graphs.networkX_edge_defaults(G)
    N = networks.Network_Layer_From_NetworkX(G, distances=[500])
    trim_to_full_idx_map, full_to_trim_idx_map = \
        data.radial_filter(N.x_arr[0], N.y_arr[1], N.x_arr, N.y_arr, 500)

    # mismatching lengths
    with pytest.raises(ValueError):
        checks.check_trim_maps(trim_to_full_idx_map[:-1], full_to_trim_idx_map)
    # full_to_trim_idx_map length can't be checked explicitly
    # corrupt trim_to_full
    corrupt_trim_to_full = trim_to_full_idx_map
    corrupt_trim_to_full[0] = np.nan
    with pytest.raises(ValueError):
        checks.check_trim_maps(corrupt_trim_to_full, full_to_trim_idx_map)
    # corrupt full_to_trim
    corrupt_full_to_trim = full_to_trim_idx_map
    corrupt_full_to_trim[0] = 100
    with pytest.raises(ValueError):
        checks.check_trim_maps(trim_to_full_idx_map, full_to_trim_idx_map)


def test_check_network_types():
    # network maps
    G, pos = mock.mock_graph()
    G = graphs.networkX_simple_geoms(G)
    G = graphs.networkX_edge_defaults(G)
    N = networks.Network_Layer_From_NetworkX(G, distances=[500])

    # check that malformed node and data maps throw errors
    with pytest.raises(ValueError):
        checks.check_network_types(N._nodes[:, :-1], N._edges)
    with pytest.raises(ValueError):
        checks.check_network_types(N._nodes, N._edges[:, :-1])


def test_check_distances_and_betas():
    betas = np.array([-0.02, -0.01, -0.005, -0.0025])
    distances = networks.distance_from_beta(betas)

    with pytest.raises(ValueError):
        checks.check_distances_and_betas(np.array(distances[:-1]), betas)
    with pytest.raises(ValueError):
        checks.check_distances_and_betas(np.array(distances), betas[:-1])
    with pytest.raises(ValueError):
        betas[0] = 4
        checks.check_distances_and_betas(np.array(distances), betas)
