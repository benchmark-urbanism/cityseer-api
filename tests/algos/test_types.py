import pytest
import numpy as np
from cityseer.util import graphs, mock, layers
from cityseer.metrics import centrality
from cityseer.algos import data, types


def test_check_index_map():

    G, pos = mock.mock_graph()
    G = graphs.networkX_simple_geoms(G)
    G = graphs.networkX_edge_defaults(G)
    n_labels, n_map, e_map = graphs.graph_maps_from_networkX(G)
    x_arr = n_map[:, 0]
    y_arr = n_map[:, 1]
    index_map = data.generate_index(x_arr, y_arr)

    with pytest.raises(ValueError):
        types.check_index_map(index_map[:,:-1])
    with pytest.raises(ValueError):
        # flip x order
        index_corrupted = index_map
        index_corrupted[:,:-1] = index_corrupted[:,:-1][::-1]
        types.check_index_map(index_corrupted)
    with pytest.raises(ValueError):
        # flip y order
        index_corrupted = index_map
        index_corrupted[:,:-3] = index_corrupted[:,:-3][::-1]
        types.check_index_map(index_corrupted)


def test_check_data_map():

    G, pos = mock.mock_graph()
    G = graphs.networkX_simple_geoms(G)
    G = graphs.networkX_edge_defaults(G)
    data_dict = mock.mock_data(G)
    d_labels, d_map, d_classes = layers.dict_to_data_map(data_dict)

    with pytest.raises(ValueError):
        types.check_data_map(d_map[:,:-1])


def test_check_trim_maps():

    G, pos = mock.mock_graph()
    G = graphs.networkX_simple_geoms(G)
    G = graphs.networkX_edge_defaults(G)
    n_labels, n_map, e_map = graphs.graph_maps_from_networkX(G)
    x_arr = n_map[:, 0]
    y_arr = n_map[:, 1]
    netw_index = data.generate_index(x_arr, y_arr)
    trim_to_full_idx_map, full_to_trim_idx_map = \
            data.distance_filter(netw_index, n_map[0][0], n_map[0][1], 500, radial=True)

    # mismatching lengths
    with pytest.raises(ValueError):
        types.check_trim_maps(trim_to_full_idx_map[:-1], full_to_trim_idx_map)
    # full_to_trim_idx_map length can't be checked explicitly
    # corrupt trim_to_full
    corrupt_trim_to_full = trim_to_full_idx_map
    corrupt_trim_to_full[0] = np.nan
    with pytest.raises(ValueError):
        types.check_trim_maps(corrupt_trim_to_full, full_to_trim_idx_map)
    # corrupt full_to_trim
    corrupt_full_to_trim = full_to_trim_idx_map
    corrupt_trim_to_full[0] = 100
    with pytest.raises(ValueError):
        types.check_trim_maps(trim_to_full_idx_map, full_to_trim_idx_map)


def test_check_network_types():

    # network maps
    G, pos = mock.mock_graph()
    G = graphs.networkX_simple_geoms(G)
    G = graphs.networkX_edge_defaults(G)
    n_labels, n_map, e_map = graphs.graph_maps_from_networkX(G)
    # index maps
    x_arr = n_map[:, 0]
    y_arr = n_map[:, 1]
    netw_index = data.generate_index(x_arr, y_arr)

    # check that malformed node and data maps throw errors
    with pytest.raises(ValueError):
        types.check_network_types(n_map[:,:-1], e_map)
    with pytest.raises(ValueError):
        types.check_network_types(n_map, e_map[:,:-1])


def test_check_distances_and_betas():

    betas = np.array([-0.02, -0.01, -0.005, -0.0025])
    distances, min_threshold_wt = centrality.distance_from_beta(betas)

    with pytest.raises(ValueError):
        types.check_distances_and_betas(distances[:-1], betas)
    with pytest.raises(ValueError):
        types.check_distances_and_betas(distances, betas[:-1])
    with pytest.raises(ValueError):
        betas[0] = 4
        types.check_distances_and_betas(distances, betas)

