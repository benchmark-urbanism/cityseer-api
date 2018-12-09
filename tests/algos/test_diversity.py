import numpy as np
import pytest
from scipy.stats import entropy

from cityseer.algos import diversity, data
from cityseer.metrics import layers, networks
from cityseer.util import mock, graphs


def test_hill_diversity():
    # test hill diversity against scipy entropy
    for counts, probs in mock.mock_species_diversity():
        # check hill q=1
        assert np.allclose(diversity.hill_diversity(counts, 1), np.exp(entropy(probs)))
        # check that hill q<1 and q>1 is reasonably close to scipy entropy
        # (different internal computation)
        assert np.allclose(diversity.hill_diversity(counts, 0.99999999), np.exp(entropy(probs)))
        assert np.allclose(diversity.hill_diversity(counts, 1.00000001), np.exp(entropy(probs)))
        # check for malformed q
        with pytest.raises(ValueError):
            diversity.hill_diversity(counts, q=-1)


def test_hill_diversity_branch_distance_wt():
    #  test hill diversity against weighted diversity where all weights = 1
    for counts, probs in mock.mock_species_diversity():

        non_weights = np.full(len(counts), 1)
        non_beta = -0
        for q in [0, 1, 2]:
            assert np.allclose(diversity.hill_diversity(counts, q),
                               diversity.hill_diversity_branch_distance_wt(counts, non_weights, q, non_beta))

        # check for malformed signatures
        with pytest.raises(ValueError):
            diversity.hill_diversity_branch_distance_wt(counts[:-1], non_weights, q=1, beta=-0.005)
        with pytest.raises(ValueError):
            diversity.hill_diversity_branch_distance_wt(counts, non_weights[:-1], q=1, beta=-0.005)
        with pytest.raises(ValueError):
            diversity.hill_diversity_branch_distance_wt(counts, non_weights, q=1, beta=0.005)
        with pytest.raises(ValueError):
            diversity.hill_diversity_branch_distance_wt(counts, non_weights, q=-1, beta=-0.005)


def test_hill_diversity_pairwise_distance_wt():
    #  test hill diversity against weighted diversity where all weights = 1
    for counts, probs in mock.mock_species_diversity():

        non_weights = np.full(len(counts), 1)
        non_beta = -0
        for q in [0, 1, 2]:
            # what to test against? For now, check successful run
            diversity.hill_diversity_pairwise_distance_wt(counts, non_weights, q, non_beta)

        # check for malformed signatures
        with pytest.raises(ValueError):
            diversity.hill_diversity_pairwise_distance_wt(counts[:-1], non_weights, q=1, beta=-0.005)
        with pytest.raises(ValueError):
            diversity.hill_diversity_pairwise_distance_wt(counts, non_weights[:-1], q=1, beta=-0.005)
        with pytest.raises(ValueError):
            diversity.hill_diversity_pairwise_distance_wt(counts, non_weights, q=1, beta=0.005)
        with pytest.raises(ValueError):
            diversity.hill_diversity_pairwise_distance_wt(counts, non_weights, q=-1, beta=-0.005)


def test_hill_diversity_pairwise_matrix_wt():

    for counts, probs in mock.mock_species_diversity():
        non_matrix = np.full((len(counts), len(counts)), 1)
        for q in [0, 1, 2]:
            # what to test against? For now, check successful run
            diversity.hill_diversity_pairwise_matrix_wt(counts, non_matrix, q)

        # check for malformed signatures
        with pytest.raises(ValueError):
            diversity.hill_diversity_pairwise_matrix_wt(counts[:-1], non_matrix, q=1)
        with pytest.raises(ValueError):
            diversity.hill_diversity_pairwise_matrix_wt(counts, non_matrix[:-1], q=1)
        with pytest.raises(ValueError):
            diversity.hill_diversity_pairwise_matrix_wt(counts, non_matrix[:, :-1], q=1)
        with pytest.raises(ValueError):
            diversity.hill_diversity_pairwise_matrix_wt(counts, non_matrix, q=-1)


def test_gini_simpson_diversity():
    '''
    USED FOR RESEARCH PURPOSES ONLY
    '''
    # just run for now to check against unexpectedly thrown errors
    for counts, probs in mock.mock_species_diversity():
        diversity.gini_simpson_diversity(counts)


def test_shannon_diversity():
    '''
    USED FOR RESEARCH PURPOSES ONLY
    '''
    # test against scipy entropy
    for counts, probs in mock.mock_species_diversity():
        assert abs(entropy(probs) - diversity.shannon_diversity(probs)) < 0.0000000001


def test_raos_quadratic_diversity():
    '''
    USED FOR RESEARCH PURPOSES ONLY
    '''
    # just run for now to check against unexpectedly thrown errors
    for counts, probs in mock.mock_species_diversity():
        mock_matrix = np.full((len(counts), len(counts)), 1)
        diversity.raos_quadratic_diversity(counts, mock_matrix)


def test_local_landuses():
    # load the test graph
    G, pos = mock.mock_graph()
    G = graphs.networkX_simple_geoms(G)
    G = graphs.networkX_edge_defaults(G)  # set default edge attributes
    node_uids, node_map, edge_map = graphs.graph_maps_from_networkX(G)  # generate node and edge maps

    data_dict = mock.mock_data(G, random_seed=13)
    data_uids, data_map, class_labels = layers.data_map_from_dict(data_dict)
    data_map = data.assign_to_network(data_map, node_map, edge_map, 500)

    betas = np.array([-0.005, -0.0025])
    distances = networks.distance_from_beta(betas)
    qs = np.array([0, 1, 2])
    div_keys = np.arange(7)
    ac_codes = np.random.randint(0, len(class_labels), 4)
    mock_matrix = np.full((len(class_labels), len(class_labels)), 1)
    class_weights = np.array([3 / 3, 2 / 3, 1 / 3, 0])

    mixed_use_hill, mixed_use_other, accessibility_data, accessibility_data_wt = \
        diversity.local_landuses(node_map,
                                 edge_map,
                                 data_map,
                                 distances,
                                 betas,
                                 qs,
                                 div_keys,
                                 ac_codes,
                                 mock_matrix)

    # catch no qs
    with pytest.raises(ValueError):
        diversity.local_landuses(node_map,
                                 edge_map,
                                 data_map,
                                 distances,
                                 betas,
                                 np.array([]),
                                 div_keys,
                                 ac_codes)
    # catch missing class tiers
    with pytest.raises(ValueError):
        diversity.local_landuses(node_map,
                                 edge_map,
                                 data_map,
                                 distances,
                                 betas,
                                 qs,
                                 div_keys,
                                 ac_codes,
                                 mock_matrix[:-1])
