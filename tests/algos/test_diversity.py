import pytest
import numpy as np
from cityseer.algos import diversity
from cityseer.util import mock
from scipy.stats import entropy


def test_hill_diversity():

    # test hill diversity against scipy entropy
    for counts, probs in mock.mock_species_diversity():
        # check hill q=1
        assert abs(diversity.hill_diversity(counts, 1) - np.exp(entropy(probs))) < 0.00000001
        # check that hill q<1 and q>1 is reasonably close to scipy entropy
        # (different internal computation)
        assert abs(diversity.hill_diversity(counts, 0.9999) - np.exp(entropy(probs))) < 0.0001
        assert abs(diversity.hill_diversity(counts, 1.0001) - np.exp(entropy(probs))) < 0.0001
        # check for malformed q
        with pytest.raises(ValueError):
            diversity.hill_diversity(counts, q=-1)

def test_hill_diversity_branch_generic():

    #  test hill diversity against weighted diversity where all weights = 1
    for counts, probs in mock.mock_species_diversity():

        q = [0, 1, 2][np.random.randint(0, 3)]

        non_weights = np.full(len(counts), 1)
        assert abs(diversity.hill_diversity(counts, q) - diversity.hill_diversity_branch_generic(counts, non_weights, q)) < 0.00000001

        # check for malformed signatures
        with pytest.raises(ValueError):
            diversity.hill_diversity_branch_generic(counts[:-1], non_weights, q)
        with pytest.raises(ValueError):
            diversity.hill_diversity_branch_generic(counts, non_weights[:-1], q)
        with pytest.raises(ValueError):
            diversity.hill_diversity_branch_generic(counts, non_weights, q=-1)


def test_hill_diversity_branch_distance_wt():

    for counts, probs in mock.mock_species_diversity():
        distances = np.random.uniform(0, 2000, len(counts))
        q = [0, 1, 2][np.random.randint(0, 3)]

        # check for malformed data
        with pytest.raises(ValueError):
            diversity.hill_diversity_branch_distance_wt(counts, distances, -0.005, q)


def test_hill_diversity_pairwise_generic():

    # what to test against?
    # just run for now to check against unexpectedly thrown errors
    for counts, probs in mock.mock_species_diversity():
        non_weights = np.full((len(counts), len(counts)), 1)
        q = [0, 1, 2][np.random.randint(0, 3)]
        diversity.hill_diversity_pairwise_generic(counts, non_weights, q)

        # check for malformed data
        with pytest.raises(ValueError):
            diversity.hill_diversity_pairwise_generic(counts[:-1], non_weights, q)
        with pytest.raises(ValueError):
            diversity.hill_diversity_pairwise_generic(counts, non_weights[:-1], q)
        with pytest.raises(ValueError):
            diversity.hill_diversity_pairwise_generic(counts, non_weights, q=-1)


def test_pairwise_distance_matrix():

    def check_matrix(wt_matrix, i, j, wt):
        # in this case the iteration handles i = i situations
        assert wt_matrix[i][j] == wt
        assert wt_matrix[j][i] == wt

    distances = [0, 100, 1000, 2, 200, 2000]
    beta = 0.005

    wt_matrix = diversity.pairwise_distance_matrix(np.array(distances), beta)
    for i, l_1 in enumerate(distances):
        for j, l_2 in enumerate(distances):
            check_matrix(wt_matrix, i, j, np.exp((l_1 + l_2) * -beta))

    # check for malformed data
    with pytest.raises(ValueError):
        diversity.pairwise_distance_matrix(np.array(distances), -0.005)


def test_hill_diversity_pairwise_distance_wt():

    for counts, probs in mock.mock_species_diversity():
        distances = np.random.uniform(0, 2000, len(counts))
        beta = [0.01, 0.005, 0.0025][np.random.randint(0, 3)]
        q = [0, 1, 2][np.random.randint(0, 3)]

        with pytest.raises(ValueError):
            diversity.hill_diversity_pairwise_distance_wt(counts[:-1], distances, beta, q)
        with pytest.raises(ValueError):
            diversity.hill_diversity_pairwise_distance_wt(counts, distances[:-1], beta, q)


def test_pairwise_disparity_matrix():

    def check_matrix(wt_matrix, i, j, wt):
        assert wt_matrix[i][i] == 0
        assert wt_matrix[j][j] == 0
        assert wt_matrix[i][j] == wt
        assert wt_matrix[j][i] == wt

    class_weights = np.array([3/3, 2/3, 1/3, 0])

    # test various levels of convergence
    class_tiers = np.array([
        [0, 10, 100, 1000],
        [0, 10, 100, 1000]
    ])
    wt_matrix = diversity.pairwise_disparity_matrix(class_tiers, class_weights)
    check_matrix(wt_matrix, 0, 1, class_weights[3])

    class_tiers = np.array([
        [0, 10, 100, 1000],
        [0, 10, 100, 1001]
    ])
    wt_matrix = diversity.pairwise_disparity_matrix(class_tiers, class_weights)
    check_matrix(wt_matrix, 0, 1, class_weights[2])

    class_tiers = np.array([
        [0, 10, 100, 1000],
        [0, 10, 101, 1000]
    ])
    wt_matrix = diversity.pairwise_disparity_matrix(class_tiers, class_weights)
    check_matrix(wt_matrix, 0, 1, class_weights[1])

    class_tiers = np.array([
        [0, 10, 100, 1000],
        [0, 11, 100, 1000]
    ])
    wt_matrix = diversity.pairwise_disparity_matrix(class_tiers, class_weights)
    check_matrix(wt_matrix, 0, 1, class_weights[0])

    # check for lack of convergence
    class_tiers = np.array([
        [0, 10, 100, 1000],
        [1, 11, 100, 1000]
    ])
    with pytest.raises(AttributeError):
        diversity.pairwise_disparity_matrix(class_tiers, class_weights)

    # check for malformed data
    with pytest.raises(ValueError):
        diversity.pairwise_disparity_matrix(class_tiers, class_weights[:-1])


def test_hill_diversity_pairwise_disparity_wt():

    def tier_factory(n):
        tiers = []
        for i in range(n):
            tiers.append([
                0,
                np.random.randint(10, 12),
                np.random.randint(100, 102),
                np.random.randint(1000, 1002)
            ])
        return np.array(tiers)

    for counts, probs in mock.mock_species_diversity():
        tiers = tier_factory(len(counts))
        weights = np.array([3/3, 2/3, 1/3, 0])
        q = float([0, 1, 2][np.random.randint(0, 3)])

        # check for malformed data
        if len(counts) > 1:
            with pytest.raises(ValueError):
                diversity.hill_diversity_pairwise_disparity_wt(counts[:-1], tiers, weights, q)
            with pytest.raises(ValueError):
                diversity.hill_diversity_pairwise_disparity_wt(counts, tiers[:-1], weights, q)


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
        non_weights = np.full((len(counts), len(counts)), 1)
        diversity.raos_quadratic_diversity(counts, non_weights)


def test_deduce_unique_species():
    pass


def test_compute_mixed_uses():
    pass
