import pytest
import random
import numpy as np
from cityseer.algos import diversity
from scipy.stats import entropy


def test_deduce_unique_species():
    pass


def probs_gen():
    '''
    convenience function for testing some probabilities in below tests
    '''
    for n in range(1, 10):
        data = np.random.random_integers(1, 3, n)
        unique = np.unique(data)
        counts = np.zeros_like(unique)
        for i, u in enumerate(unique):
            counts[i] = (data==u).sum()
        probs = counts / len(data)

        assert round(probs.sum(), 8) == 1

        yield counts, probs


def test_hill_diversity():

    # test hill diversity against scipy entropy
    for counts, probs in probs_gen():
        # check hill q=1
        assert abs(diversity.hill_diversity(counts, 1) - np.exp(entropy(probs))) < 0.00000001
        # check that hill q<1 and q>1 is reasonably close to scipy entropy
        # (different internal computation)
        assert abs(diversity.hill_diversity(counts, 0.9999) - np.exp(entropy(probs))) < 0.0001
        assert abs(diversity.hill_diversity(counts, 1.0001) - np.exp(entropy(probs))) < 0.0001


def test_hill_diversity_branch_generic():

    #  test hill diversity against weighted diversity where all weights = 1
    for counts, probs in probs_gen():
        non_weights = np.full(len(counts), 1)
        assert abs(diversity.hill_diversity(counts, 1) - diversity.hill_diversity_branch_generic(counts, non_weights, 1)) < 0.00000001


def test_hill_diversity_pairwise_generic():

    # what to test against?
    # just run for now to check against unexpectedly thrown errors
    for counts, probs in probs_gen():
        non_weights = np.full((len(counts), len(counts)), 1)
        diversity.hill_diversity_pairwise_generic(counts, non_weights, 1)


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


def test_gini_simpson_diversity():
    '''
    USED FOR RESEARCH PURPOSES ONLY
    '''
    # just run for now to check against unexpectedly thrown errors
    for counts, probs in probs_gen():
        diversity.gini_simpson_diversity(counts)


def test_shannon_diversity():
    '''
    USED FOR RESEARCH PURPOSES ONLY
    '''
    # test against scipy entropy
    for counts, probs in probs_gen():
        assert abs(entropy(probs) - diversity.shannon_diversity(probs)) < 0.0000000001


def test_raos_quadratic_diversity():
    '''
    USED FOR RESEARCH PURPOSES ONLY
    '''
    # just run for now to check against unexpectedly thrown errors
    for counts, probs in probs_gen():
        non_weights = np.full((len(counts), len(counts)), 1)
        diversity.raos_quadratic_diversity(counts, non_weights)


def test_compute_mixed_uses():
    pass