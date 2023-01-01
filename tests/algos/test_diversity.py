# pyright: basic
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import entropy

from cityseer import config
from cityseer.algos import diversity
from cityseer.tools import mock


def test_hill_diversity():
    # test hill diversity against scipy entropy
    for counts, probs in mock.mock_species_data():
        # check hill q=1 - this can be tested against scipy because hill q=1 is exponential of entropy
        assert np.allclose(
            diversity.hill_diversity(counts, q=1),
            np.exp(entropy(probs)),
            atol=config.ATOL,
            rtol=config.RTOL,
        )
        # check that hill q<1 and q>1 is reasonably close to scipy entropy
        # (different internal computation)
        assert np.allclose(
            diversity.hill_diversity(counts, 0.99999999),
            np.exp(entropy(probs)),
            atol=config.ATOL,
            rtol=config.RTOL,
        )
        assert np.allclose(
            diversity.hill_diversity(counts, 1.00000001),
            np.exp(entropy(probs)),
            atol=config.ATOL,
            rtol=config.RTOL,
        )
        # check for malformed q
        with pytest.raises(ValueError):
            diversity.hill_diversity(counts, q=-1)


def test_hill_diversity_branch_distance_wt():
    # test against hill diversity by setting all weights = 1
    for counts, probs in mock.mock_species_data():
        non_weights = np.full(len(counts), 1)
        non_beta = -0
        for q in [0, 1, 2]:
            assert np.allclose(
                diversity.hill_diversity(counts, q),
                diversity.hill_diversity_branch_distance_wt(counts, non_weights, q, non_beta),
                atol=config.ATOL,
                rtol=config.RTOL,
            )

        # check for malformed signatures
        with pytest.raises(ValueError):
            diversity.hill_diversity_branch_distance_wt(counts[:-1], non_weights, q=1, beta=0.005)
        with pytest.raises(ValueError):
            diversity.hill_diversity_branch_distance_wt(counts, non_weights[:-1], q=1, beta=0.005)
        with pytest.raises(ValueError):
            diversity.hill_diversity_branch_distance_wt(counts, non_weights, q=1, beta=-0.005)
        with pytest.raises(ValueError):
            diversity.hill_diversity_branch_distance_wt(counts, non_weights, q=-1, beta=0.005)


def test_hill_diversity_pairwise_distance_wt():
    for counts, probs in mock.mock_species_data():
        non_weights = np.full(len(counts), 1)
        non_beta = -0
        for q in [0, 1, 2]:
            # what to test against? For now, check successful run
            diversity.hill_diversity_pairwise_distance_wt(counts, non_weights, q, non_beta)

        # check for malformed signatures
        with pytest.raises(ValueError):
            diversity.hill_diversity_pairwise_distance_wt(counts[:-1], non_weights, q=1, beta=0.005)
        with pytest.raises(ValueError):
            diversity.hill_diversity_pairwise_distance_wt(counts, non_weights[:-1], q=1, beta=0.005)
        with pytest.raises(ValueError):
            diversity.hill_diversity_pairwise_distance_wt(counts, non_weights, q=1, beta=-0.005)
        with pytest.raises(ValueError):
            diversity.hill_diversity_pairwise_distance_wt(counts, non_weights, q=-1, beta=0.005)


def test_hill_diversity_pairwise_matrix_wt():
    for counts, probs in mock.mock_species_data():
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
    """
    USED FOR RESEARCH PURPOSES ONLY
    """
    # just run for now to check against unexpectedly thrown errors
    for counts, probs in mock.mock_species_data():
        diversity.gini_simpson_diversity(counts)


def test_shannon_diversity():
    """
    USED FOR RESEARCH PURPOSES ONLY
    """
    # test against scipy entropy
    for counts, probs in mock.mock_species_data():
        assert abs(entropy(probs) - diversity.shannon_diversity(probs)) < 0.0000000001


def test_raos_quadratic_diversity():
    """
    USED FOR RESEARCH PURPOSES ONLY
    """
    # just run for now to check against unexpectedly thrown errors
    for counts, probs in mock.mock_species_data():
        mock_matrix = np.full((len(counts), len(counts)), 1)
        diversity.raos_quadratic_diversity(counts, mock_matrix)
