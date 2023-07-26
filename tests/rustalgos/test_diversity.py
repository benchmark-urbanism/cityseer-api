# pyright: basic
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import entropy

from cityseer import config, rustalgos
from cityseer.tools import mock


def test_hill_diversity():
    # test hill diversity against scipy entropy
    for counts, probs in mock.mock_species_data():
        # check hill q=1 - this can be tested against scipy because hill q=1 is exponential of entropy
        assert np.allclose(
            rustalgos.hill_diversity(counts, q=1),
            np.exp(entropy(probs)),
            atol=config.ATOL,
            rtol=config.RTOL,
        )
        # check that hill q<1 and q>1 is reasonably close to scipy entropy
        # (different internal computation)
        assert np.allclose(
            rustalgos.hill_diversity(counts, 0.99999999),
            np.exp(entropy(probs)),
            atol=config.ATOL,
            rtol=config.RTOL,
        )
        assert np.allclose(
            rustalgos.hill_diversity(counts, 1.00000001),
            np.exp(entropy(probs)),
            atol=config.ATOL,
            rtol=config.RTOL,
        )
        # check for malformed q
        with pytest.raises(ValueError):
            rustalgos.hill_diversity(counts, q=-1)


def test_hill_diversity_branch_distance_wt():
    # test against hill diversity by setting all weights = 1
    for counts, _probs in mock.mock_species_data():
        non_weights = [1.0] * len(counts)
        non_beta = -0
        for q in [0, 1, 2]:
            assert np.allclose(
                rustalgos.hill_diversity(counts, q),
                rustalgos.hill_diversity_branch_distance_wt(counts, non_weights, q, non_beta, 1),
                atol=config.ATOL,
                rtol=config.RTOL,
            )


def hill_diversity_pairwise_distance_wt(
    class_counts,
    class_distances,
    q: np.float32,
    beta: np.float32,
    max_curve_wt: np.float32 = np.float32(1.0),
) -> np.float32:
    if len(class_counts) != len(class_distances):
        raise ValueError("Mismatching number of unique class counts and respective class distances.")
    if beta < 0:
        raise ValueError("Please provide the beta without the leading negative.")
    if q < 0:
        raise ValueError("Please select a non-zero value for q.")
    # catch potential division by zero situations
    num: int = class_counts.sum()
    if num == 0:
        return np.float32(0)
    # calculate Q
    agg_q = 0
    for i, class_count_i in enumerate(class_counts):
        if not class_count_i:
            continue
        a_i = class_count_i / num
        for j, class_count_j in enumerate(class_counts):
            # only need to examine the pair if j < i, otherwise double-counting
            if j > i:
                break
            if not class_count_j:
                continue
            a_j = class_count_j / num
            wt = rustalgos.clipped_beta_wt(beta, max_curve_wt, (class_distances[i] + class_distances[j]))
            # pairwise distances
            agg_q += wt * a_i * a_j
    # pairwise disparities weights can sometimes give rise to Q = 0... causing division by zero etc.
    if agg_q == 0:
        return np.float32(0)
    # if in the limit, use exponential
    if q == 1:
        div_pw_wt_lim = 0  # using the same variable name as non limit version causes errors for parallel
        for i, class_count_i in enumerate(class_counts):
            if not class_count_i:
                continue
            a_i = class_count_i / num
            for j, class_count_j in enumerate(class_counts):
                # only need to examine the pair if j < i, otherwise double-counting
                if j > i:
                    break
                if not class_count_j:
                    continue
                a_j = class_count_j / num
                # pairwise distances
                wt = rustalgos.clipped_beta_wt(beta, max_curve_wt, (class_distances[i] + class_distances[j]))
                div_pw_wt_lim += wt * a_i * a_j / agg_q * np.log(a_i * a_j / agg_q)  # sum
        # once summed
        div_pw_wt_lim = np.exp(-div_pw_wt_lim)
        return np.float32(div_pw_wt_lim ** (1 / 2))  # (FD_lim / Q) ** (1 / 2)
    # otherwise conventional form
    div_pw_wt = 0
    for i, class_count_i in enumerate(class_counts):
        if not class_count_i:
            continue
        a_i = class_count_i / num
        for j, class_count_j in enumerate(class_counts):
            # only need to examine the pair if j < i, otherwise double-counting
            if j > i:
                break
            if not class_count_j:
                continue
            a_j = class_count_j / num
            # pairwise distances
            wt = rustalgos.clipped_beta_wt(beta, max_curve_wt, (class_distances[i] + class_distances[j]))
            div_pw_wt += wt * (a_i * a_j / agg_q) ** q  # sum
    div_pw_wt = div_pw_wt ** (1 / (1 - q))
    return np.float32(div_pw_wt ** (1 / 2))  # (FD / Q) ** (1 / 2)


def test_hill_diversity_pairwise_distance_wt():
    for counts, _probs in mock.mock_species_data():
        non_weights = [1.0] * len(counts)
        non_beta = -0
        for q in [0, 1, 2]:
            assert np.allclose(
                rustalgos.hill_diversity_pairwise_distance_wt(counts, non_weights, q, non_beta, 1),
                hill_diversity_pairwise_distance_wt(np.array(counts), np.array(non_weights), q, non_beta, 1),
                rtol=config.RTOL,
                atol=config.ATOL,
            )


def gini_simpson_diversity(class_counts) -> np.float32:
    num: int = class_counts.sum()
    gini: np.float32 = np.float32(0)
    if num < 2:
        return gini
    for class_count in class_counts:
        gini += class_count / num * (class_count - 1) / (num - 1)
    return 1 - gini


def test_gini_simpson_diversity():
    """
    USED FOR RESEARCH PURPOSES ONLY
    """
    # just run for now to check against unexpectedly thrown errors
    for counts, probs in mock.mock_species_data():
        assert np.allclose(
            rustalgos.gini_simpson_diversity(counts),
            gini_simpson_diversity(np.array(counts)),
            rtol=config.RTOL,
            atol=config.ATOL,
        )


def test_shannon_diversity():
    """
    USED FOR RESEARCH PURPOSES ONLY
    """
    # test against scipy entropy
    for counts, probs in mock.mock_species_data():
        assert abs(entropy(probs) - rustalgos.shannon_diversity(counts)) < config.ATOL


def raos_quadratic_diversity(
    class_counts,
    wt_matrix,
    alpha: np.float32 = np.float32(1),
    beta: np.float32 = np.float32(1),
) -> np.float32:
    if len(class_counts) != len(wt_matrix):
        raise ValueError("Mismatching number of unique class counts and respective class taxonomy tiers.")
    if not wt_matrix.ndim == 2 or wt_matrix.shape[0] != wt_matrix.shape[1]:
        raise ValueError("Weights matrix must be an NxN pairwise matrix of disparity weights.")
    # catch potential division by zero situations
    num: int = class_counts.sum()
    if num < 2:
        return np.float32(0)
    raos: np.float32 = np.float32(0)  # variable for additive calculations of distance * p1 * p2
    for i, class_count_i in enumerate(class_counts):
        for j, class_count_j in enumerate(class_counts):
            # only need to examine the pair if j > i, otherwise double-counting
            if j > i:
                break
            p_i = class_count_i / num  # place here to catch division by zero for single element
            p_j = class_count_j / (num - 1)  # bias adjusted
            # calculate 3rd level disparity
            wt = wt_matrix[i][j]
            raos += wt**alpha * (p_i * p_j) ** beta
    return raos


def test_raos_quadratic_diversity():
    """
    USED FOR RESEARCH PURPOSES ONLY
    """
    # just run for now to check against unexpectedly thrown errors
    for counts, _probs in mock.mock_species_data():
        mock_matrix = np.full((len(counts), len(counts)), 1)
        assert np.allclose(
            rustalgos.raos_quadratic_diversity(counts, mock_matrix.tolist(), 1.0, 1.0),
            raos_quadratic_diversity(np.array(counts), mock_matrix),
            rtol=config.RTOL,
            atol=config.ATOL,
        )
