import pytest
import numpy as np
from cityseer.metrics import landuses


def test_hill_diversity_pairwise_distance_wt():

    pass


def test_hill_diversity_pairwise_disparity_wt():

    # check for no convergence
    class_tiers = np.array([
        [0, 10, 100, 1000],
        [0, 10, 100, 1000]
    ])
    class_counts = np.array([1, 1])
    class_weights = np.array([3 / 3, 2 / 3, 1 / 3, 0])
    q = 0

    # check for mismatched data
    with pytest.raises(ValueError):
        landuses.hill_diversity_pairwise_disparity_wt(class_counts[:-1], class_tiers, class_weights, q)
    with pytest.raises(ValueError):
        landuses.hill_diversity_pairwise_disparity_wt(class_counts, class_tiers[:-1], class_weights, q)
    with pytest.raises(ValueError):
        landuses.hill_diversity_pairwise_disparity_wt(class_counts, class_tiers, class_weights[:-1], q)

    # check for negative q
    with pytest.raises(ValueError):
        landuses.hill_diversity_pairwise_disparity_wt(class_counts, class_tiers, class_weights, q=-1)

    # check for failed convergences
    class_tiers_no_convergence = np.array([
        [0, 10, 100, 1000],
        [1, 10, 100, 1000]
    ])
    with pytest.raises(AttributeError):
        landuses.hill_diversity_pairwise_disparity_wt(class_counts, class_tiers_no_convergence, class_weights, q)
