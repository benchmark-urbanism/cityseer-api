"""Functions for calculating ecological diversity metrics applied to spatial data counts."""

from __future__ import annotations

def hill_diversity(class_counts: list[int], q: float) -> float:
    r"""
    Compute Hill diversity (effective number of types) for given counts and order q.

    q=0: Richness (count of unique types > 0).
    q=1: Exponential of Shannon entropy.
    q=2: Inverse Simpson index.

    Parameters
    ----------
    class_counts: list[int]
        Counts for each class/type.
    q: float
        Order of diversity (>= 0).

    Returns
    -------
    float
        Hill diversity value.
    """
    ...

def hill_diversity_branch_distance_wt(
    class_counts: list[int],
    class_distances: list[float],
    q: float,
    beta: float,
    max_curve_wt: float,
) -> float:
    r"""
    Compute Hill diversity weighted by the distance to the *nearest* instance of each class.

    Uses exponential decay weights based on `beta` and `max_curve_wt`.

    Parameters
    ----------
    class_counts: list[int]
        Counts for each class.
    class_distances: list[float]
        Distance to the nearest instance of each corresponding class.
    q: float
        Order of diversity (>= 0).
    beta: float
        Decay parameter ($\beta >= 0$).
    max_curve_wt: float
        Maximum weight from `clip_wts_curve`.

    Returns
    -------
    float
        Distance-weighted Hill diversity.
    """
    ...

def hill_diversity_pairwise_distance_wt(
    class_counts: list[int],
    class_distances: list[float],
    q: float,
    beta: float,
    max_curve_wt: float,
) -> float:
    r"""
    Compute Hill diversity weighted by *pairwise* distances between classes.

    Weights based on the sum of distances to the nearest instances of pairs of classes (i, j).
    Uses exponential decay weights based on `beta` and `max_curve_wt`.

    Parameters
    ----------
    class_counts: list[int]
        Counts for each class.
    class_distances: list[float]
        Distance to the nearest instance of each corresponding class.
    q: float
        Order of diversity (>= 0).
    beta: float
        Decay parameter ($\beta >= 0$).
    max_curve_wt: float
        Maximum weight from `clip_wts_curve`.

    Returns
    -------
    float
        Pairwise distance-weighted Hill diversity.
    """
    ...

def gini_simpson_diversity(class_counts: list[int]) -> float:
    """
    Compute Gini-Simpson diversity index (probability of inter-type encounter).

    Uses bias-corrected formula: $D = 1 - \\sum (\\frac{X_i}{N} \\cdot \\frac{X_i-1}{N-1})$.

    Parameters
    ----------
    class_counts: list[int]
        Counts for each class.

    Returns
    -------
    float
        Gini-Simpson diversity (0 to 1).
    """
    ...

def shannon_diversity(class_counts: list[int]) -> float:
    """
    Compute Shannon diversity (entropy).

    Formula: $H = -\\sum (p_i \\cdot \\ln(p_i))$, where $p_i = X_i / N$.

    Parameters
    ----------
    class_counts: list[int]
        Counts for each class.

    Returns
    -------
    float
        Shannon diversity (>= 0).
    """
    ...

def raos_quadratic_diversity(class_counts: list[int], wt_matrix: list[list[float]], alpha: float, beta: float) -> float:
    r"""
    Compute Rao's quadratic diversity (or Stirling's diversity).

    General form: $R_Q = \sum_{i,j} d_{ij}^{\alpha} (p_i p_j)^{\beta}$.
    Uses bias-corrected probabilities.

    Parameters
    ----------
    class_counts: list[int]
        Counts for each class.
    wt_matrix: list[list[float]]
        Pairwise dissimilarity/distance matrix ($d_{ij}$) between classes.
    alpha: float
        Exponent for dissimilarity weights (>= 0).
    beta: float
        Exponent for probability product weights (>= 0).

    Returns
    -------
    float
        Rao's quadratic diversity.
    """
    ...
