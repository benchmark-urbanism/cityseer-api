"""Functions for calculating diversity metrics."""

from __future__ import annotations

def hill_diversity(class_counts: list[int], q: float) -> float: ...
def hill_diversity_branch_distance_wt(
    class_counts: list[int],
    class_distances: list[float],
    q: float,
    beta: float,
    max_curve_wt: float,
) -> float: ...
def hill_diversity_pairwise_distance_wt(
    class_counts: list[int],
    class_distances: list[float],
    q: float,
    beta: float,
    max_curve_wt: float,
) -> float: ...
def gini_simpson_diversity(class_counts: list[int]) -> float: ...
def shannon_diversity(class_counts: list[int]) -> float: ...
def raos_quadratic_diversity(
    class_counts: list[int], wt_matrix: list[list[float]], alpha: float, beta: float
) -> float: ...
