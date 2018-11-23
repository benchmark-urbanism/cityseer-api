import logging
from typing import Tuple, List
import numpy as np
from cityseer.algos import diversity


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def hill_diversity_branch_distance_wt(class_counts:np.array, class_distances:np.array, beta:float, q:float) -> float:

    if len(class_counts) != len(class_distances):
        raise ValueError('Mismatching number of unique class counts and respective class weights.')

    if beta < 0:
        raise ValueError('Please provide the beta/s without the leading negative.')

    if q < 0:
        raise ValueError('Please select a non-zero value for q.')

    class_weights = np.exp(class_distances * -beta)

    return diversity.hill_diversity_branch_generic(class_counts, class_weights, q)


def hill_diversity_pairwise_distance_wt(class_counts:np.array, class_distances:np.array, beta:float, q:float) -> float:

    if len(class_counts) != len(class_distances):
        raise ValueError('Mismatching number of unique class counts and respective class distances.')

    if beta < 0:
        raise ValueError('Please provide the beta/s without the leading negative.')

    if q < 0:
        raise ValueError('Please select a non-zero value for q.')

    wt_matrix = diversity.pairwise_distance_matrix(class_distances, beta)

    # return generic hill pairwise weighted
    return diversity.hill_diversity_pairwise_generic(class_counts, wt_matrix, q)


def hill_diversity_pairwise_disparity_wt(class_counts:np.array, class_tiers:np.array, class_weights:np.array, q:float) -> float:

    if len(class_counts) != len(class_tiers):
        raise ValueError('Mismatching number of unique class counts and respective class taxonomy tiers.')

    if class_tiers.shape[1] != len(class_weights):
        raise ValueError('The number of weights must correspond to the number of tiers for nodes i and j.')

    if q < 0:
        raise ValueError('Please select a non-zero value for q.')

    wt_matrix = diversity.pairwise_disparity_matrix(class_tiers, class_weights)

    # check for failed convergences
    failed_convergence = np.where(np.isnan(wt_matrix))
    if np.any(failed_convergence):
        idx = [(i, j) for i, j in zip(failed_convergence[0], failed_convergence[1]) if j > i]
        raise AttributeError(f'Found instances of no disparity convergence for index pairs {idx}.')

    # return generic hill pairwise weighted
    return diversity.hill_diversity_pairwise_generic(class_counts, wt_matrix, q)