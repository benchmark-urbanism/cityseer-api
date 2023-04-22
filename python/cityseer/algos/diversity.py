from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numba import njit  # type: ignore

from cityseer import config
from cityseer.algos import common


@njit(cache=True, fastmath=config.FASTMATH, nogil=True)
def hill_diversity(class_counts: npt.NDArray[np.int_], q: np.float32) -> np.float32:
    """
    Compute Hill diversity.

    Hill numbers - express actual diversity as opposed e.g. to Gini-Simpson (probability) and Shannon (information)

    exponent at 1 results in undefined because of 1/0 - but limit exists as exp(entropy)
    Ssee "Entropy and diversity" by Lou Jost

    Exponent at 0 = variety - i.e. count of unique species
    Exponent at 1 = unity
    Exponent at 2 = diversity form of simpson index

    """
    if q < 0:
        raise ValueError("Please select a non-zero value for q.")
    num: int = class_counts.sum()
    # catch potential division by zero situations
    if num == 0:
        hill = 0
    # hill number defined in the limit as the exponential of information entropy
    if q == 1:
        ent = 0
        for class_count in class_counts:
            if class_count:  # if not 0
                prob = class_count / num  # the probability of this class
                ent += prob * np.log(prob)  # sum entropy
        hill = np.exp(-ent)  # return exponent of entropy
    # otherwise use the usual form of Hill numbers
    else:
        div = 0
        for class_count in class_counts:
            if class_count:
                prob = class_count / num  # the probability of this class
                div += prob**q  # sum
        hill = div ** (1 / (1 - q))  # return as equivalent species
    return hill


@njit(cache=True, fastmath=config.FASTMATH, nogil=True)
def hill_diversity_branch_distance_wt(
    class_counts: npt.NDArray[np.int_],
    class_distances: npt.NDArray[np.float32],
    q: np.float32,
    beta: np.float32,
    max_curve_wt: np.float32 = np.float32(1.0),
) -> np.float32:
    """
    Compute Hill diversity weighted by branch distances.

    Based on unified framework for species diversity in Chao, Chiu, Jost 2014.
    See table on page 308 and surrounding text

    In this case the presumption is that you supply branch weights in the form of negative exponential distance weights
    i.e. pedestrian walking distance decay which weights more distant locations more weakly than nearer locations
    This means that the walking distance to a landuse impacts how strongly it contributes to diversity

    The weighting is based on the nearest of each landuse
    This is debatably most relevant to q=0

    """
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
    # find T
    agg_t = 0
    for class_count, class_dist in zip(class_counts, class_distances):
        if class_count:
            proportion = class_count / num
            wt = common.clipped_beta_wt(beta, max_curve_wt, class_dist)
            agg_t += wt * proportion
    # hill number defined in the limit as the exponential of information entropy
    if q == 1:
        div_branch_wt_lim = 0
        # get branch lengths and class abundances
        for class_count, class_dist in zip(class_counts, class_distances):
            if class_count:
                proportion = class_count / num
                wt = common.clipped_beta_wt(beta, max_curve_wt, class_dist)
                div_branch_wt_lim += wt * proportion / agg_t * np.log(proportion / agg_t)  # sum entropy
        # return exponent of entropy
        div_branch_wt_lim = np.exp(-div_branch_wt_lim)
        return div_branch_wt_lim  # / T
    # otherwise use the usual form of Hill numbers
    div_branch_wt = 0
    # get branch lengths and class abundances
    for class_count, class_dist in zip(class_counts, class_distances):
        if class_count:
            a = class_count / num
            wt = common.clipped_beta_wt(beta, max_curve_wt, class_dist)
            div_branch_wt += wt * (a / agg_t) ** q  # sum
    # once summed, apply q
    div_branch_wt = div_branch_wt ** (1 / (1 - q))
    return div_branch_wt  # / T


@njit(cache=True, fastmath=config.FASTMATH, nogil=True)
def hill_diversity_pairwise_distance_wt(
    class_counts: npt.NDArray[np.int_],
    class_distances: npt.NDArray[np.float32],
    q: np.float32,
    beta: np.float32,
    max_curve_wt: np.float32 = np.float32(1.0),
) -> np.float32:
    """
    Compute Hill diversity weighted by pairwise distances.

    This is the distances version - see below for disparity matrix version

    Based on unified framework for species diversity in Chao, Chiu, Jost 2014
    See table on page 308 and surrounding text

    In this case the presumption is that you supply branch weights in the form of negative exponential distance weights
    i.e. pedestrian walking distance decay which weights more distant locations more weakly than nearer locations
    This means that the walking distance to a landuse impacts how strongly it contributes to diversity

    Functional diversity takes the pairwise form, thus distances are based on pairwise i to j distances via the node k
    Remember these are already distilled species counts - so it is OK to use closest distance to each species

    This is different to the non-pairwise form of the phylogenetic version which simply takes singular distance k to i

    """
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
            wt = common.clipped_beta_wt(beta, max_curve_wt, (class_distances[i] + class_distances[j]))
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
                wt = common.clipped_beta_wt(beta, max_curve_wt, (class_distances[i] + class_distances[j]))
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
            wt = common.clipped_beta_wt(beta, max_curve_wt, (class_distances[i] + class_distances[j]))
            div_pw_wt += wt * (a_i * a_j / agg_q) ** q  # sum
    div_pw_wt = div_pw_wt ** (1 / (1 - q))
    return np.float32(div_pw_wt ** (1 / 2))  # (FD / Q) ** (1 / 2)


@njit(cache=True, fastmath=config.FASTMATH, nogil=True)
def hill_diversity_pairwise_matrix_wt(
    class_counts: npt.NDArray[np.int_], wt_matrix: npt.NDArray[np.float32], q: np.float32
) -> np.float32:
    """
    Hill diversity weighted by pairwise weights matrix.

    This is the matrix version - requires a precomputed (e.g. disparity) matrix for all classes.

    See above for distance version.

    """
    if len(class_counts) != len(wt_matrix):
        raise ValueError("Mismatching number of unique class counts and dimensionality of class weights matrix.")
    if not wt_matrix.ndim == 2 or wt_matrix.shape[0] != wt_matrix.shape[1]:
        raise ValueError("Weights matrix must be an NxN pairwise matrix of disparity weights.")
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
            wt = wt_matrix[i][j]
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
                wt = wt_matrix[i][j]
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
            wt = wt_matrix[i][j]
            div_pw_wt += wt * (a_i * a_j / agg_q) ** q  # sum
    div_pw_wt = div_pw_wt ** (1 / (1 - q))
    return np.float32(div_pw_wt ** (1 / 2))  # (FD / Q) ** (1 / 2)


@njit(cache=True, fastmath=config.FASTMATH, nogil=True)
def gini_simpson_diversity(class_counts: npt.NDArray[np.int_]) -> np.float32:
    """
    Gini-Simpson diversity.

    Gini transformed to 1 − λ
    Probability that two individuals picked at random do not represent the same species (Tuomisto)

    Ordinarily:
    D = 1 - sum(p**2) where p = Xi/N

    Bias corrected:
    D = 1 - sum(Xi/N * (Xi-1/N-1))

    """
    num: int = class_counts.sum()
    gini: np.float32 = np.float32(0)
    # catch potential division by zero situations
    if num < 2:
        return gini
    # compute bias corrected gini-simpson
    for class_count in class_counts:
        gini += class_count / num * (class_count - 1) / (num - 1)
    return 1 - gini


@njit(cache=True, fastmath=config.FASTMATH, nogil=True)
def shannon_diversity(class_counts: npt.NDArray[np.int_]) -> np.float32:
    """
    Shannon diversity (information entropy).

    Entropy
    p = Xi/N
    S = -sum(p * log(p))
    Uncertainty of the species identity of an individual picked at random (Tuomisto)

    """
    num: int = class_counts.sum()
    shannon: np.float32 = np.float32(0)
    # catch potential division by zero situations
    if num == 0:
        return shannon
    # compute
    for class_count in class_counts:
        if class_count:
            prob = class_count / num  # the probability of this class
            shannon += prob * np.log(prob)  # sum entropy
    return -shannon  # remember negative


@njit(cache=True, fastmath=config.FASTMATH, nogil=True)
def raos_quadratic_diversity(
    class_counts: npt.NDArray[np.int_],
    wt_matrix: npt.NDArray[np.float32],
    alpha: np.float32 = np.float32(1),
    beta: np.float32 = np.float32(1),
) -> np.float32:
    """
    Rao's quadratic diversity.

    Bias corrected and based on disparity

    Sum of weighted pairwise products

    Note that Stirling's diversity is a rediscovery of Rao's quadratic diversity
    Though adds alpha and beta exponents to tweak weights of disparity dij and pi * pj, respectively
    This is a hybrid of the two, i.e. including alpha and beta options and adjusted for bias
    Rd = sum(dij * Xi/N * (Xj/N-1))

    Behaviour is controlled using alpha and beta exponents
    0 and 0 reduces to variety (effectively a count of unique types)
    0 and 1 reduces to balance (half-gini - pure balance, no weights)
    1 and 0 reduces to disparity (effectively a weighted count)
    1 and 1 is base stirling diversity / raos quadratic

    """
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
