import numpy as np
from numba import njit


@njit(cache=True)
def hill_diversity(class_counts: np.ndarray, q: float) -> float:
    '''
    Hill numbers - express actual diversity as opposed e.g. to Gini-Simpson (probability) and Shannon (information)

    exponent at 1 results in undefined because of 1/0 - but limit exists as exp(entropy)
    Ssee "Entropy and diversity" by Lou Jost

    Exponent at 0 = variety - i.e. count of unique species
    Exponent at 1 = unity
    Exponent at 2 = diversity form of simpson index
    '''

    if q < 0:
        raise ValueError('Please select a non-zero value for q.')

    N = class_counts.sum()
    # catch potential division by zero situations
    if N == 0:
        return 0
    # hill number defined in the limit as the exponential of information entropy
    if q == 1:
        H = 0
        for a in class_counts:
            if a:  # if not 0
                p = a / N  # the probability of this class
                H += p * np.log(p)  # sum entropy
        return np.exp(-H)  # return exponent of entropy
    # otherwise use the usual form of Hill numbers
    else:
        D = 0
        for a in class_counts:
            if a:
                p = a / N  # the probability of this class
                D += p ** q  # sum
        return D ** (1 / (1 - q))  # return as equivalent species


@njit(cache=True)
def hill_diversity_branch_distance_wt(class_counts: np.ndarray,
                                      class_distances: np.array,
                                      q: float,
                                      beta: float) -> float:
    '''
    Based on unified framework for species diversity in Chao, Chiu, Jost 2014
    See table on page 308 and surrounding text

    In this case the presumption is that you supply branch weights in the form of negative exponential distance weights
    i.e. pedestrian walking distance decay which weights more distant locations more weakly than nearer locations
    This means that the walking distance to a landuse impacts how strongly it contributes to diversity

    The weighting is based on the nearest of each landuse
    This is debatably most relevant to q=0
    '''

    if len(class_counts) != len(class_distances):
        raise ValueError('Mismatching number of unique class counts and respective class distances.')

    if beta > 0:
        raise ValueError('Please provide the beta with the leading negative.')

    if q < 0:
        raise ValueError('Please select a non-zero value for q.')

    # catch potential division by zero situations
    N = class_counts.sum()
    if N == 0:
        return 0

    # find T
    T = 0
    for i in range(len(class_counts)):
        if class_counts[i]:
            a = class_counts[i] / N
            wt = np.exp(class_distances[i] * beta)
            T += wt * a

    # hill number defined in the limit as the exponential of information entropy
    if q == 1:
        PD_lim = 0  # using the same variable name as non limit version causes errors for parallel
        # get branch lengths and class abundances
        for i in range(len(class_counts)):
            if class_counts[i]:
                a = class_counts[i] / N
                wt = np.exp(class_distances[i] * beta)
                PD_lim += wt * a / T * np.log(a / T)  # sum entropy
        # return exponent of entropy
        PD_lim = np.exp(-PD_lim)
        return PD_lim  # / T
    # otherwise use the usual form of Hill numbers
    else:
        PD = 0
        # get branch lengths and class abundances
        for i in range(len(class_counts)):
            if class_counts[i]:
                a = class_counts[i] / N
                wt = np.exp(class_distances[i] * beta)
                PD += wt * (a / T) ** q  # sum
        # once summed, apply q
        PD = PD ** (1 / (1 - q))
        return PD  # / T


@njit(cache=True)
def hill_diversity_pairwise_distance_wt(class_counts: np.ndarray,
                                        class_distances: np.ndarray,
                                        q: float,
                                        beta: float) -> float:
    '''
    This is the distances version - see below for disparity matrix version

    Based on unified framework for species diversity in Chao, Chiu, Jost 2014
    See table on page 308 and surrounding text

    In this case the presumption is that you supply branch weights in the form of negative exponential distance weights
    i.e. pedestrian walking distance decay which weights more distant locations more weakly than nearer locations
    This means that the walking distance to a landuse impacts how strongly it contributes to diversity

    Functional diversity takes the pairwise form, thus distances are based on pairwise i to j distances via the node k
    Remember these are already distilled species counts - so it is OK to use closest distance to each species

    This is different to the non-pairwise form of the phylogenetic version which simply takes singular distance k to i
    '''

    if len(class_counts) != len(class_distances):
        raise ValueError('Mismatching number of unique class counts and respective class distances.')

    if beta > 0:
        raise ValueError('Please provide the beta with the leading negative.')

    if q < 0:
        raise ValueError('Please select a non-zero value for q.')

    # catch potential division by zero situations
    N = class_counts.sum()
    if N == 0:
        return 0

    # calculate Q
    Q = 0
    for i in range(len(class_counts)):
        if class_counts[i]:
            a_i = class_counts[i] / N
            for j in range(len(class_counts)):
                # only need to examine the pair if j < i, otherwise double-counting
                if j > i:
                    break
                if class_counts[j]:
                    a_j = class_counts[j] / N
                    wt = np.exp((class_distances[i] + class_distances[j]) * beta)
                    # pairwise distances
                    Q += wt * a_i * a_j

    # pairwise disparities weights can sometimes give rise to Q = 0... causing division by zero etc.
    if Q == 0:
        return 0

    # if in the limit, use exponential
    if q == 1:
        FD_lim = 0  # using the same variable name as non limit version causes errors for parallel
        for i in range(len(class_counts)):
            if class_counts[i]:
                a_i = class_counts[i] / N
                for j in range(len(class_counts)):
                    # only need to examine the pair if j < i, otherwise double-counting
                    if j > i:
                        break
                    if class_counts[j]:
                        a_j = class_counts[j] / N
                        # pairwise distances
                        wt = np.exp((class_distances[i] + class_distances[j]) * beta)
                        FD_lim += wt * a_i * a_j / Q * np.log(a_i * a_j / Q)  # sum
        # once summed
        FD_lim = np.exp(-FD_lim)
        return FD_lim ** (1 / 2)  # (FD_lim / Q) ** (1 / 2)
    # otherwise conventional form
    else:
        FD = 0
        for i in range(len(class_counts)):
            if class_counts[i]:
                a_i = class_counts[i] / N
                for j in range(len(class_counts)):
                    # only need to examine the pair if j < i, otherwise double-counting
                    if j > i:
                        break
                    if class_counts[j]:
                        a_j = class_counts[j] / N
                        # pairwise distances
                        wt = np.exp((class_distances[i] + class_distances[j]) * beta)
                        FD += wt * (a_i * a_j / Q) ** q  # sum
        FD = FD ** (1 / (1 - q))
        return FD ** (1 / 2)  # (FD / Q) ** (1 / 2)


@njit(cache=True)
def hill_diversity_pairwise_matrix_wt(class_counts: np.ndarray, wt_matrix: np.ndarray, q: float) -> float:
    '''
    This is the matrix version - requires a precomputed (e.g. disparity) matrix for all classes.

    See above for distance version.
    '''

    if len(class_counts) != len(wt_matrix):
        raise ValueError('Mismatching number of unique class counts and dimensionality of class weights matrix.')

    if not wt_matrix.ndim == 2 or wt_matrix.shape[0] != wt_matrix.shape[1]:
        raise ValueError('Weights matrix must be an NxN pairwise matrix of disparity weights.')

    if q < 0:
        raise ValueError('Please select a non-zero value for q.')

    # catch potential division by zero situations
    N = class_counts.sum()
    if N == 0:
        return 0

    # calculate Q
    Q = 0
    for i in range(len(class_counts)):
        if class_counts[i]:
            a_i = class_counts[i] / N
            for j in range(len(class_counts)):
                # only need to examine the pair if j < i, otherwise double-counting
                if j > i:
                    break
                if class_counts[j]:
                    a_j = class_counts[j] / N
                    wt = wt_matrix[i][j]
                    # pairwise distances
                    Q += wt * a_i * a_j

    # pairwise disparities weights can sometimes give rise to Q = 0... causing division by zero etc.
    if Q == 0:
        return 0

    # if in the limit, use exponential
    if q == 1:
        FD_lim = 0  # using the same variable name as non limit version causes errors for parallel
        for i in range(len(class_counts)):
            if class_counts[i]:
                a_i = class_counts[i] / N
                for j in range(len(class_counts)):
                    # only need to examine the pair if j < i, otherwise double-counting
                    if j > i:
                        break
                    if class_counts[j]:
                        a_j = class_counts[j] / N
                        # pairwise distances
                        wt = wt_matrix[i][j]
                        FD_lim += wt * a_i * a_j / Q * np.log(a_i * a_j / Q)  # sum
        # once summed
        FD_lim = np.exp(-FD_lim)
        return FD_lim ** (1 / 2)  # (FD_lim / Q) ** (1 / 2)
    # otherwise conventional form
    else:
        FD = 0
        for i in range(len(class_counts)):
            if class_counts[i]:
                a_i = class_counts[i] / N
                for j in range(len(class_counts)):
                    # only need to examine the pair if j < i, otherwise double-counting
                    if j > i:
                        break
                    if class_counts[j]:
                        a_j = class_counts[j] / N
                        # pairwise distances
                        wt = wt_matrix[i][j]
                        FD += wt * (a_i * a_j / Q) ** q  # sum
        FD = FD ** (1 / (1 - q))
        return FD ** (1 / 2)  # (FD / Q) ** (1 / 2)


@njit(cache=True)
def gini_simpson_diversity(class_counts: np.ndarray) -> float:
    '''
    Gini-Simpson
    Gini transformed to 1 − λ
    Probability that two individuals picked at random do not represent the same species (Tuomisto)

    Ordinarily:
    D = 1 - sum(p**2) where p = Xi/N

    Bias corrected:
    D = 1 - sum(Xi/N * (Xi-1/N-1))
    '''
    N = class_counts.sum()
    G = 0
    # catch potential division by zero situations
    if N < 2:
        return G
    # compute bias corrected gini-simpson
    for c in class_counts:
        if c:
            G += c / N * (c - 1) / (N - 1)
    return 1 - G


@njit(cache=True)
def shannon_diversity(class_counts: np.ndarray) -> float:
    '''
    Entropy
    p = Xi/N
    S = -sum(p * log(p))
    Uncertainty of the species identity of an individual picked at random (Tuomisto)
    '''
    N = class_counts.sum()
    H = 0
    # catch potential division by zero situations
    if N == 0:
        return H
    # compute
    for a in class_counts:
        if a:
            p = a / N  # the probability of this class
            H += p * np.log(p)  # sum entropy
    return -H  # remember negative


@njit(cache=True)
def raos_quadratic_diversity(class_counts: np.ndarray,
                             wt_matrix: np.array,
                             alpha: float = 1,
                             beta: float = 1) -> float:
    '''
    Rao's quadratic - bias corrected and based on disparity

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
    '''

    if len(class_counts) != len(wt_matrix):
        raise ValueError('Mismatching number of unique class counts and respective class taxonomy tiers.')

    if not wt_matrix.ndim == 2 or wt_matrix.shape[0] != wt_matrix.shape[1]:
        raise ValueError('Weights matrix must be an NxN pairwise matrix of disparity weights.')

    # catch potential division by zero situations
    N = class_counts.sum()
    if N < 2:
        return 0

    R = 0  # variable for additive calculations of distance * p1 * p2
    for i in range(len(class_counts)):
        # parallelise only inner loop
        for j in range(len(class_counts)):
            # only need to examine the pair if j <= i, otherwise double-counting
            if j > i:
                break
            if class_counts[i] and class_counts[j]:
                p_i = class_counts[i] / N  # place here to catch division by zero for single element
                p_j = class_counts[j] / (N - 1)  # bias adjusted
                if p_i and p_j:  # if the probabilities aren't 0
                    # calculate 3rd level disparity
                    wt = wt_matrix[i][j]
                    R += wt ** alpha * (p_i * p_j) ** beta
    return R
