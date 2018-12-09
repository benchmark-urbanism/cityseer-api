from typing import Tuple

import numpy as np
from numba import njit

from cityseer.algos import data, checks


# cc = CC('diversity')


# @cc.export('hill_diversity', '(uint64[:], float64)')
@njit
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


# @cc.export('hill_diversity_branch_generic', '(uint64[:], float64[:], float64)')
@njit
def hill_diversity_branch_generic(class_counts: np.ndarray, tier_weights: np.ndarray, q: float) -> float:
    '''
    Based on unified framework for species diversity in Chao, Chiu, Jost 2014
    See table on page 308 and surrounding text

    In this case the presumption is that you supply branch weights in the form of negative exponential distance weights
    i.e. pedestrian walking distance decay which weights more distant locations more weakly than nearer locations
    This means that the walking distance to a landuse impacts how strongly it contributes to diversity

    The weighting is based on the nearest of each landuse
    This is debatably most relevant to q=0
    '''

    if len(class_counts) != len(tier_weights):
        raise ValueError('Mismatching number of unique class counts and respective class weights.')

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
            wt = tier_weights[i]
            T += wt * a

    # hill number defined in the limit as the exponential of information entropy
    if q == 1:
        PD_lim = 0  # using the same variable name as non limit version causes errors for parallel
        # get branch lengths and class abundances
        for i in range(len(class_counts)):
            if class_counts[i]:
                a = class_counts[i] / N
                wt = tier_weights[i]
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
                wt = tier_weights[i]
                PD += wt * (a / T) ** q  # sum
        # once summed, apply q
        PD = PD ** (1 / (1 - q))
        return PD  # / T


# @cc.export('hill_diversity_branch_distance_wt', '(uint64[:], float64[:], float64, float64)')
@njit
def hill_diversity_branch_distance_wt(class_counts: np.array,
                                      class_distances: np.array,
                                      beta: float,
                                      q: float) -> float:
    if beta >= 0:
        raise ValueError('Please provide the beta/s with the leading negative.')

    tier_weights = np.exp(class_distances * beta)

    return hill_diversity_branch_generic(class_counts, tier_weights, q)


# @cc.export('hill_diversity_pairwise_generic', '(uint64[:], float64[:,:], float64)')
@njit
def hill_diversity_pairwise_generic(class_counts: np.ndarray, wt_matrix: np.ndarray, q: float) -> float:
    '''
    Based on unified framework for species diversity in Chao, Chiu, Jost 2014
    See table on page 308 and surrounding text

    In this case the presumption is that you supply branch weights in the form of negative exponential distance weights
    i.e. pedestrian walking distance decay which weights more distant locations more weakly than nearer locations
    This means that the walking distance to a landuse impacts how strongly it contributes to diversity

    Functional diversity takes the pairwise form, thus distances are based on pairwise i to j distances via the node k

    This is different to the non-pairwise form of the phylogenetic version which simply takes singular distance k to i
    '''

    if len(class_counts) != len(wt_matrix):
        raise ValueError('Mismatching number of unique class counts vs. weights matrix dimensionality.')

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


# @cc.export('pairwise_distance_matrix', '(float64[:], float64)')
@njit
def pairwise_distance_matrix(distances: np.ndarray, beta: float) -> np.ndarray:
    if beta >= 0:
        raise ValueError('Please provide the beta/s with the leading negative.')

    # prepare the weights matrix
    wt_matrix = np.full((len(distances), len(distances)), np.inf)
    for i_idx in range(len(distances)):
        for j_idx in range(len(distances)):
            # no need to repeat where j > i
            if j_idx > i_idx:
                break
            # in this case i == j distances have to be calculated
            # this is because diversity of 1 class (but weighted) still has to be factored (e.g. hill with 1 element)
            else:
                # write in both directions - though not technically necessary
                w = np.exp((distances[i_idx] + distances[j_idx]) * beta)
                wt_matrix[i_idx][j_idx] = w
                wt_matrix[j_idx][i_idx] = w
    return wt_matrix


# @cc.export('hill_diversity_pairwise_distance_wt', '(uint64[:], float64[:], float64, float64)')
@njit
def hill_diversity_pairwise_distance_wt(class_counts: np.array,
                                        class_distances: np.array,
                                        beta: float,
                                        q: float) -> float:
    if len(class_counts) != len(class_distances):
        raise ValueError('Mismatching number of unique class counts and class distances.')

    wt_matrix = pairwise_distance_matrix(class_distances, beta)

    return hill_diversity_pairwise_generic(class_counts, wt_matrix, q)


# @cc.export('hill_diversity_pairwise_disparity_wt', '(uint64[:], uint64[:,:], float64[:], float64)')
@njit
def hill_diversity_pairwise_disparity_wt(class_counts: np.array,
                                         wt_matrix: np.array,
                                         q: float) -> float:
    if len(class_counts) != len(wt_matrix):
        raise ValueError('Mismatching number of unique class counts and respective class taxonomy tiers.')

    if not wt_matrix.ndim == 2 or wt_matrix.shape[0] != wt_matrix.shape[1]:
        raise ValueError('Weights matrix must be an NxN pairwise matrix of disparity weights.')

    return hill_diversity_pairwise_generic(class_counts, wt_matrix, q)


# explicit return required, otherwise numba throws:
# TypeError: invalid signature: 'str' instance not allowed
# @cc.export('gini_simpson_diversity', 'float64(uint64[:])')
@njit
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


# @cc.export('shannon_diversity', 'float64(uint64[:])')
@njit
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


# @cc.export('raos_quadratic_diversity', '(uint64[:], float64[:,:], float64, float64)')
@njit
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


@njit
def local_landuses(node_map: np.ndarray,
                   edge_map: np.ndarray,
                   data_map: np.ndarray,
                   distances: np.ndarray,
                   betas: np.ndarray,
                   qs: np.ndarray = np.array([]),
                   mixed_use_keys: np.ndarray = np.array([]),
                   accessibility_keys: np.ndarray = np.array([]),
                   cl_disparity_wt_matrix: np.ndarray = np.array([[]]),
                   angular: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    NODE MAP:
    0 - x
    1 - y
    2 - live
    3 - edge indx
    4 - weight

    EDGE MAP:
    0 - start node
    1 - end node
    2 - length in metres
    3 - impedance

    DATA MAP:
    0 - x
    1 - y
    2 - live
    3 - data class
    4 - assigned network index - nearest
    5 - assigned network index - next-nearest
    '''
    checks.check_data_map(data_map)
    checks.check_network_types(node_map, edge_map)
    checks.check_distances_and_betas(distances, betas)

    if len(mixed_use_keys) == 0 and len(accessibility_keys) == 0:
        raise ValueError('Please specify at least one mixed use or accessibility measure to compute.')

    for k in mixed_use_keys:
        if k < 4:
            if len(qs) == 0:
                raise ValueError(
                    'All hill diversity measures require that at least one value of q is specified.')
        if k == 3 or k == 6:
            if len(cl_disparity_wt_matrix) == 0:
                raise ValueError(
                    'Hill / Rao pairwise disparity measures require a class disparity weights matrix.')

    # establish variables
    n = len(node_map)
    d_n = len(distances)
    q_n = len(qs)
    unique_classes_n = int(data_map[:, 3].max() + 1)
    max_dist = distances.max()
    netw_nodes_live = node_map[:, 2]

    # setup data structures
    mixed_use_hill_data = np.full((4, q_n, d_n, n), np.nan)
    mixed_use_other_data = np.full((3, d_n, n), np.nan)
    accessibility_data = np.full((len(accessibility_keys), d_n, n), 0.0)
    accessibility_data_wt = np.full((len(accessibility_keys), d_n, n), 0.0)

    def mixed_use_metrics(idx, classes_counts, classes_distances, beta, q, cl_disparity_wt_matrix):

        if idx == 0:
            return hill_diversity(classes_counts, q)

        if idx == 1:
            return hill_diversity_branch_distance_wt(classes_counts, classes_distances, beta, q)

        if idx == 2:
            return hill_diversity_pairwise_distance_wt(classes_counts, classes_distances, beta, q)

        if idx == 3:
            return hill_diversity_pairwise_disparity_wt(classes_counts, cl_disparity_wt_matrix, q)

        if idx == 4:
            return shannon_diversity(classes_counts)

        if idx == 5:
            return gini_simpson_diversity(classes_counts)

        if idx == 6:
            return raos_quadratic_diversity(classes_counts, cl_disparity_wt_matrix)

        if idx > 6:
            raise ValueError('Mixed-use key exceeds the available options.')

    for src_idx in range(n):

        # numba no object mode can only handle basic printing
        if src_idx % 10000 == 0:
            print('...progress:', round(src_idx / n * 100, 2), '%')

        # only compute for live nodes
        if not netw_nodes_live[src_idx]:
            continue

        # calculate mixed uses
        # generate the reachable classes and their respective distances
        reachable_classes_trim, reachable_classes_dist_trim, _data_trim_to_full_idx_map = \
            data.aggregate_to_src_idx(src_idx,
                                      node_map,
                                      edge_map,
                                      data_map,
                                      max_dist,
                                      angular)

        classes_counts = np.full((d_n, unique_classes_n), 0)  # counts of each class type
        classes_nearest = np.full((d_n, unique_classes_n), np.inf)  # nearest of each class type

        for i in range(len(reachable_classes_trim)):
            cl_dist = reachable_classes_dist_trim[i]
            # some classes will be nan if beyond max threshold distance - so check for infinity
            if np.isinf(cl_dist):
                continue
            cl = int(reachable_classes_trim[i])

            for d_idx in range(len(distances)):

                d = distances[d_idx]
                b = betas[d_idx]

                # increment class counts at respective distances
                # TODO - not all classes though...
                # since the classes are encoded to ints, you can just use the class as an index
                if cl_dist < d:
                    classes_counts[d_idx][cl] += 1
                    # if distance is nearer, update
                    if cl_dist < classes_nearest[d_idx][cl]:
                        classes_nearest[d_idx][cl] = cl_dist

                    # if within distance, and if in accessibility keys, then aggregate accessibility
                    for ac_idx in range(len(accessibility_keys)):
                        if accessibility_keys[ac_idx] == cl:
                            accessibility_data[ac_idx][d_idx][src_idx] += 1
                            accessibility_data_wt[ac_idx][d_idx][src_idx] += np.exp(b * cl_dist)

        # iterate the distances and betas
        for d_idx in range(len(distances)):
            b = betas[d_idx]
            cl_counts = classes_counts[d_idx]
            cl_nearest = classes_nearest[d_idx]

            # compute mixed uses
            for mu_idx in mixed_use_keys:
                # the hill indices require an extra data dimension for various qs
                if mu_idx < 4:
                    for q_idx in range(len(qs)):
                        q = qs[q_idx]
                        mixed_use_hill_data[mu_idx][q_idx][d_idx][src_idx] = \
                            mixed_use_metrics(mu_idx,
                                              cl_counts,
                                              cl_nearest,
                                              b,
                                              q,
                                              cl_disparity_wt_matrix)
                # otherwise store in first dimension
                else:
                    # offset index for data structure
                    data_idx = mu_idx - 4
                    # no qs
                    mixed_use_other_data[data_idx][d_idx][src_idx] = \
                        mixed_use_metrics(mu_idx,
                                          cl_counts,
                                          cl_nearest,
                                          np.nan,  # no beta necessary
                                          np.nan,  # no q necessary
                                          cl_disparity_wt_matrix)

    return mixed_use_hill_data, mixed_use_other_data, accessibility_data, accessibility_data_wt
