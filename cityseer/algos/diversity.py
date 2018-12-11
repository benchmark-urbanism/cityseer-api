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


@njit
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


@njit
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
        raise ValueError(
            'Neither mixed-use nor accessibility keys specified, please specify at least one metric to compute.')

    if len(mixed_use_keys) != 0 and (mixed_use_keys.min() < 0 or mixed_use_keys.max() > 6):
        raise ValueError('Mixed-use keys out of range of 0:6.')

    if len(accessibility_keys) != 0 and (accessibility_keys.min() < 0):
        raise ValueError('Negative accessibility key encountered. Use positive keys corresponding to class encodings.')

    for i in range(len(mixed_use_keys)):
        for j in range(len(mixed_use_keys)):
            if j > i:
                i_key = mixed_use_keys[i]
                j_key = mixed_use_keys[j]
                if i_key == j_key:
                    raise ValueError('Duplicate mixed-use key.')

    for i in range(len(accessibility_keys)):
        for j in range(len(accessibility_keys)):
            if j > i:
                i_key = accessibility_keys[i]
                j_key = accessibility_keys[j]
                if i_key == j_key:
                    raise ValueError('Duplicate accessibility key.')

    for k in mixed_use_keys:
        if k < 4:
            if len(qs) == 0:
                raise ValueError('All hill diversity measures require that at least one value of q is specified.')
        if k == 3 or k == 6:
            if len(cl_disparity_wt_matrix) == 0:
                raise ValueError('Hill / Rao pairwise disparity measures require a class disparity weights matrix.')

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

    for src_idx in range(n):

        # numba no object mode can only handle basic printing
        if src_idx % 10000 == 0:
            print('...progress:', round(src_idx / n * 100, 2), '%')

        # only compute for live nodes
        if not netw_nodes_live[src_idx]:
            continue

        # generate the reachable classes and their respective distances
        reachable_classes_raw, reachable_classes_dist_raw, _data_raw_to_full_idx_map = \
            data.aggregate_to_src_idx(src_idx,
                                      node_map,
                                      edge_map,
                                      data_map,
                                      max_dist,
                                      angular)

        # counts of each class type (array length per all unique classes - not just those within radial max distance)
        classes_counts = np.full((d_n, unique_classes_n), 0)
        # nearest of each class type (likewise)
        classes_nearest = np.full((d_n, unique_classes_n), np.inf)
        # iterate the reachable classes and deduce reachable class counts and nearest corresponding distances
        for i in range(len(reachable_classes_raw)):
            # some classes will be nan if beyond max threshold distance - so check for infinity
            cl_dist = reachable_classes_dist_raw[i]
            if np.isinf(cl_dist):
                continue
            # get the class category in integer form
            # remember that all class codes were encoded to sequential integers - these correspond to the array indices
            cl = int(reachable_classes_raw[i])
            # iterate the distance dimensions
            for d_idx in range(len(distances)):
                d = distances[d_idx]
                b = betas[d_idx]
                # increment class counts at respective distances - but only if the distance is less than the current d
                if cl_dist < d:
                    classes_counts[d_idx][cl] += 1
                    # if distance is nearer, update the nearest distance array too
                    if cl_dist < classes_nearest[d_idx][cl]:
                        classes_nearest[d_idx][cl] = cl_dist
                    # if within distance, and if in accessibility keys, then aggregate accessibility too
                    for ac_idx in range(len(accessibility_keys)):
                        ac_code = accessibility_keys[ac_idx]
                        if ac_code == cl:
                            accessibility_data[ac_idx][d_idx][src_idx] += 1
                            accessibility_data_wt[ac_idx][d_idx][src_idx] += np.exp(b * cl_dist)

        # now that the local class counts are aggregated, mixed uses can now be calculated
        # iterate the distances and betas
        for d_idx in range(len(distances)):
            b = betas[d_idx]
            cl_counts = classes_counts[d_idx]
            cl_nearest = classes_nearest[d_idx]

            q_idx_counter = 0  # keep track of number of indices for q metrics, helps figure out indices for non q
            # mu keys determine which metrics to compute
            # don't confuse with indices
            for mu_idx, mu_key in enumerate(mixed_use_keys):
                # the hill indices require an extra data dimension for various qs
                if mu_key < 4:
                    q_idx_counter += 1
                    for q_idx, q_key in enumerate(qs):

                        if mu_key == 0:
                            mixed_use_hill_data[mu_idx][q_idx][d_idx][src_idx] = \
                                hill_diversity(cl_counts, q_key)

                        elif mu_key == 1:
                            mixed_use_hill_data[mu_idx][q_idx][d_idx][src_idx] = \
                                hill_diversity_branch_distance_wt(cl_counts, cl_nearest, q=q_key, beta=b)

                        elif mu_key == 2:
                            mixed_use_hill_data[mu_idx][q_idx][d_idx][src_idx] = \
                                hill_diversity_pairwise_distance_wt(cl_counts, cl_nearest, q=q_key, beta=b)

                        # land-use classification disparity hill diversity
                        # the wt matrix can be used without mapping because cl_counts is based on all classes
                        # regardless of whether they are reachable
                        elif mu_key == 3:
                            mixed_use_hill_data[mu_idx][q_idx][d_idx][src_idx] = \
                                hill_diversity_pairwise_matrix_wt(cl_counts, wt_matrix=cl_disparity_wt_matrix, q=q_key)

                # otherwise store in first dimension
                else:
                    # offset index for data structure
                    data_idx = int(mu_idx - q_idx_counter)

                    if mu_key == 4:
                        mixed_use_other_data[data_idx][d_idx][src_idx] = \
                            shannon_diversity(cl_counts)

                    elif mu_key == 5:
                        mixed_use_other_data[data_idx][d_idx][src_idx] = \
                            gini_simpson_diversity(cl_counts)

                    elif mu_key == 6:
                        mixed_use_other_data[data_idx][d_idx][src_idx] = \
                            raos_quadratic_diversity(cl_counts, wt_matrix=cl_disparity_wt_matrix)

    print('...done')

    return mixed_use_hill_data, mixed_use_other_data, accessibility_data, accessibility_data_wt
