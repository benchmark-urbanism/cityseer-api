import numpy as np
from numba.pycc import CC
from numba import njit


cc = CC('mixed_uses')

# TODO: refactor and document

@cc.export('deduce_unique_species',
           'Tuple((Array(f8, 1, "C"), Array(i8, 1, "C"), Array(f8, 1, "C")))'
           '(Array(f8, 1, "C"), Array(f8, 1, "C"), f8)')
@njit
def deduce_unique_species(classes, distances, max_dist=1600):
    '''
    Sifts through the classes and returns unique classes, their counts, and the nearest distance to each respective type
    Only considers classes within the max distance
    Uses the closest specimen of any particular land-use as opposed to the average distance
    e.g. if a person where trying to connect to two nearest versions of type A and B
    '''
    # check that classes and distances are the same length
    if len(classes) != len(distances):
        raise ValueError('NOTE -> the classes array and the distances array need to be the same length')
    # not using np.unique because this doesn't take into account that certain classes exceed the max distance
    # prepare arrays for aggregation
    unique_count = 0
    classes_unique_raw = np.full(len(classes), np.nan)
    classes_counts_raw = np.full(len(classes), 0)  # int array
    classes_nearest_raw = np.full(len(classes), np.inf)  # coerce to float array
    # iterate all classes
    for i in range(len(classes)):
        d = distances[i]
        # check for valid entry - in case raw array is passed where unreachable verts have be skipped - i.e. np.inf
        if not np.isfinite(d):
            continue
        # first check that this instance doesn't exceed the maximum distance
        if d > max_dist:
            continue
        # if it doesn't, get the class
        c = classes[i]
        # iterate the unique classes
        # NB -> only parallelise the inner loop
        # if parallelising the outer loop it generates some very funky outputs.... beware
        for j in range(len(classes_unique_raw)):
            u_c = classes_unique_raw[j]
            # if already in the unique list, then increment the corresponding count
            if c == u_c:
                classes_counts_raw[j] += 1
                # check if the distance to this copy is closer than the prior least distance
                if d < classes_nearest_raw[j]:
                    classes_nearest_raw[j] = d
                break
            # if no match is encountered by the end of the list (i.e. np.nan), then add it:
            if np.isnan(u_c):
                classes_unique_raw[j] = c
                classes_counts_raw[j] += 1
                classes_nearest_raw[j] = d
                unique_count += 1
                break

    classes_unique = np.full(unique_count, np.nan)
    classes_counts = np.full(unique_count, 0)
    classes_nearest = np.full(unique_count, np.inf)
    for i in range(unique_count):
        classes_unique[i] = classes_unique_raw[i]
        classes_counts[i] = classes_counts_raw[i]
        classes_nearest[i] = classes_nearest_raw[i]

    return classes_unique, classes_counts, classes_nearest


@cc.export('distance_filter',
           'Tuple((Array(f8, 1, "C"), Array(i8, 1, "C")))'
           '(Array(f8, 1, "C"), Array(f8, 1, "C"), Array(f8, 1, "C"), f8)')
@njit
def distance_filter(cl_unique_arr, cl_counts_arr, cl_nearest_arr, max_dist):
    '''
    Use this function for preparing data for non-weighted forms of mixed-uses
    (Weighted forms simply apply different betas to the full data on the full max distance)
    :param cl_unique_arr:
    :param cl_counts_arr:
    :param cl_nearest_arr:
    :param max_dist:
    :return:
    '''
    # first figure out how many valid items there are
    c = 0
    for i, d in enumerate(cl_nearest_arr):
        if d <= max_dist:
            c += 1
    # create trimmed arrays
    cl_unique_arr_trim = np.full(c, np.nan)
    cl_counts_arr_trim = np.full(c, 0)
    # then copy over valid data
    # don't parallelise - would cause issues
    c = 0
    for i, d in enumerate(cl_nearest_arr):
        if d <= max_dist:
            cl_unique_arr_trim[c] = cl_unique_arr[i]
            cl_counts_arr_trim[c] = cl_counts_arr[i]
            c += 1

    return cl_unique_arr_trim, cl_counts_arr_trim


@cc.export('gini_simpson_index',
           'f8'
           '(Array(f8, 1, "C"))')
@njit
def gini_simpson_index(classes_counts):
    '''
    Gini-Simpson
    Gini transformed to 1 − λ
    Probability that two individuals picked at random do not represent the same species (Tuomisto)

    Ordinarily:
    D = 1 - sum(p**2) where p = Xi/N

    Bias corrected:
    D = 1 - sum(Xi/N * (Xi-1/N-1))
    '''
    N = classes_counts.sum()
    G = 0
    # catch potential division by zero situations
    if N < 2:
        return G
    # compute bias corrected gini-simpson
    for c in classes_counts:
        G += c / N * (c - 1) / (N - 1)
    return 1 - G


@cc.export('shannon_index',
           'f8'
           '(Array(f8, 1, "C"))')
@njit
def shannon_index(classes_counts):
    '''
    Entropy
    p = Xi/N
    S = -sum(p * log(p))
    Uncertainty of the species identity of an individual picked at random (Tuomisto)
    '''
    N = classes_counts.sum()
    H = 0
    # catch potential division by zero situations
    if N == 0:
        return H
    # compute
    for a in classes_counts:
        p = a / N  # the probability of this class
        H += p * np.log(p)  # sum entropy
    return -H  # remember negative


@cc.export('hill_diversity',
           'f8'
           '(Array(f8, 1, "C"), f8)')
@njit
def hill_diversity(classes_counts, q):
    '''
    Hill numbers - express actual diversity as opposed e.g. to Gini-Simpson (probability) and Shannon (information)

    exponent at 1 results in undefined because of 1/0 - but limit exists as exp(entropy)
    Ssee "Entropy and diversity" by Lou Jost

    Exponent at 0 = variety - i.e. count of unique species
    Exponent at 1 = unity
    Exponent at 2 = diversity form of simpson index
    '''

    N = classes_counts.sum()
    # catch potential division by zero situations
    if N == 0:
        return 0
    # hill number defined in the limit as the exponential of information entropy
    if q == 1:
        H = 0
        for a in classes_counts:
            p = a / N  # the probability of this class
            H += p * np.log(p)  # sum entropy
        return np.exp(-H)  # return exponent of entropy
    # otherwise use the usual form of Hill numbers
    else:
        D = 0
        for a in classes_counts:
            p = a / N  # the probability of this class
            D += p ** q  # sum
        return D ** (1 / (1 - q))  # return as equivalent species


@cc.export('hill_diversity_phylogenetic',
           'f8'
           '(Array(f8, 1, "C"), Array(f8, 1, "C"), f8)')
@njit
def hill_diversity_phylogenetic(classes_counts, class_weights, q):
    '''
    Based on unified framework for species diversity in Chao, Chiu, Jost 2014
    See table on page 308

    In this case the presumption is that you supply branch weights in the form of negative exponential distance weights
    i.e. pedestrian walking distance decay which weights more distant locations more weakly than nearer locations
    This means that the average walking distance to a landuse impacts how strongly it contributes to diversity

    The weighting is based on the nearest of each landuse
    This is debatably most relevant to q=0
    '''

    # catch potential division by zero situations
    if classes_counts.sum() == 0:
        return 0

    # in this case you don't use the proportions, i.e. a[i] / N, where N is sum(a[j]) for all j
    # instead, divide by the branch length weighted abundances
    # i.e. a[i] / T, where T = sum(L[j] x a[j]) for all j
    T = 0
    for i in range(len(classes_counts)):
        weight = class_weights[i]
        a = classes_counts[i]
        T += weight * a

    # once you have T, you can proceed to do the summation
    # equation 6b on page 311 in Chao, Chiu, Jost 2014 doesn't seem to work as intended
    # instead, use branch-weighted equation in Table 1 on page 308

    # hill number defined in the limit as the exponential of information entropy
    if q == 1:
        PD_lim = 0  # using the same variable name as non limit version causes errors for parallel
        # get branch lengths and class abundances
        for i in range(len(classes_counts)):
            weight = class_weights[i]
            a = classes_counts[i]
            # note, paper shows log of abundance only...
            # but for this to work properly it has to be applied to the entire
            # weighted proportion, more in the spirit of unweighted hill numbers???
            PD_lim += weight * a / T * np.log(a / T)  # sum entropy
        PD_lim = np.exp(-PD_lim)  # return exponent of entropy
        return PD_lim #/ T  # the hill number transformation doesn't seem to work - normalises???
    # otherwise use the usual form of Hill numbers
    else:
        PD = 0
        # get branch lengths and class abundances
        for i in range(len(classes_counts)):
            weight = class_weights[i]
            a = classes_counts[i]
            # note, paper shows exponent on proportion...
            # but for this to work properly it has to be applied to the entire
            # weighted proportion, more in the spirit of unweighted hill numbers???
            PD += weight * (a / T)**q  # sum
        # once summed, apply q
        PD = PD ** (1 / (1 - q))
        return PD #/ T  # the hill number transformation doesn't seem to work - normalises???


@cc.export('hill_diversity_functional',
           'f8'
           '(Array(f8, 1, "C"), Array(f8, 1, "C"), f8)')
@njit
def hill_diversity_functional(classes_counts, class_weights, q):
    '''
    Based on unified framework for species diversity in Chao, Chiu, Jost 2014
    See table on page 308
    In this case the presumption is that you supply weights in the form of negative exponential distance weights
    i.e. pedestrian walking distance decay which weights more distant locations more weakly than nearer locations
    Functional diversity takes the pairwise form, thus distances are based on pairwise i to j distances via the node k
    This is different to the non-pairwise form of the phylogenetic version which simply takes singular distance k to i
    '''

    # catch potential division by zero situations
    if classes_counts.sum() == 0:
        return 0

    # calculate Q, the pairwise length weighted abundances,
    # i.e. a[i]a[j] / Q, where Q = sum(d[i][j] x a[i] x a[j]) for all i, j pairs
    Q = 0
    for i in range(len(classes_counts)):
        a_i = classes_counts[i]
        weight_i = class_weights[i]
        # parallelise only inner loop
        for j in range(len(classes_counts)):
            # only need to examine the pair if j < i, otherwise double-counting
            if j >= i:
                break
            a_j = classes_counts[j]
            weight_j = class_weights[j]
            # pairwise distances
            # use the multiplicative form - already in betas
            d_ij = weight_i * weight_j
            Q += d_ij * a_i * a_j

    # once you have Q, you can proceed to do the summation
    # equation 3 on page 4 in Chao, Chiu 2014 doesn't seem to work as intended
    # instead, use functional-distance-weighted equation in Table 1 on page 308
    if q == 1:
        FD_lim = 0  # using the same variable name as non limit version causes errors for parallel
        for i in range(len(classes_counts)):
            a_i = classes_counts[i]
            weight_i = class_weights[i]
            # parallelise only inner loop
            for j in range(len(classes_counts)):
                # only need to examine the pair if j < i, otherwise double-counting
                if j >= i:
                    break
                a_j = classes_counts[j]
                weight_j = class_weights[j]
                # pairwise distances
                d_ij = weight_i * weight_j
                FD_lim += d_ij * a_i * a_j / Q * np.log(a_i * a_j / Q)  # sum
        return np.exp(-0.5 * FD_lim)
    else:
        FD = 0
        for i in range(len(classes_counts)):
            a_i = classes_counts[i]
            weight_i = class_weights[i]
            # parallelise only inner loop
            for j in range(len(classes_counts)):
                # only need to examine the pair if j < i, otherwise double-counting
                if j >= i:
                    break
                a_j = classes_counts[j]
                weight_j = class_weights[j]
                # pairwise distances
                d_ij = weight_i * weight_j
                FD += d_ij * (a_i * a_j / Q) ** q  # sum
        # once summed, apply q
        return FD ** (1 / (2 * (1 - q)))
