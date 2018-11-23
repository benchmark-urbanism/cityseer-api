import numpy as np
from numba.pycc import CC
from numba import njit


cc = CC('diversity')


@cc.export('hill_diversity', '(uint64[:], float64)')
@njit
def hill_diversity(class_counts:np.ndarray, q:float) -> float:
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
            p = a / N  # the probability of this class
            H += p * np.log(p)  # sum entropy
        return np.exp(-H)  # return exponent of entropy
    # otherwise use the usual form of Hill numbers
    else:
        D = 0
        for a in class_counts:
            p = a / N  # the probability of this class
            D += p ** q  # sum
        return D ** (1 / (1 - q))  # return as equivalent species


@cc.export('hill_diversity_branch_generic', '(uint64[:], float64[:], float64)')
@njit
def hill_diversity_branch_generic(class_counts:np.ndarray, class_weights:np.ndarray, q:float) -> float:
    '''
    Based on unified framework for species diversity in Chao, Chiu, Jost 2014
    See table on page 308 and surrounding text

    In this case the presumption is that you supply branch weights in the form of negative exponential distance weights
    i.e. pedestrian walking distance decay which weights more distant locations more weakly than nearer locations
    This means that the walking distance to a landuse impacts how strongly it contributes to diversity

    The weighting is based on the nearest of each landuse
    This is debatably most relevant to q=0
    '''

    if len(class_counts) != len(class_weights):
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
        wt = class_weights[i]
        a = class_counts[i] / N
        T += wt * a

    # hill number defined in the limit as the exponential of information entropy
    if q == 1:
        PD_lim = 0  # using the same variable name as non limit version causes errors for parallel
        # get branch lengths and class abundances
        for i in range(len(class_counts)):
            wt = class_weights[i]
            a = class_counts[i] / N
            PD_lim += wt * a / T * np.log(a / T)  # sum entropy
        # return exponent of entropy
        PD_lim = np.exp(-PD_lim)
        return PD_lim #/ T
    # otherwise use the usual form of Hill numbers
    else:
        PD = 0
        # get branch lengths and class abundances
        for i in range(len(class_counts)):
            wt = class_weights[i]
            a = class_counts[i] / N
            PD += wt * (a / T)**q  # sum
        # once summed, apply q
        PD = PD ** (1 / (1 - q))
        return PD #/ T


@cc.export('hill_diversity_branch_distance_wt', '(uint64[:], float64[:], float64, float64)')
@njit
def hill_diversity_branch_distance_wt(class_counts:np.array, class_distances:np.array, beta:float, q:float) -> float:

    if beta < 0:
        raise ValueError('Please provide the beta/s without the leading negative.')

    class_weights = np.exp(class_distances * -beta)

    return hill_diversity_branch_generic(class_counts, class_weights, q)


@cc.export('hill_diversity_pairwise_generic', '(uint64[:], float64[:,:], float64)')
@njit
def hill_diversity_pairwise_generic(class_counts:np.ndarray, wt_matrix:np.ndarray, q:float) -> float:
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
    if N < 2:
        return 0

    # calculate Q
    Q = 0
    for i in range(len(class_counts)):
        a_i = class_counts[i] / N
        for j in range(len(class_counts)):
            # only need to examine the pair if j < i, otherwise double-counting
            if j >= i:
                break
            a_j = class_counts[j] / N
            wt = wt_matrix[i][j]
            # pairwise distances
            Q += wt * a_i * a_j

    # if in the limit, use exponential
    if q == 1:
        FD_lim = 0  # using the same variable name as non limit version causes errors for parallel
        for i in range(len(class_counts)):
            a_i = class_counts[i] / N
            for j in range(len(class_counts)):
                # only need to examine the pair if j < i, otherwise double-counting
                if j >= i:
                    break
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
            a_i = class_counts[i] / N
            for j in range(len(class_counts)):
                # only need to examine the pair if j < i, otherwise double-counting
                if j >= i:
                    break
                a_j = class_counts[j] / N
                # pairwise distances
                wt = wt_matrix[i][j]
                FD += wt * (a_i * a_j / Q) ** q  # sum
        FD = FD ** (1 / (1 - q))
        return FD ** (1 / 2)  # (FD / Q) ** (1 / 2)


@cc.export('pairwise_distance_matrix', '(float64[:], float64)')
@njit
def pairwise_distance_matrix(distances:np.ndarray, beta:float) -> np.ndarray:

    if beta < 0:
        raise ValueError('Please provide the beta/s without the leading negative.')

    # prepare the weights matrix
    wt_matrix = np.full((len(distances), len(distances)), np.inf)
    for i_idx in range(len(distances)):
        for j_idx in range(len(distances)):
            # no need to repeat where j > i
            if j_idx > i_idx:
                continue
            # in this case i == j distances have to be calculated
            else:
                # write in both directions - though not technically necessary
                w = np.exp((distances[i_idx] + distances[j_idx]) * -beta)
                wt_matrix[i_idx][j_idx] = w
                wt_matrix[j_idx][i_idx] = w
    return wt_matrix


@cc.export('hill_diversity_pairwise_distance_wt', '(uint64[:], float64[:], float64, float64)')
@njit
def hill_diversity_pairwise_distance_wt(class_counts:np.array, class_distances:np.array, beta:float, q:float) -> float:

    if len(class_counts) != len(class_distances):
        raise ValueError('Mismatching number of unique class counts and class distances.')

    wt_matrix = pairwise_distance_matrix(class_distances, beta)

    return hill_diversity_pairwise_generic(class_counts, wt_matrix, q)


@cc.export('pairwise_disparity_matrix', '(uint64[:,:], float64[:])')
@njit
def pairwise_disparity_matrix(class_tiers:np.ndarray, class_weights:np.ndarray) -> np.ndarray:

    if class_tiers.shape[1] != len(class_weights):
        raise ValueError('The number of weights must correspond to the number of tiers for nodes i and j.')

    # prepare the weights matrix
    wt_matrix = np.full((len(class_tiers), len(class_tiers)), np.inf)
    for i_idx in range(len(class_tiers)):
        for j_idx in range(len(class_tiers)):
            # no need to repeat where j > i
            if j_idx > i_idx:
                continue
            elif j_idx == i_idx:
                wt_matrix[i_idx][j_idx] = 0  # because disparity is 0
            else:
                w = np.nan
                for t_idx, (i, j) in enumerate(zip(class_tiers[i_idx], class_tiers[j_idx])):
                    if i == j:
                        w = class_weights[t_idx]
                    else:
                        break
                if np.isnan(w):
                    raise AttributeError('Failed convergence in species tiers. Check that all tiers converge at the first level.')
                # write in both directions - though not technically necessary
                wt_matrix[i_idx][j_idx] = w
                wt_matrix[j_idx][i_idx] = w
    return wt_matrix


@cc.export('hill_diversity_pairwise_disparity_wt', '(uint64[:], uint64[:,:], float64[:], float64)')
@njit
def hill_diversity_pairwise_disparity_wt(class_counts:np.array, class_tiers:np.array, class_weights:np.array, q:float) -> float:

    if len(class_counts) != len(class_tiers):
        raise ValueError('Mismatching number of unique class counts and respective class taxonomy tiers.')

    wt_matrix = pairwise_disparity_matrix(class_tiers, class_weights)

    return hill_diversity_pairwise_generic(class_counts, wt_matrix, q)


# explicit return required, otherwise numba throws:
# TypeError: invalid signature: 'str' instance not allowed
@cc.export('gini_simpson_diversity', 'float64(uint64[:])')
@njit
def gini_simpson_diversity(class_counts:np.ndarray) -> float:
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
        G += c / N * (c - 1) / (N - 1)
    return 1 - G


@cc.export('shannon_diversity', 'float64(uint64[:])')
@njit
def shannon_diversity(class_counts:np.ndarray) -> float:
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
        p = a / N  # the probability of this class
        H += p * np.log(p)  # sum entropy
    return -H  # remember negative


@cc.export('raos_quadratic_diversity', '(uint64[:], float64[:,:], float64, float64)')
@njit
def raos_quadratic_diversity(class_counts:np.ndarray, wt_matrix:np.ndarray, alpha:float=1, beta:float=1) -> float:
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
    # catch potential division by zero situations
    N = class_counts.sum()
    if N < 2:
        return 0

    R = 0  # variable for additive calculations of distance * p1 * p2
    for i in range(len(class_counts)):
        # parallelise only inner loop
        for j in range(len(class_counts)):
            # only need to examine the pair if j < i, otherwise double-counting
            if j >= i:
                break
            p_i = class_counts[i] / N  # place here to catch division by zero for single element
            p_j = class_counts[j] / (N - 1)  # bias adjusted
            # calculate 3rd level disparity
            wt = wt_matrix[i][j]
            R += wt**alpha * (p_i * p_j)**beta
    return R


#@njit
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


#@njit
def compute_mixed_uses(node_map, edge_map, distances, betas, overlay):
    max_dist = max(distances)

    # establish the number of nodes
    n = node_map.shape[1]

    # create the distance map
    d_map = np.full((len(distances), 2), np.nan)
    for i, d in enumerate(distances):
        d_map[i] = [d, ]

    # prepare data arrays
    closeness = np.full((4, n), 0.0)
    gravity = np.full((4, n), 0.0)
    betweenness_wt = np.full((4, n), 0.0)
    betweenness_wt = np.full((4, n), 0.0)

    # prepare data arrays
    gravity = np.zeros((4, total_count))
    betweenness_wt = np.zeros((4, total_count))
    mixed_uses_wt = np.zeros((4, total_count))
    pois = np.zeros((40, total_count))

    beta_100 = -0.04
    beta_200 = -0.02
    beta_400 = -0.01
    beta_800 = -0.005

    data_assign_map, data_assign_dist = networks.assign_accessibility_data(netw_x_arr, netw_y_arr, data_x_arr,
                                                                           data_y_arr, max_dist)

    # iterate through each vert and calculate the shortest path tree
    for netw_src_idx in range(total_count):

        # if netw_src_idx % 1000 == 0:
        #    print('...progress')
        #    print(round(netw_src_idx / total_count * 100, 2))

        # only compute for nodes in current city
        if not hot_node[netw_src_idx]:
            continue

        netw_src_idx_trim, netw_trim_count, netw_idx_map_trim_to_full, nbs_trim, lens_trim = \
            networks.graph_window(netw_src_idx, max_dist, netw_x_arr, netw_y_arr, nbs, lens)

        # use np.inf for max distance for data POI mapping, which uses an overshoot and backtracking workflow
        # not a huge penalty because graph is already windowed per above
        netw_dist_map_trim, netw_pred_map_trim = networks.shortest_path_tree(nbs_trim, lens_trim, netw_src_idx_trim,
                                                                             netw_trim_count, np.inf)

        # calculate mixed uses
        # generate the reachable classes and their respective distances
        reachable_classes, reachable_classes_dist, data_trim_to_full_idx_map = networks.accessibility_agg(netw_src_idx,
                                                                                                          max_dist,
                                                                                                          netw_dist_map_trim,
                                                                                                          netw_pred_map_trim,
                                                                                                          netw_idx_map_trim_to_full,
                                                                                                          netw_x_arr,
                                                                                                          netw_y_arr,
                                                                                                          data_classes,
                                                                                                          data_x_arr,
                                                                                                          data_y_arr,
                                                                                                          data_assign_map,
                                                                                                          data_assign_dist)

        # get unique classes, their counts, and nearest - use the default max distance of 1600m
        classes_unique, classes_counts, classes_nearest = mixed_uses.deduce_unique_species(reachable_classes,
                                                                                           reachable_classes_dist)

        # compute mixed uses
        mixed_uses_wt[0][netw_src_idx] = mixed_uses.hill_diversity_functional(classes_counts,
                                                                              np.exp(beta_100 * classes_nearest), 0)
        mixed_uses_wt[1][netw_src_idx] = mixed_uses.hill_diversity_functional(classes_counts,
                                                                              np.exp(beta_200 * classes_nearest), 0)
        mixed_uses_wt[2][netw_src_idx] = mixed_uses.hill_diversity_functional(classes_counts,
                                                                              np.exp(beta_400 * classes_nearest), 0)
        mixed_uses_wt[3][netw_src_idx] = mixed_uses.hill_diversity_functional(classes_counts,
                                                                              np.exp(beta_800 * classes_nearest), 0)

        # compute accessibilities
        # reachable is same as for mixed uses, use the data_trim_to_full_idx_map array as an index
        poi_idx = data_trim_to_full_idx_map[np.isfinite(data_trim_to_full_idx_map)]
        # throwing errors so setting int indices with loop
        # some data_trim_to_full_idx_map values are np.nan, hence not int from get-go
        poi_idx_int = np.full(len(poi_idx), 0)
        for i, idx in enumerate(poi_idx):
            poi_idx_int[i] = np.int(idx)
        # calculate accessibilities
        pois[:, netw_src_idx] = accessibility.accessibility_osm_poi(poi_cats[poi_idx_int], reachable_classes_dist, 40,
                                                                    beta_800)

        # use corresponding indices for reachable verts
        ind = np.where(np.isfinite(netw_dist_map_trim))[0]
        for trim_to_idx in ind:

            # skip self node
            if trim_to_idx == netw_src_idx_trim:
                continue

            dist_m = netw_dist_map_trim[trim_to_idx]

            # some crow-flies max distance nodes won't be reached within max distance threshold over the network
            if np.isinf(dist_m):
                continue

            # remember that the shortest_path_tree is set to np.inf for mixed-uses purposes, so check here for distance
            if dist_m > max_dist:
                continue

            # calculate gravity and betweenness
            # the strength of the weight is based on the start and end vertices, not the intermediate locations
            netw_wt_100 = np.exp(beta_100 * dist_m)
            netw_wt_200 = np.exp(beta_200 * dist_m)
            netw_wt_400 = np.exp(beta_400 * dist_m)
            netw_wt_800 = np.exp(beta_800 * dist_m)

            # gravity -> an accessibility measure, or effectively a closeness consisting of inverse distance weighted node count
            gravity[0][netw_src_idx] += netw_wt_100
            gravity[1][netw_src_idx] += netw_wt_200
            gravity[2][netw_src_idx] += netw_wt_400
            gravity[3][netw_src_idx] += netw_wt_800

            # betweenness - only counting truly between vertices, not starting and ending verts
            intermediary_idx_trim = np.int(netw_pred_map_trim[trim_to_idx])
            intermediary_idx_mapped = np.int(netw_idx_map_trim_to_full[intermediary_idx_trim])  # cast to int
            # only counting betweenness in one 'direction' since the graph is symmetrical (non-directed)
            while True:
                # break out of while loop if the intermediary has reached the source node
                if intermediary_idx_trim == netw_src_idx_trim:
                    break

                # weighted variants - summed at all distances
                betweenness_wt[0][intermediary_idx_mapped] += netw_wt_100
                betweenness_wt[1][intermediary_idx_mapped] += netw_wt_200
                betweenness_wt[2][intermediary_idx_mapped] += netw_wt_400
                betweenness_wt[3][intermediary_idx_mapped] += netw_wt_800

                # unlike the dist_map the pred_map contains all vertices, so no offset required
                intermediary_idx_trim = np.int(netw_pred_map_trim[intermediary_idx_trim])
                intermediary_idx_mapped = np.int(netw_idx_map_trim_to_full[intermediary_idx_trim])  # cast to int

    return gravity, betweenness_wt, mixed_uses_wt, pois