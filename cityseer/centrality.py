'''
.. module:: centrality
   :synopsis: Cityseer centrality methods

.. moduleauthor:: Gareth Simons

'''

import logging
import numpy as np
from numba.pycc import CC
from numba import njit
from . import networks, mixed_uses, accessibility

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

cc = CC('centrality')


def custom_decay_betas(beta:(float, list, np.ndarray), threshold_weight:float=0.01831563888873418):
    '''
    A convenience function for mapping :math:`-\\beta` decay parameters to equivalent distance thresholds corresponding to a `threshold_weight` cutoff parameter.

    :param beta: The :math:`-\\beta` that you wish to convert to distance thresholds.
    :param threshold_weight: The :math:`w_{min}` threshold.
    :return: A numpy array of effective :math:`d_{max}` distances at the corresponding :math:`w_{min}` threshold.

    .. note:: There is no need to use this function unless you wish to provide your own :math:`-\\beta` or `threshold_weight` parameters to the :meth:`cityseer.centrality.compute_centrality` method.

    .. warning:: Remember to pass both the :math:`d_{max}` and :math:`w_{min}` to :meth:`cityseer.centrality.compute_centrality`

    The weighted variants of centrality, e.g. gravity or weighted betweenness, are computed using a negative exponential decay function of the form:

    .. math::
       weight = exp(-\\beta \cdot distance)

    The strength of the decay is controlled by the :math:`-\\beta` parameter, which reflects a decreasing willingness to walk correspondingly farther distances.
    For example, if :math:`-\\beta=0.005` represents a person's willingness to walk to a bus stop, then a location 100m distant would be weighted at 60% and a location 400m away would be weighted at 13.5%. After an initially rapid decrease, the weightings decay ever more gradually in perpetuity. At some point, it becomes futile to consider locations any farther away, and this is what is meant by a minimum weight threshold :math:`w_{min}` corresponding to a maximum distance threshold of :math:`d_{max}`.

    The :meth:`cityseer.centrality.compute_centrality` method computes the :math:`-\\beta` parameters automatically, using a default `threshold_weight` of :math:`w_{min}=0.01831563888873418`.

    .. math::

       \\beta = \\frac{log\\big(\\frac{1}{w_{min}}\\big)}{d_{max}}

    Therefore, :math:`-\\beta` weights corresponding to walking thresholds of 400m, 800m, and 1600m would give:

    .. table::
       :align: center

       ================= =================
        :math:`d_{max}`   :math:`-\\beta`
       ----------------- -----------------
              400m             -0.01
              800m             -0.005
              1600m            -0.0025
       ================= =================

    In reality, people may be more or less willing to walk based on the specific purpose of the trip and the pedestrian-friendliness of the urban context. Therefore, if you wish to override the defaults, or simply want to use a custom :math:`-\\beta` or a different :math:`w_{min}` threshold, then use this function to generate effective :math:`d_{max}` values, which can then be passed to :meth:`cityseer.centrality.compute_centrality`.

    '''

    # check for mistaken usage
    if round(threshold_weight, 3) == round(0.01831563888873418, 3):
        logger.warning('The current min_weight parameter matches the default, simply pass your distances directly to the centrality method.')

    # cast to list form
    if isinstance(beta, (int, float)):
        beta = [beta]

    # check that the betas do not have leading negatives
    for b in beta:
        if b < 0:
            raise ValueError('Please provide the beta/s without the leading negative')

    # cast to numpy
    beta = np.array(beta)

    # deduce the effective distance thresholds
    return np.log(threshold_weight) / -beta, threshold_weight


def centrality(node_map, link_map, distances):

    if node_map.shape[0] != 4:
        raise ValueError('The node map must have a dimensionality of 4, consisting of x, y, live, and link idx parameters')
    if link_map.shape[0] != 3:
        raise ValueError('The link map must have a dimensionality of 3, consisting of start, end, and distance parameters')
    if isinstance(distances, (int, float)):
        distances = [distances]
    if not isinstance(distances, (list, np.ndarray)):
        raise ValueError('Please provide a distance or an array of distances')




y = 0.01831563888873418
betas = []
for d in [50, 100, 150, 200, 300, 400, 600, 800, 1200, 1600]:
    betas.append(np.log(1/y)/d)


# NOTE -> didn't work with boolean so using unsigned int...
@cc.export('compute_centrality',
           'Tuple((Array(f8, 2, "C"), Array(f8, 2, "C"), Array(f8, 2, "C"), Array(f8, 2, "C")))'
           '(Array(f8, 2, "C"), Array(f8, 2, "C"), Array(f8, 1, "C"), Array(f8, 1, "C"), Array(boolean, 1, "C"), Array(f8, 1, "C"), Array(f8, 1, "C"), Array(f8, 1, "C"), Array(f8, 1, "C"))')
@njit
def compute_centrality(node_map, link_map, distances):
    '''

    :param node_map:
    :param link_map:
    :return:
    '''


    # used for calculating a corresponding beta value
    y = 0.01831563888873418

    max_dist = 800
    # establish the number of nodes
    n = node_map.shape[1]
    # create the distance map
    d_map = np.full((len(distances), 2), np.nan)
    for i, d in enumerate(distances):
        d_map[i] = [d, ]

    # prepare data arrays
    closeness = np.zeros((4, n))
    gravity = np.zeros((4, n))
    betweenness_wt = np.zeros((4, n))






    # prepare data arrays
    gravity = np.zeros((4, total_count))
    betweenness_wt = np.zeros((4, total_count))
    mixed_uses_wt = np.zeros((4, total_count))
    pois = np.zeros((40, total_count))

    beta_100 = -0.04
    beta_200 = -0.02
    beta_400 = -0.01
    beta_800 = -0.005

    data_assign_map, data_assign_dist = networks.assign_accessibility_data(netw_x_arr, netw_y_arr, data_x_arr, data_y_arr, max_dist)

    # iterate through each vert and calculate the shortest path tree
    for netw_src_idx in range(total_count):

        #if netw_src_idx % 1000 == 0:
        #    print('...progress')
        #    print(round(netw_src_idx / total_count * 100, 2))

        # only compute for nodes in current city
        if not hot_node[netw_src_idx]:
            continue

        netw_src_idx_trim, netw_trim_count, netw_idx_map_trim_to_full, nbs_trim, lens_trim = \
            networks.graph_window(netw_src_idx, max_dist, netw_x_arr, netw_y_arr, nbs, lens)

        # use np.inf for max distance for data POI mapping, which uses an overshoot and backtracking workflow
        # not a huge penalty because graph is already windowed per above
        netw_dist_map_trim, netw_pred_map_trim = networks.shortest_path_tree(nbs_trim, lens_trim, netw_src_idx_trim, netw_trim_count, np.inf)

        # calculate mixed uses
        # generate the reachable classes and their respective distances
        reachable_classes, reachable_classes_dist, data_trim_to_full_idx_map = networks.accessibility_agg(netw_src_idx, max_dist,
            netw_dist_map_trim, netw_pred_map_trim, netw_idx_map_trim_to_full, netw_x_arr, netw_y_arr, data_classes,
                data_x_arr, data_y_arr, data_assign_map, data_assign_dist)

        # get unique classes, their counts, and nearest - use the default max distance of 1600m
        classes_unique, classes_counts, classes_nearest = mixed_uses.deduce_unique_species(reachable_classes, reachable_classes_dist)

        # compute mixed uses
        mixed_uses_wt[0][netw_src_idx] = mixed_uses.hill_diversity_functional(classes_counts, np.exp(beta_100 * classes_nearest), 0)
        mixed_uses_wt[1][netw_src_idx] = mixed_uses.hill_diversity_functional(classes_counts, np.exp(beta_200 * classes_nearest), 0)
        mixed_uses_wt[2][netw_src_idx] = mixed_uses.hill_diversity_functional(classes_counts, np.exp(beta_400 * classes_nearest), 0)
        mixed_uses_wt[3][netw_src_idx] = mixed_uses.hill_diversity_functional(classes_counts, np.exp(beta_800 * classes_nearest), 0)

        # compute accessibilities
        # reachable is same as for mixed uses, use the data_trim_to_full_idx_map array as an index
        poi_idx = data_trim_to_full_idx_map[np.isfinite(data_trim_to_full_idx_map)]
        # throwing errors so setting int indices with loop
        # some data_trim_to_full_idx_map values are np.nan, hence not int from get-go
        poi_idx_int = np.full(len(poi_idx), 0)
        for i, idx in enumerate(poi_idx):
            poi_idx_int[i] = np.int(idx)
        # calculate accessibilities
        pois[:,netw_src_idx] = accessibility.accessibility_osm_poi(poi_cats[poi_idx_int], reachable_classes_dist, 40, beta_800)

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
