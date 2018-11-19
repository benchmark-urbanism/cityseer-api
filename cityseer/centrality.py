'''
Centrality methods
'''
import logging
from typing import Union, Tuple, Any
import numpy as np
from . import networks


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def distance_from_beta(beta:Union[float, list, np.ndarray], min_threshold_wt:float=0.01831563888873418) -> Tuple[np.ndarray, float]:

    # cast to list form
    if isinstance(beta, (int, float)):
        beta = [beta]

    # check that the betas do not have leading negatives
    for b in beta:
        if b < 0:
            raise ValueError('Please provide the beta/s without the leading negative.')

    # cast to numpy
    beta = np.array(beta)

    # deduce the effective distance thresholds
    return np.log(min_threshold_wt) / -beta, min_threshold_wt


def compute_centrality(node_map:np.ndarray, edge_map:np.ndarray, distances:list, close_metrics:list=None,
        between_metrics:list=None, min_threshold_wt:float=0.01831563888873418, angular:bool=False) -> Tuple[Any, ...]:

    if node_map.shape[1] != 5:
        raise ValueError('The node map must have a dimensionality of nx5, consisting of x, y, live, link idx, and weight parameters.')

    if edge_map.shape[1] != 4:
        raise ValueError('The link map must have a dimensionality of nx4, consisting of start, end, distance, and impedance parameters.')

    if distances == []:
        raise ValueError('A list of local centrality distance thresholds is required.')

    if isinstance(distances, (int, float)):
        distances = [distances]

    if not isinstance(distances, (list, np.ndarray)):
        raise ValueError('Please provide a distance or an array of distances.')

    betas = []
    for d in distances:
        betas.append(np.log(min_threshold_wt) / d)

    if not close_metrics and not between_metrics:
        raise ValueError(f'Neither closeness nor betweenness metrics specified, please specify at least one metric to compute.')

    closeness_options = ['node_density', 'farness_impedance', 'farness_distance', 'harmonic', 'improved', 'gravity', 'cycles']
    closeness_map = []
    if close_metrics:
        for cl in close_metrics:
            if cl not in closeness_options:
                raise ValueError(f'Invalid closeness option: {cl}. Must be one of {", ".join(closeness_options)}.')
            closeness_map.append(closeness_options.index(cl))
    # improved closeness is extrapolated from node density and farness_distance, so these may have to be added regardless:
    # assign to new variable so as to keep closeness_map pure for later use in unpacking the results
    closeness_map_extra = closeness_map
    if close_metrics and 'improved' in close_metrics:
        closeness_map_extra = list(set(closeness_map_extra + [
            closeness_options.index('node_density'),
            closeness_options.index('farness_distance')]))

    betweenness_options = ['betweenness', 'betweenness_gravity']
    betweenness_map = []
    if between_metrics:
        for bt in between_metrics:
            if bt not in betweenness_options:
                raise ValueError(f'Invalid betweenness option: {bt}. Must be one of {", ".join(betweenness_options)}.')
            betweenness_map.append(betweenness_options.index(bt))

    closeness_data, betweenness_data = networks.network_centralities(node_map, edge_map, np.array(distances),
                                    np.array(betas), np.array(closeness_map_extra), np.array(betweenness_map), angular)

    # return statement tuple unpacking supported from Python 3.8... till then, unpack first
    return_data = *closeness_data[closeness_map], *betweenness_data[betweenness_map], betas
    return return_data


def compute_betweenness(node_map:np.ndarray, edge_map:np.ndarray, distances:list) -> Tuple[Any, ...]:
    return compute_centrality(node_map, edge_map, distances, between_metrics=['betweenness'])


def compute_harmonic_closeness(node_map:np.ndarray, edge_map:np.ndarray, distances:list) -> Tuple[Any, ...]:
    return compute_centrality(node_map, edge_map, distances, close_metrics=['harmonic'])


# TODO: add harmonic closeness with automatic beta


# TODO: add weighted betweenness with automatic beta


def compute_angular_betweenness(node_map:np.ndarray, edge_map:np.ndarray, distances:list) -> Tuple[Any, ...]:
    return compute_centrality(node_map, edge_map, distances, between_metrics=['betweenness'], angular=True)


def compute_angular_harmonic_closeness(node_map:np.ndarray, edge_map:np.ndarray, distances:list) -> Tuple[Any, ...]:
    return compute_centrality(node_map, edge_map, distances, close_metrics=['harmonic'], angular=True)


# TODO: add mixed-uses algo
'''
@njit
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
'''