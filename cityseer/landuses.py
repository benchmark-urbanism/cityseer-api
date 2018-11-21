import logging
from typing import Tuple
import numpy as np
from numba.pycc import CC
from numba import njit


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


cc = CC('landuses')


'''
@njit
def accessibility_agg(netw_src_idx, max_dist, netw_dist_map_trim, netw_pred_map_trim, netw_idx_map_trim_to_full,
                      netw_x_arr, netw_y_arr, data_classes, data_x_arr, data_y_arr, data_assign_map, data_assign_dist):
    # window the data
    source_x = netw_x_arr[netw_src_idx]
    source_y = netw_y_arr[netw_src_idx]
    data_trim_count, data_trim_to_full_idx_map, data_full_to_trim_idx_map = \
        crow_flies(source_x, source_y, max_dist, data_x_arr, data_y_arr)

    # iterate the distance trimmed data point
    reachable_classes_trim = np.full(data_trim_count, np.nan)
    reachable_classes_dist_trim = np.full(data_trim_count, np.inf)
    for i, original_data_idx in enumerate(data_trim_to_full_idx_map):
        # find the network node that it was assigned to
        assigned_network_idx = data_assign_map[np.int(original_data_idx)]
        # now iterate the trimmed network distances
        for j, (original_network_idx, dist) in enumerate(zip(netw_idx_map_trim_to_full, netw_dist_map_trim)):
            # no need to continue if it doesn't match the data point's assigned network node idx
            if original_network_idx != assigned_network_idx:
                continue
            # check both current and previous nodes for valid distances before continuing
            # first calculate the distance to the current node
            # in many cases, dist_calc will be np.inf, though still works for the logic below
            dist_calc = dist + data_assign_dist[np.int(original_data_idx)]
            # get the predecessor node so that distance to prior node can be compared
            # in some cases this is closer and therefore use the closer corner, especially for full networks
            prev_netw_node_trim_idx = netw_pred_map_trim[j]
            # some will be unreachable causing dist = np.inf or prev_netw_node_trim_idx = np.nan
            if not np.isfinite(prev_netw_node_trim_idx):
                # in this cases, just check whether dist_calc is less than max and continue
                if dist_calc <= max_dist:
                    reachable_classes_dist_trim[i] = dist_calc
                    reachable_classes_trim[i] = data_classes[np.int(original_data_idx)]
                continue
            # otherwise, go-ahead and calculate for the prior node
            prev_netw_node_full_idx = np.int(netw_idx_map_trim_to_full[np.int(prev_netw_node_trim_idx)])
            prev_dist_calc = netw_dist_map_trim[np.int(prev_netw_node_trim_idx)] + \
                             np.sqrt((netw_x_arr[prev_netw_node_full_idx] - data_x_arr[np.int(original_data_idx)]) ** 2
                                     + (netw_y_arr[prev_netw_node_full_idx] - data_y_arr[
                                 np.int(original_data_idx)]) ** 2)
            # use the shorter distance between the current and prior nodes
            # but only if less than the maximum distance
            if dist_calc < prev_dist_calc and dist_calc <= max_dist:
                reachable_classes_dist_trim[i] = dist_calc
                reachable_classes_trim[i] = data_classes[np.int(original_data_idx)]
            elif prev_dist_calc < dist_calc and prev_dist_calc <= max_dist:
                reachable_classes_dist_trim[i] = prev_dist_calc
                reachable_classes_trim[i] = data_classes[np.int(original_data_idx)]

    # note that some entries will be nan values if the max distance was exceeded
    # return the trim to full idx map as well in case other forms of data also need to be processed
    return reachable_classes_trim, reachable_classes_dist_trim, data_trim_to_full_idx_map



@njit
def accessibility_agg_angular(netw_src_idx, max_dist, netw_dist_map_a_m_trim, netw_pred_map_a_trim, netw_idx_map_trim_to_full, netw_x_arr, netw_y_arr, data_classes, data_x_arr, data_y_arr, data_assign_map, data_assign_dist):

    # window the data
    source_x = netw_x_arr[netw_src_idx]
    source_y = netw_y_arr[netw_src_idx]
    data_trim_count, data_trim_to_full_idx_map, data_full_to_trim_idx_map = \
        crow_flies(source_x, source_y, max_dist, data_x_arr, data_y_arr)

    # iterate the distance trimmed data point
    reachable_classes_trim = np.full(data_trim_count, np.nan)
    reachable_classes_dist_trim = np.full(data_trim_count, np.inf)
    for i, original_data_idx in enumerate(data_trim_to_full_idx_map):
        # find the network node that it was assigned to
        assigned_network_idx = data_assign_map[np.int(original_data_idx)]
        # now iterate the trimmed network distances
        # use the angular route (simplest paths) version of distances
        for j, (original_network_idx, dist) in enumerate(zip(netw_idx_map_trim_to_full, netw_dist_map_a_m_trim)):
            # no need to continue if it doesn't match the data point's assigned network node idx
            if original_network_idx != assigned_network_idx:
                continue
            # check both current and previous nodes for valid distances before continuing
            # first calculate the distance to the current node
            # in many cases, dist_calc will be np.inf, though still works for the logic below
            dist_calc = dist + data_assign_dist[np.int(original_data_idx)]
            # get the predecessor node so that distance to prior node can be compared
            # in some cases this is closer and therefore use the closer corner, especially for full networks
            prev_netw_node_trim_idx = netw_pred_map_a_trim[j]
            # some will be unreachable causing dist = np.inf or prev_netw_node_trim_idx = np.nan
            if not np.isfinite(prev_netw_node_trim_idx):
                # in this cases, just check whether dist_calc is less than max and continue
                if dist_calc <= max_dist:
                    reachable_classes_dist_trim[i] = dist_calc
                    reachable_classes_trim[i] = data_classes[np.int(original_data_idx)]
                continue
            # otherwise, go-ahead and calculate for the prior node
            prev_netw_node_full_idx = np.int(netw_idx_map_trim_to_full[np.int(prev_netw_node_trim_idx)])
            prev_dist_calc = netw_dist_map_a_m_trim[np.int(prev_netw_node_trim_idx)] + \
                             np.sqrt((netw_x_arr[prev_netw_node_full_idx] - data_x_arr[np.int(original_data_idx)]) ** 2
                                     + (netw_y_arr[prev_netw_node_full_idx] - data_y_arr[np.int(original_data_idx)]) ** 2)
            # use the shorter distance between the current and prior nodes
            # but only if less than the maximum distance
            if dist_calc < prev_dist_calc and dist_calc <= max_dist:
                reachable_classes_dist_trim[i] = dist_calc
                reachable_classes_trim[i] = data_classes[np.int(original_data_idx)]
            elif prev_dist_calc < dist_calc and prev_dist_calc <= max_dist:
                reachable_classes_dist_trim[i] = prev_dist_calc
                reachable_classes_trim[i] = data_classes[np.int(original_data_idx)]

    # note that some entries will be nan values if the max distance was exceeded
    return reachable_classes_trim, reachable_classes_dist_trim

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