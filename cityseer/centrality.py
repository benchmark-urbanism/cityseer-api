'''

'''

import logging
from typing import Union, Tuple
import utm
from shapely import geometry, ops
import networkx as nx
import numpy as np
from numba.pycc import CC
from numba import njit
from . import networks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

cc = CC('centrality')


def distance_from_beta(beta:Union[float, list, np.ndarray], min_threshold_wt:float=0.01831563888873418) \
        -> Tuple[np.ndarray, float]:

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
    return np.log(min_threshold_wt) / -beta, min_threshold_wt


def graph_from_networkx(network_x_graph:nx.Graph, wgs84_coords:bool=False, decompose:int=None, geom:geometry.Polygon=None) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Strategic decisions because of too many edge cases:
    - decided to not discard disconnected components to avoid unintended consequences
    - no internal simplification - use prior methods or tools to clean or simplify the graph before calling this method
    '''

    # check that it is an undirected graph
    if not isinstance(network_x_graph, nx.Graph):
        raise ValueError('This method requires an undirected networkx graph')

    # copy the graph to avoid modifying the original
    g_copy = network_x_graph.copy()

    # if necessary, convert from WGS84
    for n, d in g_copy.nodes(data=True):
        x = d['x']
        y = d['y']
        # in case lng lat accidentally passed-in without WGS flag
        if x <= 90 and not wgs84_coords:
            raise ValueError(f'Coordinate error, if using lng, lat coordinates, set the wgs84_coords param to True')
        # if the coords are WGS84, then convert to local UTM
        if wgs84_coords:
            # remember - accepts and returns in y, x order
            y, x = utm.from_latlon(y, x)[:2]
        # round to 1cm
        g_copy.node[n]['x'] = np.round(x, 2)
        g_copy.node[n]['y'] = np.round(y, 2)

    # process the vertices and generate the lengths if not present
    # note -> write to a duplicated graph to avoid in-place errors
    g_dup = g_copy.copy()
    for s, e, d in g_copy.edges(data=True):
        # get start coords
        s_x = g_copy.node[s]['x']
        s_y = g_copy.node[s]['y']
        # get end coords
        e_x = g_copy.node[e]['x']
        e_y = g_copy.node[e]['y']
        # generate the geometry
        ls = geometry.LineString([[s_x, s_y], [e_x, e_y]])
        # write length to g_dup edge if no value already present
        if 'length' in g_dup[s][e]:
            # check that weight is positive
            if not g_dup[s][e]['length'] >= 0:
                raise ValueError('Only positive lengths can be passed to this method')
        else:
            g_dup[s][e]['length'] = ls.length
        # add weight attribute if not present, and set to length
        if 'weight' in g_dup[s][e]:
            # check that weight is positive
            if not g_dup[s][e]['weight'] >= 0:
                raise ValueError('Only positive weights can be passed to this method')
        else:
            # NB - set from length attribute and not geom in case input provided
            g_dup[s][e]['weight'] = g_dup[s][e]['length']
        # continue if not decomposing
        if not decompose:
            continue
        # see how many segments are necessary so as not to exceed decomposition max distance
        # note that a length less than the decompose threshold will result in a single 'sub'-string
        l = g_dup[s][e]['length']
        n = np.ceil(l / decompose)
        # find the length and weight subdivisions
        sub_l = l / n
        w = g_dup[s][e]['weight']
        sub_w = w / n
        # since decomposing, remove the prior edge... but only after properties have been read
        g_dup.remove_edge(s, e)
        # then add the new sub-edge/s
        d_step = 0
        prior_node_id = s
        sub_node_counter = 0
        # everything inside this loop is a new node - i.e. this loop is effectively skipped if n = 1
        # note, using actual length attribute / n for new length, as provided lengths may not match crow-flies distance
        for i in range(int(n) - 1):
            # create the new node ID
            new_node_id = f'{s}_{sub_node_counter}_{e}'
            sub_node_counter += 1
            # create the split linestring geom for measuring the new length
            s_g = ops.substring(ls, d_step, d_step + l/n)
            # get the x, y of the new end node
            x = s_g.coords.xy[0][-1]
            y = s_g.coords.xy[1][-1]
            # add the new node and edge
            g_dup.add_node(new_node_id, x=x, y=y)
            g_dup.add_edge(prior_node_id, new_node_id, length=sub_l, weight=sub_w)
            # increment the step and node id
            prior_node_id = new_node_id
            d_step += l / n
        # set the last link manually to avoid rounding errors at end of linestring
        # the nodes already exist, so just add link
        g_dup.add_edge(prior_node_id, e, length=sub_l, weight=sub_w)

    # convert the nodes to sequential - this permits implicit indices with benefits to speed and structure
    g_dup = nx.convert_node_labels_to_integers(g_dup, 0)

    # set the live nodes and sum degrees
    total_out_degrees = 0
    for n, d in g_dup.nodes(data=True):
        total_out_degrees += nx.degree(g_dup, n)
        live = True
        if geom and not geom.contains(geometry.Point(d['x'], d['y'])):
            live = False
        g_dup.node[n]['live'] = live

    # prepare the node and link maps
    n = g_dup.number_of_nodes()
    node_map = np.full((n, 4), 0.0)  # float - for consistency
    edge_map = np.full((total_out_degrees, 4), 0.0)  # float - allows for nan and inf
    edge_idx = 0
    # populate the nodes
    for n, d in g_dup.nodes(data=True):
        idx = int(n)
        node_map[idx][0] = d['x']
        node_map[idx][1] = d['y']
        node_map[idx][2] = d['live']
        node_map[idx][3] = edge_idx
        # follow all out links and add these to the edge_map
        # this happens for both directions
        for nb in g_dup.neighbors(n):
            # start node
            edge_map[edge_idx][0] = idx
            # end node
            edge_map[edge_idx][1] = nb
            # length
            edge_map[edge_idx][2] = g_dup[idx][nb]['length']
            # weight
            edge_map[edge_idx][3] = g_dup[idx][nb]['weight']
            # increment the link_idx
            edge_idx += 1

    assert len(node_map) == g_dup.number_of_nodes()
    assert len(edge_map) == g_dup.number_of_edges() * 2

    return node_map, edge_map


# TODO: consider adding primary to dual networkx converter? - too many edge cases and requires geoms...?


# TODO: have a look at autogenerating signatures? ints vs. floats etc? or are tests sufficient?
def centrality(node_map, edge_map, distances, min_threshold_wt=0.01831563888873418):

    if node_map.shape[1] != 4:
        raise ValueError('The node map must have a dimensionality of nx4, consisting of x, y, live, and link idx parameters')

    if edge_map.shape[1] != 4:
        raise ValueError('The link map must have a dimensionality of nx4, consisting of start, end, distance, and weight parameters')

    if isinstance(distances, (int, float)):
        distances = [distances]

    if not isinstance(distances, (list, np.ndarray)):
        raise ValueError('Please provide a distance or an array of distances')

    betas = []
    for d in distances:
        betas.append(np.log(min_threshold_wt) / d)

    node_density, imp_closeness, harm_closeness, gravity, betweenness, betweenness_wt = \
        networks.compute_centrality(node_map, edge_map, np.array(distances), np.array(betas))

    return node_density, imp_closeness, harm_closeness, gravity, betweenness, betweenness_wt, betas


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