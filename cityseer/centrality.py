'''

'''

import logging
from typing import Union, Tuple, Any
import utm
from shapely import geometry, ops
import networkx as nx
from tqdm import tqdm
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


def networkX_wgs_to_utm(networkX_graph:nx.Graph) -> nx.Graph:

    if not isinstance(networkX_graph, nx.Graph):
        raise ValueError('This method requires an undirected networkX graph.')

    logger.info('Converting networkX graph from WGS to UTM.')
    g_copy = networkX_graph.copy()

    logger.info('Processing node x, y coordinates.')
    for n, d in tqdm(g_copy.nodes(data=True)):
        # x coordinate
        if 'x' not in d:
            raise ValueError(f'Encountered node missing "x" coordinate attribute at node {n}.')
        x = d['x']
        # y coordinate
        if 'y' not in d:
            raise ValueError(f'Encountered node missing "y" coordinate attribute at node {n}.')
        y = d['y']
        # check for unintentional use of conversion
        if x > 180 or y > 90:
            raise ValueError('x, y coordinates exceed WGS bounds. Please check your coordinate system.')
        # remember - accepts and returns in y, x order
        y, x = utm.from_latlon(y, x)[:2]
        # write back to graph
        g_copy.nodes[n]['x'] = x
        g_copy.nodes[n]['y'] = y

    # if line geom property provided, then convert as well
    logger.info('Processing edge geom coordinates, if present.')
    for s, e, d in tqdm(g_copy.edges(data=True)):
        # check if geom present - optional step
        if 'geom' in d:
            line_geom = d['geom']
            if line_geom.type != 'LineString':
                raise ValueError(f'Expecting linestring geometry but found {line_geom.type} geometry.')
            # convert the coords to UTM - remember to flip back to lng, lat
            utm_coords = [utm.from_latlon(lat, lng)[:2][::-1] for lng, lat in zip(line_geom.coords.xy[0], line_geom.coords.xy[1])]
            # write back to edge
            g_copy[s][e]['geom'] = geometry.LineString(utm_coords)

    return g_copy


def networkX_decompose(networkX_graph:nx.Graph, decompose_max:float) -> nx.Graph:

    if not isinstance(networkX_graph, nx.Graph):
        raise ValueError('This method requires an undirected networkX graph.')

    logger.info(f'Decomposing graph to maximum edge lengths of {decompose_max}.')
    g_copy = networkX_graph.copy()

    # note -> write to a duplicated graph to avoid in-place errors
    for s, e, d in tqdm(networkX_graph.edges(data=True)):
        # test for x coordinates
        if 'x' not in networkX_graph.nodes[s] or 'x' not in networkX_graph.nodes[e]:
            raise ValueError(f'Encountered node missing "x" coordinate attribute at node {s}.')
        # test for y coordinates
        if 'y' not in networkX_graph.nodes[s] or 'y' not in networkX_graph.nodes[e]:
            raise ValueError(f'Encountered node missing "y" coordinate attribute at node {e}.')
        s_x = networkX_graph.nodes[s]['x']
        s_y = networkX_graph.nodes[s]['y']
        e_x = networkX_graph.nodes[e]['x']
        e_y = networkX_graph.nodes[e]['y']
        # test for geom
        if 'geom' not in d:
            raise ValueError(f'No edge geom found for edge {s}-{e}: Please add an edge "geom" attribute consisting of a shapely LineString.')
        # get edge geometry
        line_geom = d['geom']
        if line_geom.type != 'LineString':
            raise ValueError(f'Expecting linestring geometry but found {line_geom.type} geometry.')
        # check geom coordinates directionality - flip if facing backwards direction
        if (s_x, s_y) == line_geom.coords[-1] and (e_x, e_y) == line_geom.coords[0]:
            flipped_coords = np.fliplr(line_geom.coords.xy)
            line_geom = geometry.LineString([[x, y] for x, y in zip(flipped_coords[0], flipped_coords[1])])
        # double check that coordinates now face the forwards direction
        if not (s_x, s_y) == line_geom.coords[0] and (e_x, e_y) == line_geom.coords[-1]:
            raise ValueError(f'Edge geometry endpoint coordinate mismatch for edge {s}-{e}')
        # see how many segments are necessary so as not to exceed decomposition max distance
        # note that a length less than the decompose threshold will result in a single 'sub'-string
        n = np.ceil(line_geom.length / decompose_max)
        step_size = line_geom.length / n
        # since decomposing, remove the prior edge... but only after properties have been read
        g_copy.remove_edge(s, e)
        # then add the new sub-edge/s
        step = 0
        prior_node_id = s
        sub_node_counter = 0
        # everything inside this loop is a new node - i.e. this loop is effectively skipped if n = 1
        # note, using actual length attribute / n for new length, as provided lengths may not match crow-flies distance
        for i in range(int(n) - 1):
            # create the new node label and id
            new_node_id = f'{s}_{sub_node_counter}_{e}'
            sub_node_counter += 1
            # create the split linestring geom for measuring the new length
            line_segment = ops.substring(line_geom, step, step + step_size)
            # get the x, y of the new end node
            x = line_segment.coords.xy[0][-1]
            y = line_segment.coords.xy[1][-1]
            # add the new node and edge
            g_copy.add_node(new_node_id, x=x, y=y)
            l = line_segment.length
            g_copy.add_edge(prior_node_id, new_node_id, length=l, impedance=l, weight=l)
            # increment the step and node id
            prior_node_id = new_node_id
            step += step_size
        # set the last link manually to avoid rounding errors at end of linestring
        # the nodes already exist, so just add link
        line_segment = ops.substring(line_geom, step, line_geom.length)
        l = line_segment.length
        g_copy.add_edge(prior_node_id, e, length=l, impedance=l, weight=l)

    return g_copy


def networkX_edge_defaults(networkX_graph:nx.Graph) -> nx.Graph:

    if not isinstance(networkX_graph, nx.Graph):
        raise ValueError('This method requires an undirected networkX graph.')

    logger.info('Generating default edge attributes from edge geoms.')
    g_copy = networkX_graph.copy()

    logger.info('Preparing graph')
    for s, e, d in tqdm(g_copy.edges(data=True)):
        if 'geom' not in d:
            raise ValueError(f'No edge geom found for edge {s}-{e}: Please add an edge "geom" attribute consisting of a shapely LineString.')
        # get edge geometry
        line_geom = d['geom']
        if line_geom.type != 'LineString':
            raise ValueError(f'Expecting linestring geometry but found {line_geom.type} geometry.')
        g_copy[s][e]['length'] = line_geom.length
        g_copy[s][e]['impedance'] = line_geom.length
        g_copy[s][e]['weight'] = line_geom.length

    return g_copy


def graph_maps_from_networkX(networkX_graph:nx.Graph) -> Tuple[list, np.ndarray, np.ndarray]:
    '''
    Strategic decisions because of too many edge cases:
    - decided to not discard disconnected components to avoid unintended consequences
    - no internal simplification - use prior methods or tools to clean or simplify the graph before calling this method
    '''

    if not isinstance(networkX_graph, nx.Graph):
        raise ValueError('This method requires an undirected networkX graph.')

    logger.info('Preparing node and edge arrays from networkX graph.')
    g_copy = networkX_graph.copy()

    logger.info('Preparing graph')
    total_out_degrees = 0
    for n in tqdm(g_copy.nodes()):
        # writing to 'labels' in case conversion to integers method interferes with order
        g_copy.nodes[n]['label'] = n
        # sum edges
        total_out_degrees += nx.degree(g_copy, n)

    logger.info('Generating data arrays')
    # convert the nodes to sequential - this permits implicit indices with benefits to speed and structure
    g_copy = nx.convert_node_labels_to_integers(g_copy, 0)
    # prepare the node and link maps
    node_labels = []
    node_map = np.full((g_copy.number_of_nodes(), 4), np.nan)  # float - for consistency
    edge_map = np.full((total_out_degrees, 5), np.nan)  # float - allows for nan and inf
    edge_idx = 0
    # populate the nodes
    for n, d in tqdm(g_copy.nodes(data=True)):
        # label
        node_labels.append(d['label'])
        # cast to int for indexing
        idx = int(n)
        # NODE MAP INDEX POSITION 0 = x coordinate
        if 'x' not in d:
            raise ValueError(f'Encountered node missing "x" coordinate attribute at node {n}.')
        node_map[idx][0] = d['x']
        # NODE MAP INDEX POSITION 1 = y coordinate
        if 'y' not in d:
            raise ValueError(f'Encountered node missing "y" coordinate attribute at node {n}.')
        node_map[idx][1] = d['y']
        # NODE MAP INDEX POSITION 2 = live or not
        if 'live' in d:
            node_map[idx][2] = d['live']
        else:
            node_map[idx][2] = True
        # NODE MAP INDEX POSITION 3 = starting index for edges in edge map
        node_map[idx][3] = edge_idx
        # follow all out links and add these to the edge_map
        # this happens for both directions
        for nb in g_copy.neighbors(n):
            # EDGE MAP INDEX POSITION 0 = start node
            edge_map[edge_idx][0] = idx
            # EDGE MAP INDEX POSITION 1 = end node
            edge_map[edge_idx][1] = nb
            # EDGE MAP INDEX POSITION 2 = length
            if 'length' not in g_copy[idx][nb]:
                raise ValueError(f'No "length" attribute for edge {idx}-{nb}.')
            # cannot have zero length
            l = g_copy[idx][nb]['length']
            if not np.isfinite(l) or l <= 0:
                raise ValueError(f'Length attribute {l} for edge {idx}-{nb} must be a finite positive value.')
            edge_map[edge_idx][2] = l
            # EDGE MAP INDEX POSITION 3 = impedance
            if 'impedance' not in g_copy[idx][nb]:
                raise ValueError(f'No "impedance" attribute for edge {idx}-{nb}.')
            # cannot have impedance less than zero (but == 0 is OK)
            imp = g_copy[idx][nb]['impedance']
            if not (np.isfinite(imp) or np.isinf(imp)) or imp < 0:
                raise ValueError(f'Impedance attribute {imp} for edge {idx}-{nb} must be a finite positive value or positive infinity.')
            edge_map[edge_idx][3] = imp
            # EDGE MAP INDEX POSITION 4 = weight
            if 'weight' not in g_copy[idx][nb]:
                raise ValueError(f'No "weight" attribute for edge {idx}-{nb}.')
            edge_map[edge_idx][4] = g_copy[idx][nb]['weight']
            # increment the link_idx
            edge_idx += 1

    return node_labels, node_map, edge_map


def compute_centrality(node_map:np.ndarray, edge_map:np.ndarray, distances:list, close_metrics:list=[],
                        between_metrics:list=[], min_threshold_wt:float=0.01831563888873418, angular_wt:bool=False) \
                        -> Tuple[Any, ...]:

    if node_map.shape[1] != 4:
        raise ValueError('The node map must have a dimensionality of nx4, consisting of x, y, live, and link idx parameters.')

    if edge_map.shape[1] != 5:
        raise ValueError('The link map must have a dimensionality of nx4, consisting of start, end, distance, and weight parameters.')

    if not distances:
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

    closeness_options = ['count', 'farness', 'farness_meters', 'harmonic', 'improved', 'gravity', 'cycles']
    closeness_map = []
    for cl in close_metrics:
        if cl not in closeness_options:
            raise ValueError(f'Invalid closeness option: {cl}. Must be one of {", ".join(closeness_options)}.')
        closeness_map.append(closeness_options.index(cl))

    betweenness_options = ['count', 'weighted', 'gravity_weighted']
    betweenness_map = []
    for bt in between_metrics:
        if bt not in betweenness_options:
            raise ValueError(f'Invalid betweenness option: {bt}. Must be one of {", ".join(betweenness_options)}.')
        betweenness_map.append(betweenness_options.index(bt))

    closeness_data, betweenness_data = networks.network_centralities(node_map, edge_map, np.array(distances),
                                        np.array(betas), np.array(closeness_map), np.array(betweenness_map), angular_wt)

    # return statement tuple unpacking supported from Python 3.8... till then, unpack first
    return_data = *closeness_data[closeness_map], *betweenness_data[betweenness_map], betas
    return return_data


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