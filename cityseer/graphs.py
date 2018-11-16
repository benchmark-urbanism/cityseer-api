'''
General graph manipulation
'''
import logging
from typing import Union, Tuple, Any
import utm
from shapely import geometry, ops
import networkx as nx
from tqdm import tqdm
import numpy as np


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
            g_copy.add_edge(prior_node_id, new_node_id, length=l, impedance=l)
            # increment the step and node id
            prior_node_id = new_node_id
            step += step_size
        # set the last link manually to avoid rounding errors at end of linestring
        # the nodes already exist, so just add link
        line_segment = ops.substring(line_geom, step, line_geom.length)
        l = line_segment.length
        g_copy.add_edge(prior_node_id, e, length=l, impedance=l)

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

    return g_copy


def networkX_length_weighted_nodes(networkX_graph:nx.Graph) -> nx.Graph:

    if not isinstance(networkX_graph, nx.Graph):
        raise ValueError('This method requires an undirected networkX graph.')

    logger.info('Generating default edge attributes from edge geoms.')
    g_copy = networkX_graph.copy()

    for n in g_copy.nodes():
        agg_length = 0
        for nb in g_copy.neighbors(n):
            # test for length attribute
            if 'length' not in g_copy[n][nb]:
                raise ValueError(f'No "length" attribute available for edge {n}-{nb}.')
            agg_length += g_copy[n][nb]['length'] / 2
        g_copy.nodes[n]['weight'] = agg_length

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
    node_map = np.full((g_copy.number_of_nodes(), 5), np.nan)  # float - for consistency
    edge_map = np.full((total_out_degrees, 4), np.nan)  # float - allows for nan and inf
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
        # NODE MAP INDEX POSITION 4 = weight
        if 'weight' in d:
            node_map[idx][4] = d['weight']
        else:
            node_map[idx][4] = 1
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
            # increment the link_idx
            edge_idx += 1

    return node_labels, node_map, edge_map