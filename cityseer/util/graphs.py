'''
General graph manipulation
'''
import logging
from typing import Union, Tuple

import networkx as nx
import numpy as np
import utm
from shapely import geometry, ops
from tqdm import tqdm

from cityseer.algos import checks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# TODO: this corrected shapely function is temporary until the fix is released in Shapely 1.7 - submitted PR#658. Note, also used from test_networkX_remove_straight_intersections()
def substring(geom, start_dist, end_dist, normalized=False):
    assert (isinstance(geom, geometry.LineString))

    # Filter out cases in which to return a point
    if start_dist == end_dist:
        return geom.interpolate(start_dist, normalized)
    elif not normalized and start_dist >= geom.length and end_dist >= geom.length:
        return geom.interpolate(geom.length, normalized)
    elif not normalized and -start_dist >= geom.length and -end_dist >= geom.length:
        return geom.interpolate(0, normalized)
    elif normalized and start_dist >= 1 and end_dist >= 1:
        return geom.interpolate(1, normalized)
    elif normalized and -start_dist >= 1 and -end_dist >= 1:
        return geom.interpolate(0, normalized)

    start_point = geom.interpolate(start_dist, normalized)
    end_point = geom.interpolate(end_dist, normalized)

    min_dist = min(start_dist, end_dist)
    max_dist = max(start_dist, end_dist)
    if normalized:
        min_dist *= geom.length
        max_dist *= geom.length

    vertex_list = [(start_point.x, start_point.y)]
    coords = list(geom.coords)
    for p in coords:
        pd = geom.project(geometry.Point(p))
        if pd <= min_dist:
            pass
        elif min_dist < pd < max_dist:
            vertex_list.append(p)
        else:
            break
    vertex_list.append((end_point.x, end_point.y))

    # reverse direction of section
    if start_dist > end_dist:
        vertex_list = reversed(vertex_list)

    return geometry.LineString(vertex_list)


def nX_simple_geoms(networkX_graph: nx.Graph) -> nx.Graph:
    if not isinstance(networkX_graph, nx.Graph):
        raise TypeError('This method requires an undirected networkX graph.')

    logger.info('Generating simple (straight) edge geometries.')
    g_copy = networkX_graph.copy()

    # unpack coordinates and build simple edge geoms
    for s, e in tqdm(g_copy.edges()):

        # start x coordinate
        if 'x' not in g_copy.nodes[s]:
            raise AttributeError(f'Encountered node missing "x" coordinate attribute at node {s}.')
        s_x = g_copy.nodes[s]['x']
        # start y coordinate
        if 'y' not in g_copy.nodes[s]:
            raise AttributeError(f'Encountered node missing "y" coordinate attribute at node {s}.')
        s_y = g_copy.nodes[s]['y']
        # end x coordinate
        if 'x' not in g_copy.nodes[e]:
            raise AttributeError(f'Encountered node missing "x" coordinate attribute at node {e}.')
        e_x = g_copy.nodes[e]['x']
        # end y coordinate
        if 'y' not in g_copy.nodes[e]:
            raise AttributeError(f'Encountered node missing "y" coordinate attribute at node {e}.')
        e_y = g_copy.nodes[e]['y']

        g_copy[s][e]['geom'] = geometry.LineString([[s_x, s_y], [e_x, e_y]])

    return g_copy


def nX_wgs_to_utm(networkX_graph: nx.Graph) -> nx.Graph:
    if not isinstance(networkX_graph, nx.Graph):
        raise TypeError('This method requires an undirected networkX graph.')

    logger.info('Converting networkX graph from WGS to UTM.')
    g_copy = networkX_graph.copy()

    logger.info('Processing node x, y coordinates.')
    for n, d in tqdm(g_copy.nodes(data=True)):
        # x coordinate
        if 'x' not in d:
            raise AttributeError(f'Encountered node missing "x" coordinate attribute at node {n}.')
        x = d['x']
        # y coordinate
        if 'y' not in d:
            raise AttributeError(f'Encountered node missing "y" coordinate attribute at node {n}.')
        y = d['y']
        # check for unintentional use of conversion
        if x > 180 or y > 90:
            raise AttributeError('x, y coordinates exceed WGS bounds. Please check your coordinate system.')
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
                raise TypeError(f'Expecting LineString geometry but found {line_geom.type} geometry.')
            # convert the coords to UTM - remember to flip back to lng, lat
            utm_coords = [utm.from_latlon(lat, lng)[:2][::-1] for lng, lat in line_geom.coords]
            # write back to edge
            g_copy[s][e]['geom'] = geometry.LineString(utm_coords)

    return g_copy


def nX_remove_filler_nodes(networkX_graph: nx.Graph) -> nx.Graph:
    if not isinstance(networkX_graph, nx.Graph):
        raise TypeError('This method requires an undirected networkX graph.')

    logger.info(f'Simplifying graph intersections.')
    g_copy = networkX_graph.copy()

    # remove self-edges, otherwise nx.degree includes self-loops
    for s, e in nx.selfloop_edges(g_copy):
        g_copy.remove_edge(s, e)

    # iterate the nodes and weld edges where encountering simple intersections
    # use the original graph so as to write changes to new graph
    for n in networkX_graph.nodes():
        if nx.degree(networkX_graph, n) == 2:

            # get neighbours and geoms either side
            nb_a, nb_b = list(nx.neighbors(networkX_graph, n))

            # geom A
            if 'geom' not in networkX_graph[n][nb_a]:
                raise AttributeError(f'Missing "geom" attribute for edge {n}-{nb_a}')
            geom_a = networkX_graph[n][nb_a]['geom']
            if geom_a.type != 'LineString':
                raise AttributeError(f'Expecting LineString geometry but found {geom_a.type} geometry.')
            # geom B
            if 'geom' not in networkX_graph[n][nb_b]:
                raise AttributeError(f'Missing "geom" attribute for edge {n}-{nb_b}')
            geom_b = networkX_graph[n][nb_b]['geom']
            if geom_b.type != 'LineString':
                raise AttributeError(f'Expecting LineString geometry but found {geom_b.type} geometry.')

            # remove old node - edges are removed implicitly
            g_copy.remove_node(n)

            # add new edge
            merged_line = ops.linemerge([geom_a, geom_b])
            if merged_line.type != 'LineString':
                raise AttributeError(
                    f'Found {merged_line.type} geometry instead of "LineString" for new geom {merged_line.wkt}. Check that the adjacent LineStrings for {nb_a}-{n} and {n}-{nb_b} actually touch.')
            g_copy.add_edge(nb_a, nb_b, geom=merged_line)

    return g_copy


def nX_decompose(networkX_graph: nx.Graph, decompose_max: float) -> nx.Graph:
    if not isinstance(networkX_graph, nx.Graph):
        raise TypeError('This method requires an undirected networkX graph.')

    logger.info(f'Decomposing graph to maximum edge lengths of {decompose_max}.')
    g_copy = networkX_graph.copy()

    # note -> write to a duplicated graph to avoid in-place errors
    for s, e, d in tqdm(networkX_graph.edges(data=True)):
        # test for x coordinates
        if 'x' not in networkX_graph.nodes[s] or 'y' not in networkX_graph.nodes[s]:
            raise AttributeError(f'Encountered node missing "x" or "y" coordinate attributes at node {s}.')
        # test for y coordinates
        if 'x' not in networkX_graph.nodes[e] or 'y' not in networkX_graph.nodes[e]:
            raise AttributeError(f'Encountered node missing "x" or "y" coordinate attributes at node {e}.')
        s_x = networkX_graph.nodes[s]['x']
        s_y = networkX_graph.nodes[s]['y']
        e_x = networkX_graph.nodes[e]['x']
        e_y = networkX_graph.nodes[e]['y']
        # test for geom
        if 'geom' not in d:
            raise AttributeError(
                f'No edge geom found for edge {s}-{e}: Please add an edge "geom" attribute consisting of a shapely LineString.')
        # get edge geometry
        line_geom = d['geom']
        if line_geom.type != 'LineString':
            raise TypeError(f'Expecting LineString geometry but found {line_geom.type} geometry for edge {s}-{e}.')
        # check geom coordinates directionality - flip if facing backwards direction
        if not (s_x, s_y) == line_geom.coords[0][:2]:
            line_geom = geometry.LineString(line_geom.coords[::-1])
        # double check that coordinates now face the forwards direction
        if not (s_x, s_y) == line_geom.coords[0][:2] or not (e_x, e_y) == line_geom.coords[-1][:2]:
            raise AttributeError(f'Edge geometry endpoint coordinate mismatch for edge {s}-{e}')
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
        for i in range(int(n) - 1):
            # create the new node label and id
            new_node_id = f'{s}_{sub_node_counter}_{e}'
            sub_node_counter += 1
            # create the split LineString geom for measuring the new length
            line_segment = substring(line_geom, step, step + step_size)
            # get the x, y of the new end node
            x, y = line_segment.coords[-1]
            # add the new node and edge
            g_copy.add_node(new_node_id, x=x, y=y)
            # add and set live property if present in parent graph
            if 'live' in networkX_graph.nodes[s] and 'live' in networkX_graph.nodes[e]:
                live = True
                # if BOTH parents are not live, then set child to not live
                if not networkX_graph.nodes[s]['live'] and not networkX_graph.nodes[e]['live']:
                    live = False
                g_copy.nodes[new_node_id]['live'] = live
            # add the edge
            l = line_segment.length
            g_copy.add_edge(prior_node_id, new_node_id, length=l, impedance=l)
            # increment the step and node id
            prior_node_id = new_node_id
            step += step_size
        # set the last edge manually to avoid rounding errors at end of LineString
        # the nodes already exist, so just add edge
        line_segment = substring(line_geom, step, line_geom.length)
        l = line_segment.length
        g_copy.add_edge(prior_node_id, e, length=l, impedance=l)

    return g_copy


def nX_to_dual(networkX_graph: nx.Graph) -> nx.Graph:
    '''
    Not to be used on angular graphs - would overwrite angular impedance
    '''

    if not isinstance(networkX_graph, nx.Graph):
        raise TypeError('This method requires an undirected networkX graph.')

    logger.info('Converting graph to dual with angular impedances.')
    g_dual = nx.Graph()

    def get_half_geoms(g, a_node, b_node):
        '''
        For splitting and orienting half geoms
        '''
        # get edge data
        edge_data = g[a_node][b_node]
        # test for x coordinates
        if 'x' not in g.nodes[a_node] or 'y' not in g.nodes[a_node]:
            raise AttributeError(f'Encountered node missing "x" or "y" coordinate attributes at node {a_node}.')
        # test for y coordinates
        if 'x' not in g.nodes[b_node] or 'y' not in g.nodes[b_node]:
            raise AttributeError(f'Encountered node missing "x" or "y" coordinate attributes at node {b_node}.')
        a_x = g.nodes[a_node]['x']
        a_y = g.nodes[a_node]['y']
        b_x = g.nodes[b_node]['x']
        b_y = g.nodes[b_node]['y']
        # test for geom
        if 'geom' not in edge_data:
            raise AttributeError(
                f'No edge geom found for edge {a_node}-{b_node}: Please add an edge "geom" attribute consisting of a shapely LineString.')
        # get edge geometry
        line_geom = edge_data['geom']
        if line_geom.type != 'LineString':
            raise TypeError(
                f'Expecting LineString geometry but found {line_geom.type} geometry for edge {a_node}-{b_node}.')
        # check geom coordinates directionality - flip if facing backwards direction - beware 3d coords
        if not (a_x, a_y) == line_geom.coords[0][:2]:
            line_geom = geometry.LineString(line_geom.coords[::-1])
        # double check that coordinates now face the forwards direction
        if not (a_x, a_y) == line_geom.coords[0][:2] or not (b_x, b_y) == line_geom.coords[-1][:2]:
            raise AttributeError(f'Edge geometry endpoint coordinate mismatch for edge {a_node}-{b_node}')
        # generate the two half geoms
        a_half_geom = substring(line_geom, 0, line_geom.length / 2)
        b_half_geom = substring(line_geom, line_geom.length / 2, line_geom.length)
        assert a_half_geom.coords[-1][:2] == b_half_geom.coords[0][:2]

        return a_half_geom, b_half_geom

    # iterate the primal graph's edges
    for s, e, d in tqdm(networkX_graph.edges(data=True)):

        # get the first and second half geoms
        s_half_geom, e_half_geom = get_half_geoms(networkX_graph, s, e)

        # create a new dual node corresponding to the current primal edge
        s_e = sorted([s, e])
        hub_node_dual = f'{s_e[0]}_{s_e[1]}'
        x, y = s_half_geom.coords[-1][:2]
        g_dual.add_node(hub_node_dual, x=x, y=y)
        # add and set live property if present in parent graph
        if 'live' in networkX_graph.nodes[s] and 'live' in networkX_graph.nodes[e]:
            live = True
            # if BOTH parents are not live, then set child to not live
            if not networkX_graph.nodes[s]['live'] and not networkX_graph.nodes[e]['live']:
                live = False
            g_dual.nodes[hub_node_dual]['live'] = live

        # process either side
        for n_side, half_geom in zip([s, e], [s_half_geom, e_half_geom]):

            # add the spoke edges on the dual
            for nb in nx.neighbors(networkX_graph, n_side):

                # don't follow neighbour back to current edge combo
                if nb in [s, e]:
                    continue

                # get the near and far half geoms
                spoke_half_geom, _discard_geom = get_half_geoms(networkX_graph, n_side, nb)

                # add the neighbouring primal edge as dual node
                s_nb = sorted([n_side, nb])
                spoke_node_dual = f'{s_nb[0]}_{s_nb[1]}'
                x, y = spoke_half_geom.coords[-1][:2]
                g_dual.add_node(spoke_node_dual, x=x, y=y)
                # add and set live property if present in parent graph
                if 'live' in networkX_graph.nodes[n_side] and 'live' in networkX_graph.nodes[nb]:
                    live = True
                    # if BOTH parents are not live, then set child to not live
                    if not networkX_graph.nodes[n_side]['live'] and not networkX_graph.nodes[nb]['live']:
                        live = False
                    g_dual.nodes[spoke_node_dual]['live'] = live

                # weld the lines
                merged_line = ops.linemerge([half_geom, spoke_half_geom])
                if merged_line.type != 'LineString':
                    raise TypeError(
                        f'Found {merged_line.type} geometry instead of "LineString" for new geom {merged_line.wkt}. Check that the LineStrings for {s}-{e} and {n_side}-{nb} actually touch.')

                # iterate the coordinates and sum the calculate the angular change
                sum_angles = 0
                for i in range(len(merged_line.coords) - 2):
                    x_1, y_1 = merged_line.coords[i][:2]
                    x_2, y_2 = merged_line.coords[i + 1][:2]
                    x_3, y_3 = merged_line.coords[i + 2][:2]
                    # arctan2 is y / x order
                    a_1 = np.rad2deg(np.arctan2(y_2 - y_1, x_2 - x_1))
                    a_2 = np.rad2deg(np.arctan2(y_3 - y_2, x_3 - x_2))

                    sum_angles += np.abs((a_2 - a_1 + 180) % 360 - 180)

                    # A = np.array(merged_line.coords[i + 1]) - np.array(merged_line.coords[i])
                    # B = np.array(merged_line.coords[i + 2]) - np.array(merged_line.coords[i + 1])
                    # angle = np.abs(np.degrees(np.math.atan2(np.linalg.det([A, B]), np.dot(A, B))))

                # add the dual edge
                g_dual.add_edge(hub_node_dual, spoke_node_dual, parent_primal_node=n_side, length=merged_line.length,
                                impedance=sum_angles, geom=merged_line)

    return g_dual


def nX_auto_edge_params(networkX_graph: nx.Graph) -> nx.Graph:
    '''
    Not to be used on angular graphs - would overwrite angular impedance
    '''

    if not isinstance(networkX_graph, nx.Graph):
        raise TypeError('This method requires an undirected networkX graph.')

    logger.info('Generating default edge attributes from edge geoms.')
    g_copy = networkX_graph.copy()

    logger.info('Preparing graph')
    for s, e, d in tqdm(g_copy.edges(data=True)):
        if 'geom' not in d:
            raise AttributeError(
                f'No edge geom found for edge {s}-{e}: Please add an edge "geom" attribute consisting of a shapely LineString.')
        # get edge geometry
        line_geom = d['geom']
        if line_geom.type != 'LineString':
            raise TypeError(f'Expecting LineString geometry but found {line_geom.type} geometry for edge {s}-{e}.')
        g_copy[s][e]['length'] = line_geom.length
        g_copy[s][e]['impedance'] = line_geom.length

    return g_copy


def nX_m_weighted_nodes(networkX_graph: nx.Graph) -> nx.Graph:
    if not isinstance(networkX_graph, nx.Graph):
        raise TypeError('This method requires an undirected networkX graph.')

    logger.info('Generating default edge attributes from edge geoms.')
    g_copy = networkX_graph.copy()

    for n in tqdm(g_copy.nodes()):
        agg_length = 0
        for nb in g_copy.neighbors(n):
            # test for length attribute
            if 'length' not in g_copy[n][nb]:
                raise AttributeError(f'No "length" attribute available for edge {n}-{nb}.')
            agg_length += g_copy[n][nb]['length'] / 2
        g_copy.nodes[n]['weight'] = agg_length

    return g_copy


def graph_maps_from_nX(networkX_graph: nx.Graph) -> Tuple[tuple, np.ndarray, np.ndarray]:
    '''
    Strategic decisions because of too many edge cases:
    - decided to not discard disconnected components to avoid unintended consequences
    - no internal simplification - use prior methods or tools to clean or simplify the graph before calling this method
    '''

    if not isinstance(networkX_graph, nx.Graph):
        raise TypeError('This method requires an undirected networkX graph.')

    logger.info('Preparing node and edge arrays from networkX graph.')
    g_copy = networkX_graph.copy()

    # remove self-edges, otherwise nx.degree includes self-loops
    for s, e in nx.selfloop_edges(g_copy):
        g_copy.remove_edge(s, e)

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
    # prepare the node and edge maps
    node_uids = []
    node_map = np.full((g_copy.number_of_nodes(), 5), np.nan)  # float - for consistency
    edge_map = np.full((total_out_degrees, 4), np.nan)  # float - allows for nan and inf
    edge_idx = 0
    # populate the nodes
    for n, d in tqdm(g_copy.nodes(data=True)):
        # label
        node_uids.append(d['label'])
        # cast to int for indexing
        i = int(n)
        # NODE MAP INDEX POSITION 0 = x coordinate
        if 'x' not in d:
            raise AttributeError(f'Encountered node missing "x" coordinate attribute at node {n}.')
        node_map[i][0] = d['x']
        # NODE MAP INDEX POSITION 1 = y coordinate
        if 'y' not in d:
            raise AttributeError(f'Encountered node missing "y" coordinate attribute at node {n}.')
        node_map[i][1] = d['y']
        # NODE MAP INDEX POSITION 2 = live or not
        if 'live' in d:
            node_map[i][2] = d['live']
        else:
            node_map[i][2] = True
        # NODE MAP INDEX POSITION 3 = starting index for edges in edge map
        # NB - if an isolated node, then this should be np.nan
        # otherwise it will refer to the incorrect edge since the edge won't be iterated below
        if nx.degree(g_copy, i) == 0:
            node_map[i][3] = np.nan
        else:
            node_map[i][3] = edge_idx
        # NODE MAP INDEX POSITION 4 = weight
        if 'weight' in d:
            node_map[i][4] = d['weight']
        else:
            node_map[i][4] = 1
        # follow all out edges and add these to the edge_map
        # this happens for both directions
        for nb in g_copy.neighbors(n):
            # EDGE MAP INDEX POSITION 0 = start node
            edge_map[edge_idx][0] = i
            # EDGE MAP INDEX POSITION 1 = end node
            edge_map[edge_idx][1] = nb
            # EDGE MAP INDEX POSITION 2 = length
            if 'length' not in g_copy[i][nb]:
                raise AttributeError(f'No "length" attribute for edge {i}-{nb}.')
            # cannot have zero length
            l = g_copy[i][nb]['length']
            if not np.isfinite(l) or l <= 0:
                raise AttributeError(f'Length attribute {l} for edge {i}-{nb} must be a finite positive value.')
            edge_map[edge_idx][2] = l
            # EDGE MAP INDEX POSITION 3 = impedance
            if 'impedance' not in g_copy[i][nb]:
                raise AttributeError(f'No "impedance" attribute for edge {i}-{nb}.')
            # cannot have impedance less than zero (but == 0 is OK)
            imp = g_copy[i][nb]['impedance']
            if not (np.isfinite(imp) or np.isinf(imp)) or imp < 0:
                raise AttributeError(
                    f'Impedance attribute {imp} for edge {i}-{nb} must be a finite positive value or positive infinity.')
            edge_map[edge_idx][3] = imp
            # increment the edge_idx
            edge_idx += 1

    return tuple(node_uids), node_map, edge_map


def nX_from_graph_maps(node_uids: Union[tuple, list],
                       node_map: np.ndarray,
                       edge_map: np.ndarray,
                       networkX_graph: nx.Graph = None,
                       metrics_dict: dict = None) -> nx.Graph:
    logger.info('Populating node and edge map data to a networkX graph.')

    checks.check_network_maps(node_map, edge_map)

    if networkX_graph is not None:
        logger.info('Reusing existing graph as backbone.')
        if networkX_graph.number_of_nodes() != len(node_map):
            raise ValueError('The number of nodes in the graph does not match the number of nodes in the node map.')
        g_copy = networkX_graph.copy()
        for uid in node_uids:
            if uid not in g_copy:
                raise AttributeError(
                    f'Node uid {uid} not found in graph. If passing a graph as backbone, the uids must match those supplied with the node and edge maps.')
    else:
        logger.info('No existing graph found, creating new.')
        g_copy = nx.Graph()
        for uid in node_uids:
            g_copy.add_node(uid)

    logger.info('Unpacking node data.')
    for uid, node in tqdm(zip(node_uids, node_map)):
        x, y, live, edge_idx, wt = node
        g_copy.nodes[uid]['x'] = x
        g_copy.nodes[uid]['y'] = y
        g_copy.nodes[uid]['live'] = bool(live)
        g_copy.nodes[uid]['weight'] = wt

    logger.info('Unpacking edge data.')
    for edge in tqdm(edge_map):
        start, end, length, impedance = edge
        start_uid = node_uids[int(start)]
        end_uid = node_uids[int(end)]
        # networkX will silently add new edges / data over existing edges
        g_copy.add_edge(start_uid, end_uid, length=length, impedance=impedance)

    if metrics_dict is not None:
        logger.info('Unpacking metrics to nodes.')
        for uid, metrics in tqdm(metrics_dict.items()):
            if uid not in g_copy:
                raise AttributeError(
                    f'Node uid {uid} not found in graph. Data dictionary uids must match those supplied with the node and edge maps.')
            g_copy.nodes[uid]['metrics'] = metrics

    return g_copy
