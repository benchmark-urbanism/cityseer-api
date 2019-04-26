'''
General graph manipulation
'''
import logging
import os
import uuid
from typing import Union, Tuple

import networkx as nx
import numpy as np
import utm
from shapely import geometry, ops, strtree
from tqdm import tqdm

from cityseer.algos import checks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

tqdm_suppress = False
if 'GCP_PROJECT' in os.environ:
    tqdm_suppress = True


# TODO: this corrected shapely function is temporary until the fix is released in Shapely 1.7 - submitted PR#658.
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
    for s, e in tqdm(g_copy.edges(), disable=tqdm_suppress):

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


def nX_wgs_to_utm(networkX_graph: nx.Graph, force_zone_number=None) -> nx.Graph:
    if not isinstance(networkX_graph, nx.Graph):
        raise TypeError('This method requires an undirected networkX graph.')

    logger.info('Converting networkX graph from WGS to UTM.')
    g_copy = networkX_graph.copy()

    zone_number = None
    if force_zone_number is not None:
        zone_number = force_zone_number

    logger.info('Processing node x, y coordinates.')
    for n, d in tqdm(g_copy.nodes(data=True), disable=tqdm_suppress):
        # x coordinate
        if 'x' not in d:
            raise AttributeError(f'Encountered node missing "x" coordinate attribute at node {n}.')
        lng = d['x']
        # y coordinate
        if 'y' not in d:
            raise AttributeError(f'Encountered node missing "y" coordinate attribute at node {n}.')
        lat = d['y']
        # check for unintentional use of conversion
        if abs(lng) > 180 or abs(lat) > 90:
            raise AttributeError('x, y coordinates exceed WGS bounds. Please check your coordinate system.')
        # to avoid issues across UTM boundaries, use the first point to set (and subsequently force) the UTM zone
        if zone_number is None:
            zone_number = utm.from_latlon(lat, lng)[2]  # zone number is position 2
        # be cognisant of parameter and return order
        # returns in easting, northing order
        easting, northing = utm.from_latlon(lat, lng, force_zone_number=zone_number)[:2]
        # write back to graph
        g_copy.nodes[n]['x'] = easting
        g_copy.nodes[n]['y'] = northing

    # if line geom property provided, then convert as well
    logger.info('Processing edge geom coordinates, if present.')
    for s, e, d in tqdm(g_copy.edges(data=True), disable=tqdm_suppress):
        # check if geom present - optional step
        if 'geom' in d:
            line_geom = d['geom']
            if line_geom.type != 'LineString':
                raise TypeError(f'Expecting LineString geometry but found {line_geom.type} geometry.')
            # be cognisant of parameter and return order
            # returns in easting, northing order
            utm_coords = [utm.from_latlon(lat, lng, force_zone_number=zone_number)[:2] for lng, lat in line_geom.coords]
            # write back to edge
            g_copy[s][e]['geom'] = geometry.LineString(utm_coords)

    return g_copy


def nX_remove_dangling_nodes(networkX_graph: nx.Graph,
                             despine: float = 25,
                             remove_disconnected: bool = True) -> nx.Graph:
    logger.info(f'Removing dangling nodes.')
    g_copy = networkX_graph.copy()

    if remove_disconnected:
        connected_components = list(nx.connected_component_subgraphs(g_copy))
        g_copy = sorted(connected_components, key=len, reverse=True)[0]

    if despine:
        remove_nodes = []
        for n, d in tqdm(g_copy.nodes(data=True), disable=tqdm_suppress):
            if nx.degree(g_copy, n) == 1:
                nb = list(nx.neighbors(g_copy, n))[0]
                if g_copy[n][nb]['geom'].length <= despine:
                    remove_nodes.append(n)
        g_copy.remove_nodes_from(remove_nodes)

    return g_copy


def nX_remove_filler_nodes(networkX_graph: nx.Graph) -> nx.Graph:
    if not isinstance(networkX_graph, nx.Graph):
        raise TypeError('This method requires an undirected networkX graph.')

    logger.info(f'Removing filler nodes.')
    g_copy = networkX_graph.copy()
    removed_nodes = set()

    def manual_weld(_G, _start_node, _geom_a, _geom_b):
        s_x = _G.nodes[_start_node]['x']
        s_y = _G.nodes[_start_node]['y']
        # check geom coordinates directionality - flip to wind in same direction
        # i.e. _geom_a should start at _start_node whereas _geom_b should end at _start_node
        if not (s_x, s_y) == _geom_a.coords[0][:2]:
            _geom_a = geometry.LineString(_geom_a.coords[::-1])
        if not (s_x, s_y) == _geom_b.coords[-1][:2]:
            _geom_b = geometry.LineString(_geom_b.coords[::-1])
        # now concatenate
        _new_agg_geom = geometry.LineString(list(_geom_a.coords) + list(_geom_b.coords))
        # check
        assert _new_agg_geom.coords[0] == (s_x, s_y)
        assert _new_agg_geom.coords[-1] == (s_x, s_y)
        return _new_agg_geom

    def recursive_weld(_G, start_node, agg_geom, agg_del_nodes, curr_node, next_node):

        # if the next node has a degree of 2, then follow the chain
        # for disconnected components, check that the next node is not back at the start node...
        if nx.degree(_G, next_node) == 2 and next_node != start_node:
            # next node becomes new current
            _new_curr = next_node
            # add next node to delete list
            agg_del_nodes.append(next_node)
            # get its neighbours
            _a, _b = list(nx.neighbors(networkX_graph, next_node))
            # proceed to the new_next node
            if _a == curr_node:
                _new_next = _b
            else:
                _new_next = _a
            # get the geom and weld
            if 'geom' not in _G[_new_curr][_new_next]:
                raise AttributeError(f'Missing "geom" attribute for edge {_new_curr}-{_new_next}')
            new_geom = _G[_new_curr][_new_next]['geom']
            if new_geom.type != 'LineString':
                raise AttributeError(f'Expecting LineString geometry but found {new_geom.type} geometry.')
            # when welding an isolated circular component, the ops linemerge will potentially weld onto the wrong end
            # i.e. start-side instead of end-side... so orient and merge manually
            if _new_next == start_node:
                _new_agg_geom = manual_weld(_G, start_node, new_geom, agg_geom)
            else:
                _new_agg_geom = ops.linemerge([agg_geom, new_geom])
            if _new_agg_geom.type != 'LineString':
                raise AttributeError(
                    f'Found {_new_agg_geom.type} geometry instead of "LineString" for new geom {_new_agg_geom.wkt}.'
                    f'Check that the adjacent LineStrings in the vicinity of {curr_node}-{next_node} are not corrupted.')
            return recursive_weld(_G, start_node, _new_agg_geom, agg_del_nodes, _new_curr, _new_next)
        else:
            end_node = next_node
            return agg_geom, agg_del_nodes, end_node

    # iterate the nodes and weld edges where encountering simple intersections
    # use the original graph so as to write changes to new graph
    for n in tqdm(networkX_graph.nodes(), disable=tqdm_suppress):

        # some nodes will already have been removed via recursive function
        if n in removed_nodes:
            continue

        if nx.degree(networkX_graph, n) == 2:

            # get neighbours and geoms either side
            nb_a, nb_b = list(nx.neighbors(networkX_graph, n))

            # geom A
            if 'geom' not in networkX_graph[n][nb_a]:
                raise AttributeError(f'Missing "geom" attribute for edge {n}-{nb_a}')
            geom_a = networkX_graph[n][nb_a]['geom']
            if geom_a.type != 'LineString':
                raise AttributeError(f'Expecting LineString geometry but found {geom_a.type} geometry.')
            # start the A direction recursive weld
            agg_geom_a, agg_del_nodes_a, end_node_a = recursive_weld(networkX_graph, n, geom_a, [], n, nb_a)

            # only follow geom B if geom A doesn't return an isolated (disconnected) looping component
            # e.g. circular disconnected walkway
            if end_node_a == n:
                logger.warning(f'Disconnected looping component encountered around {n}')
                # in this case, do not remove the starting node because it suspends the loop
                g_copy.remove_nodes_from(agg_del_nodes_a)
                removed_nodes.update(agg_del_nodes_a)
                g_copy.add_edge(n, n, geom=agg_geom_a)
                continue

            # geom B
            if 'geom' not in networkX_graph[n][nb_b]:
                raise AttributeError(f'Missing "geom" attribute for edge {n}-{nb_b}')
            geom_b = networkX_graph[n][nb_b]['geom']
            if geom_b.type != 'LineString':
                raise AttributeError(f'Expecting LineString geometry but found {geom_b.type} geometry.')
            # start the B direction recursive weld
            agg_geom_b, agg_del_nodes_b, end_node_b = recursive_weld(networkX_graph, n, geom_b, [], n, nb_b)

            # remove old nodes - edges are removed implicitly
            agg_del_nodes = agg_del_nodes_a + agg_del_nodes_b
            # also remove origin node n
            agg_del_nodes.append(n)
            g_copy.remove_nodes_from(agg_del_nodes)
            removed_nodes.update(agg_del_nodes)

            # merge the lines
            # disconnected self-loops are caught above per geom a, i.e. where the whole loop is degree == 2
            # however, lollipop scenarios are not, so weld manually
            # lollipop scenarios are where a looping component (all degrees == 2) suspends off a node with degree > 2
            if end_node_a == end_node_b:
                merged_line = manual_weld(networkX_graph, end_node_a, agg_geom_a, agg_geom_b)
            else:
                merged_line = ops.linemerge([agg_geom_a, agg_geom_b])

            # run checks
            if merged_line.type != 'LineString':
                raise AttributeError(
                    f'Found {merged_line.type} geometry instead of "LineString" for new geom {merged_line.wkt}. '
                    f'Check that the adjacent LineStrings for {nb_a}-{n} and {n}-{nb_b} actually touch.')

            # add new edge
            g_copy.add_edge(end_node_a, end_node_b, geom=merged_line)

    return g_copy


def _dissolve_adjacent(_target_graph: nx.Graph,
                       _parent_node_name: str,
                       _node_group: Union[set, list, tuple],
                       highest_degree=False) -> nx.Graph:

    # set the new centroid from the centroid of the node group's Multipoint:
    node_geoms = []
    if not highest_degree:
        for n_uid in _node_group:
            x = _target_graph.nodes[n_uid]['x']
            y = _target_graph.nodes[n_uid]['y']
            node_geoms.append(geometry.Point(x, y))
    # if by highest_degree, then find the centroid of the highest degree nodes
    else:
        highest_degree = 0
        for n_uid in _node_group:
            if n_uid in _target_graph:
                if nx.degree(_target_graph, n_uid) > highest_degree:
                    highest_degree = nx.degree(_target_graph, n_uid)

        # aggregate the highest degree nodes
        node_geoms = []
        for n_uid in _node_group:
            if n_uid not in _target_graph:
                continue
            if nx.degree(_target_graph, n_uid) != highest_degree:
                continue
            x = _target_graph.nodes[n_uid]['x']
            y = _target_graph.nodes[n_uid]['y']
            # append geom
            node_geoms.append(geometry.Point(x, y))

    # find the new centroid
    c = geometry.MultiPoint(node_geoms).centroid
    _target_graph.add_node(_parent_node_name, x=c.x, y=c.y)

    # remove old nodes and reassign to new parent node
    # first determine new edges
    new_edge_data = []
    for uid in _node_group:
        for nb_uid in nx.neighbors(_target_graph, uid):
            # drop geoms between merged nodes
            # note that this will also drop self-loops
            if uid in _node_group and nb_uid in _node_group:
                continue
            else:
                if 'geom' not in _target_graph[uid][nb_uid]:
                    raise AttributeError(f'Missing "geom" attribute for edge {uid}-{nb_uid}')
                line_geom = _target_graph[uid][nb_uid]['geom']
                if line_geom.type != 'LineString':
                    raise TypeError(
                        f'Expecting LineString geometry but found {line_geom.type} geometry for edge {uid}-{nb_uid}.')
                # first orient geom in correct direction
                s_x = _target_graph.nodes[uid]['x']
                s_y = _target_graph.nodes[uid]['y']
                # check geom coordinates directionality - flip if facing backwards direction
                if not (s_x, s_y) == line_geom.coords[0][:2]:
                    line_geom = geometry.LineString(line_geom.coords[::-1])
                # double check that coordinates now face the forwards direction
                if not (s_x, s_y) == line_geom.coords[0][:2]:
                    raise AttributeError(f'Edge geometry endpoint coordinate mismatch for edge {uid}-{nb_uid}')
                # update geom starting point to new parent node's coordinates
                coords = list(line_geom.coords)
                coords[0] = (c.x, c.y)
                new_line_geom = geometry.LineString(coords)
                new_edge_data.append((_parent_node_name, nb_uid, new_line_geom))
    # remove the nodes from the target graph, this will also implicitly drop related edges
    _target_graph.remove_nodes_from(_node_group)
    # add the edges
    for s, e, geom in new_edge_data:
        # when dealing with a collapsed linestring, this should be a rare occurance
        if geom.length == 0:
            logger.warning(f'Encountered a geom of length 0m: check edge {s}-{e}.')
            continue
        # don't add edge duplicates from respectively merged nodes
        if (s, e) not in _target_graph.edges():
            _target_graph.add_edge(s, e, geom=geom, length=geom.length, impedance=geom.length)
        # however, do add if substantially different geom...
        else:
            diff = _target_graph[s][e]['geom'].length / geom.length
            if abs(diff) > 1.25:
                _target_graph.add_edge(s, e, geom=geom, length=geom.length, impedance=geom.length)

    return _target_graph


def _create_strtree(_graph: nx.Graph) -> strtree.STRtree:
    # create an STRtree
    points = []
    for n, n_d in _graph.nodes(data=True):
        # x coordinate
        if 'x' not in _graph.nodes[n]:
            raise AttributeError(f'Encountered node missing "x" coordinate attribute at node {n}.')
        x = _graph.nodes[n]['x']
        # y coordinate
        if 'y' not in _graph.nodes[n]:
            raise AttributeError(f'Encountered node missing "y" coordinate attribute at node {n}.')
        y = _graph.nodes[n]['y']
        p = geometry.Point(x, y)
        p.uid = n
        points.append(p)
    return strtree.STRtree(points)


def nX_consolidate_spatial(networkX_graph: nx.Graph, buffer_dist: float = 14) -> nx.Graph:
    if not isinstance(networkX_graph, nx.Graph):
        raise TypeError('This method requires an undirected networkX graph.')

    logger.info(f'Consolidating network by distance buffer.')
    g_copy = networkX_graph.copy()

    # create an STRtree
    tree = _create_strtree(networkX_graph)

    # setup template for new node names
    n_n_template = uuid.uuid4().hex.upper()[0:3]
    n_n_count = 0

    # keep track of removed nodes
    removed_nodes = set()

    # iterate origin graph, remove overlapping nodes within buffer, replace with new
    for n, n_d in tqdm(networkX_graph.nodes(data=True), disable=tqdm_suppress):
        # skip if already consolidated from an adjacent node
        if n in removed_nodes:
            continue
        # get all other nodes within buffer distance
        js = tree.query(geometry.Point(n_d['x'], n_d['y']).buffer(buffer_dist))
        # if only self-node, then continue
        if len(js) <= 1:
            continue

        # new parent node name - only used if match found
        parent_node_name = None
        # keep track of the uids to be consolidated
        node_group = set()

        # iterate geoms within buffer
        # this includes the self-node, hence no special logic to handle
        for j in js:
            if j.uid in removed_nodes:
                continue
            # initialise the parent node name, if necessary
            if parent_node_name is None:
                parent_node_name = f'{n_n_template}_{n_n_count}'
                n_n_count += 1
            # if not already in the removed nodes, go ahead and add the point
            node_group.add(j.uid)
            # add to the removed_nodes dict and point to new parent node uid
            removed_nodes.add(j.uid)

        if not node_group:
            continue

        g_copy = _dissolve_adjacent(g_copy, parent_node_name, node_group, highest_degree=True)

    return g_copy


def _find_parallel(_networkX_graph: nx.Graph, _line_start_nd, _line_end_nd, _parallel_nd, _buffer_dist):

    line_geom = _networkX_graph[_line_start_nd][_line_end_nd]['geom']
    p_x = _networkX_graph.nodes[_parallel_nd]['x']
    p_y = _networkX_graph.nodes[_parallel_nd]['y']
    parallel_point = geometry.Point(p_x, p_y)

    # returns tuple of nearest from respective input geom
    # want the nearest point on the line at index 1
    nearest_point = ops.nearest_points(parallel_point, line_geom)[1]

    # check if the distance from the parallel point to the nearest point is within buffer distance
    if parallel_point.distance(nearest_point) > _buffer_dist:
        return None

    # in some cases the line will be pointing away, but is still short enough to be within max
    # in these cases, check that the closest point is not actually the start of the line geom (or very near to it)
    s_x = _networkX_graph.nodes[_line_start_nd]['x']
    s_y = _networkX_graph.nodes[_line_start_nd]['y']
    line_start_point = geometry.Point(s_x, s_y)
    if nearest_point.distance(line_start_point) < 1:
        return None

    # if a valid nearest point has been found, go ahead and split the geom
    # use a snap because rounding precision errors will otherwise cause issues
    split_geoms = ops.split(ops.snap(line_geom, nearest_point, 0.01), nearest_point)
    # if (relatively rare) the split is still not successful
    if len(split_geoms) != 2:
        logger.warning(f'Attempt to split line geom for {_line_start_nd}-{_line_end_nd} did not return two geoms: '
                       f'{split_geoms}')
        return None
    # otherwise, unpack the geoms
    part_a, part_b = split_geoms
    # generate a new node name by concatenating the source nodes
    new_nd_name = f'{_line_start_nd}_{_line_end_nd}'
    _networkX_graph.add_node(new_nd_name, x=nearest_point.x, y=nearest_point.y)
    _networkX_graph.add_edge(_line_start_nd, new_nd_name)
    _networkX_graph.add_edge(_line_end_nd, new_nd_name)
    if (s_x, s_y) == part_a.coords[0][:2] or (s_x, s_y) == part_a.coords[-1][:2]:
        _networkX_graph[_line_start_nd][new_nd_name]['geom'] = part_a
        _networkX_graph[_line_start_nd][new_nd_name]['length'] = part_a.length
        _networkX_graph[_line_start_nd][new_nd_name]['impedance'] = part_a.length
        _networkX_graph[_line_end_nd][new_nd_name]['geom'] = part_b
        _networkX_graph[_line_end_nd][new_nd_name]['length'] = part_b.length
        _networkX_graph[_line_end_nd][new_nd_name]['impedance'] = part_b.length
    else:
        # double check matching geoms
        if (s_x, s_y) != part_b.coords[0][:2] and (s_x, s_y) != part_b.coords[-1][:2]:
            raise ValueError('Unable to match split geoms to existing nodes')
        _networkX_graph[_line_start_nd][new_nd_name]['geom'] = part_b
        _networkX_graph[_line_start_nd][new_nd_name]['length'] = part_b.length
        _networkX_graph[_line_start_nd][new_nd_name]['impedance'] = part_b.length
        _networkX_graph[_line_end_nd][new_nd_name]['geom'] = part_a
        _networkX_graph[_line_end_nd][new_nd_name]['length'] = part_a.length
        _networkX_graph[_line_end_nd][new_nd_name]['impedance'] = part_a.length

    # the existing edge should be removed later to avoid in-place errors during loop cycle
    # also return the parallel point and the newly paired parallel node
    return (_line_start_nd, _line_end_nd), (_parallel_nd, new_nd_name)


def nX_consolidate_parallel(networkX_graph: nx.Graph, buffer_dist: float = 14) -> nx.Graph:
    if not isinstance(networkX_graph, nx.Graph):
        raise TypeError('This method requires an undirected networkX graph.')

    logger.info(f'Consolidating network by parallel edges.')
    g_copy = networkX_graph.copy()

    # create an STRtree
    tree = _create_strtree(networkX_graph)

    # setup template for new node names
    n_n_template = uuid.uuid4().hex.upper()[0:3]
    n_n_count = 0

    # keep track of removed nodes
    removed_nodes = set()

    # keep track of manually split node locations for post-processing
    merge_pairs = []

    # iterate origin graph
    for n, n_d in tqdm(networkX_graph.nodes(data=True), disable=tqdm_suppress):
        # skip if already consolidated from an adjacent node
        if n in removed_nodes:
            continue
        # get all other nodes within buffer distance
        js = tree.query(geometry.Point(n_d['x'], n_d['y']).buffer(buffer_dist))
        # if only self-node, then continue
        if len(js) <= 1:
            continue

        # new parent node name - only used if match found
        parent_node_name = None
        # keep track of the uids to be consolidated
        node_group = set()
        # delay removals until after each iteration of loop to avoid in-place modification errors
        removals = []

        # iterate each node's neighbours
        # check if any of the neighbour's neighbours are within the buffer distance of other direct neighbours
        # if so, parallel set of edges may have been found
        nbs = list(nx.neighbors(g_copy, n))
        for j_point in js:
            j = j_point.uid
            # ignore self-node
            if j == n:
                continue
            # only review if not already in the removed nodes,
            if j in removed_nodes:
                continue
            # matching can happen in one of several situations, so use a flag
            matched = False
            # cross check n's neighbours against j's neighbours
            # if they have respective neighbours within buffer dist of each other, then merge
            j_nbs = list(nx.neighbors(g_copy, j))
            for n_nb in nbs:
                # skip this neighbour if already removed
                if n_nb in removed_nodes:
                    continue
                # if j is a direct neighbour, then ignore
                if n_nb == j:
                    continue
                # get the n node neighbour and create a point
                n_nb_point = geometry.Point(g_copy.nodes[n_nb]['x'], g_copy.nodes[n_nb]['y'])
                # compare against j node neighbours
                for j_nb in j_nbs:
                    # skip this neighbour if already removed
                    if j_nb in removed_nodes:
                        continue
                    # don't match against origin node
                    if j_nb == n:
                        continue
                    # if the respective neighbours are the same node, then match
                    if n_nb == j_nb:
                        matched = True
                        break
                    # otherwise, get the j node neighbour and create a point
                    j_nb_point = geometry.Point(g_copy.nodes[j_nb]['x'], g_copy.nodes[j_nb]['y'])
                    # check whether the neighbours are within the buffer distance of each other
                    if n_nb_point.distance(j_nb_point) < buffer_dist:
                        matched = True
                        break
                    # if not, then check along length of lines
                    # this is necessary for situations where certain lines are broken by other links
                    # i.e. where nodes are out of lock-step
                    # check first for j_nb point against n - n_nb line geom
                    response = _find_parallel(g_copy, n, n_nb, j_nb, buffer_dist)
                    if response is not None:
                        removal_pair, merge_pair = response
                        removals.append(removal_pair)
                        merge_pairs.append(merge_pair)
                        matched = True
                        break
                    # similarly check for n_nb point against j - j_nb line geom
                    response = _find_parallel(g_copy, j, j_nb, n_nb, buffer_dist)
                    if response is not None:
                        removal_pair, merge_pair = response
                        removals.append(removal_pair)
                        merge_pairs.append(merge_pair)
                        matched = True
                        break

                # break out if match found
                if matched:
                    break

            # if successful match, go ahead and add a new parent node, and merge n and j
            if matched:
                if parent_node_name is None:
                    parent_node_name = f'{n_n_template}_{n_n_count}'
                    n_n_count += 1
                node_group.update([n, j])
                removed_nodes.update([n, j])

        for s, e in removals:
            # in some cases, the edge may not exist anymore
            if (s, e) in g_copy.edges():
                g_copy.remove_edge(s, e)

        if not node_group:
            continue

        g_copy = _dissolve_adjacent(g_copy, parent_node_name, node_group)

    for pair in merge_pairs:
        # in some cases one of the pair of nodes may not exist anymore
        if pair[0] in g_copy and pair[1] in g_copy:
            parent_node_name = f'{n_n_template}_{n_n_count}'
            n_n_count += 1
            g_copy = _dissolve_adjacent(g_copy, parent_node_name, pair)

    return g_copy


def nX_decompose(networkX_graph: nx.Graph, decompose_max: float) -> nx.Graph:
    if not isinstance(networkX_graph, nx.Graph):
        raise TypeError('This method requires an undirected networkX graph.')

    logger.info(f'Decomposing graph to maximum edge lengths of {decompose_max}.')
    g_copy = networkX_graph.copy()

    # note -> write to a duplicated graph to avoid in-place errors
    for s, e, d in tqdm(networkX_graph.edges(data=True), disable=tqdm_suppress):
        # test for x, y in start coordinates
        if 'x' not in networkX_graph.nodes[s] or 'y' not in networkX_graph.nodes[s]:
            raise AttributeError(f'Encountered node missing "x" or "y" coordinate attributes at node {s}.')
        # test for x, y in end coordinates
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
            g_copy.add_edge(prior_node_id, new_node_id, length=l, impedance=l, geom=line_segment)
            # increment the step and node id
            prior_node_id = new_node_id
            step += step_size
        # set the last edge manually to avoid rounding errors at end of LineString
        # the nodes already exist, so just add edge
        line_segment = substring(line_geom, step, line_geom.length)
        l = line_segment.length
        g_copy.add_edge(prior_node_id, e, length=l, impedance=l, geom=line_segment)

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
                f'No edge geom found for edge {a_node}-{b_node}: '
                f'Please add an edge "geom" attribute consisting of a shapely LineString.')
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
    for s, e, d in tqdm(networkX_graph.edges(data=True), disable=tqdm_suppress):

        # get the first and second half geoms
        s_half_geom, e_half_geom = get_half_geoms(networkX_graph, s, e)

        # create a new dual node corresponding to the current primal edge
        s_e = sorted([str(s), str(e)])
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
                s_nb = sorted([str(n_side), str(nb)])
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
                        f'Found {merged_line.type} geometry instead of "LineString" for new geom {merged_line.wkt}. '
                        f'Check that the LineStrings for {s}-{e} and {n_side}-{nb} actually touch.')

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
    for s, e, d in tqdm(g_copy.edges(data=True), disable=tqdm_suppress):
        if 'geom' not in d:
            raise AttributeError(
                f'No edge geom found for edge {s}-{e}: '
                f'Please add an edge "geom" attribute consisting of a shapely LineString.')
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

    for n in tqdm(g_copy.nodes(), disable=tqdm_suppress):
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
    for n in tqdm(g_copy.nodes(), disable=tqdm_suppress):
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
    for n, d in tqdm(g_copy.nodes(data=True), disable=tqdm_suppress):
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
                    f'Node uid {uid} not found in graph. '
                    f'If passing a graph as backbone, the uids must match those supplied with the node and edge maps.')
    else:
        logger.info('No existing graph found, creating new.')
        g_copy = nx.Graph()
        for uid in node_uids:
            g_copy.add_node(uid)

    logger.info('Unpacking node data.')
    for uid, node in tqdm(zip(node_uids, node_map), disable=tqdm_suppress):
        x, y, live, edge_idx, wt = node
        g_copy.nodes[uid]['x'] = x
        g_copy.nodes[uid]['y'] = y
        g_copy.nodes[uid]['live'] = bool(live)
        g_copy.nodes[uid]['weight'] = wt

    logger.info('Unpacking edge data.')
    for edge in tqdm(edge_map, disable=tqdm_suppress):
        start, end, length, impedance = edge
        start_uid = node_uids[int(start)]
        end_uid = node_uids[int(end)]
        # networkX will silently add new edges / data over existing edges
        g_copy.add_edge(start_uid, end_uid, length=length, impedance=impedance)

    if metrics_dict is not None:
        logger.info('Unpacking metrics to nodes.')
        for uid, metrics in tqdm(metrics_dict.items(), disable=tqdm_suppress):
            if uid not in g_copy:
                raise AttributeError(
                    f'Node uid {uid} not found in graph. '
                    f'Data dictionary uids must match those supplied with the node and edge maps.')
            g_copy.nodes[uid]['metrics'] = metrics

    return g_copy
