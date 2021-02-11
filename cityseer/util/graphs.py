'''
General graph manipulation
'''
import json
import logging
from typing import Union, Tuple

import networkx as nx
from numba import types
from numba.typed import Dict
import numpy as np
from shapely import geometry, ops, strtree, coords
from tqdm.auto import tqdm
import utm

from cityseer.algos import checks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def nX_simple_geoms(networkX_multigraph: nx.MultiGraph) -> nx.MultiGraph:
    """
    Adds straight linestrings between coordinates of nodes. Useful for converting OSM graphs to geometry.
    """
    if not isinstance(networkX_multigraph, nx.MultiGraph):
        raise TypeError('This method requires an undirected networkX MultiGraph.')
    logger.info('Generating simple (straight) edge geometries.')
    g_multi_copy = networkX_multigraph.copy()
    # unpack coordinates and build simple edge geoms
    for s, e, k in tqdm(g_multi_copy.edges(keys=True), disable=checks.quiet_mode):
        # start x coordinate
        if 'x' not in g_multi_copy.nodes[s]:
            raise KeyError(f'Encountered node missing "x" coordinate attribute at node {s}.')
        s_x = g_multi_copy.nodes[s]['x']
        # start y coordinate
        if 'y' not in g_multi_copy.nodes[s]:
            raise KeyError(f'Encountered node missing "y" coordinate attribute at node {s}.')
        s_y = g_multi_copy.nodes[s]['y']
        # end x coordinate
        if 'x' not in g_multi_copy.nodes[e]:
            raise KeyError(f'Encountered node missing "x" coordinate attribute at node {e}.')
        e_x = g_multi_copy.nodes[e]['x']
        # end y coordinate
        if 'y' not in g_multi_copy.nodes[e]:
            raise KeyError(f'Encountered node missing "y" coordinate attribute at node {e}.')
        e_y = g_multi_copy.nodes[e]['y']
        g_multi_copy[s][e][k]['geom'] = geometry.LineString([[s_x, s_y], [e_x, e_y]])

    return g_multi_copy


def nX_from_osm(osm_json) -> nx.MultiGraph:
    """
    Parses an osm response into a NetworkX graph.
    """
    osm_network_data = json.loads(osm_json)
    G = nx.MultiGraph()
    for e in osm_network_data['elements']:
        if e['type'] == 'node':
            G.add_node(e['id'], x=e['lon'], y=e['lat'])
    for e in osm_network_data['elements']:
        if e['type'] == 'way':
            count = len(e['nodes'])
            for idx in range(count - 1):
                G.add_edge(e['nodes'][idx], e['nodes'][idx + 1])

    return G


def nX_wgs_to_utm(networkX_multigraph: nx.MultiGraph, force_zone_number=None) -> nx.MultiGraph:
    """
    Converts a WGS CRS graph to a local UTM CRS.
    """
    if not isinstance(networkX_multigraph, nx.MultiGraph):
        raise TypeError('This method requires an undirected networkX MultiGraph.')
    logger.info('Converting networkX graph from WGS to UTM.')
    g_multi_copy = networkX_multigraph.copy()
    zone_number = None
    if force_zone_number is not None:
        zone_number = force_zone_number
    logger.info('Processing node x, y coordinates.')
    for n, d in tqdm(g_multi_copy.nodes(data=True), disable=checks.quiet_mode):
        # x coordinate
        if 'x' not in d:
            raise KeyError(f'Encountered node missing "x" coordinate attribute at node {n}.')
        lng = d['x']
        # y coordinate
        if 'y' not in d:
            raise KeyError(f'Encountered node missing "y" coordinate attribute at node {n}.')
        lat = d['y']
        # check for unintentional use of conversion
        if abs(lng) > 180 or abs(lat) > 90:
            raise ValueError('x, y coordinates exceed WGS bounds. Please check your coordinate system.')
        # to avoid issues across UTM boundaries, use the first point to set (and subsequently force) the UTM zone
        if zone_number is None:
            zone_number = utm.from_latlon(lat, lng)[2]  # zone number is position 2
        # be cognisant of parameter and return order
        # returns in easting, northing order
        easting, northing = utm.from_latlon(lat, lng, force_zone_number=zone_number)[:2]
        # write back to graph
        g_multi_copy.nodes[n]['x'] = easting
        g_multi_copy.nodes[n]['y'] = northing
    # if line geom property provided, then convert as well
    logger.info('Processing edge geom coordinates, if present.')
    for s, e, k, d in tqdm(g_multi_copy.edges(data=True, keys=True), disable=checks.quiet_mode):
        # check if geom present - optional step
        if 'geom' in d:
            line_geom = d['geom']
            if line_geom.type != 'LineString':
                raise TypeError(f'Expecting LineString geometry but found {line_geom.type} geometry.')
            # be cognisant of parameter and return order
            # returns in easting, northing order
            utm_coords = [utm.from_latlon(lat, lng, force_zone_number=zone_number)[:2] for lng, lat in line_geom.coords]
            # write back to edge
            g_multi_copy[s][e][k]['geom'] = geometry.LineString(utm_coords)

    return g_multi_copy


def nX_remove_dangling_nodes(networkX_multigraph: nx.MultiGraph,
                             despine: float = None,
                             remove_disconnected: bool = True) -> nx.MultiGraph:
    """
    Strips out all dead-end edges that are shorter than the despine parameter.
    Removes disconnected components remove_disconnected is set to True.
    """
    logger.info(f'Removing dangling nodes.')
    g_multi_copy = networkX_multigraph.copy()
    if remove_disconnected:
        # finds connected components - this behaviour changed with networkx v2.4
        connected_components = list(nx.algorithms.components.connected_components(g_multi_copy))
        # sort by largest component
        g_nodes = sorted(connected_components, key=len, reverse=True)[0]
        # make a copy of the graph using the largest component
        g_multi_copy = nx.MultiGraph(g_multi_copy.subgraph(g_nodes))
    if despine:
        remove_nodes = []
        for n, d in tqdm(g_multi_copy.nodes(data=True), disable=checks.quiet_mode):
            if nx.degree(g_multi_copy, n) == 1:
                # only a single neighbour, so index-in directly and update at key = 0
                nb = list(nx.neighbors(g_multi_copy, n))[0]
                if g_multi_copy[n][nb][0]['geom'].length <= despine:
                    remove_nodes.append(n)
        g_multi_copy.remove_nodes_from(remove_nodes)

    return g_multi_copy


def _snap_linestring_idx(linestring_coords: Union[list, tuple, np.ndarray, coords.CoordinateSequence],
                         idx: int,
                         xy: Tuple[float, float]) -> list:
    """
    Snaps a LineString's coordinate at the specified index to the provided xy coordinate.
    """
    # check types
    if not isinstance(linestring_coords, (list, tuple, np.ndarray, coords.CoordinateSequence)):
        raise ValueError('Expecting a list, tuple, numpy array, or shapely LineString coordinate sequence.')
    linestring_coords = list(linestring_coords)
    # check that the index is either 0 or 1
    if idx not in [0, -1]:
        raise ValueError('Expecting either a start index of "0" or an end index of "-1"')
    # handle 3D
    coord = list(linestring_coords[idx])  # tuples don't support indexed assignment
    coord[:2] = xy
    linestring_coords[idx] = coord

    return linestring_coords


def _snap_linestring_startpoint(linestring_coords: Union[list, tuple, np.ndarray, coords.CoordinateSequence],
                                xy: Tuple[float, float]) -> list:
    """
    Snaps a LineString's start-point coordinate to a specified xy coordinate.
    """
    return _snap_linestring_idx(linestring_coords, 0, xy)


def _snap_linestring_endpoint(linestring_coords: Union[list, tuple, np.ndarray, coords.CoordinateSequence],
                              xy: Tuple[float, float]) -> list:
    """
    Snaps a LineString's end-point coordinate to a specified xy coordinate.
    """
    return _snap_linestring_idx(linestring_coords, -1, xy)


def _align_linestring_coords(linestring_coords: Union[list, tuple, np.ndarray, coords.CoordinateSequence],
                             xy: Tuple[float, float],
                             reverse: bool = False,
                             tolerance=checks.tolerance) -> list:
    """
    Aligns a LineString's coordinate order to either start or end at the xy coordinate within a given tolerance.
    If reverse=False the coordinate order will be aligned to start from the given xy coordinate.
    If reverse=True the coordinate order will be aligned to end at the given xy coordinate.
    """
    # check types
    if not isinstance(linestring_coords, (list, tuple, np.ndarray, coords.CoordinateSequence)):
        raise ValueError('Expecting a list, tuple, numpy array, or shapely LineString coordinate sequence.')
    linestring_coords = list(linestring_coords)
    # the target indices depend on whether reversed or not
    if not reverse:
        xy_idx = 0
        opposite_idx = -1
    else:
        xy_idx = -1
        opposite_idx = 0
    # flip if necessary
    if np.allclose(xy, linestring_coords[opposite_idx][:2], atol=tolerance, rtol=0):
        return linestring_coords[::-1]
    # if still not aligning, then there is an issue
    elif not np.allclose(xy, linestring_coords[xy_idx][:2], atol=tolerance, rtol=0):
        raise ValueError(f'Unable to align the LineString to starting point {xy} given the tolerance of {tolerance}.')
    # otherwise no flipping is required and the coordinates can simply be returned
    else:
        return linestring_coords


def _weld_linestring_coords(linestring_coords_a: Union[list, tuple, np.ndarray, coords.CoordinateSequence],
                            linestring_coords_b: Union[list, tuple, np.ndarray, coords.CoordinateSequence],
                            force_xy: Tuple[float, float] = None,
                            tolerance=checks.tolerance) -> list:
    """
    Takes two geometries, finds a matching start / end point combination and merges the coordinates accordingly.
    If the optional force_xy is provided then the weld will be performed at the xy end of the LineStrings.
    The force_xy parameter is useful for looping geometries or overlapping geometries:
    in these cases it can happen that welding works from either of the two ends, thus mis-aligning the start point.
    """
    # check types
    for lc in [linestring_coords_a, linestring_coords_b]:
        if not isinstance(lc, (list, tuple, np.ndarray, coords.CoordinateSequence)):
            raise ValueError('Expecting a list, tuple, numpy array, or shapely LineString coordinate sequence.')
    linestring_coords_a = list(linestring_coords_a)
    linestring_coords_b = list(linestring_coords_b)
    # if both lists are empty, raise
    if len(linestring_coords_a) == 0 and len(linestring_coords_b) == 0:
        raise ValueError('Neither of the provided linestring coordinate lists contain any coordinates.')
    # if one of the lists is empty, return only the other
    elif not len(linestring_coords_b):
        return linestring_coords_a
    elif not len(linestring_coords_a):
        return linestring_coords_b
    # match the directionality of the linestrings
    # if override_xy is provided, then make sure that the sides with the specified xy are merged
    # this is useful for looping components or overlapping components
    # i.e. where both the start and end points match an endpoint on the opposite line
    # in this case it is necessary to know which is the inner side of the weld and which is the outer endpoint
    if force_xy:
        if not np.allclose(linestring_coords_a[-1][:2], force_xy, atol=tolerance, rtol=0):
            coords_a = _align_linestring_coords(linestring_coords_a, force_xy, reverse=True)
        else:
            coords_a = linestring_coords_a
        if not np.allclose(linestring_coords_b[0][:2], force_xy, atol=tolerance, rtol=0):
            coords_b = _align_linestring_coords(linestring_coords_b, force_xy, reverse=False)
        else:
            coords_b = linestring_coords_b
    # case A: the linestring_b has to be flipped to start from x, y
    elif np.allclose(linestring_coords_a[-1][:2], linestring_coords_b[-1][:2], atol=tolerance, rtol=0):
        anchor_xy = linestring_coords_a[-1][:2]
        coords_a = linestring_coords_a
        coords_b = _align_linestring_coords(linestring_coords_b, anchor_xy)
    # case B: linestring_a has to be flipped to end at x, y
    elif np.allclose(linestring_coords_a[0][:2], linestring_coords_b[0][:2], atol=tolerance, rtol=0):
        anchor_xy = linestring_coords_a[0][:2]
        coords_a = _align_linestring_coords(linestring_coords_a, anchor_xy)
        coords_b = linestring_coords_b
    # case C: merge in the b -> a order (saves flipping both)
    elif np.allclose(linestring_coords_a[0][:2], linestring_coords_b[-1][:2], atol=tolerance, rtol=0):
        coords_a = linestring_coords_b
        coords_b = linestring_coords_a
    # case D: no further alignment is necessary
    else:
        coords_a = linestring_coords_a
        coords_b = linestring_coords_b
    # double check weld
    if not np.allclose(coords_a[-1][:2], coords_b[0][:2], atol=tolerance, rtol=0):
        raise ValueError(f'Unable to weld LineString geometries with the given tolerance of {tolerance}.')
    # drop the duplicate interleaving coordinate
    return coords_a[:-1] + coords_b


def nX_remove_filler_nodes(networkX_multigraph: nx.MultiGraph) -> nx.MultiGraph:
    """
    Iterates a networkX graph's nodes and removes nodes of degree = 2.
    The associated edges are removed and replaced with a new spliced geometry.
    """
    if not isinstance(networkX_multigraph, nx.MultiGraph):
        raise TypeError('This method requires an undirected networkX MultiGraph.')
    logger.info(f'Removing filler nodes.')
    g_multi_copy = networkX_multigraph.copy()
    removed_nodes = set()
    # iterates the original graph, but changes are written to the copied version (to avoid in-place snafus)
    for n in tqdm(networkX_multigraph.nodes(), disable=checks.quiet_mode):
        # some nodes will already have been removed
        if n in removed_nodes:
            continue
        # proceed if a "simple" node is discovered, i.e. degree = 2
        if nx.degree(networkX_multigraph, n) == 2:
            # pick the first neighbour and follow the chain until a non-simple node is encountered
            # this will become the starting point of the chain of simple nodes to be consolidated
            nb = list(nx.neighbors(networkX_multigraph, n))[0]
            # anchor_nd should be the first node of the chain of nodes to be merged, and should be a non-simple node
            anchor_nd = None
            # next_link_nd should be a direct neighbour of anchor_nd and must be a simple node
            next_link_nd = n
            # find the non-simple start node
            while anchor_nd is None:
                # follow the chain of neighbours and break once a non-simple node is found
                # catch disconnected looping components by checking for re-encountering start-node
                if nx.degree(networkX_multigraph, nb) != 2 or nb == n:
                    anchor_nd = nb
                    break
                # probe neighbours in one-direction only - i.e. don't backtrack
                nb_a, nb_b = list(nx.neighbors(networkX_multigraph, nb))
                if nb_a == next_link_nd:
                    next_link_nd = nb
                    nb = nb_b
                else:
                    next_link_nd = nb
                    nb = nb_a
            # from anchor_nd, proceed along the chain in the next_link_nd direction
            # accumulate and weld geometries along the way
            # break once finding another non-simple node
            trailing_nd = anchor_nd
            end_nd = None
            drop_nodes = []
            agg_geom = []
            while end_nd is None:
                # aggregate the geom
                try:
                    # the edge from trailing_nd to ext_link_nd is always singular i.e. index-in directly:
                    # simple nodes on one end or the other end implies a single-in and single-out for at least one end
                    assert networkX_multigraph.number_of_edges(trailing_nd, next_link_nd) == 1
                    geom = networkX_multigraph[trailing_nd][next_link_nd][0]['geom']
                except KeyError:
                    raise KeyError(f'Missing "geom" attribute for edge {trailing_nd}-{next_link_nd}')
                if geom.type != 'LineString':
                    raise TypeError(f'Expecting LineString geometry but found {geom.type} geometry.')
                # welds can be done automatically, but there are edge cases, e.g.:
                # looped roadways or overlapping edges such as stairways don't know which sides of two segments to join
                # i.e. in these cases the edges can sometimes be matched from one of two possible configurations
                # since the xy join is known for all cases it is used here regardless
                override_xy = (networkX_multigraph.nodes[trailing_nd]['x'], networkX_multigraph.nodes[trailing_nd]['y'])
                # weld
                agg_geom = _weld_linestring_coords(agg_geom, geom.coords, force_xy=override_xy)
                # if the next node has a degree other than 2, then break
                # for circular components, break if the next node matches the start node
                if nx.degree(networkX_multigraph, next_link_nd) != 2 or next_link_nd == anchor_nd:
                    end_nd = next_link_nd
                # otherwise, follow the chain
                else:
                    # add next_link_nd to drop list
                    drop_nodes.append(next_link_nd)
                    # get the next set of neighbours
                    nb_a, nb_b = list(nx.neighbors(networkX_multigraph, next_link_nd))
                    # proceed to the new_next node
                    if nb_a == trailing_nd:
                        trailing_nd = next_link_nd
                        next_link_nd = nb_b
                    else:
                        trailing_nd = next_link_nd
                        next_link_nd = nb_a
            # double-check that the geom's endpoints match within tolerance
            # then snap to remove any potential side-effects from minor tolerance issues
            s_xy = (networkX_multigraph.nodes[anchor_nd]['x'], networkX_multigraph.nodes[anchor_nd]['y'])
            if not np.allclose(agg_geom[0], s_xy, atol=checks.tolerance, rtol=0):
                raise ValueError('New Linestring geometry does not match starting node coordinates.')
            else:
                agg_geom = _snap_linestring_startpoint(agg_geom, s_xy)
            e_xy = (networkX_multigraph.nodes[end_nd]['x'], networkX_multigraph.nodes[end_nd]['y'])
            if not np.allclose(agg_geom[-1], e_xy, atol=checks.tolerance, rtol=0):
                raise ValueError('New Linestring geometry does not match ending node coordinates.')
            else:
                agg_geom = _snap_linestring_endpoint(agg_geom, e_xy)
            # create a new linestring
            new_geom = geometry.LineString(agg_geom)
            if new_geom.type != 'LineString':
                raise TypeError(
                    f'Found {new_geom.type} geometry instead of "LineString" for new geom {new_geom.wkt}.'
                    f'Check that the adjacent LineStrings in the vicinity of {n} are not corrupted.')
            # add a new edge from anchor_nd to end_nd
            g_multi_copy.add_edge(anchor_nd, end_nd, geom=new_geom)
            # drop the removed nodes, which will also implicitly drop the related edges
            g_multi_copy.remove_nodes_from(drop_nodes)
            removed_nodes.update(drop_nodes)

    return g_multi_copy


def _squash_adjacent(_multi_graph: nx.MultiGraph,
                     _node_group: Union[set, list, tuple],
                     _highest_degree: bool) -> nx.MultiGraph:
    """
    Squashes nodes from the node group down to a new node.
    The new node can either be based on the centroid of all nodes, else the centroid of the highest degree nodes.
    Modifies edge geoms accordingly.
    """
    if not isinstance(_multi_graph, nx.MultiGraph):
        raise TypeError('This method requires an undirected networkX multi-graph (for multiple edges).')
    # set the new centroid from the centroid of the node group's Multipoint:
    node_geoms = []
    node_names = []
    # if not using the high degree basis, then simply take the multipoint of all nodes
    if not _highest_degree:
        node_dupes = set()
        for n_uid in _node_group:
            node_names.append(n_uid)
            x = _multi_graph.nodes[n_uid]['x']
            y = _multi_graph.nodes[n_uid]['y']
            # in rare cases opposing geom splitting can cause overlaying nodes
            # these can swing the gravity of multipoint centroids so screen these out
            xy_key = f'{x}-{y}'
            if xy_key in node_dupes:
                continue
            else:
                node_dupes.add(xy_key)
            node_geoms.append(geometry.Point(x, y))
    # otherwise, identify all nodes of highest degree and use only their centroid
    else:
        highest_degree = 0
        for n_uid in _node_group:
            if n_uid in _multi_graph:
                if nx.degree(_multi_graph, n_uid) > highest_degree:
                    highest_degree = nx.degree(_multi_graph, n_uid)
        node_geoms = []
        for n_uid in _node_group:
            if n_uid in _multi_graph:
                if nx.degree(_multi_graph, n_uid) == highest_degree:
                    # append geom
                    x = _multi_graph.nodes[n_uid]['x']
                    y = _multi_graph.nodes[n_uid]['y']
                    node_geoms.append(geometry.Point(x, y))
                    node_names.append(n_uid)
    # find the new centroid
    c = geometry.MultiPoint(node_geoms).centroid
    # prepare the new node name
    new_node_names = []
    for new_name in node_names:
        new_name = str(new_name)
        new_name = ''.join([s for s in new_name if s.isalnum()])
        if len(new_name) < 4:
            new_node_names.append(new_name)
        else:
            new_node_names.append(f'{new_name[:2]}-{new_name[-2:]}')
    new_node_name = '|'.join(new_node_names)
    # add the new node
    _multi_graph.add_node(new_node_name, x=c.x, y=c.y)
    # iterate the nodes to be removed and connect their existing edge geometries to the new centroid's coordinate
    for uid in _node_group:
        # iterate the node's existing neighbours
        for nb_uid in nx.neighbors(_multi_graph, uid):
            # if a neighbour is also going to be dropped, then no need to create new between edges
            # an exception exists when a geom is looped, in which case the neighbour is also the current node
            if nb_uid in _node_group and nb_uid != uid:
                continue
            # MultiGraph - so iter edges
            for edge in _multi_graph[uid][nb_uid].values():
                if 'geom' not in edge:
                    raise KeyError(f'Missing "geom" attribute for edge {uid}-{nb_uid}')
                line_geom = edge['geom']
                if line_geom.type != 'LineString':
                    raise TypeError(
                        f'Expecting LineString geometry but found {line_geom.type} geometry for edge {uid}-{nb_uid}.')
                # orient the LineString so that the starting point matches the node's xy
                s_xy = (_multi_graph.nodes[uid]['x'], _multi_graph.nodes[uid]['y'])
                line_coords = _align_linestring_coords(line_geom.coords, s_xy)
                # update geom starting point to new parent node's coordinates
                line_coords = _snap_linestring_startpoint(line_coords, (c.x, c.y))
                # if self-loop, then the end also needs updating
                if uid == nb_uid:
                    line_coords = _snap_linestring_endpoint(line_coords, (c.x, c.y))
                    target_uid = new_node_name
                else:
                    target_uid = nb_uid
                # build the new geom
                new_line_geom = geometry.LineString(line_coords)
                # add to the list of new edges
                _multi_graph.add_edge(new_node_name, target_uid, geom=new_line_geom)
        # drop the node, this will also implicitly drop the old edges
        _multi_graph.remove_node(uid)

    return _multi_graph


def _merge_parallel_edges(_multi_graph: nx.MultiGraph,
                          use_midline: bool,
                          max_length_discrepancy_ratio: float) -> nx.MultiGraph:
    """
    Checks a multiGraph for duplicate edges, which are then consolidated.
    If use_midline is True, then the duplicates are replaced with a new edge following the merged centre-line.
    Otherwise, if use_midline is False, then the shortest of the edges is used and the others are simply dropped.
    In cases where one line is significantly longer than another (e.g. a crescent streets) and not merging by midline:
    then the longer edge is retained as separate if exceeding the max_length_discrepancy_ratio.
    """
    if not isinstance(_multi_graph, nx.MultiGraph):
        raise TypeError('This method requires an undirected networkX multi-graph (for multiple edges).')
    # don't use copy() - add nodes only
    deduped_graph = nx.MultiGraph()
    deduped_graph.add_nodes_from(_multi_graph.nodes(data=True))
    # iter the edges
    for s, e, d in tqdm(_multi_graph.edges(data=True), disable=checks.quiet_mode):
        # if only one edge is associated with this node pair, then add
        if _multi_graph.number_of_edges(s, e) == 1:
            deduped_graph.add_edge(s, e, **d)
        # otherwise, add if not already added from another (parallel) edge
        elif not deduped_graph.has_edge(s, e):
            # there are normally two edges, but sometimes three or possibly more
            edges = _multi_graph[s][e].values()
            # find the shortest of the geoms
            edge_geoms = [edge['geom'] for edge in edges]
            edge_lens = [geom.length for geom in edge_geoms]
            shortest_idx = edge_lens.index(min(edge_lens))
            shortest_len = edge_lens[shortest_idx]
            shortest_geom = edge_geoms.pop(shortest_idx)
            # retain distinct edges where additional length exceeds the max_length_discrepancy_ratio
            if max(edge_lens) > shortest_len * max_length_discrepancy_ratio:
                for idx, edge_len in enumerate(edge_lens):
                    if edge_len > shortest_len * max_length_discrepancy_ratio:
                        # add edge
                        g = edge_geoms[idx]
                        deduped_graph.add_edge(s, e, geom=g)
                        # remove from remainder
                        edge_geoms.pop(idx)
            # otherwise, if not merging on a midline basis
            # or, if no other edges to process (in cases where longer geom has been retained per above)
            elif not use_midline or len(edge_geoms) == 0:
                deduped_graph.add_edge(s, e, geom=shortest_geom)
            # otherwise weld the geoms, using the shortest as a yardstick
            else:
                # iterate the coordinates along the shorter geom
                new_coords = []
                for coord in shortest_geom.coords:
                    # from the current short_geom coordinate
                    short_point = geometry.Point(coord)
                    # find the nearest points on the longer geoms
                    multi_coords = [short_point]
                    for longer_geom in edge_geoms:
                        # get the nearest point on the longer geom
                        # returns a tuple of nearest geom for respective input geoms
                        longer_point = ops.nearest_points(short_point, longer_geom)[-1]
                        # aggregate
                        multi_coords.append(longer_point)
                    # create a midpoint between the geoms and add to the new coordinate array
                    mid_point = geometry.MultiPoint(multi_coords).centroid
                    new_coords.append(mid_point)
                # generate the new mid-line geom
                new_geom = geometry.LineString(new_coords)
                # add to the graph
                deduped_graph.add_edge(s, e, geom=new_geom)

    return deduped_graph


def _create_nodes_strtree(_graph: nx.MultiGraph) -> strtree.STRtree:
    points = []
    for n, d in _graph.nodes(data=True):
        # x coordinate
        if 'x' not in d:
            raise KeyError(f'Encountered node missing "x" coordinate attribute at node {n}.')
        x = d['x']
        # y coordinate
        if 'y' not in d:
            raise KeyError(f'Encountered node missing "y" coordinate attribute at node {n}.')
        y = d['y']
        p = geometry.Point(x, y)
        p.uid = n
        p.degree = nx.degree(_graph, n)
        points.append(p)
    return strtree.STRtree(points)


def _create_edges_strtree(_graph: nx.MultiGraph) -> strtree.STRtree:
    lines = []
    for s, e, d in _graph.edges(data=True):
        if 'geom' not in d:
            raise KeyError(f'Encountered edge missing "geom" attribute.')
        linestring = d['geom']
        linestring.start_uid = s
        linestring.end_uid = e
        lines.append(linestring)
    return strtree.STRtree(lines)


def nX_consolidate_spatial(networkX_multigraph: nx.MultiGraph,
                           buffer_dist: float = 5,
                           min_node_threshold: int = 2,
                           min_node_degree: int = 1,
                           min_cumulative_degree: int = None,
                           max_cumulative_degree: int = None,
                           neigh_policy: str = None,
                           merge_by_midline: bool = False,
                           squash_by_highest_degree: bool = False,
                           max_length_discrepancy_ratio: float = 1.2) -> nx.MultiGraph:
    """
    Consolidates nodes if they are within a buffer distance of each other.
    """
    if not isinstance(networkX_multigraph, nx.MultiGraph):
        raise TypeError('This method requires an undirected networkX MultiGraph.')
    if min_node_threshold < 2:
        raise ValueError('The minimum node threshold should be set to at least two.')
    if neigh_policy is not None and neigh_policy not in ('direct', 'indirect'):
        raise ValueError('Neighbour policy should be one of "direct" or "indirect".')
    logger.info(f'Consolidating network by distance buffer.')
    _multi_graph = networkX_multigraph.copy()
    # create a nodes STRtree
    nodes_tree = _create_nodes_strtree(_multi_graph)
    # keep track of removed nodes
    removed_nodes = set()
    # iterate origin graph (else node structure changes in place)
    for n, n_d in tqdm(networkX_multigraph.nodes(data=True), disable=checks.quiet_mode):
        # skip if already consolidated from an adjacent node
        if n in removed_nodes:
            continue
        # get all other nodes within buffer distance - the self-node is also returned
        js = nodes_tree.query(geometry.Point(n_d['x'], n_d['y']).buffer(buffer_dist))
        # extract the node uids if they've not already been removed and their degrees exceed min_node_degree
        node_group = []
        cumulative_degree = 0
        for j in js:
            if j.uid in removed_nodes or j.degree < min_node_degree:
                continue
            neighbours = nx.neighbors(networkX_multigraph, n)
            if neigh_policy == 'indirect' and j.uid in neighbours:
                continue
            elif neigh_policy == 'direct' and j.uid not in neighbours:
                continue
            node_group.append(j.uid)
            cumulative_degree += j.degree
        # check for min_node_threshold
        if len(node_group) < min_node_threshold:
            continue
        # check for cumulative degree threshold
        if min_cumulative_degree is not None and cumulative_degree < min_cumulative_degree:
            continue
        if max_cumulative_degree is not None and cumulative_degree > max_cumulative_degree:
            continue
        # update removed nodes
        removed_nodes.update(node_group)
        # consolidate if nodes have been identified within buffer and if these exceed min_node_threshold
        _multi_graph = _squash_adjacent(_multi_graph, node_group, squash_by_highest_degree)
    # squashing nodes can result in edge duplicates
    deduped_graph1 = _merge_parallel_edges(_multi_graph,
                                           merge_by_midline,
                                           max_length_discrepancy_ratio)
    deduped_graph2 = nX_remove_filler_nodes(deduped_graph1)

    return deduped_graph2


def nX_split_opposing_geoms(networkX_multigraph: nx.MultiGraph,
                            buffer_dist: float = 10,
                            use_midline: bool = False,
                            max_length_discrepancy_ratio: float = 1.2) -> nx.MultiGraph:
    if not isinstance(networkX_multigraph, nx.MultiGraph):
        raise TypeError('This method requires an undirected networkX MultiGraph.')

    def make_edge_key(s, e):
        return '-'.join(sorted([str(s), str(e)]))

    logger.info(f'Consolidating network by parallel edges.')
    _multi_graph = networkX_multigraph.copy()
    # create an edges STRtree (nodes and edges)
    edges_tree = _create_edges_strtree(_multi_graph)
    # where edges are deleted, keep track of new children edges
    edge_children = {}
    # iterate origin graph (else node structure changes in place)
    for n, n_d in tqdm(networkX_multigraph.nodes(data=True), disable=checks.quiet_mode):
        # don't split opposing geoms from nodes of degree 1
        if nx.degree(_multi_graph, n) < 2:
            continue
        # get all other edges within the buffer distance
        # the spatial index using bounding boxes, so further filtering is required (see further down)
        # furthermore, successive iterations may remove old edges, so keep track of removed parent vs new child edges
        n_point = geometry.Point(n_d['x'], n_d['y'])
        # spatial query from point returns all buffers with buffer_dist
        edges = edges_tree.query(n_point.buffer(buffer_dist))
        # extract the start node, end node, geom
        edges = [(edge.start_uid, edge.end_uid, edge) for edge in edges]
        # check against removed edges
        current_edges = []
        for s, e, edge_geom in edges:
            # if removed, use new children edges instead
            edge_key = make_edge_key(s, e)
            if edge_key in edge_children:
                current_edges += edge_children[edge_key]
            # otherwise it is safe to use original edge
            else:
                current_edges.append((s, e, edge_geom))
        # get neighbouring nodes from new graph
        neighbours = list(_multi_graph.neighbors(n))
        # abort if only direct neighbours
        if len(current_edges) <= len(neighbours):
            continue
        # prepare a list of out-bound edges
        out_edges_str = [make_edge_key(n, nb) for nb in neighbours]
        # filter current_edges
        gapped_edges = []
        for s, e, edge_geom in current_edges:
            # skip direct neighbours
            if make_edge_key(s, e) in out_edges_str:
                continue
            # check whether the geom is truly within the buffer distance
            if edge_geom.distance(n_point) > buffer_dist:
                continue
            gapped_edges.append((s, e, edge_geom))
        # abort if no gapped edges
        if not gapped_edges:
            continue
        # prepare the root node's point geom
        n_geom = geometry.Point(n_d['x'], n_d['y'])
        # iter gapped edges
        for s, e, edge_geom in gapped_edges:
            # see if start node is within buffer distance already
            s_nd_data = _multi_graph.nodes[s]
            s_nd_geom = geometry.Point(s_nd_data['x'], s_nd_data['y'])
            if s_nd_geom.distance(n_geom) <= buffer_dist:
                continue
            # likewise for end node
            e_nd_data = _multi_graph.nodes[e]
            e_nd_geom = geometry.Point(e_nd_data['x'], e_nd_data['y'])
            if e_nd_geom.distance(n_geom) <= buffer_dist:
                continue
            # otherwise, project a point and split the opposing geom
            # ops.nearest_points returns tuple of nearest from respective input geoms
            # want the nearest point on the line at index 1
            nearest_point = ops.nearest_points(n_geom, edge_geom)[-1]
            # if a valid nearest point has been found, go ahead and split the geom
            # use a snap because rounding precision errors will otherwise cause issues
            split_geoms = ops.split(ops.snap(edge_geom, nearest_point, 0.01), nearest_point)
            # in some cases the line will be pointing away, but is still near enough to be within max
            # in these cases a single geom will be returned
            if len(split_geoms) < 2:
                continue
            new_edge_geom_a, new_edge_geom_b = split_geoms
            # generate a new node name
            new_nd_name = f'{s}|{n}|{e}'
            # add the new node and edges to _multi_graph (don't modify networkX_multigraph because of iter in place)
            _multi_graph.add_node(new_nd_name, x=nearest_point.x, y=nearest_point.y)
            _multi_graph.add_edge(s, new_nd_name)
            _multi_graph.add_edge(e, new_nd_name)
            # TODO: more than one splice??
            if np.allclose(s_nd_geom.coords, new_edge_geom_a.coords[0][:2], atol=0.001, rtol=0) or \
                    np.allclose(s_nd_geom.coords, new_edge_geom_a.coords[-1][:2], atol=0.001, rtol=0):
                s_new_geom = new_edge_geom_a
                e_new_geom = new_edge_geom_b
            else:
                # double check matching geoms
                if not np.allclose(s_nd_geom.coords, new_edge_geom_b.coords[0][:2], atol=0.001, rtol=0) and \
                        not np.allclose(s_nd_geom.coords, new_edge_geom_b.coords[-1][:2], atol=0.001, rtol=0):
                    raise ValueError('Unable to match split geoms to existing nodes')
                s_new_geom = new_edge_geom_b
                e_new_geom = new_edge_geom_a
            # write the new edges
            _multi_graph[s][new_nd_name][0]['geom'] = s_new_geom
            _multi_graph[e][new_nd_name][0]['geom'] = e_new_geom
            # add the new edges to the edge_children dictionary
            edge_key = make_edge_key(s, e)
            edge_children[edge_key] = [(s, new_nd_name, s_new_geom),
                                       (e, new_nd_name, e_new_geom)]
            # TODO: more than one splice?
            # drop the old edge from _multi_graph
            if _multi_graph.has_edge(s, e):
                _multi_graph.remove_edge(s, e)
    # squashing nodes can result in edge duplicates
    deduped_graph = _merge_parallel_edges(_multi_graph,
                                          use_midline,
                                          max_length_discrepancy_ratio)

    return deduped_graph


def nX_decompose(networkX_multigraph: nx.MultiGraph, decompose_max: float) -> nx.MultiGraph:
    if not isinstance(networkX_multigraph, nx.MultiGraph):
        raise TypeError('This method requires an undirected networkX MultiGraph.')
    logger.info(f'Decomposing graph to maximum edge lengths of {decompose_max}.')
    g_multi_copy = networkX_multigraph.copy()
    # note -> write to a duplicated graph to avoid in-place errors
    for s, e, d in tqdm(networkX_multigraph.edges(data=True), disable=checks.quiet_mode):
        # test for x, y in start coordinates
        if 'x' not in networkX_multigraph.nodes[s] or 'y' not in networkX_multigraph.nodes[s]:
            raise KeyError(f'Encountered node missing "x" or "y" coordinate attributes at node {s}.')
        # test for x, y in end coordinates
        if 'x' not in networkX_multigraph.nodes[e] or 'y' not in networkX_multigraph.nodes[e]:
            raise KeyError(f'Encountered node missing "x" or "y" coordinate attributes at node {e}.')
        s_x = networkX_multigraph.nodes[s]['x']
        s_y = networkX_multigraph.nodes[s]['y']
        e_x = networkX_multigraph.nodes[e]['x']
        e_y = networkX_multigraph.nodes[e]['y']
        # test for geom
        if 'geom' not in d:
            raise KeyError(
                f'No edge geom found for edge {s}-{e}: Please add an edge "geom" attribute consisting of a shapely LineString.')
        # get edge geometry
        line_geom = d['geom']
        if line_geom.type != 'LineString':
            raise TypeError(f'Expecting LineString geometry but found {line_geom.type} geometry for edge {s}-{e}.')
        # check geom coordinates directionality - flip if facing backwards direction
        if not np.allclose((s_x, s_y), line_geom.coords[0][:2], atol=0.001, rtol=0):
            line_geom = geometry.LineString(line_geom.coords[::-1])
        # double check that coordinates now face the forwards direction
        if not np.allclose((s_x, s_y), line_geom.coords[0][:2], atol=0.001, rtol=0) or \
                not np.allclose((e_x, e_y), line_geom.coords[-1][:2], atol=0.001, rtol=0):
            raise ValueError(f'Edge geometry endpoint coordinate mismatch for edge {s}-{e}')
        # see how many segments are necessary so as not to exceed decomposition max distance
        # note that a length less than the decompose threshold will result in a single 'sub'-string
        n = np.ceil(line_geom.length / decompose_max)
        step_size = line_geom.length / n
        # since decomposing, remove the prior edge... but only after properties have been read
        g_multi_copy.remove_edge(s, e)
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
            line_segment = ops.substring(line_geom, step, step + step_size)
            # get the x, y of the new end node
            x, y = line_segment.coords[-1]
            # add the new node and edge
            g_multi_copy.add_node(new_node_id, x=x, y=y)
            # add and set live property if present in parent graph
            if 'live' in networkX_multigraph.nodes[s] and 'live' in networkX_multigraph.nodes[e]:
                live = True
                # if BOTH parents are not live, then set child to not live
                if not networkX_multigraph.nodes[s]['live'] and not networkX_multigraph.nodes[e]['live']:
                    live = False
                g_multi_copy.nodes[new_node_id]['live'] = live
            # add the edge
            # TODO: multi
            g_multi_copy.add_edge(prior_node_id, new_node_id, geom=line_segment)
            # increment the step and node id
            prior_node_id = new_node_id
            step += step_size
        # set the last edge manually to avoid rounding errors at end of LineString
        # the nodes already exist, so just add edge
        line_segment = ops.substring(line_geom, step, line_geom.length)
        l = line_segment.length
        g_multi_copy.add_edge(prior_node_id, e, geom=line_segment)

    return g_multi_copy


def nX_to_dual(networkX_multigraph: nx.MultiGraph) -> nx.MultiGraph:
    if not isinstance(networkX_multigraph, nx.MultiGraph):
        raise TypeError('This method requires an undirected networkX MultiGraph.')
    logger.info('Converting graph to dual.')
    g_dual = nx.MultiGraph()

    def get_half_geoms(g, a_node, b_node):
        '''
        For splitting and orienting half geoms
        '''
        # get edge data
        #TODO: convert to multi
        edge_data = g[a_node][b_node]
        # test for x coordinates
        if 'x' not in g.nodes[a_node] or 'y' not in g.nodes[a_node]:
            raise KeyError(f'Encountered node missing "x" or "y" coordinate attributes at node {a_node}.')
        # test for y coordinates
        if 'x' not in g.nodes[b_node] or 'y' not in g.nodes[b_node]:
            raise KeyError(f'Encountered node missing "x" or "y" coordinate attributes at node {b_node}.')
        a_x = g.nodes[a_node]['x']
        a_y = g.nodes[a_node]['y']
        b_x = g.nodes[b_node]['x']
        b_y = g.nodes[b_node]['y']
        # test for geom
        if 'geom' not in edge_data:
            raise KeyError(
                f'No edge geom found for edge {a_node}-{b_node}: '
                f'Please add an edge "geom" attribute consisting of a shapely LineString.')
        # get edge geometry
        line_geom = edge_data['geom']
        if line_geom.type != 'LineString':
            raise TypeError(
                f'Expecting LineString geometry but found {line_geom.type} geometry for edge {a_node}-{b_node}.')
        # check geom coordinates directionality - flip if facing backwards direction - beware 3d coords
        if not np.allclose((a_x, a_y), line_geom.coords[0][:2], atol=0.001, rtol=0):
            line_geom = geometry.LineString(line_geom.coords[::-1])
        # double check that coordinates now face the forwards direction
        if not np.allclose((a_x, a_y), line_geom.coords[0][:2], atol=0.001, rtol=0) or \
                not np.allclose((b_x, b_y), line_geom.coords[-1][:2], atol=0.001, rtol=0):
            raise ValueError(f'Edge geometry endpoint coordinate mismatch for edge {a_node}-{b_node}')
        # generate the two half geoms
        a_half_geom = ops.substring(line_geom, 0, line_geom.length / 2)
        b_half_geom = ops.substring(line_geom, line_geom.length / 2, line_geom.length)
        assert np.allclose(a_half_geom.coords[-1][:2], b_half_geom.coords[0][:2], atol=0.001, rtol=0)

        return a_half_geom, b_half_geom

    # iterate the primal graph's edges
    for s, e, d in tqdm(networkX_multigraph.edges(data=True), disable=checks.quiet_mode):

        # get the first and second half geoms
        s_half_geom, e_half_geom = get_half_geoms(networkX_multigraph, s, e)

        # create a new dual node corresponding to the current primal edge
        s_e = sorted([str(s), str(e)])
        hub_node_dual = f'{s_e[0]}_{s_e[1]}'
        x, y = s_half_geom.coords[-1][:2]
        g_dual.add_node(hub_node_dual, x=x, y=y)
        # add and set live property if present in parent graph
        if 'live' in networkX_multigraph.nodes[s] and 'live' in networkX_multigraph.nodes[e]:
            live = True
            # if BOTH parents are not live, then set child to not live
            if not networkX_multigraph.nodes[s]['live'] and not networkX_multigraph.nodes[e]['live']:
                live = False
            g_dual.nodes[hub_node_dual]['live'] = live

        # process either side
        for n_side, half_geom in zip([s, e], [s_half_geom, e_half_geom]):

            # add the spoke edges on the dual
            for nb in nx.neighbors(networkX_multigraph, n_side):

                # don't follow neighbour back to current edge combo
                if nb in [s, e]:
                    continue

                # get the near and far half geoms
                spoke_half_geom, _discard_geom = get_half_geoms(networkX_multigraph, n_side, nb)

                # add the neighbouring primal edge as dual node
                s_nb = sorted([str(n_side), str(nb)])
                spoke_node_dual = f'{s_nb[0]}_{s_nb[1]}'
                x, y = spoke_half_geom.coords[-1][:2]
                g_dual.add_node(spoke_node_dual, x=x, y=y)
                # add and set live property if present in parent graph
                if 'live' in networkX_multigraph.nodes[n_side] and 'live' in networkX_multigraph.nodes[nb]:
                    live = True
                    # if BOTH parents are not live, then set child to not live
                    if not networkX_multigraph.nodes[n_side]['live'] and not networkX_multigraph.nodes[nb]['live']:
                        live = False
                    g_dual.nodes[spoke_node_dual]['live'] = live

                # weld the lines
                merged_line = ops.linemerge([half_geom, spoke_half_geom])
                if merged_line.type != 'LineString':
                    raise TypeError(
                        f'Found {merged_line.type} geometry instead of "LineString" for new geom {merged_line.wkt}. '
                        f'Check that the LineStrings for {s}-{e} and {n_side}-{nb} actually touch.')

                # add the dual edge
                g_dual.add_edge(hub_node_dual, spoke_node_dual, parent_primal_node=n_side, geom=merged_line)

    return g_dual


def graph_maps_from_nX(networkX_multigraph: nx.MultiGraph) -> Tuple[tuple, np.ndarray, np.ndarray, Dict]:
    '''
    Strategic decisions because of too many edge cases:
    - decided to not discard disconnected components to avoid unintended consequences
    - no internal simplification - use prior methods or tools to clean or simplify the graph before calling this method
    - length and angle now set automatically inside this method because in and out bearing are set here regardless.
    - returns node_data, edge_data, a map from nodes to edges
    '''

    if not isinstance(networkX_multigraph, nx.MultiGraph):
        raise TypeError('This method requires an undirected networkX MultiGraph.')
    logger.info('Preparing node and edge arrays from networkX graph.')
    g_multi_copy = networkX_multigraph.copy()
    logger.info('Preparing graph')
    total_out_degrees = 0
    for n in tqdm(g_multi_copy.nodes(), disable=checks.quiet_mode):
        # writing node identifier to 'labels' in case conversion to integers method interferes with order
        g_multi_copy.nodes[n]['label'] = n
        # sum edges
        for nb in g_multi_copy.neighbors(n):
            total_out_degrees += 1
    logger.info('Generating data arrays')
    # convert the nodes to sequential - this permits implicit indices with benefits to speed and structure
    g_multi_copy = nx.convert_node_labels_to_integers(g_multi_copy, 0)
    # prepare the node and edge maps
    node_uids = []
    # float - for consistency - requires higher accuracy for x, y work
    node_data = np.full((g_multi_copy.number_of_nodes(), 3), np.nan, dtype=np.float64)
    # float - allows for nan and inf - float32 should be ample...
    edge_data = np.full((total_out_degrees, 7), np.nan, dtype=np.float32)
    # nodes have a one-to-many mapping to edges
    node_edge_map = Dict.empty(
        key_type=types.int64,
        value_type=types.int64[:]
    )
    edge_idx = 0
    # populate the nodes
    for n, d in tqdm(g_multi_copy.nodes(data=True), disable=checks.quiet_mode):
        # label
        # don't cast to string because otherwise correspondence between original and round-trip graph indices is lost
        node_uids.append(d['label'])
        # cast to int for indexing
        node_idx = int(n)

        # NODE MAP INDEX POSITION 0 = x coordinate
        if 'x' not in d:
            raise KeyError(f'Encountered node missing "x" coordinate attribute at node {n}.')
        node_data[node_idx][0] = d['x']

        # NODE MAP INDEX POSITION 1 = y coordinate
        if 'y' not in d:
            raise KeyError(f'Encountered node missing "y" coordinate attribute at node {n}.')
        node_data[node_idx][1] = d['y']

        # NODE MAP INDEX POSITION 2 = live or not
        if 'live' in d:
            node_data[node_idx][2] = d['live']
        else:
            node_data[node_idx][2] = True

        # build edges
        out_edges = []
        for nb in g_multi_copy.neighbors(n):
            # add the new edge index to the node's out edges
            out_edges.append(edge_idx)
            # EDGE MAP INDEX POSITION 0 = start node
            edge_data[edge_idx][0] = node_idx
            # EDGE MAP INDEX POSITION 1 = end node
            edge_data[edge_idx][1] = nb
            # get edge data
            edge = g_multi_copy[node_idx][nb]
            # EDGE MAP INDEX POSITION 2 = length
            if not 'geom' in edge:
                raise KeyError(
                    f'No edge geom found for edge {node_idx}-{nb}: '
                    f'Please add an edge "geom" attribute consisting of a shapely LineString.'
                    f'Simple (straight) geometries can be inferred automatically through use of the nX_simple_geoms() method.')
            line_geom = edge['geom']
            if line_geom.type != 'LineString':
                raise TypeError(
                    f'Expecting LineString geometry but found {line_geom.type} geometry for edge {node_idx}-{nb}.')
            # cannot have zero or negative length - division by zero
            l = line_geom.length
            if not np.isfinite(l) or l <= 0:
                raise ValueError(f'Length attribute {l} for edge {node_idx}-{nb} must be a finite positive value.')
            edge_data[edge_idx][2] = l
            # EDGE MAP INDEX POSITION 3 = angle_sum
            # check geom coordinates directionality (for bearings at index 5 / 6)
            # flip if facing backwards direction
            s_x, s_y = node_data[node_idx][:2]
            if not np.allclose((s_x, s_y), line_geom.coords[0][:2], atol=0.001, rtol=0):
                line_geom = geometry.LineString(line_geom.coords[::-1])
            e_x, e_y = (g_multi_copy.nodes[nb]['x'], g_multi_copy.nodes[nb]['y'])
            # double check that coordinates now face the forwards direction
            if not np.allclose((s_x, s_y), line_geom.coords[0][:2]) or \
                    not np.allclose((e_x, e_y), line_geom.coords[-1][:2], atol=0.001, rtol=0):
                raise ValueError(f'Edge geometry endpoint coordinate mismatch for edge {node_idx}-{nb}')
            # iterate the coordinates and calculate the angular change
            angle_sum = 0
            for c in range(len(line_geom.coords) - 2):
                x_1, y_1 = line_geom.coords[c][:2]
                x_2, y_2 = line_geom.coords[c + 1][:2]
                x_3, y_3 = line_geom.coords[c + 2][:2]
                # arctan2 is y / x order
                a_1 = np.rad2deg(np.arctan2(y_2 - y_1, x_2 - x_1))
                a_2 = np.rad2deg(np.arctan2(y_3 - y_2, x_3 - x_2))
                angle_sum += np.abs((a_2 - a_1 + 180) % 360 - 180)
                # alternative
                # A = np.array(merged_line.coords[c + 1]) - np.array(merged_line.coords[c])
                # B = np.array(merged_line.coords[c + 2]) - np.array(merged_line.coords[c + 1])
                # angle = np.abs(np.degrees(np.math.atan2(np.linalg.det([A, B]), np.dot(A, B))))
            if not np.isfinite(angle_sum) or angle_sum < 0:
                raise ValueError(
                    f'Angle-sum attribute {angle_sum} for edge {node_idx}-{nb} must be a finite positive value.')
            edge_data[edge_idx][3] = angle_sum
            # EDGE MAP INDEX POSITION 4 = imp_factor
            # if imp_factor is set explicitly, then use
            if 'imp_factor' in edge:
                # cannot have imp_factor less than zero (but == 0 is OK)
                imp_factor = edge['imp_factor']
                if not (np.isfinite(imp_factor) or np.isinf(imp_factor)) or imp_factor < 0:
                    raise ValueError(
                        f'Impedance factor: {imp_factor} for edge {node_idx}-{nb} must be a finite positive value or positive infinity.')
                edge_data[edge_idx][4] = imp_factor
            else:
                # fallback imp_factor of 1
                edge_data[edge_idx][4] = 1
            # EDGE MAP INDEX POSITION 5 - in bearing
            x_1, y_1 = line_geom.coords[0][:2]
            x_2, y_2 = line_geom.coords[1][:2]
            edge_data[edge_idx][5] = np.rad2deg(np.arctan2(y_2 - y_1, x_2 - x_1))
            # EDGE MAP INDEX POSITION 6 - out bearing
            x_1, y_1 = line_geom.coords[-2][:2]
            x_2, y_2 = line_geom.coords[-1][:2]
            edge_data[edge_idx][6] = np.rad2deg(np.arctan2(y_2 - y_1, x_2 - x_1))
            # increment the edge_idx
            edge_idx += 1
        # add the node to the node_edge_map
        node_edge_map[node_idx] = np.array(out_edges, dtype='int64')

    return tuple(node_uids), node_data, edge_data, node_edge_map


def nX_from_graph_maps(node_uids: Union[tuple, list],
                       node_data: np.ndarray,
                       edge_data: np.ndarray,
                       node_edge_map: Dict,
                       networkX_multigraph: nx.MultiGraph = None,
                       metrics_dict: dict = None) -> nx.MultiGraph:
    logger.info('Populating node and edge map data to a networkX graph.')

    if networkX_multigraph is not None:
        logger.info('Reusing existing graph as backbone.')
        if networkX_multigraph.number_of_nodes() != len(node_data):
            raise ValueError('The number of nodes in the graph does not match the number of nodes in the node map.')
        g_multi_copy = networkX_multigraph.copy()
        for uid in node_uids:
            if uid not in g_multi_copy:
                raise KeyError(
                    f'Node uid {uid} not found in graph. '
                    f'If passing a graph as backbone, the uids must match those supplied with the node and edge maps.')
    else:
        logger.info('No existing graph found, creating new.')
        g_multi_copy = nx.MultiGraph()
        for uid in node_uids:
            g_multi_copy.add_node(uid)

    # after above so that errors caught first
    checks.check_network_maps(node_data, edge_data, node_edge_map)

    logger.info('Unpacking node data.')
    for uid, node in tqdm(zip(node_uids, node_data), disable=checks.quiet_mode):
        x, y, live = node
        g_multi_copy.nodes[uid]['x'] = x
        g_multi_copy.nodes[uid]['y'] = y
        g_multi_copy.nodes[uid]['live'] = bool(live)

    logger.info('Unpacking edge data.')
    for edge in tqdm(edge_data, disable=checks.quiet_mode):
        start, end, length, angle_sum, imp_factor, start_bearing, end_bearing = edge
        start_uid = node_uids[int(start)]
        end_uid = node_uids[int(end)]
        # networkX will silently add new edges / data over existing edges
        g_multi_copy.add_edge(start_uid,
                              end_uid,
                              length=length,
                              angle_sum=angle_sum,
                              imp_factor=imp_factor)

    if metrics_dict is not None:
        logger.info('Unpacking metrics to nodes.')
        for uid, metrics in tqdm(metrics_dict.items(), disable=checks.quiet_mode):
            if uid not in g_multi_copy:
                raise KeyError(
                    f'Node uid {uid} not found in graph. '
                    f'Data dictionary uids must match those supplied with the node and edge maps.')
            g_multi_copy.nodes[uid]['metrics'] = metrics

    return g_multi_copy
