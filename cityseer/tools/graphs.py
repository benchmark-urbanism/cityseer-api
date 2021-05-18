"""
A collection of convenience functions for the preparation and conversion of [`NetworkX`](https://networkx.github.io/)
graphs to and from `cityseer` data structures. Note that the `cityseer` network data structures can be created and
manipulated directly, if so desired.
"""

import json
import logging
from typing import Union, Tuple, Optional

import networkx as nx
import numpy as np
import utm
from numba import types
from numba.typed import Dict
from shapely import geometry, ops, strtree, coords
from tqdm.auto import tqdm

from cityseer.algos import checks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def nX_simple_geoms(networkX_multigraph: nx.MultiGraph) -> nx.MultiGraph:
    """
    Generates straight-line geometries for each edge based on the the `x` and `y` coordinates of the adjacent nodes.
    The edge geometry will be stored to the edge `geom` attribute.

    Parameters
    ----------
    networkX_multigraph
        A `networkX` `MultiGraph` with `x` and `y` node attributes.

    Returns
    -------
    nx.MultiGraph
        A `networkX` `MultiGraph` with `shapely` [`Linestring`](https://shapely.readthedocs.io/en/latest/manual.html#linestrings)
        geometries assigned to the edge `geom` attributes.
    """
    if not isinstance(networkX_multigraph, nx.MultiGraph):
        raise TypeError('This method requires an undirected networkX MultiGraph.')
    logger.info('Generating simple (straight) edge geometries.')
    g_multi_copy = networkX_multigraph.copy()

    def _process_node(n):
        # x coordinate
        if 'x' not in g_multi_copy.nodes[n]:
            raise KeyError(f'Encountered node missing "x" coordinate attribute at node {n}.')
        x = g_multi_copy.nodes[n]['x']
        # y coordinate
        if 'y' not in g_multi_copy.nodes[n]:
            raise KeyError(f'Encountered node missing "y" coordinate attribute at node {n}.')
        y = g_multi_copy.nodes[n]['y']

        return x, y

    # unpack coordinates and build simple edge geoms
    remove_edges = []
    for s, e, k in tqdm(g_multi_copy.edges(keys=True), disable=checks.quiet_mode):
        s_x, s_y = _process_node(s)
        e_x, e_y = _process_node(e)
        g = geometry.LineString([[s_x, s_y], [e_x, e_y]])
        if s == e and g.length == 0:
            remove_edges.append((s, e, k))
        else:
            g_multi_copy[s][e][k]['geom'] = g
    for s, e, k in remove_edges:
        logger.warning(f'Found zero length looped edge for node {s}, removing from graph.')
        g_multi_copy.remove_edge(s, e, key=k)

    return g_multi_copy


def _add_node(networkX_multigraph: nx.MultiGraph,
              node_names: Union[list, tuple],
              x: float,
              y: float,
              live: Optional[bool] = None) -> Optional[str]:
    """
    Adds a node to a networkX `MultiGraph`. Assembles a new name from source node names. Checks for duplicates.
    """
    # suggest a name based on the given names
    if len(node_names) == 1:
        new_nd_name = str(node_names[0])
    # if concatenating existing nodes, suggest a name based on a combination of existing names
    else:
        names = []
        for name in node_names:
            name = str(name)
            if len(name) > 10:
                name = f'{name[:5]}|{name[-5:]}'
            names.append(name)
        new_nd_name = '±'.join(names)
    # first check whether the node already exists
    append = 2
    target_name = new_nd_name
    dupe = False
    while True:
        if f'{new_nd_name}' in networkX_multigraph:
            dupe = True
            # if the coordinates also match, then it is probable that the same node is being re-added...
            nd = networkX_multigraph.nodes[f'{new_nd_name}']
            if nd['x'] == x and nd['y'] == y:
                logger.debug(f'Proposed new node {new_nd_name} would overlay a node that already exists '
                             f'at the same coordinates. Skipping.')
                return None
            # otherwise, warn and bump the appended node number
            new_nd_name = f'{target_name}§v{append}'
            append += 1
        else:
            if dupe:
                logger.debug(f'A node of the same name already exists in the graph, '
                             f'adding this node as {new_nd_name} instead.')
            break
    # add
    attributes = {'x': x, 'y': y}
    if live is not None:
        attributes['live'] = live
    networkX_multigraph.add_node(new_nd_name, **attributes)
    return new_nd_name


def nX_from_osm(osm_json: str) -> nx.MultiGraph:
    """
    Generates a `NetworkX` `MultiGraph` from [Open Street Map](https://www.openstreetmap.org) data.

    Parameters
    -----------
    osm_json
        A `json` string response from the [OSM overpass API](https://wiki.openstreetmap.org/wiki/Overpass_API),
        consisting of `nodes` and `ways`.

    Returns
    -------
    nx.MultiGraph
        A `NetworkX` `MultiGraph` with `x` and `y` attributes in [WGS84](https://epsg.io/4326) `lng`, `lat` geographic
        coordinates.

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


def nX_wgs_to_utm(networkX_multigraph: nx.MultiGraph,
                  force_zone_number: int = None) -> nx.MultiGraph:
    """
    Converts `x` and `y` node attributes from [WGS84](https://epsg.io/4326) `lng`, `lat` geographic coordinates to the
    local UTM projected coordinate system. If edge `geom` attributes are found, the associated `LineString` geometries
    will also be converted. The UTM zone derived from the first processed node will be used for the conversion of all
    other nodes and geometries contained in the graph. This ensures consistent behaviour in cases where a graph spans
    a UTM boundary.

    Parameters
    ----------
    networkX_multigraph
        A `networkX` `MultiGraph` with `x` and `y` node attributes in the WGS84 coordinate system. Optional `geom` edge
        attributes containing `LineString` geoms to be converted.
    force_zone_number
        An optional UTM zone number for coercing all conversions to an explicit UTM zone. Use with caution: mismatched
        UTM zones may introduce substantial distortions in the results. Defaults to None.

    Returns
    -------
    nx.MultiGraph
        A `networkX` `MultiGraph` with `x` and `y` node attributes converted to the local UTM coordinate system. If edge
         `geom` attributes are present, these will also be converted.
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
    Optionally removes short dead-ends or disconnected graph components, which may be prevalent on poor quality network
    datasets.

    Parameters
    ----------
    networkX_multigraph
        A `networkX` `MultiGraph` in a projected coordinate system, containing `x` and `y` node attributes, and `geom`
        edge attributes containing `LineString` geoms.
    despine
        The maximum cutoff distance for removal of dead-ends. Use `None` or `0` where no despining should occur.
        Defaults to None.
    remove_disconnected
        Whether to remove disconnected components. If set to `True`, only the largest connected component will be
        returned. Defaults to True.

    Returns
    -------
    nx.MultiGraph
        A `networkX` `MultiGraph` with disconnected components optionally removed, and dead-ends removed where less than
         the `despine` parameter distance.
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
    if despine is not None and despine > 0:
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
    linestring_coords[idx] = tuple(coord)

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
    The force_xy parameter is useful for looping geometries or overlapping geometries where it can happen that
    welding works from either of the two ends, thus potentially mis-aligning the start point unless explicit.
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
    Removes nodes of degree=2: such nodes represent no route-choices other than traversal to the next edge.
    The edges on either side of the deleted nodes will be removed and replaced with a new spliced edge.

    :::tip Comment
    Filler nodes may be prevalent in poor quality datasets, or in situations where curved roadways have been represented
    through the addition of nodes to describe arced geometries. `cityseer` uses `shapely` [`Linestrings`](https://shapely.readthedocs.io/en/latest/manual.html#linestrings)
    to describe arbitrary road geometries without the need for filler nodes. Filler nodes can therefore be removed, thus
    reducing side-effects as a function of varied node intensities when computing network centralities.
    :::

    Parameters
    ----------
    networkX_multigraph
        A `networkX` `MultiGraph` in a projected coordinate system, containing `x` and `y` node attributes, and `geom`
        edge attributes containing `LineString` geoms.

    Returns
    -------
    nx.MultiGraph
        A `networkX` `MultiGraph` with nodes of degree=2 removed. Adjacent edges will be combined into a unified new
        edge with associated `geom` attributes spliced together.
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
            nbs = list(nx.neighbors(networkX_multigraph, n))
            # catch the edge case where the a single dead-end node has two out-edges to a single neighbour
            if len(nbs) == 1:
                continue
            # if only one neighbour, then ignore (e.g. dead-end with two edges linking back to another node)
            if nbs == 1:
                continue
            # otherwise randomly select one side and find a non-simple node as a starting point.
            else:
                nb = nbs[0]
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
                    # there is ordinarily a single edge from trailing to next
                    # however, there is an edge case where next is a dead-end with two edges linking back to trailing
                    # (i.e. where one of those edges is longer than the maximum length discrepancy for merging edges)
                    # in either case, use the first geom
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
                    # in the above-mentioned edge-case, a single dead-end node with two edges back to a start node
                    # will only have one neighbour
                    new_nbs = list(nx.neighbors(networkX_multigraph, next_link_nd))
                    if len(new_nbs) == 1:
                        trailing_nd = next_link_nd
                        next_link_nd = new_nbs[0]
                    # but in almost all cases there will be two neighbours, one of which will be the previous node
                    else:
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


def _squash_adjacent(networkX_multigraph: nx.MultiGraph,
                     node_group: Union[set, list, tuple],
                     cent_min_degree: Optional[int] = None,
                     cent_min_len_factor: Optional[float] = None) -> nx.MultiGraph:
    """
    Squashes nodes from the node group down to a new node. The new node can either be based on:
    - The centroid of all nodes;
    - Else, all nodes of degree greater or equal to cent_min_degree;
    - Else, all nodes with aggregate adjacent edge lengths greater than cent_min_len_factor as a factor of the node with
      the greatest overall aggregate lengths. Edges are adjusted from the old nodes to the new combined node.
    """
    if not isinstance(networkX_multigraph, nx.MultiGraph):
        raise TypeError('This method requires an undirected networkX MultiGraph (for multiple edges).')
    if cent_min_degree is not None and cent_min_degree < 1:
        raise ValueError('merge_node_min_degree should be a positive integer.')
    if cent_min_len_factor is not None and not 1 >= cent_min_len_factor >= 0:
        raise ValueError('cent_min_len_factor should be a decimal between 0 and 1.')
    # remove any node uids no longer in the graph
    node_group = [n for n in node_group if n in networkX_multigraph]
    # filter out nodes if using cent_min_degree or cent_min_len_factor
    filtered_nodes = []
    if cent_min_degree is not None:
        for n_uid in node_group:
            if nx.degree(networkX_multigraph, n_uid) >= cent_min_degree:
                filtered_nodes.append(n_uid)
    # else if merging on a longest adjacent edges basis
    if cent_min_len_factor is not None:
        # if nodes are pre-filtered by edge degrees, then use the filtered nodes as a starting point
        if filtered_nodes:
            node_pool = filtered_nodes.copy()
            filtered_nodes = []  # reset
        # else use the full original node group
        else:
            node_pool = node_group
        agg_lens = []
        for n_uid in node_pool:
            agg_len = 0
            # iterate each node's neighbours, aggregating neighbouring edge lengths along the way
            for nb_uid in nx.neighbors(networkX_multigraph, n_uid):
                for nb_edge in networkX_multigraph[n_uid][nb_uid].values():
                    agg_len += nb_edge['geom'].length
            agg_lens.append(agg_len)
        # find the longest
        max_len = max(agg_lens)
        # select all nodes with an agg_len within a small tolerance of longest
        for n_uid, agg_len in zip(node_pool, agg_lens):
            if agg_len >= max_len * cent_min_len_factor:
                filtered_nodes.append(n_uid)
    # otherwise, derive the centroid from all nodes
    # this is also a fallback if no nodes selected via minimum degree basis
    if not filtered_nodes:
        filtered_nodes = node_group
    # prepare the names and geoms for all points used for the new centroid
    node_geoms = []
    coords_set = set()
    for n_uid in filtered_nodes:
        x = networkX_multigraph.nodes[n_uid]['x']
        y = networkX_multigraph.nodes[n_uid]['y']
        # in rare cases opposing geom splitting can cause overlaying nodes
        # these can swing the gravity of multipoint centroids so screen these out
        xy_key = f'{round(x)}-{round(y)}'
        if xy_key in coords_set:
            continue
        else:
            coords_set.add(xy_key)
            node_geoms.append(geometry.Point(x, y))
    # set the new centroid from the centroid of the node group's Multipoint:
    c = geometry.MultiPoint(node_geoms).centroid
    # now that the centroid is known, go ahead and merge the _node_group
    # add the new node
    new_nd_name = _add_node(networkX_multigraph, node_group, x=c.x, y=c.y)
    if new_nd_name is None:
        raise ValueError(f'Attempted to add a duplicate node for node_group {node_group}.')
    # iterate the nodes to be removed and connect their existing edge geometries to the new centroid
    for uid in node_group:
        # iterate the node's existing neighbours
        for nb_uid in nx.neighbors(networkX_multigraph, uid):
            # if a neighbour is also going to be dropped, then no need to create new between edges
            # an exception exists when a geom is looped, in which case the neighbour is also the current node
            if nb_uid in node_group and nb_uid != uid:
                continue
            # MultiGraph - so iter edges
            for edge in networkX_multigraph[uid][nb_uid].values():
                if 'geom' not in edge:
                    raise KeyError(f'Missing "geom" attribute for edge {uid}-{nb_uid}')
                line_geom = edge['geom']
                if line_geom.type != 'LineString':
                    raise TypeError(
                        f'Expecting LineString geometry but found {line_geom.type} geometry for edge {uid}-{nb_uid}.')
                # orient the LineString so that the starting point matches the node's xy
                s_xy = (networkX_multigraph.nodes[uid]['x'], networkX_multigraph.nodes[uid]['y'])
                line_coords = _align_linestring_coords(line_geom.coords, s_xy)
                # update geom starting point to new parent node's coordinates
                line_coords = _snap_linestring_startpoint(line_coords, (c.x, c.y))
                # if self-loop, then the end also needs updating
                if uid == nb_uid:
                    line_coords = _snap_linestring_endpoint(line_coords, (c.x, c.y))
                    target_uid = new_nd_name
                else:
                    target_uid = nb_uid
                # build the new geom
                new_edge_geom = geometry.LineString(line_coords)
                # check that a duplicate is not being added
                dupe = False
                if networkX_multigraph.has_edge(new_nd_name, target_uid):
                    # only add parallel edges if substantially different from any existing edges
                    n_edges = networkX_multigraph.number_of_edges(new_nd_name, target_uid)
                    for k in range(n_edges):
                        existing_edge_geom = networkX_multigraph[new_nd_name][target_uid][k]["geom"]
                        # don't add if the edges have the same number of coords and the coords are similar
                        # 5m x and y tolerance across all coordinates
                        if len(new_edge_geom.coords) == len(existing_edge_geom.coords) and \
                                np.allclose(new_edge_geom.coords, existing_edge_geom.coords, atol=5, rtol=0):
                            dupe = True
                            logger.debug(
                                f'Not adding edge {new_nd_name} to {target_uid}: a similar edge already exists. '
                                f'Length: {new_edge_geom.length} vs. {existing_edge_geom.length}. '
                                f'Num coords: {len(new_edge_geom.coords)} vs. {len(existing_edge_geom.coords)}.')
                if not dupe:
                    # add the new edge
                    networkX_multigraph.add_edge(new_nd_name, target_uid, geom=new_edge_geom)
        # drop the node, this will also implicitly drop the old edges
        networkX_multigraph.remove_node(uid)

    return networkX_multigraph


def _merge_parallel_edges(networkX_multigraph: nx.MultiGraph,
                          merge_edges_by_midline: bool,
                          multi_edge_len_factor: float,
                          multi_edge_min_len: float) -> nx.MultiGraph:
    """
    Checks a MultiGraph for duplicate edges, which are then consolidated.
    If merge_edges_by_midline is False, then the shortest of the edges is used and the others are simply dropped.
    If merge_edges_by_midline is True, then the duplicates are replaced with a new edge following the merged centreline.
    In cases where one line is significantly longer than another (e.g. a crescent streets),
    then the longer edge is retained as separate if exceeding the multi_edge_len_factor as a factor of the shortest
    length but with the exception that (longer) edges still shorter than multi_edge_min_len are removed regardless.
    """
    if not isinstance(networkX_multigraph, nx.MultiGraph):
        raise TypeError('This method requires an undirected networkX MultiGraph (for multiple edges).')
    if multi_edge_len_factor <= 1:
        raise TypeError('multi_edge_len_factor should be a factor greater than 1. ')
    if multi_edge_len_factor < 1.25:
        logger.warning('Merging by midline and setting multi_edge_len_factor too low (e.g. lower than 1.25) may '
                       'result in an undesirable number of relatively similar parallel edges.')
    # don't use copy() - add nodes only
    deduped_graph = nx.MultiGraph()
    deduped_graph.add_nodes_from(networkX_multigraph.nodes(data=True))
    # iter the edges
    for s, e, d in tqdm(networkX_multigraph.edges(data=True), disable=checks.quiet_mode):
        # if only one edge is associated with this node pair, then add
        if networkX_multigraph.number_of_edges(s, e) == 1:
            deduped_graph.add_edge(s, e, **d)
        # otherwise, add if not already added from another (parallel) edge
        elif not deduped_graph.has_edge(s, e):
            # there are normally two edges, but sometimes three or possibly more
            edges = networkX_multigraph[s][e].values()
            # find the shortest of the geoms
            edge_geoms = [edge['geom'] for edge in edges]
            edge_lens = [geom.length for geom in edge_geoms]
            shortest_idx = edge_lens.index(min(edge_lens))
            shortest_len = edge_lens.pop(shortest_idx)
            shortest_geom = edge_geoms.pop(shortest_idx)
            longer_geoms = []
            for edge_len, edge_geom in zip(edge_lens, edge_geoms):
                # retain distinct edges where they are substantially longer than the shortest geom
                if edge_len > shortest_len * multi_edge_len_factor and edge_len > multi_edge_min_len:
                    deduped_graph.add_edge(s, e, geom=edge_geom)
                # otherwise, add to the list of longer geoms to be merged along with shortest
                else:
                    longer_geoms.append(edge_geom)
            # otherwise, if not merging on a midline basis
            # or, if no other edges to process (in cases where longer geom has been retained per above)
            # then use the shortest geom
            if not merge_edges_by_midline or len(longer_geoms) == 0:
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
                    for longer_geom in longer_geoms:
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


def _create_nodes_strtree(networkX_multigraph: nx.MultiGraph) -> strtree.STRtree:
    """
    Creates a nodes-based STRtree spatial index.
    """
    points = []
    for n, d in networkX_multigraph.nodes(data=True):
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
        p.degree = nx.degree(networkX_multigraph, n)
        points.append(p)
    return strtree.STRtree(points)


def _create_edges_strtree(networkX_multigraph: nx.MultiGraph) -> strtree.STRtree:
    """
    Creates an edges-based STRtree spatial index.
    """
    lines = []
    for s, e, k, d in networkX_multigraph.edges(keys=True, data=True):
        if 'geom' not in d:
            raise KeyError(f'Encountered edge missing "geom" attribute.')
        linestring = d['geom']
        linestring.start_uid = s
        linestring.end_uid = e
        linestring.k = k
        lines.append(linestring)
    return strtree.STRtree(lines)


def nX_consolidate_nodes(networkX_multigraph: nx.MultiGraph,
                         buffer_dist: float = 5,
                         min_node_group: int = 2,
                         min_node_degree: int = 1,
                         min_cumulative_degree: int = None,
                         max_cumulative_degree: int = None,
                         neighbour_policy: str = None,
                         crawl: bool = True,
                         cent_min_degree: int = 3,
                         cent_min_len_factor: float = None,
                         merge_edges_by_midline: bool = True,
                         multi_edge_len_factor: float = 1.25,
                         multi_edge_min_len: float = 100) -> nx.MultiGraph:
    """
    Consolidates nodes if they are within a buffer distance of each other. Several parameters provide more control over
    the conditions used for deciding whether or not to merge nodes. The algorithm proceeds in two steps:
    - Nodes within the buffer distance of each other are merged. A new centroid will be determined and all existing
        edge endpoints will be updated accordingly. The new centroid for the merged nodes can be based on:
        - The centroid of the node group;
        - Else, all nodes of degree greater or equal to `cent_min_degree`;
        - Else, all nodes with aggregate adjacent edge lengths greater than a factor of `cent_min_len_factor` of the node
        with the greatest aggregate length for adjacent edges.
    - The merging of nodes creates parallel edges which may start and end at a shared node on either side. These edges
        are replaced by a single new edge, with the new geometry selected from either:
        - An imaginary centreline of the combined edges if `merge_edges_by_midline` is set to `True`;
        - Else, the shortest edge, with longer edges discarded;
        - Note that substantially longer parallel edges are retained, instead of discarded, if they exceed
          `multi_edge_len_factor` and are longer than `multi_edge_min_len`.

    Parameters
    ----------
    networkX_multigraph
        A `networkX` `MultiGraph` in a projected coordinate system, containing `x` and `y` node attributes, and `geom`
        edge attributes containing `LineString` geoms.
    buffer_dist
        The buffer distance to be used for consolidating nearby nodes. Defaults to 5.
    min_node_group
        The minimum number of nodes to consider a valid group for consolidation. Defaults to 2.
    min_node_degree
        The least number of edges a node should have in order to be considered for consolidation. Defaults to 1.
    min_cumulative_degree
        An optional minimum cumulative degree to consider a valid node group for consolidation. Defaults to None.
    max_cumulative_degree
        An optional maximum cumulative degree to consider a valid node group for consolidation. Defaults to None.
    neighbour_policy
        Whether all nodes within the buffer distance are merged, or only "direct" or "indirect" neighbours. Defaults to
        None.
    crawl
        Whether the algorithm will recursively explore neighbours of neighbours if those neighbours are within the
        buffer distance from the prior node. Defaults to True.
    cent_min_degree
        The minimum node degree for a node to be considered when calculating the new centroid for the merged node
        cluster. Defaults to 3.
    cent_min_len_factor
        The minimum aggregate adjacent edge lengths an existing node should have to be considered when calculating the
        centroid for the new node cluster. Expressed as a factor of the node with the greatest aggregate adjacent edge
        lengths. Defaults to None.
    merge_edges_by_midline
        Whether to merge parallel edges by an imaginary centreline. If set to False, then the shortest edge will be
        retained as the new geometry and the longer edges will be discarded. Defaults to True.
    multi_edge_len_factor
        In cases where one line is significantly longer than another (e.g. crescent streets) then the longer edge is
        retained as separate if exceeding the multi_edge_len_factor as a factor of the shortest length but with the
        exception that (longer) edges still shorter than multi_edge_min_len are removed regardless. Defaults to 1.5.
    multi_edge_min_len
        See `multi_edge_len_factor`. Defaults to 100.

    Returns
    -------
    nx.MultiGraph
        A `networkX` `MultiGraph` with consolidated nodes.

    Notes
    -----
    See the guide on [graph cleaning](/guide/#graph-cleaning) for more information.

    ![Example raw graph from OSM](../../src/assets/plots/images/graph_cleaning_1.png)
    _The pre-consolidation OSM street network for Soho, London. © OpenStreetMap contributors._

    ![Example cleaned graph](../../src/assets/plots/images/graph_cleaning_5.png)
    _The consolidated OSM street network for Soho, London. © OpenStreetMap contributors._
    """
    if not isinstance(networkX_multigraph, nx.MultiGraph):
        raise TypeError('This method requires an undirected networkX MultiGraph.')
    if min_node_group < 2:
        raise ValueError('The minimum node threshold should be set to at least two.')
    if neighbour_policy is not None and neighbour_policy not in ('direct', 'indirect'):
        raise ValueError('Neighbour policy should be one "direct", "indirect", else the default of "None"')
    if crawl and buffer_dist > 25:
        logger.warning('Be cautious with the buffer distance when using crawl.')
    logger.info('Consolidating nodes.')
    _multi_graph = networkX_multigraph.copy()
    # create a nodes STRtree
    nodes_tree = _create_nodes_strtree(_multi_graph)
    # keep track of removed nodes
    removed_nodes = set()

    def recursive_squash(nd_uid: Union[int, str],
                         x: float,
                         y: float,
                         node_group: list,
                         processed_nodes: list,
                         recursive: bool = False):
        # keep track of which nodes have been processed as part of recursion
        processed_nodes.append(nd_uid)
        # get all other nodes within buffer distance - the self-node and previously processed nodes are also returned
        js = nodes_tree.query(geometry.Point(x, y).buffer(buffer_dist))
        # review each node within the buffer
        for j in js:
            j_uid = j.uid
            if j_uid in removed_nodes or j_uid in processed_nodes or j.degree < min_node_degree:
                continue
            # check neighbour policy
            if neighbour_policy is not None:
                # use the original graph prior to in-place modifications
                neighbours = nx.neighbors(networkX_multigraph, nd_uid)
                if neighbour_policy == 'indirect' and j_uid in neighbours:
                    continue
                elif neighbour_policy == 'direct' and j_uid not in neighbours:
                    continue
            # otherwise add the node
            node_group.append(j_uid)
            # if recursive, follow the chain
            if recursive:
                j_nd = networkX_multigraph.nodes[j_uid]
                return recursive_squash(j_uid,
                                        j_nd['x'],
                                        j_nd['y'],
                                        node_group,
                                        processed_nodes,
                                        recursive=crawl)

        return node_group

    # iterate origin graph (else node structure changes in place)
    for n, n_d in tqdm(networkX_multigraph.nodes(data=True), disable=checks.quiet_mode):
        # skip if already consolidated from an adjacent node, or if the node's degree doesn't meet min_node_degree
        if n in removed_nodes or nx.degree(networkX_multigraph, n) < min_node_degree:
            continue
        node_group = recursive_squash(n,  # node uid
                                      n_d['x'],  # x point for buffer
                                      n_d['y'],  # y point for buffer
                                      [n],  # node group for consolidation (with starting node)
                                      [],  # processed nodes tracked through recursion
                                      crawl)  # whether to recursively probe neighbours per distance
        # check for min_node_threshold
        if len(node_group) < min_node_group:
            continue
        # check for cumulative degree thresholds if requested
        if min_cumulative_degree is not None or max_cumulative_degree is not None:
            cumulative_degree = sum([nx.degree(networkX_multigraph, n) for n in node_group])
            if min_cumulative_degree is not None and cumulative_degree < min_cumulative_degree:
                continue
            if max_cumulative_degree is not None and cumulative_degree > max_cumulative_degree:
                continue
        # update removed nodes
        removed_nodes.update(node_group)
        # consolidate if nodes have been identified within buffer and if these exceed min_node_threshold
        _multi_graph = _squash_adjacent(_multi_graph,
                                        node_group,
                                        cent_min_degree,
                                        cent_min_len_factor)
    # remove filler nodes
    deduped_graph = nX_remove_filler_nodes(_multi_graph)
    # remove any parallel edges that may have resulted from squashing nodes
    deduped_graph = _merge_parallel_edges(deduped_graph,
                                          merge_edges_by_midline,
                                          multi_edge_len_factor,
                                          multi_edge_min_len)

    return deduped_graph


def nX_split_opposing_geoms(networkX_multigraph: nx.MultiGraph,
                            buffer_dist: float = 10,
                            merge_edges_by_midline: bool = True,
                            multi_edge_len_factor: float = 1.25,
                            multi_edge_min_len: float = 100) -> nx.MultiGraph:
    """
    Projects nodes to pierce opposing edges within a buffer distance. The pierced nodes facilitate subsequent
    merging for scenarios such as divided boulevards.

    Parameters
    ----------
    networkX_multigraph
        A `networkX` `MultiGraph` in a projected coordinate system, containing `x` and `y` node attributes, and `geom`
        edge attributes containing `LineString` geoms.
    buffer_dist
        The buffer distance to be used for splitting nearby nodes. Defaults to 5.
    merge_edges_by_midline
        Whether to merge parallel edges by an imaginary centreline. If set to False, then the shortest edge will be
        retained as the new geometry and the longer edges will be discarded. Defaults to True.
    multi_edge_len_factor
        In cases where one line is significantly longer than another (e.g. crescent streets) then the longer edge is
        retained as separate if exceeding the `multi_edge_len_factor` as a factor of the shortest length but with the
        exception that (longer) edges still shorter than `multi_edge_min_len` are removed regardless. Defaults to 1.5.
    multi_edge_min_len
        See `multi_edge_len_factor`. Defaults to 100.

    Returns
    -------
    nx.MultiGraph
        A `networkX` `MultiGraph` with consolidated nodes.
    """

    def make_edge_key(s, e, k):
        return '-'.join(sorted([str(s), str(e)])) + f'-k{k}'

    # where edges are deleted, keep track of new children edges
    edge_children = {}

    # recursive function for retrieving nested layers of successively replaced edges
    def recurse_child_keys(s, e, k, geom, current_edges):
        """
        Checks if an edge has been replaced by children, if so, use children instead.
        Children may also have children, so recurse downwards.
        """
        edge_key = make_edge_key(s, e, k)
        # if an edge does not have children, add to current_edges and return
        if edge_key not in edge_children:
            current_edges.append((s, e, k, geom))
        # otherwise recursively drill-down until newest edges are found
        else:
            for child_s, child_e, child_k, child_geom in edge_children[edge_key]:
                recurse_child_keys(child_s, child_e, child_k, child_geom, current_edges)

    if not isinstance(networkX_multigraph, nx.MultiGraph):
        raise TypeError('This method requires an undirected networkX MultiGraph.')
    logger.info(f'Splitting opposing edges.')
    _multi_graph = networkX_multigraph.copy()
    # create an edges STRtree (nodes and edges)
    edges_tree = _create_edges_strtree(_multi_graph)
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
        edges = [(edge.start_uid, edge.end_uid, edge.k, edge) for edge in edges]
        # check against removed edges
        current_edges = []
        for s, e, k, edge_geom in edges:
            recurse_child_keys(s, e, k, edge_geom, current_edges)
        # get neighbouring nodes from new graph
        neighbours = list(_multi_graph.neighbors(n))
        # abort if only direct neighbours
        if len(current_edges) <= len(neighbours):
            continue
        # filter current_edges
        gapped_edges = []
        for s, e, k, edge_geom in current_edges:
            # skip direct neighbours
            if s == n or e == n:
                continue
            # check whether the geom is truly within the buffer distance
            if edge_geom.distance(n_point) > buffer_dist:
                continue
            gapped_edges.append((s, e, k, edge_geom))
        # abort if no gapped edges
        if not gapped_edges:
            continue
        # prepare the root node's point geom
        n_geom = geometry.Point(n_d['x'], n_d['y'])
        # iter gapped edges
        for s, e, k, edge_geom in gapped_edges:
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
            # add the new node and edges to _multi_graph (don't modify networkX_multigraph because of iter in place)
            new_nd_name = _add_node(_multi_graph, [s, n, e], x=nearest_point.x, y=nearest_point.y)
            # if a node already exists at this location, add_node will return None
            if new_nd_name is None:
                continue
            _multi_graph.add_edge(s, new_nd_name)
            _multi_graph.add_edge(e, new_nd_name)
            if np.allclose(s_nd_geom.coords, new_edge_geom_a.coords[0][:2], atol=checks.tolerance, rtol=0) or \
                    np.allclose(s_nd_geom.coords, new_edge_geom_a.coords[-1][:2], atol=checks.tolerance, rtol=0):
                s_new_geom = new_edge_geom_a
                e_new_geom = new_edge_geom_b
            else:
                # double check matching geoms
                if not np.allclose(s_nd_geom.coords, new_edge_geom_b.coords[0][:2], atol=checks.tolerance, rtol=0) and \
                        not np.allclose(s_nd_geom.coords, new_edge_geom_b.coords[-1][:2], atol=checks.tolerance,
                                        rtol=0):
                    raise ValueError('Unable to match split geoms to existing nodes')
                s_new_geom = new_edge_geom_b
                e_new_geom = new_edge_geom_a
            # if splitting a looped component, then both new edges will have the same starting and ending nodes
            # in these cases, there will be multiple edges
            if s == e:
                assert _multi_graph.number_of_edges(s, new_nd_name) == 2
                s_k = 0
                e_k = 1
            else:
                assert _multi_graph.number_of_edges(s, new_nd_name) == 1
                assert _multi_graph.number_of_edges(e, new_nd_name) == 1
                s_k = e_k = 0
            # write the new edges
            _multi_graph[s][new_nd_name][s_k]['geom'] = s_new_geom
            _multi_graph[e][new_nd_name][e_k]['geom'] = e_new_geom
            # add the new edges to the edge_children dictionary
            edge_key = make_edge_key(s, e, k)
            edge_children[edge_key] = [(s, new_nd_name, s_k, s_new_geom),
                                       (e, new_nd_name, e_k, e_new_geom)]
            # drop the old edge from _multi_graph
            if _multi_graph.has_edge(s, e, k):
                _multi_graph.remove_edge(s, e, k)
    # squashing nodes can result in edge duplicates
    deduped_graph = _merge_parallel_edges(_multi_graph,
                                          merge_edges_by_midline,
                                          multi_edge_len_factor,
                                          multi_edge_min_len)

    return deduped_graph


def nX_decompose(networkX_multigraph: nx.MultiGraph,
                 decompose_max: float) -> nx.MultiGraph:
    """
    Decomposes a graph so that no edge is longer than a set maximum. Decomposition provides a more granular
    representation of potential variations along street lengths, while reducing network centrality side-effects that
    arise as a consequence of varied node densities.

    :::warning Comment
    Setting the `decompose` parameter too small in relation to the size of the graph may increase the computation time
    unnecessarily for subsequent analysis. For larger-scale urban analysis, it is generally not necessary to go smaller
    20m, and 50m may already be sufficient for the majority of cases.
    :::

    Parameters
    ----------
    networkX_multigraph
        A `networkX` `MultiGraph` in a projected coordinate system, containing `x` and `y` node attributes, and `geom`
        edge attributes containing `LineString` geoms.

    decompose_max
        The maximum length threshold for decomposed edges.

    Returns
    -------
    nx.MultiGraph
        A decomposed `networkX` graph with no edge longer than the `decompose_max` parameter. If `live` node attributes
        were provided, then the `live` attribute for child-nodes will be set to `True` if either or both parent nodes
        were `live`. Otherwise, all nodes wil be set to `live=True`. The `length` and `impedance` edge attributes will
        be set to match the lengths of the new edges.

    Notes
    -----
    ```python
    from cityseer.tools import mock, graphs, plot

    G = mock.mock_graph()
    G_simple = graphs.nX_simple_geoms(G)
    G_decomposed = graphs.nX_decompose(G_simple, 100)
    plot.plot_nX(G_decomposed)
    ```

    ![Example graph](../../src/assets/plots/images/graph_simple.png)
    _Example graph prior to decomposition._

    ![Example decomposed graph](../../src/assets/plots/images/graph_decomposed.png)
    _Example graph after decomposition._

    """
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
                f'No edge geom found for edge {s}-{e}: '
                f'Please add an edge "geom" attribute consisting of a shapely LineString.')
        # get edge geometry
        line_geom = d['geom']
        if line_geom.type != 'LineString':
            raise TypeError(f'Expecting LineString geometry but found {line_geom.type} geometry for edge {s}-{e}.')
        # check geom coordinates directionality - flip if facing backwards direction
        line_geom_coords = _align_linestring_coords(line_geom.coords, (s_x, s_y))
        # double check that coordinates now face the forwards direction
        if not np.allclose((s_x, s_y), line_geom_coords[0][:2], atol=checks.tolerance, rtol=0) or \
                not np.allclose((e_x, e_y), line_geom_coords[-1][:2], atol=checks.tolerance, rtol=0):
            raise ValueError(f'Edge geometry endpoint coordinate mismatch for edge {s}-{e}')
        line_geom = geometry.LineString(line_geom_coords)
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
            # create the split LineString geom for measuring the new length
            line_segment = ops.substring(line_geom, step, step + step_size)
            # get the x, y of the new end node
            x, y = line_segment.coords[-1]
            # add the new node and edge
            new_nd_name = _add_node(g_multi_copy, [s, sub_node_counter, e], x=x, y=y)
            if new_nd_name is None:
                raise ValueError(f'Attempted to add a duplicate node. '
                                 f'Check for existence of duplicate edges in the vicinity of {s}-{e}.')
            sub_node_counter += 1
            # add and set live property if present in parent graph
            if 'live' in networkX_multigraph.nodes[s] and 'live' in networkX_multigraph.nodes[e]:
                live = True
                # if BOTH parents are not live, then set child to not live
                if not networkX_multigraph.nodes[s]['live'] and not networkX_multigraph.nodes[e]['live']:
                    live = False
                g_multi_copy.nodes[new_nd_name]['live'] = live
            # add the edge
            g_multi_copy.add_edge(prior_node_id, new_nd_name, geom=line_segment)
            # increment the step and node id
            prior_node_id = new_nd_name
            step += step_size
        # set the last edge manually to avoid rounding errors at end of LineString
        # the nodes already exist, so just add edge
        line_segment = ops.substring(line_geom, step, line_geom.length)
        g_multi_copy.add_edge(prior_node_id, e, geom=line_segment)

    return g_multi_copy


def nX_to_dual(networkX_multigraph: nx.MultiGraph) -> nx.MultiGraph:
    """
    Converts a primal graph representation, where intersections are represented as nodes and streets as edges, to the
    dual representation. So doing, edges are converted to nodes and intersections become edges. Primal edge `geom`
    attributes will be welded to adjacent edges and split into the new dual edge `geom` attributes.

    :::tip Comment
    Note that a `MultiGraph` is useful for primal but not for dual, so the output `MultiGraph` will have single edges.
    e.g. a crescent street that spans the same intersections as parallel straight street requires multiple edges in
    primal. The same type of situation does not arise in the dual because the nodes map to distinct edges regardless.
    :::

    Parameters
    ----------
    networkX_multigraph
        A `networkX` `MultiGraph` in a projected coordinate system, containing `x` and `y` node attributes, and `geom`
        edge attributes containing `LineString` geoms.

    Returns
    -------
    nx.MultiGraph
        A dual representation `networkX` graph. The new dual nodes will have `x` and `y` node attributes corresponding
        to the mid-points of the original primal edges. If `live` node attributes were provided, then the `live`
        attribute for the new dual nodes will be set to `True` if either or both of the adjacent primal nodes were set
        to `live=True`. Otherwise, all dual nodes wil be set to `live=True`. The primal `geom` edge attributes will be
        split and welded to form the new dual `geom` edge attributes. A `parent_primal_node` edge attribute will be
        added, corresponding to the node identifier of the primal graph.

    Notes
    -----
    ```python
    from cityseer.tools import graphs, mock, plot

    G = mock.mock_graph()
    G_simple = graphs.nX_simple_geoms(G)
    G_dual = graphs.nX_to_dual(G_simple)
    plot.plot_nX_primal_or_dual(G_simple,
                                G_dual,
                                plot_geoms=False)
    ```

    ![Example dual graph](../../src/assets/plots/images/graph_dual.png)
    _Dual graph (blue) overlaid on the source primal graph (red)._

    """

    if not isinstance(networkX_multigraph, nx.MultiGraph):
        raise TypeError('This method requires an undirected networkX MultiGraph.')
    logger.info('Converting graph to dual.')
    g_dual = nx.MultiGraph()

    def get_half_geoms(g, a_node, b_node, edge_k):
        '''
        For splitting and orienting half geoms
        '''
        # get edge data
        edge_data = g[a_node][b_node][edge_k]
        # test for x coordinates
        if 'x' not in g.nodes[a_node] or 'y' not in g.nodes[a_node]:
            raise KeyError(f'Encountered node missing "x" or "y" coordinate attributes at node {a_node}.')
        # test for y coordinates
        if 'x' not in g.nodes[b_node] or 'y' not in g.nodes[b_node]:
            raise KeyError(f'Encountered node missing "x" or "y" coordinate attributes at node {b_node}.')
        a_xy = (g.nodes[a_node]['x'], g.nodes[a_node]['y'])
        b_xy = (g.nodes[b_node]['x'], g.nodes[b_node]['y'])
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
        # align geom coordinates to start from A side
        line_geom_coords = _align_linestring_coords(line_geom.coords, a_xy)
        line_geom = geometry.LineString(line_geom_coords)
        # generate the two half geoms
        a_half_geom = ops.substring(line_geom, 0, line_geom.length / 2)
        b_half_geom = ops.substring(line_geom, line_geom.length / 2, line_geom.length)
        # check that nothing odd happened with new midpoint
        assert np.allclose(a_half_geom.coords[-1][:2], b_half_geom.coords[0][:2], atol=checks.tolerance, rtol=0)
        # snap to prevent creeping tolerance issues
        # A side geom starts at node A and ends at new midpoint
        a_half_geom_coords = _snap_linestring_startpoint(a_half_geom.coords, a_xy)
        # snap new midpoint to geom A's endpoint (i.e. no need to snap endpoint of geom A)
        mid_xy = a_half_geom_coords[-1][:2]
        # B side geom starts at mid and ends at B node
        b_half_geom_coords = _snap_linestring_startpoint(b_half_geom.coords, mid_xy)
        b_half_geom_coords = _snap_linestring_endpoint(b_half_geom_coords, b_xy)
        # double check coords
        assert a_half_geom_coords[0][:2] == a_xy
        assert a_half_geom_coords[-1][:2] == mid_xy
        assert b_half_geom_coords[0][:2] == mid_xy
        assert b_half_geom_coords[-1][:2] == b_xy

        return geometry.LineString(a_half_geom_coords), geometry.LineString(b_half_geom_coords)

    def set_live(s, e, dual_n):
        # add and set live property to dual node if present in adjacent primal graph nodes
        if 'live' in networkX_multigraph.nodes[s] and 'live' in networkX_multigraph.nodes[e]:
            live = True
            # if BOTH parents are not live, then set child to not live
            if not networkX_multigraph.nodes[s]['live'] and not networkX_multigraph.nodes[e]['live']:
                live = False
            g_dual.nodes[dual_n]['live'] = live

    # iterate the primal graph's edges
    for s, e, k, d in tqdm(networkX_multigraph.edges(data=True, keys=True), disable=checks.quiet_mode):
        # get the first and second half geoms
        s_half_geom, e_half_geom = get_half_geoms(networkX_multigraph, s, e, k)
        # create a new dual node corresponding to the current primal edge
        # nodes are added manually to retain link to origin node names and to check for duplicates
        s_e = sorted([str(s), str(e)])
        hub_node_dual = f'{s_e[0]}_{s_e[1]}'
        # the node may already have been added from a neighbouring node that has already been processed
        if hub_node_dual not in g_dual:
            x, y = s_half_geom.coords[-1][:2]
            g_dual.add_node(hub_node_dual, x=x, y=y)
            # add and set live property if present in parent graph
            set_live(s, e, hub_node_dual)
        # process either side
        for n_side, half_geom in zip([s, e], [s_half_geom, e_half_geom]):
            # add the spoke edges on the dual
            for nb in nx.neighbors(networkX_multigraph, n_side):
                # don't follow neighbour back to current edge combo
                if nb in [s, e]:
                    continue
                # add the neighbouring primal edge as dual node
                s_nb = sorted([str(n_side), str(nb)])
                spoke_node_dual = f'{s_nb[0]}_{s_nb[1]}'
                # skip if the edge has already been processed from another direction
                if g_dual.has_edge(hub_node_dual, spoke_node_dual):
                    continue
                # get the near and far half geoms
                spoke_half_geom, _discard_geom = get_half_geoms(networkX_multigraph, n_side, nb, k)
                # nodes will be added if not already present (i.e. from first direction processed)
                if spoke_node_dual not in g_dual:
                    x, y = spoke_half_geom.coords[-1][:2]
                    g_dual.add_node(spoke_node_dual, x=x, y=y)
                    # add and set live property if present in parent graph
                    set_live(s, e, spoke_node_dual)
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
    """
    Transposes a `networkX` `MultiGraph` into `numpy` arrays for use by `NetworkLayer` classes. Calculates length and
    angle attributes, as well as in and out bearings and stores these in the returned data maps.

    :::warning Comment
    It is generally not necessary to use this function directly. This function will be called internally when invoking
    [NetworkLayerFromNX](/metrics/networks/#class-networklayerfromnx)
    :::

    Parameters
    ----------
    networkX_multigraph
        A `networkX` `MultiGraph` in a projected coordinate system, containing `x` and `y` node attributes, and `geom`
        edge attributes containing `LineString` geoms.

    Returns
    -------
    node_uids
        A tuple of node `uids` corresponding to the node identifiers in the source `networkX` graph.
    node_data
        A 2d `numpy` array representing the graph's nodes. The indices of the second dimension correspond as follows:
        
        | idx | property |
        |-----|:---------|
        | 0 | `x` coordinate |
        | 1 | `y` coordinate |
        | 2 | `bool` describing whether the node is `live`. Metrics are only computed for `live` nodes. |

    edge_data
        A 2d `numpy` array representing the graph's edges. Each edge will be described separately for each direction of
        travel. The indices of the second dimension correspond as follows:

        | idx | property |
        |-----|:---------|
        | 0 | start node `idx` |
        | 1 | end node `idx` |
        | 2 | the segment length in metres |
        | 3 | the sum of segment's angular change |
        | 4 | an 'impedance factor' which can be applied to magnify or reduce the effect of the edge's impedance on
        shortest-path calculations. e.g. for gradients or other such considerations. Use with caution. |
        | 5 | the edge's entry angular bearing |
        | 6 | the edge's exit angular bearing |

        All edge attributes will be generated automatically, however, the impedance factor parameter can be over-ridden by supplying a `imp_factor` attribute on the input graph's edges.
    node_edge_map
        A `numba` `Dict` with `node_data` indices as keys and `numba` `List` types as values containing the out-edge
        indices for each node.
    """

    if not isinstance(networkX_multigraph, nx.MultiGraph):
        raise TypeError('This method requires an undirected networkX MultiGraph.')
    logger.info('Preparing node and edge arrays from networkX graph.')
    g_multi_copy = networkX_multigraph.copy()
    # accumulate degrees
    total_out_degrees = 0
    for n in tqdm(g_multi_copy.nodes(), disable=checks.quiet_mode):
        # writing node identifier to 'labels' in case conversion to integers method interferes with order
        g_multi_copy.nodes[n]['label'] = n
        for nb in nx.neighbors(g_multi_copy, n):
            total_out_degrees += g_multi_copy.number_of_edges(n, nb)
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
            for nx_edge_idx, nx_edge_data in g_multi_copy[n][nb].items():
                # add the new edge index to the node's out edges
                out_edges.append(edge_idx)
                # EDGE MAP INDEX POSITION 0 = start node
                edge_data[edge_idx][0] = node_idx
                # EDGE MAP INDEX POSITION 1 = end node
                edge_data[edge_idx][1] = nb
                # EDGE MAP INDEX POSITION 2 = length
                if not 'geom' in nx_edge_data:
                    raise KeyError(
                        f'No edge geom found for edge {node_idx}-{nb}: '
                        f'Please add an edge "geom" attribute consisting of a shapely LineString.'
                        f'Simple (straight) geometries can be inferred automatically through use of the nX_simple_geoms() method.')
                line_geom = nx_edge_data['geom']
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
                line_geom_coords = _align_linestring_coords(line_geom.coords, (s_x, s_y))
                # iterate the coordinates and calculate the angular change
                angle_sum = 0
                for c in range(len(line_geom_coords) - 2):
                    x_1, y_1 = line_geom_coords[c][:2]
                    x_2, y_2 = line_geom_coords[c + 1][:2]
                    x_3, y_3 = line_geom_coords[c + 2][:2]
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
                if 'imp_factor' in nx_edge_data:
                    # cannot have imp_factor less than zero (but == 0 is OK)
                    imp_factor = nx_edge_data['imp_factor']
                    if not (np.isfinite(imp_factor) or np.isinf(imp_factor)) or imp_factor < 0:
                        raise ValueError(
                            f'Impedance factor: {imp_factor} for edge {node_idx}-{nb} must be a finite positive value or positive infinity.')
                    edge_data[edge_idx][4] = imp_factor
                else:
                    # fallback imp_factor of 1
                    edge_data[edge_idx][4] = 1
                # EDGE MAP INDEX POSITION 5 - in bearing
                x_1, y_1 = line_geom_coords[0][:2]
                x_2, y_2 = line_geom_coords[1][:2]
                edge_data[edge_idx][5] = np.rad2deg(np.arctan2(y_2 - y_1, x_2 - x_1))
                # EDGE MAP INDEX POSITION 6 - out bearing
                x_1, y_1 = line_geom_coords[-2][:2]
                x_2, y_2 = line_geom_coords[-1][:2]
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
    """
    Writes cityseer data graph maps back to a `MultiGraph`. Can write back to an existing `MultiGraph` if an existing
    graph is provided as an argument to the `networkX_multigraph` parameter.

    :::warning Comment
    It is generally not necessary to use this function directly. This function will be called internally when invoking
    [NetworkLayer.to_networkX](/metrics/networks/#networklayerto_networkx)
    :::

    Parameters
    ----------
    node_uids
        A tuple of node ids corresponding to the node identifiers for the target `networkX` graph.
    node_data
        A 2d `numpy` array representing the graph's nodes. The indices of the second dimension should correspond as
        follows:

        | idx | property |
        | :-: | :------- |
        | 0   | `x` coordinate |
        | 1   | `y` coordinate |
        | 2   | `bool` describing whether the node is `live` |

    edge_data
        A 2d `numpy` array representing the graph's directional edges. The indices of the second dimension should 
        correspond as follows:

        | idx | property |
        | :-: | :------- |
        | 0   | start node `idx` |
        | 1   | end node `idx` |
        | 2   | the segment length in metres |
        | 3   | the sum of segment's angular change |
        | 4   | 'impedance factor' applied to magnify or reduce the edge impedance. |
        | 5   | the edge's entry angular bearing |
        | 6   | the edge's exit angular bearing |

    node_edge_map
        A `numba` `Dict` with `node_data` indices as keys and `numba` `List` types as values containing the out-edge
        indices for each node.
    networkX_multigraph
        An optional `networkX` graph to use as a backbone for unpacking the data. The number of nodes and edges should
        correspond to the `cityseer` data maps and the node identifiers should correspond to the `node_uids`. If not
        provided, then a new `networkX` graph will be returned. This function is intended to be used for situations
        where `cityseer` data is being transposed back to a source `networkX` graph. Defaults to None.
    metrics_dict
        An optional dictionary with keys corresponding to the identifiers in `node_uids`. The dictionary's `values` will
        be unpacked to the corresponding nodes in the `networkX` graph. Defaults to None.

    Returns
    -------
    nx.MultiGraph
        A `networkX` graph. If a backbone graph was provided, a copy of the same graph will be returned with the data
        overridden as described below. If no graph was provided, then a new graph will be generated.
        `x`, `y`, `live`, `ghosted` node attributes will be copied from `node_data` to the graph nodes. `length`,
        `angle_sum`, `imp_factor`, `start_bearing`, and `end_bearing` attributes will be copied from the `edge_data`
        to the graph edges. If a `metrics_dict` is provided, all data will be copied to the graph nodes based on
        matching node identifiers.
    """
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
        g_multi_copy.add_nodes_from(node_uids)
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
        # note that the original geom is lost with round trip unless retained in a supplied backbone graph.
        # the edge map is directional, so each original edge will be processed twice, once from either direction.
        # edges are only added if A) not using a backbone graph and B) the edge hasn't already been added
        if networkX_multigraph is None:
            # if the edge doesn't already exist, then simply add
            if not g_multi_copy.has_edge(start_uid, end_uid):
                add_edge = True
            # else, only add if not matching an already added edge
            # i.e. don't add the same edge when processed from opposite direction
            else:
                add_edge = True  # tentatively set to True
                # iter the edges
                for edge_item_idx, edge_item_data in g_multi_copy[start_uid][end_uid].items():
                    # set add_edge to false if a matching edge length is found
                    if edge_item_data['length'] == length:
                        add_edge = False
            # add the edge if not existent
            if add_edge:
                g_multi_copy.add_edge(start_uid,
                                      end_uid,
                                      length=length,
                                      angle_sum=angle_sum,
                                      imp_factor=imp_factor)
        # if a new edge is not being added then add the attributes to the appropriate backbone edge if not already done
        # this is only relevant if processing a backbone graph
        else:
            # raise if the edge doesn't exist
            if not g_multi_copy.has_edge(start_uid, end_uid):
                raise KeyError(f'The backbone graph is missing an edge spanning from {start_uid} to {end_uid}'
                               f'The original graph (with all original edges) has to be reused.')
            # due working with a MultiGraph it is necessary to check that the correct edge index is matched
            for edge_item_idx, edge_item_data in g_multi_copy[start_uid][end_uid].items():
                if np.isclose(edge_item_data['geom'].length, length, atol=0.1, rtol=0.0):
                    # check whether the attributes have already been added from the other direction?
                    if 'length' in edge_item_data and edge_item_data['length'] == length:
                        continue
                    # otherwise add the edge attributes and move on
                    existing_edge = g_multi_copy[start_uid][end_uid][edge_item_idx]
                    existing_edge['length'] = length
                    existing_edge['angle_sum'] = angle_sum
                    existing_edge['imp_factor'] = imp_factor
    # unpack any metrics written to the nodes
    if metrics_dict is not None:
        logger.info('Unpacking metrics to nodes.')
        for uid, metrics in tqdm(metrics_dict.items(), disable=checks.quiet_mode):
            if uid not in g_multi_copy:
                raise KeyError(
                    f'Node uid {uid} not found in graph. '
                    f'Data dictionary uids must match those supplied with the node and edge maps.')
            g_multi_copy.nodes[uid]['metrics'] = metrics

    return g_multi_copy


def nX_from_OSMnx(networkX_multidigraph: nx.MultiDiGraph,
                  node_attributes: Union[list, tuple] = None,
                  edge_attributes: Union[list, tuple] = None,
                  tolerance: float = checks.tolerance) -> nx.MultiGraph:
    """
    Copies an [`OSMnx`](https://osmnx.readthedocs.io/) directed `MultiDiGraph` to an undirected `cityseer` compatible
    `MultiGraph`. See the [`OSMnx`](/guide/#osmnx) section of the guide for a more general discussion (and example) on
    workflows combining `OSMnx` with `cityseer`.

    `x` and `y` node attributes will be copied directly and `geometry` edge attributes will be copied to a `geom` edge
    attribute. The conversion process will snap the `shapely` `LineString` endpoints to the corresponding start and end
    node coordinates.

    Note that `OSMnx` `geometry` attributes only exist for simplified edges: if a `geometry` edge attribute is not
    found, then a simple (straight) `shapely` `LineString` geometry will be inferred from the respective start and end
    nodes.

    Other attributes will be ignored to avoid potential downstream misinterpretations of the attributes as a consequence
    of subsequent steps of graph manipulation, i.e. to avoid situations where attributes may fall out of lock-step with
    the state of the graph. If particular attributes need to be copied across, and assuming cognisance of downstream
    implications, then these can be manually specified by providing a list of node attributes keys per the
    `node_attributes` parameter or edge attribute keys per the `edge_attributes` parameter.

    Parameters
    ----------
    networkX_multidigraph
        A `OSMnx` derived `networkX` `MultiDiGraph` containing `x` and `y` node attributes, with optional `geometry`
        edge attributes containing `LineString` geoms (for simplified edges).
    node_attributes
        Optional node attributes to copy to the new MultiGraph. (In addition to the default `x` and `y` attributes.)
    edge_attributes
        Optional edge attributes to copy to the new MultiGraph. (In addition to the optional `geometry` attribute.)
    tolerance
        Tolerance at which to raise errors for mismatched geometry end-points vis-a-vis corresponding node coordinates.
        Prior to conversion, this method will check edge geometry end-points for alignment with the corresponding
        end-point nodes. Where these don't align within the given tolerance an exception will be raised. Otherwise, if
        within the tolerance, the conversion function will snap the geometry end-points to the corresponding node
        coordinates so that downstream exceptions are not subsequently raised. It is preferable to minimise graph
        manipulation prior to conversion to a `cityseer` compatible `MultiGraph` otherwise particularly large tolerances
        may be required, and this may lead to some unexpected or undesirable effects due to aggressive snapping.

    Returns
    -------
    nx.MultiGraph
        A `cityseer` compatible `networkX` graph with `x` and `y` node attributes and `geom` edge attribute.
    """
    if not isinstance(networkX_multidigraph, nx.MultiDiGraph):
        raise TypeError('This method requires a directed networkX MultiDiGraph as derived from `OSMnx`.')
    if node_attributes is not None and not isinstance(node_attributes, (list, tuple)):
        raise TypeError('Node attributes to be copied should be provided as either a list or tuple of attribute keys.')
    if edge_attributes is not None and not isinstance(edge_attributes, (list, tuple)):
        raise TypeError('Edge attributes to be copied should be provided as either a list or tuple of attribute keys.')
    logger.info('Converting OSMnx MultiDiGraph to cityseer MultiGraph.')
    # target MultiGraph
    g_multi = nx.MultiGraph()

    def _process_node(n):
        # x
        if 'x' not in networkX_multidigraph.nodes[n]:
            raise KeyError(f'Encountered node missing "x" coordinate attribute for node {n}.')
        x = networkX_multidigraph.nodes[n]['x']
        # y
        if 'y' not in networkX_multidigraph.nodes[s]:
            raise KeyError(f'Encountered node missing "y" coordinate attribute for node {n}.')
        y = networkX_multidigraph.nodes[n]['y']
        # add attributes if necessary
        if n not in g_multi:
            g_multi.add_node(n, x=x, y=y)
            if node_attributes is not None:
                for node_att in node_attributes:
                    if node_att not in networkX_multidigraph.nodes[n]:
                        raise ValueError(f'Attribute {node_att} is not available for node {n}.')
                    g_multi.nodes[n][node_att] = networkX_multidigraph.nodes[n][node_att]

        return x, y

    # copy nodes and edges
    for s, e, k, d in tqdm(networkX_multidigraph.edges(data=True, keys=True), disable=checks.quiet_mode):
        s_x, s_y = _process_node(s)
        e_x, e_y = _process_node(e)
        # copy edge if present
        if 'geometry' in d:
            line_geom = d['geometry']
        # otherwise create
        else:
            line_geom = geometry.LineString([[s_x, s_y], [e_x, e_y]])
        # check for LineString validity
        if line_geom.type != 'LineString':
            raise TypeError(f'Expecting LineString geometry but found {line_geom.type} geometry for edge {s}-{e}.')
        # orient LineString
        geom_coords = line_geom.coords
        if not np.allclose((s_x, s_y), geom_coords[0][:2], atol=tolerance, rtol=0):
            geom_coords = _align_linestring_coords(geom_coords, (s_x, s_y))
        # check starting and ending tolerances
        if not np.allclose((s_x, s_y), geom_coords[0][:2], atol=tolerance, rtol=0):
            raise ValueError(f"Starting node coordinates don't match LineString geometry starting coordinates.")
        if not np.allclose((e_x, e_y), geom_coords[-1][:2], atol=tolerance, rtol=0):
            raise ValueError(f"Ending node coordinates don't match LineString geometry ending coordinates.")
        # snap starting and ending coords to avoid rounding error issues
        geom_coords = _snap_linestring_startpoint(geom_coords, (s_x, s_y))
        geom_coords = _snap_linestring_endpoint(geom_coords, (e_x, e_y))
        g_multi.add_edge(s, e, key=k, geom=geometry.LineString(geom_coords))
        if edge_attributes is not None:
            for edge_att in edge_attributes:
                if edge_att not in d:
                    raise ValueError(f'Attribute {edge_att} is not available for edge {s}-{e}.')
                g_multi[s][e][k][edge_att] = d[edge_att]

    return g_multi
