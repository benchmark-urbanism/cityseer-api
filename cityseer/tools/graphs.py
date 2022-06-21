"""
Convenience functions for the preparation and conversion of `networkX` graphs to and from `cityseer` data structures.

Note that the `cityseer` network data structures can be created and manipulated directly, if so desired.

"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional, Union, cast

import geopandas as gpd
import networkx as nx
import numpy as np
import numpy.typing as npt
import utm
from shapely import coords, geometry, ops, strtree
from tqdm import tqdm

from cityseer import config, structures

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# define types
# type hack until networkx supports type-hinting
MultiGraph = Any
MultiDiGraph = Any
# coords can be 2d or 3d
NodeKey = Union[int, str]
NodeData = dict[str, Any]
EdgeType = Union[tuple[NodeKey, NodeKey], tuple[NodeKey, NodeKey, int]]
EdgeData = dict[str, Any]
EdgeMapping = tuple[NodeKey, NodeKey, int, geometry.LineString]
CoordsType = Union[tuple[float, float], tuple[float, float, float]]
AnyCoordsType = Union[list[CoordsType], npt.NDArray[np.float_], coords.CoordinateSequence]
ListCoordsType = list[CoordsType]


def nx_simple_geoms(nx_multigraph: MultiGraph) -> MultiGraph:
    """
    Generate straight-line geometries for each edge.

    Prepares "simple" straight-lined geometries spanning the `x` and `y` coordinates of each node-pair. The resultant
    edge geometry will be stored to the edge `geom` attribute.

    Parameters
    ----------
    nx_multigraph: MultiGraph
        A `networkX` `MultiGraph` with `x` and `y` node attributes.

    Returns
    -------
    MultiGraph
        A `networkX` `MultiGraph` with `shapely`
        [`Linestring`](https://shapely.readthedocs.io/en/latest/manual.html#linestrings) geometries assigned to the edge
        `geom` attributes.

    """
    if not isinstance(nx_multigraph, nx.MultiGraph):
        raise TypeError("This method requires an undirected networkX MultiGraph.")
    logger.info("Generating simple (straight) edge geometries.")
    g_multi_copy: MultiGraph = nx_multigraph.copy()

    def _process_node(nd_key: NodeKey):
        # x coordinate
        if "x" not in g_multi_copy.nodes[nd_key]:
            raise KeyError(f'Encountered node missing "x" coordinate attribute at node {nd_key}.')
        x: float = g_multi_copy.nodes[nd_key]["x"]
        # y coordinate
        if "y" not in g_multi_copy.nodes[nd_key]:
            raise KeyError(f'Encountered node missing "y" coordinate attribute at node {nd_key}.')
        y: float = g_multi_copy.nodes[nd_key]["y"]

        return x, y

    # unpack coordinates and build simple edge geoms
    remove_edges: list[tuple[NodeKey, NodeKey, int]] = []
    start_nd_key: NodeKey
    end_nd_key: NodeKey
    edge_idx: int
    for start_nd_key, end_nd_key, edge_idx in tqdm(
        g_multi_copy.edges(keys=True), disable=config.QUIET_MODE
    ):  # pylint: disable=line-too-long
        s_x, s_y = _process_node(start_nd_key)
        e_x, e_y = _process_node(end_nd_key)
        seg = geometry.LineString([[s_x, s_y], [e_x, e_y]])
        if start_nd_key == end_nd_key and seg.length == 0:
            remove_edges.append((start_nd_key, end_nd_key, edge_idx))
        else:
            g_multi_copy[start_nd_key][end_nd_key][edge_idx]["geom"] = seg
    for start_nd_key, end_nd_key, edge_idx in remove_edges:
        logger.warning(f"Found zero length looped edge for node {start_nd_key}, removing from graph.")
        g_multi_copy.remove_edge(start_nd_key, end_nd_key, key=edge_idx)

    return g_multi_copy


def _add_node(
    nx_multigraph: MultiGraph,
    nodes_names: list[NodeKey],
    x: float,
    y: float,
    live: Optional[bool] = None,
) -> Optional[str]:
    """
    Add a node to a networkX `MultiGraph`. Assembles a new name from source node names. Checks for duplicates.
    """
    # suggest a name based on the given names
    if len(nodes_names) == 1:
        new_nd_name = str(nodes_names[0])
    # if concatenating existing nodes, suggest a name based on a combination of existing names
    else:
        names = []
        for name in nodes_names:
            name = str(name)
            if len(name) > 10:
                name = f"{name[:5]}|{name[-5:]}"
            names.append(name)
        new_nd_name = "±".join(names)
    # first check whether the node already exists
    append = 2
    target_name = new_nd_name
    dupe = False
    while True:
        if f"{new_nd_name}" in nx_multigraph:
            dupe = True
            # if the coordinates also match, then it is probable that the same node is being re-added...
            nd_data: dict[str, float] = nx_multigraph.nodes[f"{new_nd_name}"]
            if nd_data["x"] == x and nd_data["y"] == y:
                logger.debug(
                    f"Proposed new node {new_nd_name} would overlay a node that already exists "
                    f"at the same coordinates. Skipping."
                )
                return None
            # otherwise, warn and bump the appended node number
            new_nd_name = f"{target_name}§v{append}"
            append += 1
        else:
            if dupe:
                logger.debug(
                    f"A node of the same name already exists in the graph, "
                    f"adding this node as {new_nd_name} instead."
                )
            break
    # add
    attributes = {"x": x, "y": y}
    if live is not None:
        attributes["live"] = live
    nx_multigraph.add_node(new_nd_name, **attributes)
    return new_nd_name


def nx_from_osm(osm_json: str) -> MultiGraph:
    """
    Generate a `NetworkX` `MultiGraph` from [Open Street Map](https://www.openstreetmap.org) data.

    Parameters
    ----------
    osm_json: str
        A `json` string response from the [OSM overpass API](https://wiki.openstreetmap.org/wiki/Overpass_API),
        consisting of `nodes` and `ways`.

    Returns
    -------
    MultiGraph
        A `NetworkX` `MultiGraph` with `x` and `y` attributes in [WGS84](https://epsg.io/4326) `lng`, `lat` geographic
        coordinates.

    """
    osm_network_data = json.loads(osm_json)
    nx_multigraph: MultiGraph = nx.MultiGraph()
    for elem in osm_network_data["elements"]:
        if elem["type"] == "node":
            nx_multigraph.add_node(elem["id"], x=elem["lon"], y=elem["lat"])
    for elem in osm_network_data["elements"]:
        if elem["type"] == "way":
            count = len(elem["nodes"])
            for idx in range(count - 1):
                nx_multigraph.add_edge(elem["nodes"][idx], elem["nodes"][idx + 1])

    return nx_multigraph


def nx_wgs_to_utm(nx_multigraph: MultiGraph, force_zone_number: Optional[int] = None) -> MultiGraph:
    """
    Convert a graph from WGS84 geographic coordinates to UTM projected coordinates.

    Converts `x` and `y` node attributes from [WGS84](https://epsg.io/4326) `lng`, `lat` geographic coordinates to the
    local UTM projected coordinate system. If edge `geom` attributes are found, the associated `LineString` geometries
    will also be converted. The UTM zone derived from the first processed node will be used for the conversion of all
    other nodes and geometries contained in the graph. This ensures consistent behaviour in cases where a graph spans
    a UTM boundary.

    Parameters
    ----------
    nx_multigraph: MultiGraph
        A `networkX` `MultiGraph` with `x` and `y` node attributes in the WGS84 coordinate system. Optional `geom` edge
        attributes containing `LineString` geoms to be converted.
    force_zone_number: int
        An optional UTM zone number for coercing all conversions to an explicit UTM zone. Use with caution: mismatched
        UTM zones may introduce substantial distortions in the results. By Default None.

    Returns
    -------
    MultiGraph
        A `networkX` `MultiGraph` with `x` and `y` node attributes converted to the local UTM coordinate system. If edge
         `geom` attributes are present, these will also be converted.

    """
    if not isinstance(nx_multigraph, nx.MultiGraph):
        raise TypeError("This method requires an undirected networkX MultiGraph.")
    logger.info("Converting networkX graph from WGS to UTM.")
    g_multi_copy: MultiGraph = nx_multigraph.copy()
    utm_zone_number = None
    utm_zone_letter = None
    if force_zone_number is not None:
        utm_zone_number = force_zone_number
    logger.info("Processing node x, y coordinates.")
    nd_key: NodeKey
    node_data: NodeData
    for nd_key, node_data in tqdm(g_multi_copy.nodes(data=True), disable=config.QUIET_MODE):
        # x coordinate
        if "x" not in node_data:
            raise KeyError(f'Encountered node missing "x" coordinate attribute at node {nd_key}.')
        lng: float = node_data["x"]
        # y coordinate
        if "y" not in node_data:
            raise KeyError(f'Encountered node missing "y" coordinate attribute at node {nd_key}.')
        lat: float = node_data["y"]
        # check for unintentional use of conversion
        if abs(lng) > 180 or abs(lat) > 90:
            raise ValueError(f"x, y coordinates {lng}, {lat} exceed WGS bounds. Please check your coordinate system.")
        # to avoid issues across UTM boundaries, use the first point to set (and subsequently force) the UTM zone
        if utm_zone_number is None:
            utm_zone_number, utm_zone_letter = utm.from_latlon(lat, lng)[2:]  # zone number is position 2
        # be cognisant of parameter and return order
        # returns in easting, northing order
        easting, northing = utm.from_latlon(lat, lng, force_zone_number=utm_zone_number)[:2]
        # write back to graph
        g_multi_copy.nodes[nd_key]["x"] = easting
        g_multi_copy.nodes[nd_key]["y"] = northing
    logger.info(f"UTM conversion info: UTM zone number: {utm_zone_number}, UTM zone letter: {utm_zone_letter}")
    # if line geom property provided, then convert as well
    logger.info("Processing edge geom coordinates, if present.")
    start_nd_key: NodeKey
    end_nd_key: NodeKey
    edge_idx: int
    edge_data: EdgeData
    for start_nd_key, end_nd_key, edge_idx, edge_data in tqdm(
        g_multi_copy.edges(data=True, keys=True), disable=config.QUIET_MODE
    ):
        # check if geom present - optional step
        if "geom" in edge_data:
            line_geom: geometry.LineString = edge_data["geom"]
            if line_geom.type != "LineString":
                raise TypeError(f"Expecting LineString geometry but found {line_geom.type} geometry.")
            # be cognisant of parameter and return order
            # returns in easting, northing order
            utm_coords = [
                utm.from_latlon(lat, lng, force_zone_number=utm_zone_number)[:2] for lng, lat in line_geom.coords
            ]
            # write back to edge
            g_multi_copy[start_nd_key][end_nd_key][edge_idx]["geom"] = geometry.LineString(utm_coords)

    return g_multi_copy


def nx_remove_dangling_nodes(
    nx_multigraph: MultiGraph,
    despine: Optional[float] = None,
    remove_disconnected: bool = True,
) -> MultiGraph:
    """
    Remove disconnected components and optionally removes short dead-end street stubs.

    Parameters
    ----------
    nx_multigraph: MultiGraph
        A `networkX` `MultiGraph` in a projected coordinate system, containing `x` and `y` node attributes, and `geom`
        edge attributes containing `LineString` geoms.
    despine: bool
        The maximum cutoff distance for removal of dead-ends. Use `None` or `0` where no despining should occur.
        Defaults to None.
    remove_disconnected: bool
        Whether to remove disconnected components. If set to `True`, only the largest connected component will be
        returned. Defaults to True.

    Returns
    -------
    MultiGraph
        A `networkX` `MultiGraph` with disconnected components optionally removed, and dead-ends removed where less than
         the `despine` parameter distance.

    """
    logger.info("Removing dangling nodes.")
    g_multi_copy: MultiGraph = nx_multigraph.copy()
    if remove_disconnected:
        # finds connected components - this behaviour changed with networkx v2.4
        connected_components: list[list[NodeKey]] = list(
            nx.algorithms.components.connected_components(g_multi_copy)  # type: ignore
        )  # pylint: disable=line-too-long
        # sort by largest component
        g_nodes: list[NodeKey] = sorted(connected_components, key=len, reverse=True)[0]
        # make a copy of the graph using the largest component
        g_multi_copy: MultiGraph = nx.MultiGraph(g_multi_copy.subgraph(g_nodes))  # type: ignore
    if despine is not None and despine > 0:
        remove_nodes = []
        nd_key: NodeKey
        for nd_key in tqdm(g_multi_copy.nodes(data=False), disable=config.QUIET_MODE):
            if nx.degree(g_multi_copy, nd_key) == 1:
                # only a single neighbour, so index-in directly and update at key = 0
                nb_nd_key: NodeKey = list(nx.neighbors(g_multi_copy, nd_key))[0]  # type: ignore
                if g_multi_copy[nd_key][nb_nd_key][0]["geom"].length <= despine:
                    remove_nodes.append(nd_key)
        g_multi_copy.remove_nodes_from(remove_nodes)

    return g_multi_copy


def _snap_linestring_idx(
    linestring_coords: AnyCoordsType,
    idx: int,
    x_y: CoordsType,
) -> ListCoordsType:
    """
    Snaps a LineString's coordinate at the specified index to the provided x_y coordinate.
    """
    # check types
    if not isinstance(linestring_coords, (list, np.ndarray, coords.CoordinateSequence)):
        raise ValueError("Expecting a list, tuple, numpy array, or shapely LineString coordinate sequence.")
    list_linestring_coords: ListCoordsType = list(linestring_coords)
    # check that the index is either 0 or -1
    if idx not in [0, -1]:
        raise ValueError('Expecting either a start index of "0" or an end index of "-1"')
    # handle 3D
    coord = list(list_linestring_coords[idx])  # tuples don't support indexed assignment
    coord[:2] = x_y
    list_linestring_coords[idx] = tuple(coord)

    return list_linestring_coords


def _snap_linestring_startpoint(
    linestring_coords: AnyCoordsType,
    x_y: CoordsType,
) -> ListCoordsType:
    """
    Snaps a LineString's start-point coordinate to a specified x_y coordinate.
    """
    return _snap_linestring_idx(linestring_coords, 0, x_y)


def _snap_linestring_endpoint(
    linestring_coords: AnyCoordsType,
    x_y: CoordsType,
) -> ListCoordsType:
    """
    Snaps a LineString's end-point coordinate to a specified x_y coordinate.
    """
    return _snap_linestring_idx(linestring_coords, -1, x_y)


def _align_linestring_coords(
    linestring_coords: AnyCoordsType,
    x_y: CoordsType,
    reverse: bool = False,
    tolerance: float = 0.5,
) -> ListCoordsType:
    """
    Align a LineString's coordinate order to either start or end at a specified x_y coordinate within a given tolerance.

    If reverse=False the coordinate order will be aligned to start from the given x_y coordinate.
    If reverse=True the coordinate order will be aligned to end at the given x_y coordinate.

    """
    # check types
    if not isinstance(linestring_coords, (list, np.ndarray, coords.CoordinateSequence)):
        raise ValueError("Expecting a list, numpy array, or shapely LineString coordinate sequence.")
    linestring_coords = list(linestring_coords)
    # the target indices depend on whether reversed or not
    if not reverse:
        xy_idx = 0
        opposite_idx = -1
    else:
        xy_idx = -1
        opposite_idx = 0
    # flip if necessary
    if np.allclose(x_y, linestring_coords[opposite_idx][:2], atol=tolerance, rtol=0):
        return linestring_coords[::-1]
    # if still not aligning, then there is an issue
    if not np.allclose(x_y, linestring_coords[xy_idx][:2], atol=tolerance, rtol=0):
        raise ValueError(f"Unable to align the LineString to starting point {x_y} given the tolerance of {tolerance}.")
    # otherwise no flipping is required and the coordinates can simply be returned
    return linestring_coords


def _weld_linestring_coords(
    linestring_coords_a: AnyCoordsType,
    linestring_coords_b: AnyCoordsType,
    force_xy: Optional[CoordsType] = None,
    tolerance: float = config.ATOL,
) -> ListCoordsType:
    """
    Welds two linestrings.

    Finds a matching start / end point combination and merges the coordinates accordingly. If the optional force_xy is
    provided then the weld will be performed at the x_y end of the LineStrings. The force_xy parameter is useful for
    looping geometries or overlapping geometries where it can happen that welding works from either of the two ends,
    thus potentially mis-aligning the start point unless explicit.

    """
    # check types
    for line_coords in [linestring_coords_a, linestring_coords_b]:
        if not isinstance(line_coords, (list, np.ndarray, coords.CoordinateSequence)):
            raise ValueError("Expecting a list, tuple, numpy array, or shapely LineString coordinate sequence.")
    linestring_coords_a = list(linestring_coords_a)
    linestring_coords_b = list(linestring_coords_b)
    # if both lists are empty, raise
    if len(linestring_coords_a) == 0 and len(linestring_coords_b) == 0:
        raise ValueError("Neither of the provided linestring coordinate lists contain any coordinates.")
    # if one of the lists is empty, return only the other
    if not linestring_coords_b:
        return linestring_coords_a
    if not linestring_coords_a:
        return linestring_coords_b
    # match the directionality of the linestrings
    # if override_xy is provided, then make sure that the sides with the specified x_y are merged
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
        raise ValueError(f"Unable to weld LineString geometries with the given tolerance of {tolerance}.")
    # drop the duplicate interleaving coordinate
    return coords_a[:-1] + coords_b


def nx_remove_filler_nodes(nx_multigraph: MultiGraph) -> MultiGraph:
    """
    Remove nodes of degree=2.

    Nodes of degree=2 represent no route-choice options other than traversal to the next edge. These are frequently
    found on network topologies as a means of describing roadway geometry, but are meaningless from a network topology
    point of view. This method will find and deleted these nodes, and replaces the two edges on either side with a new
    spliced edge. The new edge's `geom` attribute will retain the geometric properties of the original edges.

    :::note
    Filler nodes may be prevalent in poor quality datasets, or in situations where curved roadways have been represented
    through the addition of nodes to describe arced geometries. `cityseer` uses `shapely` `Linestrings` to describe
    arbitrary road geometries without the need for filler nodes. Filler nodes can therefore be removed, thus reducing
    side-effects as a function of varied node intensities when computing network centralities.
    :::

    Parameters
    ----------
    nx_multigraph: MultiGraph
        A `networkX` `MultiGraph` in a projected coordinate system, containing `x` and `y` node attributes, and `geom`
        edge attributes containing `LineString` geoms.

    Returns
    -------
    MultiGraph
        A `networkX` `MultiGraph` with nodes of degree=2 removed. Adjacent edges will be combined into a unified new
        edge with associated `geom` attributes spliced together.

    """
    if not isinstance(nx_multigraph, nx.MultiGraph):
        raise TypeError("This method requires an undirected networkX MultiGraph.")
    logger.info("Removing filler nodes.")
    g_multi_copy: MultiGraph = nx_multigraph.copy()
    removed_nodes: set[NodeKey] = set()
    # iterates the original graph, but changes are written to the copied version (to avoid in-place snafus)
    nd_key: NodeKey
    for nd_key in tqdm(nx_multigraph.nodes(), disable=config.QUIET_MODE):
        # some nodes will already have been removed
        if nd_key in removed_nodes:
            continue
        # proceed if a "simple" node is discovered, i.e. degree = 2
        if nx.degree(nx_multigraph, nd_key) == 2:
            # pick the first neighbour and follow the chain until a non-simple node is encountered
            # this will become the starting point of the chain of simple nodes to be consolidated
            nbs: list[NodeKey] = list(nx.neighbors(nx_multigraph, nd_key))  # type: ignore
            # catch the edge case where the a single dead-end node has two out-edges to a single neighbour
            if len(nbs) == 1:
                continue
            # otherwise randomly select one side and find a non-simple node as a starting point.
            nb_nd_key = nbs[0]
            # anchor_nd should be the first node of the chain of nodes to be merged, and should be a non-simple node
            anchor_nd: Optional[NodeKey] = None
            # next_link_nd should be a direct neighbour of anchor_nd and must be a simple node
            next_link_nd: NodeKey = nd_key
            # find the non-simple start node
            while anchor_nd is None:
                # follow the chain of neighbours and break once a non-simple node is found
                # catch disconnected looping components by checking for re-encountering start-node
                if nx.degree(nx_multigraph, nb_nd_key) != 2 or nb_nd_key == nd_key:
                    anchor_nd = nb_nd_key
                    break
                # probe neighbours in one-direction only - i.e. don't backtrack
                nb_a: NodeKey
                nb_b: NodeKey
                nb_a, nb_b = list(nx.neighbors(nx_multigraph, nb_nd_key))  # type: ignore
                if nb_a == next_link_nd:
                    next_link_nd = nb_nd_key
                    nb_nd_key = nb_b
                else:
                    next_link_nd = nb_nd_key
                    nb_nd_key = nb_a
            # from anchor_nd, proceed along the chain in the next_link_nd direction
            # accumulate and weld geometries along the way
            # break once finding another non-simple node
            trailing_nd: NodeKey = anchor_nd
            end_nd: Optional[NodeKey] = None
            drop_nodes: list[NodeKey] = []
            agg_geom: ListCoordsType = []
            while True:
                # aggregate the geom
                try:
                    # there is ordinarily a single edge from trailing to next
                    # however, there is an edge case where next is a dead-end with two edges linking back to trailing
                    # (i.e. where one of those edges is longer than the maximum length discrepancy for merging edges)
                    # in either case, use the first geom
                    geom: geometry.LineString = nx_multigraph[trailing_nd][next_link_nd][0]["geom"]
                except KeyError as err:
                    raise KeyError(f'Missing "geom" attribute for edge {trailing_nd}-{next_link_nd}') from err
                if geom.type != "LineString":
                    raise TypeError(f"Expecting LineString geometry but found {geom.type} geometry.")
                # welds can be done automatically, but there are edge cases, e.g.:
                # looped roadways or overlapping edges such as stairways don't know which sides of two segments to join
                # i.e. in these cases the edges can sometimes be matched from one of two possible configurations
                # since the x_y join is known for all cases it is used here regardless
                override_xy: CoordsType = (
                    cast(float, nx_multigraph.nodes[trailing_nd]["x"]),
                    cast(float, nx_multigraph.nodes[trailing_nd]["y"]),
                )
                # weld
                agg_geom = _weld_linestring_coords(agg_geom, geom.coords, force_xy=override_xy)
                # if the next node has a degree other than 2, then break
                # for circular components, break if the next node matches the start node
                if nx.degree(nx_multigraph, next_link_nd) != 2 or next_link_nd == anchor_nd:
                    end_nd = next_link_nd
                    break
                # otherwise, follow the chain
                # add next_link_nd to drop list
                drop_nodes.append(next_link_nd)
                # get the next set of neighbours
                # in the above-mentioned edge-case, a single dead-end node with two edges back to a start node
                # will only have one neighbour
                new_nbs: list[NodeKey] = list(nx.neighbors(nx_multigraph, next_link_nd))  # type: ignore
                if len(new_nbs) == 1:
                    trailing_nd = next_link_nd
                    next_link_nd = new_nbs[0]
                # but in almost all cases there will be two neighbours, one of which will be the previous node
                else:
                    nb_a, nb_b = list(nx.neighbors(nx_multigraph, next_link_nd))  # type: ignore
                    # proceed to the new_next node
                    if nb_a == trailing_nd:
                        trailing_nd = next_link_nd
                        next_link_nd = nb_b
                    else:
                        trailing_nd = next_link_nd
                        next_link_nd = nb_a
            # double-check that the geom's endpoints match within tolerance
            # then snap to remove any potential side-effects from minor tolerance issues
            s_xy: CoordsType = (
                cast(float, nx_multigraph.nodes[anchor_nd]["x"]),
                cast(float, nx_multigraph.nodes[anchor_nd]["y"]),
            )
            if not np.allclose(agg_geom[0], s_xy, atol=config.ATOL, rtol=config.RTOL):
                raise ValueError("New Linestring geometry does not match starting node coordinates.")
            agg_geom = _snap_linestring_startpoint(agg_geom, s_xy)
            e_xy: CoordsType = (
                cast(float, nx_multigraph.nodes[end_nd]["x"]),
                cast(float, nx_multigraph.nodes[end_nd]["y"]),
            )
            if not np.allclose(agg_geom[-1], e_xy, atol=config.ATOL, rtol=config.RTOL):
                raise ValueError("New Linestring geometry does not match ending node coordinates.")
            agg_geom = _snap_linestring_endpoint(agg_geom, e_xy)
            # create a new linestring
            new_geom = geometry.LineString(agg_geom)
            if new_geom.type != "LineString":
                raise TypeError(
                    f'Found {new_geom.type} geometry instead of "LineString" for new geom {new_geom.wkt}.'
                    f"Check that the adjacent LineStrings in the vicinity of {nd_key} are not corrupted."
                )
            # add a new edge from anchor_nd to end_nd
            g_multi_copy.add_edge(anchor_nd, end_nd, geom=new_geom)
            # drop the removed nodes, which will also implicitly drop the related edges
            g_multi_copy.remove_nodes_from(drop_nodes)
            removed_nodes.update(drop_nodes)

    return g_multi_copy


def _squash_adjacent(
    nx_multigraph: MultiGraph,
    node_group: list[NodeKey],
    cent_min_degree: Optional[int] = None,
    cent_min_len_factor: Optional[float] = None,
) -> MultiGraph:
    """
    Squash nodes from a specified node group down to a new node.

    The new node can either be based on:
    - The centroid of all nodes;
    - Else, all nodes of degree greater or equal to cent_min_degree;
    - Else, all nodes with aggregate adjacent edge lengths greater than cent_min_len_factor as a factor of the node with
      the greatest overall aggregate lengths. Edges are adjusted from the old nodes to the new combined node.

    """
    if not isinstance(nx_multigraph, nx.MultiGraph):
        raise TypeError("This method requires an undirected networkX MultiGraph (for multiple edges).")
    if cent_min_degree is not None and cent_min_degree < 1:
        raise ValueError("merge_node_min_degree should be a positive integer.")
    if cent_min_len_factor is not None and not 1 >= cent_min_len_factor >= 0:
        raise ValueError("cent_min_len_factor should be a decimal between 0 and 1.")
    # remove any node keys no longer in the graph
    node_group = [nd_key for nd_key in node_group if nd_key in nx_multigraph]
    # filter out nodes if using cent_min_degree or cent_min_len_factor
    filtered_nodes: list[NodeKey] = []
    if cent_min_degree is not None:
        for nd_key in node_group:
            if nx.degree(nx_multigraph, nd_key) >= cent_min_degree:
                filtered_nodes.append(nd_key)
    # else if merging on a longest adjacent edges basis
    if cent_min_len_factor is not None:
        # if nodes are pre-filtered by edge degrees, then use the filtered nodes as a starting point
        if filtered_nodes:
            node_pool = filtered_nodes.copy()
            filtered_nodes = []  # reset
        # else use the full original node group
        else:
            node_pool = node_group
        agg_lens: list[int] = []
        for nd_key in node_pool:
            agg_len = 0
            # iterate each node's neighbours, aggregating neighbouring edge lengths along the way
            nb_nd_key: NodeKey
            for nb_nd_key in nx.neighbors(nx_multigraph, nd_key):
                nb_edge_data: EdgeData
                for nb_edge_data in nx_multigraph[nd_key][nb_nd_key].values():
                    agg_len += nb_edge_data["geom"].length
            agg_lens.append(agg_len)
        # find the longest
        max_len: int = max(agg_lens)
        # select all nodes with an agg_len within a small tolerance of longest
        nd_key: NodeKey
        agg_len: int
        for nd_key, agg_len in zip(node_pool, agg_lens):
            if agg_len >= max_len * cent_min_len_factor:
                filtered_nodes.append(nd_key)
    # otherwise, derive the centroid from all nodes
    # this is also a fallback if no nodes selected via minimum degree basis
    if not filtered_nodes:
        filtered_nodes = node_group
    # prepare the names and geoms for all points used for the new centroid
    node_geoms = []
    coords_set: set[str] = set()
    for nd_key in filtered_nodes:
        x: float = nx_multigraph.nodes[nd_key]["x"]
        y: float = nx_multigraph.nodes[nd_key]["y"]
        # in rare cases opposing geom splitting can cause overlaying nodes
        # these can swing the gravity of multipoint centroids so screen these out
        xy_key: str = f"{round(x)}-{round(y)}"
        if xy_key in coords_set:
            continue
        coords_set.add(xy_key)
        node_geoms.append(geometry.Point(x, y))
    # set the new centroid from the centroid of the node group's Multipoint:
    c: geometry.Point = geometry.MultiPoint(node_geoms).centroid  # type: ignore
    # now that the centroid is known, go ahead and merge the _node_group
    # add the new node
    new_nd_name = _add_node(nx_multigraph, node_group, x=c.x, y=c.y)  # pylint: disable=no-member
    if new_nd_name is None:
        raise ValueError(f"Attempted to add a duplicate node for node_group {node_group}.")
    # iterate the nodes to be removed and connect their existing edge geometries to the new centroid
    for nd_key in node_group:
        # iterate the node's existing neighbours
        for nb_nd_key in nx.neighbors(nx_multigraph, nd_key):
            # if a neighbour is also going to be dropped, then no need to create new between edges
            # an exception exists when a geom is looped, in which case the neighbour is also the current node
            if nb_nd_key in node_group and nb_nd_key != nd_key:
                continue
            # MultiGraph - so iter edges
            edge_data: EdgeData
            for edge_data in nx_multigraph[nd_key][nb_nd_key].values():
                if "geom" not in edge_data:
                    raise KeyError(f'Missing "geom" attribute for edge {nd_key}-{nb_nd_key}')
                line_geom: geometry.LineString = edge_data["geom"]
                if line_geom.type != "LineString":
                    raise TypeError(
                        f"Expecting LineString geometry but found {line_geom.type} geometry "
                        f"for edge {nd_key}-{nb_nd_key}."
                    )
                # orient the LineString so that the starting point matches the node's x_y
                s_xy: CoordsType = (
                    cast(float, nx_multigraph.nodes[nd_key]["x"]),
                    cast(float, nx_multigraph.nodes[nd_key]["y"]),
                )
                line_coords = _align_linestring_coords(line_geom.coords, s_xy)
                # update geom starting point to new parent node's coordinates
                line_coords = _snap_linestring_startpoint(line_coords, (c.x, c.y))  # pylint: disable=no-member
                # if self-loop, then the end also needs updating
                if nd_key == nb_nd_key:
                    line_coords = _snap_linestring_endpoint(line_coords, (c.x, c.y))  # pylint: disable=no-member
                    target_nd_key = new_nd_name
                else:
                    target_nd_key = nb_nd_key
                # build the new geom
                new_edge_geom = geometry.LineString(line_coords)
                # check that a duplicate is not being added
                dupe = False
                if nx_multigraph.has_edge(new_nd_name, target_nd_key):
                    # only add parallel edges if substantially different from any existing edges
                    n_edges: int = nx_multigraph.number_of_edges(new_nd_name, target_nd_key)  # type: ignore
                    for edge_idx in range(n_edges):
                        exist_geom: geometry.LineString = nx_multigraph[new_nd_name][target_nd_key][edge_idx]["geom"]
                        # don't add if the edges have the same number of coords and the coords are similar
                        # 5m x and y tolerance across all coordinates
                        if len(new_edge_geom.coords) == len(exist_geom.coords) and np.allclose(
                            new_edge_geom.coords,
                            exist_geom.coords,
                            atol=5,
                            rtol=0,
                        ):
                            dupe = True
                            logger.debug(
                                f"Not adding edge {new_nd_name} to {target_nd_key}: a similar edge already exists. "
                                f"Length: {new_edge_geom.length} vs. {exist_geom.length}. "
                                f"Num coords: {len(new_edge_geom.coords)} vs. {len(exist_geom.coords)}."
                            )
                if not dupe:
                    # add the new edge
                    nx_multigraph.add_edge(new_nd_name, target_nd_key, geom=new_edge_geom)
        # drop the node, this will also implicitly drop the old edges
        nx_multigraph.remove_node(nd_key)

    return nx_multigraph


def _merge_parallel_edges(
    nx_multigraph: MultiGraph,
    merge_edges_by_midline: bool,
    multi_edge_len_factor: float,
    multi_edge_min_len: float,
) -> MultiGraph:
    """
    Check a MultiGraph for duplicate edges; which, if found, will be consolidated.

    If merge_edges_by_midline is False, then the shortest of the edges is used and the others are simply dropped.
    If merge_edges_by_midline is True, then the duplicates are replaced with a new edge following the merged centreline.
    In cases where one line is significantly longer than another (e.g. a crescent streets),
    then the longer edge is retained as separate if exceeding the multi_edge_len_factor as a factor of the shortest
    length but with the exception that (longer) edges still shorter than multi_edge_min_len are removed regardless.

    """
    if not isinstance(nx_multigraph, nx.MultiGraph):
        raise TypeError("This method requires an undirected networkX MultiGraph (for multiple edges).")
    if multi_edge_len_factor <= 1:
        raise TypeError("multi_edge_len_factor should be a factor greater than 1. ")
    if multi_edge_len_factor < 1.25:
        logger.warning(
            "Merging by midline and setting multi_edge_len_factor too low (e.g. lower than 1.25) may "
            "result in an undesirable number of relatively similar parallel edges."
        )
    # don't use copy() - add nodes only
    deduped_graph = nx.MultiGraph()
    deduped_graph.add_nodes_from(nx_multigraph.nodes(data=True))
    # iter the edges
    start_nd_key: NodeKey
    end_nd_key: NodeKey
    edge_data: EdgeData
    for start_nd_key, end_nd_key, edge_data in tqdm(
        nx_multigraph.edges(data=True), disable=config.QUIET_MODE
    ):  # pylint: disable=line-too-long
        # if only one edge is associated with this node pair, then add
        if nx_multigraph.number_of_edges(start_nd_key, end_nd_key) == 1:
            deduped_graph.add_edge(start_nd_key, end_nd_key, **edge_data)
        # otherwise, add if not already added from another (parallel) edge
        elif not deduped_graph.has_edge(start_nd_key, end_nd_key):
            # there are normally two edges, but sometimes three or possibly more
            edges_data: list[EdgeData] = nx_multigraph[start_nd_key][end_nd_key].values()
            # find the shortest of the geoms
            edge_geoms = [edge["geom"] for edge in edges_data]
            edge_lens = [geom.length for geom in edge_geoms]
            shortest_idx = edge_lens.index(min(edge_lens))
            shortest_len = edge_lens.pop(shortest_idx)
            shortest_geom = edge_geoms.pop(shortest_idx)
            longer_geoms: list[geometry.LineString] = []
            for edge_len, edge_geom in zip(edge_lens, edge_geoms):
                # retain distinct edges where they are substantially longer than the shortest geom
                if edge_len > shortest_len * multi_edge_len_factor and edge_len > multi_edge_min_len:
                    deduped_graph.add_edge(start_nd_key, end_nd_key, geom=edge_geom)
                # otherwise, add to the list of longer geoms to be merged along with shortest
                else:
                    longer_geoms.append(edge_geom)
            # otherwise, if not merging on a midline basis
            # or, if no other edges to process (in cases where longer geom has been retained per above)
            # then use the shortest geom
            if not merge_edges_by_midline or len(longer_geoms) == 0:
                deduped_graph.add_edge(start_nd_key, end_nd_key, geom=shortest_geom)
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
                deduped_graph.add_edge(start_nd_key, end_nd_key, geom=new_geom)

    return deduped_graph


def _create_nodes_strtree(nx_multigraph: MultiGraph) -> strtree.STRtree:
    """
    Create a nodes-based STRtree spatial index.
    """
    point_geoms = []
    nd_key: NodeKey
    node_data: NodeData
    for nd_key, node_data in nx_multigraph.nodes(data=True):  # type: ignore
        # x coordinate
        if "x" not in node_data:  # type: ignore
            raise KeyError(f'Encountered node missing "x" coordinate attribute at node {nd_key}.')
        x: float = node_data["x"]  # type: ignore
        # y coordinate
        if "y" not in node_data:  # type: ignore
            raise KeyError(f'Encountered node missing "y" coordinate attribute at node {nd_key}.')
        y: float = node_data["y"]  # type: ignore
        point_geom = geometry.Point(x, y)
        point_geom.nd_key = nd_key
        point_geom.degree = nx.degree(nx_multigraph, nd_key)
        point_geoms.append(point_geom)
    return strtree.STRtree(point_geoms)


def _create_edges_strtree(nx_multigraph: MultiGraph) -> strtree.STRtree:
    """
    Create an edges-based STRtree spatial index.
    """
    lines = []
    start_nd_key: NodeKey
    end_nd_key: NodeKey
    edge_idx: int
    edge_data: EdgeData
    for start_nd_key, end_nd_key, edge_idx, edge_data in nx_multigraph.edges(keys=True, data=True):  # type: ignore
        if "geom" not in edge_data:  # type: ignore
            raise KeyError('Encountered edge missing "geom" attribute.')
        linestring = edge_data["geom"]  # type: ignore
        linestring.start_nd_key = start_nd_key
        linestring.end_nd_key = end_nd_key
        linestring.edge_idx = edge_idx
        lines.append(linestring)
    return strtree.STRtree(lines)


def nx_consolidate_nodes(
    nx_multigraph: MultiGraph,
    buffer_dist: float = 5,
    min_node_group: int = 2,
    min_node_degree: int = 1,
    min_cumulative_degree: Optional[int] = None,
    max_cumulative_degree: Optional[int] = None,
    neighbour_policy: Optional[str] = None,
    crawl: bool = True,
    cent_min_degree: int = 3,
    cent_min_len_factor: Optional[float] = None,
    merge_edges_by_midline: bool = True,
    multi_edge_len_factor: float = 1.25,
    multi_edge_min_len: float = 100,
) -> MultiGraph:
    """
    Consolidates nodes if they are within a buffer distance of each other.

    Several parameters provide more control over the conditions used for deciding whether or not to merge nodes. The
    algorithm proceeds in two steps:

    Nodes within the buffer distance of each other are merged. A new centroid will be determined and all existing
    edge endpoints will be updated accordingly. The new centroid for the merged nodes can be based on:
    - The centroid of the node group;
    - Else, all nodes of degree greater or equal to `cent_min_degree`;
    - Else, all nodes with aggregate adjacent edge lengths greater than a factor of `cent_min_len_factor` of the node
      with the greatest aggregate length for adjacent edges.

    The merging of nodes creates parallel edges which may start and end at a shared node on either side. These edges
    are replaced by a single new edge, with the new geometry selected from either:
    - An imaginary centreline of the combined edges if `merge_edges_by_midline` is set to `True`;
    - Else, the shortest edge, with longer edges discarded;
    - Note that substantially longer parallel edges are retained, instead of discarded, if they exceed
      `multi_edge_len_factor` and are longer than `multi_edge_min_len`.

    Parameters
    ----------
    nx_multigraph: MultiGraph
        A `networkX` `MultiGraph` in a projected coordinate system, containing `x` and `y` node attributes, and `geom`
        edge attributes containing `LineString` geoms.
    buffer_dist: float
        The buffer distance to be used for consolidating nearby nodes. Defaults to 5.
    min_node_group: int
        The minimum number of nodes to consider a valid group for consolidation. Defaults to 2.
    min_node_degree: int
        The least number of edges a node should have in order to be considered for consolidation. Defaults to 1.
    min_cumulative_degree: int
        An optional minimum cumulative degree to consider a valid node group for consolidation. Defaults to None.
    max_cumulative_degree: int
        An optional maximum cumulative degree to consider a valid node group for consolidation. Defaults to None.
    neighbour_policy: str
        Whether all nodes within the buffer distance are merged, or only "direct" or "indirect" neighbours. Defaults to
        None.
    crawl: bool
        Whether the algorithm will recursively explore neighbours of neighbours if those neighbours are within the
        buffer distance from the prior node. Defaults to True.
    cent_min_degree: int
        The minimum node degree for a node to be considered when calculating the new centroid for the merged node
        cluster. Defaults to 3.
    cent_min_len_factor: float
        The minimum aggregate adjacent edge lengths an existing node should have to be considered when calculating the
        centroid for the new node cluster. Expressed as a factor of the node with the greatest aggregate adjacent edge
        lengths. Defaults to None.
    merge_edges_by_midline: bool
        Whether to merge parallel edges by an imaginary centreline. If set to False, then the shortest edge will be
        retained as the new geometry and the longer edges will be discarded. Defaults to True.
    multi_edge_len_factor: float
        In cases where one line is significantly longer than another (e.g. crescent streets) then the longer edge is
        retained as separate if exceeding the multi_edge_len_factor as a factor of the shortest length but with the
        exception that (longer) edges still shorter than multi_edge_min_len are removed regardless. Defaults to 1.5.
    multi_edge_min_len: float
        See `multi_edge_len_factor`. Defaults to 100.

    Returns
    -------
    MultiGraph
        A `networkX` `MultiGraph` with consolidated nodes.

    Examples
    --------
    See the guide on [graph cleaning](/guide/#graph-cleaning) for more information.

    ![Example raw graph from OSM](/images/graph_cleaning_1.png)
    _The pre-consolidation OSM street network for Soho, London. © OpenStreetMap contributors._

    ![Example cleaned graph](/images/graph_cleaning_5.png)
    _The consolidated OSM street network for Soho, London. © OpenStreetMap contributors._

    """
    if not isinstance(nx_multigraph, nx.MultiGraph):
        raise TypeError("This method requires an undirected networkX MultiGraph.")
    if min_node_group < 2:
        raise ValueError("The minimum node threshold should be set to at least two.")
    if neighbour_policy is not None and neighbour_policy not in ("direct", "indirect"):
        raise ValueError('Neighbour policy should be one "direct", "indirect", else the default of "None"')
    if crawl and buffer_dist > 25:
        logger.warning("Be cautious with the buffer distance when using crawl.")
    logger.info("Consolidating nodes.")
    _multi_graph: MultiGraph = nx_multigraph.copy()
    # create a nodes STRtree
    nodes_tree = _create_nodes_strtree(_multi_graph)
    # keep track of removed nodes
    removed_nodes: set[NodeKey] = set()

    def recursive_squash(
        nd_key: NodeKey,
        x: float,
        y: float,
        node_group: list[NodeKey],
        processed_nodes: list[NodeKey],
        recursive: bool = False,
    ) -> list[NodeKey]:
        # keep track of which nodes have been processed as part of recursion
        processed_nodes.append(nd_key)
        # get all other nodes within buffer distance - the self-node and previously processed nodes are also returned
        js = nodes_tree.query(geometry.Point(x, y).buffer(buffer_dist))
        # review each node within the buffer
        for j in js:
            j_nd_key: NodeKey = j.nd_key  # type: ignore
            if j_nd_key in removed_nodes or j_nd_key in processed_nodes or j.degree < min_node_degree:  # type: ignore
                continue
            # check neighbour policy
            if neighbour_policy is not None:
                # use the original graph prior to in-place modifications
                neighbours: list[NodeKey] = nx.neighbors(nx_multigraph, nd_key)
                if neighbour_policy == "indirect" and j_nd_key in neighbours:
                    continue
                if neighbour_policy == "direct" and j_nd_key not in neighbours:
                    continue
            # otherwise add the node
            node_group.append(j_nd_key)
            # if recursive, follow the chain
            if recursive:
                j_nd_data: NodeData = nx_multigraph.nodes[j_nd_key]
                return recursive_squash(
                    j_nd_key,
                    j_nd_data["x"],
                    j_nd_data["y"],
                    node_group,
                    processed_nodes,
                    recursive=crawl,
                )

        return node_group

    # iterate origin graph (else node structure changes in place)
    nd_key: NodeKey
    nd_data: NodeData
    for nd_key, nd_data in tqdm(nx_multigraph.nodes(data=True), disable=config.QUIET_MODE):
        # skip if already consolidated from an adjacent node, or if the node's degree doesn't meet min_node_degree
        if nd_key in removed_nodes or nx.degree(nx_multigraph, nd_key) < min_node_degree:
            continue
        node_group = recursive_squash(
            nd_key,  # node nd_key
            nd_data["x"],  # x point for buffer
            nd_data["y"],  # y point for buffer
            [nd_key],  # node group for consolidation (with starting node)
            [],  # processed nodes tracked through recursion
            crawl,
        )  # whether to recursively probe neighbours per distance
        # check for min_node_threshold
        if len(node_group) < min_node_group:
            continue
        # check for cumulative degree thresholds if requested
        if min_cumulative_degree is not None or max_cumulative_degree is not None:
            gather_degrees: list[int] = [nx.degree(nx_multigraph, nd_key) for nd_key in node_group]
            cumulative_degree: int = sum(gather_degrees)
            if min_cumulative_degree is not None and cumulative_degree < min_cumulative_degree:
                continue
            if max_cumulative_degree is not None and cumulative_degree > max_cumulative_degree:
                continue
        # update removed nodes
        removed_nodes.update(node_group)
        # consolidate if nodes have been identified within buffer and if these exceed min_node_threshold
        _multi_graph = _squash_adjacent(_multi_graph, node_group, cent_min_degree, cent_min_len_factor)
    # remove filler nodes
    deduped_graph = nx_remove_filler_nodes(_multi_graph)
    # remove any parallel edges that may have resulted from squashing nodes
    deduped_graph = _merge_parallel_edges(
        deduped_graph, merge_edges_by_midline, multi_edge_len_factor, multi_edge_min_len
    )

    return deduped_graph


def nx_split_opposing_geoms(
    nx_multigraph: MultiGraph,
    buffer_dist: float = 10,
    merge_edges_by_midline: bool = True,
    multi_edge_len_factor: float = 1.25,
    multi_edge_min_len: float = 100,
) -> MultiGraph:
    """
    Split edges opposite nodes on parallel edge segments if within a buffer distance.

    This facilitates merging parallel roadways through subsequent use of
    [`nx-consolidate-nodes`](#nx-consolidate-nodes).

    Parameters
    ----------
    nx_multigraph: MultiGraph
        A `networkX` `MultiGraph` in a projected coordinate system, containing `x` and `y` node attributes, and `geom`
        edge attributes containing `LineString` geoms.
    buffer_dist: int
        The buffer distance to be used for splitting nearby nodes. Defaults to 5.
    merge_edges_by_midline: bool
        Whether to merge parallel edges by an imaginary centreline. If set to False, then the shortest edge will be
        retained as the new geometry and the longer edges will be discarded. Defaults to True.
    multi_edge_len_factor: float
        In cases where one line is significantly longer than another (e.g. crescent streets) then the longer edge is
        retained as separate if exceeding the `multi_edge_len_factor` as a factor of the shortest length but with the
        exception that (longer) edges still shorter than `multi_edge_min_len` are removed regardless. Defaults to 1.5.
    multi_edge_min_len: float
        See `multi_edge_len_factor`. Defaults to 100.

    Returns
    -------
    MultiGraph
        A `networkX` `MultiGraph` with consolidated nodes.

    """

    def make_edge_key(start_nd_key: NodeKey, end_nd_key: NodeKey, edge_idx: int) -> str:
        return "-".join(sorted([str(start_nd_key), str(end_nd_key)])) + f"-k{edge_idx}"

    # where edges are deleted, keep track of new children edges
    edge_children: dict[str, list[EdgeMapping]] = {}

    # recursive function for retrieving nested layers of successively replaced edges
    def recurse_child_keys(
        start_nd_key: NodeKey,
        end_nd_key: NodeKey,
        edge_idx: int,
        geom: geometry.LineString,
        current_edges: list[EdgeMapping],
    ):
        """
        Recursively checks if an edge has been replaced by children, if so, use children instead.
        """
        edge_key = make_edge_key(start_nd_key, end_nd_key, edge_idx)
        # if an edge does not have children, add to current_edges and return
        if edge_key not in edge_children:
            current_edges.append((start_nd_key, end_nd_key, edge_idx, geom))
        # otherwise recursively drill-down until newest edges are found
        else:
            for child_s, child_e, child_k, child_geom in edge_children[edge_key]:
                recurse_child_keys(child_s, child_e, child_k, child_geom, current_edges)

    if not isinstance(nx_multigraph, nx.MultiGraph):
        raise TypeError("This method requires an undirected networkX MultiGraph.")
    logger.info("Splitting opposing edges.")
    _multi_graph: MultiGraph = nx_multigraph.copy()
    # create an edges STRtree (nodes and edges)
    edges_tree = _create_edges_strtree(_multi_graph)
    # iterate origin graph (else node structure changes in place)
    nd_key: NodeKey
    nd_data: NodeData
    for nd_key, nd_data in tqdm(nx_multigraph.nodes(data=True), disable=config.QUIET_MODE):
        # don't split opposing geoms from nodes of degree 1
        if nx.degree(_multi_graph, nd_key) < 2:
            continue
        # get all other edges within the buffer distance
        # the spatial index using bounding boxes, so further filtering is required (see further down)
        # furthermore, successive iterations may remove old edges, so keep track of removed parent vs new child edges
        n_point = geometry.Point(nd_data["x"], nd_data["y"])
        # spatial query from point returns all buffers with buffer_dist
        edge_geoms: list[geometry.LineString] = edges_tree.query(n_point.buffer(buffer_dist))  # type: ignore
        # extract the start node, end node, geom
        edges: list[EdgeMapping] = [
            (edge_geom.start_nd_key, edge_geom.end_nd_key, edge_geom.edge_idx, edge_geom)  # type: ignore
            for edge_geom in edge_geoms  # pylint: disable=line-too-long
        ]
        # check against removed edges
        current_edges: list[EdgeMapping] = []
        for start_nd_key, end_nd_key, edge_idx, edge_geom in edges:
            recurse_child_keys(start_nd_key, end_nd_key, edge_idx, edge_geom, current_edges)
        # get neighbouring nodes from new graph
        neighbours: list[NodeKey] = list(_multi_graph.neighbors(nd_key))  # type: ignore
        # abort if only direct neighbours
        if len(current_edges) <= len(neighbours):
            continue
        # filter current_edges
        gapped_edges: list[EdgeMapping] = []
        for start_nd_key, end_nd_key, edge_idx, edge_geom in current_edges:
            # skip direct neighbours
            if start_nd_key == nd_key or end_nd_key == nd_key:  # pylint: disable=consider-using-in
                continue
            # check whether the geom is truly within the buffer distance
            if edge_geom.distance(n_point) > buffer_dist:
                continue
            gapped_edges.append((start_nd_key, end_nd_key, edge_idx, edge_geom))
        # abort if no gapped edges
        if not gapped_edges:
            continue
        # prepare the root node's point geom
        n_geom = geometry.Point(nd_data["x"], nd_data["y"])
        # iter gapped edges
        for start_nd_key, end_nd_key, edge_idx, edge_geom in gapped_edges:
            # see if start node is within buffer distance already
            s_nd_data: NodeData = _multi_graph.nodes[start_nd_key]
            s_nd_geom = geometry.Point(s_nd_data["x"], s_nd_data["y"])
            if s_nd_geom.distance(n_geom) <= buffer_dist:
                continue
            # likewise for end node
            e_nd_data: NodeData = _multi_graph.nodes[end_nd_key]
            e_nd_geom = geometry.Point(e_nd_data["x"], e_nd_data["y"])
            if e_nd_geom.distance(n_geom) <= buffer_dist:
                continue
            # otherwise, project a point and split the opposing geom
            # ops.nearest_points returns tuple of nearest from respective input geoms
            # want the nearest point on the line at index 1
            nearest_point = ops.nearest_points(n_geom, edge_geom)[-1]
            # if a valid nearest point has been found, go ahead and split the geom
            # use a snap because rounding precision errors will otherwise cause issues
            split_geoms: geometry.GeometryCollection = ops.split(
                ops.snap(edge_geom, nearest_point, 0.01), nearest_point
            )
            # in some cases the line will be pointing away, but is still near enough to be within max
            # in these cases a single geom will be returned
            if len(split_geoms.geoms) < 2:  # type: ignore
                continue
            new_edge_geom_a: geometry.LineString
            new_edge_geom_b: geometry.LineString
            new_edge_geom_a, new_edge_geom_b = split_geoms.geoms
            # add the new node and edges to _multi_graph (don't modify nx_multigraph because of iter in place)
            new_nd_name = _add_node(
                _multi_graph, [start_nd_key, nd_key, end_nd_key], x=nearest_point.x, y=nearest_point.y
            )
            # if a node already exists at this location, add_node will return None
            if new_nd_name is None:
                continue
            _multi_graph.add_edge(start_nd_key, new_nd_name)
            _multi_graph.add_edge(end_nd_key, new_nd_name)
            if np.allclose(s_nd_geom.coords, new_edge_geom_a.coords[0][:2], atol=config.ATOL, rtol=0,) or np.allclose(
                s_nd_geom.coords,
                new_edge_geom_a.coords[-1][:2],
                atol=config.ATOL,
                rtol=0,
            ):
                s_new_geom = new_edge_geom_a
                e_new_geom = new_edge_geom_b
            else:
                # double check matching geoms
                if not np.allclose(
                    s_nd_geom.coords,
                    new_edge_geom_b.coords[0][:2],
                    atol=config.ATOL,
                    rtol=0,
                ) and not np.allclose(
                    s_nd_geom.coords,
                    new_edge_geom_b.coords[-1][:2],
                    atol=config.ATOL,
                    rtol=0,
                ):
                    raise ValueError("Unable to match split geoms to existing nodes")
                s_new_geom = new_edge_geom_b
                e_new_geom = new_edge_geom_a
            # if splitting a looped component, then both new edges will have the same starting and ending nodes
            # in these cases, there will be multiple edges
            if start_nd_key == end_nd_key:
                if _multi_graph.number_of_edges(start_nd_key, new_nd_name) != 2:
                    raise ValueError(f"Number of edges between {start_nd_key} and {new_nd_name} does not equal 2")
                s_k = 0
                e_k = 1
            else:
                if _multi_graph.number_of_edges(start_nd_key, new_nd_name) != 1:
                    raise ValueError(f"Number of edges between {start_nd_key} and {new_nd_name} does not equal 1.")
                if _multi_graph.number_of_edges(end_nd_key, new_nd_name) != 1:
                    raise ValueError(f"Number of edges between {end_nd_key} and {new_nd_name} does not equal 1.")
                s_k = e_k = 0
            # write the new edges
            _multi_graph[start_nd_key][new_nd_name][s_k]["geom"] = s_new_geom
            _multi_graph[end_nd_key][new_nd_name][e_k]["geom"] = e_new_geom
            # add the new edges to the edge_children dictionary
            edge_key = make_edge_key(start_nd_key, end_nd_key, edge_idx)
            edge_children[edge_key] = [
                (start_nd_key, new_nd_name, s_k, s_new_geom),
                (end_nd_key, new_nd_name, e_k, e_new_geom),
            ]
            # drop the old edge from _multi_graph
            if _multi_graph.has_edge(start_nd_key, end_nd_key, edge_idx):
                _multi_graph.remove_edge(start_nd_key, end_nd_key, edge_idx)
    # squashing nodes can result in edge duplicates
    deduped_graph = _merge_parallel_edges(
        _multi_graph, merge_edges_by_midline, multi_edge_len_factor, multi_edge_min_len
    )

    return deduped_graph


def nx_decompose(nx_multigraph: MultiGraph, decompose_max: float) -> MultiGraph:
    """
    Decomposes a graph so that no edge is longer than a set maximum.

    Decomposition provides a more granular representation of potential variations along street lengths, while reducing
    network centrality side-effects that arise as a consequence of varied node densities.

    :::note
    Setting the `decompose` parameter too small in relation to the size of the graph may increase the computation time
    unnecessarily for subsequent analysis. For larger-scale urban analysis, it is generally not necessary to go smaller
    20m, and 50m may already be sufficient for the majority of cases.
    :::

    Parameters
    ----------
    nx_multigraph: MultiGraph
        A `networkX` `MultiGraph` in a projected coordinate system, containing `x` and `y` node attributes, and `geom`
        edge attributes containing `LineString` geoms.

    decompose_max: float
        The maximum length threshold for decomposed edges.

    Returns
    -------
    MultiGraph
        A decomposed `networkX` graph with no edge longer than the `decompose_max` parameter. If `live` node attributes
        were provided, then the `live` attribute for child-nodes will be set to `True` if either or both parent nodes
        were `live`. Otherwise, all nodes wil be set to `live=True`. The `length` and `imp_factor` edge attributes will
        be set to match the lengths of the new edges.

    Examples
    --------
    ```python
    from cityseer.tools import mock, graphs, plot

    G = mock.mock_graph()
    G_simple = graphs.nx_simple_geoms(G)
    G_decomposed = graphs.nx_decompose(G_simple, 100)
    plot.plot_nx(G_decomposed)
    ```

    ![Example graph](/images/graph_simple.png)
    _Example graph prior to decomposition._

    ![Example decomposed graph](/images/graph_decomposed.png)
    _Example graph after decomposition._

    """
    if not isinstance(nx_multigraph, nx.MultiGraph):
        raise TypeError("This method requires an undirected networkX MultiGraph.")
    logger.info(f"Decomposing graph to maximum edge lengths of {decompose_max}.")
    g_multi_copy: MultiGraph = nx_multigraph.copy()
    # note -> write to a duplicated graph to avoid in-place errors
    start_nd_key: NodeKey
    end_nd_key: NodeKey
    edge_data: EdgeData
    for start_nd_key, end_nd_key, edge_data in tqdm(
        nx_multigraph.edges(data=True), disable=config.QUIET_MODE
    ):  # pylint: disable=line-too-long
        # test for x, y in start coordinates
        if "x" not in nx_multigraph.nodes[start_nd_key] or "y" not in nx_multigraph.nodes[start_nd_key]:
            raise KeyError(f'Encountered node missing "x" or "y" coordinate attributes at node {start_nd_key}.')
        # test for x, y in end coordinates
        if "x" not in nx_multigraph.nodes[end_nd_key] or "y" not in nx_multigraph.nodes[end_nd_key]:
            raise KeyError(f'Encountered node missing "x" or "y" coordinate attributes at node {end_nd_key}.')
        s_x: float = nx_multigraph.nodes[start_nd_key]["x"]
        s_y: float = nx_multigraph.nodes[start_nd_key]["y"]
        e_x: float = nx_multigraph.nodes[end_nd_key]["x"]
        e_y: float = nx_multigraph.nodes[end_nd_key]["y"]
        # test for geom
        if "geom" not in edge_data:
            raise KeyError(
                f"No edge geom found for edge {start_nd_key}-{end_nd_key}: "
                f'Please add an edge "geom" attribute consisting of a shapely LineString.'
            )
        # get edge geometry
        line_geom: geometry.LineString = edge_data["geom"]
        if line_geom.type != "LineString":
            raise TypeError(
                f"Expecting LineString geometry but found {line_geom.type} geometry for "
                f"edge {start_nd_key}-{end_nd_key}."
            )
        # check geom coordinates directionality - flip if facing backwards direction
        line_geom_coords = _align_linestring_coords(line_geom.coords, (s_x, s_y))
        # double check that coordinates now face the forwards direction
        if not np.allclose((s_x, s_y), line_geom_coords[0][:2], atol=config.ATOL, rtol=config.RTOL) or not np.allclose(
            (e_x, e_y), line_geom_coords[-1][:2], atol=config.ATOL, rtol=config.RTOL
        ):
            raise ValueError(f"Edge geometry endpoint coordinate mismatch for edge {start_nd_key}-{end_nd_key}")
        line_geom: geometry.LineString = geometry.LineString(line_geom_coords)
        # see how many segments are necessary so as not to exceed decomposition max distance
        # note that a length less than the decompose threshold will result in a single 'sub'-string
        cuts: int = int(np.ceil(line_geom.length / decompose_max))  # type: ignore
        step_size: float = line_geom.length / cuts
        # since decomposing, remove the prior edge... but only after properties have been read
        g_multi_copy.remove_edge(start_nd_key, end_nd_key)
        # then add the new sub-edge/s
        step = 0
        prior_node_id = start_nd_key
        sub_node_counter = 0
        # everything inside this loop is a new node - i.e. this loop is effectively skipped if cuts = 1
        for _ in range(cuts - 1):
            # create the split LineString geom for measuring the new length
            line_segment: geometry.LineString = ops.substring(line_geom, step, step + step_size)
            # get the x, y of the new end node
            x, y = line_segment.coords[-1]
            # add the new node and edge
            new_nd_name = _add_node(g_multi_copy, [start_nd_key, sub_node_counter, end_nd_key], x=x, y=y)
            if new_nd_name is None:
                raise ValueError(
                    f"Attempted to add a duplicate node. "
                    f"Check for existence of duplicate edges in the vicinity of {start_nd_key}-{end_nd_key}."
                )
            sub_node_counter += 1
            # add and set live property if present in parent graph
            if "live" in nx_multigraph.nodes[start_nd_key] and "live" in nx_multigraph.nodes[end_nd_key]:
                live = True
                # if BOTH parents are not live, then set child to not live
                if not nx_multigraph.nodes[start_nd_key]["live"] and not nx_multigraph.nodes[end_nd_key]["live"]:
                    live = False
                g_multi_copy.nodes[new_nd_name]["live"] = live
            # add the edge
            g_multi_copy.add_edge(prior_node_id, new_nd_name, geom=line_segment)
            # increment the step and node id
            prior_node_id = new_nd_name
            step += step_size
        # set the last edge manually to avoid rounding errors at end of LineString
        # the nodes already exist, so just add edge
        line_segment = ops.substring(line_geom, step, line_geom.length)  # type: ignore
        g_multi_copy.add_edge(prior_node_id, end_nd_key, geom=line_segment)

    return g_multi_copy


def nx_to_dual(nx_multigraph: MultiGraph) -> MultiGraph:
    """
    Convert a primal graph representation to the dual representation.

    Primal graphs represent intersections as nodes and streets as edges. This method will invert this representation
    so that edges are converted to nodes and intersections become edges. Primal edge `geom` attributes will be welded to
    adjacent edges and split into the new dual edge `geom` attributes.

    :::note
    Note that a `MultiGraph` is useful for primal but not for dual, so the output `MultiGraph` will have single edges.
    e.g. a crescent street that spans the same intersections as parallel straight street requires multiple edges in
    primal. The same type of situation does not arise in the dual because the nodes map to distinct edges regardless.
    :::

    Parameters
    ----------
    nx_multigraph: MultiGraph
        A `networkX` `MultiGraph` in a projected coordinate system, containing `x` and `y` node attributes, and `geom`
        edge attributes containing `LineString` geoms.

    Returns
    -------
    MultiGraph
        A dual representation `networkX` graph. The new dual nodes will have `x` and `y` node attributes corresponding
        to the mid-points of the original primal edges. If `live` node attributes were provided, then the `live`
        attribute for the new dual nodes will be set to `True` if either or both of the adjacent primal nodes were set
        to `live=True`. Otherwise, all dual nodes wil be set to `live=True`. The primal `geom` edge attributes will be
        split and welded to form the new dual `geom` edge attributes. A `parent_primal_node` edge attribute will be
        added, corresponding to the node identifier of the primal graph.

    Examples
    --------
    ```python
    from cityseer.tools import graphs, mock, plot

    G = mock.mock_graph()
    G_simple = graphs.nx_simple_geoms(G)
    G_dual = graphs.nx_to_dual(G_simple)
    plot.plot_nx_primal_or_dual(G_simple,
                                G_dual,
                                plot_geoms=False)
    ```

    ![Example dual graph](/images/graph_dual.png)
    _Dual graph (blue) overlaid on the source primal graph (red)._

    """
    if not isinstance(nx_multigraph, nx.MultiGraph):
        raise TypeError("This method requires an undirected networkX MultiGraph.")
    logger.info("Converting graph to dual.")
    g_dual = nx.MultiGraph()

    def get_half_geoms(nx_multigraph_ref: MultiGraph, a_node: NodeKey, b_node: NodeKey, edge_idx: int):
        """
        Split geom and orient half-geoms.
        """
        # get edge data
        edge_data: EdgeData = nx_multigraph_ref[a_node][b_node][edge_idx]
        # test for x coordinates
        if "x" not in nx_multigraph_ref.nodes[a_node] or "y" not in nx_multigraph_ref.nodes[a_node]:
            raise KeyError(f'Encountered node missing "x" or "y" coordinate attributes at node {a_node}.')
        # test for y coordinates
        if "x" not in nx_multigraph_ref.nodes[b_node] or "y" not in nx_multigraph_ref.nodes[b_node]:
            raise KeyError(f'Encountered node missing "x" or "y" coordinate attributes at node {b_node}.')
        a_xy: CoordsType = cast(
            CoordsType, (nx_multigraph_ref.nodes[a_node]["x"], nx_multigraph_ref.nodes[a_node]["y"])
        )
        b_xy: CoordsType = cast(
            CoordsType, (nx_multigraph_ref.nodes[b_node]["x"], nx_multigraph_ref.nodes[b_node]["y"])
        )
        # test for geom
        if "geom" not in edge_data:
            raise KeyError(
                f"No edge geom found for edge {a_node}-{b_node}: "
                f'Please add an edge "geom" attribute consisting of a shapely LineString.'
            )
        # get edge geometry
        line_geom = edge_data["geom"]
        if line_geom.type != "LineString":
            raise TypeError(
                f"Expecting LineString geometry but found {line_geom.type} geometry for edge {a_node}-{b_node}."
            )
        # align geom coordinates to start from A side
        line_geom_coords = _align_linestring_coords(line_geom.coords, a_xy)
        line_geom = geometry.LineString(line_geom_coords)
        # generate the two half geoms
        a_half_geom: geometry.LineString = ops.substring(line_geom, 0, line_geom.length / 2)  # type: ignore
        b_half_geom: geometry.LineString = ops.substring(
            line_geom, line_geom.length / 2, line_geom.length  # type: ignore
        )  # pylint: disable=line-too-long
        # check that nothing odd happened with new midpoint
        if not np.allclose(
            a_half_geom.coords[-1][:2],
            b_half_geom.coords[0][:2],
            atol=config.ATOL,
            rtol=0,
        ):
            raise ValueError("Nodes of half geoms don't match")
        # snap to prevent creeping tolerance issues
        # A side geom starts at node A and ends at new midpoint
        a_half_geom_coords = _snap_linestring_startpoint(a_half_geom.coords, a_xy)
        # snap new midpoint to geom A's endpoint (i.e. no need to snap endpoint of geom A)
        mid_xy = a_half_geom_coords[-1][:2]
        # B side geom starts at mid and ends at B node
        b_half_geom_coords = _snap_linestring_startpoint(b_half_geom.coords, mid_xy)
        b_half_geom_coords = _snap_linestring_endpoint(b_half_geom_coords, b_xy)
        # double check coords
        if (
            a_half_geom_coords[0][:2] != a_xy
            or a_half_geom_coords[-1][:2] != mid_xy
            or b_half_geom_coords[0][:2] != mid_xy
            or b_half_geom_coords[-1][:2] != b_xy
        ):
            raise ValueError("Nodes of half geoms don't match")

        return geometry.LineString(a_half_geom_coords), geometry.LineString(b_half_geom_coords)

    def set_live(start_nd_key: NodeKey, end_nd_key: NodeKey, dual_node_key: NodeKey):
        # add and set live property to dual node if present in adjacent primal graph nodes
        if "live" in nx_multigraph.nodes[start_nd_key] and "live" in nx_multigraph.nodes[end_nd_key]:
            live = True
            # if BOTH parents are not live, then set child to not live
            if not nx_multigraph.nodes[start_nd_key]["live"] and not nx_multigraph.nodes[end_nd_key]["live"]:
                live = False
            g_dual.nodes[dual_node_key]["live"] = live

    # iterate the primal graph's edges
    start_nd_key: NodeKey
    end_nd_key: NodeKey
    edge_idx: int
    for start_nd_key, end_nd_key, edge_idx in tqdm(
        nx_multigraph.edges(data=False, keys=True), disable=config.QUIET_MODE
    ):
        # get the first and second half geoms
        s_half_geom, e_half_geom = get_half_geoms(nx_multigraph, start_nd_key, end_nd_key, edge_idx)
        # create a new dual node corresponding to the current primal edge
        # nodes are added manually to retain link to origin node names and to check for duplicates
        s_e = sorted([str(start_nd_key), str(end_nd_key)])
        hub_node_dual = f"{s_e[0]}_{s_e[1]}"
        # the node may already have been added from a neighbouring node that has already been processed
        if hub_node_dual not in g_dual:
            x, y = s_half_geom.coords[-1][:2]
            g_dual.add_node(hub_node_dual, x=x, y=y)
            # add and set live property if present in parent graph
            set_live(start_nd_key, end_nd_key, hub_node_dual)
        # process either side
        for n_side, half_geom in zip([start_nd_key, end_nd_key], [s_half_geom, e_half_geom]):
            # add the spoke edges on the dual
            nb_nd_key: NodeKey
            for nb_nd_key in nx.neighbors(nx_multigraph, n_side):
                # don't follow neighbour back to current edge combo
                if nb_nd_key in [start_nd_key, end_nd_key]:
                    continue
                # add the neighbouring primal edge as dual node
                s_nb = sorted([str(n_side), str(nb_nd_key)])
                spoke_node_dual = f"{s_nb[0]}_{s_nb[1]}"
                # skip if the edge has already been processed from another direction
                if g_dual.has_edge(hub_node_dual, spoke_node_dual):
                    continue
                # get the near and far half geoms
                spoke_half_geom, _discard_geom = get_half_geoms(
                    nx_multigraph, n_side, nb_nd_key, edge_idx
                )  # pylint: disable=line-too-long
                # nodes will be added if not already present (i.e. from first direction processed)
                if spoke_node_dual not in g_dual:
                    x, y = spoke_half_geom.coords[-1][:2]
                    g_dual.add_node(spoke_node_dual, x=x, y=y)
                    # add and set live property if present in parent graph
                    set_live(start_nd_key, end_nd_key, spoke_node_dual)
                # weld the lines
                merged_line: geometry.LineString = ops.linemerge([half_geom, spoke_half_geom])
                if merged_line.type != "LineString":
                    raise TypeError(
                        f'Found {merged_line.type} geometry instead of "LineString" for new geom {merged_line.wkt}. '
                        f"Check that the LineStrings for {start_nd_key}-{end_nd_key} and {n_side}-{nb_nd_key} touch."
                    )
                # add the dual edge
                g_dual.add_edge(
                    hub_node_dual,
                    spoke_node_dual,
                    parent_primal_node=n_side,
                    geom=merged_line,
                )

    return g_dual


def network_structure_from_nx(
    nx_multigraph: MultiGraph,
    crs: str,
) -> tuple[gpd.GeoDataFrame, structures.NetworkStructure]:
    """
    Transpose a `networkX` `MultiGraph` into a `GeoDataFrame` and `NetworkStructure` for use by `cityseer`.

    Calculates length and angle attributes, as well as in and out bearings, and stores this information in the returned
    data maps.

    Parameters
    ----------
    nx_multigraph: MultiGraph
        A `networkX` `MultiGraph` in a projected coordinate system, containing `x` and `y` node attributes, and `geom`
        edge attributes containing `LineString` geoms.
    crs: str
        CRS for initialising the returned structures. This is used for initialising the GeoPandas
        [`GeoDataFrame`](https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoDataFrame.html#geopandas-geodataframe).  # pylint: disable=line-too-long

    Returns
    -------
    nodes_gdf: GeoDataFrame
        A `GeoDataFrame` with `live` and `geometry` attributes. The original `networkX` graph's node keys will be used
        for the `GeoDataFrame` index.
    network_structure: structures.NetworkStructure
        A [`structures.NetworkStructure`](/structures/#networkstructure) instance.

    """
    if not isinstance(nx_multigraph, nx.MultiGraph):
        raise TypeError("This method requires an undirected networkX MultiGraph.")
    logger.info("Preparing node and edge arrays from networkX graph.")
    g_multi_copy: MultiGraph = nx_multigraph.copy()
    # accumulate degrees
    total_out_degrees = 0
    nd_key: NodeKey
    for nd_key in tqdm(g_multi_copy.nodes(), disable=config.QUIET_MODE):
        # writing node identifier to 'labels' in case conversion to integers method interferes with order
        g_multi_copy.nodes[nd_key]["label"] = nd_key
        nb_nd_key: NodeKey
        for nb_nd_key in nx.neighbors(g_multi_copy, nd_key):
            total_out_degrees += g_multi_copy.number_of_edges(nd_key, nb_nd_key)
    # convert the nodes to sequential - this permits implicit indices with benefits to speed and structure
    g_multi_copy = nx.convert_node_labels_to_integers(g_multi_copy, 0)
    # prepare the network structure
    node_keys: list[NodeKey] = []
    nodes_n: int = g_multi_copy.number_of_nodes()
    edges_n: int = total_out_degrees
    network_structure: structures.NetworkStructure = structures.NetworkStructure(nodes_n, edges_n)
    # generate the network information
    # NOTE: node keys have been converted to int - so use int directly for jitclass
    start_node_key: int
    node_data: NodeData
    for start_node_key, node_data in tqdm(g_multi_copy.nodes(data=True), disable=config.QUIET_MODE):
        # don't cast label to string otherwise correspondence between original and round-trip graph indices is lost
        node_keys.append(node_data["label"])
        if "x" not in node_data:
            raise KeyError(f'Encountered node missing "x" coordinate attribute at node {start_node_key}.')
        node_x: float = node_data["x"]
        if "y" not in node_data:
            raise KeyError(f'Encountered node missing "y" coordinate attribute at node {start_node_key}.')
        node_y: float = node_data["y"]
        is_live: bool = True
        if "live" in node_data:
            is_live = bool(node_data["live"])
        # set node
        network_structure.set_node(start_node_key, node_x, node_y, is_live)
        # build edges
        end_node_key: int
        for end_node_key in g_multi_copy.neighbors(start_node_key):
            # add the new edge index to the node's out edges
            _nx_edge_idx: int
            nx_edge_data: EdgeData
            for _nx_edge_idx, nx_edge_data in g_multi_copy[start_node_key][end_node_key].items():
                if not "geom" in nx_edge_data:
                    raise KeyError(
                        f"No edge geom found for edge {start_node_key}-{end_node_key}: Please add an edge 'geom' "
                        "attribute consisting of a shapely LineString. Simple (straight) geometries can be inferred "
                        "automatically through the nx_simple_geoms() method."
                    )
                line_geom = nx_edge_data["geom"]
                if line_geom.type != "LineString":
                    raise TypeError(
                        f"Expecting LineString geometry but found {line_geom.type} geom for edge "
                        f"{start_node_key}-{end_node_key}."
                    )
                # cannot have zero or negative length - division by zero
                line_len = line_geom.length
                if not np.isfinite(line_len) or line_len <= 0:
                    raise ValueError(
                        f"Length {line_len} for edge {start_node_key}-{end_node_key} must be finite and positive."
                    )
                # check geom coordinates directionality (for bearings at index 5 / 6)
                # flip if facing backwards direction
                line_geom_coords = _align_linestring_coords(line_geom.coords, (node_x, node_y))
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
                        f"Angle sum {angle_sum} for edge {start_node_key}-{end_node_key} must be finite and positive."
                    )
                # if imp_factor is set explicitly, then use
                # fallback imp_factor of 1
                imp_factor: float = 1
                if "imp_factor" in nx_edge_data:
                    # cannot have imp_factor less than zero (but == 0 is OK)
                    imp_factor = nx_edge_data["imp_factor"]
                    if not (np.isfinite(imp_factor) or np.isinf(imp_factor)) or imp_factor < 0:
                        raise ValueError(
                            f"Impedance factor: {imp_factor} for edge {start_node_key}-{end_node_key} must be finite "
                            " and positive or positive infinity."
                        )
                # in bearing
                x_1: float = line_geom_coords[0][0]
                y_1: float = line_geom_coords[0][1]
                x_2: float = line_geom_coords[1][0]
                y_2: float = line_geom_coords[1][1]
                in_bearing: float = np.rad2deg(np.arctan2(y_2 - y_1, x_2 - x_1))
                # out bearing
                x_1: float = line_geom_coords[-2][0]
                y_1: float = line_geom_coords[-2][1]
                x_2: float = line_geom_coords[-1][0]
                y_2: float = line_geom_coords[-1][1]
                out_bearing: float = np.rad2deg(np.arctan2(y_2 - y_1, x_2 - x_1))
                network_structure.set_edge(
                    start_node_key, end_node_key, line_len, angle_sum, imp_factor, in_bearing, out_bearing
                )
    # create geopandas for node keys and data state
    data = {
        "node_key": node_keys,
        "live": network_structure.nodes.live,
        "geometry": gpd.points_from_xy(network_structure.nodes.xs, network_structure.nodes.ys),
    }
    nodes_gdf = gpd.GeoDataFrame(data, crs=crs)
    nodes_gdf = nodes_gdf.set_index("node_key")
    nodes_gdf = cast(gpd.GeoDataFrame, nodes_gdf)

    return nodes_gdf, network_structure


def nx_from_network_structure(
    nodes_gdf: gpd.GeoDataFrame,
    network_structure: structures.NetworkStructure,
    nx_multigraph: Optional[MultiGraph] = None,
) -> MultiGraph:
    """
    Write `cityseer` data graph maps back to a `networkX` `MultiGraph`.

    This method will write back to an existing `MultiGraph` if an existing graph is provided as an argument to the
    `nx_multigraph` parameter.

    Parameters
    ----------
    nodes_gdf: GeoDataFrame
        A `GeoDataFrame` with `live` and Point `geometry` attributes. The index will be used for the returned `networkX`
        graph's node keys.
    network_structure: structures.NetworkStructure
        A [`structures.NetworkStructure`](/structures/#networkstructure) instance corresponding to the `nodes_gdf`
        parameter.
    nx_multigraph: MultiGraph
        An optional `networkX` graph to use as a backbone for unpacking the data. The number of nodes and edges should
        correspond to the `cityseer` data maps and the node identifiers should correspond to the `node_keys`. If not
        provided, then a new `networkX` graph will be returned. This function is intended to be used for situations
        where `cityseer` data is being transposed back to a source `networkX` graph. Defaults to None.

    Returns
    -------
    nx_multigraph: MultiGraph
        A `networkX` graph. If a backbone graph was provided, a copy of the same graph will be returned. If no graph was
        provided, then a new graph will be generated. `x`, `y`, `live` node attributes will be copied from `nodes_gdf`
        to the graph nodes. `length`, `angle_sum`, `imp_factor`, `in_bearing`, and `out_bearing` attributes will be
        copied from the `network_structure` to the graph edges. `cc_metric` columns will be copied from the `nodes_gdf`
        `GeoDataFrame` to the corresponding nodes in the returned `MultiGraph`.

    """
    logger.info("Populating node and edge map data to a networkX graph.")
    if nx_multigraph is not None:
        logger.info("Reusing existing graph as backbone.")
        if nx_multigraph.number_of_nodes() != network_structure.nodes.count:
            raise ValueError("The number of nodes in the graph does not match the number of nodes in the node map.")
        g_multi_copy: MultiGraph = nx_multigraph.copy()
        for nd_key in nodes_gdf.index:  # type: ignore
            if nd_key not in g_multi_copy:
                raise KeyError(
                    f"Node key {nd_key} not found in graph. If passing an existing nx graph as backbone "
                    "then the keys must match those supplied with the node and edge maps."
                )
    else:
        logger.info("No existing graph found, creating new nx multigraph.")
        g_multi_copy = nx.MultiGraph()
        g_multi_copy.add_nodes_from(nodes_gdf.index.values.tolist())
    # after above so that errors caught first
    network_structure.validate()
    logger.info("Unpacking node data.")
    for nd_key, nd_data in tqdm(nodes_gdf.iterrows(), disable=config.QUIET_MODE):  # type: ignore
        g_multi_copy.nodes[nd_key]["x"] = nd_data.geometry.x
        g_multi_copy.nodes[nd_key]["y"] = nd_data.geometry.y
        g_multi_copy.nodes[nd_key]["live"] = nd_data.live
    logger.info("Unpacking edge data.")
    for edge_idx in tqdm(range(network_structure.edges.count), disable=config.QUIET_MODE):  # type: ignore
        start_nd_idx: int = network_structure.edges.start[edge_idx]  # type: ignore
        end_nd_idx: int = network_structure.edges.end[edge_idx]  # type: ignore
        length: float = network_structure.edges.length[edge_idx]  # type: ignore
        angle_sum: float = network_structure.edges.angle_sum[edge_idx]  # type: ignore
        imp_factor: float = network_structure.edges.imp_factor[edge_idx]  # type: ignore
        # find corresponding node keys
        start_nd_key: NodeKey = nodes_gdf.index[start_nd_idx]
        end_nd_key: NodeKey = nodes_gdf.index[end_nd_idx]
        # note that the original geom is lost with round trip unless retained in a supplied backbone graph.
        # the edge map is directional, so each original edge will be processed twice, once from either direction.
        # edges are only added if A) not using a backbone graph and B) the edge hasn't already been added
        if nx_multigraph is None:
            # if the edge doesn't already exist, then simply add
            if not g_multi_copy.has_edge(start_nd_key, end_nd_key):
                add_edge = True
            # else, only add if not matching an already added edge
            # i.e. don't add the same edge when processed from opposite direction
            else:
                add_edge = True  # tentatively set to True
                # iter the edges
                edge_item_idx: int
                edge_item_data: EdgeData
                for edge_item_idx, edge_item_data in g_multi_copy[start_nd_key][end_nd_key].items():
                    # set add_edge to false if a matching edge length is found
                    if edge_item_data["length"] == length:
                        add_edge = False
            # add the edge if not existent
            if add_edge:
                g_multi_copy.add_edge(
                    start_nd_key,
                    end_nd_key,
                    length=length,
                    angle_sum=angle_sum,
                    imp_factor=imp_factor,
                )
        # if a new edge is not being added then add the attributes to the appropriate backbone edge if not already done
        # this is only relevant if processing a backbone graph
        else:
            # raise if the edge doesn't exist
            if not g_multi_copy.has_edge(start_nd_key, end_nd_key):
                raise KeyError(
                    f"The backbone graph is missing an edge spanning from {start_nd_key} to {end_nd_key}"
                    f"The original graph (with all original edges) has to be reused."
                )
            # due working with a MultiGraph it is necessary to check that the correct edge index is matched
            for edge_item_idx, edge_item_data in g_multi_copy[start_nd_key][end_nd_key].items():
                if np.isclose(edge_item_data["geom"].length, length, atol=config.ATOL, rtol=config.RTOL):
                    # check whether the attributes have already been added from the other direction?
                    if "length" in edge_item_data and edge_item_data["length"] == length:
                        continue
                    # otherwise add the edge attributes and move on
                    exist_edge_data: EdgeData
                    exist_edge_data = g_multi_copy[start_nd_key][end_nd_key][edge_item_idx]
                    exist_edge_data["length"] = length
                    exist_edge_data["angle_sum"] = angle_sum
                    exist_edge_data["imp_factor"] = imp_factor
    # unpack any metrics written to the nodes
    metrics_column_labels: list[str] = [c for c in nodes_gdf.columns if c.startswith("cc_metric")]  # type: ignore
    if metrics_column_labels is not None:
        logger.info("Unpacking metrics to nodes.")
        for metrics_column_label in metrics_column_labels:
            for nd_key, node_row in tqdm(nodes_gdf.iterrows(), disable=config.QUIET_MODE):  # type: ignore
                g_multi_copy.nodes[nd_key][metrics_column_label] = node_row[metrics_column_label]

    return g_multi_copy


def nx_from_osm_nx(
    nx_multidigraph: MultiDiGraph,
    node_attributes: Optional[Union[list[str], tuple[str]]] = None,
    edge_attributes: Optional[Union[list[str], tuple[str]]] = None,
    tolerance: float = config.ATOL,
) -> MultiGraph:
    """
    Copy an [`OSMnx`](https://osmnx.readthedocs.io/) directed `MultiDiGraph` to an undirected `cityseer` `MultiGraph`.

    See the [`OSMnx`](/guide/#osm-and-networkx) section of the guide for a more general discussion (and example) on
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
    nx_multidigraph: MultiDiGraph
        A `OSMnx` derived `networkX` `MultiDiGraph` containing `x` and `y` node attributes, with optional `geometry`
        edge attributes containing `LineString` geoms (for simplified edges).
    node_attributes: tuple[str]
        Optional node attributes to copy to the new MultiGraph. (In addition to the default `x` and `y` attributes.)
    edge_attributes: tuple[str]
        Optional edge attributes to copy to the new MultiGraph. (In addition to the optional `geometry` attribute.)
    tolerance: float
        Tolerance at which to raise errors for mismatched geometry end-points vis-a-vis corresponding node coordinates.
        Prior to conversion, this method will check edge geometry end-points for alignment with the corresponding
        end-point nodes. Where these don't align within the given tolerance an exception will be raised. Otherwise, if
        within the tolerance, the conversion function will snap the geometry end-points to the corresponding node
        coordinates so that downstream exceptions are not subsequently raised. It is preferable to minimise graph
        manipulation prior to conversion to a `cityseer` compatible `MultiGraph` otherwise particularly large tolerances
        may be required, and this may lead to some unexpected or undesirable effects due to aggressive snapping.

    Returns
    -------
    MultiGraph
        A `cityseer` compatible `networkX` graph with `x` and `y` node attributes and `geom` edge attribute.

    """
    if not isinstance(nx_multidigraph, nx.MultiDiGraph):
        raise TypeError("This method requires a directed networkX MultiDiGraph as derived from `OSMnx`.")
    if node_attributes is not None and not isinstance(node_attributes, (list, tuple)):
        raise TypeError("Node attributes to be copied should be provided as either a list or tuple of attribute keys.")
    if edge_attributes is not None and not isinstance(edge_attributes, (list, tuple)):
        raise TypeError("Edge attributes to be copied should be provided as either a list or tuple of attribute keys.")
    logger.info("Converting OSMnx MultiDiGraph to cityseer MultiGraph.")
    # target MultiGraph
    g_multi = nx.MultiGraph()

    def _process_node(nd_key: NodeKey) -> tuple[float, float]:
        # x
        if "x" not in nx_multidigraph.nodes[nd_key]:
            raise KeyError(f'Encountered node missing "x" coordinate attribute for node {nd_key}.')
        x: float = nx_multidigraph.nodes[nd_key]["x"]
        # y
        if "y" not in nx_multidigraph.nodes[nd_key]:
            raise KeyError(f'Encountered node missing "y" coordinate attribute for node {nd_key}.')
        y: float = nx_multidigraph.nodes[nd_key]["y"]
        # add attributes if necessary
        if nd_key not in g_multi:
            g_multi.add_node(nd_key, x=x, y=y)
            if node_attributes is not None:
                for node_att in node_attributes:
                    if node_att not in nx_multidigraph.nodes[nd_key]:
                        raise ValueError(f"Specified attribute {node_att} is not available for node {nd_key}.")
                    g_multi.nodes[nd_key][node_att] = nx_multidigraph.nodes[nd_key][node_att]

        return x, y

    # copy nodes and edges
    start_nd_key: NodeKey
    end_nd_key: NodeKey
    edge_idx: int
    edge_data: EdgeData
    for start_nd_key, end_nd_key, edge_idx, edge_data in tqdm(
        nx_multidigraph.edges(data=True, keys=True), disable=config.QUIET_MODE
    ):
        edge_data = cast(EdgeData, edge_data)  # type: ignore
        s_x, s_y = _process_node(start_nd_key)
        e_x, e_y = _process_node(end_nd_key)
        # copy edge if present
        if "geometry" in edge_data:
            line_geom: geometry.LineString = edge_data["geometry"]
        # otherwise create
        else:
            line_geom = geometry.LineString([[s_x, s_y], [e_x, e_y]])
        # check for LineString validity
        if line_geom.type != "LineString":
            raise TypeError(
                f"Expecting LineString geometry but found {line_geom.type} geometry for "
                f"edge {start_nd_key}-{end_nd_key}."
            )
        # orient LineString
        geom_coords = line_geom.coords
        if not np.allclose((s_x, s_y), geom_coords[0][:2], atol=tolerance, rtol=0):
            geom_coords = _align_linestring_coords(geom_coords, (s_x, s_y))
        # check starting and ending tolerances
        if not np.allclose((s_x, s_y), geom_coords[0][:2], atol=tolerance, rtol=0):
            raise ValueError("Starting node coordinates don't match LineString geometry starting coordinates.")
        if not np.allclose((e_x, e_y), geom_coords[-1][:2], atol=tolerance, rtol=0):
            raise ValueError("Ending node coordinates don't match LineString geometry ending coordinates.")
        # snap starting and ending coords to avoid rounding error issues
        geom_coords = _snap_linestring_startpoint(geom_coords, (s_x, s_y))
        geom_coords = _snap_linestring_endpoint(geom_coords, (e_x, e_y))
        g_multi.add_edge(start_nd_key, end_nd_key, key=edge_idx, geom=geometry.LineString(geom_coords))
        if edge_attributes is not None:
            for edge_att in edge_attributes:
                if edge_att not in edge_data:
                    raise ValueError(f"Attribute {edge_att} is not available for edge {start_nd_key}-{end_nd_key}.")
                g_multi[start_nd_key][end_nd_key][edge_idx][edge_att] = edge_data[edge_att]

    return g_multi
