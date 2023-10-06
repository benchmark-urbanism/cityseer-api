"""
Convenience functions for the preparation and conversion of `networkX` graphs to and from `cityseer` data structures.

Note that the `cityseer` network data structures can be created and manipulated directly, if so desired.

"""
# workaround until networkx adopts types
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportGeneralTypeIssues=false

from __future__ import annotations

import json
import logging
from typing import Any, Optional, Union

import geopandas as gpd
import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd
import utm
from pyproj import CRS, Transformer
from shapely import coords, geometry, ops, strtree
from shapely.errors import GeometryTypeError
from shapely.geometry import LineString
from tqdm import tqdm

from cityseer import config, rustalgos

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# define types
MultiGraph = Any
MultiDiGraph = Any
# coords can be 2d or 3d
NodeKey = str
NodeData = dict[str, Any]
EdgeType = Union[tuple[NodeKey, NodeKey], tuple[NodeKey, NodeKey, int]]
EdgeData = dict[str, Any]
EdgeMapping = tuple[NodeKey, NodeKey, int, geometry.LineString]
CoordsType = Union[tuple[float, float], tuple[float, float, float]]
AnyCoordsType = Union[list[CoordsType], npt.NDArray[np.float_], coords.CoordinateSequence]
ListCoordsType = list[CoordsType]


def _substring(  # type: ignore # pylint: disable=too-many-return-statements
    geom, start_dist, end_dist, normalized=False  # type: ignore
):
    """Temporary copy of shapely substring method until issue #1699 is fixed (re: z coords)."""
    if not isinstance(geom, LineString):
        raise GeometryTypeError(
            "Can only calculate a substring of LineString geometries. " f"A {geom.geom_type} was provided."
        )

    # Filter out cases in which to return a point
    if start_dist == end_dist:
        return geom.interpolate(start_dist, normalized)  # type: ignore
    if not normalized and start_dist >= geom.length and end_dist >= geom.length:
        return geom.interpolate(geom.length, normalized)  # type: ignore
    if not normalized and -start_dist >= geom.length and -end_dist >= geom.length:
        return geom.interpolate(0, normalized)  # type: ignore
    if normalized and start_dist >= 1 and end_dist >= 1:
        return geom.interpolate(1, normalized)  # type: ignore
    if normalized and -start_dist >= 1 and -end_dist >= 1:
        return geom.interpolate(0, normalized)  # type: ignore

    if normalized:
        start_dist *= geom.length  # type: ignore
        end_dist *= geom.length  # type: ignore

    # Filter out cases where distances meet at a middle point from opposite ends.
    if start_dist < 0 < end_dist and abs(start_dist) + end_dist == geom.length:  # type: ignore
        return geom.interpolate(end_dist)  # type: ignore
    if end_dist < 0 < start_dist and abs(end_dist) + start_dist == geom.length:  # type: ignore
        return geom.interpolate(start_dist)  # type: ignore

    start_point = geom.interpolate(start_dist)  # type: ignore
    end_point = geom.interpolate(end_dist)  # type: ignore

    if start_dist < 0:
        start_dist = geom.length + start_dist  # Values may still be negative,# type: ignore
    if end_dist < 0:  # but only in the out-of-range
        end_dist = geom.length + end_dist  # sense, not the wrap-around sense.# type: ignore

    reverse = start_dist > end_dist  # type: ignore
    if reverse:
        start_dist, end_dist = end_dist, start_dist  # type: ignore

    if start_dist < 0:  # pylint: disable=consider-using-max-builtin
        start_dist = 0  # to avoid duplicating the first vertex

    if reverse:
        vertex_list = [tuple(*end_point.coords)]  # type: ignore
    else:
        vertex_list = [tuple(*start_point.coords)]  # type: ignore

    _coords = list(geom.coords)  # type: ignore
    current_distance = 0
    for p1, p2 in zip(_coords, _coords[1:]):  # pylint: disable=invalid-name # type: ignore
        if start_dist < current_distance < end_dist:
            vertex_list.append(p1)
        elif current_distance >= end_dist:
            break

        current_distance += ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5  # type: ignore

    if reverse:
        vertex_list.append(tuple(*start_point.coords))  # type: ignore
        # reverse direction result
        vertex_list = reversed(vertex_list)  # type: ignore
    else:
        vertex_list.append(tuple(*end_point.coords))  # type: ignore

    return LineString(vertex_list)


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


def snap_linestring_startpoint(
    linestring_coords: AnyCoordsType,
    x_y: CoordsType,
) -> ListCoordsType:
    """
    Snaps a LineString's start-point coordinate to a specified x_y coordinate.

    Parameters
    ----------
    linestring_coords: tuple | list | np.ndarray
        A list, tuple, or numpy array of x, y coordinate tuples.
    x_y: tuple[float, float]
        A tuple of floats representing the target x, y coordinates against which to align the linestring start point.

    Returns
    -------
    linestring_coords
        A list of linestring coords aligned to the specified starting point.

    """
    return _snap_linestring_idx(linestring_coords, 0, x_y)


def snap_linestring_endpoint(
    linestring_coords: AnyCoordsType,
    x_y: CoordsType,
) -> ListCoordsType:
    """
    Snaps a LineString's end-point coordinate to a specified x_y coordinate.

    Parameters
    ----------
    linestring_coords: tuple | list | np.ndarray
        A list, tuple, or numpy array of x, y coordinate tuples.
    x_y: tuple[float, float]
        A tuple of floats representing the target x, y coordinates against which to align the linestring end point.

    Returns
    -------
    linestring_coords
        A list of linestring coords aligned to the specified ending point.

    """
    return _snap_linestring_idx(linestring_coords, -1, x_y)


def align_linestring_coords(
    linestring_coords: AnyCoordsType,
    x_y: CoordsType,
    reverse: bool = False,
    tolerance: float = 0.5,
) -> ListCoordsType:
    """
    Align a LineString's coordinate order to either start or end at a specified x_y coordinate within a given tolerance.

    Parameters
    ----------
    linestring_coords: tuple | list | np.ndarray
        A list, tuple, or numpy array of x, y coordinate tuples.
    x_y: tuple[float, float]
        A tuple of floats representing the target x, y coordinates against which to align the linestring coords.
    reverse: bool
        If reverse=False the coordinate order will be aligned to start from the given x_y coordinate. If reverse=True
        the coordinate order will be aligned to end at the given x_y coordinate.
    tolerance: float
        Distance tolerance in metres for matching the x_y coordinate to the linestring_coords.

    Returns
    -------
    linestring_coords
        A list of linestring coords aligned to the specified endpoint.

    """
    # check types
    if not isinstance(linestring_coords, (list, np.ndarray, coords.CoordinateSequence)):
        raise ValueError("Expecting a list, numpy array, or shapely LineString coordinate sequence.")
    linestring_coords = list(linestring_coords)
    a_dist = np.hypot(linestring_coords[0][0] - x_y[0], linestring_coords[0][1] - x_y[1])
    b_dist = np.hypot(linestring_coords[-1][0] - x_y[0], linestring_coords[-1][1] - x_y[1])
    # the target indices depend on whether reversed or not
    if not reverse:
        if a_dist > b_dist:
            linestring_coords = linestring_coords[::-1]
        tol_dist = np.hypot(linestring_coords[0][0] - x_y[0], linestring_coords[0][1] - x_y[1])
    else:
        if a_dist < b_dist:
            linestring_coords = linestring_coords[::-1]
        tol_dist = np.hypot(linestring_coords[-1][0] - x_y[0], linestring_coords[-1][1] - x_y[1])
    if tol_dist > tolerance:
        raise ValueError(f"Closest side of edge geom is {tol_dist} from node, exceeding tolerance of {tolerance}.")
    # otherwise no flipping is required and the coordinates can simply be returned
    return linestring_coords


def _snap_linestring_endpoints(
    nx_multigraph: nx.MultiGraph,
    start_nd_key: NodeKey,
    end_nd_key: NodeKey,
    linestring_coords: ListCoordsType,
    tolerance: float = 0.5,
) -> ListCoordsType:
    """
    Snaps edge geom coordinate sequence to the nodes on either side.

    Parameters
    ----------
    nx_multigraph: MultiGraph
        A `networkX` `MultiGraph` with `x` and `y` node attributes and edge `geom` attributes.
    start_nd_key: NodeKey
        A node key corresponding to the edge's start node.
    end_nd_key: NodeKey
        A node key corresponding to the edge's end node.
    linestring_coords: tuple | list | np.ndarray
        A list, tuple, or numpy array of x, y coordinate tuples.
    tolerance: float
        Distance tolerance in metres for matching the x_y coordinate to the linestring_coords.

    Returns
    -------
    linestring_coords
        A list of linestring coords aligned to the specified ending point.

    """
    # unpack node data
    start_nd_data: NodeData = nx_multigraph.nodes[start_nd_key]
    s_xy = (start_nd_data["x"], start_nd_data["y"])
    end_nd_data: NodeData = nx_multigraph.nodes[end_nd_key]
    e_xy = (end_nd_data["x"], end_nd_data["y"])
    # align and snap edge geom
    linestring_coords = align_linestring_coords(linestring_coords, s_xy, tolerance=tolerance)
    if not np.allclose(linestring_coords[0], s_xy, atol=tolerance, rtol=0):
        raise ValueError("Linestring geometry does not match starting node coordinates.")
    if not np.allclose(linestring_coords[-1], e_xy, atol=tolerance, rtol=0):
        raise ValueError("Linestring geometry does not match ending node coordinates.")
    linestring_coords = snap_linestring_startpoint(linestring_coords, s_xy)
    linestring_coords = snap_linestring_endpoint(linestring_coords, e_xy)
    return linestring_coords


def _weld_linestring_coords(
    linestring_coords_a: AnyCoordsType,
    linestring_coords_b: AnyCoordsType,
    force_xy: Optional[CoordsType] = None,
    tolerance: float = 0.01,
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
            coords_a = align_linestring_coords(linestring_coords_a, force_xy, reverse=True)
        else:
            coords_a = linestring_coords_a
        if not np.allclose(linestring_coords_b[0][:2], force_xy, atol=tolerance, rtol=0):
            coords_b = align_linestring_coords(linestring_coords_b, force_xy, reverse=False)
        else:
            coords_b = linestring_coords_b
    # case A: the linestring_b has to be flipped to start from x, y
    elif np.allclose(linestring_coords_a[-1][:2], linestring_coords_b[-1][:2], atol=tolerance, rtol=0):
        anchor_xy = linestring_coords_a[-1][:2]
        coords_a = linestring_coords_a
        coords_b = align_linestring_coords(linestring_coords_b, anchor_xy)
    # case B: linestring_a has to be flipped to end at x, y
    elif np.allclose(linestring_coords_a[0][:2], linestring_coords_b[0][:2], atol=tolerance, rtol=0):
        anchor_xy = linestring_coords_a[0][:2]
        coords_a = align_linestring_coords(linestring_coords_a, anchor_xy)
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


class _EdgeInfo:
    _names: list[str]
    _refs: list[str]
    _highways: list[str]

    @property
    def names(self):
        """Returns a set of street names."""
        return tuple(set(self._names))

    @property
    def routes(self):
        """Returns a set of routes - e.g. route numbers."""
        return tuple(set(self._refs))

    @property
    def highways(self):
        """Returns a set of highway types - e.g. footway."""
        return tuple(set(self._highways))

    def __init__(self):
        """Initialises a network information structure."""
        self._names = []
        self._refs = []
        self._highways = []

    def gather_edge_info(self, edge_data: dict[str, Any]):
        """Gather edge data from provided edge_data."""
        # agg names, routes, highway attributes if present
        if "names" in edge_data:
            self._names += edge_data["names"]
        if "routes" in edge_data:
            self._refs += edge_data["routes"]
        if "highways" in edge_data:
            self._highways += edge_data["highways"]

    def set_edge_info(
        self,
        nx_multigraph: nx.MultiGraph,
        start_node_key: NodeKey,
        end_node_key: NodeKey,
        edge_idx: int,
    ):
        """Set accumulated edge data to specified graph and edge."""
        nx_multigraph[start_node_key][end_node_key][edge_idx]["names"] = self.names
        nx_multigraph[start_node_key][end_node_key][edge_idx]["routes"] = self.routes
        nx_multigraph[start_node_key][end_node_key][edge_idx]["highways"] = self.highways


def nx_simple_geoms(nx_multigraph: MultiGraph, simplify_dist: int = 5) -> MultiGraph:
    """
    Inferring geometries from node to node.

    Infers straight-lined geometries connecting the `x` and `y` coordinates of each node-pair. The resultant edge
    geometry will be stored to each edge's `geom` attribute.

    Parameters
    ----------
    nx_multigraph: MultiGraph
        A `networkX` `MultiGraph` with `x` and `y` node attributes.
    simplify_dist: int
        Simplification distance to use for simplifying the linestring geometries.

    Returns
    -------
    MultiGraph
        A `networkX` `MultiGraph` with `shapely`
        [`Linestring`](https://shapely.readthedocs.io/en/latest/manual.html#linestrings) geometries assigned to the edge
        `geom` attributes.

    """
    if not isinstance(nx_multigraph, nx.MultiGraph):
        raise TypeError("This method requires an undirected networkX MultiGraph.")
    logger.info("Generating interpolated edge geometries.")
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
    for start_nd_key, end_nd_key, edge_idx in tqdm(g_multi_copy.edges(keys=True), disable=config.QUIET_MODE):
        s_x, s_y = _process_node(start_nd_key)
        e_x, e_y = _process_node(end_nd_key)
        seg = geometry.LineString([[s_x, s_y], [e_x, e_y]])
        seg: geometry.LineString = seg.simplify(simplify_dist)
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
) -> tuple[str, bool]:
    """
    Add a node to a networkX `MultiGraph`. Assembles a new name from source node names. Checks for duplicates.

    Returns new name and is_dupe
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
                return new_nd_name, True  # is_dupe
            # otherwise, bump the appended node number
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
    return new_nd_name, False  # is_dupe


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
    # deduplicate nodes based on x, y matches
    xy_nd_map: dict[str, str] = {}  # from x y to original node index
    nd_merge_map: dict[str, str] = {}  # from original node index to deduplicated node index

    for elem in osm_network_data["elements"]:
        if elem["type"] == "node":
            # all nodes should be string type
            nd_idx = str(elem["id"])
            x = elem["lon"]
            y = elem["lat"]
            nd_x_y = f"{x}-{y}"
            # when deduplicating, sometimes overlapping stairways / underpasses might share an x, y space...
            if "tags" in elem and "level" in elem["tags"]:
                nd_x_y = f'{nd_x_y}-{elem["tags"]["level"]}'
            # only add if non-duplicate
            if nd_x_y not in xy_nd_map:
                xy_nd_map[nd_x_y] = nd_idx
            else:
                logger.warning(f"Merging node {nd_idx} into {xy_nd_map[nd_x_y]} due to identical x, y coords.")
            # ok_nd_idx will correspond to deduplicated node index
            ok_nd_idx = xy_nd_map[nd_x_y]
            # track original node idx to deduplicated idx for reference when building edges
            nd_merge_map[nd_idx] = ok_nd_idx
            nx_multigraph.add_node(ok_nd_idx, x=x, y=y)

    def get_merged_nd_keys(_idx: int) -> tuple[str, str]:
        start_nd_key = str(elem["nodes"][_idx])
        end_nd_key = str(elem["nodes"][_idx + 1])
        return nd_merge_map[start_nd_key], nd_merge_map[end_nd_key]

    for elem in osm_network_data["elements"]:
        if elem["type"] == "way":
            count = len(elem["nodes"])
            if "tags" in elem:
                tags = elem["tags"]
                name = tags["name"] if "name" in tags else None
                ref = tags["ref"] if "ref" in tags else None
                highway = tags["highway"] if "highway" in tags else None
                for idx in range(count - 1):
                    start_nd_key, end_nd_key = get_merged_nd_keys(idx)
                    nx_multigraph.add_edge(
                        start_nd_key,
                        end_nd_key,
                        names=[name],
                        routes=[ref],
                        highways=[highway],
                    )
            else:
                for idx in range(count - 1):
                    start_nd_key, end_nd_key = get_merged_nd_keys(idx)
                    nx_multigraph.add_edge(start_nd_key, end_nd_key)

    return nx_multigraph


def nx_epsg_conversion(nx_multigraph: MultiGraph, from_epsg_code: int, to_epsg_code: int) -> MultiGraph:
    """
    Convert a graph from the `from_epsg_code` EPSG CRS to the `to_epsg_code` EPSG CRS.

    The `to_epsg_code` must be for a projected CRS. If edge `geom` attributes are found, the associated `LineString`
    geometries will also be converted.

    Parameters
    ----------
    nx_multigraph: MultiGraph
        A `networkX` `MultiGraph` with `x` and `y` node attributes in the `from_epsg_code` coordinate system. Optional
        `geom` edge attributes containing `LineString` geoms to be converted.
    from_epsg_code: int
        An integer representing a valid EPSG code specifying the CRS from which the graph must be converted. For
        example, [4326](https://epsg.io/4326) if converting data from an OpenStreetMap response.
    to_epsg_code: int
        An integer representing a valid EPSG code specifying the CRS into which the graph must be projected. For
        example, [27700](https://epsg.io/27700) if converting to British National Grid.

    Returns
    -------
    MultiGraph
        A `networkX` `MultiGraph` with `x` and `y` node attributes converted to the specified `to_epsg_code` coordinate
        system. Edge `geom` attributes will also be converted if found.

    """
    if not isinstance(nx_multigraph, nx.MultiGraph):
        raise TypeError("This method requires an undirected networkX MultiGraph.")
    logger.info(f"Converting networkX graph from EPSG code {from_epsg_code} to EPSG code {to_epsg_code}.")
    g_multi_copy: MultiGraph = nx_multigraph.copy()
    test_crs = CRS.from_epsg(to_epsg_code)
    if not test_crs.is_projected:
        raise ValueError("The to_epsg_code parameter must be for a projected CRS")
    transformer = Transformer.from_crs(from_epsg_code, to_epsg_code, always_xy=True)
    logger.info("Processing node x, y coordinates.")
    nd_key: NodeKey
    node_data: NodeData
    for nd_key, node_data in tqdm(g_multi_copy.nodes(data=True), disable=config.QUIET_MODE):
        # x coordinate
        if "x" not in node_data:
            raise KeyError(f'Encountered node missing "x" coordinate attribute at node {nd_key}.')
        x: float = node_data["x"]
        # y coordinate
        if "y" not in node_data:
            raise KeyError(f'Encountered node missing "y" coordinate attribute at node {nd_key}.')
        y: float = node_data["y"]
        # be cognisant of parameter and return order, using always_xy for transformer
        easting, northing = transformer.transform(x, y)  # pylint: disable=unpacking-non-sequence
        # write back to graph
        g_multi_copy.nodes[nd_key]["x"] = easting
        g_multi_copy.nodes[nd_key]["y"] = northing
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
            if line_geom.geom_type != "LineString":
                raise TypeError(f"Expecting LineString geometry but found {line_geom.geom_type} geometry.")
            # convert
            edge_coords: ListCoordsType = [transformer.transform(x, y) for x, y in line_geom.coords]
            # snap ends
            edge_coords = _snap_linestring_endpoints(g_multi_copy, start_nd_key, end_nd_key, edge_coords)
            # write back to edge
            g_multi_copy[start_nd_key][end_nd_key][edge_idx]["geom"] = geometry.LineString(edge_coords)

    return g_multi_copy


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
        UTM zones may introduce substantial distortions in the results.

    Returns
    -------
    MultiGraph
        A `networkX` `MultiGraph` with `x` and `y` node attributes converted to the local UTM coordinate system. If edge
         `geom` attributes are present, these will also be converted.

    """
    # sample the first node for UTM
    utm_zone_number = force_zone_number
    nd_key = list(nx_multigraph.nodes())[0]
    lng = nx_multigraph.nodes[nd_key]["x"]
    lat = nx_multigraph.nodes[nd_key]["y"]
    is_north = lat >= 0
    is_south = lat < 0
    utm_zone_number, _utm_zone_letter = utm.from_latlon(lat, lng)[2:]  # zone number is position 2
    if force_zone_number is not None:
        utm_zone_number = force_zone_number
    # or dictionary
    crs = CRS.from_dict({"proj": "utm", "zone": utm_zone_number, "north": is_north, "south": is_south})
    target_epsg = crs.to_epsg()
    if not isinstance(target_epsg, int):
        raise ValueError("Unable to extract an EPSG code from the provided network.")
    return nx_epsg_conversion(nx_multigraph, 4326, target_epsg)


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
            nbs: list[NodeKey] = list(nx.neighbors(nx_multigraph, nd_key))
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
                nb_a, nb_b = list(nx.neighbors(nx_multigraph, nb_nd_key))
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
            edge_info = _EdgeInfo()
            while True:
                edge_data: EdgeData = nx_multigraph[trailing_nd][next_link_nd][0]
                edge_info.gather_edge_info(edge_data)
                # aggregate the geom
                try:
                    # there is ordinarily a single edge from trailing to next
                    # however, there is an edge case where next is a dead-end with two edges linking back to trailing
                    # (i.e. where one of those edges is longer than the maximum length discrepancy for merging edges)
                    # in either case, use the first geom
                    geom: geometry.LineString = edge_data["geom"]
                except KeyError as err:
                    raise KeyError(f'Missing "geom" attribute for edge {trailing_nd}-{next_link_nd}') from err
                if geom.geom_type != "LineString":
                    raise TypeError(f"Expecting LineString geometry but found {geom.geom_type} geometry.")
                # welds can be done automatically, but there are edge cases, e.g.:
                # looped roadways or overlapping edges such as stairways don't know which sides of two segments to join
                # i.e. in these cases the edges can sometimes be matched from one of two possible configurations
                # since the x_y join is known for all cases it is used here regardless
                trailing_nd_data: NodeData = nx_multigraph.nodes[trailing_nd]
                override_xy = (trailing_nd_data["x"], trailing_nd_data["y"])
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
                new_nbs: list[NodeKey] = list(nx.neighbors(nx_multigraph, next_link_nd))
                if len(new_nbs) == 1:
                    trailing_nd = next_link_nd
                    next_link_nd = new_nbs[0]
                # but in almost all cases there will be two neighbours, one of which will be the previous node
                else:
                    nb_a, nb_b = list(nx.neighbors(nx_multigraph, next_link_nd))
                    # proceed to the new_next node
                    if nb_a == trailing_nd:
                        trailing_nd = next_link_nd
                        next_link_nd = nb_b
                    else:
                        trailing_nd = next_link_nd
                        next_link_nd = nb_a
            # checks and snapping
            agg_geom = _snap_linestring_endpoints(g_multi_copy, anchor_nd, end_nd, agg_geom)
            # create a new linestring
            new_geom = geometry.LineString(agg_geom)
            if new_geom.geom_type != "LineString":
                raise TypeError(
                    f'Found {new_geom.geom_type} geometry instead of "LineString" for new geom {new_geom.wkt}.'
                    f"Check that the adjacent LineStrings in the vicinity of {nd_key} are not corrupted."
                )
            # add a new edge from anchor_nd to end_nd
            edge_idx = g_multi_copy.add_edge(
                anchor_nd,
                end_nd,
                geom=new_geom,
            )
            edge_info.set_edge_info(g_multi_copy, anchor_nd, end_nd, edge_idx)
            # drop the removed nodes, which will also implicitly drop the related edges
            g_multi_copy.remove_nodes_from(drop_nodes)
            removed_nodes.update(drop_nodes)

    return g_multi_copy


def nx_remove_dangling_nodes(
    nx_multigraph: MultiGraph,
    despine: Optional[float] = None,
    remove_disconnected: bool = True,
    cleanup_filler_nodes: bool = True,
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
    cleanup_filler_nodes: bool
        Removal of dangling nodes can result in "filler nodes" of degree two where dangling streets were removed.
        If cleanup_filler_nodes is `True` then these will be removed.

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
        connected_components: list[list[NodeKey]] = list(nx.algorithms.components.connected_components(g_multi_copy))
        # sort by largest component
        g_nodes: list[NodeKey] = sorted(connected_components, key=len, reverse=True)[0]
        # make a copy of the graph using the largest component
        g_multi_copy: MultiGraph = nx.MultiGraph(g_multi_copy.subgraph(g_nodes))
    # remove dangleres
    if despine is not None and despine > 0:
        remove_nodes = []
        nd_key: NodeKey
        for nd_key in tqdm(g_multi_copy.nodes(data=False), disable=config.QUIET_MODE):
            if nx.degree(g_multi_copy, nd_key) == 1:
                # only a single neighbour, so index-in directly and update at key = 0
                nb_nd_key: NodeKey = list(nx.neighbors(g_multi_copy, nd_key))[0]
                if g_multi_copy[nd_key][nb_nd_key][0]["geom"].length <= despine:
                    remove_nodes.append(nd_key)
        g_multi_copy.remove_nodes_from(remove_nodes)
    # cleanup leftover fillers
    if cleanup_filler_nodes:
        g_multi_copy = nx_remove_filler_nodes(g_multi_copy)

    return g_multi_copy


def merge_parallel_edges(
    nx_multigraph: MultiGraph,
    merge_edges_by_midline: bool,
    contains_buffer_dist: int,
) -> MultiGraph:
    """
    Check a MultiGraph for duplicate edges; which, if found, will be merged.

    The shortest of these parallel edges is selected and buffered by `contains_buffer_dist`. If this buffer contains an
    adjacent edge, then the adjacent edge is merged. Edges falling outside this buffer are retained.

    When candidate edges are found for merging, they are replaced by a single new edge. The new geometry selected from
    either:
    - An imaginary centreline of the combined edges if `merge_edges_by_midline` is set to `True`;
    - Else, the shortest edge is retained, with longer edges discarded.

    Parameters
    ----------
    nx_multigraph: MultiGraph
        A `networkX` `MultiGraph` in a projected coordinate system, containing `x` and `y` node attributes, and `geom`
        edge attributes containing `LineString` geoms.
    merge_edges_by_midline: bool
        Whether to merge parallel edges by an imaginary centreline. If set to False, then the shortest edge will be
        retained as the new geometry and the longer edges will be discarded. Defaults to True.
    contains_buffer_dist: int
        The buffer distance to consider when checking if parallel edges are sufficiently similar to be merged.

    Returns
    -------
    MultiGraph
        A `networkX` `MultiGraph` with consolidated nodes.

    """
    if not isinstance(nx_multigraph, nx.MultiGraph):
        raise TypeError("This method requires an undirected networkX MultiGraph (for multiple edges).")
    if contains_buffer_dist < 1:
        raise TypeError("contains_buffer_dist should be greater or equal to 1. ")
    logger.info(f"Merging parallel edges within buffer of {contains_buffer_dist}.")
    # don't use copy() - add nodes only
    deduped_graph: MultiGraph = nx.MultiGraph()
    deduped_graph.add_nodes_from(nx_multigraph.nodes(data=True))
    # iter the edges
    start_nd_key: NodeKey
    end_nd_key: NodeKey
    edge_data: EdgeData
    for start_nd_key, end_nd_key, edge_data in tqdm(nx_multigraph.edges(data=True), disable=config.QUIET_MODE):
        # if only one edge is associated with this node pair, then add
        if nx_multigraph.number_of_edges(start_nd_key, end_nd_key) == 1:
            deduped_graph.add_edge(start_nd_key, end_nd_key, **edge_data)
        # otherwise, add if not already added from another (parallel) edge
        elif not deduped_graph.has_edge(start_nd_key, end_nd_key):
            # there are normally max two edges, but sometimes three or more
            edges_data: list[EdgeData] = []
            for edge_data in nx_multigraph.get_edge_data(start_nd_key, end_nd_key).values():  # type: ignore
                edges_data.append(edge_data)
            edge_info = _EdgeInfo()
            # find the shortest of the geoms
            edge_geoms = [edge["geom"] for edge in edges_data]
            edge_lens = [geom.length for geom in edge_geoms]
            shortest_idx = edge_lens.index(min(edge_lens))
            shortest_geom = edge_geoms.pop(shortest_idx)
            shortest_data = edges_data.pop(shortest_idx)
            # start by gathering shortest's data
            edge_info.gather_edge_info(shortest_data)
            # process longer geoms
            longer_geoms: list[geometry.LineString] = []
            for edge_geom, edge_data in zip(edge_geoms, edges_data):
                # discard distinct edges where the buffer of the shorter contains the longer
                is_contained = shortest_geom.buffer(contains_buffer_dist).contains(edge_geom)
                if is_contained:
                    edge_info.gather_edge_info(edge_data)
                    longer_geoms.append(edge_geom)
                else:
                    edge_data_copy = {k: v for k, v in edge_data.items() if k != "geom"}
                    deduped_graph.add_edge(start_nd_key, end_nd_key, geom=edge_geom, **edge_data_copy)
            # otherwise, if not merging on a midline basis
            # or, if no other edges to process (in cases where longer geom has been retained per above)
            # then use the shortest geom
            if not merge_edges_by_midline or len(longer_geoms) == 0:
                edge_idx = deduped_graph.add_edge(start_nd_key, end_nd_key, geom=shortest_geom)
                edge_info.set_edge_info(deduped_graph, start_nd_key, end_nd_key, edge_idx)
            # otherwise weld the geoms, using the shortest as a yardstick
            else:
                # iterate the coordinates along the shorter geom
                # starting and endpoint geoms already match
                new_coords = []
                for coord in shortest_geom.coords:
                    # from the current short_geom coordinate
                    short_point = geometry.Point(coord)
                    # find the nearest points on the longer geoms
                    multi_coords = [short_point]
                    for longer_geom in longer_geoms:
                        # get the nearest point on the longer geom
                        # returns a tuple of nearest geom for respective input geoms
                        longer_point: geometry.Point = ops.nearest_points(short_point, longer_geom)[-1]
                        # only use for new coord centroid if not the starting or end point
                        lg_start_point = geometry.Point(longer_geom.coords[0])
                        lg_end_point = geometry.Point(longer_geom.coords[-1])
                        if lg_start_point.distance(longer_point) < 1 or lg_end_point.distance(longer_point) < 1:
                            continue
                        # aggregate
                        multi_coords.append(longer_point)
                    # create a midpoint between the geoms and add to the new coordinate array
                    mid_point: geometry.Point = geometry.MultiPoint(multi_coords).centroid
                    new_coords.append((mid_point.x, mid_point.y))
                # generate the new mid-line geom
                new_coords = _snap_linestring_endpoints(deduped_graph, start_nd_key, end_nd_key, new_coords)
                new_geom = geometry.LineString(new_coords)
                # add to the graph
                edge_idx = deduped_graph.add_edge(start_nd_key, end_nd_key, geom=new_geom)
                edge_info.set_edge_info(deduped_graph, start_nd_key, end_nd_key, edge_idx)

    return deduped_graph


def nx_snap_endpoints(nx_multigraph: MultiGraph) -> MultiGraph:
    """
    Snaps geom endpoints to adjacent node coordinates.

    Parameters
    ----------
    nx_multigraph: MultiGraph
        A `networkX` `MultiGraph` in a projected coordinate system, containing `x` and `y` node attributes, and `geom`
        edge attributes containing `LineString` geoms.

    Returns
    -------
    MultiGraph
        A `networkX` `MultiGraph`.

    """
    logger.info("Snapping edge endpoints.")
    g_multi_copy: MultiGraph = nx_multigraph.copy()
    start_nd_key: NodeKey
    end_nd_key: NodeKey
    for start_nd_key, end_nd_key, edge_idx, edge_data in tqdm(
        g_multi_copy.edges(keys=True, data=True)  # , disable=config.QUIET_MODE
    ):
        edge_geom: geometry.LineString = edge_data["geom"]
        edge_geom = geometry.LineString(
            _snap_linestring_endpoints(g_multi_copy, start_nd_key, end_nd_key, edge_geom.coords)
        )
        g_multi_copy[start_nd_key][end_nd_key][edge_idx]["geom"] = edge_geom
    return g_multi_copy


def nx_iron_edges(
    nx_multigraph: MultiGraph,
) -> MultiGraph:
    """
    Simplifies edges.

    Parameters
    ----------
    nx_multigraph: MultiGraph
        A `networkX` `MultiGraph` in a projected coordinate system, containing `x` and `y` node attributes, and `geom`
        edge attributes containing `LineString` geoms.

    Returns
    -------
    MultiGraph
        A `networkX` `MultiGraph` with simplified edges.

    """
    logger.info("Ironing edges.")
    g_multi_copy: MultiGraph = nx_multigraph.copy()
    start_nd_key: NodeKey
    end_nd_key: NodeKey
    for start_nd_key, end_nd_key, edge_idx, edge_data in tqdm(
        g_multi_copy.edges(keys=True, data=True), disable=config.QUIET_MODE
    ):
        edge_geom: geometry.LineString = edge_data["geom"]
        # for all changes - write over edge_geom and also update in place
        total_angle = _measure_cumulative_angle(edge_geom.coords)
        if total_angle < 5:
            edge_geom = edge_geom.simplify(20, preserve_topology=False)
        elif total_angle < 10:
            edge_geom = edge_geom.simplify(10, preserve_topology=False)
        elif total_angle < 20:
            edge_geom = edge_geom.simplify(5, preserve_topology=False)
        else:
            edge_geom = edge_geom.simplify(2, preserve_topology=True)
        g_multi_copy[start_nd_key][end_nd_key][edge_idx]["geom"] = edge_geom
    # straightening parallel edges can create duplicates
    g_multi_copy = merge_parallel_edges(g_multi_copy, False, 1)

    return g_multi_copy


def _squash_adjacent(
    nx_multigraph: MultiGraph,
    node_group: set[NodeKey],
    centroid_by_straightness: bool,
    centroid_by_min_len_factor: float | None = 0.9,
) -> MultiGraph:
    """
    Squash nodes from a specified node group down to a new node.

    The new node can either be based on:
    - The centroid of all nodes;
    - else, all nodes of degree greater or equal to cent_min_degree;
    - and / else, all nodes with cumulative adjacent OSM street names or routes greater than cent_min_names;
    - and / else, all nodes with aggregate adjacent edge lengths greater than centroid_by_min_len_factor as a factor of
    the node with the greatest overall aggregate lengths. Edges are adjusted from the old nodes to the new combined
    node.

    """
    if not isinstance(nx_multigraph, nx.MultiGraph):
        raise TypeError("This method requires an undirected networkX MultiGraph (for multiple edges).")
    if centroid_by_min_len_factor is not None and not 1 >= centroid_by_min_len_factor >= 0:
        raise ValueError("centroid_by_min_len_factor should be a float between 0 and 1.")
    # remove any node keys no longer in the graph
    node_group = [nd_key for nd_key in node_group if nd_key in nx_multigraph]
    # straightness basis - looks for intersections bridging relatively straight through routes
    centroid_nodes: list[int] = []
    if centroid_by_straightness is True:
        for nd_key in node_group:
            nd_x_y = (nx_multigraph.nodes[nd_key]["x"], nx_multigraph.nodes[nd_key]["y"])
            for nb_nd_key_a in nx.neighbors(nx_multigraph, nd_key):
                for nb_edge_data_a in nx_multigraph[nd_key][nb_nd_key_a].values():
                    geom_a_coords = align_linestring_coords(nb_edge_data_a["geom"].coords, nd_x_y, reverse=True)
                    for nb_nd_key_b in nx.neighbors(nx_multigraph, nd_key):
                        if nb_nd_key_b == nb_nd_key_a:
                            continue
                        for nb_edge_data_b in nx_multigraph[nd_key][nb_nd_key_b].values():
                            geom_b_coords = align_linestring_coords(
                                nb_edge_data_b["geom"].coords, nd_x_y, reverse=False
                            )
                            # take the last linestring coordinates before / first coordinate after intersection
                            angular_change = _measure_coords_angle(geom_a_coords[-2][:2], nd_x_y, geom_b_coords[1][:2])
                            # if straight through - then prioritise this centroid
                            if angular_change < 5:
                                centroid_nodes.append(nd_key)
                                continue
    # if merging on a longest adjacent edges basis
    if centroid_by_min_len_factor is not None:
        # if nodes are pre-filtered by edge degrees, then use the filtered nodes as a starting point
        if centroid_nodes:
            node_pool = centroid_nodes.copy()
            centroid_nodes = []  # reset
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
            if agg_len >= max_len * centroid_by_min_len_factor:
                centroid_nodes.append(nd_key)
    # fallback
    if not centroid_nodes:
        centroid_nodes = node_group
    # prepare the names and geoms for filtered points to be used for the new centroid
    node_geoms = []
    coords_set: set[str] = set()
    for nd_key in centroid_nodes:
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
    new_cent: geometry.Point = geometry.MultiPoint(node_geoms).centroid
    # now that the centroid is known, add the new node
    new_nd_name, is_dupe = _add_node(nx_multigraph, node_group, x=new_cent.x, y=new_cent.y)
    if is_dupe:
        # an edge case: if the potential duplicate was one of the node group then it doesn't need adding
        if new_nd_name in node_group:
            # but remove from the node group since it doesn't need to be removed and replumbed
            node_group.remove(new_nd_name)
        else:
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
                if line_geom.geom_type != "LineString":
                    raise TypeError(
                        f"Expecting LineString geometry but found {line_geom.geom_type} geometry "
                        f"for edge {nd_key}-{nb_nd_key}."
                    )
                # orient the LineString so that the geom starts from the node's x_y
                nd_data: NodeData = nx_multigraph.nodes[nd_key]
                nd_xy = (nd_data["x"], nd_data["y"])
                line_coords = align_linestring_coords(line_geom.coords, nd_xy)
                # update geom starting point to new parent node's coordinates
                line_coords = snap_linestring_startpoint(line_coords, (new_cent.x, new_cent.y))
                # if self-loop, then the end also needs updating to the new centroid
                if nd_key == nb_nd_key:
                    line_coords = snap_linestring_endpoint(line_coords, (new_cent.x, new_cent.y))
                    target_nd_key = new_nd_name
                else:
                    target_nd_key = nb_nd_key
                # build the new geom
                new_edge_geom = geometry.LineString(line_coords)
                # check that a duplicate is not being added
                dupe = False
                if nx_multigraph.has_edge(new_nd_name, target_nd_key):
                    # only add parallel edges if substantially different from any existing edges
                    n_edges: int = nx_multigraph.number_of_edges(new_nd_name, target_nd_key)
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
                    edge_idx = nx_multigraph.add_edge(
                        new_nd_name,
                        target_nd_key,
                        geom=new_edge_geom,
                    )
                    edge_info = _EdgeInfo()
                    edge_info.gather_edge_info(edge_data)
                    edge_info.set_edge_info(nx_multigraph, new_nd_name, target_nd_key, edge_idx)
        # drop the node, this will also implicitly drop the old edges
        nx_multigraph.remove_node(nd_key)

    return nx_multigraph


def _create_nodes_strtree(nx_multigraph: MultiGraph) -> tuple[strtree.STRtree, list[dict[str, Any]]]:
    """
    Create a nodes-based STRtree spatial index.
    """
    node_geoms = []
    node_lookups: list[dict[str, Any]] = []
    nd_key: NodeKey
    node_data: NodeData
    logger.info("Creating nodes STR tree")
    for nd_key, node_data in tqdm(nx_multigraph.nodes(data=True), disable=config.QUIET_MODE):
        # x coordinate
        if "x" not in node_data:
            raise KeyError(f'Encountered node missing "x" coordinate attribute at node {nd_key}.')
        x: float = node_data["x"]
        # y coordinate
        if "y" not in node_data:
            raise KeyError(f'Encountered node missing "y" coordinate attribute at node {nd_key}.')
        y: float = node_data["y"]
        point_geom = geometry.Point(x, y)
        node_geoms.append(point_geom)
        node_lookups.append({"nd_key": nd_key, "nd_degree": nx.degree(nx_multigraph, nd_key)})
    nodes_tree = strtree.STRtree(node_geoms)
    return nodes_tree, node_lookups


def _create_edges_strtree(nx_multigraph: MultiGraph) -> tuple[strtree.STRtree, list[dict[str, Any]]]:
    """
    Create an edges-based STRtree spatial index.
    """
    edge_geoms = []
    edge_lookups: list[dict[str, Any]] = []
    start_nd_key: NodeKey
    end_nd_key: NodeKey
    edge_idx: int
    edge_data: EdgeData
    logger.info("Creating edges STR tree.")
    for start_nd_key, end_nd_key, edge_idx, edge_data in tqdm(
        nx_multigraph.edges(keys=True, data=True), disable=config.QUIET_MODE
    ):
        if "geom" not in edge_data:
            raise KeyError('Encountered edge missing "geom" attribute.')
        linestring = edge_data["geom"]
        edge_geoms.append(linestring)
        edge_lookups.append({"start_nd_key": start_nd_key, "end_nd_key": end_nd_key, "edge_idx": edge_idx})
    edge_tree = strtree.STRtree(edge_geoms)
    return edge_tree, edge_lookups


def nx_consolidate_nodes(
    nx_multigraph: MultiGraph,
    buffer_dist: float = 5,
    neighbour_policy: str | None = None,
    crawl: bool = False,
    centroid_by_straightness: bool = True,
    centroid_by_min_len_factor: Optional[float] = None,
    merge_edges_by_midline: bool = True,
    contains_buffer_dist: int = 20,
) -> MultiGraph:
    """
    Consolidates nodes if they are within a buffer distance of each other.

    Several parameters provide more control over the conditions used for deciding whether or not to merge nodes. The
    algorithm proceeds in two steps:

    Nodes within the buffer distance of each other are merged. A new centroid will be determined and all existing
    edge endpoints will be updated accordingly. The new centroid for the merged nodes can be based on:
    - The centroid of the node group;
    - Else, all nodes of degree greater or equal to `cent_min_degree`;
    - Else, all nodes with aggregate adjacent edge lengths greater than a factor of `centroid_by_min_len_factor` of the
    node with the greatest aggregate length for adjacent edges.

    The merging of nodes can create parallel edges with mutually shared nodes on either side. These edges are replaced
    by a single new edge, with the new geometry selected from either:
    - An imaginary centreline of the combined edges if `merge_edges_by_midline` is set to `True`;
    - Else, the shortest edge, with longer edges discarded;
    See [`merge_parallel_edges`](#merge-parallel-edges) for more information.

    Parameters
    ----------
    nx_multigraph: MultiGraph
        A `networkX` `MultiGraph` in a projected coordinate system, containing `x` and `y` node attributes, and `geom`
        edge attributes containing `LineString` geoms.
    buffer_dist: float
        The buffer distance to be used for consolidating nearby nodes. Defaults to 5.
    neighbour_policy: str
        Whether all nodes within the buffer distance are merged, or only "direct" or "indirect" neighbours. Defaults to
        None.
    crawl: bool
        Whether the algorithm will recursively explore neighbours of neighbours if those neighbours are within the
        buffer distance from the prior node. Defaults to True.
    centroid_by_straightness: bool
        Whether to use an intersection straightness heuristic to select new centroids. True by default.
    centroid_by_min_len_factor: float
        The minimum aggregate adjacent edge lengths an existing node should have to be considered when calculating the
        centroid for the new node cluster. Expressed as a factor of the node with the greatest aggregate adjacent edge
        lengths. Defaults to None.
    merge_edges_by_midline: bool
        Whether to merge parallel edges by an imaginary centreline. If set to False, then the shortest edge will be
        retained as the new geometry and the longer edges will be discarded. Defaults to True.
    contains_buffer_dist: int
        The buffer distance to consider when checking if parallel edges are sufficiently similar to be merged.

    Returns
    -------
    MultiGraph
        A `networkX` `MultiGraph` with consolidated nodes.

    Examples
    --------
    See the guide on [graph cleaning](/guide#graph-cleaning) for more information.

    ![Example raw graph from OSM](/images/graph_cleaning_1.png)
    _The pre-consolidation OSM street network for Soho, London. © OpenStreetMap contributors._

    ![Example cleaned graph](/images/graph_cleaning_5.png)
    _The consolidated OSM street network for Soho, London. © OpenStreetMap contributors._

    """
    if not isinstance(nx_multigraph, nx.MultiGraph):
        raise TypeError("This method requires an undirected networkX MultiGraph.")
    if neighbour_policy is not None and neighbour_policy not in ("direct", "indirect"):
        raise ValueError('Neighbour policy should be "direct", "indirect", or the default of "None"')
    if crawl and buffer_dist > 25:
        logger.warning("Be cautious with large buffer distances when using crawl!")
    _multi_graph: MultiGraph = nx_multigraph.copy()
    # create a nodes STRtree
    nodes_tree, node_lookups = _create_nodes_strtree(_multi_graph)
    # iter
    logger.info("Consolidating nodes.")
    # keep track of removed nodes
    removed_nodes: set[NodeKey] = set()

    def recursive_squash(
        nd_key: NodeKey,
        x: float,
        y: float,
        node_group: set[NodeKey],
        processed_nodes: set[NodeKey],
        recursive: bool = False,
    ) -> set[NodeKey]:
        # keep track of which nodes have been processed as part of recursion
        processed_nodes.add(nd_key)
        # get all other nodes within buffer distance - the self-node and previously processed nodes are also returned
        j_hits: list[int] = nodes_tree.query(geometry.Point(x, y).buffer(buffer_dist))
        # review each node within the buffer
        j_nd_key: NodeKey
        for j_hit_idx in j_hits:
            j_lookup: dict[str, Any] = node_lookups[j_hit_idx]
            j_nd_key = j_lookup["nd_key"]
            if j_nd_key in removed_nodes or j_nd_key in processed_nodes:
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
            node_group.add(j_nd_key)
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
        if nd_key in removed_nodes:
            continue
        node_group = recursive_squash(
            nd_key,  # node nd_key
            nd_data["x"],  # x point for buffer
            nd_data["y"],  # y point for buffer
            set([nd_key]),  # node group for consolidation (with starting node)
            set(),  # processed nodes tracked through recursion
            crawl,
        )  # whether to recursively probe neighbours per distance
        # update removed nodes
        removed_nodes.update(node_group)
        # consolidate if nodes have been identified within buffer and if these exceed min_node_threshold
        _multi_graph = _squash_adjacent(
            _multi_graph,
            node_group,
            centroid_by_straightness=centroid_by_straightness,
            centroid_by_min_len_factor=centroid_by_min_len_factor,
        )
    # remove new filler nodes
    _multi_graph = nx_remove_filler_nodes(_multi_graph)
    # remove wonky endings from consolidation process
    _multi_graph = nx_iron_edges(_multi_graph)
    # remove parallel edges resulting from squashing nodes
    _multi_graph = merge_parallel_edges(_multi_graph, merge_edges_by_midline, contains_buffer_dist)

    return _multi_graph


def nx_split_opposing_geoms(
    nx_multigraph: MultiGraph,
    buffer_dist: float = 10,
    merge_edges_by_midline: bool = True,
    contains_buffer_dist: float = 20,
) -> MultiGraph:
    """
    Split edges opposite nodes on parallel edge segments if within a buffer distance.

    This facilitates merging parallel roadways through subsequent use of
    [`nx-consolidate-nodes`](#nx-consolidate-nodes).

    The merging of nodes can create parallel edges with mutually shared nodes on either side. These edges are replaced
        by a single new edge, with the new geometry selected from either:
        - An imaginary centreline of the combined edges if `merge_edges_by_midline` is set to `True`;
        - Else, the shortest edge, with longer edges discarded;
    See [`merge_parallel_edges`](#merge-parallel-edges) for more information.

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
    contains_buffer_dist: float
        The buffer distance to consider when checking if parallel edges are sufficiently similar to be merged.

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
    _multi_graph: MultiGraph = nx_multigraph.copy()
    # create an edges STRtree (nodes and edges)
    edges_tree, edge_lookups = _create_edges_strtree(_multi_graph)
    # iter
    logger.info("Splitting opposing edges.")
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
        edge_hits: list[int] = edges_tree.query(n_point.buffer(buffer_dist))
        # extract the start node, end node, geom
        edges: list[EdgeMapping] = []
        for edge_hit_idx in edge_hits:
            edge_lookup = edge_lookups[edge_hit_idx]
            start_nd_key = edge_lookup["start_nd_key"]
            end_nd_key = edge_lookup["end_nd_key"]
            edge_idx = edge_lookup["edge_idx"]
            edge_geom: geometry.LineString = nx_multigraph[start_nd_key][end_nd_key][edge_idx]["geom"]
            edges.append((start_nd_key, end_nd_key, edge_idx, edge_geom))
        # check against removed edges
        current_edges: list[EdgeMapping] = []
        for start_nd_key, end_nd_key, edge_idx, edge_geom in edges:
            recurse_child_keys(start_nd_key, end_nd_key, edge_idx, edge_geom, current_edges)
        # get neighbouring nodes from new graph
        neighbours: list[NodeKey] = list(_multi_graph.neighbors(nd_key))
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
            # don't proceed with splits for short geoms
            if edge_geom.length < buffer_dist:
                continue
            # project a point and split the opposing geom
            # ops.nearest_points returns tuple of nearest from respective input geoms
            # want the nearest point on the line at index 1
            nearest_point: geometry.Point = ops.nearest_points(n_geom, edge_geom)[-1]
            # if a valid nearest point has been found, go ahead and split the geom
            # use a snap because rounding precision errors will otherwise cause issues
            split_geoms: geometry.GeometryCollection = ops.split(
                ops.snap(edge_geom, nearest_point, 0.01), nearest_point
            )
            # in some cases the line will be pointing away, but is still near enough to be within max
            # in these cases a single geom will be returned
            if len(split_geoms.geoms) < 2:
                continue
            new_edge_geom_a: geometry.LineString
            new_edge_geom_b: geometry.LineString
            new_edge_geom_a, new_edge_geom_b = split_geoms.geoms
            # don't proceed with split for overly short split segments
            if new_edge_geom_a.length < buffer_dist or new_edge_geom_b.length < buffer_dist:
                continue
            # add the new node and edges to _multi_graph (don't modify nx_multigraph because of iter in place)
            new_nd_name, is_dupe = _add_node(
                _multi_graph, [start_nd_key, nd_key, end_nd_key], x=nearest_point.x, y=nearest_point.y
            )
            # continue if a node already exists at this location
            if is_dupe:
                continue
            edge_data: EdgeData = _multi_graph[start_nd_key][end_nd_key][edge_idx]
            edge_data_copy = {k: v for k, v in edge_data.items() if k != "geom"}
            _multi_graph.add_edge(start_nd_key, new_nd_name, **edge_data_copy)
            _multi_graph.add_edge(end_nd_key, new_nd_name, **edge_data_copy)
            # get starting geom for orientation
            s_nd_data: NodeData = _multi_graph.nodes[start_nd_key]
            s_nd_geom = geometry.Point(s_nd_data["x"], s_nd_data["y"])
            if np.allclose(
                s_nd_geom.coords,
                new_edge_geom_a.coords[0][:2],
                atol=config.ATOL,
                rtol=0,
            ) or np.allclose(
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
    deduped_graph = merge_parallel_edges(_multi_graph, merge_edges_by_midline, contains_buffer_dist)

    return deduped_graph


def _measure_bearing(xy_1: npt.NDArray[np.float_], xy_2: npt.NDArray[np.float_]) -> float:
    """Measures the angular bearing between two coordinate pairs."""
    y_1, x_1 = xy_1[::-1]
    y_2, x_2 = xy_2[::-1]
    return np.rad2deg(np.arctan2(y_2 - y_1, x_2 - x_1))


def _measure_coords_angle(
    coords_1: npt.NDArray[np.float_], coords_2: npt.NDArray[np.float_], coords_3: npt.NDArray[np.float_]
) -> float:
    """Measures angle between three coordinate pairs."""
    # arctan2 is y / x order
    a_1: float = _measure_bearing(coords_2, coords_1)
    a_2: float = _measure_bearing(coords_3, coords_2)
    angle = np.abs((a_2 - a_1 + 180) % 360 - 180)
    return angle


def _measure_linestring_angle(linestring_coords: ListCoordsType, idx_a: int, idx_b: int, idx_c: int) -> float:
    """Measures angle between two segment bearings per indices."""
    coords_1: npt.NDArray[np.float_] = np.array(linestring_coords[idx_a])[:2]
    coords_2: npt.NDArray[np.float_] = np.array(linestring_coords[idx_b])[:2]
    coords_3: npt.NDArray[np.float_] = np.array(linestring_coords[idx_c])[:2]
    # arctan2 is y / x order
    a_1: float = _measure_bearing(coords_2, coords_1)
    a_2: float = _measure_bearing(coords_3, coords_2)
    angle = np.abs((a_2 - a_1 + 180) % 360 - 180)
    # alternative
    # A: npt.NDArray[np.float_] = coords_2 - coords_1
    # B: npt.NDArray[np.float_] = coords_3 - coords_2
    # alt_angle = np.abs(np.degrees(np.math.atan2(np.linalg.det([A, B]), np.dot(A, B))))

    return angle


def _measure_cumulative_angle(linestring_coords: ListCoordsType) -> float:
    """Measures the cumulative angle along a LineString geom's coords."""
    angle_sum: float = 0
    for c_idx in range(len(linestring_coords) - 2):
        angle_sum += _measure_linestring_angle(linestring_coords, c_idx, c_idx + 1, c_idx + 2)

    return angle_sum


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
    for start_nd_key, end_nd_key, edge_data in tqdm(nx_multigraph.edges(data=True), disable=config.QUIET_MODE):
        # test for x, y in start coordinates
        if "x" not in nx_multigraph.nodes[start_nd_key] or "y" not in nx_multigraph.nodes[start_nd_key]:
            raise KeyError(f'Encountered node missing "x" or "y" coordinate attributes at node {start_nd_key}.')
        # test for x, y in end coordinates
        if "x" not in nx_multigraph.nodes[end_nd_key] or "y" not in nx_multigraph.nodes[end_nd_key]:
            raise KeyError(f'Encountered node missing "x" or "y" coordinate attributes at node {end_nd_key}.')
        # test for geom
        if "geom" not in edge_data:
            raise KeyError(
                f"No edge geom found for edge {start_nd_key}-{end_nd_key}: "
                f'Please add an edge "geom" attribute consisting of a shapely LineString.'
            )
        # get edge geometry
        line_geom: geometry.LineString = edge_data["geom"]
        if line_geom.geom_type != "LineString":
            raise TypeError(
                f"Expected LineString geometry but found {line_geom.geom_type} for edge {start_nd_key}-{end_nd_key}."
            )
        # check geom coordinates directionality - flip if facing backwards direction
        line_geom_coords = _snap_linestring_endpoints(nx_multigraph, start_nd_key, end_nd_key, line_geom.coords)
        line_geom: geometry.LineString = geometry.LineString(line_geom_coords)
        # see how many segments are necessary so as not to exceed decomposition max distance
        # note that a length less than the decompose threshold will result in a single 'sub'-string
        cuts: int = int(np.ceil(line_geom.length / decompose_max))
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
            line_segment: geometry.LineString = _substring(line_geom, step, step + step_size)
            # get the x, y of the new end node
            x, y = line_segment.coords[-1]
            # add the new node and edge
            new_nd_name, is_dupe = _add_node(g_multi_copy, [start_nd_key, sub_node_counter, end_nd_key], x=x, y=y)
            if is_dupe:
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
            edge_data_copy = {k: v for k, v in edge_data.items() if k != "geom"}
            g_multi_copy.add_edge(prior_node_id, new_nd_name, geom=line_segment, **edge_data_copy)
            # increment the step and node id
            prior_node_id = new_nd_name
            step += step_size
        # set the last edge manually to avoid rounding errors at end of LineString
        # the nodes already exist, so just add edge
        line_segment = _substring(line_geom, step, line_geom.length)
        edge_data_copy = {k: v for k, v in edge_data.items() if k != "geom"}
        g_multi_copy.add_edge(prior_node_id, end_nd_key, geom=line_segment, **edge_data_copy)

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
        split and welded to form the new dual `geom` edge attributes. `primal_edge_node_a`, `primal_edge_node_b`, and
        `primal_edge_idx` attributes will be added to the new (dual) nodes, and a `primal_node_id` edge attribute
        will be added to the new (dual) edges. This is useful for welding the primal geometry to the dual
        representations where useful for purposes such as visualisation, or otherwise welding downstream metrics to
        source (primal) geometries.

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
    g_dual: MultiGraph = nx.MultiGraph()
    g_dual.graph["is_dual"] = True

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
        a_node_data: NodeData = nx_multigraph_ref.nodes[a_node]
        a_xy = (a_node_data["x"], a_node_data["y"])
        b_node_data: NodeData = nx_multigraph_ref.nodes[b_node]
        b_xy = (b_node_data["x"], b_node_data["y"])
        # test for geom
        if "geom" not in edge_data:
            raise KeyError(
                f"No edge geom found for edge {a_node}-{b_node}: "
                f'Please add an edge "geom" attribute consisting of a shapely LineString.'
            )
        # get edge geometry
        line_geom = edge_data["geom"]
        if line_geom.geom_type != "LineString":
            raise TypeError(
                f"Expecting LineString geometry but found {line_geom.geom_type} geometry for edge {a_node}-{b_node}."
            )
        # align geom coordinates to start from A side
        line_geom_coords = align_linestring_coords(line_geom.coords, a_xy)
        line_geom = geometry.LineString(line_geom_coords)
        # generate the two half geoms
        a_half_geom: geometry.LineString = _substring(line_geom, 0.0, 0.5, normalized=True)
        b_half_geom: geometry.LineString = _substring(line_geom, 0.5, 1.0, normalized=True)
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
        a_half_geom_coords = snap_linestring_startpoint(a_half_geom.coords, a_xy)
        # snap new midpoint to geom A's endpoint (i.e. no need to snap endpoint of geom A)
        mid_xy = a_half_geom_coords[-1][:2]
        # B side geom starts at mid and ends at B node
        b_half_geom_coords = snap_linestring_startpoint(b_half_geom.coords, mid_xy)
        b_half_geom_coords = snap_linestring_endpoint(b_half_geom_coords, b_xy)
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

    def prepare_dual_node_key(start_nd_key: NodeKey, end_nd_key: NodeKey, edge_idx: int) -> str:
        s_e = sorted([str(start_nd_key), str(end_nd_key)])
        return f"{s_e[0]}_{s_e[1]}_k{edge_idx}"

    # add dual nodes
    start_nd_key: NodeKey
    end_nd_key: NodeKey
    edge_idx: int
    logger.info("Preparing dual nodes")
    for start_nd_key, end_nd_key, edge_idx, edge_data in tqdm(
        nx_multigraph.edges(data=True, keys=True), disable=config.QUIET_MODE
    ):
        mid_point = edge_data["geom"].interpolate(0.5, normalized=True)  # type: ignore
        dual_node_key = prepare_dual_node_key(start_nd_key, end_nd_key, edge_idx)
        # create a new dual node corresponding to the current primal edge
        g_dual.add_node(
            dual_node_key,
            x=mid_point.x,
            y=mid_point.y,
            primal_edge_node_a=start_nd_key,
            primal_edge_node_b=end_nd_key,
            primal_edge_idx=edge_idx,
        )
        # add and set live property if present in parent graph
        set_live(start_nd_key, end_nd_key, dual_node_key)
    # add dual edges
    logger.info("Preparing dual edges (splitting and welding geoms)")
    for start_nd_key, end_nd_key, edge_idx in tqdm(
        nx_multigraph.edges(data=False, keys=True), disable=config.QUIET_MODE
    ):
        hub_node_dual = prepare_dual_node_key(start_nd_key, end_nd_key, edge_idx)
        # get the first and second half geoms
        s_half_geom, e_half_geom = get_half_geoms(nx_multigraph, start_nd_key, end_nd_key, edge_idx)
        # process either side
        for n_side, m_side, half_geom in zip(
            [start_nd_key, end_nd_key], [end_nd_key, start_nd_key], [s_half_geom, e_half_geom]
        ):
            # add the spoke edges on the dual
            nb_nd_key: NodeKey
            for nb_nd_key in nx.neighbors(nx_multigraph, n_side):
                # don't follow neighbour back to current edge combo
                if nb_nd_key == m_side:
                    continue
                # add the neighbouring primal edges as dual nodes
                for edge_idx in nx_multigraph[n_side][nb_nd_key]:
                    spoke_node_dual = prepare_dual_node_key(n_side, nb_nd_key, edge_idx)
                    # skip if the edge has already been processed from another direction
                    if g_dual.has_edge(hub_node_dual, spoke_node_dual):
                        continue
                    # get the near and far half geoms
                    spoke_half_geom, _discard_geom = get_half_geoms(nx_multigraph, n_side, nb_nd_key, edge_idx)
                    # weld the lines
                    merged_line: geometry.LineString = ops.linemerge([half_geom, spoke_half_geom])
                    if merged_line.geom_type != "LineString":
                        raise TypeError(
                            f'Found {merged_line.geom_type} instead of "LineString" for new geom {merged_line.wkt}. '
                            f"Check that the LineStrings for {start_nd_key}-{end_nd_key} & {n_side}-{nb_nd_key} touch."
                        )
                    # add the dual edge
                    g_dual.add_edge(
                        hub_node_dual,
                        spoke_node_dual,
                        primal_node_id=n_side,
                        geom=merged_line,
                    )

    return g_dual


def network_structure_from_nx(
    nx_multigraph: MultiGraph,
    crs: Union[str, int],
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, rustalgos.NetworkStructure]:
    """
    Transpose a `networkX` `MultiGraph` into a `GeoDataFrame` and `NetworkStructure` for use by `cityseer`.

    Calculates length and angle attributes, as well as in and out bearings, and stores this information in the returned
    data maps.

    Parameters
    ----------
    nx_multigraph: MultiGraph
        A `networkX` `MultiGraph` in a projected coordinate system, containing `x` and `y` node attributes, and `geom`
        edge attributes containing `LineString` geoms.
    crs: str | int
        CRS for initialising the returned structures. This is used for initialising the GeoPandas
        [`GeoDataFrame`](https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoDataFrame.html#geopandas-geodataframe).  # pylint: disable=line-too-long

    Returns
    -------
    nodes_gdf: GeoDataFrame
        A `GeoDataFrame` with `live` and `geometry` attributes. The original `networkX` graph's node keys will be used
        for the `GeoDataFrame` index.
    edges_gdf: GeoDataFrame
        A `GeoDataFrame` with `ns_edge_idx`, `start_ns_node_idx`, `end_ns_node_idx`, `edge_idx`, `nx_start_node_key`,
        `nx_end_node_key`, `length`, `angle_sum`, `imp_factor`, `in_bearing`, `out_bearing`, `total_bearing`, `geom`
        attributes.
    network_structure: rustalgos.NetworkStructure
        A [`rustalgos.NetworkStructure`](/rustalgos#networkstructure) instance.

    """
    if not isinstance(nx_multigraph, nx.MultiGraph):
        raise TypeError("This method requires an undirected networkX MultiGraph.")
    logger.info("Preparing node and edge arrays from networkX graph.")
    g_multi_copy: MultiGraph = nx_multigraph.copy()
    # prepare the network structure
    network_structure = rustalgos.NetworkStructure()
    # generate the network information
    agg_node_data: dict[str, tuple[Any, ...]] = {}
    agg_node_dual_data: dict[str, tuple[Any]] = {}
    agg_edge_data: dict[str, tuple[Any, ...]] = {}
    agg_edge_dual_data: list[str] = []
    node_data: NodeData
    # set nodes
    for node_key, node_data in tqdm(g_multi_copy.nodes(data=True), disable=config.QUIET_MODE):
        # node_key must be string
        if not isinstance(node_key, str):
            raise TypeError(f"Node key must be of type string but encountered {type(node_key)}")
        if "x" not in node_data:
            raise KeyError(f'Encountered node missing "x" coordinate attribute at node {node_key}.')
        node_x: float = node_data["x"]
        if "y" not in node_data:
            raise KeyError(f'Encountered node missing "y" coordinate attribute at node {node_key}.')
        node_y: float = node_data["y"]
        is_live: bool = True
        if "live" in node_data:
            is_live = bool(node_data["live"])
        # set node
        ns_node_idx = network_structure.add_node(node_key, node_x, node_y, is_live)
        agg_node_data[node_key] = (ns_node_idx, node_x, node_y, is_live, geometry.Point(node_x, node_y))
        if "is_dual" in g_multi_copy.graph and g_multi_copy.graph["is_dual"]:
            agg_node_dual_data[node_key] = (
                node_data["primal_edge_node_a"],
                node_data["primal_edge_node_b"],
                node_data["primal_edge_idx"],
            )
    # set edges
    for start_node_key in tqdm(g_multi_copy.nodes(), disable=config.QUIET_MODE):
        # build edges
        start_ns_node_idx, start_node_x, start_node_y, _, _ = agg_node_data[start_node_key]
        end_node_key: int
        for end_node_key in g_multi_copy.neighbors(start_node_key):
            end_ns_node_idx, _, _, _, _ = agg_node_data[end_node_key]
            # add the new edge index to the node's out edges
            nx_edge_data: EdgeData
            for edge_idx, nx_edge_data in g_multi_copy[start_node_key][end_node_key].items():
                if not "geom" in nx_edge_data:
                    raise KeyError(
                        f"No edge geom found for edge {start_node_key}-{end_node_key}: Please add an edge 'geom' "
                        "attribute consisting of a shapely LineString. Simple (straight) geometries can be inferred "
                        "automatically through the nx_simple_geoms() method."
                    )
                line_geom = nx_edge_data["geom"]
                if line_geom.geom_type != "LineString":
                    raise TypeError(
                        f"Expecting LineString geometry but found {line_geom.geom_type} geom for edge "
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
                line_geom_coords = align_linestring_coords(line_geom.coords, (start_node_x, start_node_y))
                # iterate the coordinates and calculate the angular change
                angle_sum = _measure_cumulative_angle(line_geom_coords)
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
                xy_1: npt.NDArray[np.float_] = np.array(line_geom_coords[0])
                xy_2: npt.NDArray[np.float_] = np.array(line_geom_coords[1])
                in_bearing: float = _measure_bearing(xy_1, xy_2)
                # out bearing
                xy_3: npt.NDArray[np.float_] = np.array(line_geom_coords[-2])
                xy_4: npt.NDArray[np.float_] = np.array(line_geom_coords[-1])
                out_bearing: float = _measure_bearing(xy_3, xy_4)
                total_bearing = _measure_bearing(xy_1, xy_4)
                # set edge
                ns_edge_idx = network_structure.add_edge(
                    start_ns_node_idx,
                    end_ns_node_idx,
                    edge_idx,
                    start_node_key,
                    end_node_key,
                    line_len,
                    angle_sum,
                    imp_factor,
                    in_bearing,
                    out_bearing,
                )
                # add to edge data
                agg_edge_data[f"{start_node_key}-{end_node_key}"] = (
                    ns_edge_idx,
                    start_ns_node_idx,
                    end_ns_node_idx,
                    edge_idx,
                    start_node_key,
                    end_node_key,
                    line_len,
                    angle_sum,
                    imp_factor,
                    in_bearing,
                    out_bearing,
                    total_bearing,
                    line_geom,
                )
                if "is_dual" in g_multi_copy.graph and g_multi_copy.graph["is_dual"]:
                    agg_edge_dual_data.append(nx_edge_data["primal_node_id"])
    # create geopandas for node keys and data state
    nodes_gdf = gpd.GeoDataFrame.from_dict(
        agg_node_data,
        orient="index",
        columns=["ns_node_idx", "x", "y", "live", "geom"],
        geometry="geom",
        crs=crs,
    )
    edges_gdf = gpd.GeoDataFrame.from_dict(
        agg_edge_data,
        orient="index",
        columns=[
            "ns_edge_idx",
            "start_ns_node_idx",
            "end_ns_node_idx",
            "edge_idx",
            "nx_start_node_key",
            "nx_end_node_key",
            "length",
            "angle_sum",
            "imp_factor",
            "in_bearing",
            "out_bearing",
            "total_bearing",
            "geom",
        ],
        geometry="geom",
        crs=crs,
    )
    if "is_dual" in g_multi_copy.graph and g_multi_copy.graph["is_dual"]:
        nodes_dual_gdf = pd.DataFrame.from_dict(
            agg_node_dual_data,
            orient="index",
            columns=[
                "primal_edge_node_a",
                "primal_edge_node_b",
                "primal_edge_idx",
            ],
        )
        nodes_gdf = nodes_gdf.join(nodes_dual_gdf)
        edges_gdf["primal_node_id"] = agg_edge_dual_data
    return nodes_gdf, edges_gdf, network_structure


def blend_metrics(
    nodes_gdf: gpd.GeoDataFrame,
    edges_gdf: gpd.GeoDataFrame,
    method: str,
) -> MultiGraph:
    """
    Blends metrics from a nodes GeoDataFrame into an edges GeoDataFrame.

    This is useful for situations where it is preferable to visualise the computed metrics as LineStrings instead of
    points. The line will be assigned the value from the adjacent two nodes based on the selected "min", "max", or "avg"
    method.

    Parameters
    ----------
    nodes_gdf: GeoDataFrame
        A nodes `GeoDataFrame` as derived from [`network_structure_from_nx`](#network-structure-from-nx).
    edges_gdf: GeoDataFrame
        An edges `GeoDataFrame` as derived from [`network_structure_from_nx`](#network-structure-from-nx).
    method: str
        The method used for determining the line value from the adjacent points. Must be one of "min", "max", or "avg".

    Returns
    -------
    merged_gdf: GeoDataFrame
        An edges `GeoDataFrame` created by merging the node metrics from the provided nodes `GeoDataFrame` into the
        provided edges `GeoDataFrame`.

    """
    if method not in ["min", "max", "avg"]:
        raise ValueError('Method should be one of "min", "max", or "avg"')
    merged_edges_gdf = edges_gdf.copy()
    for node_column in nodes_gdf.columns:
        if not node_column.startswith("cc_metric"):
            continue
        # suffix is only applied for overlapping column names
        merged_edges_gdf = pd.merge(
            merged_edges_gdf, nodes_gdf[[node_column]], left_on="nx_start_node_key", right_index=True
        )
        merged_edges_gdf = pd.merge(
            merged_edges_gdf,
            nodes_gdf[[node_column]],
            left_on="nx_end_node_key",
            right_index=True,
            suffixes=("", "_end_nd"),
        )
        # merge
        end_nd_col = f"{node_column}_end_nd"
        if method == "avg":
            merged_edges_gdf[node_column] = (merged_edges_gdf[node_column] + merged_edges_gdf[end_nd_col]) / 2
        elif method == "min":
            merged_edges_gdf[node_column] = merged_edges_gdf[[node_column, end_nd_col]].min(axis=1)
        else:
            merged_edges_gdf[node_column] = merged_edges_gdf[[node_column, end_nd_col]].max(axis=1)
        # cleanup
        merged_edges_gdf = merged_edges_gdf.drop(columns=[end_nd_col])

    return merged_edges_gdf


def nx_from_geopandas(
    nodes_gdf: gpd.GeoDataFrame,
    edges_gdf: gpd.GeoDataFrame,
) -> MultiGraph:
    """
    Write nodes and edges `GeoDataFrames` to a `networkX` `MultiGraph`.

    Parameters
    ----------
    nodes_gdf: GeoDataFrame
        A `GeoDataFrame` with `live` and Point `geometry` attributes. The index will be used for the returned `networkX`
        graph's node keys.
    edges_gdf: GeoDataFrame
        An edges `GeoDataFrame` as derived from [`network_structure_from_nx`](#network-structure-from-nx).

    Returns
    -------
    nx_multigraph: MultiGraph
        A `networkX` graph with geometries and attributes as copied from the input `GeoDataFrames`.

    """
    logger.info("Populating node and edge map data to a networkX graph.")
    g_multi_copy: MultiGraph = nx.MultiGraph()
    g_multi_copy.add_nodes_from(nodes_gdf.index.values.tolist())
    # after above so that errors caught first
    logger.info("Unpacking node data.")
    for nd_key, nd_data in tqdm(nodes_gdf.iterrows(), disable=config.QUIET_MODE):
        g_multi_copy.nodes[nd_key]["x"] = nd_data.x
        g_multi_copy.nodes[nd_key]["y"] = nd_data.y
        g_multi_copy.nodes[nd_key]["live"] = nd_data.live
    logger.info("Unpacking edge data.")
    for _, row_data in tqdm(edges_gdf.iterrows(), disable=config.QUIET_MODE):
        g_multi_copy.add_edge(
            row_data.nx_start_node_key,
            row_data.nx_end_node_key,
            row_data.edge_idx,
            length=row_data.length,
            angle_sum=row_data.angle_sum,
            imp_factor=row_data.imp_factor,
            in_bearing=row_data.in_bearing,
            out_bearing=row_data.out_bearing,
            total_bearing=row_data.total_bearing,
            geom=row_data.geom,
        )
    # unpack any metrics written to the nodes
    metrics_column_labels: list[str] = [c for c in nodes_gdf.columns if c.startswith("cc_metric")]
    if metrics_column_labels:
        logger.info("Unpacking metrics to nodes.")
        for metrics_column_label in metrics_column_labels:
            for nd_key, node_row in tqdm(nodes_gdf.iterrows(), disable=config.QUIET_MODE):
                g_multi_copy.nodes[nd_key][metrics_column_label] = node_row[metrics_column_label]

    return g_multi_copy
