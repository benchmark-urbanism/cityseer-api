"""
Convenience functions for the preparation and conversion of `networkX` graphs to and from `cityseer` data structures.

Note that the `cityseer` network data structures can be created and manipulated directly, if so desired.
"""
# workaround until networkx adopts types
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportGeneralTypeIssues=false
from __future__ import annotations

import logging
from typing import Any, Union

import geopandas as gpd
import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd
from shapely import coords, geometry, strtree
from shapely.errors import GeometryTypeError
from shapely.geometry import LineString
from tqdm import tqdm

from cityseer import config

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


def measure_bearing(xy_1: npt.NDArray[np.float_], xy_2: npt.NDArray[np.float_]) -> float:
    """Measures the angular bearing between two coordinate pairs."""
    y_1, x_1 = xy_1[::-1]
    y_2, x_2 = xy_2[::-1]
    return np.rad2deg(np.arctan2(y_2 - y_1, x_2 - x_1))


def measure_coords_angle(
    coords_1: npt.NDArray[np.float_], coords_2: npt.NDArray[np.float_], coords_3: npt.NDArray[np.float_]
) -> float:
    """Measures angle between three coordinate pairs."""
    # arctan2 is y / x order
    a_1: float = measure_bearing(coords_2, coords_1)
    a_2: float = measure_bearing(coords_3, coords_2)
    angle = np.abs((a_2 - a_1 + 180) % 360 - 180)
    return angle


def _measure_linestring_angle(linestring_coords: ListCoordsType, idx_a: int, idx_b: int, idx_c: int) -> float:
    """Measures angle between two segment bearings per indices."""
    coords_1: npt.NDArray[np.float_] = np.array(linestring_coords[idx_a])[:2]
    coords_2: npt.NDArray[np.float_] = np.array(linestring_coords[idx_b])[:2]
    coords_3: npt.NDArray[np.float_] = np.array(linestring_coords[idx_c])[:2]
    # arctan2 is y / x order
    a_1: float = measure_bearing(coords_2, coords_1)
    a_2: float = measure_bearing(coords_3, coords_2)
    angle = np.abs((a_2 - a_1 + 180) % 360 - 180)
    # alternative
    # A: npt.NDArray[np.float_] = coords_2 - coords_1
    # B: npt.NDArray[np.float_] = coords_3 - coords_2
    # alt_angle = np.abs(np.degrees(np.math.atan2(np.linalg.det([A, B]), np.dot(A, B))))

    return angle


def measure_cumulative_angle(linestring_coords: ListCoordsType) -> float:
    """Measures the cumulative angle along a LineString geom's coords."""
    angle_sum: float = 0
    for c_idx in range(len(linestring_coords) - 2):
        angle_sum += _measure_linestring_angle(linestring_coords, c_idx, c_idx + 1, c_idx + 2)

    return angle_sum


def substring(  # type: ignore # pylint: disable=too-many-return-statements
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


def snap_linestring_endpoints(
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


def weld_linestring_coords(
    linestring_coords_a: AnyCoordsType,
    linestring_coords_b: AnyCoordsType,
    force_xy: CoordsType | None = None,
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


class EdgeInfo:
    """Encapsulates EdgeInfo logic."""

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


def add_node(
    nx_multigraph: MultiGraph,
    nodes_names: list[NodeKey],
    x: float,
    y: float,
    live: bool | None = None,
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


def create_nodes_strtree(nx_multigraph: MultiGraph) -> tuple[strtree.STRtree, list[dict[str, Any]]]:
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


def create_edges_strtree(nx_multigraph: MultiGraph) -> tuple[strtree.STRtree, list[dict[str, Any]]]:
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
