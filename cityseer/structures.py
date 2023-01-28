"""
The `structures` module defines data structures used by the lower-level `cityseer` API.

The data structures defined in this modules are created and managed automatically by the user-facing API. It is
therefore not necessary to create these structures directly unless interaction the lower-level API is intentional.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import numpy.typing as npt
from numba.core import types
from numba.experimental import jitclass  # type: ignore
from numba.typed import Dict, List

node_map_spec: list[tuple[str, Any]] = [
    ("nodes_n", types.int_),
    ("node_idx", types.int_),
    ("xs", types.float32[:]),
    ("ys", types.float32[:]),
    ("live", types.bool_[:]),
]


@jitclass(node_map_spec)
class NodeMap:
    """
    `NodeMap` structure representing the `x`, `y`, and `live` information for the network.

    Each attribute contains a `numpy` array with indices corresponding to the graph's node indices.

    It is not necessary to invoke this class directly if using a `NetworkStructure` class, which will generate the
    `NodeMap` implicitly.
    """

    xs: npt.NDArray[np.float32]
    """`x` coordinates."""
    ys: npt.NDArray[np.float32]
    """`y` coordinates."""
    live: npt.NDArray[np.bool_]
    """`live` status indicators."""

    # Alternative to length dunder - which is not yet supported by jitclass.
    @property
    def count(self):
        """
        The number of nodes represented by the instanced `NodeMap`.
        """
        return len(self.xs)

    def __init__(self, nodes_n: int):
        """
        Instance a `NodeMap`.

        Parameters
        ----------
        nodes_n: int
            The number of nodes to be contained by this `NodeMap` instance.

        """
        self.xs = np.full(nodes_n, np.nan, dtype=np.float32)
        self.ys = np.full(nodes_n, np.nan, dtype=np.float32)
        self.live = np.full(nodes_n, False, dtype=np.bool_)

    def x_y(self, node_idx: int) -> npt.NDArray[np.float32]:
        """
        Return the `x` and `y` coordinates for a given node index.

        Parameters
        ----------
        node_idx: int
            The node index for which to return `x` and `y` coordinates.

        Returns
        -------
        x: float
            `x` coordinate.
        y: float
            `y` coordinate.

        """
        return np.array([self.xs[node_idx], self.ys[node_idx]], dtype=np.float32)

    def validate(self):
        """Validate this `NodeMap` instance."""
        if self.count == 0:
            raise ValueError("Zero length NodeMap")
        if len(self.ys) != self.count or len(self.live) != self.count:
            raise ValueError("X, Y and 'live' arrays are not the same length")
        if not np.all(np.isfinite(self.xs)):
            raise ValueError("Non finite x coordinate encountered.")
        if not np.all(np.isfinite(self.ys)):
            raise ValueError("Non finite y coordinate encountered.")
        if np.all(~self.live):
            raise ValueError("NodeMap has no live nodes.")


edge_map_spec: list[tuple[str, Any]] = [
    ("edges_n", types.int_),
    ("start", types.int_[:]),
    ("end", types.int_[:]),
    ("length", types.float32[:]),
    ("angle_sum", types.float32[:]),
    ("imp_factor", types.float32[:]),
    ("in_bearing", types.float32[:]),
    ("out_bearing", types.float32[:]),
]


@jitclass(edge_map_spec)
class EdgeMap:
    """
    `EdgeMap` structure containing edge (segment) information for the network.

    Each attribute contains a `numpy` array with indices corresponding to the graph's edges.

    It is not necessary to invoke this class directly if using a `NetworkStructure` class, which will generate the
    `EdgeMap` implicitly.
    """

    start: npt.NDArray[np.int_]
    """The start node index."""
    end: npt.NDArray[np.int_]
    """The end node index."""
    length: npt.NDArray[np.float32]
    """Edge lengths."""
    angle_sum: npt.NDArray[np.float32]
    """The sum of angular change for a given edge."""
    imp_factor: npt.NDArray[np.float32]
    """Edge impedance factor."""
    in_bearing: npt.NDArray[np.float32]
    """Angular inwards bearing."""
    out_bearing: npt.NDArray[np.float32]
    """Agnular outwards bearing."""

    # Alternative to length dunder - which is not yet supported by jitclass.
    @property
    def count(self):
        """Number of edges represented by the instanced `EdgeMap`."""
        return len(self.start)

    def __init__(self, edges_n: int):
        """
        Create an `EdgeMap` instance.

        Parameters
        ----------
        edges_n: int
            The number of edges to be contained by this `EdgeMap` instance.

        """
        self.start = np.full(edges_n, -1, dtype=np.int_)
        self.end = np.full(edges_n, -1, dtype=np.int_)
        self.length = np.full(edges_n, np.nan, dtype=np.float32)
        self.angle_sum = np.full(edges_n, np.nan, dtype=np.float32)
        self.imp_factor = np.full(edges_n, np.nan, dtype=np.float32)
        self.in_bearing = np.full(edges_n, np.nan, dtype=np.float32)
        self.out_bearing = np.full(edges_n, np.nan, dtype=np.float32)

    def validate(self):
        """Validate this Edgemap instance."""
        if self.count == 0:
            raise ValueError("Zero length NodeMap")
        if (
            len(self.end) != self.count  # pylint: disable=too-many-boolean-expressions
            or len(self.length) != self.count
            or len(self.angle_sum) != self.count
            or len(self.imp_factor) != self.count
            or len(self.in_bearing) != self.count
            or len(self.out_bearing) != self.count
        ):
            raise ValueError("Arrays are not of the same length.")
        if not np.all(np.isfinite(self.start)) or not np.all(self.start >= 0):
            raise ValueError("Missing or invalid start node index encountered.")
        if not np.all(np.isfinite(self.end)) or not np.all(self.end >= 0):
            raise ValueError("Missing or invalid end node index encountered.")
        if not np.all(np.isfinite(self.length)) or not np.all(self.length >= 0):
            raise ValueError("Invalid edge length encountered. Should be finite number greater than or equal to zero.")
        if not np.all(np.isfinite(self.angle_sum)) or not np.all(self.angle_sum >= 0):
            raise ValueError(
                "Invalid edge angle sum encountered. Should be finite number greater than or equal to zero."
            )
        if not np.all(np.isfinite(self.imp_factor)) or not np.all(self.imp_factor >= 0):
            raise ValueError(
                "Invalid impedance factor encountered. Should be finite number greater than or equal to zero."
            )


network_spec: list[tuple[str, Any]] = [
    ("nodes_n", types.int_),
    ("edges_n", types.int_),
    ("node_idx", types.int_),
    ("node_x", types.float32),
    ("node_y", types.float32),
    ("node_live", types.bool_),
    ("node_edge_map", types.DictType(types.int64, types.ListType(types.int64))),
    ("next_edge_idx", types.int_),
]


@jitclass(network_spec)
class NetworkStructure:
    """
    `NetworkStructure` instance consisting of `nodes`, `edges` and `node_edge_map` attributes.

    Each of these attributes will be created automatically when instancing this class.
    """

    nodes: NodeMap
    """A `NodeMap` instance. See [`NodeMap`](#nodemap)."""
    edges: EdgeMap
    """A `EdgeMap` instance. See [`EdgeMap`](#edgemap)."""
    node_edge_map: dict[int, list[int]]
    """A `node_edge_map` describing the relationship from each node to all directly connected edges."""
    _next_edge_idx: int

    def __init__(self, nodes_n: int, edges_n: int):
        """
        Instances a `NetworkStructure`.

        Parameters
        ----------
        nodes_n: int
            The number of nodes this `NetworkStructure` will contain.
        edges_n: int
            The number of edges this `NetworkStructure` will contain.

        """
        self.nodes = NodeMap(nodes_n)
        self.edges = EdgeMap(edges_n)
        # NOTE
        # List.empty_list(types.int64) only works in jitclass mode
        # i.e. testing in pure python use types.ListType(types.int64) otherwise hash issue arises
        self.node_edge_map = Dict.empty(types.int64, List.empty_list(types.int64))
        self._next_edge_idx = 0

    def set_node(self, node_idx: int, node_x: float, node_y: float, node_live: bool = True):
        """
        Add a node to the `NetworkStructure`.

        Parameters
        ----------
        node_idx: int
            The index at which to add the node.
        node_x: float
            The `x` coordinate for the added node.
        node_y: float
            The `y` coordinate for the added node.
        node_live: bool
            The `live` node attribute identifies nodes falling within the areal boundary of interest as opposed to those
            that fall within the surrounding buffered area. See the [edge-rolloff](/guide#edge-rolloff) section in the
            guide.

        """
        self.nodes.xs[node_idx] = node_x
        self.nodes.ys[node_idx] = node_y
        self.nodes.live[node_idx] = node_live
        self.node_edge_map[node_idx] = List.empty_list(types.int64)  # type: ignore

    def set_edge(
        self,
        start_node_idx: int,
        end_node_idx: int,
        length: float,
        angle_sum: float,
        imp_factor: float,
        in_bearing: float,
        out_bearing: float,
    ):
        """
        Add an edge to the `NetworkStructure`.

        Edges are directed, meaning that each bidirectionaly street is represented twice: once in each direction;
        start/end nodes and in/out bearings will differ accordingly.

        Parameters
        ----------
        start_node_idx: int
            Index for the starting node for the added edge.
        end_node_idx: int
            Index for the ending node for the added edge.
        length: float
            The `length` edge attribute should always correspond to the edge lengths in metres. This is used when
            calculating the distances traversed by the shortest-path algorithm so that the respective $d_{max}$ maximum
            distance thresholds can be enforced: these distance thresholds are based on the actual network-paths
            traversed by the algorithm as opposed to crow-flies distances.
        angle_sum: float
            The `angle_sum` edge bearing should correspond to the total angular change along the length of
            the segment. This is used when calculating angular impedances for simplest-path measures. The
            `in_bearing` and `out_bearing` attributes respectively represent the starting and
            ending bearing of the segment. This is also used when calculating simplest-path measures when the algorithm
            steps from one edge to another.
        imp_factor: float
            The `imp_factor` edge attribute represents an impedance multiplier for increasing or diminishing
            the impedance of an edge. This is ordinarily set to 1, therefor not impacting calculations. By setting
            this to greater or less than 1, the edge will have a correspondingly higher or lower impedance. This can
            be used to take considerations such as street gradients into account, but should be used with caution.
        in_bearing: float
            The edge's inwards angular bearing.
        out_bearing: float
            The edge's outwards angular bearing.

        """
        self.node_edge_map[start_node_idx].append(self._next_edge_idx)
        self.edges.start[self._next_edge_idx] = start_node_idx
        self.edges.end[self._next_edge_idx] = end_node_idx
        self.edges.length[self._next_edge_idx] = length
        self.edges.angle_sum[self._next_edge_idx] = angle_sum
        self.edges.imp_factor[self._next_edge_idx] = imp_factor
        self.edges.in_bearing[self._next_edge_idx] = in_bearing
        self.edges.out_bearing[self._next_edge_idx] = out_bearing
        self._next_edge_idx += 1

    def validate(self):
        """Validate Network Structure."""
        self.nodes.validate()
        self.edges.validate()
        # check sequential and reciprocal node to edge map indices
        edge_counts: npt.NDArray[np.float_] = np.full(self.edges.count, 0)
        for n_idx in range(self.nodes.count):
            # zip through all edges for current node
            for edge_idx in self.node_edge_map[n_idx]:
                # check that the start node matches the current node index
                start_nd_idx: int = self.edges.start[edge_idx]
                if start_nd_idx != n_idx:
                    raise ValueError("Start node does not match current node index")
                # check that each edge has a matching pair in the opposite direction
                end_nd_idx: int = self.edges.end[edge_idx]
                paired = False
                for return_edge_idx in self.node_edge_map[end_nd_idx]:
                    if self.edges.end[return_edge_idx] == n_idx:
                        paired = True
                        break
                if not paired:
                    raise ValueError("Missing matching edge pair in opposite direction.")
                # add to the counter
                edge_counts[edge_idx] += 1
        if not np.all(edge_counts == 1):
            raise ValueError("Mismatched node and edge maps encountered.")


data_map_spec: list[tuple[str, Any]] = [
    ("xs", types.float32[:]),
    ("ys", types.float32[:]),
    ("nearest_assign", types.int_[:]),
    ("next_nearest_assign", types.int_[:]),
    ("data_id", types.int_[:]),
]


@jitclass(data_map_spec)
class DataMap:
    """
    `NodeMap` instance representing the `x`, `y` data coordinates and the nearest adjacent network node indices.

    Each attribute contains a `numpy` array with indices corresponding to the graph's node indices.
    """

    xs: npt.NDArray[np.float32]
    """`x` coordinates."""
    ys: npt.NDArray[np.float32]
    """`y` coordinates."""
    nearest_assign: npt.NDArray[np.int_]
    """Nearest assigned network node index."""
    next_nearest_assign: npt.NDArray[np.int_]
    """Next-nearest assigned network node index."""
    data_id: npt.NDArray[np.int_]
    """Optional data key for deduplicating aggregations"""

    # Alternative to length dunder - which is not yet supported by jitclass.
    @property
    def count(self):
        """
        The number of data points represented by the instanced `DataMap`.
        """
        return len(self.xs)

    def __init__(self, data_n: int):
        """
        Instance a `DataMap`.

        Parameters
        ----------
        data_n: int
            The number of data points to be contained by this `DataMap` instance.

        """
        self.xs = np.full(data_n, np.nan, dtype=np.float32)
        self.ys = np.full(data_n, np.nan, dtype=np.float32)
        self.nearest_assign = np.full(data_n, -1, dtype=np.int_)
        self.next_nearest_assign = np.full(data_n, -1, dtype=np.int_)
        self.data_id = np.full(data_n, -1, dtype=np.int_)

    def set_data_point(self, data_idx: int, data_x: np.float32, data_y: np.float32, data_id: Optional[np.int_] = None):
        """
        Add a data point.

        Parameters
        ----------
        data_idx: int
            The index for the added node.
        data_x: np.float32
            The x coordinate for the added node.
        data_y: np.float32
            The y coordinate for the added node.
        data_id: np.int_ | None
            An optional key for each datapoint. Used for deduplication.

        """
        if not isinstance(data_x, (float, int)):
            raise ValueError(f'Expecting float or int for "data_x" but encountered {type(data_x)}')
        if not isinstance(data_y, (float, int)):
            raise ValueError(f'Expecting float or int for "data_y" but encountered {type(data_y)}')
        if not isinstance(data_id, (int)):
            raise ValueError(f'Expecting int for "data_id" but encountered {type(data_id)}')
        self.xs[data_idx] = data_x
        self.ys[data_idx] = data_y
        if data_id is not None:
            self.data_id[data_idx] = data_id

    def x_y(self, data_idx: int) -> npt.NDArray[np.float32]:
        """
        Return the `x` and `y` coordinates for a given data point index.

        Parameters
        ----------
        data_idx: int
            The data point index for which to return `x` and `y` coordinates.

        Returns
        -------
        x: float
            `x` coordinate.
        y: float
            `y` coordinate.

        """
        return np.array([self.xs[data_idx], self.ys[data_idx]], dtype=np.float32)

    def validate(self, check_assigned: bool = False):
        """
        Validate this `DataMap` instance.

        Parameters
        ----------
        check_assigned: bool
            Whether to check if data points have been assigned to a network.

        """
        if self.count == 0:
            raise ValueError("Zero length DataMap")
        if (
            len(self.xs) != self.count
            or len(self.ys) != self.count
            or len(self.nearest_assign) != self.count
            or len(self.nearest_assign) != self.count
        ):
            raise ValueError("Arrays are not of the same length.")
        if np.any(np.isnan(self.xs)) or np.any(self.xs < 0):
            raise ValueError("X coordinates must be positive finite values.")
        if np.any(np.isnan(self.ys)) or np.any(self.ys < 0):
            raise ValueError("Y coordinates must be positive finite values.")
        if check_assigned:
            # check that data map has been assigned - only if explicitly requested
            if np.all(self.nearest_assign == -1):
                raise ValueError(
                    "Data map has not been assigned to a network. (Else data-points were not assignable "
                    "for the given max_dist parameter passed to assign_to_network."
                )
        # either all or no data keys should be set
        if not (np.all(self.data_id == -1) or np.all(self.data_id != -1)):
            raise ValueError("Either all or none of the data keys should be set.")
