from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict, Union

import numpy as np
import numpy.typing as npt
from numba.core import types
from numba.experimental import jitclass  # type: ignore
from numba.typed import Dict, List

qsType = Union[  # pylint: disable=invalid-name
    int,
    float,
    Union[list[int], list[float]],
    Union[tuple[int], tuple[float]],
    Union[npt.NDArray[np.int_], npt.NDArray[np.float32]],
    None,
]


tree_map_spec: list[tuple[str, Any]] = [
    ("visited_nodes", types.bool_[:]),
    ("preds", types.int_[:]),
    ("short_dist", types.float32[:]),
    ("simpl_dist", types.float32[:]),
    ("cycles", types.float32[:]),
    ("origin_seg", types.int_[:]),
    ("last_seg", types.int_[:]),
    ("out_bearings", types.float32[:]),
    ("visited_edges", types.bool_[:]),
]


@jitclass(tree_map_spec)
class TreeMap:
    """Tree Map for shortest-path algo."""

    def __init__(self, nodes_n: int, edges_n: int):
        """Instance TreeMap."""
        self.visited_nodes: npt.NDArray[np.bool_] = np.full(nodes_n, False, dtype=np.bool_)
        self.preds: npt.NDArray[np.int_] = np.full(nodes_n, -1, dtype=np.int_)
        self.short_dist: npt.NDArray[np.float32] = np.full(nodes_n, np.inf, dtype=np.float32)
        self.simpl_dist: npt.NDArray[np.float32] = np.full(nodes_n, np.inf, dtype=np.float32)
        self.cycles: npt.NDArray[np.float32] = np.full(nodes_n, 0.0, dtype=np.float32)
        self.origin_seg: npt.NDArray[np.int_] = np.full(nodes_n, -1, dtype=np.int_)
        self.last_seg: npt.NDArray[np.int_] = np.full(nodes_n, -1, dtype=np.int_)
        self.out_bearings: npt.NDArray[np.float32] = np.full(nodes_n, np.nan, dtype=np.float32)
        self.visited_edges: npt.NDArray[np.bool_] = np.full(edges_n, False, dtype=np.bool_)


node_map_spec: list[tuple[str, Any]] = [
    ("nodes_n", types.int_),
    ("node_idx", types.int_),
    ("xs", types.float32[:]),
    ("ys", types.float32[:]),
    ("live", types.bool_[:]),
]


@jitclass(node_map_spec)
class NodeMap:
    """Node Map for network."""

    xs: npt.NDArray[np.float32]
    ys: npt.NDArray[np.float32]
    live: npt.NDArray[np.bool_]

    @property
    def count(self):
        """Alternative to length dunder - which is not yet supported by jitclass."""
        return len(self.xs)

    def __init__(self, nodes_n: int):
        """Instance NodeMap."""
        self.xs = np.full(nodes_n, np.nan, dtype=np.float32)
        self.ys = np.full(nodes_n, np.nan, dtype=np.float32)
        self.live = np.full(nodes_n, False, dtype=np.bool_)

    def x_y(self, node_idx: int) -> npt.NDArray[np.float32]:
        """XY coordinates for given node index."""
        return np.array([self.xs[node_idx], self.ys[node_idx]], dtype=np.float32)  # type: ignore

    def validate(self):
        """Validate NodeMap."""
        if self.count == 0:
            raise ValueError("Zero length NodeMap")
        if len(self.ys) != self.count or len(self.live) != self.count:
            raise ValueError("X, Y and 'live' arrays are not the same length")
        if not np.all(np.isfinite(self.xs)) or not np.all(self.xs >= 0):  # type: ignore
            raise ValueError("Missing or invalid start x data encountered.")
        if not np.all(np.isfinite(self.ys)) or not np.all(self.ys >= 0):  # type: ignore
            raise ValueError("Missing or invalid start y data encountered.")
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
    """Edge Map for network."""

    start: npt.NDArray[np.int_]
    end: npt.NDArray[np.int_]
    length: npt.NDArray[np.float32]
    angle_sum: npt.NDArray[np.float32]
    imp_factor: npt.NDArray[np.float32]
    in_bearing: npt.NDArray[np.float32]
    out_bearing: npt.NDArray[np.float32]

    @property
    def count(self):
        """Alternative to length dunder - which is not yet supported by jitclass."""
        return len(self.start)

    def __init__(self, edges_n: int):
        """Instance EdgeMap."""
        self.start = np.full(edges_n, -1, dtype=np.int_)
        self.end = np.full(edges_n, -1, dtype=np.int_)
        self.length = np.full(edges_n, np.nan, dtype=np.float32)
        self.angle_sum = np.full(edges_n, np.nan, dtype=np.float32)
        self.imp_factor = np.full(edges_n, np.nan, dtype=np.float32)
        self.in_bearing = np.full(edges_n, np.nan, dtype=np.float32)
        self.out_bearing = np.full(edges_n, np.nan, dtype=np.float32)

    def validate(self):
        """Validate Edgemap."""
        if self.count == 0:
            raise ValueError("Zero length NodeMap")
        if (
            len(self.end) != self.count
            or len(self.length) != self.count
            or len(self.angle_sum) != self.count
            or len(self.imp_factor) != self.count
            or len(self.in_bearing) != self.count
            or len(self.out_bearing) != self.count
        ):
            raise ValueError("Arrays are not of the same length.")
        if not np.all(np.isfinite(self.start)) or not np.all(self.start >= 0):  # type: ignore
            raise ValueError("Missing or invalid start node index encountered.")
        if not np.all(np.isfinite(self.end)) or not np.all(self.end >= 0):  # type: ignore
            raise ValueError("Missing or invalid end node index encountered.")
        if not np.all(np.isfinite(self.length)) or not np.all(self.length >= 0):  # type: ignore
            raise ValueError("Invalid edge length encountered. Should be finite number greater than or equal to zero.")
        if not np.all(np.isfinite(self.angle_sum)) or not np.all(self.angle_sum >= 0):  # type: ignore
            raise ValueError(
                "Invalid edge angle sum encountered. Should be finite number greater than or equal to zero."
            )
        if not np.all(np.isfinite(self.imp_factor)) or not np.all(self.imp_factor >= 0):  # type: ignore
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
    """Network with nodes, edges, node_edge_map."""

    nodes: NodeMap
    edges: EdgeMap
    node_edge_map: dict[int, list[int]]
    next_edge_idx: int

    def __init__(self, nodes_n: int, edges_n: int):
        """Instance Network."""
        self.nodes = NodeMap(nodes_n)
        self.edges = EdgeMap(edges_n)
        # NOTE
        # List.empty_list(types.int64) only works in jitclass mode
        # i.e. testing in pure python use types.ListType(types.int64) otherwise hash issue arises
        self.node_edge_map = Dict.empty(types.int64, List.empty_list(types.int64))
        self.next_edge_idx = 0

    def set_node(self, node_idx: int, node_x: float, node_y: float, node_live: bool = True):
        """Add a node to the network."""
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
        """Add an edge to the network."""
        self.node_edge_map[start_node_idx].append(self.next_edge_idx)  # type: ignore
        self.edges.start[self.next_edge_idx] = start_node_idx
        self.edges.end[self.next_edge_idx] = end_node_idx
        self.edges.length[self.next_edge_idx] = length
        self.edges.angle_sum[self.next_edge_idx] = angle_sum
        self.edges.imp_factor[self.next_edge_idx] = imp_factor
        self.edges.in_bearing[self.next_edge_idx] = in_bearing
        self.edges.out_bearing[self.next_edge_idx] = out_bearing
        self.next_edge_idx += 1

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
                    if self.edges.end[return_edge_idx] == n_idx:  # type: ignore
                        paired = True
                        break
                if not paired:
                    raise ValueError("Missing matching edge pair in opposite direction.")
                # add to the counter
                edge_counts[edge_idx] += 1
        if not np.all(edge_counts == 1):  # type: ignore
            raise ValueError("Mismatched node and edge maps encountered.")


class DataPoint(TypedDict):
    """DataPoint type for type-hinting."""

    x: float
    y: float


DataDictType = dict[Union[str, int], DataPoint]

data_map_spec: list[tuple[str, Any]] = [
    ("xs", types.float32[:]),
    ("ys", types.float32[:]),
    ("nearest_assign", types.int_[:]),
    ("next_nearest_assign", types.int_[:]),
]


@jitclass(data_map_spec)
class DataMap:
    """Node Map for network."""

    xs: npt.NDArray[np.float32]
    ys: npt.NDArray[np.float32]
    nearest_assign: npt.NDArray[np.int_]
    next_nearest_assign: npt.NDArray[np.int_]

    @property
    def count(self):
        """Alternative to length dunder - which is not yet supported by jitclass."""
        return len(self.xs)

    def __init__(self, data_n: int):
        """Instance DataMap."""
        self.xs = np.full(data_n, np.nan, dtype=np.float32)
        self.ys = np.full(data_n, np.nan, dtype=np.float32)
        self.nearest_assign = np.full(data_n, -1, dtype=np.int_)
        self.next_nearest_assign = np.full(data_n, -1, dtype=np.int_)

    def set_data_point(self, data_idx: int, data_x: np.float32, data_y: np.float32):
        """Add a data point."""
        self.xs[data_idx] = data_x
        self.ys[data_idx] = data_y

    def x_y(self, data_idx: int) -> npt.NDArray[np.float32]:
        """XY coordinates for given data index."""
        return np.array([self.xs[data_idx], self.ys[data_idx]], dtype=np.float32)  # type: ignore

    def validate(self, check_assigned: bool = False):
        """Validate DataMap."""
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
            if np.all(self.nearest_assign == -1):  # type: ignore
                raise ValueError(
                    "Data map has not been assigned to a network. (Else data-points were not assignable "
                    "for the given max_dist parameter passed to assign_to_network."
                )


class CentralityMetricsState(TypedDict, total=False):
    """Centrality metrics typing scaffold."""

    # node shortest
    node_density: dict[int, npt.NDArray[np.float_]]
    node_farness: dict[int, npt.NDArray[np.float_]]
    node_cycles: dict[int, npt.NDArray[np.float_]]
    node_harmonic: dict[int, npt.NDArray[np.float_]]
    node_beta: dict[int, npt.NDArray[np.float_]]
    node_betweenness: dict[int, npt.NDArray[np.float_]]
    node_betweenness_beta: dict[int, npt.NDArray[np.float_]]
    # node simplest
    node_harmonic_angular: dict[int, npt.NDArray[np.float_]]
    node_betweenness_angular: dict[int, npt.NDArray[np.float_]]
    # segment shortest
    segment_density: dict[int, npt.NDArray[np.float_]]
    segment_harmonic: dict[int, npt.NDArray[np.float_]]
    segment_beta: dict[int, npt.NDArray[np.float_]]
    segment_betweenness: dict[int, npt.NDArray[np.float_]]
    # segment simplest
    segment_harmonic_hybrid: dict[int, npt.NDArray[np.float_]]
    segment_betweeness_hybrid: dict[int, npt.NDArray[np.float_]]


class MixedUsesMetricsState(TypedDict, total=False):
    """Mixed-uses metrics typing scaffold."""

    # hill measures have q keys
    hill: dict[float | int, dict[int, npt.NDArray[np.float_]]]
    hill_branch_wt: dict[float | int, dict[int, npt.NDArray[np.float_]]]
    hill_pairwise_wt: dict[float | int, dict[int, npt.NDArray[np.float_]]]
    hill_pairwise_disparity: dict[float | int, dict[int, npt.NDArray[np.float_]]]
    # non-hill do not have q keys
    shannon: dict[int, npt.NDArray[np.float_]]
    gini_simpson: dict[int, npt.NDArray[np.float_]]
    raos_pairwise_disparity: dict[int, npt.NDArray[np.float_]]


@dataclass()
class AccessibilityMetricsState:
    """Accessibility metrics typing scaffold."""

    weighted: dict[str, dict[int, npt.NDArray[np.float_]]] = field(default_factory=dict)
    non_weighted: dict[str, dict[int, npt.NDArray[np.float_]]] = field(default_factory=dict)


class StatsMetricsState(TypedDict, total=False):
    """Stats metrics typing scaffold."""

    max: dict[int, npt.NDArray[np.float_]]
    min: dict[int, npt.NDArray[np.float_]]
    sum: dict[int, npt.NDArray[np.float_]]
    sum_weighted: dict[int, npt.NDArray[np.float_]]
    mean: dict[int, npt.NDArray[np.float_]]
    mean_weighted: dict[int, npt.NDArray[np.float_]]
    variance: dict[int, npt.NDArray[np.float_]]
    variance_weighted: dict[int, npt.NDArray[np.float_]]


NodeMetrics = dict[str, dict[str, Any]]
DictNodeMetrics = dict[Union[str, int], NodeMetrics]


class MetricsState:
    """Metrics typing scaffold."""

    centrality: CentralityMetricsState = CentralityMetricsState()
    mixed_uses: MixedUsesMetricsState = MixedUsesMetricsState()
    accessibility: AccessibilityMetricsState = AccessibilityMetricsState()
    stats: dict[str, StatsMetricsState] = {}

    def extract_node_metrics(self, node_idx: int) -> NodeMetrics:
        """Extract metrics for a given node idx."""
        node_state: NodeMetrics = {}
        # centrality
        node_state["centrality"] = {}
        # pylint: disable=duplicate-code
        for key in [
            "node_density",
            "node_farness",
            "node_cycles",
            "node_harmonic",
            "node_beta",
            "node_betweenness",
            "node_betweenness_beta",
            "node_harmonic_angular",
            "node_betweenness_angular",
            "segment_density",
            "segment_harmonic",
            "segment_beta",
            "segment_betweenness",
            "segment_harmonic_hybrid",
            "segment_betweeness_hybrid",
        ]:
            if key in self.centrality:
                node_state["centrality"][key] = {}
                for d_key, d_val in self.centrality[key].items():  # type: ignore
                    node_state["centrality"][key][d_key] = d_val[node_idx]
        # mixed-uses - hill
        node_state["mixed_uses"] = {}
        for key in [
            "hill",
            "hill_branch_wt",
            "hill_pairwise_wt",
            "hill_pairwise_disparity",
        ]:
            if key in self.mixed_uses:
                node_state["mixed_uses"][key] = {}
                for q_key, q_val in self.mixed_uses[key].items():  # type: ignore
                    node_state["mixed_uses"][key][q_key] = {}
                    for d_key, d_val in q_val.items():  # type: ignore
                        node_state["mixed_uses"][key][q_key][d_key] = d_val[node_idx]
        # mixed-uses non-hill
        for key in [
            "shannon",
            "gini_simpson",
            "raos_pairwise_disparity",
        ]:
            if key in self.mixed_uses:
                node_state["mixed_uses"][key] = {}
                for d_key, d_val in self.mixed_uses[key].items():  # type: ignore
                    node_state["mixed_uses"][key][d_key] = d_val[node_idx]
        # accessibility
        node_state["accessibility"] = {"non_weighted": {}, "weighted": {}}
        # non-weighted
        for cl_key, cl_val in self.accessibility.non_weighted.items():  # type: ignore
            node_state["accessibility"]["non_weighted"][cl_key] = {}
            for d_key, d_val in cl_val.items():  # type: ignore
                node_state["accessibility"]["non_weighted"][cl_key][d_key] = d_val[node_idx]
        # weighted
        for cl_key, cl_val in self.accessibility.weighted.items():  # type: ignore
            node_state["accessibility"]["weighted"][cl_key] = {}
            for d_key, d_val in cl_val.items():  # type: ignore
                node_state["accessibility"]["weighted"][cl_key][d_key] = d_val[node_idx]
        # stats
        node_state["stats"] = {}
        for th_key in self.stats:  # pylint: disable=consider-using-dict-items
            node_state["stats"][th_key] = {}
            for stat_attr in [
                "max",
                "min",
                "sum",
                "sum_weighted",
                "mean",
                "mean_weighted",
                "variance",
                "variance_weighted",
            ]:
                node_state["stats"][th_key][stat_attr] = {}
                stat_val = getattr(self.stats[th_key], stat_attr)
                for d_key, d_val in stat_val.items():
                    node_state["stats"][th_key][stat_attr][d_key] = d_val[node_idx]

        return node_state
