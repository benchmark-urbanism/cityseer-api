from __future__ import annotations

from typing import Any
import numpy as np
import numpy.typing as npt
from numba.core import types
from numba.typed import Dict, List
from numba.experimental import jitclass  # type: ignore


tree_map_spec: list[tuple[str, Any]] = [
    ("nodes_n", types.int_),
    ("edges_n", types.int_),
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
    ("x", types.float_[:]),
    ("y", types.float_[:]),
    ("live", types.bool_[:]),
]


@jitclass(node_map_spec)
class NodeMap:
    """Node Map for network."""

    x: npt.NDArray[np.float_]
    y: npt.NDArray[np.float_]
    live: npt.NDArray[np.bool_]

    def __init__(self, nodes_n: int):
        """Instance NodeMap."""
        self.x = np.full(nodes_n, np.nan, dtype=np.float_)
        self.y = np.full(nodes_n, np.nan, dtype=np.float_)
        self.live = np.full(nodes_n, False, dtype=np.bool_)


edge_map_spec: list[tuple[str, Any]] = [
    ("edges_n", types.int_),
    ("start", types.int_[:]),
    ("end", types.int_[:]),
    ("length", types.float32[:]),
    ("angular", types.float32[:]),
    ("impedance", types.float32[:]),
    ("in_bearing", types.float32[:]),
    ("out_bearing", types.float32[:]),
]


@jitclass(edge_map_spec)
class EdgeMap:
    """Edge Map for network."""

    start: npt.NDArray[np.int_]
    end: npt.NDArray[np.int_]
    length: npt.NDArray[np.float32]
    angular: npt.NDArray[np.float32]
    impedance: npt.NDArray[np.float32]
    in_bearing: npt.NDArray[np.float32]
    out_bearing: npt.NDArray[np.float32]

    def __init__(self, edges_n: int):
        """Instance EdgeMap."""
        self.start = np.full(edges_n, -1, dtype=np.int_)
        self.end = np.full(edges_n, -1, dtype=np.int_)
        self.length = np.full(edges_n, np.nan, dtype=np.float32)
        self.angular = np.full(edges_n, np.nan, dtype=np.float32)
        self.impedance = np.full(edges_n, np.nan, dtype=np.float32)
        self.in_bearing = np.full(edges_n, np.nan, dtype=np.float32)
        self.out_bearing = np.full(edges_n, np.nan, dtype=np.float32)


network_spec: list[tuple[str, Any]] = [
    # init
    ("nodes_n", types.int_),
    ("edges_n", types.int_),
    # set_node
    ("node_idx", types.int_),
    ("label", types.unicode_type),
    ("x", types.float_),
    ("y", types.float_),
    ("live", types.bool_),
    # set_edge
    ("root_node_idx", types.int_),
    ("start", types.int_),
    ("end", types.int_),
    ("length", types.float32),
    ("angular", types.float32),
    ("impedance", types.float32),
    ("in_bearing", types.float32),
    ("out_bearing", types.float32),
    # attributes
    ("node_uids", types.uchar[:]),
    ("node_edge_map", types.DictType(types.int64, types.ListType(types.int64))),
    ("next_edge_idx", types.int_),
]


# @jitclass(network_spec)
class NetworkStructure:
    """Network with nodes, edges, node_edge_map."""

    node_uids: npt.NDArray[np.str_]
    nodes: NodeMap
    edges: EdgeMap
    node_edge_map: dict[int, list[int]]
    next_edge_idx: int

    def __init__(self, nodes_n: int, edges_n: int):
        """Instance Network."""
        self.node_uids = np.full(nodes_n, "", dtype=np.unicode_)
        self.nodes = NodeMap(nodes_n)
        self.edges = EdgeMap(edges_n)
        self.node_edge_map = Dict.empty(key_type=types.int64, value_type=types.List(types.int64))
        self.next_edge_idx = 0

    def set_node(self, node_idx: int, label: str, x: float, y: float, live: bool):
        """Add a node to the network."""
        self.node_uids[node_idx] = label
        self.nodes.x[node_idx] = x
        self.nodes.y[node_idx] = y
        self.nodes.live[node_idx] = live
        self.node_edge_map[node_idx] = List.empty_list(types.int64)  # type: ignore

    def set_edge(
        self,
        root_node_idx: int,
        start: int,
        end: int,
        length: float,
        angular: float,
        impedance: float,
        in_bearing: float,
        out_bearing: float,
    ):
        """Add an edge to the network."""
        self.node_edge_map[root_node_idx].append(self.next_edge_idx)
        self.edges.start[self.next_edge_idx] = start
        self.edges.end[self.next_edge_idx] = end
        self.edges.length[self.next_edge_idx] = length
        self.edges.angular[self.next_edge_idx] = angular
        self.edges.impedance[self.next_edge_idx] = impedance
        self.edges.in_bearing[self.next_edge_idx] = in_bearing
        self.edges.out_bearing[self.next_edge_idx] = out_bearing
        self.next_edge_idx += 1
