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
    ("x", types.float32[:]),
    ("y", types.float32[:]),
    ("live", types.bool_[:]),
]


@jitclass(node_map_spec)
class NodeMap:
    """Node Map for network."""

    x: npt.NDArray[np.float32]
    y: npt.NDArray[np.float32]
    live: npt.NDArray[np.bool_]

    def __init__(self, nodes_n: int):
        """Instance NodeMap."""
        self.x = np.full(nodes_n, np.nan, dtype=np.float32)
        self.y = np.full(nodes_n, np.nan, dtype=np.float32)
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
    ("node_label", types.unicode_type),
    ("node_x", types.float32),
    ("node_y", types.float32),
    ("node_live", types.bool_),
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
    ("node_uids", types.ListType(types.unicode_type)),
    ("node_edge_map", types.DictType(types.int64, types.ListType(types.int64))),
    ("next_edge_idx", types.int_),
]


@jitclass(network_spec)
class NetworkStructure:
    """Network with nodes, edges, node_edge_map."""

    node_uids: list[str]
    nodes: NodeMap
    edges: EdgeMap
    node_edge_map: dict[int, list[int]]
    next_edge_idx: int

    def __init__(self, nodes_n: int, edges_n: int):
        """Instance Network."""
        self.node_uids = List.empty_list(types.unicode_type)
        self.nodes = NodeMap(nodes_n)
        self.edges = EdgeMap(edges_n)
        self.node_edge_map = Dict.empty(types.int64, List.empty_list(types.int64))
        self.next_edge_idx = 0

    def set_node(self, node_idx: int, node_label: str, node_x: float, node_y: float, node_live: bool):
        """Add a node to the network."""
        self.node_uids.append(node_label)
        self.nodes.x[node_idx] = node_x
        self.nodes.y[node_idx] = node_y
        self.nodes.live[node_idx] = node_live
        self.node_edge_map[node_idx] = List.empty_list(types.int64)  # type: ignore

    def set_edge(
        self,
        root_node_idx: int,
        edge_idx: int,
        start: int,
        end: int,
        length: float,
        angular: float,
        impedance: float,
        in_bearing: float,
        out_bearing: float,
    ):
        """Add an edge to the network."""
        self.node_edge_map[root_node_idx].append(edge_idx)  # type: ignore
        self.edges.start[edge_idx] = start
        self.edges.end[edge_idx] = end
        self.edges.length[edge_idx] = length
        self.edges.angular[edge_idx] = angular
        self.edges.impedance[edge_idx] = impedance
        self.edges.in_bearing[edge_idx] = in_bearing
        self.edges.out_bearing[edge_idx] = out_bearing
