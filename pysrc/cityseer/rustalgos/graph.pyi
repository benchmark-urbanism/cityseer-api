"""Graph data structures and utilities for network analysis."""

from __future__ import annotations

from typing import Any

from .centrality import CentralitySegmentResult, CentralityShortestResult, CentralitySimplestResult

__doc__: str

class NodePayload:
    node_key: str
    live: bool
    weight: float
    def validate(self) -> bool: ...
    @property
    def coord(self) -> tuple[float, float]: ...

class EdgePayload:
    start_nd_key_py: Any
    end_nd_key_py: Any
    edge_idx: int
    length: float
    angle_sum: float
    imp_factor: float
    in_bearing: float
    out_bearing: float
    seconds: float
    geom_wkt: str | None
    is_transport: bool
    def validate(self) -> bool: ...

class NodeVisit:
    visited: bool
    discovered: bool
    pred: int | None
    short_dist: float
    simpl_dist: float
    cycles: float
    origin_seg: int | None
    last_seg: int | None
    out_bearing: float
    agg_seconds: float
    @classmethod
    def new(cls) -> NodeVisit: ...

class EdgeVisit:
    visited: bool
    start_nd_idx: int | None
    end_nd_idx: int | None
    edge_idx: int | None
    @classmethod
    def new(cls) -> EdgeVisit: ...

class DiGraph: ...

class NetworkStructure:
    graph: DiGraph
    edge_rtree: object | None
    @classmethod
    def new(cls) -> NetworkStructure: ...
    def progress_init(self) -> None: ...
    @property
    def progress(self) -> int: ...
    def add_street_node(self, node_key: str, x: float, y: float, live: bool, weight: float) -> int:
        """
        Add a street node to the `NetworkStructure`.

        Parameters
        ----------
        node_key: str
            The node key as `str`.
        x: float
            The node's `x` coordinate.
        y: float
            The node's `y` coordinate.
        live: bool
            The `live` node attribute identifying if this node falls within the areal boundary of interest.
        weight: float
            The node's weight.

        Returns
        -------
        int
            The internal index of the newly added node.
        """
        ...

    def add_transport_node(self, node_key: str, x: float, y: float) -> int:
        """
        Add a transport node (e.g., station, stop) to the `NetworkStructure`. `weight` is implicitly 0 and ignored for
        computations (`live=False`).

        Parameters
        ----------
        node_key: str
            The node key as `str`.
        x: float
            The node's `x` coordinate.
        y: float
            The node's `y` coordinate.

        Returns
        -------
        int
            The internal index of the newly added node.
        """
        ...

    def get_node_payload(self, node_idx: int) -> NodePayload: ...
    def get_node_weight(self, node_idx: int) -> float: ...
    def is_node_live(self, node_idx: int) -> bool: ...
    def node_count(self) -> int: ...
    def node_indices(self) -> list[int]: ...
    @property
    def node_xs(self) -> list[float]:
        """`x` coordinates."""
        ...

    @property
    def node_ys(self) -> list[float]:
        """`y` coordinates."""
        ...

    @property
    def node_xys(self) -> list[tuple[float, float]]:
        """`x` and `y` node coordinates."""
        ...

    @property
    def node_lives(self) -> list[bool]:
        """`live` status indicators."""
        ...

    @property
    def edge_count(self) -> int: ...
    def add_street_edge(
        self,
        start_nd_idx: int,
        end_nd_idx: int,
        edge_idx: int,
        start_nd_key_py: Any,
        end_nd_key_py: Any,
        geom_wkt: str,
        imp_factor: float | None = None,
    ) -> int:
        """
        Add a street edge with geometry to the `NetworkStructure`.

        Edges are directed. Calculates length, bearings, and angle sum from the provided WKT geometry.
        The edge's `seconds` attribute will be NaN, intended to be calculated later based on length and speed.

        Parameters
        ----------
        start_node_idx: int
            Node index for the starting node.
        end_node_idx: int
            Node index for the ending node.
        edge_idx: int
            The edge index, such that multiple edges can span between the same node pair.
        start_nd_key_py: Any
            Node key for the starting node.
        end_nd_key_py: Any
            Node key for the ending node.
        geom_wkt: str
            The geometry of the edge in WKT format.
        imp_factor: float | None
            Impedance multiplier (default 1.0). Applied during pathfinding.

        Raises
        ------
        ValueError
            If `geom_wkt` cannot be parsed or has fewer than 2 coordinates.

        Returns
        -------
        int
            The internal index of the newly added edge.
        """
        ...

    def add_transport_edge(
        self,
        start_nd_idx: int,
        end_nd_idx: int,
        edge_idx: int,
        start_nd_key_py: Any,
        end_nd_key_py: Any,
        seconds: float,
        imp_factor: float | None = None,
    ) -> int:
        """
        Add an abstract transport edge defined by travel time to the `NetworkStructure`.

        Edges are directed. Calculates length based on the straight-line distance between nodes.
        Geometry-related attributes (bearings, angle sum, geom_wkt, geom) will be NaN or None.

        Parameters
        ----------
        start_node_idx: int
            Node index for the starting node.
        end_node_idx: int
            Node index for the ending node.
        edge_idx: int
            The edge index, such that multiple edges can span between the same node pair.
        start_nd_key_py: Any
            Node key for the starting node.
        end_nd_key_py: Any
            Node key for the ending node.
        seconds: float
            The edge's traversal time in seconds.
        imp_factor: float | None
            Impedance multiplier (default 1.0). Applied during pathfinding.

        Returns
        -------
        int
            The internal index of the newly added edge.
        """
        ...

    def edge_references(self) -> list[tuple[int, int, int]]: ...
    def get_edge_payload(self, start_nd_idx: int, end_nd_idx: int, edge_idx: int) -> EdgePayload: ...
    def validate(self) -> bool:
        """Validate Network Structure."""
        ...

    def build_edge_rtree(self) -> None: ...
    def set_barriers(self, barriers_wkt: list[str]) -> None: ...
    def unset_barriers(self) -> None: ...
    def find_assignments_for_entry(self, data_key: str, geom: Any, max_dist: float) -> list[tuple[int, str, float]]: ...
    def dijkstra_tree_shortest(
        self, src_idx: int, max_seconds: int, speed_m_s: float, jitter_scale: float | None = None
    ) -> tuple[list[int], list[NodeVisit]]: ...
    def dijkstra_tree_simplest(
        self, src_idx: int, max_seconds: int, speed_m_s: float, jitter_scale: float | None = None
    ) -> tuple[list[int], list[NodeVisit]]: ...
    def dijkstra_tree_segment(
        self, src_idx: int, max_seconds: int, speed_m_s: float, jitter_scale: float | None = None
    ) -> tuple[list[int], list[int], list[NodeVisit], list[EdgeVisit]]: ...
    def local_node_centrality_shortest(
        self,
        distances: list[int] | None = None,
        betas: list[float] | None = None,
        minutes: list[float] | None = None,
        compute_closeness: bool | None = True,
        compute_betweenness: bool | None = True,
        min_threshold_wt: float | None = None,
        speed_m_s: float | None = None,
        jitter_scale: float | None = None,
        pbar_disabled: bool | None = None,
    ) -> CentralityShortestResult: ...
    def local_node_centrality_simplest(
        self,
        distances: list[int] | None = None,
        betas: list[float] | None = None,
        minutes: list[float] | None = None,
        compute_closeness: bool | None = True,
        compute_betweenness: bool | None = True,
        min_threshold_wt: float | None = None,
        speed_m_s: float | None = None,
        angular_scaling_unit: float | None = None,
        farness_scaling_offset: float | None = None,
        jitter_scale: float | None = None,
        pbar_disabled: bool | None = None,
    ) -> CentralitySimplestResult: ...
    def local_segment_centrality(
        self,
        distances: list[int] | None = None,
        betas: list[float] | None = None,
        minutes: list[float] | None = None,
        compute_closeness: bool | None = True,
        compute_betweenness: bool | None = True,
        min_threshold_wt: float | None = None,
        speed_m_s: float | None = None,
        jitter_scale: float | None = None,
        pbar_disabled: bool | None = None,
    ) -> CentralitySegmentResult: ...
