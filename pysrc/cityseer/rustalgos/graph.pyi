"""Graph data structures and utilities for network analysis."""

from __future__ import annotations

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
    start_nd_key: str
    end_nd_key: str
    edge_idx: int
    length: float
    angle_sum: float
    imp_factor: float
    in_bearing: float
    out_bearing: float
    seconds: float
    geom_wkt: str
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
    progress: int
    edge_rtree: object | None
    edge_rtree_built: bool
    @classmethod
    def new(cls) -> NetworkStructure: ...
    def progress_init(self) -> None: ...
    def add_node(self, node_key: str, x: float, y: float, live: bool, weight: float) -> int:
        """
        Parameters
        ----------
        node_key: str
            The node key as `str`.
        x: float
            The node's `x` coordinate.
        y: float
            The node's `y` coordinate.
        live: bool
            The `live` node attribute identifying if this node falls within the areal boundary of interest as opposed to
            those that fall within the surrounding buffered area. See the [edge-rolloff](/guide#edge-rolloff) section in
            the guide.
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
    def add_edge(
        self,
        start_nd_idx: int,
        end_nd_idx: int,
        edge_idx: int,
        start_nd_key: str,
        end_nd_key: str,
        geom_wkt: str,
        imp_factor: float | None = None,
        seconds: float | None = None,
    ) -> int:
        """
        Add an edge to the `NetworkStructure`.

        Edges are directed, meaning that each bidirectional street is represented twice: once in each direction;
        start/end nodes and in/out bearings will differ accordingly.

        Parameters
        ----------
        start_node_idx: str
            Node index for the starting node.
        end_node_idx: str
            Node index for the ending node.
        edge_idx: int
            The edge index, such that multiple edges can span between the same node pair.
        start_node_key: str
            Node key for the starting node.
        end_node_key: str
            Node key for the ending node.
        geom_wkt: str
            The geometry of the edge in WKT format.
        imp_factor: float
            The `imp_factor` edge attribute represents an impedance multiplier for increasing or diminishing
            the impedance of an edge. This is ordinarily set to 1, therefore not impacting calculations. By setting
            this to greater or less than 1, the edge will have a correspondingly higher or lower impedance. This can
            be used to take considerations such as street gradients into account, but should be used with caution.
        seconds: int
            The edge's traversal time in seconds.

        """
        ...

    def edge_references(self) -> list[tuple[int, int, int]]: ...
    def get_edge_payload(self, start_nd_idx: int, end_nd_idx: int, edge_idx: int) -> EdgePayload: ...
    def validate(self) -> bool:
        """Validate Network Structure."""
        ...

    def build_edge_rtree(self) -> None: ...
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
