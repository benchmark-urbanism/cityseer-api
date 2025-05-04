"""Graph data structures and utilities for network analysis, including pathfinding and centrality."""

from __future__ import annotations

from typing import Any

from .centrality import CentralitySegmentResult, CentralityShortestResult, CentralitySimplestResult

__doc__: str

class NodePayload:
    """Payload data associated with a network node."""

    node_key: Any
    live: bool
    weight: float
    is_transport: bool
    def validate(self) -> None:
        """Validate node payload attributes (e.g., weight non-negative)."""
        ...
    @property
    def coord(self) -> tuple[float, float]:
        """Get the (x, y) coordinates of the node."""
        ...

class EdgePayload:
    """Payload data associated with a network edge."""

    start_nd_key_py: Any | None
    end_nd_key_py: Any | None
    edge_idx: int
    length: float
    angle_sum: float
    imp_factor: float
    in_bearing: float
    out_bearing: float
    seconds: float
    geom_wkt: str | None
    is_transport: bool
    def validate(self) -> None:
        """Validate edge payload attributes (e.g., impedance positive, consistency)."""
        ...

class NodeVisit:
    """State information for a node during a graph traversal (e.g., Dijkstra)."""

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
    def new(cls) -> NodeVisit:
        """Initialize a new NodeVisit state."""
        ...

class EdgeVisit:
    """State information for an edge during a graph traversal."""

    visited: bool
    start_nd_idx: int | None
    end_nd_idx: int | None
    edge_idx: int | None
    @classmethod
    def new(cls) -> EdgeVisit:
        """Initialize a new EdgeVisit state."""
        ...

class DiGraph: ...  # Placeholder for the internal graph representation

class NetworkStructure:
    """Manages the network graph, including nodes, edges, barriers, and spatial indexing."""

    graph: DiGraph
    edge_rtree: object | None  # R-tree for efficient spatial queries on edges
    @classmethod
    def new(cls) -> NetworkStructure:
        """Create a new, empty NetworkStructure."""
        ...
    def progress_init(self) -> None:
        """Reset the internal progress counter (used for long operations)."""
        ...
    @property
    def progress(self) -> int:
        """Get the current value of the internal progress counter."""
        ...
    def add_street_node(self, node_key: Any, x: float, y: float, live: bool, weight: float) -> int:
        """
        Add a standard street network node.

        Parameters
        ----------
        node_key: Any
            Unique identifier for the node.
        x: float
            Node's x-coordinate.
        y: float
            Node's y-coordinate.
        live: bool
            Indicates if the node is within the primary analysis area.
        weight: float
            Node weight (e.g., for weighted centrality calculations, >= 0).

        Returns
        -------
        int
            The internal index assigned to the node.
        """
        ...

    def add_transport_node(
        self,
        node_key: Any,
        x: float,
        y: float,
        linking_radius: float | None = None,
        speed_m_s: float | None = None,
    ) -> int:
        """
        Add a transport node (e.g., station, stop) and optionally link it to nearby street nodes.

        Transport nodes have `live=False` and `weight=0` implicitly.
        Linking creates bi-directional 'transport' edges between the transport node
        and valid street nodes within `linking_radius`.
        Requires `build_edge_rtree()` to be called first if linking.

        Parameters
        ----------
        node_key: Any
            Unique identifier for the transport node.
        x: float
            Node's x-coordinate.
        y: float
            Node's y-coordinate.
        linking_radius: float | None
            Max distance (meters) to search for street nodes to link to (default: 100.0).
        speed_m_s: float | None
            Speed used to calculate travel time for linking edges (default: standard walking speed).

        Returns
        -------
        int
            The internal index assigned to the transport node.
        """
        ...

    def get_node_payload(self, node_idx: int) -> NodePayload:
        """Retrieve the payload data for a specific node index."""
        ...
    def get_node_weight(self, node_idx: int) -> float:
        """Get the weight of a specific node index."""
        ...
    def is_node_live(self, node_idx: int) -> bool:
        """Check if a specific node index is marked as 'live'."""
        ...
    def node_count(self) -> int:
        """Get the total number of nodes in the graph."""
        ...
    def street_node_count(self) -> int:
        """Get the number of street nodes in the graph."""
        ...
    def node_indices(self) -> list[int]:
        """Get indices for all nodes."""
        ...
    def street_node_indices(self) -> list[int]:
        """Get indices for street nodes only."""
        ...
    @property
    def node_xs(self) -> list[float]:
        """Get x-coordinates for all nodes."""
        ...
    @property
    def street_node_xs(self) -> list[float]:
        """Get x-coordinates for street nodes only."""
        ...
    @property
    def node_ys(self) -> list[float]:
        """Get y-coordinates for all nodes."""
        ...
    @property
    def street_node_ys(self) -> list[float]:
        """Get y-coordinates for street nodes only."""
        ...
    @property
    def node_xys(self) -> list[tuple[float, float]]:
        """Get (x, y) coordinates for all nodes."""
        ...
    @property
    def street_node_xys(self) -> list[tuple[float, float]]:
        """Get (x, y) coordinates for street nodes only."""
        ...
    @property
    def node_lives(self) -> list[bool]:
        """Get 'live' status for all nodes."""
        ...
    @property
    def street_node_lives(self) -> list[bool]:
        """Get 'live' status for street nodes only."""
        ...
    @property
    def edge_count(self) -> int:
        """Get the total number of edges in the graph."""
        ...
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
        Add a directed street edge with geometry.

        Calculates length, bearings, angle sum from WKT. `seconds` is NaN (calculated during traversal).
        Invalidates the edge R-tree; call `build_edge_rtree()` afterwards if needed.

        Parameters
        ----------
        start_nd_idx: int
            Index of the starting node.
        end_nd_idx: int
            Index of the ending node.
        edge_idx: int
            External identifier for the edge (allows multiple edges between nodes).
        start_nd_key_py: Any
            Original key of the starting node.
        end_nd_key_py: Any
            Original key of the ending node.
        geom_wkt: str
            Edge geometry in WKT format (must have >= 2 points).
        imp_factor: float | None
            Impedance multiplier (> 0.0, default 1.0).

        Returns
        -------
        int
            The internal index assigned to the edge.
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
        Add a directed abstract transport edge defined by travel time.

        Length, geometry, bearings, angle sum are NaN/None.

        Parameters
        ----------
        start_nd_idx: int
            Index of the starting node.
        end_nd_idx: int
            Index of the ending node.
        edge_idx: int
            External identifier for the edge.
        start_nd_key_py: Any
            Original key of the starting node.
        end_nd_key_py: Any
            Original key of the ending node.
        seconds: float
            Travel time in seconds (>= 0.0).
        imp_factor: float | None
            Impedance multiplier (> 0.0, default 1.0).

        Returns
        -------
        int
            The internal index assigned to the edge.
        """
        ...

    def edge_references(self) -> list[tuple[int, int, int]]:
        """Get list of (start_node_idx, end_node_idx, edge_idx) for all edges."""
        ...
    def get_edge_payload(self, start_nd_idx: int, end_nd_idx: int, edge_idx: int) -> EdgePayload:
        """Retrieve the payload for a specific edge defined by nodes and edge_idx."""
        ...
    def validate(self) -> None:
        """Check internal consistency of all nodes and edges in the graph."""
        ...

    def build_edge_rtree(self) -> None:
        """Build or rebuild the R-tree spatial index for street edges. Deduplicates based on geometry."""
        ...
    def set_barriers(self, barriers_wkt: list[str]) -> None:
        """Set impassable barrier geometries (from WKT) and build their R-tree."""
        ...
    def unset_barriers(self) -> None:
        """Remove all barrier geometries and their R-tree."""
        ...
    def find_assignments_for_entry(self, data_key: str, geom: Any, max_dist: float) -> list[tuple[int, str, float]]:
        """
        Find valid network node assignments for a data entry's geometry.

        Checks proximity, max distance, barrier intersections, and street intersections.
        Requires `build_edge_rtree()` to have been called.

        Parameters
        ----------
        data_key: str
            Identifier for the data entry (used in return value).
        geom: Any
            Geometry object of the data entry (must support `centroid`, `closest_point`, `distance`).
        max_dist: float
            Maximum assignment distance (meters).

        Returns
        -------
        list[tuple[int, str, float]]
            List of (assigned_node_idx, data_key, assignment_distance).
        """
        ...
    def dijkstra_tree_shortest(
        self, src_idx: int, max_seconds: int, speed_m_s: float, jitter_scale: float | None = None
    ) -> tuple[list[int], list[NodeVisit]]:
        """
        Compute shortest path tree (metric distance) from a source node using Dijkstra.

        Parameters
        ----------
        src_idx: int
            Starting node index.
        max_seconds: int
            Maximum travel time cutoff.
        speed_m_s: float
            Travel speed (m/s) to convert edge lengths to time.
        jitter_scale: float | None
            Optional scale for random cost jitter (tie-breaking).

        Returns
        -------
        tuple[list[int], list[NodeVisit]]
            (List of reachable node indices, List of NodeVisit states for all nodes).
        """
        ...
    def dijkstra_tree_simplest(
        self, src_idx: int, max_seconds: int, speed_m_s: float, jitter_scale: float | None = None
    ) -> tuple[list[int], list[NodeVisit]]:
        """
        Compute simplest path tree (angular distance) from a source node using Dijkstra.

        Parameters
        ----------
        src_idx: int
            Starting node index.
        max_seconds: int
            Maximum travel time cutoff.
        speed_m_s: float
            Travel speed (m/s).
        jitter_scale: float | None
            Optional scale for random cost jitter.

        Returns
        -------
        tuple[list[int], list[NodeVisit]]
            (List of reachable node indices, List of NodeVisit states for all nodes).
        """
        ...
    def dijkstra_tree_segment(
        self, src_idx: int, max_seconds: int, speed_m_s: float, jitter_scale: float | None = None
    ) -> tuple[list[int], list[int], list[NodeVisit], list[EdgeVisit]]:
        """
        Compute shortest path tree for segment-based analysis.

        Parameters
        ----------
        src_idx: int
            Starting node index.
        max_seconds: int
            Maximum travel time cutoff.
        speed_m_s: float
            Travel speed (m/s).
        jitter_scale: float | None
            Optional scale for random cost jitter.

        Returns
        -------
        tuple[list[int], list[int], list[NodeVisit], list[EdgeVisit]]
            (Reachable node indices, Visited edge indices, NodeVisit states, EdgeVisit states).
        """
        ...
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
    ) -> CentralityShortestResult:
        """
        Calculate local node centrality metrics based on shortest paths (metric distance).

        Computes closeness and/or betweenness centrality within specified catchment thresholds.
        Requires exactly one of `distances`, `betas`, or `minutes`.

        Parameters
        ----------
        distances: list[int] | None
            Distance thresholds (meters).
        betas: list[float] | None
            Decay parameters (beta).
        minutes: list[float] | None
            Time thresholds (minutes).
        compute_closeness: bool | None
            Compute closeness centrality if True.
        compute_betweenness: bool | None
            Compute betweenness centrality if True.
        min_threshold_wt: float | None
            Minimum weight for beta/distance conversion.
        speed_m_s: float | None
            Travel speed (m/s).
        jitter_scale: float | None
            Path cost jitter scale.
        pbar_disabled: bool | None
            Disable progress bar if True.

        Returns
        -------
        CentralityShortestResult
            Object containing calculated centrality metrics.
        """
        ...
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
    ) -> CentralitySimplestResult:
        """
        Calculate local node centrality metrics based on simplest paths (angular distance).

        Computes closeness and/or betweenness centrality within specified catchment thresholds.
        Requires exactly one of `distances`, `betas`, or `minutes`.

        Parameters
        ----------
        distances: list[int] | None
            Distance thresholds (meters).
        betas: list[float] | None
            Decay parameters (beta).
        minutes: list[float] | None
            Time thresholds (minutes).
        compute_closeness: bool | None
            Compute closeness centrality if True.
        compute_betweenness: bool | None
            Compute betweenness centrality if True.
        min_threshold_wt: float | None
            Minimum weight for beta/distance conversion.
        speed_m_s: float | None
            Travel speed (m/s).
        angular_scaling_unit: float | None
            Scaling unit for angular cost (default: 90 degrees).
        farness_scaling_offset: float | None
            Offset for farness calculation (default: 1.0).
        jitter_scale: float | None
            Path cost jitter scale.
        pbar_disabled: bool | None
            Disable progress bar if True.

        Returns
        -------
        CentralitySimplestResult
            Object containing calculated centrality metrics.
        """
        ...
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
    ) -> CentralitySegmentResult:
        """
        Calculate local segment centrality metrics based on shortest paths.

        Computes closeness and/or betweenness centrality for network segments within specified thresholds.
        Requires exactly one of `distances`, `betas`, or `minutes`.

        Parameters
        ----------
        distances: list[int] | None
            Distance thresholds (meters).
        betas: list[float] | None
            Decay parameters (beta).
        minutes: list[float] | None
            Time thresholds (minutes).
        compute_closeness: bool | None
            Compute closeness centrality if True.
        compute_betweenness: bool | None
            Compute betweenness centrality if True.
        min_threshold_wt: float | None
            Minimum weight for beta/distance conversion.
        speed_m_s: float | None
            Travel speed (m/s).
        jitter_scale: float | None
            Path cost jitter scale.
        pbar_disabled: bool | None
            Disable progress bar if True.

        Returns
        -------
        CentralitySegmentResult
            Object containing calculated segment centrality metrics.
        """
        ...
