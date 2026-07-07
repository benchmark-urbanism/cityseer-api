"""Graph data structures and utilities for network analysis, including pathfinding and centrality."""

from __future__ import annotations

from typing import Any

from .centrality import (
    CentralityResult,
    OdMatrix,
)

__doc__: str

class NodePayload:
    """Payload data associated with a network node."""

    node_key: Any  # In Rust: Py<PyAny>
    z: float | None  # In Rust: Option<f64>
    live: bool
    weight: float  # In Rust: f32
    is_transport: bool
    def validate(self) -> None:  # In Rust: validate(self, py: Python) -> PyResult<()>
        """Validate node payload attributes (e.g., weight non-negative)."""
        ...
    @property
    def coord(self) -> tuple[float, float]:  # In Rust: getter returns (f64, f64)
        """Get the (x, y) coordinates of the node."""
        ...
    @property
    def coord_z(self) -> tuple[float, float, float | None]:  # In Rust: getter returns (f64, f64, Option<f64>)
        """Get the (x, y, z) coordinates of the node, where z may be None."""
        ...

class EdgePayload:
    """Payload data associated with a network edge."""

    start_nd_key_py: Any | None  # In Rust: Option<Py<PyAny>>
    end_nd_key_py: Any | None  # In Rust: Option<Py<PyAny>>
    edge_idx: int  # In Rust: usize
    length: float  # In Rust: f32
    angle_sum: float  # In Rust: f32
    imp_factor: float  # In Rust: f32
    in_bearing: float  # In Rust: f32
    out_bearing: float  # In Rust: f32
    seconds: float  # In Rust: f32
    geom_wkt: str | None  # In Rust: Option<String>
    is_transport: bool
    def validate(self) -> None:  # In Rust: validate(self, py: Python) -> PyResult<()>
        """Validate edge payload attributes (e.g., impedance positive, consistency)."""
        ...

class NodeVisit:
    """State information for a node during a graph traversal (e.g., Dijkstra)."""

    visited: bool
    discovered: bool
    pred: int | None  # In Rust: Option<usize>
    short_dist: float  # In Rust: f32
    simpl_dist: float  # In Rust: f32
    agg_seconds: float  # In Rust: f32
    @classmethod
    def new(cls) -> NodeVisit:  # In Rust: #[new] pub fn new() -> Self
        """Initialize a new NodeVisit state."""
        ...

class StableGraph: ...  # Placeholder for the internal graph representation (petgraph::stable_graph::StableGraph)

class NetworkStructure:
    """Manages the network graph, including nodes, edges, barriers, and spatial indexing."""

    graph: StableGraph  # Actual type is petgraph::stable_graph::StableGraph<NodePayload, EdgePayload>
    is_dual: bool
    is_directed: bool
    edge_rtree: (
        object | None
    )  # R-tree for efficient spatial queries on edges. Type in Rust: Option<RTree<EdgeRtreeItem>>
    # barrier_geoms and barrier_rtree are internal and managed via set/unset methods.
    @classmethod
    def new(cls) -> NetworkStructure:  # In Rust: #[new] pub fn new() -> Self
        """Create a new, empty NetworkStructure."""
        ...
    def progress_init(self) -> None:  # In Rust: pub fn progress_init(&self)
        """Reset the internal progress counter (used for long operations)."""
        ...
    @property
    def progress(self) -> int:  # In Rust: pub fn progress(&self) -> usize
        """Get the current value of the internal progress counter."""
        ...
    @property
    def is_dual(self) -> bool:
        """Whether this network structure was ingested from a dual graph."""
        ...
    def set_is_dual(self, is_dual: bool) -> None:
        """Set whether this network structure represents a dual graph."""
        ...
    @property
    def is_directed(self) -> bool:
        """Whether this network structure represents a directed graph."""
        ...
    def set_is_directed(self, is_directed: bool) -> None:
        """Set whether this network structure represents a directed graph."""
        ...
    def add_street_node(
        self,
        node_key: Any,
        x: float,
        y: float,
        live: bool,
        weight: float,
        z: float | None = None,
    ) -> int:  # Returns usize in Rust
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
        z: float | None
            Optional z-coordinate (elevation). Default None. When z is provided for both endpoints
            of an edge, a slope-based walking impedance (Tobler's hiking function) is automatically
            applied during shortest-path and simplest-path computations.
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
        z: float | None = None,
    ) -> int:  # Returns PyResult<usize> in Rust
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
            Max distance (meters) to search for street nodes to link to (default: 100.0 from Rust).
        z: float | None
            Optional z-coordinate (elevation). Default None.

        Returns
        -------
        int
            The internal index assigned to the transport node.
        """
        ...

    def get_node_payload_py(self, node_idx: int) -> NodePayload:  # Returns PyResult<NodePayload>
        """Retrieve the payload data for a specific node index."""
        ...
    def get_node_weight(self, node_idx: int) -> float:  # Returns PyResult<f32>
        """Get the weight of a specific node index."""
        ...
    def set_node_weight(self, node_idx: int, weight: float) -> None:  # Returns PyResult<()>
        """Set the weight of a specific node index."""
        ...
    def is_node_live(self, node_idx: int) -> bool:  # Returns PyResult<bool>
        """Check if a specific node index is marked as 'live'."""
        ...
    def set_node_live(self, node_idx: int, live: bool) -> None:  # Returns PyResult<()>
        """Set the live status of a node (e.g. based on a boundary polygon)."""
        ...
    def node_count(self) -> int:  # Returns usize
        """Get the total number of nodes in the graph."""
        ...
    def street_node_count(self) -> int:  # Returns usize
        """Get the number of street nodes in the graph."""
        ...
    def node_indices(self) -> list[int]:  # Returns Vec<usize>
        """Get indices for all nodes."""
        ...
    def node_keys_py(self) -> list[Any]:  # In Rust: pub fn node_keys_py(&self, py: Python) -> Vec<Py<PyAny>>
        """Get a list of original keys for all nodes (street and transport)."""
        ...
    def street_node_indices(self) -> list[int]:  # In Rust: pub fn street_node_indices(&self) -> Vec<usize>
        """Get indices for non-transport (street) nodes."""
        ...
    @property
    def node_xs(self) -> list[float]:  # Getter returns Vec<f64>
        """Get x-coordinates for all nodes."""
        ...
    # street_node_xs removed as no direct public getter in Rust
    @property
    def node_ys(self) -> list[float]:  # Getter returns Vec<f64>
        """Get y-coordinates for all nodes."""
        ...
    # street_node_ys removed as no direct public getter in Rust
    @property
    def node_xys(self) -> list[tuple[float, float]]:  # Getter returns Vec<(f64, f64)>
        """Get (x, y) coordinates for all nodes."""
        ...
    @property
    def node_zs(self) -> list[float | None]:  # Getter returns Vec<Option<f64>>
        """Get optional z-coordinates for all nodes."""
        ...
    @property
    def node_xyzs(self) -> list[tuple[float, float, float | None]]:  # Getter returns Vec<(f64, f64, Option<f64>)>
        """Get (x, y, z) coordinates for all nodes, where z may be None."""
        ...
    # street_node_xys removed as no direct public getter in Rust
    @property
    def node_lives(self) -> list[bool]:  # Getter returns Vec<bool>
        """Get 'live' status for all nodes."""
        ...
    @property
    def street_node_lives(self) -> list[bool]:  # Getter returns Vec<bool>
        """Get 'live' status for street nodes only."""
        ...
    @property
    def edge_count(self) -> int:  # Getter returns usize
        """Get the total number of edges in the graph."""
        ...
    def add_street_edge(
        self,
        start_nd_idx: int,  # usize
        end_nd_idx: int,  # usize
        edge_idx: int,  # usize
        start_nd_key_py: Any,
        end_nd_key_py: Any,
        geom_wkt: str,
        imp_factor: float | None = None,
        shared_primal_node_key: str | None = None,
    ) -> int:  # Returns PyResult<usize>
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
        shared_primal_node_key: str | None
            Optional primal junction key for dual-graph transitions.
        imp_factor: float | None
            Impedance multiplier (> 0.0, default 1.0).

        Returns
        -------
        int
            The internal index assigned to the edge.
        """
        ...

    def remove_street_node(self, node_idx: int) -> None:
        """
        Remove a street node and all its connected edges.

        Parameters
        ----------
        node_idx: int
            The internal index of the node to remove.

        Raises
        ------
        ValueError
            If the node does not exist or is a transport node.
        """
        ...
    def remove_street_edge(self, start_nd_idx: int, end_nd_idx: int, edge_idx: int) -> None:
        """
        Remove a specific directed edge.

        Parameters
        ----------
        start_nd_idx: int
            Index of the starting node.
        end_nd_idx: int
            Index of the ending node.
        edge_idx: int
            The external edge identifier to match.

        Raises
        ------
        ValueError
            If no matching edge is found.
        """
        ...
    def add_transport_edge(
        self,
        start_nd_idx: int,  # usize
        end_nd_idx: int,  # usize
        edge_idx: int,  # usize
        start_nd_key_py: Any,
        end_nd_key_py: Any,
        seconds: float,
        imp_factor: float | None = None,
    ) -> int:  # Returns PyResult<usize>
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

    def edge_references(self) -> list[tuple[int, int, int]]:  # Returns Vec<(usize, usize, usize)>
        """Get list of (start_node_idx, end_node_idx, edge_idx) for all edges."""
        ...
    def get_edge_payload_py(
        self, start_nd_idx: int, end_nd_idx: int, edge_idx: int
    ) -> EdgePayload:  # PyResult<EdgePayload>
        """Retrieve the payload for a specific edge defined by nodes and edge_idx."""
        ...
    def get_edge_length(self, start_nd_idx: int, end_nd_idx: int, edge_idx: int) -> float:
        """Get the length of a specific edge."""
        ...
    def get_edge_impedance(self, start_nd_idx: int, end_nd_idx: int, edge_idx: int) -> float:
        """Get the impedance factor of a specific edge."""
        ...
    def validate(self) -> None:  # PyResult<()>
        """Check internal consistency of all nodes and edges in the graph."""
        ...

    def build_edge_rtree(self) -> None:  # PyResult<()>
        """Build or rebuild the R-tree spatial index for street edges. Deduplicates based on geometry."""
        ...
    def set_barriers(self, barriers_wkt: list[str]) -> None:  # PyResult<()>
        """Set impassable barrier geometries (from WKT) and build their R-tree."""
        ...
    def unset_barriers(self) -> None:
        """Remove all barrier geometries and their R-tree."""
        ...
    def dijkstra_tree_shortest(
        self,
        src_idx: int,
        max_seconds: int,
        speed_m_s: float,
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

        Returns
        -------
        tuple[list[int], list[NodeVisit]]
            (List of reachable node indices, List of NodeVisit states for all nodes).
        """
        ...
    def poll_reach_hits(
        self,
        src_idxs: list[int],
        distances: list[int],
        speed_m_s: float,
    ) -> list[list[int]]:
        """
        Count, per distance threshold, how many of the given sources reach each node.

        One bounded Dijkstra per source (parallel, to the largest threshold). Backs the
        sampling pilot: hit counts are binomial in reach. Returns one list of length
        node_bound() per distance, indexed by raw node index.
        """
        ...
    def dijkstra_tree_simplest(
        self,
        src_idx: int,
        max_seconds: int,
        speed_m_s: float,
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

        Returns
        -------
        tuple[list[int], list[NodeVisit]]
            (List of reachable node indices, List of NodeVisit states for all nodes).

        Notes
        -----
        Requires `self.is_dual == True`.
        """
        ...
    def centrality_shortest(
        self,
        distances: list[int] | None = None,
        minutes: list[float] | None = None,
        closeness_exprs: list[tuple[str, str]] | None = None,
        betweenness_exprs: list[tuple[str, str]] | None = None,
        compute_cycles: bool | None = None,
        speed_m_s: float | None = None,
        tolerance: float | None = None,
        segment_weighted: bool | None = None,
        sample_probability: float | None = None,
        sampling_weights: list[float] | None = None,
        random_seed: int | None = None,
        pbar_disabled: bool | None = None,
    ) -> CentralityResult:
        """
        Compute centrality using shortest paths. Expressions use ``c`` (metric
        distance) and ``p`` (normalised progress = c / threshold).

        Parameters
        ----------
        distances: list[int] | None
            Distance thresholds (meters).
        minutes: list[float] | None
            Time thresholds (minutes).
        closeness_exprs: list[tuple[str, str]] | None
            Named closeness expressions: ``[(name, expr), ...]``.
        betweenness_exprs: list[tuple[str, str]] | None
            Named betweenness expressions: ``[(name, expr), ...]``.
        compute_cycles: bool | None
            Compute circuit rank. Default False.
        speed_m_s: float | None
            Travel speed (m/s).
        tolerance: float | None
            Relative tolerance for near-equal path detection in betweenness.
        segment_weighted: bool | None
            Weight by primal edge lengths.
        sample_probability: float | None
            Bernoulli sampling probability with IPW.
        sampling_weights: list[float] | None
            Per-node sampling weights in [0.0, 1.0].
        random_seed: int | None
            Optional seed for reproducible sampling.
        pbar_disabled: bool | None
            Disable progress bar if True.

        Returns
        -------
        CentralityResult
            Object with ``metrics`` dict: ``{name: {distance: array}}``.
        """
        ...
    def centrality_simplest(
        self,
        distances: list[int] | None = None,
        minutes: list[float] | None = None,
        closeness_exprs: list[tuple[str, str]] | None = None,
        betweenness_exprs: list[tuple[str, str]] | None = None,
        speed_m_s: float | None = None,
        tolerance: float | None = None,
        segment_weighted: bool | None = None,
        sample_probability: float | None = None,
        sampling_weights: list[float] | None = None,
        random_seed: int | None = None,
        pbar_disabled: bool | None = None,
    ) -> CentralityResult:
        """
        Compute centrality using simplest (angular) paths. Expressions use ``c``
        (angular cost) and ``p`` (normalised time progress).

        Parameters
        ----------
        distances: list[int] | None
            Distance thresholds (meters).
        minutes: list[float] | None
            Time thresholds (minutes).
        closeness_exprs: list[tuple[str, str]] | None
            Named closeness expressions: ``[(name, expr), ...]``.
        betweenness_exprs: list[tuple[str, str]] | None
            Named betweenness expressions: ``[(name, expr), ...]``.
        speed_m_s: float | None
            Travel speed (m/s).
        tolerance: float | None
            Relative tolerance for near-equal path detection in betweenness.
        segment_weighted: bool | None
            Weight by primal edge lengths.
        sample_probability: float | None
            Bernoulli sampling probability with IPW.
        sampling_weights: list[float] | None
            Per-node sampling weights in [0.0, 1.0].
        random_seed: int | None
            Optional seed for reproducible sampling.
        pbar_disabled: bool | None
            Disable progress bar if True.

        Returns
        -------
        CentralityResult
            Object with ``metrics`` dict: ``{name: {distance: array}}``.

        Notes
        -----
        Requires `self.is_dual == True`.
        """
        ...
    def betweenness_od_shortest(
        self,
        od_matrix: OdMatrix,
        distances: list[int] | None = None,
        minutes: list[float] | None = None,
        betweenness_exprs: list[tuple[str, str]] | None = None,
        speed_m_s: float | None = None,
        tolerance: float | None = None,
        pbar_disabled: bool | None = None,
    ) -> CentralityResult:
        """
        Compute OD-weighted betweenness centrality using shortest paths.

        Parameters
        ----------
        od_matrix: OdMatrix
            Sparse OD weight matrix mapping (origin, destination) pairs to trip weights.
        distances: list[int] | None
            Distance thresholds (meters).
        minutes: list[float] | None
            Time thresholds (minutes).
        betweenness_exprs: list[tuple[str, str]] | None
            Named betweenness expressions: ``[(name, expr), ...]``.
        speed_m_s: float | None
            Travel speed (m/s).
        tolerance: float | None
            Relative tolerance for near-equal path detection in betweenness.
        pbar_disabled: bool | None
            Disable progress bar if True.

        Returns
        -------
        CentralityResult
            Object with ``metrics`` dict: ``{name: {distance: array}}``.
        """
        ...
    def betweenness_demand_shortest(
        self,
        origins: list[tuple[int, float]],
        destinations: list[tuple[int, float]],
        decay_fn: str,
        distances: list[int] | None = None,
        minutes: list[float] | None = None,
        closest_destination: bool = False,
        metric_name: str | None = None,
        speed_m_s: float | None = None,
        tolerance: float | None = None,
        pbar_disabled: bool | None = None,
    ) -> CentralityResult:
        """
        Demand-weighted (flow) betweenness from a singly / origin-constrained spatial interaction model.

        Each origin distributes its full weight across reachable destinations in proportion to
        ``W_d * decay_fn(c)`` and the allocated flows are routed along shortest paths. Origins and
        destinations are aggregated by node first (duplicate-snapped points sum their weights).

        Parameters
        ----------
        origins: list[tuple[int, float]]
            ``(node_idx, weight)`` pairs for demand origins.
        destinations: list[tuple[int, float]]
            ``(node_idx, weight)`` pairs for demand destinations / attractors.
        decay_fn: str
            Distance-decay expression using ``c`` (metric cost) and ``p`` (normalised progress).
        distances: list[int] | None
            Distance thresholds (meters).
        minutes: list[float] | None
            Time thresholds (minutes).
        closest_destination: bool
            If True, route each origin's full weight to its single nearest reachable destination.
        metric_name: str | None
            Output metric name (default ``"demand"``).
        speed_m_s: float | None
            Travel speed (m/s).
        tolerance: float | None
            Relative tolerance for near-equal path detection in betweenness.
        pbar_disabled: bool | None
            Disable progress bar if True.

        Returns
        -------
        CentralityResult
            Object with ``metrics`` dict: ``{name: {distance: array}}``.
        """
        ...
