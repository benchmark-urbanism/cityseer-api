"""Graph data structures and utilities for network analysis, including pathfinding and centrality."""

from __future__ import annotations

from typing import Any

from .centrality import CentralitySegmentResult, CentralityShortestResult, CentralitySimplestResult

__doc__: str

class NodePayload:
    """Payload data associated with a network node."""

    node_key: Any  # In Rust: Py<PyAny>
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
    cycles: float  # In Rust: f32
    origin_seg: int | None  # In Rust: Option<usize>
    last_seg: int | None  # In Rust: Option<usize>
    out_bearing: float  # In Rust: f32
    agg_seconds: float  # In Rust: f32
    @classmethod
    def new(cls) -> NodeVisit:  # In Rust: #[new] pub fn new() -> Self
        """Initialize a new NodeVisit state."""
        ...

class EdgeVisit:
    """State information for an edge during a graph traversal."""

    visited: bool
    start_nd_idx: int | None  # In Rust: Option<usize>
    end_nd_idx: int | None  # In Rust: Option<usize>
    edge_idx: int | None  # In Rust: Option<usize>
    @classmethod
    def new(cls) -> EdgeVisit:  # In Rust: #[new] pub fn new() -> Self
        """Initialize a new EdgeVisit state."""
        ...

class DiGraph: ...  # Placeholder for the internal graph representation (petgraph::graph::DiGraph)

class NetworkStructure:
    """Manages the network graph, including nodes, edges, barriers, and spatial indexing."""

    graph: DiGraph  # Actual type is petgraph::graph::DiGraph<NodePayload, EdgePayload>
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
    def add_street_node(
        self, node_key: Any, x: float, y: float, live: bool, weight: float
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
    def is_node_live(self, node_idx: int) -> bool:  # Returns PyResult<bool>
        """Check if a specific node index is marked as 'live'."""
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
        jitter_scale: float | None = None,
        random_seed: int | None = None,
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
        random_seed: int | None
            Optional seed for deterministic random cost jitter.

        Returns
        -------
        tuple[list[int], list[NodeVisit]]
            (List of reachable node indices, List of NodeVisit states for all nodes).
        """
        ...
    def dijkstra_tree_simplest(
        self,
        src_idx: int,
        max_seconds: int,
        speed_m_s: float,
        jitter_scale: float | None = None,
        random_seed: int | None = None,
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
        random_seed: int | None
            Optional seed for deterministic random cost jitter.

        Returns
        -------
        tuple[list[int], list[NodeVisit]]
            (List of reachable node indices, List of NodeVisit states for all nodes).
        """
        ...
    def dijkstra_tree_segment(
        self,
        src_idx: int,
        max_seconds: int,
        speed_m_s: float,
        jitter_scale: float | None = None,
        random_seed: int | None = None,
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
        random_seed: int | None
            Optional seed for deterministic random cost jitter.

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
        sample_probability: float | None = None,
        sampling_weights: list[float] | None = None,
        random_seed: int | None = None,
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
        sample_probability: float | None
            Probability of sampling a node as a source for centrality calculations.
        sampling_weights: list[float] | None
            Per-node sampling weights in range [0.0, 1.0]. When provided, sampling probability
            for each node becomes sample_probability * sampling_weights[node_idx].
        random_seed: int | None
            Optional seed for reproducible sampling and jitter.
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
        sample_probability: float | None = None,
        sampling_weights: list[float] | None = None,
        random_seed: int | None = None,
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
        sample_probability: float | None
            Probability of sampling a node as a source for centrality calculations.
        sampling_weights: list[float] | None
            Per-node sampling weights in range [0.0, 1.0]. When provided, sampling probability
            for each node becomes sample_probability * sampling_weights[node_idx].
        random_seed: int | None
            Optional seed for reproducible sampling and jitter.
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
        random_seed: int | None = None,
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
        random_seed: int | None
            Optional seed for random cost jitter.
        pbar_disabled: bool | None
            Disable progress bar if True.

        Returns
        -------
        CentralitySegmentResult
            Object containing calculated segment centrality metrics.
        """
        ...
