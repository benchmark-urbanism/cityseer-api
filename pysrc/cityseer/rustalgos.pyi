"""Rust based algorithms used from cityseer."""

# pyright: basic
# pylint: disable=unused-argument,missing-function-docstring,unnecessary-ellipsis,missing-class-docstring

from __future__ import annotations

from typing import Any

import numpy.typing as npt

class Coord:
    """Class representing a coordinate."""

    x: float
    y: float

    def __init__(self, x: float, y: float) -> None:
        """
        Creates a `Coord` with `x` and `y` coordinates.

        Parameters
        ----------
        `x`: x coordinate.
        `y`: y coordinate.
        """
        ...

    def xy(self) -> tuple[float, float]:
        """
        Returns the `Coord` as a `tuple` of `x` and `y`.

        Returns
        -------
        `xy`: tuple[float, float]
        """
        ...

    def validate(self) -> bool:
        """Validates the Coord."""
        ...

    def hypot(self, other_coord: Coord) -> float:
        """
        Returns the pythagorean distance from this `Coord` to another.

        Parameters
        ----------
        `other_coord`: Coord
            The other coordinate to which to compute the Pythagorean distance.
        """
        ...

    def difference(self, other_coord: Coord) -> Coord:
        """
        Returns the vector of the spatial difference between this `Coord` and another.

        Parameters
        ----------
        `other_coord`: Coord
            The other coordinate to which to compute the Pythagorean distance.
        """
        ...

def calculate_rotation(point_a: Coord, point_b: Coord) -> float: ...
def calculate_rotation_smallest(vec_a: Coord, vec_b: Coord) -> float:
    """
    Calculates the angle between `vec_a` and `vec_b`.

    Parameters
    ----------
    `vec_a`: Coord
        The vector of `vec_a`.
    `vec_b`: Coord
        The vector of `vec_b`.

    Returns
    -------
    """
    ...

def check_numerical_data(data_arr: list[float]) -> None:
    """
    Checks the integrity of a numerical data array.
    data_arr: list[float]
    """
    ...

def distances_from_betas(betas: list[float], min_threshold_wt: float | None = None) -> list[int]:
    r"""
    Map distance thresholds $d_{max}$ to equivalent decay parameters $\beta$ at the specified cutoff weight $w_{min}$.

    See [`distance_from_beta`](#distance-from-beta) for additional discussion.

    :::note
    It is generally not necessary to utilise this function directly.
    :::

    Parameters
    ----------
    distance: int | tuple[int]
        $d_{max}$ value/s to convert to decay parameters $\beta$.
    min_threshold_wt: float
        The cutoff weight $w_{min}$ on which to model the decay parameters $\beta$.

    Returns
    -------
    tuple[float]
        A numpy array of decay parameters $\beta$.

    Examples
    --------
    ```python
    from cityseer.metrics import networks
    # a list of betas
    distances = [400, 200]
    # convert to betas
    betas = networks.beta_from_distance(distances)
    print(betas)  # prints: array([0.01, 0.02])
    ```

    Most `networks` module methods can be invoked with either `distances` or `betas` parameters, but not both. If using
    the `distances` parameter, then this function will be called in order to extrapolate the decay parameters
    implicitly, using:

    $$\beta = -\frac{log(w_{min})}{d_{max}}$$

    The default `min_threshold_wt` of $w_{min}=0.01831563888873418$ yields conveniently rounded $\beta$ parameters, for
    example:

    | $d_{max}$ | $\beta$ |
    |:---------:|:-------:|
    | 200m | 0.02 |
    | 400m | 0.01 |
    | 800m | 0.005 |
    | 1600m | 0.0025 |

    """
    ...

def betas_from_distances(distances: list[int], min_threshold_wt: float | None = None) -> list[float]:
    r"""
    Map decay parameters $\beta$ to equivalent distance thresholds $d_{max}$ at the specified cutoff weight $w_{min}$.

    :::note
    It is generally not necessary to utilise this function directly.
    :::

    Parameters
    ----------
    betas: list[float]
        $\beta$ value/s to convert to distance thresholds $d_{max}$.
    min_threshold_wt: float | None
        An optional cutoff weight $w_{min}$ at which to set the distance threshold $d_{max}$.

    Returns
    -------
    distances: list[int]
        A list of distance thresholds $d_{max}$.

    Examples
    --------
    ```python
    from cityseer import rustalgos
    # a list of betas
    betas = [0.01, 0.02]
    # convert to distance thresholds
    d_max = rustalgos.distances_from_betas(betas)
    print(d_max)
    # prints: [400, 200]
    ```

    Weighted measures such as the gravity index, weighted betweenness, and weighted land-use accessibilities are
    computed using a negative exponential decay function in the form of:

    $$weight = exp(-\beta \cdot distance)$$

    The strength of the decay is controlled by the $\beta$ parameter, which reflects a decreasing willingness to walk
    correspondingly farther distances. For example, if $\beta=0.005$ were to represent a person's willingness to walk
    to a bus stop, then a location 100m distant would be weighted at 60% and a location 400m away would be weighted at
    13.5%. After an initially rapid decrease, the weightings decay ever more gradually in perpetuity; thus, once a
    sufficiently small weight is encountered it becomes computationally expensive to consider locations any farther
    away. The minimum weight at which this cutoff occurs is represented by $w_{min}$, and the corresponding maximum
    distance threshold by $d_{max}$.

    ![Example beta decays](/images/betas.png)

    Most `networks` module methods can be invoked with either `distances` or `betas` parameters, but not both. If using
    the `betas` parameter, then this function will be called in order to extrapolate the distance thresholds implicitly,
    using:

    $$d_{max} = \frac{log(w_{min})}{-\beta}$$

    The default `min_threshold_wt` of $w_{min}=0.01831563888873418$ yields conveniently rounded $d_{max}$ walking
    thresholds, for example:

    | $\beta$ | $d_{max}$ |
    |:-------:|:---------:|
    | 0.02 | 200m |
    | 0.01 | 400m |
    | 0.005 | 800m |
    | 0.0025 | 1600m |

    Overriding the default $w_{min}$ will adjust the $d_{max}$ accordingly, for example:

    | $\beta$ | $w_{min}$ | $d_{max}$ |
    |:-------:|:---------:|:---------:|
    | 0.02 | 0.01 | 230m |
    | 0.01 | 0.01 | 461m |
    | 0.005 | 0.01 | 921m |
    | 0.0025 | 0.01 | 1842m |

    """
    ...

def pair_distances_and_betas(
    distances: list[int] | None = None, betas: list[float] | None = None, min_threshold_wt: float | None = None
) -> tuple[list[int], list[float]]:
    r"""
    Pair distances and betas, where one or the other parameter is provided.

    Parameters
    ----------
    distances: list[int] | tuple[int]
        Distances corresponding to the local $d_{max}$ thresholds to be used for calculations. The $\beta$ parameters
        (for distance-weighted metrics) will be determined implicitly. If the `distances` parameter is not provided,
        then the `beta` parameter must be provided instead.
    betas: tuple[float]
        A $\beta$, or array of $\beta$ to be used for the exponential decay function for weighted metrics. The
        `distance` parameters for unweighted metrics will be determined implicitly. If the `betas` parameter is not
        provided, then the `distance` parameter must be provided instead.
    min_threshold_wt: float
        The default `min_threshold_wt` parameter can be overridden to generate custom mappings between the
        `distance` and `beta` parameters. See [`distance_from_beta`](#distance-from-beta) for more information.

    Returns
    -------
    distances: tuple[int]
        Distances corresponding to the local $d_{max}$ thresholds to be used for calculations. The $\beta$ parameters
        (for distance-weighted metrics) will be determined implicitly. If the `distances` parameter is not provided,
        then the `beta` parameter must be provided instead.
    betas: tuple[float]
        A $\beta$, or array of $\beta$ to be used for the exponential decay function for weighted metrics. The
        `distance` parameters for unweighted metrics will be determined implicitly. If the `betas` parameter is not
        provided, then the `distance` parameter must be provided instead.

    Examples
    --------
    :::warning
    Networks should be buffered according to the largest distance threshold that will be used for analysis. This
    protects nodes near network boundaries from edge falloffs. Nodes outside the area of interest but within these
    buffered extents should be set to 'dead' so that centralities or other forms of measures are not calculated.
    Whereas metrics are not calculated for 'dead' nodes, they can still be traversed by network analysis algorithms
    when calculating shortest paths and landuse accessibilities.
    :::

    """
    ...

def avg_distances_for_betas(betas: list[float], min_threshold_wt: float | None = None) -> list[float]:
    r"""
    Calculate the mean distance for a given $\beta$ parameter.

    Parameters
    ----------
    beta: tuple[float]
        $\beta$ representing a spatial impedance / distance decay for which to compute the average walking distance.
    min_threshold_wt: float
        The cutoff weight $w_{min}$ on which to model the decay parameters $\beta$.

    Returns
    -------
    tuple[float]
        The average walking distance for a given $\beta$.

    Examples
    --------
    ```python
    from cityseer.metrics import networks
    import numpy as np

    distances = [100, 200, 400, 800, 1600]
    print('distances', distances)
    # distances [ 100  200  400  800 1600]

    betas = networks.beta_from_distance(distances)
    print('betas', betas)
    # betas [0.04   0.02   0.01   0.005  0.0025]

    print('avg', networks.avg_distance_for_beta(betas))
    # avg [ 35.11949  70.23898 140.47797 280.95593 561.91187]
    ```

    """
    ...

def clip_wts_curve(distances: list[int], betas: list[float], spatial_tolerance: int) -> list[float]:
    r"""
    Calculate the upper bounds for clipping weights produced by spatial impedance functions.

    Determine the upper weights threshold of the distance decay curve for a given $\beta$ based on the
    `spatial_tolerance` parameter. This is used by downstream functions to determine the upper extent at which weights
    derived for spatial impedance functions are flattened and normalised. This functionality is only intended for
    situations where the location of datapoints is uncertain for a given spatial tolerance.

    :::warning
    Use distance based clipping with caution for smaller distance thresholds. For example, if using a 200m distance
    threshold clipped by 100m, then substantial distortion is introduced by the process of clipping and normalising the
    distance decay curve. More generally, smaller distance thresholds should generally be avoided for situations where
    datapoints are not located with high spatial precision.
    :::

    Parameters
    ----------
    distances: tuple[int]
        An array of distances corresponding to the local $d_{max}$ thresholds to be used for calculations.
    betas: tuple[float]
        An array of $\beta$ to be used for the exponential decay function for weighted metrics.
    spatial_tolerance: int
        The spatial buffer distance corresponding to the tolerance for spatial inaccuracy.

    Returns
    -------
    max_curve_wts: tuple[float]
        An array of maximum weights at which curves for corresponding $\beta$ will be clipped.

    """
    ...

def clipped_beta_wt(beta: float, max_curve_wt: float, data_dist: float) -> float: ...

class NodePayload:
    node_key: str
    coord: Coord
    live: bool
    weight: float
    def validate(self) -> bool: ...

class EdgePayload:
    start_nd_key: str
    end_nd_key: str
    edge_idx: int
    length: float
    angle_sum: float
    imp_factor: float
    in_bearing: float
    out_bearing: float
    def validate(self) -> bool: ...

class NodeVisit:
    visited: bool
    pred: int | None
    short_dist: float
    simpl_dist: float
    cycles: float
    origin_seg: int | None
    last_seg: int | None
    out_bearing: float
    @classmethod
    def new(cls) -> NodeVisit: ...

class EdgeVisit:
    visited: bool
    start_nd_idx: int | None
    end_nd_idx: int | None
    edge_idx: int | None
    @classmethod
    def new(cls) -> EdgeVisit: ...

class CentralityShortestResult:
    node_density: dict[int, npt.ArrayLike]
    node_farness: dict[int, npt.ArrayLike]
    node_cycles: dict[int, npt.ArrayLike]
    node_harmonic: dict[int, npt.ArrayLike]
    node_beta: dict[int, npt.ArrayLike]
    node_betweenness: dict[int, npt.ArrayLike]
    node_betweenness_beta: dict[int, npt.ArrayLike]

class CentralitySimplestResult:
    node_density: dict[int, npt.ArrayLike]
    node_farness: dict[int, npt.ArrayLike]
    node_harmonic: dict[int, npt.ArrayLike]
    node_betweenness: dict[int, npt.ArrayLike]

class CentralitySegmentResult:
    segment_density: dict[int, npt.ArrayLike]
    segment_harmonic: dict[int, npt.ArrayLike]
    segment_beta: dict[int, npt.ArrayLike]
    segment_betweenness: dict[int, npt.ArrayLike]

class DiGraph: ...  # pylint: disable=multiple-statements

class NetworkStructure:
    # pylint: disable=too-many-public-methods
    graph: DiGraph
    @classmethod
    def new(cls) -> NetworkStructure: ...
    def progress(self) -> int: ...
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
        length: float,
        angle_sum: float,
        imp_factor: float,
        in_bearing: float,
        out_bearing: float,
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
            the impedance of an edge. This is ordinarily set to 1, therefore not impacting calculations. By setting
            this to greater or less than 1, the edge will have a correspondingly higher or lower impedance. This can
            be used to take considerations such as street gradients into account, but should be used with caution.
        in_bearing: float
            The edge's inwards angular bearing.
        out_bearing: float
            The edge's outwards angular bearing.

        """
        ...

    def edge_references(self) -> list[tuple[int, int, int]]: ...
    def get_edge_payload(self, start_nd_idx: int, end_nd_idx: int, edge_idx: int) -> EdgePayload: ...
    def validate(self) -> bool:
        """Validate Network Structure."""
        ...

    def find_nearest(self, data_coord: Any, max_dist: float) -> tuple[int | None, float, int | None]: ...
    def road_distance(self, data_coord: Any, nd_a_idx: int, nd_b_idx: int) -> tuple[float, int | None, int | None]: ...
    def closest_intersections(
        self, data_coord: Any, pred_map: list[int | None], last_nd_idx: int
    ) -> tuple[float, int | None, int | None]: ...
    def assign_to_network(self, data_coord: Any, max_dist: float) -> tuple[int | None, int | None]: ...
    def dijkstra_tree_shortest(
        self, src_idx: int, max_dist: int, jitter_scale: float | None = None
    ) -> tuple[list[int], list[NodeVisit]]: ...
    def dijkstra_tree_simplest(
        self, src_idx: int, max_dist: int, jitter_scale: float | None = None
    ) -> tuple[list[int], list[NodeVisit]]: ...
    def dijkstra_tree_segment(
        self, src_idx: int, max_dist: int, jitter_scale: float | None = None
    ) -> tuple[list[int], list[int], list[NodeVisit], list[EdgeVisit]]: ...
    def local_node_centrality_shortest(
        self,
        distances: list[int] | None = None,
        betas: list[float] | None = None,
        compute_closeness: bool | None = True,
        compute_betweenness: bool | None = True,
        min_threshold_wt: float | None = None,
        jitter_scale: float | None = None,
        pbar_disabled: bool | None = None,
    ) -> CentralityShortestResult: ...
    def local_node_centrality_simplest(
        self,
        distances: list[int] | None = None,
        betas: list[float] | None = None,
        compute_closeness: bool | None = True,
        compute_betweenness: bool | None = True,
        min_threshold_wt: float | None = None,
        jitter_scale: float | None = None,
        pbar_disabled: bool | None = None,
    ) -> CentralitySimplestResult: ...
    def local_segment_centrality(
        self,
        distances: list[int] | None = None,
        betas: list[float] | None = None,
        compute_closeness: bool | None = True,
        compute_betweenness: bool | None = True,
        min_threshold_wt: float | None = None,
        jitter_scale: float | None = None,
        pbar_disabled: bool | None = None,
    ) -> CentralitySegmentResult: ...

def hill_diversity(class_counts: list[int], q: float) -> float: ...
def hill_diversity_branch_distance_wt(
    class_counts: list[int], class_distances: list[float], q: float, beta: float, max_curve_wt: float
) -> float: ...
def hill_diversity_pairwise_distance_wt(
    class_counts: list[int], class_distances: list[float], q: float, beta: float, max_curve_wt: float
) -> float: ...
def gini_simpson_diversity(class_counts: list[int]) -> float: ...
def shannon_diversity(class_counts: list[int]) -> float: ...
def raos_quadratic_diversity(
    class_counts: list[int], wt_matrix: list[list[float]], alpha: float, beta: float
) -> float: ...

class AccessibilityResult:
    weighted: dict[int, npt.ArrayLike]
    unweighted: dict[int, npt.ArrayLike]
    distance: dict[int, npt.ArrayLike]

class MixedUsesResult:
    hill: dict[int, dict[int, npt.ArrayLike]] | None
    hill_weighted: dict[int, dict[int, npt.ArrayLike]] | None
    shannon: dict[int, npt.ArrayLike] | None
    gini: dict[int, npt.ArrayLike] | None

class StatsResult:
    sum: dict[int, npt.ArrayLike]
    sum_wt: dict[int, npt.ArrayLike]
    mean: dict[int, npt.ArrayLike]
    mean_wt: dict[int, npt.ArrayLike]
    count: dict[int, npt.ArrayLike]
    count_wt: dict[int, npt.ArrayLike]
    variance: dict[int, npt.ArrayLike]
    variance_wt: dict[int, npt.ArrayLike]
    max: dict[int, npt.ArrayLike]
    min: dict[int, npt.ArrayLike]

class ClassesState:
    count: int
    nearest: float

class DataEntry:
    data_key: str
    coord: Coord
    data_id: str | None
    nearest_assign: int | None
    next_nearest_assign: int | None

    def __init__(
        self,
        data_key: str,
        x: float,
        y: float,
        data_id: str | None = None,
        nearest_assign: int | None = None,
        next_nearest_assign: int | None = None,
    ) -> None: ...
    def is_assigned(self) -> bool: ...

class DataMap:
    entries: dict[str, DataEntry]
    def __init__(self) -> None: ...
    def progress(self) -> int: ...
    def insert(
        self,
        data_key: str,
        x: float,
        y: float,
        data_id: str | None = None,
        nearest_assign: int | None = None,
        next_nearest_assign: int | None = None,
    ) -> None:
        """
        data_key: str
            The key for the added node.
        data_x: float
            The x coordinate for the added node.
        data_y: float
            The y coordinate for the added node.
        data_id: str | None
            An optional key for each datapoint. Used for deduplication.
        """
        ...

    def entry_keys(self) -> list[str]: ...
    def get_entry(self, data_key: str) -> DataEntry | None: ...
    def get_data_coord(self, data_key: str) -> Coord | None: ...
    def count(self) -> int: ...
    def is_empty(self) -> bool: ...
    def all_assigned(self) -> bool: ...
    def none_assigned(self) -> bool: ...
    def set_nearest_assign(self, data_key: str, assign_idx: int) -> None: ...
    def set_next_nearest_assign(self, data_key: str, assign_idx: int) -> None: ...
    def aggregate_to_src_idx(
        self,
        netw_src_idx: int,
        network_structure: NetworkStructure,
        max_dist: int,
        jitter_scale: float | None = None,
        angular: bool | None = None,
    ) -> dict[str, float]: ...
    def accessibility(
        self,
        network_structure: NetworkStructure,
        landuses_map: dict[str, str],
        accessibility_keys: list[str],
        distances: list[int] | None = None,
        betas: list[float] | None = None,
        angular: bool | None = None,
        spatial_tolerance: int | None = None,
        min_threshold_wt: float | None = None,
        jitter_scale: float | None = None,
        pbar_disabled: bool | None = None,
    ) -> dict[str, AccessibilityResult]: ...
    def mixed_uses(
        self,
        network_structure: NetworkStructure,
        landuses_map: dict[str, str],
        compute_hill: bool | None = True,
        compute_hill_weighted: bool | None = True,
        compute_shannon: bool | None = False,
        compute_gini: bool | None = False,
        distances: list[int] | None = None,
        betas: list[float] | None = None,
        angular: bool | None = None,
        spatial_tolerance: int | None = None,
        min_threshold_wt: float | None = None,
        jitter_scale: float | None = None,
        pbar_disabled: bool | None = None,
    ) -> MixedUsesResult: ...
    def stats(
        self,
        network_structure: NetworkStructure,
        numerical_map: dict[str, float],
        distances: list[int] | None = None,
        betas: list[float] | None = None,
        angular: bool | None = None,
        spatial_tolerance: int | None = None,
        min_threshold_wt: float | None = None,
        jitter_scale: float | None = None,
        pbar_disabled: bool | None = None,
    ) -> StatsResult: ...

class Viewshed:
    @classmethod
    def new(cls) -> Viewshed: ...
    def progress(self) -> int: ...
    def visibility_graph(
        self, bldgs_rast: npt.ArrayLike, view_distance: int, pbar_disabled: bool = False
    ) -> tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]: ...
    def viewshed(
        self, bldgs_rast: npt.ArrayLike, view_distance: int, origin_x: int, origin_y: int, pbar_disabled: bool = False
    ) -> npt.ArrayLike: ...
