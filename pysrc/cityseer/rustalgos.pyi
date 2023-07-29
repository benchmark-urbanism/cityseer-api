from __future__ import annotations
from typing import Any
import numpy as np

class Coord:
    x: float
    y: float

    def __init__(self, x: float, y: float) -> None: ...
    def xy(self) -> tuple[float, float]: ...
    def validate(self) -> bool: ...
    def hypot(self, other_coord: Coord) -> float: ...
    def difference(self, other_coord: Coord) -> Coord: ...

def calculate_rotation(point_a: Coord, point_b: Coord) -> float: ...
def calculate_rotation_smallest(vec_a: Coord, vec_b: Coord) -> float: ...
def check_numerical_data(data_arr: list[list[float]]) -> None: ...
def check_categorical_data(data_arr: list[list[int]]) -> None: ...
def distances_from_betas(betas: list[float], min_threshold_wt: float | None = None) -> list[int]: ...
def betas_from_distances(distances: list[int], min_threshold_wt: float | None = None) -> list[float]: ...
def pair_distances_and_betas(
    distances: list[int] | None = None, betas: list[float] | None = None, min_threshold_wt: float | None = None
) -> tuple[list[int], list[float]]: ...
def avg_distances_for_betas(betas: list[float], min_threshold_wt: float | None = None) -> list[float]: ...
def clip_wts_curve(distances: list[int], betas: list[float], spatial_tolerance: int) -> list[float]: ...
def clipped_beta_wt(beta: float, max_curve_wt: float, data_dist: float) -> float: ...

class NodePayload:
    node_key: str
    coord: Coord
    live: bool
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
    node_density: dict[int, np.ndarray]
    node_farness: dict[int, np.ndarray]
    node_cycles: dict[int, np.ndarray]
    node_harmonic: dict[int, np.ndarray]
    node_beta: dict[int, np.ndarray]
    node_betweenness: dict[int, np.ndarray]
    node_betweenness_beta: dict[int, np.ndarray]

class CentralitySimplestResult:
    node_harmonic: dict[int, np.ndarray]
    node_betweenness: dict[int, np.ndarray]

class CentralitySegmentResult:
    segment_density: dict[int, np.ndarray]
    segment_harmonic: dict[int, np.ndarray]
    segment_beta: dict[int, np.ndarray]
    segment_betweenness: dict[int, np.ndarray]

class DiGraph: ...

class NetworkStructure:
    graph: DiGraph
    progress: int
    @classmethod
    def new(cls) -> NetworkStructure: ...
    def progress(self) -> int: ...
    def add_node(self, node_key: str, x: float, y: float, live: bool) -> int: ...
    def get_node_payload(self, node_idx: int) -> NodePayload: ...
    def is_node_live(self, node_idx: int) -> bool: ...
    def node_count(self) -> int: ...
    def node_indices(self) -> list[int]: ...
    @property
    def node_xs(self) -> list[float]: ...
    @property
    def node_ys(self) -> list[float]: ...
    @property
    def node_xys(self) -> list[tuple[float, float]]: ...
    @property
    def node_lives(self) -> list[bool]: ...
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
    ) -> int: ...
    def edge_references(self) -> list[tuple[int, int, int]]: ...
    def get_edge_payload(self, start_nd_idx: int, end_nd_idx: int, edge_idx: int) -> EdgePayload: ...
    def validate(self) -> bool: ...
    def find_nearest(self, data_coord: Any, max_dist: float) -> tuple[int | None, float, int | None]: ...
    def road_distance(self, data_coord: Any, nd_a_idx: int, nd_b_idx: int) -> tuple[float, int | None, int | None]: ...
    def closest_intersections(
        self, data_coord: Any, pred_map: list[int | None], last_nd_idx: int
    ) -> tuple[float, int | None, int | None]: ...
    def assign_to_network(self, data_coord: Any, max_dist: float) -> tuple[int | None, int | None]: ...
    def shortest_path_tree(
        self, src_idx: int, max_dist: int, angular: bool | None = None, jitter_scale: float | None = None
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
    weighted: dict[int, np.ndarray]
    unweighted: dict[int, np.ndarray]

class MixedUsesResult:
    hill: dict[int, dict[int, np.ndarray]] | None
    hill_weighted: dict[int, dict[int, np.ndarray]] | None
    shannon: dict[int, np.ndarray] | None
    gini: dict[int, np.ndarray] | None

class StatsResult:
    sum: dict[int, np.ndarray]
    sum_wt: dict[int, np.ndarray]
    mean: dict[int, np.ndarray]
    mean_wt: dict[int, np.ndarray]
    count: dict[int, np.ndarray]
    count_wt: dict[int, np.ndarray]
    variance: dict[int, np.ndarray]
    variance_wt: dict[int, np.ndarray]
    max: dict[int, np.ndarray]
    min: dict[int, np.ndarray]

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
    def insert(
        self,
        data_key: str,
        x: float,
        y: float,
        data_id: str | None = None,
        nearest_assign: int | None = None,
        next_nearest_assign: int | None = None,
    ) -> None: ...
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
