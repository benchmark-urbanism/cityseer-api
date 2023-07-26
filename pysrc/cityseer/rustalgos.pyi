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

class CloseShortestResult:
    node_density: dict[int, np.ndarray]
    node_farness: dict[int, np.ndarray]
    node_cycles: dict[int, np.ndarray]
    node_harmonic: dict[int, np.ndarray]
    node_beta: dict[int, np.ndarray]

class CloseSimplestResult:
    node_harmonic: dict[int, np.ndarray]

class CloseSegmentShortestResult:
    segment_density: dict[int, np.ndarray]
    segment_harmonic: dict[int, np.ndarray]
    segment_beta: dict[int, np.ndarray]

class BetwShortestResult:
    node_betweenness: dict[int, np.ndarray]
    node_betweenness_beta: dict[int, np.ndarray]

class BetwSimplestResult:
    node_betweenness: dict[int, np.ndarray]

class BetwSegmentShortestResult:
    segment_betweenness: dict[int, np.ndarray]

class DiGraph: ...

class NetworkStructure:
    graph: DiGraph[NodePayload, EdgePayload]
    @classmethod
    def new(cls) -> NetworkStructure: ...
    def add_node(self, node_key: str, x: float, y: float, live: bool) -> int: ...
    def get_node_payload(self, node_idx: int) -> NodePayload | None: ...
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
    def get_edge_payload(self, start_nd_idx: int, end_nd_idx: int, edge_idx: int) -> EdgePayload | None: ...
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
        distances: list[int] | None,
        betas: list[float] | None,
        closeness: bool | None,
        betweenness: bool | None,
        min_threshold_wt: float | None,
        jitter_scale: float | None,
        pbar_disabled: bool | None,
    ) -> tuple[CloseShortestResult | None, BetwShortestResult | None]: ...
    def local_node_centrality_simplest(
        self,
        distances: list[int] | None,
        betas: list[float] | None,
        closeness: bool | None,
        betweenness: bool | None,
        min_threshold_wt: float | None,
        jitter_scale: float | None,
        pbar_disabled: bool | None,
    ) -> tuple[CloseSimplestResult | None, BetwSimplestResult | None]: ...
    def local_segment_centrality_shortest(
        self,
        distances: list[int] | None,
        betas: list[float] | None,
        closeness: bool | None,
        betweenness: bool | None,
        min_threshold_wt: float | None,
        jitter_scale: float | None,
        pbar_disabled: bool | None,
    ) -> tuple[CloseSegmentShortestResult | None, BetwSegmentShortestResult | None]: ...

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
