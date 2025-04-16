"""Data structures and utilities."""

from __future__ import annotations

from collections.abc import Hashable

import numpy.typing as npt

from . import Coord
from .graph import NetworkStructure

class ClassesState:
    count: int
    nearest: float

class DataEntry:
    python_key: Hashable
    data_key: str
    coord: Coord
    data_id: str | None
    node_matches: NodeMatches | None

    def __init__(
        self,
        python_key: Hashable,
        x: float,
        y: float,
        data_id: str | None = None,
        node_matches: NodeMatches | None = None,
    ) -> None: ...

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

class NodeMatch:
    idx: int
    dist: float

class NodeMatches:
    nearest: NodeMatch | None
    next_nearest: NodeMatch | None

class DataMap:
    entries: dict[str, DataEntry]
    assigned_to_network: bool
    def __init__(self) -> None: ...
    def progress_init(self) -> None: ...
    def progress(self) -> int: ...
    def insert(
        self,
        python_key: Hashable,
        x: float,
        y: float,
        data_id: str | None = None,
        node_matches: NodeMatches | None = None,
    ) -> None: ...
    def entry_keys(self) -> list[str]: ...
    def get_entry(self, data_key: str) -> DataEntry | None: ...
    def get_data_coord(self, data_key: str) -> Coord | None: ...
    def count(self) -> int: ...
    def is_empty(self) -> bool: ...
    def assign_to_network(
        self,
        network_structure: NetworkStructure,
        max_dist: float,
        pbar_disabled: bool | None = None,
    ) -> None: ...
    def aggregate_to_src_idx(
        self,
        netw_src_idx: int,
        network_structure: NetworkStructure,
        max_walk_seconds: float,
        speed_m_s: float,
        jitter_scale: float | None = None,
        angular: bool | None = None,
    ) -> dict[str, float]: ...
    def accessibility(
        self,
        network_structure: NetworkStructure,
        landuses_map: dict[Hashable, str],
        accessibility_keys: list[str],
        distances: list[int] | None = None,
        betas: list[float] | None = None,
        minutes: list[float] | None = None,
        angular: bool | None = None,
        spatial_tolerance: int | None = None,
        min_threshold_wt: float | None = None,
        speed_m_s: float | None = None,
        jitter_scale: float | None = None,
        pbar_disabled: bool | None = None,
    ) -> dict[str, AccessibilityResult]: ...
    def mixed_uses(
        self,
        network_structure: NetworkStructure,
        landuses_map: dict[Hashable, str],
        distances: list[int] | None = None,
        betas: list[float] | None = None,
        minutes: list[float] | None = None,
        compute_hill: bool | None = True,
        compute_hill_weighted: bool | None = True,
        compute_shannon: bool | None = False,
        compute_gini: bool | None = False,
        angular: bool | None = None,
        spatial_tolerance: int | None = None,
        min_threshold_wt: float | None = None,
        speed_m_s: float | None = None,
        jitter_scale: float | None = None,
        pbar_disabled: bool | None = None,
    ) -> MixedUsesResult: ...
    def stats(
        self,
        network_structure: NetworkStructure,
        numerical_maps: list[dict[Hashable, str]],
        distances: list[int] | None = None,
        betas: list[float] | None = None,
        minutes: list[float] | None = None,
        angular: bool | None = None,
        spatial_tolerance: int | None = None,
        min_threshold_wt: float | None = None,
        speed_m_s: float | None = None,
        jitter_scale: float | None = None,
        pbar_disabled: bool | None = None,
    ) -> list[StatsResult]: ...
