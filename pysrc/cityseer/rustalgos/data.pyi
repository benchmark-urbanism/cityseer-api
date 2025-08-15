"""Data structures and utilities for mapping and analyzing spatial data relative to a network."""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any

import numpy as np
import numpy.typing as npt

from .graph import NetworkStructure

class LanduseAccess:
    """Holds accessibility calculation results for a specific land use category."""

    @property
    def weighted(self) -> dict[int, npt.NDArray[np.float32]]: ...
    @property
    def unweighted(self) -> dict[int, npt.NDArray[np.float32]]: ...
    @property
    def distance(self) -> dict[int, npt.NDArray[np.float32]]: ...

class AccessibilityResult:
    """Holds overall accessibility calculation results."""

    distances: list[int]
    node_keys_py: list[Any]
    node_indices: list[int]
    @property
    def result(self) -> dict[str, LanduseAccess]: ...

class MixedUsesResult:
    """Holds mixed-use diversity calculation results (Hill, Shannon, Gini)."""

    distances: list[int]
    node_keys_py: list[Any]
    node_indices: list[int]
    @property
    def hill(self) -> dict[int, dict[int, npt.NDArray[np.float32]]]: ...
    @property
    def hill_weighted(self) -> dict[int, dict[int, npt.NDArray[np.float32]]]: ...
    @property
    def shannon(self) -> dict[int, npt.NDArray[np.float32]]: ...
    @property
    def gini(self) -> dict[int, npt.NDArray[np.float32]]: ...

class Stats:
    """Holds statistical aggregation results for a single numerical map."""

    @property
    def sum(self) -> dict[int, npt.NDArray[np.float32]]: ...
    @property
    def sum_wt(self) -> dict[int, npt.NDArray[np.float32]]: ...
    @property
    def mean(self) -> dict[int, npt.NDArray[np.float32]]: ...
    @property
    def mean_wt(self) -> dict[int, npt.NDArray[np.float32]]: ...
    @property
    def median(self) -> dict[int, npt.NDArray[np.float32]]: ...
    @property
    def median_wt(self) -> dict[int, npt.NDArray[np.float32]]: ...
    @property
    def count(self) -> dict[int, npt.NDArray[np.float32]]: ...
    @property
    def count_wt(self) -> dict[int, npt.NDArray[np.float32]]: ...
    @property
    def variance(self) -> dict[int, npt.NDArray[np.float32]]: ...
    @property
    def variance_wt(self) -> dict[int, npt.NDArray[np.float32]]: ...
    @property
    def max(self) -> dict[int, npt.NDArray[np.float32]]: ...
    @property
    def min(self) -> dict[int, npt.NDArray[np.float32]]: ...

class StatsResult:
    """Holds overall statistical aggregation results for multiple numerical maps."""

    distances: list[int]
    node_keys_py: list[Any]
    node_indices: list[int]
    @property
    def result(self) -> list[Stats]: ...

class DataEntry:
    """Represents a single spatial data point with geometry and keys."""

    data_key_py: Hashable
    data_key: str
    dedupe_key_py: Hashable
    dedupe_key: str
    geom_wkt: str

    def __init__(
        self,
        data_key_py: Hashable,
        geom_wkt: str,
        dedupe_key_py: Hashable | None = None,
    ) -> None:
        """
        Initialize a DataEntry.

        Parameters
        ----------
        data_key_py: Hashable
            Unique identifier for the data entry.
        geom_wkt: str
            Geometry in WKT format.
        dedupe_key_py: Hashable | None
            Optional key for deduplication during aggregation. If None, `data_key_py` is used.
        """
        ...

class DataMap:
    """Manages a collection of DataEntry objects and their assignment to a NetworkStructure."""

    entries: dict[str, DataEntry]
    node_data_map: dict[int, list[tuple[str, float]]]

    def __init__(self) -> None: ...
    def progress_init(self) -> None:
        """Reset the internal progress counter."""
        ...
    @property
    def progress(self) -> int:
        """Get the current value of the internal progress counter."""
        ...
    def insert(
        self,
        data_key_py: Hashable,
        geom_wkt: str,
        dedupe_key_py: Hashable | None = None,
    ) -> None:
        """
        Insert a new data entry into the map.

        Parameters
        ----------
        data_key_py: Hashable
            Unique identifier for the data entry.
        geom_wkt: str
            Geometry in WKT format.
        dedupe_key_py: Hashable | None
            Optional key for deduplication. If None, `data_key_py` is used.
        """
        ...
    def entry_keys(self) -> list[str]:
        """Get a list of all data keys (as strings) in the map."""
        ...
    def get_entry(self, data_key: str) -> DataEntry | None:
        """Retrieve a specific DataEntry by its string key."""
        ...
    def count(self) -> int:
        """Return the total number of data entries in the map."""
        ...
    def is_empty(self) -> bool:
        """Check if the map contains any data entries."""
        ...
    def assign_data_to_network(
        self,
        network_structure: NetworkStructure,
        max_assignment_dist: float,
        n_nearest_candidates: int,
    ) -> None:
        """
        Assign data entries to the nearest valid network nodes.

        Considers `max_assignment_dist`, barriers, and street intersections.
        Requires `network_structure.build_edge_rtree()` to be called first.

        Parameters
        ----------
        network_structure: NetworkStructure
            The network to assign data to.
        max_assignment_dist: float
            Maximum distance allowed for assignment.
        n_nearest_candidates: int
            Number of nearest street edge candidates to consider for assignment.
        """
        ...
    def aggregate_to_src_idx(
        self,
        netw_src_idx: int,
        network_structure: NetworkStructure,
        max_walk_seconds: int,
        speed_m_s: float,
        jitter_scale: float | None = None,
        angular: bool | None = None,
    ) -> dict[str, float]:
        """
        Find reachable data entries from a source node within a time limit.

        Performs Dijkstra search (shortest or simplest path) and aggregates reachable data,
        applying deduplication based on `dedupe_key`.

        Parameters
        ----------
        netw_src_idx: int
            The starting node index in the network.
        network_structure: NetworkStructure
            The network structure.
        max_walk_seconds: int
            Maximum travel time in seconds.
        speed_m_s: float
            Travel speed in meters per second.
        jitter_scale: float | None
            Optional scale for adding random jitter to path costs (for tie-breaking).
        angular: bool | None
            If True, use simplest path (angular distance); otherwise, use shortest path (metric distance).

        Returns
        -------
        dict[str, float]
            Dictionary mapping reachable `data_key` strings to their minimum travel distance (meters).
        """
        ...
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
    ) -> AccessibilityResult:
        """
        Calculate accessibility metrics (counts, weighted counts, nearest distance) for specified land uses.

        Aggregates data reachable from each network node within given distance/time thresholds.
        Requires exactly one of `distances`, `betas`, or `minutes`.

        Parameters
        ----------
        network_structure: NetworkStructure
            The network structure.
        landuses_map: dict[Hashable, str]
            Mapping from `data_key_py` to land use category string.
        accessibility_keys: list[str]
            List of land use categories to calculate accessibility for.
        distances: list[int] | None
            Distance thresholds (meters).
        betas: list[float] | None
            Decay parameters (beta).
        minutes: list[float] | None
            Time thresholds (minutes).
        angular: bool | None
            Use simplest path if True.
        spatial_tolerance: int | None
            Spatial uncertainty buffer for weight clipping.
        min_threshold_wt: float | None
            Minimum weight threshold for beta/distance conversion.
        speed_m_s: float | None
            Travel speed (m/s).
        jitter_scale: float | None
            Path cost jitter scale.
        pbar_disabled: bool | None
            Disable progress bar if True.

        Returns
        -------
        AccessibilityResult
            Object containing the accessibility metrics. Access detailed results via its `result` attribute.
        """
        ...
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
    ) -> MixedUsesResult:
        """
        Calculate mixed-use diversity metrics (Hill, Shannon, Gini) based on reachable land uses.

        Aggregates land use counts within catchments and computes selected diversity indices.
        Requires exactly one of `distances`, `betas`, or `minutes`.

        Parameters
        ----------
        network_structure: NetworkStructure
            The network structure.
        landuses_map: dict[Hashable, str]
            Mapping from `data_key_py` to land use category string.
        distances: list[int] | None
            Distance thresholds (meters).
        betas: list[float] | None
            Decay parameters (beta).
        minutes: list[float] | None
            Time thresholds (minutes).
        compute_hill: bool | None
            Compute Hill diversity (q=0, 1, 2) if True. Default True.
        compute_hill_weighted: bool | None
            Compute distance-weighted Hill diversity if True. Default True.
        compute_shannon: bool | None
            Compute Shannon diversity if True. Default False.
        compute_gini: bool | None
            Compute Gini-Simpson diversity if True. Default False.
        angular: bool | None
            Use simplest path if True.
        spatial_tolerance: int | None
            Spatial uncertainty buffer for weight clipping.
        min_threshold_wt: float | None
            Minimum weight threshold for beta/distance conversion.
        speed_m_s: float | None
            Travel speed (m/s).
        jitter_scale: float | None
            Path cost jitter scale.
        pbar_disabled: bool | None
            Disable progress bar if True.

        Returns
        -------
        MixedUsesResult
            Object containing the calculated diversity metrics.
        """
        ...
    def stats(
        self,
        network_structure: NetworkStructure,
        numerical_maps: list[dict[Hashable, float]],
        distances: list[int] | None = None,
        betas: list[float] | None = None,
        minutes: list[float] | None = None,
        angular: bool | None = None,
        spatial_tolerance: int | None = None,
        min_threshold_wt: float | None = None,
        speed_m_s: float | None = None,
        jitter_scale: float | None = None,
        pbar_disabled: bool | None = None,
    ) -> StatsResult:
        """
        Calculate statistics (sum, mean, count, variance, min, max) for numerical data within catchments.

        Aggregates numerical values associated with reachable data entries.
        Requires exactly one of `distances`, `betas`, or `minutes`.

        Parameters
        ----------
        network_structure: NetworkStructure
            The network structure.
        numerical_maps: list[dict[Hashable, float]]
            List of dictionaries, each mapping `data_key_py` to a numerical value.
        distances: list[int] | None
            Distance thresholds (meters).
        betas: list[float] | None
            Decay parameters (beta).
        minutes: list[float] | None
            Time thresholds (minutes).
        angular: bool | None
            Use simplest path if True.
        spatial_tolerance: int | None
            Spatial uncertainty buffer for weight clipping.
        min_threshold_wt: float | None
            Minimum weight threshold for beta/distance conversion.
        speed_m_s: float | None
            Travel speed (m/s).
        jitter_scale: float | None
            Path cost jitter scale.
        pbar_disabled: bool | None
            Disable progress bar if True.

        Returns
        -------
        StatsResult
            Object containing the statistical results. Access detailed results for each input map via its `result`
            attribute.
        """
        ...
