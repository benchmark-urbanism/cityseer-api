"""Data structures and utilities for mapping and analyzing spatial data relative to a network."""

from __future__ import annotations

from collections.abc import Hashable

import numpy.typing as npt

from .graph import NetworkStructure

class AccessibilityResult:
    """Holds accessibility calculation results (weighted, unweighted counts, nearest distance)."""

    weighted: dict[int, npt.ArrayLike]
    unweighted: dict[int, npt.ArrayLike]
    distance: dict[int, npt.ArrayLike]

class MixedUsesResult:
    """Holds mixed-use diversity calculation results (Hill, Shannon, Gini)."""

    hill: dict[int, dict[int, npt.ArrayLike]] | None
    hill_weighted: dict[int, dict[int, npt.ArrayLike]] | None
    shannon: dict[int, npt.ArrayLike] | None
    gini: dict[int, npt.ArrayLike] | None

class StatsResult:
    """Holds statistical aggregation results for numerical data within catchments."""

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
    ) -> dict[str, AccessibilityResult]:
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
        dict[str, AccessibilityResult]
            Dictionary mapping land use keys to their accessibility results.
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
            Compute Hill diversity (q=0, 1, 2) if True.
        compute_hill_weighted: bool | None
            Compute distance-weighted Hill diversity if True.
        compute_shannon: bool | None
            Compute Shannon diversity if True.
        compute_gini: bool | None
            Compute Gini-Simpson diversity if True.
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
    ) -> list[StatsResult]:
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
        list[StatsResult]
            List containing a StatsResult object for each input numerical map.
        """
        ...
