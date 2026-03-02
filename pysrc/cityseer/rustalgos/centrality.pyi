"""Network centrality calculation results and OD matrix."""

from __future__ import annotations

from typing import Any

import numpy as np  # For np.float32
import numpy.typing as npt

class OdMatrix:
    """Sparse origin-destination weight matrix for OD-weighted centrality.

    Constructed from parallel arrays of origin node indices, destination node indices,
    and trip weights (COO sparse format). Can be reused across multiple centrality calls.
    """

    def __init__(
        self,
        origins: list[int],
        destinations: list[int],
        weights: list[float],
    ) -> None: ...
    def len(self) -> int:
        """Number of non-zero OD pairs."""
        ...
    def n_origins(self) -> int:
        """Number of unique origin nodes."""
        ...

class CentralityShortestResult:
    """Holds combined closeness + betweenness results for shortest path centrality."""

    distances: list[int]
    node_keys_py: list[Any]
    node_indices: list[int]

    @property
    def node_density(self) -> dict[int, npt.NDArray[np.float32]]: ...
    @property
    def node_farness(self) -> dict[int, npt.NDArray[np.float32]]: ...
    @property
    def node_cycles(self) -> dict[int, npt.NDArray[np.float32]]: ...
    @property
    def node_harmonic(self) -> dict[int, npt.NDArray[np.float32]]: ...
    @property
    def node_beta(self) -> dict[int, npt.NDArray[np.float32]]: ...
    @property
    def node_betweenness(self) -> dict[int, npt.NDArray[np.float32]]: ...
    @property
    def node_betweenness_beta(self) -> dict[int, npt.NDArray[np.float32]]: ...
    @property
    def reachability_totals(self) -> list[int]:
        """Total reachability counts per distance from sampled sources."""
        ...
    @property
    def sampled_source_count(self) -> int:
        """Number of sources that were sampled."""
        ...

class CentralitySimplestResult:
    """Holds combined closeness + betweenness results for simplest (angular) path centrality."""

    distances: list[int]
    node_keys_py: list[Any]
    node_indices: list[int]

    @property
    def node_density(self) -> dict[int, npt.NDArray[np.float32]]: ...
    @property
    def node_farness(self) -> dict[int, npt.NDArray[np.float32]]: ...
    @property
    def node_harmonic(self) -> dict[int, npt.NDArray[np.float32]]: ...
    @property
    def node_betweenness(self) -> dict[int, npt.NDArray[np.float32]]: ...
    @property
    def node_betweenness_beta(self) -> dict[int, npt.NDArray[np.float32]]: ...
    @property
    def reachability_totals(self) -> list[int]:
        """Total reachability counts per distance from sampled sources."""
        ...
    @property
    def sampled_source_count(self) -> int:
        """Number of sources that were sampled."""
        ...

class BetweennessShortestResult:
    """Holds results for shortest path betweenness centrality calculations."""

    distances: list[int]
    node_keys_py: list[Any]
    node_indices: list[int]

    @property
    def node_betweenness(self) -> dict[int, npt.NDArray[np.float32]]: ...
    @property
    def node_betweenness_beta(self) -> dict[int, npt.NDArray[np.float32]]: ...
    @property
    def reachability_totals(self) -> list[int]:
        """Total reachability counts per distance from sampled sources."""
        ...
    @property
    def sampled_source_count(self) -> int:
        """Number of sources that were sampled."""
        ...

class CentralitySegmentResult:
    """Holds results for segment-based centrality calculations."""

    distances: list[int]
    node_keys_py: list[Any]
    node_indices: list[int]

    @property
    def segment_density(self) -> dict[int, npt.NDArray[np.float32]]: ...
    @property
    def segment_harmonic(self) -> dict[int, npt.NDArray[np.float32]]: ...
    @property
    def segment_beta(self) -> dict[int, npt.NDArray[np.float32]]: ...
    @property
    def segment_betweenness(self) -> dict[int, npt.NDArray[np.float32]]: ...
