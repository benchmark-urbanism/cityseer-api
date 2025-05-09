"""Network centrality calculation results."""

from __future__ import annotations

from typing import Any

import numpy as np  # For np.float32
import numpy.typing as npt

class CentralityShortestResult:
    """Holds results for shortest path (metric distance) centrality calculations."""

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
    def node_harmonic(self) -> dict[int, npt.NDArray[np.float32]]: ...  # Closeness based on harmonic mean
    @property
    def node_beta(self) -> dict[int, npt.NDArray[np.float32]]: ...  # Beta-weighted closeness
    @property
    def node_betweenness(self) -> dict[int, npt.NDArray[np.float32]]: ...
    @property
    def node_betweenness_beta(self) -> dict[int, npt.NDArray[np.float32]]: ...  # Beta-weighted betweenness

class CentralitySimplestResult:
    """Holds results for simplest path (angular distance) centrality calculations."""

    distances: list[int]
    node_keys_py: list[Any]
    node_indices: list[int]

    @property
    def node_density(self) -> dict[int, npt.NDArray[np.float32]]: ...
    @property
    def node_farness(self) -> dict[int, npt.NDArray[np.float32]]: ...  # Angular farness
    @property
    def node_harmonic(self) -> dict[int, npt.NDArray[np.float32]]: ...  # Angular closeness (harmonic)
    @property
    def node_betweenness(self) -> dict[int, npt.NDArray[np.float32]]: ...

class CentralitySegmentResult:
    """Holds results for segment-based centrality calculations."""

    distances: list[int]
    node_keys_py: list[Any]
    node_indices: list[int]  # Note: these are still node indices, results are mapped to segments via nodes

    @property
    def segment_density(self) -> dict[int, npt.NDArray[np.float32]]: ...
    @property
    def segment_harmonic(self) -> dict[int, npt.NDArray[np.float32]]: ...  # Segment closeness (harmonic mean)
    @property
    def segment_beta(self) -> dict[int, npt.NDArray[np.float32]]: ...  # Segment beta-weighted closeness
    @property
    def segment_betweenness(self) -> dict[int, npt.NDArray[np.float32]]: ...
