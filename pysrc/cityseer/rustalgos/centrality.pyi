"""Network centrality calculation results."""

from __future__ import annotations

from typing import Any  # For Py<PyAny>

import numpy as np  # For np.float32
import numpy.typing as npt

class CentralityShortestResult:
    """Holds results for shortest path (metric distance) centrality calculations."""

    distances: list[int]
    node_keys_py: list[Any]
    node_indices: list[int]

    node_density: dict[int, npt.NDArray[np.float32]]
    node_farness: dict[int, npt.NDArray[np.float32]]
    node_cycles: dict[int, npt.NDArray[np.float32]]
    node_harmonic: dict[int, npt.NDArray[np.float32]]  # Closeness based on harmonic mean
    node_beta: dict[int, npt.NDArray[np.float32]]  # Beta-weighted closeness
    node_betweenness: dict[int, npt.NDArray[np.float32]]
    node_betweenness_beta: dict[int, npt.NDArray[np.float32]]  # Beta-weighted betweenness

class CentralitySimplestResult:
    """Holds results for simplest path (angular distance) centrality calculations."""

    distances: list[int]
    node_keys_py: list[Any]
    node_indices: list[int]

    node_density: dict[int, npt.NDArray[np.float32]]
    node_farness: dict[int, npt.NDArray[np.float32]]  # Angular farness
    node_harmonic: dict[int, npt.NDArray[np.float32]]  # Angular closeness (harmonic)
    node_betweenness: dict[int, npt.NDArray[np.float32]]

class CentralitySegmentResult:
    """Holds results for segment-based centrality calculations."""

    distances: list[int]
    node_keys_py: list[Any]
    node_indices: list[int]  # Note: these are still node indices, results are mapped to segments via nodes

    segment_density: dict[int, npt.NDArray[np.float32]]
    segment_harmonic: dict[int, npt.NDArray[np.float32]]  # Segment closeness (harmonic mean)
    segment_beta: dict[int, npt.NDArray[np.float32]]  # Segment beta-weighted closeness
    segment_betweenness: dict[int, npt.NDArray[np.float32]]
