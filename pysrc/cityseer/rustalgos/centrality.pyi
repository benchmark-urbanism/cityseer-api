"""Network centrality calculation results."""

from __future__ import annotations

import numpy.typing as npt

class CentralityShortestResult:
    """Holds results for shortest path (metric distance) centrality calculations."""

    node_density: dict[int, npt.ArrayLike]
    node_farness: dict[int, npt.ArrayLike]
    node_closeness: dict[int, npt.ArrayLike]
    node_betweenness: dict[int, npt.ArrayLike]

class CentralitySimplestResult:
    """Holds results for simplest path (angular distance) centrality calculations."""

    node_density: dict[int, npt.ArrayLike]
    node_farness: dict[int, npt.ArrayLike]
    node_closeness: dict[int, npt.ArrayLike]
    node_betweenness: dict[int, npt.ArrayLike]

class CentralitySegmentResult:
    """Holds results for segment-based centrality calculations."""

    segment_density: dict[int, npt.ArrayLike]
    segment_farness: dict[int, npt.ArrayLike]
    segment_closeness: dict[int, npt.ArrayLike]
    segment_betweenness: dict[int, npt.ArrayLike]
