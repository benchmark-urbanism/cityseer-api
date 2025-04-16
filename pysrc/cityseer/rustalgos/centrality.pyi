"""Centrality analysis utilities for network structures."""

from __future__ import annotations

import numpy.typing as npt

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
