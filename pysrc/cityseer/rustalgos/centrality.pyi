"""Network centrality calculation results and OD matrix."""

from __future__ import annotations

from typing import Any

import numpy as np
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

class CentralityResult:
    """Holds centrality results from closeness, betweenness, and/or cycles computation."""

    distances: list[int]
    node_keys_py: list[Any]
    node_indices: list[int]

    @property
    def metrics(self) -> dict[str, dict[int, npt.NDArray[np.float64]]]:
        """All computed metrics as {name: {distance: array}}."""
        ...
    @property
    def reachability_totals(self) -> list[int]:
        """Total reachability counts per distance from sampled sources."""
        ...
    @property
    def sampled_source_count(self) -> int:
        """Number of sources that were sampled."""
        ...
