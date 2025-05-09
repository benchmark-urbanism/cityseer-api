"""Viewshed analysis utilities for spatial visibility studies on raster grids."""

from __future__ import annotations

import numpy.typing as npt

class Viewshed:
    """Performs viewshed calculations on a 2D raster grid representing obstructions."""
    @classmethod
    def new(cls) -> Viewshed:
        """Initialize a new Viewshed instance."""
        ...
    def progress(self) -> int:
        """Get the current value of the internal progress counter."""
        ...
    def visibility_graph(
        self, bldgs_rast: npt.ArrayLike, view_distance: float, pbar_disabled: bool | None = None
    ) -> tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
        """
        Compute visibility metrics for all cells in a raster grid.

        Calculates line-of-sight from each cell to all others within `view_distance`,
        considering obstructions defined by non-zero values in `bldgs_rast`.
        Uses parallel processing.

        Parameters
        ----------
        bldgs_rast: npt.ArrayLike
            2D NumPy array where 1 indicates an obstruction, 0 is open space.
        view_distance: float
            Maximum distance (in cell units) to check visibility.
        pbar_disabled: bool | None
            Disable progress bar if True.

        Returns
        -------
        tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]
            Tuple of 2D NumPy arrays (same shape as input):
            - Visible cell count (density).
            - Sum of distances to visible cells (farness).
            - Sum of inverse distances to visible cells (harmonic mean distance indicator).
        """
        ...
    def viewshed(
        self,
        bldgs_rast: npt.ArrayLike,
        view_distance: float,
        origin_x: int,
        origin_y: int,
    ) -> npt.ArrayLike:
        """
        Compute the viewshed (visible cells) from a single origin point.

        Parameters
        ----------
        bldgs_rast: npt.ArrayLike
            2D NumPy array where 1 indicates an obstruction, 0 is open space.
        view_distance: float
            Maximum distance (in cell units) to check visibility.
        origin_x: int
            X-coordinate (column index) of the origin cell.
        origin_y: int
            Y-coordinate (row index) of the origin cell.

        Returns
        -------
        npt.ArrayLike
            2D NumPy array (same shape as input) where 1 indicates a visible cell, 0 otherwise.
        """
        ...
