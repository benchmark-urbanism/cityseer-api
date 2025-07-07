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
    def visibility(
        self,
        bldgs_rast: npt.ArrayLike,
        view_distance: float,
        resolution: float,
        observer_height: float,
        pbar_disabled: bool | None = None,
    ) -> tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
        """
        Compute visibility metrics for all cells in a raster grid.

        Calculates line-of-sight from each cell to all others within `view_distance`,
        considering obstructions and elevation in `bldgs_rast` (float32 elevation values).
        Uses parallel processing.

        Parameters
        ----------
        bldgs_rast: npt.ArrayLike
            2D NumPy array of float32 elevations (meters above datum).
        view_distance: float
            Maximum view distance (in meters).
        resolution: float
            Raster cell size (meters per pixel).
        observer_height: float
            Height of observer above ground (meters).
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
        resolution: float,
        observer_height: float,
        origin_x: int,
        origin_y: int,
    ) -> npt.ArrayLike:
        """
        Compute the viewshed (visible cells) from a single origin point.

        Parameters
        ----------
        bldgs_rast: npt.ArrayLike
            2D NumPy array of float32 elevations (meters above datum).
        view_distance: float
            Maximum view distance (in meters).
        resolution: float
            Raster cell size (meters per pixel).
        observer_height: float
            Height of observer above ground (meters).
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
