"""Viewshed analysis utilities for spatial visibility studies."""

from __future__ import annotations

import numpy.typing as npt

class Viewshed:
    @classmethod
    def new(cls) -> Viewshed: ...
    def progress(self) -> int: ...
    def visibility_graph(
        self, bldgs_rast: npt.ArrayLike, view_distance: float, pbar_disabled: bool | None = None
    ) -> tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]: ...
    def viewshed(
        self,
        bldgs_rast: npt.ArrayLike,
        view_distance: float,
        origin_x: int,
        origin_y: int,
    ) -> npt.ArrayLike: ...
