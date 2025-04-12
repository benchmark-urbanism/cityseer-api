"""Viewshed analysis utilities for spatial visibility studies."""

from __future__ import annotations

import numpy.typing as npt

class Viewshed:
    @classmethod
    def new(cls) -> Viewshed: ...
    def progress(self) -> int: ...
    def visibility_graph(
        self, bldgs_rast: npt.ArrayLike, view_distance: int, pbar_disabled: bool = False
    ) -> tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]: ...
    def viewshed(
        self,
        bldgs_rast: npt.ArrayLike,
        view_distance: int,
        origin_x: int,
        origin_y: int,
        pbar_disabled: bool = False,
    ) -> npt.ArrayLike: ...
