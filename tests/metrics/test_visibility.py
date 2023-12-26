# pyright: basic

# %%
from __future__ import annotations

import pytest
from shapely import geometry

from cityseer.metrics import visibility

# %%
# def test_vga():
""" """
poly = geometry.box(-0.13169766943736105, 51.509755641119575, -0.11945904078380688, 51.517436276694816)
# poly = geometry.box(-0.130, 51.512, -0.124, 51.515)
bldgs_gdf = visibility._buildings_from_osmnx(poly)
crs = 27700
bldgs_gdf = bldgs_gdf.to_crs(crs)
resolution = 1  # for example, 1 meter
# %%
# 5.59
visibility.visibility_graph(bldgs_gdf, crs, "../../temp/test3.tif", distance=50)

# %%
visibility.viewshed(bldgs_gdf, crs, "../../temp/test_vs.tif", 10, 500, distance=50)
