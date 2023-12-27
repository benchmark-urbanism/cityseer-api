# pyright: basic

# %%
from __future__ import annotations

import pytest

from cityseer.metrics import visibility

# %%
soho_bounds = [-0.13169766943736105, 51.509755641119575, -0.11945904078380688, 51.517436276694816]
hackney_bounds = [-0.0595339, 51.5428946, -0.0401002, 51.5503604]

# %%
visibility.visibility_graph_from_osm(
    hackney_bounds, "../../temp/test_vga_rd.tif", view_distance=300, to_epsg_code=27700, resolution=2
)

# %%
visibility.viewshed_from_osm(
    soho_bounds,
    -0.126952,
    51.513783,
    "../../temp/test_viewshed.tif",
    view_distance=200,
    to_epsg_code=27700,
    resolution=1,
)
