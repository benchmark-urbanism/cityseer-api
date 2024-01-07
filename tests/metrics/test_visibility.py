# pyright: basic

# %%
from __future__ import annotations

import pytest

from cityseer.metrics import visibility

# %%
soho_bounds = [-0.13169766943736105, 51.509755641119575, -0.11945904078380688, 51.517436276694816]
hackney_bounds = [-0.0595339, 51.5428946, -0.0401002, 51.5503604]

# %%
if False:
    visibility.visibility_graph_from_osm(
        soho_bounds, "../../temp/test_vga_rd.tif", to_crs_code=27700, view_distance=200, resolution=1
    )

# %%
if False:
    visibility.viewshed_from_osm(
        soho_bounds,
        -0.1291582,
        51.5133715,
        "../../temp/test_viewshed.tif",
        to_crs_code=27700,
        view_distance=400,
        resolution=1,
    )
