""" """
# pyright: basic
from __future__ import annotations

from functools import partial
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely import geometry

from cityseer import config, rustalgos
from cityseer.tools import util


def viewsheds(bldgs_gdf: gpd.GeoDataFrame, crs: int, out_path: str, resolution: int = 1):
    """ """
    if bldgs_gdf.crs and not bldgs_gdf.crs.is_projected:
        raise ValueError("Buildings GeoDataFrame must be in a projected coordinate reference system.")
    write_path = Path(out_path)
    if not write_path.parent.exists():
        raise ValueError(f"Directory {write_path.parent.resolve()} does not exist")
    # build raster
    unioned_gdf = gpd.GeoDataFrame(geometry=[bldgs_gdf.unary_union])
    unioned_gdf.set_crs(crs)
    bbox = unioned_gdf.iloc[0].geometry.bounds
    w, s = np.floor(bbox[:2]).astype(int)
    e, n = np.floor(bbox[2:]).astype(int)
    width = int(abs(e - w) / resolution)
    height = int(abs(n - s) / resolution)
    # Create a base raster
    raster = np.zeros((height, width), dtype=np.uint8)
    # Rasterize building polygons
    transform = from_bounds(w, s, e, n, width, height)
    rasterized_buildings = rasterize(
        [(geom, 1) for geom in unioned_gdf.geometry], out_shape=raster.shape, transform=transform
    )
    viewshed = rustalgos.Viewshed()
    partial_func = partial(viewshed.process_raster, rasterized_buildings, 200)
    bands = config.wrap_progress(total=width * height, rust_struct=viewshed, partial_func=partial_func)
    with rasterio.open(
        str(write_path.resolve()),
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype=np.float32,
        crs=crs,
        transform=transform,
    ) as dst:
        for i in range(3):
            dst.write(bands[i], i + 1)


def _buildings_from_osmnx(poly: geometry.Polygon) -> gpd.GeoDataFrame:
    """ """
    bldgs_gdf = ox.features_from_polygon(poly, {"building": True})
    bldgs_gdf = bldgs_gdf.explode()
    bldgs_gdf = bldgs_gdf.reset_index(drop=True)
    return bldgs_gdf[["geometry"]]


def vga_from_osm(poly: geometry.Polygon, from_epsg_code: int, to_epsg_code: int) -> None:
    """ """
    # util.project_geom
    pass
