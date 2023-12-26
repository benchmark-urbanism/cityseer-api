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


def prepare_bldgs_rast(bldgs_gdf: gpd.GeoDataFrame, crs: int, resolution: int):
    """ """
    # build raster
    unioned_gdf = gpd.GeoDataFrame(geometry=[bldgs_gdf.unary_union])
    unioned_gdf.set_crs(crs)
    bbox = unioned_gdf.iloc[0].geometry.bounds
    w, s = np.floor(bbox[:2]).astype(int)
    e, n = np.floor(bbox[2:]).astype(int)
    width = int(abs(e - w) / resolution)
    height = int(abs(n - s) / resolution)
    # Rasterize building polygons
    transform = from_bounds(w, s, e, n, width, height)
    bldgs_rast = rasterize([(geom, 1) for geom in unioned_gdf.geometry], out_shape=(height, width), transform=transform)
    return bldgs_rast, transform


def visibility_graph(bldgs_gdf: gpd.GeoDataFrame, crs: int, out_path: str, distance: int = 100, resolution: int = 1):
    """ """
    if bldgs_gdf.crs and not bldgs_gdf.crs.is_projected:
        raise ValueError("Buildings GeoDataFrame must be in a projected coordinate reference system.")
    write_path = Path(out_path)
    if not write_path.parent.exists():
        raise ValueError(f"Directory {write_path.parent.resolve()} does not exist")
    bldgs_rast, transform = prepare_bldgs_rast(bldgs_gdf, crs, resolution)
    viewshed = rustalgos.Viewshed()
    partial_func = partial(viewshed.visibility_graph, bldgs_rast, distance)
    bands = config.wrap_progress(
        total=bldgs_rast.shape[0] * bldgs_rast.shape[1], rust_struct=viewshed, partial_func=partial_func
    )
    with rasterio.open(
        str(write_path.resolve()),
        "w",
        driver="GTiff",
        height=bldgs_rast.shape[0],
        width=bldgs_rast.shape[1],
        count=3,
        dtype=np.float32,
        crs=crs,
        transform=transform,
    ) as dst:
        for i in range(3):
            dst.write(bands[i], i + 1)


def viewshed(
    bldgs_gdf: gpd.GeoDataFrame,
    crs: int,
    out_path: str,
    origin_x: int,
    origin_y: int,
    distance: int = 100,
    resolution: int = 1,
):
    """ """
    if bldgs_gdf.crs and not bldgs_gdf.crs.is_projected:
        raise ValueError("Buildings GeoDataFrame must be in a projected coordinate reference system.")
    write_path = Path(out_path)
    if not write_path.parent.exists():
        raise ValueError(f"Directory {write_path.parent.resolve()} does not exist")
    bldgs_rast, transform = prepare_bldgs_rast(bldgs_gdf, crs, resolution)
    viewshed = rustalgos.Viewshed()
    rast = viewshed.viewshed(bldgs_rast, distance, origin_x, origin_y)
    with rasterio.open(
        str(write_path.resolve()),
        "w",
        driver="GTiff",
        height=bldgs_rast.shape[0],
        width=bldgs_rast.shape[1],
        count=3,
        dtype=np.float32,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(rast, 1)


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
