""" """
# pyright: basic
from __future__ import annotations

from functools import partial
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import numpy.typing as npt
import osmnx as ox
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely import geometry

from cityseer import config, rustalgos
from cityseer.tools import util


def _buildings_from_osmnx(bounds_poly: geometry.Polygon) -> gpd.GeoDataFrame:
    """ """
    # fetch buildings
    bldgs_gdf = ox.features_from_polygon(bounds_poly, {"building": True})
    # explode features
    bldgs_gdf = bldgs_gdf.explode()
    # clean up index
    bldgs_gdf = bldgs_gdf.reset_index(drop=True)
    # return minimal info
    return bldgs_gdf[["geometry"]]


def _prepare_bldgs_rast(
    bounds: tuple[float, float, float, float], bldgs_gdf: gpd.GeoDataFrame, resolution: int
) -> tuple[npt.ArrayLike, rasterio.Affine]:
    """ """
    # prepare extents from original bounds - i.e. don't use bldgs GDF because this will overshoot per building polys
    w, s = np.floor(bounds[:2]).astype(int)
    e, n = np.floor(bounds[2:]).astype(int)
    width = int(abs(e - w) / resolution)
    height = int(abs(n - s) / resolution)
    # prepare transform
    transform = from_bounds(w, s, e, n, width, height)
    # rasterize building polygons
    unioned_gdf = gpd.GeoDataFrame(geometry=[bldgs_gdf.unary_union])
    unioned_gdf.set_crs(bldgs_gdf.crs)
    bldgs_rast = rasterize([(geom, 1) for geom in unioned_gdf.geometry], out_shape=(height, width), transform=transform)

    return bldgs_rast, transform


def visibility_graph(
    bounds: tuple[float, float, float, float],
    bldgs_gdf: gpd.GeoDataFrame,
    to_epsg_code: int,
    out_path: str,
    view_distance: int = 100,
    resolution: int = 1,
):
    """ """
    if bldgs_gdf.crs and not bldgs_gdf.crs.is_projected:
        raise ValueError("Buildings GeoDataFrame must be in a projected coordinate reference system.")
    write_path = Path(out_path)
    if not write_path.parent.exists():
        raise ValueError(f"Directory {write_path.parent.resolve()} does not exist")
    # prepare raster and transform
    bldgs_rast, transform = _prepare_bldgs_rast(bounds, bldgs_gdf, resolution)
    # run viewshed
    viewshed = rustalgos.Viewshed()
    # convert distance to cells
    resolution_distance = int(view_distance / resolution)
    # wrap with progress monitor
    partial_func = partial(viewshed.visibility_graph, bldgs_rast, resolution_distance)
    bands = config.wrap_progress(
        total=bldgs_rast.shape[0] * bldgs_rast.shape[1], rust_struct=viewshed, partial_func=partial_func
    )
    # write output TIF
    with rasterio.open(
        str(write_path.resolve()),
        "w",
        driver="GTiff",
        height=bldgs_rast.shape[0],
        width=bldgs_rast.shape[1],
        count=3,
        dtype=np.float32,
        crs=to_epsg_code,
        transform=transform,
    ) as dst:
        for i in range(3):
            dst.write(bands[i], i + 1)


def visibility_graph_from_osm(
    bounds: tuple[float, float, float, float],
    out_path: str,
    view_distance: int = 100,
    resolution: int = 1,
    to_epsg_code: int | None = None,
) -> None:
    """ """
    # create box from extent
    extents_wgs = geometry.box(*bounds)
    # get buildings for buffered extents
    bldgs_gdf = _buildings_from_osmnx(extents_wgs)
    # convert GDF to target EPSG
    if to_epsg_code is None:
        to_epsg_code = util.extract_utm_epsg_code(extents_wgs.centroid.x, extents_wgs.centroid.y)
    bldgs_gdf = bldgs_gdf.to_crs(to_epsg_code)
    extents = util.project_geom(extents_wgs, 4326, to_epsg_code)
    # run visibility graph
    visibility_graph(extents.bounds, bldgs_gdf, to_epsg_code, out_path, view_distance, resolution)


def viewshed_from_osm(
    bounds: tuple[float, float, float, float],
    origin_lng: int,
    origin_lat: int,
    out_path: str,
    view_distance: int = 100,
    resolution: int = 1,
    to_epsg_code: int | None = None,
) -> None:
    """ """
    # create box from extent
    extents_wgs = geometry.box(*bounds)
    # get buildings for buffered extents
    bldgs_gdf = _buildings_from_osmnx(extents_wgs)
    # convert GDF to target EPSG
    if to_epsg_code is None:
        to_epsg_code = util.extract_utm_epsg_code(extents_wgs.centroid.x, extents_wgs.centroid.y)
    bldgs_gdf = bldgs_gdf.to_crs(to_epsg_code)
    extents = util.project_geom(extents_wgs, 4326, to_epsg_code)
    # run visibility graph
    if bldgs_gdf.crs and not bldgs_gdf.crs.is_projected:
        raise ValueError("Buildings GeoDataFrame must be in a projected coordinate reference system.")
    write_path = Path(out_path)
    if not write_path.parent.exists():
        raise ValueError(f"Directory {write_path.parent.resolve()} does not exist")
    # prepare raster and transform
    bldgs_rast, transform = _prepare_bldgs_rast(extents.bounds, bldgs_gdf, resolution)
    # prepare cell coordinates
    point_projected = util.project_geom(geometry.Point(origin_lng, origin_lat), 4326, to_epsg_code)
    y_idx, x_idx = ~transform * (point_projected.x, point_projected.y)
    # run viewshed
    viewshed = rustalgos.Viewshed()
    # convert distance to cells
    resolution_distance = int(view_distance / resolution)
    # find the viewshed
    rast = viewshed.viewshed(bldgs_rast, resolution_distance, int(x_idx), int(y_idx))
    with rasterio.open(
        str(write_path.resolve()),
        "w",
        driver="GTiff",
        height=bldgs_rast.shape[0],
        width=bldgs_rast.shape[1],
        count=3,
        dtype=np.float32,
        crs=to_epsg_code,
        transform=transform,
    ) as dst:
        dst.write(rast, 1)
