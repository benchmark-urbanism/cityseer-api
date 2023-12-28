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


def _buildings_from_osmnx(bounds: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    """ """
    # fetch buildings
    bounds_geom = geometry.box(*bounds)
    bldgs_gdf = ox.features_from_polygon(bounds_geom, {"building": True})
    # explode features
    bldgs_gdf = bldgs_gdf.explode()
    # clean up index
    bldgs_gdf = bldgs_gdf.reset_index(drop=True)
    # return minimal info
    return bldgs_gdf[["geometry"]]


def _prepare_path(out_path: str | Path) -> Path:
    """ """
    write_path = Path(out_path)
    if not write_path.parent.exists():
        raise ValueError(f"Directory {write_path.parent.resolve()} does not exist")
    if write_path.is_dir():
        raise IOError("Specified write path is a directory but should be a file name")
    # remove file extension
    write_path = Path(write_path.parent / write_path.stem)

    return write_path


def _prepare_epsg_code(bounds: tuple[float, float, float, float], to_epsg_code: int | None) -> int:
    """ """
    bounds_geom = geometry.box(*bounds)
    if to_epsg_code is None:
        to_epsg_code = util.extract_utm_epsg_code(bounds_geom.centroid.x, bounds_geom.centroid.y)
    return to_epsg_code


def _prepare_bldgs_rast(
    bldgs_gdf: gpd.GeoDataFrame,
    bounds: tuple[float, float, float, float],
    from_epsg_code: int,
    to_epsg_code: int,
    resolution: int,
) -> tuple[npt.ArrayLike, rasterio.Affine]:
    """ """
    bldgs_gdf = bldgs_gdf.to_crs(to_epsg_code)
    if not bldgs_gdf.crs.is_projected:
        raise ValueError("Buildings GeoDataFrame must be in a projected coordinate reference system.")
    # prepare extents from original bounds
    # i.e. don't use bldgs GDF because this will overshoot per building polys
    projected_bounds = util.project_geom(geometry.box(*bounds), from_epsg_code, to_epsg_code)
    w, s = np.floor(projected_bounds.bounds[:2]).astype(int)
    e, n = np.floor(projected_bounds.bounds[2:]).astype(int)
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
    bldgs_gdf: gpd.GeoDataFrame,
    bounds: tuple[float, float, float, float],
    out_path: str,
    from_epsg_code: int,
    to_epsg_code: int | None = None,
    view_distance: int = 100,
    resolution: int = 1,
):
    """ """
    write_path = _prepare_path(out_path)
    to_epsg_code = _prepare_epsg_code(bounds, to_epsg_code)
    bldgs_rast, transform = _prepare_bldgs_rast(bldgs_gdf, bounds, from_epsg_code, to_epsg_code, resolution)
    # run viewshed
    viewshed = rustalgos.Viewshed()
    # convert distance to cells
    resolution_distance = int(view_distance / resolution)
    # wrap with progress monitor
    partial_func = partial(viewshed.visibility_graph, bldgs_rast, resolution_distance)
    bands = config.wrap_progress(
        total=bldgs_rast.shape[0] * bldgs_rast.shape[1], rust_struct=viewshed, partial_func=partial_func
    )
    for band_idx, key in enumerate(["density", "farness", "harmonic"]):
        path = Path(f"{write_path}_{key}").with_suffix(".tif")
        with rasterio.open(
            str(path.resolve()),
            "w",
            driver="GTiff",
            height=bldgs_rast.shape[0],
            width=bldgs_rast.shape[1],
            count=1,
            dtype=np.float32,
            crs=to_epsg_code,
            transform=transform,
        ) as dst:
            dst.write(bands[band_idx], 1)


def visibility_graph_from_osm(
    bounds_wgs: tuple[float, float, float, float],
    out_path: str,
    to_epsg_code: int | None = None,
    view_distance: int = 100,
    resolution: int = 1,
) -> None:
    """ """
    # get buildings for buffered extents
    bldgs_gdf = _buildings_from_osmnx(bounds_wgs)
    # run visibility graph
    visibility_graph(bldgs_gdf, bounds_wgs, out_path, 4326, to_epsg_code, view_distance, resolution)


def viewshed(
    bldgs_gdf: gpd.GeoDataFrame,
    bounds: tuple[float, float, float, float],
    origin_lng: int,
    origin_lat: int,
    out_path: str,
    from_epsg_code: int,
    to_epsg_code: int | None = None,
    view_distance: int = 100,
    resolution: int = 1,
):
    """ """
    write_path = _prepare_path(out_path)
    to_epsg_code = _prepare_epsg_code(bounds, to_epsg_code)
    bldgs_rast, transform = _prepare_bldgs_rast(bldgs_gdf, bounds, from_epsg_code, to_epsg_code, resolution)
    # prepare cell coordinates
    point_projected = util.project_geom(geometry.Point(origin_lng, origin_lat), from_epsg_code, to_epsg_code)
    x_idx, y_idx = ~transform * (point_projected.x, point_projected.y)
    x_idx = int(x_idx)
    y_idx = int(y_idx)
    # run viewshed
    viewshed = rustalgos.Viewshed()
    # convert distance to cells
    resolution_distance = int(view_distance / resolution)
    # find the viewshed
    rast = viewshed.viewshed(bldgs_rast, resolution_distance, x_idx, y_idx)
    # write to raster
    rgb_arr = np.dstack((rast, rast, rast, np.full_like(rast, 255))).astype(np.uint8)  # Fully opaque alpha channel
    rgb_arr[rast == 1] = [255, 0, 0, 255]  # red for view
    rgb_arr[rast == 0] = [0, 0, 0, 0]  # transparent for non-built
    rgb_arr[bldgs_rast == 1] = [10, 10, 10, 255]  # black for built
    rgb_arr[y_idx - 2 : y_idx + 2, x_idx - 2 : x_idx + 2] = [255, 255, 0, 255]  # yellow for origin

    with rasterio.open(
        str(write_path.with_suffix(".tif").resolve()),
        "w",
        driver="GTiff",
        height=bldgs_rast.shape[0],
        width=bldgs_rast.shape[1],
        count=4,  # RGBA, so 4 channels
        dtype=np.uint8,
        crs=to_epsg_code,
        transform=transform,
    ) as dst:
        dst.write(rgb_arr[:, :, 0], 1)  # Red channel
        dst.write(rgb_arr[:, :, 1], 2)  # Green channel
        dst.write(rgb_arr[:, :, 2], 3)  # Blue channel
        dst.write(rgb_arr[:, :, 3], 4)  # Alpha channel


def viewshed_from_osm(
    bounds_wgs: tuple[float, float, float, float],
    origin_lng: int,
    origin_lat: int,
    out_path: str,
    to_epsg_code: int | None = None,
    view_distance: int = 100,
    resolution: int = 1,
) -> None:
    """ """
    # get buildings for buffered extents
    bldgs_gdf = _buildings_from_osmnx(bounds_wgs)
    # run viewshed
    viewshed(bldgs_gdf, bounds_wgs, origin_lng, origin_lat, out_path, 4326, to_epsg_code, view_distance, resolution)
