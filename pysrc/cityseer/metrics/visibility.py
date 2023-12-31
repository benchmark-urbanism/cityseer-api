# pyright: basic
"""
Visibility and viewshed analysis.
"""
from __future__ import annotations

from functools import partial
from pathlib import Path

import geopandas as gpd
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
    """
    Retrieve buildings from OSM given WGS bounds.
    """
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
    """
    Prepare an output path for writing TIFF data.
    """
    write_path = Path(out_path)
    if not write_path.parent.exists():
        raise ValueError(f"Directory {write_path.parent.resolve()} does not exist")
    if write_path.is_dir():
        raise IOError("Specified write path is a directory but should be a file name")
    # remove file extension
    write_path = Path(write_path.parent / write_path.stem)

    return write_path


def _prepare_epsg_code(bounds: tuple[float, float, float, float], to_crs_code: int | str | None) -> int | str:
    """
    Find a UTM EPSG code if no output EPSG code is provided.
    """
    bounds_geom = geometry.box(*bounds)
    if to_crs_code is None:
        to_crs_code = util.extract_utm_epsg_code(bounds_geom.centroid.x, bounds_geom.centroid.y)
    return to_crs_code


def _prepare_bldgs_rast(
    bldgs_gdf: gpd.GeoDataFrame,
    bounds: tuple[float, float, float, float],
    from_crs_code: int | str,
    to_crs_code: int | str,
    resolution: int,
) -> tuple[npt.ArrayLike, rasterio.Affine]:
    """
    Convert a buildings GeoDataFrame into a raster with accompanying Transform object.
    """
    bldgs_gdf = bldgs_gdf.to_crs(to_crs_code)  # type: ignore
    if not bldgs_gdf.crs.is_projected:
        raise ValueError("Buildings GeoDataFrame must be in a projected coordinate reference system.")
    # prepare extents from original bounds
    # i.e. don't use bldgs GDF because this will overshoot per building polys
    projected_bounds = util.project_geom(geometry.box(*bounds), from_crs_code, to_crs_code)
    w, s = np.floor(projected_bounds.bounds[:2]).astype(int)
    e, n = np.floor(projected_bounds.bounds[2:]).astype(int)
    width = int(abs(e - w) / resolution)
    height = int(abs(n - s) / resolution)
    # prepare transform
    transform = from_bounds(w, s, e, n, width, height)
    # rasterize building polygons
    unioned_gdf = gpd.GeoDataFrame(geometry=[bldgs_gdf.unary_union])  # type: ignore
    unioned_gdf.set_crs(bldgs_gdf.crs)
    bldgs_rast = rasterize([(geom, 1) for geom in unioned_gdf.geometry], out_shape=(height, width), transform=transform)

    return bldgs_rast, transform


def visibility_graph(
    bldgs_gdf: gpd.GeoDataFrame,
    bounds: tuple[float, float, float, float],
    out_path: str,
    from_crs_code: int | str,
    to_crs_code: int | str | None = None,
    view_distance: int = 100,
    resolution: int = 1,
):
    """
    Run a visibility graph analysis.

    This will return three TIFF image files, respectively showing a density, farness, and harmonic closeness based
    measure.

    Parameters
    ----------
    bldgs_gdf: gpd.GeoDataFrame
        A GeoDataFrame containing building polygons.
    bounds: tuple[float, float, float, float]
        A tuple specifying the bounds corresponding to the provided `from_crs_code` parameter.
    out_path: str
        An output path to which the generated TIFF images will be written. The pathname will be appended to correspond
        to the density, farness, and harmonic closeness measures.
    from_crs_code: int | str
        The EPSG coordinate reference code corresponding to the input data.
    to_crs_code: int | str | None = None
        An output EPSG coordinate reference code. `None` by default, in which case a UTM projection will be used.
    view_distance: int = 100
        The view distance within which to run the visibility analysis. 100m by default.
    resolution: int = 1
        The spatial resolution in metres to use when generating the raster. Lower resolutions will result in faster
        analysis.

    """
    write_path = _prepare_path(out_path)
    to_crs_code = _prepare_epsg_code(bounds, to_crs_code)
    bldgs_rast, transform = _prepare_bldgs_rast(bldgs_gdf, bounds, from_crs_code, to_crs_code, resolution)
    # run viewshed
    viewshed_struct = rustalgos.Viewshed()
    # convert distance to cells
    resolution_distance = int(view_distance / resolution)
    # wrap with progress monitor
    partial_func = partial(viewshed_struct.visibility_graph, bldgs_rast, resolution_distance)
    bands = config.wrap_progress(
        total=bldgs_rast.shape[0] * bldgs_rast.shape[1],  # type: ignore
        rust_struct=viewshed_struct,
        partial_func=partial_func,
    )
    for band_idx, key in enumerate(["density", "farness", "harmonic"]):
        path = Path(f"{write_path}_{key}").with_suffix(".tif")
        with rasterio.open(
            str(path.resolve()),
            "w",
            driver="GTiff",
            height=bldgs_rast.shape[0],  # type: ignore
            width=bldgs_rast.shape[1],  # type: ignore
            count=1,
            dtype=np.float32,
            crs=to_crs_code,
            transform=transform,
        ) as dst:
            dst.write(bands[band_idx], 1)  # type: ignore


def visibility_graph_from_osm(
    bounds_wgs: tuple[float, float, float, float],
    out_path: str,
    to_crs_code: int | str | None = None,
    view_distance: int = 100,
    resolution: int = 1,
) -> None:
    """
    Retrieves OSM buildings for the specified WGS bounds and runs a visibility analysis.

    This will return three TIFF image files, respectively showing a density, farness, and harmonic closeness based
    measure.

    Parameters
    ----------
    bounds_wgs: tuple[float, float, float, float]
        A tuple specifying the bounds corresponding to the provided `from_crs_code` parameter.
    out_path: str
        An output path to which the generated TIFF images will be written. The pathname will be appended to correspond
        to the density, farness, and harmonic closeness measures.
    to_crs_code: int | str | None = None
        An output EPSG coordinate reference code. `None` by default, in which case a UTM projection will be used.
    view_distance: int = 100
        The view distance within which to run the visibility analysis. 100m by default.
    resolution: int = 1
        The spatial resolution in metres to use when generating the raster. Lower resolutions will result in faster
        analysis.

    """
    # get buildings for buffered extents
    bldgs_gdf = _buildings_from_osmnx(bounds_wgs)
    # run visibility graph
    visibility_graph(bldgs_gdf, bounds_wgs, out_path, 4326, to_crs_code, view_distance, resolution)


def viewshed(
    bldgs_gdf: gpd.GeoDataFrame,
    bounds: tuple[float, float, float, float],
    origin_x: float,
    origin_y: float,
    out_path: str,
    from_crs_code: int | str,
    to_crs_code: int | str | None = None,
    view_distance: int = 100,
    resolution: int = 1,
):
    """
    Run a viewshed analysis from a specified point. Writes an output image to the specified output path.

    Parameters
    ----------
    bldgs_gdf: gpd.GeoDataFrame
        A GeoDataFrame containing building polygons.
    bounds: tuple[float, float, float, float]
        A tuple specifying the bounds corresponding to the provided `from_crs_code` parameter.
    origin_x: float
        An easting or longitude for the origin of the viewshed in the `from_crs_code` coordinate reference system.
    origin_y: float
        A northing or latitude for the origin of the viewshed in the `from_crs_code` coordinate reference system.
    out_path: str
        An output path to which the generated TIFF images will be written. The pathname will be appended to correspond
        to the density, farness, and harmonic closeness measures.
    from_crs_code: int | str
        The EPSG coordinate reference code corresponding to the input data.
    to_crs_code: int | str | None = None
        An output EPSG coordinate reference code. `None` by default, in which case a UTM projection will be used.
    view_distance: int = 100
        The view distance within which to run the visibility analysis. 100m by default.
    resolution: int = 1
        The spatial resolution in metres to use when generating the raster. Lower resolutions will result in faster
        analysis.

    """
    write_path = _prepare_path(out_path)
    to_crs_code = _prepare_epsg_code(bounds, to_crs_code)
    bldgs_rast, transform = _prepare_bldgs_rast(bldgs_gdf, bounds, from_crs_code, to_crs_code, resolution)
    # prepare cell coordinates
    point_projected = util.project_geom(geometry.Point(origin_x, origin_y), from_crs_code, to_crs_code)
    x_idx, y_idx = ~transform * (point_projected.x, point_projected.y)  # type: ignore
    x_idx = int(x_idx)
    y_idx = int(y_idx)
    # run viewshed
    viewshed_struct = rustalgos.Viewshed()
    # convert distance to cells
    resolution_distance = int(view_distance / resolution)
    # find the viewshed
    rast = viewshed_struct.viewshed(bldgs_rast, resolution_distance, x_idx, y_idx)
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
        height=bldgs_rast.shape[0],  # type: ignore
        width=bldgs_rast.shape[1],  # type: ignore
        count=4,  # RGBA, so 4 channels
        dtype=np.uint8,
        crs=to_crs_code,
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
    to_crs_code: int | str | None = None,
    view_distance: int = 100,
    resolution: int = 1,
) -> None:
    """
    Run a viewshed analysis from a specified point using OSM data. Writes an output image to the specified output path.

    Parameters
    ----------
    bounds_wgs: tuple[float, float, float, float]
        A tuple specifying the bounds corresponding to the provided `from_crs_code` parameter.
    origin_lng: float
        A longitude for the origin of the viewshed in WGS84 coordinates.
    origin_lat: float
        A latitude for the origin of the viewshed in WGS84 coordinates.
    out_path: str
        An output path to which the generated TIFF images will be written. The pathname will be appended to correspond
        to the density, farness, and harmonic closeness measures.
    to_crs_code: int | str | None = None
        An output EPSG coordinate reference code. `None` by default, in which case a UTM projection will be used.
    view_distance: int = 100
        The view distance within which to run the visibility analysis. 100m by default.
    resolution: int = 1
        The spatial resolution in metres to use when generating the raster. Lower resolutions will result in faster
        analysis.

    """
    # get buildings for buffered extents
    bldgs_gdf = _buildings_from_osmnx(bounds_wgs)
    # run viewshed
    viewshed(bldgs_gdf, bounds_wgs, origin_lng, origin_lat, out_path, 4326, to_crs_code, view_distance, resolution)
