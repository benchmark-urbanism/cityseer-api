# pyright: basic
"""
Visibility and viewshed analysis.
"""

from __future__ import annotations

import logging
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

from .. import config, rustalgos
from ..tools import util

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    return bldgs_gdf[["geometry"]]  # type: ignore


def _prepare_path(out_path: str | Path) -> Path:
    """
    Prepare an output path for writing TIFF data.
    """
    write_path = Path(out_path)
    if not write_path.parent.exists():
        raise ValueError(f"Directory {write_path.parent.resolve()} does not exist")
    if write_path.is_dir():
        raise OSError("Specified write path is a directory but should be a file name")
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
    logger.info("Preparing buildings raster.")
    bldgs_gdf = bldgs_gdf.to_crs(to_crs_code)
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
    bldgs_rast = rasterize([(geom, 3) for geom in bldgs_gdf.geometry], out_shape=(height, width), transform=transform)
    bldgs_rast = np.ascontiguousarray(bldgs_rast, dtype=np.float32)

    return bldgs_rast, transform


def visibility_graph(
    bldgs_gdf: gpd.GeoDataFrame,
    bounds: tuple[float, float, float, float],
    out_path: str,
    from_crs_code: int | str,
    to_crs_code: int | str | None = None,
    view_distance: int = 100,
    resolution: int = 1,
    observer_height: float = 1.5,
):
    logger.warning("visibility_graph is deprecated, please switch to visibility_from_gpd instead.")
    return visibility_from_gpd(
        bldgs_gdf,
        bounds,
        out_path,
        from_crs_code,
        to_crs_code,
        view_distance,
        resolution,
        observer_height,
    )


def visibility_from_gpd(
    bldgs_gdf: gpd.GeoDataFrame,
    bounds: tuple[float, float, float, float],
    out_path: str,
    from_crs_code: int | str,
    to_crs_code: int | str | None = None,
    view_distance: int = 100,
    resolution: int = 1,
    observer_height: float = 1.5,
) -> None:
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
    observer_height: float = 1.5
        The height of the observer in metres. 1.5m by default.

    """
    write_path = _prepare_path(out_path)
    to_crs_code = _prepare_epsg_code(bounds, to_crs_code)
    bldgs_rast, transform = _prepare_bldgs_rast(bldgs_gdf, bounds, from_crs_code, to_crs_code, resolution)
    # save raster for debugging
    path = Path(f"{write_path}_bldgs_rast").with_suffix(".tif")
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
        dst.write(bldgs_rast, 1)
    logger.info("Running visibility.")
    # run viewshed
    viewshed_struct = rustalgos.viewshed.Viewshed()
    # wrap with progress monitor
    partial_func = partial(viewshed_struct.visibility, bldgs_rast, view_distance, resolution, observer_height)
    bands = config.wrap_progress(
        total=bldgs_rast.shape[0] * bldgs_rast.shape[1],  # type: ignore
        rust_struct=viewshed_struct,
        partial_func=partial_func,
    )
    for band_idx, key in enumerate(["density", "farness", "harmonic"]):
        path = Path(f"{write_path}_{key}_dist_{view_distance}_res_{resolution}").with_suffix(".tif")
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
    observer_height: float = 1.5,
) -> None:
    logger.warning("visibility_graph_from_osm is deprecated, please switch to visibility_from_osm instead.")
    return visibility_from_osm(
        bounds_wgs,
        out_path,
        to_crs_code,
        view_distance,
        resolution,
        observer_height,
    )


def visibility_from_osm(
    bounds_wgs: tuple[float, float, float, float],
    out_path: str,
    to_crs_code: int | str | None = None,
    view_distance: int = 100,
    resolution: int = 1,
    observer_height: float = 1.5,
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
    observer_height: float = 1.5
        The height of the observer in metres. 1.5m by default.

    """
    # get buildings for buffered extents
    bldgs_gdf = _buildings_from_osmnx(bounds_wgs)
    # run visibility graph
    visibility_from_gpd(bldgs_gdf, bounds_wgs, out_path, 4326, to_crs_code, view_distance, resolution, observer_height)


def visibility_from_raster(
    input_path: str,
    out_path: str,
    view_distance: int = 100,
    observer_height: float = 1.5,
):
    """
    Run a visibility graph analysis directly from a raster file.

    This will return three TIFF image files, respectively showing a density, farness, and harmonic closeness based
    measure.

    Parameters
    ----------
    input_path: str
        Path to the input raster file representing building heights or presence.
    out_path: str
        An output path to which the generated TIFF images will be written. The pathname will be appended to correspond
        to the density, farness, and harmonic closeness measures.
    view_distance: int = 100
        The view distance within which to run the visibility analysis. 100m by default.
    observer_height: float = 1.5
        The height of the observer in metres. 1.5m by default.

    """
    read_path = Path(input_path)
    if not read_path.exists():
        raise FileNotFoundError(f"Input raster file {read_path.resolve()} does not exist.")
    logger.info(f"Running visibility graph from raster file: {read_path.resolve()}.")
    write_path = _prepare_path(out_path)

    # Read the raster from the input path
    with rasterio.open(str(read_path.resolve())) as src:
        bldgs_rast = src.read(1).astype(np.float32)
        transform = src.transform

    # Calculate raster resolution in meters per pixel from extent and shape
    width = bldgs_rast.shape[1]
    height = bldgs_rast.shape[0]
    # Get bounds: left, bottom, right, top
    left, top = transform * (0, 0)
    right, bottom = transform * (width, height)
    res_x = abs((right - left) / width)
    res_y = abs((top - bottom) / height)
    if not np.isclose(res_x, res_y, atol=0.01):
        raise ValueError(f"Non-square pixels detected: res_x={res_x}, res_y={res_y}")
    resolution = round(res_x, 2)
    logger.info(f"Raster resolution: {resolution} meters per pixel")

    # run viewshed
    viewshed_struct = rustalgos.viewshed.Viewshed()
    partial_func = partial(
        viewshed_struct.visibility,
        bldgs_rast,
        view_distance,
        resolution,
        observer_height,
    )

    # Validate the shape attribute of bldgs_rast
    if not hasattr(bldgs_rast, "shape") or len(bldgs_rast.shape) != 2:
        raise ValueError("bldgs_rast must be a 2D array.")

    # Use the validated shape for progress wrapping
    total = bldgs_rast.shape[0] * bldgs_rast.shape[1]
    bands = config.wrap_progress(
        total=total,
        rust_struct=viewshed_struct,
        partial_func=partial_func,
    )
    # Ensure `bands` is correctly unpacked and iterable
    if not isinstance(bands, list | tuple) or len(bands) != 3:
        raise ValueError("Expected `bands` to be a list or tuple with three elements.")

    # Write each band to the corresponding TIFF file
    for band_idx, key in enumerate(["density", "farness", "harmonic"]):
        path = Path(f"{write_path}_{key}_dist_{view_distance}_res_{resolution}").with_suffix(".tif")
        with rasterio.open(
            str(path.resolve()),
            "w",
            driver="GTiff",
            height=bldgs_rast.shape[0],
            width=bldgs_rast.shape[1],
            count=1,
            dtype=np.float32,
            transform=transform,
        ) as dst:
            dst.write(bands[band_idx], 1)


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
    observer_height: float = 1.5,
):
    logger.warning("viewshed is deprecated, please switch to viewshed_from_gpd instead.")
    return viewshed_from_gpd(
        bldgs_gdf,
        bounds,
        origin_x,
        origin_y,
        out_path,
        from_crs_code,
        to_crs_code,
        view_distance,
        resolution,
        observer_height,
    )


def viewshed_from_gpd(
    bldgs_gdf: gpd.GeoDataFrame,
    bounds: tuple[float, float, float, float],
    origin_x: float,
    origin_y: float,
    out_path: str,
    from_crs_code: int | str,
    to_crs_code: int | str | None = None,
    view_distance: int = 100,
    resolution: int = 1,
    observer_height: float = 1.5,
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
    observer_height: float = 1.5
        The height of the observer in metres. 1.5m by default.

    """
    write_path = _prepare_path(out_path)
    to_crs_code = _prepare_epsg_code(bounds, to_crs_code)
    bldgs_rast, transform = _prepare_bldgs_rast(bldgs_gdf, bounds, from_crs_code, to_crs_code, resolution)
    # prepare cell coordinates
    point_projected = util.project_geom(geometry.Point(origin_x, origin_y), from_crs_code, to_crs_code)
    x_idx, y_idx = ~transform * (point_projected.x, point_projected.y)
    x_idx = int(x_idx)
    y_idx = int(y_idx)
    # run viewshed
    viewshed_struct = rustalgos.viewshed.Viewshed()
    # find the viewshed
    rast = viewshed_struct.viewshed(bldgs_rast, view_distance, resolution, observer_height, x_idx, y_idx)
    # write to raster
    rgb_arr = np.dstack((rast, rast, rast, np.full_like(rast, 255))).astype(np.uint8)  # Fully opaque alpha channel
    rgb_arr[rast == 1] = [255, 0, 0, 255]  # red for view
    rgb_arr[rast == 0] = [0, 0, 0, 0]  # transparent for non-built
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
    observer_height: float = 1.5,
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
    observer_height: float = 1.5
        The height of the observer in metres. 1.5m by default.

    """
    # get buildings for buffered extents
    bldgs_gdf = _buildings_from_osmnx(bounds_wgs)
    # run viewshed
    viewshed(
        bldgs_gdf,
        bounds_wgs,
        origin_lng,
        origin_lat,
        out_path,
        4326,
        to_crs_code,
        view_distance,
        resolution,
        observer_height,
    )
