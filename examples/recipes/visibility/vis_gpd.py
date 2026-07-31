import marimo

__generated_with = "0.23.13"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Visibility from GeoPandas Data

    Generate a visibility analysis from GeoPandas data (or a file opened with GeoPandas).

    The bundled datasets and their source attributions are documented on the [datasets page](https://cityseer.benchmarkurbanism.com/examples/datasets).

    Visibility analysis is provided by the standalone `visibility` module; it is not wrapped by the higher-level `CityNetwork` class.
    """)
    return


@app.cell
def _():
    import geopandas as gpd
    from cityseer.metrics import visibility

    return gpd, visibility


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Open a GeoPandas dataset.
    """)
    return


@app.cell
def _(gpd):
    bldgs_gdf = gpd.read_file("../../data/madrid_buildings/madrid_bldgs.gpkg")
    bldgs_gdf.head()
    return (bldgs_gdf,)


@app.cell
def _(bldgs_gdf):
    bldgs_gdf.crs
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Use the [`visibility.visibility_from_gpd`](https://cityseer.benchmarkurbanism.com/metrics/visibility#visibility-from-gpd) method to run a visibility analysis over the provided building polygons.

    Pay particular attention to:

    - The `bounds` parameter, which defines the geographical area in the same CRS as your dataset. The order is left, bottom, right, top.
    - The `out_path` parameter, which specifies where to save the output files. The directory must exist. Outputs are written in GeoTIFF format (`.tif`), which can be opened in QGIS.
    - The `from_crs_code` parameter, which specifies the coordinate reference system (CRS) of the input data. This should match the CRS of your GeoPandas dataset; check it with `.crs` if unsure.
    - The `to_crs_code` parameter, which specifies the CRS of the output. The default is the local UTM zone, but any valid CRS code can be used.
    - The `view_distance` and `resolution` parameters, which control the analysis granularity.

    There is a performance trade-off between the bounds, view distance, and resolution: a larger view distance or finer resolution gives more detailed output but takes longer to compute and requires more memory.
    """)
    return


@app.cell
def _(bldgs_gdf, visibility):
    visibility.visibility_from_gpd(
        bldgs_gdf,
        bounds=(439658, 4473632, 440914, 4474972),
        out_path="images/madrid_vis",
        from_crs_code=25830,  # set geopandas CRS
        to_crs_code=25830,  # set output CRS
        view_distance=150,  # can use a larger view distance - but slower
        resolution=2,  # can use a higher resolution - but slower
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The analysis writes a GeoTIFF per measure, plus a rasterised-buildings file for checking the inputs:

    - A `density` measure: the number of visible pixels.
    - A `farness` measure: the summed distance to visible pixels, which favours farther views (up to `view_distance`).
    - A `harmonic` closeness measure: the summed inverse distance to visible pixels, which favours close adjacency to open spaces.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The output below shows `farness`, with values clipped to de-accentuate rooftops while recovering detail at ground level. The figure was rendered at resolution 1 (1m cells); the notebook runs at the coarser resolution 2 to keep the runtime short, so a fresh run reproduces the pattern at lower detail.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="images/madrid_farness_150_1.png" style="width:100%;">
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusion

    This notebook generated a visibility analysis from a GeoPandas GeoDataFrame of building footprints using `visibility_from_gpd`. Given bounds, CRS, view distance, and resolution, the method produces density, farness, and harmonic closeness rasters that capture the spatial pattern of visual openness and enclosure across the study area.

    **Next steps:** To incorporate terrain elevation data, see [Visibility from Raster](https://cityseer.benchmarkurbanism.com/examples/visibility/vis-raster).
    """)
    return


if __name__ == "__main__":
    app.run()
