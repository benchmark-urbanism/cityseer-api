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
    # Visibility from Raster Data

    Generate a visibility analysis from raster data. This approach allows for blending building footprints with Digital Elevation Model (DEM) data to take elevation into account.

    Visibility analysis is provided by the standalone `visibility` module; it is not wrapped by the higher-level `CityNetwork` class.

    The bundled raster derives from building footprint and terrain data; sources and licensing are documented on the [datasets page](https://cityseer.benchmarkurbanism.com/examples/datasets).
    """)
    return


@app.cell
def _():
    from cityseer.metrics import visibility

    return (visibility,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Use the [`visibility.visibility_from_raster`](https://cityseer.benchmarkurbanism.com/metrics/visibility#visibility_from_raster) method to run a visibility analysis over the provided raster.

    Pay particular attention to:

    - The `input_path` parameter, which specifies the path to the input raster file. The file must exist and be in a format readable by `rasterio`, such as GeoTIFF. The resolution is inferred from the raster.
    - The `out_path` parameter, which specifies where to save the output files. The directory must exist. Outputs are written in GeoTIFF format (`.tif`), which can be opened in QGIS.

    There is a performance trade-off between the view distance and the input raster resolution: a larger view distance or finer resolution gives more detailed output but takes longer to compute and requires more memory.
    """)
    return


@app.cell
def _(visibility):
    visibility.visibility_from_raster(
        "images/madrid_rast_ht.tif",
        "images/madrid_vis_from_rast",
        view_distance=100,
        observer_height=1.5,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The output below shows the `harmonic` measure computed from a building raster that includes rooftop heights. No DEM data is used in this example.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="images/madrid_harmonic_100_2.png" style="width:100%;">
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusion

    This notebook generated a visibility analysis from raster data using `visibility_from_raster`, which can incorporate building heights and Digital Elevation Model (DEM) data to account for terrain. The `observer_height` parameter simulates pedestrian-level views, and the output rasters (density, farness, harmonic) capture visibility patterns influenced by both buildings and topography.

    **Next steps:** For street continuity analysis, see [Continuity from OSM](https://cityseer.benchmarkurbanism.com/examples/continuity/continuity-osm).
    """)
    return


if __name__ == "__main__":
    app.run()
