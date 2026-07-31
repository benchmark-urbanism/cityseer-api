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
    # Visibility from OSM Data

    Generate a visibility analysis from OpenStreetMap (OSM) data.

    Network data © [OpenStreetMap](https://www.openstreetmap.org/copyright) contributors, available under the Open Database Licence.

    Visibility analysis is provided by the standalone `visibility` module; it is not wrapped by the higher-level `CityNetwork` class.
    """)
    return


@app.cell
def _():
    from cityseer.metrics import visibility

    return (visibility,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Use the [`visibility.visibility_from_osm`](https://cityseer.benchmarkurbanism.com/metrics/visibility#visibility-from-osm) method to run a visibility analysis over building footprints downloaded from OSM.

    Pay particular attention to:

    - The `bounds_wgs` parameter, which defines the geographical area to analyse in longitude and latitude. The order is left, bottom, right, top.
    - The `out_path` parameter, which specifies where to save the output files. The directory must exist. Outputs are written in GeoTIFF format (`.tif`), which can be opened in QGIS.
    - The `to_crs_code` parameter, which specifies the coordinate reference system (CRS) of the output. The default is the local UTM zone, but any valid CRS code can be used.
    - The `view_distance` and `resolution` parameters, which control the analysis granularity.

    There is a performance trade-off between the bounds, view distance, and resolution: a larger view distance or finer resolution gives more detailed output but takes longer to compute and requires more memory.
    """)
    return


@app.cell
def _(visibility):
    visibility.visibility_from_osm(
        bounds_wgs=(18.41077, -33.93154, 18.42755, -33.91626),
        out_path="images/ct_vis",
        to_crs_code=None,  # defaults to local UTM
        view_distance=100,  # can use a larger view distance - but slower
        resolution=2,  # set resolution - e.g. 2m
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
    The output below shows `farness`, with building rooftops included.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="images/ct_farness_100_2.png" style="width:100%;">
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Higher visibility values indicate more open street environments with fewer obstructions, while lower values indicate more enclosed streets surrounded by buildings.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusion

    This notebook generated a visibility analysis directly from OpenStreetMap building footprints using `visibility_from_osm`. The method produces density, farness, and harmonic closeness rasters that quantify visual openness and enclosure across the urban area, written as GeoTIFF files for use in QGIS or further analysis.

    **Next steps:** To use building data from files, see [Visibility from GeoPandas](https://cityseer.benchmarkurbanism.com/examples/visibility/vis-gpd).
    """)
    return


if __name__ == "__main__":
    app.run()
