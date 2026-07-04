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
    # Urban Analytics

    This lesson introduces two packages from the wider Python geospatial ecosystem: `osmnx`, for downloading data from OpenStreetMap, and `momepy`, for analysing urban form.

    Both packages are included in this repository's `examples` dependency group, so they are available automatically when you open the notebook with `uv run marimo edit`. If you are working in your own environment, install them first (e.g. `pip install osmnx momepy`).

    ### Importing Libraries

    Let's import the necessary packages:

    Network data © [OpenStreetMap](https://www.openstreetmap.org/copyright) contributors, available under the Open Database Licence.
    """)
    return


@app.cell
def _():
    # use an alias for convenience
    import osmnx as ox
    import momepy
    import matplotlib.pyplot as plt

    return momepy, ox, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## osmnx

    [`osmnx`](https://osmnx.readthedocs.io/) is a Python package that simplifies downloading and working with geospatial data from OpenStreetMap (OSM), such as building footprints or points of interest.

    This example demonstrates how to download building footprints within a 1km radius of a specified location in Nicosia, Cyprus, defined by its coordinates. `osmnx` provides several methods for downloading data; here, we will use [`features_from_point`](https://osmnx.readthedocs.io/en/stable/user-reference.html#osmnx.features.features_from_point).
    """)
    return


@app.cell
def _(ox):
    # Define the point of interest (latitude, longitude) and distance
    center_point = (35.17526, 33.36402)
    distance_m = 1000

    # Download building footprints
    gdf_buildings = ox.features_from_point(center_point, tags={"building": True}, dist=distance_m)
    return center_point, distance_m, gdf_buildings


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `osmnx` returns GeoDataFrames which, as shown in the previous lesson, are ideal for spatial analysis in Python. Note the `tags={"building": True}` argument, which instructs `osmnx` to fetch all features tagged as buildings in OSM. By changing these tags, you can also download other types of features, such as roads or parks.

    It is good practice to inspect the data you have downloaded. The `head()` function displays the first few rows of the GeoDataFrame, allowing you to quickly check the data structure and attributes.
    """)
    return


@app.cell
def _(gdf_buildings):
    # Display the first few rows of the buildings GeoDataFrame
    gdf_buildings.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can also plot the downloaded data:
    """)
    return


@app.cell
def _(center_point, distance_m, gdf_buildings, plt):
    _fig, _ax = plt.subplots(figsize=(10, 10), dpi=150)
    gdf_buildings.plot(ax=_ax)
    _ax.set_title(f"Buildings within {distance_m} m of the study point")
    _ax.set_axis_off()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For more detailed information on different ways to query OSM features data, refer to the `osmnx` [features documentation](https://osmnx.readthedocs.io/en/stable/user-reference.html#module-osmnx.features).

    ## Data Preparation

    To streamline the subsequent analysis, it is advisable to first filter for the types of geometry you intend to work with. In this instance, we are interested in polygon or multi-polygon geometries and will discard other types, such as points or linestrings.
    """)
    return


@app.cell
def _(gdf_buildings):
    # Filter out non-polygon geometries
    gdf_buildings_1 = gdf_buildings[gdf_buildings.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    gdf_buildings_1.head()
    return (gdf_buildings_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Secondly, we will reset the index so that all features are neatly indexed from zero upwards, without duplicates.
    """)
    return


@app.cell
def _(gdf_buildings_1):
    # Reset the index
    gdf_buildings_2 = gdf_buildings_1.reset_index(drop=True)
    gdf_buildings_2.head()
    return (gdf_buildings_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Thirdly, we will drop any columns not relevant to our analysis. In this case, we will retain the geometry column and the `building` column.
    """)
    return


@app.cell
def _(gdf_buildings_2):
    gdf_buildings_3 = gdf_buildings_2[["geometry", "building"]]
    gdf_buildings_3.head()
    return (gdf_buildings_3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Before performing morphological analysis, it is necessary to ensure your data is in a **projected Coordinate Reference System (CRS)**. Morphological metrics often involve measurements of distance and area, which are only accurate in a projected CRS. For Nicosia, we will use EPSG:3035 (ETRS89 / LAEA Europe), a European projection.
    """)
    return


@app.cell
def _(gdf_buildings_3):
    # Set the target CRS
    gdf_buildings_proj = gdf_buildings_3.to_crs(3035)
    return (gdf_buildings_proj,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's replot the building data to ensure everything is in order.
    """)
    return


@app.cell
def _(center_point, distance_m, gdf_buildings_proj, plt):
    _fig, _ax = plt.subplots(figsize=(10, 10), dpi=150)
    gdf_buildings_proj.plot(ax=_ax)
    _ax.set_title(f"Buildings within {distance_m} m of the study point, projected to EPSG:3035")
    _ax.set_axis_off()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## momepy

    [`momepy`](https://momepy.readthedocs.io/) is a library for the quantitative analysis of urban form, also known as urban morphology. It operates primarily on GeoDataFrames and provides a range of functions for calculating various morphological metrics.

    By way of example, we will explore two of these functions.

    ### Building Orientations

    `momepy` can calculate the orientation of building footprints using the [`orientation`](https://docs.momepy.org/en/stable/api/momepy.orientation.html) function.
    """)
    return


@app.cell
def _(gdf_buildings_proj, momepy, plt):
    gdf_buildings_proj["orientation"] = momepy.orientation(gdf_buildings_proj)
    _fig, _ax = plt.subplots(figsize=(10, 10), dpi=150)
    gdf_buildings_proj.plot(
        ax=_ax,
        column="orientation",
        cmap="viridis",
        legend=True,
        legend_kwds={"label": "Orientation (degrees from cardinal)", "shrink": 0.6},
    )
    _ax.set_title("Building orientations")
    _ax.set_axis_off()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Shared Walls

    `momepy` can calculate shared wall lengths between buildings using the [`shared_walls`](https://docs.momepy.org/en/stable/api/momepy.shared_walls.html) function.
    """)
    return


@app.cell
def _(gdf_buildings_proj, momepy, plt):
    gdf_buildings_proj["shared_wall_length"] = momepy.shared_walls(gdf_buildings_proj)
    _fig, _ax = plt.subplots(figsize=(10, 10), dpi=150)
    gdf_buildings_proj.plot(
        ax=_ax,
        column="shared_wall_length",
        cmap="magma",
        legend=True,
        legend_kwds={"label": "Shared wall length (m)", "shrink": 0.6},
    )
    _ax.set_title("Building shared wall lengths")
    _ax.set_axis_off()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Many other functions are available in `momepy` for calculating various morphological metrics. For a comprehensive list, please refer to the [momepy documentation](https://docs.momepy.org/en/stable/api.html).

    ### Exercises

    - Use `osmnx` to download a different feature type (e.g., parks with `tags={"leisure": "park"}`) for the same area. Plot the result.
    - Explore the `momepy` documentation and calculate a different morphological metric (e.g., `momepy.elongation` or `momepy.circular_compactness`) on the building footprints. What do the values tell you about building shapes?
    - Download building footprints for a location of your choice (a different city or neighbourhood). Compare building orientations between the two areas.

    ## Summary

    This has been a brief exploration of the broader Python geospatial ecosystem, focusing on `osmnx` for downloading urban data from OpenStreetMap and `momepy` for conducting urban morphology analysis. This is just a small sample of what these and other tools can achieve.

    Remember that with GeoPandas, you can export your data. So, after downloading data and running any variety of metrics using available Python packages, you can then export the file, which can be further visualised or manipulated with packages such as QGIS. For example, you can export your data to a GeoPackage file using the `to_file` method:
    """)
    return


@app.cell
def _(gdf_buildings_proj):
    gdf_buildings_proj.to_file("nicosia_buildings_metrics.gpkg", driver="GPKG")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next: [Data Science](https://cityseer.benchmarkurbanism.com/start/6-data-science) introduces dimensionality reduction and prediction with `scikit-learn`. The `osmnx` skills from this chapter are used in many of the [Cityseer Recipes](https://cityseer.benchmarkurbanism.com/examples), particularly the [Network Preparation](https://cityseer.benchmarkurbanism.com/examples/networks) examples.
    """)
    return


if __name__ == "__main__":
    app.run()
