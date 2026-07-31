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
    # Custom Network from a Streets Dataset

    Use `geopandas` to open a street network file and convert it to a `networkx` graph.

    If you have a street network file in a GeoPackage, shapefile, or similar format, then you can load this file and convert it to a `networkx` graph.

    The bundled datasets and their source attributions are documented on the [datasets page](https://cityseer.benchmarkurbanism.com/examples/datasets).
    """)
    return


@app.cell
def _():
    import geopandas as gpd
    from cityseer.tools import io, plot

    return gpd, io, plot


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Use `geopandas` to load the street network file. Check that your file path is correct!
    """)
    return


@app.cell
def _(gpd):
    streets_gpd = gpd.read_file("../../data/madrid_streets/street_network.gpkg")
    streets_gpd.head()
    return (streets_gpd,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Take note of your data's CRS.
    """)
    return


@app.cell
def _(streets_gpd):
    epsg_code = streets_gpd.crs.to_epsg()
    print(epsg_code)
    print(streets_gpd.crs.is_projected)
    return (epsg_code,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If your street network is not in a projected CRS, it is recommended to convert your dataset to a locally projected coordinate system. This can be done with the built-in `to_crs` method in `geopandas`. The EPSG code for the UTM zone can be found at [epsg.io](https://epsg.io/).

    Alternatively, you can project the graph after creation.
    """)
    return


@app.cell
def _(epsg_code, streets_gpd):
    # shown as example - unnecessary step for current dataset
    streets_gpd_1 = streets_gpd.to_crs(epsg=25830)
    print(epsg_code)
    print(streets_gpd_1.crs.is_projected)
    return (streets_gpd_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Take note of the geometry type.
    """)
    return


@app.cell
def _(streets_gpd_1):
    streets_gpd_1.geometry.type.unique()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If your data consists of `MultiLineString` geometries, then these should first be converted to unnested `LineString` geometries.
    """)
    return


@app.cell
def _(streets_gpd_1):
    streets_gpd_2 = streets_gpd_1.explode(ignore_index=True)
    # the source data contains a small number of exact duplicate geometries
    streets_gpd_2 = streets_gpd_2.drop_duplicates(subset="geometry")
    streets_gpd_2.geometry.type.unique()
    return (streets_gpd_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Use the [`nx_from_generic_geopandas`](https://cityseer.benchmarkurbanism.com/tools/io#nx-from-generic-geopandas) function to convert the `geopandas` LineStrings dataset to a `networkx` graph. This function will automatically create nodes and edges from the LineStrings in the dataset.

    The function expects that all features are `LineString` geometries, where the geometries represent individual street segments that meet at intersections. Street segments which share endpoints will be connected by a node in the graph.
    """)
    return


@app.cell
def _(io, streets_gpd_2):
    G = io.nx_from_generic_geopandas(streets_gpd_2)
    print(G)
    return (G,)


@app.cell
def _(G, plot):
    plot.plot_nx(
        G,
        plot_geoms=True,
        x_lim=(438500, 438500 + 3500),
        y_lim=(4472500, 4472500 + 3500),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If your data is not in a projected CRS (and if you did not already reproject the `GeoDataFrame` before creating the graph), then reproject your `networkx` graph to a locally projected coordinate system before doing further analysis.

    If the data is currently in geographic longitudes and latitudes (WGS84 / 4326) then the [`nx_wgs_to_utm`](https://cityseer.benchmarkurbanism.com/tools/io#nx-wgs-to-utm) function can be used to convert it to the local UTM projection. Alternatively, the [nx_epsg_conversion](https://cityseer.benchmarkurbanism.com/tools/io#nx-epsg-conversion) function can be used to specify input and output CRS.
    """)
    return


@app.cell
def _(G, io):
    # shown as an example - unnecessary step for current dataset
    G_utm = io.nx_epsg_conversion(G, to_crs_code=32630)
    print(G_utm)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusion

    This notebook demonstrated how to load a custom street network file (e.g. GeoPackage or shapefile) using `geopandas` and convert it to a `cityseer`-compatible `networkx` graph with `nx_from_generic_geopandas`. It covered handling MultiLineString geometries by exploding them to LineStrings, verifying the CRS, and optionally reprojecting the resulting graph.

    To continue into analysis, hand the prepared graph to the high-level API with [`CityNetwork.from_nx`](https://cityseer.benchmarkurbanism.com/api/network#from-nx), which builds the dual network and exposes the centrality and land-use methods.

    **Next steps:** To convert networks from other libraries, see [osmnx conversion](https://cityseer.benchmarkurbanism.com/examples/networks/osmnx-to-cityseer) or [momepy conversion](https://cityseer.benchmarkurbanism.com/examples/networks/momepy-to-cityseer).
    """)
    return


if __name__ == "__main__":
    app.run()
