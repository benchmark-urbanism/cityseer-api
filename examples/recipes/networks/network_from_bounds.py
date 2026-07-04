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
    # OSM Network from a Boundary File

    Use a custom boundary file to create a `networkx` graph from OSM.

    If you have a boundary file in a GeoPackage, shapefile, or similar format, then you can load this file and use it as the boundary for creating a network from OSM.

    Network data © [OpenStreetMap](https://www.openstreetmap.org/copyright) contributors, available under the Open Database Licence.

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
    Use `geopandas` to load the boundary file. Check that your file path is correct!
    """)
    return


@app.cell
def _(gpd):
    bounds_gpd = gpd.read_file("../../data/madrid_nbhds/madrid_nbhds.gpkg")
    bounds_gpd
    return (bounds_gpd,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Extract the EPSG code from the boundary file.
    """)
    return


@app.cell
def _(bounds_gpd):
    epsg_code = bounds_gpd.crs.to_epsg()
    print(epsg_code)
    print(bounds_gpd.crs.is_projected)
    return (epsg_code,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If the geometry is not in a projected coordinate system, reproject it to a locally projected coordinate system before doing buffering or simplification. This can be done with the built-in `to_crs` method in `geopandas`. The EPSG code for the UTM zone can be found at [epsg.io](https://epsg.io/). For example, if your boundary is in the UTM zone 30N, you can use code 32630.

    For the currently opened file, the boundary is already in a projected coordinate system, so this step is unnecessary. When working with a file that does need to be reprojected, use the following code to project to a given EPSG code.
    """)
    return


@app.cell
def _(bounds_gpd, epsg_code):
    # shown as example - unnecessary step for current dataset
    bounds_gpd_1 = bounds_gpd.to_crs(epsg=25830)
    print(epsg_code)
    print(bounds_gpd_1.crs.is_projected)
    return (bounds_gpd_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Convert the `GeoDataFrame` into a Polygon using the `geopandas` `union_all` method. This will create a single unified `shapely` geometry.
    """)
    return


@app.cell
def _(bounds_gpd_1):
    centro_gpd = bounds_gpd_1[bounds_gpd_1["NOMDIS"] == "Centro"]
    bounds_geom = centro_gpd.union_all()
    bounds_geom
    return bounds_geom, centro_gpd


@app.cell
def _(centro_gpd):
    centro_gpd
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If the geometries are complex, then the OSM data request may fail due to URL length limitations (because each coordinate has to be passed in the URL). To avoid this, you can use the `convex_hull` or else the `simplify` method to reduce the number of points in the geometry.
    """)
    return


@app.cell
def _(bounds_geom):
    bounds_geom_simpl = bounds_geom.convex_hull
    bounds_geom_simpl
    return (bounds_geom_simpl,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    It is best practice to buffer the geometry by an amount matching the farthest distance used for centrality or accessibility calculations, which prevents edge rolloff effects.
    """)
    return


@app.cell
def _(bounds_geom_simpl):
    bounds_geom_buff = bounds_geom_simpl.buffer(800)
    bounds_geom_buff
    return (bounds_geom_buff,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `shapely` Polygon can now be used as before to download and prepare an OSM graph.
    """)
    return


@app.cell
def _(bounds_geom_buff, epsg_code, io):
    G = io.osm_graph_from_poly(
        bounds_geom_buff,
        poly_crs_code=epsg_code,
        simplify=True,
    )
    print(G)
    return (G,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Setting the `x_lim` and `y_lim` extents will effectively zoom-in so that the results of the query can be seen more clearly.
    """)
    return


@app.cell
def _(G, plot):
    plot.plot_nx(
        G,
        plot_geoms=True,
        x_lim=(439500, 439500 + 2000),
        y_lim=(4473500, 4473500 + 2000),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusion
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This notebook demonstrated how to load a custom boundary file (e.g. GeoPackage or shapefile) with `geopandas`, prepare the geometry through union, simplification, and buffering, and then use it to download and create an OSM street network graph. Buffering the boundary by the analysis distance is recommended to prevent edge rolloff effects in subsequent centrality or accessibility calculations.

    To continue into analysis, hand the prepared graph to the high-level API with [`CityNetwork.from_nx`](https://cityseer.benchmarkurbanism.com/api/network#from_nx), which builds the dual network and exposes the centrality and land-use methods.

    **Next steps:** To use an OSM relation ID instead, see [Relation ID](https://cityseer.benchmarkurbanism.com/examples/networks/network-from-relation-id).
    """)
    return


if __name__ == "__main__":
    app.run()
