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
    # OSM Network from a Bounding Box

    Use a bounding box to create a `networkx` graph from OpenStreetMap data.

    Network data © [OpenStreetMap](https://www.openstreetmap.org/copyright) contributors, available under the Open Database Licence.
    """)
    return


@app.cell
def _():
    from cityseer.tools import io, plot
    from shapely import geometry

    return geometry, io, plot


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Use a shapely Polygon in WGS84 coordinates to define a boundary for the graph.
    """)
    return


@app.cell
def _(geometry):
    poly_wgs = geometry.box(-0.14115725966109327, 51.509220662095714, -0.12676440185383622, 51.51820111033659)
    poly_wgs
    return (poly_wgs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Use the [osm_graph_from_poly](https://cityseer.benchmarkurbanism.com/tools/io#osm-graph-from-poly) function available from the `cityseer` package's `io` module.

    The function expects a `shapely` polygon and returns a `networkx` graph. `cityseer` will automatically extract the graph topology while also creating accurate street geometries represented as `shapely` LineString geometries, which are linked to the graph's edges through the `geom` edge attribute.
    """)
    return


@app.cell
def _(io, poly_wgs):
    G = io.osm_graph_from_poly(poly_wgs)
    print(G)
    return (G,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The [`plot_nx`](https://cityseer.benchmarkurbanism.com/tools/plot#plot-nx) function can be used to visualize the graph. It accepts a `cityseer` prepared `networkx` graph and will plot street geometries when the `plot_geoms` parameter is set to `True`.
    """)
    return


@app.cell
def _(G, plot):
    plot.plot_nx(G, plot_geoms=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If using a different CRS, then specify this using the `poly_crs_code` parameter. For example, if you are using UTM zone 30N, then set this to the corresponding EPSG code, in this case [32630](https://epsg.io/32630).
    """)
    return


@app.cell
def _(geometry, io, plot):
    poly_utm = geometry.box(698361, 5710348, 699361, 5711348)
    G_utm = io.osm_graph_from_poly(poly_utm, poly_crs_code=32630)
    print(G_utm)
    plot.plot_nx(G_utm, plot_geoms=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `cityseer` will set node and edge geometry coordinates to the local projected UTM coordinate reference system. If you want to create the resulting graph in a specific CRS, then specify this using the `to_crs_code` parameter. For example, in the UK you might want to use the British National Grid, which has the EPSG code [27700](https://epsg.io/27700).
    """)
    return


@app.cell
def _(io, plot, poly_wgs):
    G_bng = io.osm_graph_from_poly(poly_wgs, to_crs_code=27700)
    print(G_bng)
    plot.plot_nx(G_bng, plot_geoms=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `cityseer` will automatically attempt to clean-up the graph by removing unnecessary nodes and simplifying the graph. If you only want basic cleaning without simplification, then set the `simplify` parameter to `False`. Note that in this case the graph has significantly more nodes and edges because simplification has not been applied.
    """)
    return


@app.cell
def _(io, plot, poly_wgs):
    G_raw = io.osm_graph_from_poly(poly_wgs, simplify=False)
    print(G_raw)
    plot.plot_nx(G_raw, plot_geoms=True)
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
    This notebook demonstrated how to create a `networkx` street network graph from OpenStreetMap data using a bounding box defined as a `shapely` Polygon. It covered specifying custom CRS codes for both the input polygon and the output graph, as well as toggling automatic network simplification on or off.

    To continue into analysis, hand the prepared graph to the high-level API with [`CityNetwork.from_nx`](https://cityseer.benchmarkurbanism.com/api/network#from_nx), which builds the dual network and exposes the centrality and land-use methods.

    **Next steps:** To create a network from a buffered point instead, see [Buffered Point](https://cityseer.benchmarkurbanism.com/examples/networks/create-from-buffered-point).
    """)
    return


if __name__ == "__main__":
    app.run()
