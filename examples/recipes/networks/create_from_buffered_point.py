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
    # OSM Network from a Buffered Coordinate

    Use a buffered point to create a `networkx` graph from OSM.

    Use this approach when you have a centre-point, such as a longitude and latitude, and you wish to create a network within a certain radius distance.

    Network data © [OpenStreetMap](https://www.openstreetmap.org/copyright) contributors, available under the Open Database Licence.
    """)
    return


@app.cell
def _():
    from cityseer.tools import io, plot

    return io, plot


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Define a longitude and latitude, and a buffer distance in meters.
    """)
    return


@app.cell
def _():
    lng, lat = -0.13396079424572427, 51.51371088849723
    buffer = 500
    return buffer, lat, lng


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A geometry can now be created using the [`buffered_point_poly`](https://cityseer.benchmarkurbanism.com/tools/io#buffered-point-poly) function available from the `cityseer` package's `io` module.
    """)
    return


@app.cell
def _(buffer, io, lat, lng):
    poly_wgs, epsg_code = io.buffered_point_poly(lng, lat, buffer)
    print(epsg_code)
    poly_wgs
    return (poly_wgs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can then use this geometry to specify the boundaries to be used for creating a network using the [osm_graph_from_poly](https://cityseer.benchmarkurbanism.com/tools/io#osm-graph-from-poly) function.
    """)
    return


@app.cell
def _(io, plot, poly_wgs):
    G = io.osm_graph_from_poly(poly_wgs, green_footways=False)  # False by default
    print(G)
    plot.plot_nx(G, plot_geoms=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Set the `projected` parameter to `True` if you would rather use the local UTM projection for the boundary.
    """)
    return


@app.cell
def _(buffer, io, lat, lng):
    poly_utm, epsg_code_1 = io.buffered_point_poly(lng, lat, buffer, projected=True)
    print(epsg_code_1)
    poly_utm
    return epsg_code_1, poly_utm


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this case, remember to set the `poly_crs_code` parameter to the appropriate UTM CRS if using the resultant geometry as a parameter for the [osm_graph_from_poly](https://cityseer.benchmarkurbanism.com/tools/io#osm-graph-from-poly) function.
    """)
    return


@app.cell
def _(epsg_code_1, io, plot, poly_utm):
    G_utm = io.osm_graph_from_poly(poly_utm, poly_crs_code=epsg_code_1)
    print(G_utm)
    plot.plot_nx(G_utm, plot_geoms=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusion

    This notebook demonstrated how to create a street network graph from OpenStreetMap by buffering a longitude/latitude coordinate to a specified radius using `buffered_point_poly`. This approach is useful when you have a centre-point and want to extract the surrounding network within a given distance, with options for both WGS84 and projected UTM boundaries.

    To continue into analysis, hand the prepared graph to the high-level API with [`CityNetwork.from_nx`](https://cityseer.benchmarkurbanism.com/api/network#from_nx), which builds the dual network and exposes the centrality and land-use methods.

    **Next steps:** To use a custom boundary file instead, see [Boundary File](https://cityseer.benchmarkurbanism.com/examples/networks/network-from-bounds).
    """)
    return


if __name__ == "__main__":
    app.run()
