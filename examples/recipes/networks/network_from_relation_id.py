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
    # OSM Network from a Relation ID

    Use an OSM relation id to create a `networkx` graph from OSM.

    Use OpenStreetMap to identify a boundary relations id. Then, use `osmnx` to retrieve the relation id as a `GeoDataFrame`.

    Network data © [OpenStreetMap](https://www.openstreetmap.org/copyright) contributors, available under the Open Database Licence.
    """)
    return


@app.cell
def _():
    import osmnx as ox
    from cityseer.tools import io, plot

    return io, ox, plot


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this case, we'll retrieve the London [Soho](https://www.openstreetmap.org/relation/17710512) boundary relation `17710512`. For this, we can use the `osmnx` [`geocode_to_gdf`](https://osmnx.readthedocs.io/en/stable/user-reference.html#osmnx.geocoder.geocode_to_gdf) function with the `by_osmid` parameter set to `True`. Note the preceding `R` prepended to the id as per the API documentation.
    """)
    return


@app.cell
def _(ox):
    bounds_gdf = ox.geocode_to_gdf(
        "R17710512",  # OSM relation ID
        by_osmid=True,
    )
    bounds_gdf
    return (bounds_gdf,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Convert the `GeoDataFrame` into a unified Polygon using the `geopandas` `union_all` method.
    """)
    return


@app.cell
def _(bounds_gdf):
    # returns a geoDataFrame - union for the geom
    bounds_geom = bounds_gdf.union_all()
    bounds_geom
    return (bounds_geom,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If the geometry is complex, then the subsequent OSM API request (which happens behind the scenes) can fail due to a long URL request. In this case, you can take the geometrically simpler `convex_hull` instead. Alternatively, you can simplify the geometry, but remember to use spatial units matching the CRS of the geometry.
    """)
    return


@app.cell
def _(bounds_geom):
    bounds_geom_1 = bounds_geom.convex_hull
    bounds_geom_1
    return (bounds_geom_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `shapely` geometry can then be passed to the `cityseer` [osm_graph_from_poly](https://cityseer.benchmarkurbanism.com/tools/io#osm-graph-from-poly) function, as per other examples.
    """)
    return


@app.cell
def _(bounds_geom_1, io):
    G = io.osm_graph_from_poly(bounds_geom_1)
    print(G)
    return (G,)


@app.cell
def _(G, plot):
    plot.plot_nx(G, plot_geoms=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusion

    This notebook demonstrated how to retrieve an administrative or neighbourhood boundary from OpenStreetMap using an OSM relation ID via `osmnx`, and then use that boundary to create a `cityseer`-compatible street network graph. This approach is convenient when a well-defined boundary already exists in OSM, avoiding the need to supply a custom boundary file.

    To continue into analysis, hand the prepared graph to the high-level API with [`CityNetwork.from_nx`](https://cityseer.benchmarkurbanism.com/api/network#from_nx), which builds the dual network and exposes the centrality and land-use methods.

    **Next steps:** To create a network from a custom streets dataset, see [Streets Dataset](https://cityseer.benchmarkurbanism.com/examples/networks/network-from-streets).
    """)
    return


if __name__ == "__main__":
    app.run()
