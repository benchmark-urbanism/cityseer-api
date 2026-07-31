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
    # Saving a Network to a File

    Save a `cityseer` prepared `networkx` graph to a file.

    To save a graph or to visualise it from QGIS, convert it to a `GeoDataFrame` and save it using `geopandas`.

    First, let's create a simple OSM graph from a bounding box, using the same approach as before.

    Network data © [OpenStreetMap](https://www.openstreetmap.org/copyright) contributors, available under the Open Database Licence.
    """)
    return


@app.cell
def _():
    from cityseer.tools import io
    from shapely import geometry

    poly_wgs = geometry.box(
        -0.14115725966109327,
        51.509220662095714,
        -0.12676440185383622,
        51.51820111033659,
    )
    G = io.osm_graph_from_poly(poly_wgs)
    print(G)
    return G, io


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Use the [`geopandas_from_nx`](https://cityseer.benchmarkurbanism.com/tools/io#geopandas-from-nx) function to convert the `networkx` dataset into a `geopandas` LineString `GeoDataFrame`.
    """)
    return


@app.cell
def _(G, io):
    streets_gpd = io.geopandas_from_nx(G)
    streets_gpd.head()
    return (streets_gpd,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Since this is now a `geopandas` `GeoDataFrame`, you can use it accordingly.
    """)
    return


@app.cell
def _(streets_gpd):
    streets_gpd.plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `geopandas` can then save the file to disk, after which it can be accessed and edited from an application such as QGIS.
    """)
    return


@app.cell
def _(streets_gpd):
    from pathlib import Path

    out_dir = Path("temp")
    out_dir.mkdir(exist_ok=True)
    streets_gpd.to_file(
        out_dir / "save_streets_demo.gpkg",
        driver="GPKG",
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
    This notebook demonstrated how to convert a `cityseer`-prepared `networkx` graph into a `geopandas` GeoDataFrame using `geopandas_from_nx`, and then save it to disk as a GeoPackage file. This workflow is useful for exporting networks for inspection in GIS applications such as QGIS or for sharing datasets with collaborators.

    When working with the high-level [`CityNetwork`](https://cityseer.benchmarkurbanism.com/api/network) class, networks and computed metrics can instead be persisted with its `save` method and restored with `CityNetwork.load`. To continue into analysis, hand the prepared graph to the high-level API with [`CityNetwork.from_nx`](https://cityseer.benchmarkurbanism.com/api/network#from-nx), which builds the dual network and exposes the centrality and land-use methods.

    **Next steps:** For analysis, see [Network Centrality](https://cityseer.benchmarkurbanism.com/examples/centrality) or [Accessibility](https://cityseer.benchmarkurbanism.com/examples/accessibility).
    """)
    return


if __name__ == "__main__":
    app.run()
