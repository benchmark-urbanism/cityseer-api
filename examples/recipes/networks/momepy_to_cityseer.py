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
    # Convert a Network from momepy

    Convert a network from `momepy` to a `cityseer` compatible `networkx` graph.

    The process for converting `momepy` networks is the same as for other street network datasets opened via `geopandas`.
    """)
    return


@app.cell
def _():
    import geopandas as gpd
    import momepy
    from cityseer.tools import io, plot

    return gpd, io, momepy, plot


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Prepare a `momepy` street network. This example uses the `streets` layer of the `momepy` `bubenec` example dataset, a `GeoDataFrame` of street segments.
    """)
    return


@app.cell
def _(gpd, momepy):
    streets_gpd = gpd.read_file(
        momepy.datasets.get_path("bubenec"),
        layer="streets",
    )
    streets_gpd.head()
    return (streets_gpd,)


@app.cell
def _(streets_gpd):
    streets_gpd.plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Use the [`nx_from_generic_geopandas`](https://cityseer.benchmarkurbanism.com/tools/io#nx-from-generic-geopandas) function to convert the `geopandas` LineStrings dataset to a `networkx` graph. This function will automatically create nodes and edges from the LineStrings in the dataset.
    """)
    return


@app.cell
def _(io, plot, streets_gpd):
    G = io.nx_from_generic_geopandas(streets_gpd)
    print(G)
    plot.plot_nx(G, plot_geoms=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusion

    This notebook demonstrated how to convert a `momepy` street network (stored as a `geopandas` GeoDataFrame of LineStrings) into a `cityseer`-compatible `networkx` graph using `nx_from_generic_geopandas`. The same approach applies to any LineString-based street dataset loaded through `geopandas`.

    To continue into analysis, hand the prepared graph to the high-level API with [`CityNetwork.from_nx`](https://cityseer.benchmarkurbanism.com/api/network#from_nx), which builds the dual network and exposes the centrality and land-use methods.

    **Next steps:** Once your network is ready, proceed to [Network Centrality](https://cityseer.benchmarkurbanism.com/examples/centrality) to compute metrics.
    """)
    return


if __name__ == "__main__":
    app.run()
