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
    # Convert a Network from osmnx

    Convert a network from `osmnx` to a `cityseer` compatible `networkx` graph.

    Network data © [OpenStreetMap](https://www.openstreetmap.org/copyright) contributors, available under the Open Database Licence.
    """)
    return


@app.cell
def _():
    import osmnx as ox
    from cityseer.tools import plot, io

    return io, ox, plot


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Use one of the available `osmnx` methods to create a `networkx` graph.
    """)
    return


@app.cell
def _(ox):
    lng, lat = -0.14115725966109327, 51.509220662095714
    buff = 500

    multi_di_graph = ox.graph_from_point((lat, lng), dist=buff, simplify=True)
    print(multi_di_graph)
    return (multi_di_graph,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Use the [`nx_from_osm_nx`](https://cityseer.benchmarkurbanism.com/tools/io#nx-from-osm-nx) function from the `cityseer` `io` module to convert the `osmnx` dataset to a `cityseer` compatible `networkx` graph. By default the result is an undirected `MultiGraph`, appropriate for pedestrian analysis; pass `directed=True` to preserve one-way directionality for cycling or vehicular analysis.
    """)
    return


@app.cell
def _(io, multi_di_graph, plot):
    G = io.nx_from_osm_nx(multi_di_graph)
    print(G)
    plot.plot_nx(G, plot_geoms=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusion

    This notebook demonstrated how to convert an `osmnx` MultiDiGraph into a `cityseer`-compatible `networkx` MultiGraph using the `nx_from_osm_nx` function. This allows you to leverage `osmnx` for network retrieval while using `cityseer` for downstream analysis such as centrality and accessibility computations.

    To continue into analysis, hand the prepared graph to the high-level API with [`CityNetwork.from_nx`](https://cityseer.benchmarkurbanism.com/api/network#from_nx), which builds the dual network and exposes the centrality and land-use methods.

    **Next steps:** See also [momepy conversion](https://cityseer.benchmarkurbanism.com/examples/networks/momepy-to-cityseer) or proceed to [Network Centrality](https://cityseer.benchmarkurbanism.com/examples/centrality) to compute metrics.
    """)
    return


if __name__ == "__main__":
    app.run()
