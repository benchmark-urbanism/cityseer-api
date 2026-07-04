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
    # Create a Dual Graph

    Create a dual graph representation from a primal graph.

    For purposes of visualisation and intuition, it is preferable to work with the network in its dual representation. Metrics such as centrality and accessibility are then calculated relative to streets instead of intersections, and the results can be visualised on the street geometries themselves. Angular (simplest-path) analysis requires the dual graph, since angular distances are tracked from street to street. Metric (shortest-path) analysis can also be run on the primal graph if preferred.

    > **Note:**
    > The high-level [`CityNetwork`](https://cityseer.benchmarkurbanism.com/api/network) class builds the dual graph automatically. The manual conversion shown here applies when working with the lower-level `networkx` based API.

    Network data © [OpenStreetMap](https://www.openstreetmap.org/copyright) contributors, available under the Open Database Licence.
    """)
    return


@app.cell
def _():
    from cityseer.tools import graphs, io, plot

    return graphs, io, plot


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    First, create a primal graph as before.
    """)
    return


@app.cell
def _(io, plot):
    lng, lat = -0.13396079424572427, 51.51371088849723
    buffer = 500
    poly_wgs, epsg_code = io.buffered_point_poly(lng, lat, buffer)
    G = io.osm_graph_from_poly(poly_wgs)
    print(G)
    plot.plot_nx(G, plot_geoms=True)
    return (G,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A dual graph representation can then be created using the [`nx_to_dual`](https://cityseer.benchmarkurbanism.com/tools/graphs#nx-to-dual) function available from the `cityseer` package's `graphs` module.
    """)
    return


@app.cell
def _(G, graphs):
    G_dual = graphs.nx_to_dual(G)
    print(G_dual)
    return (G_dual,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice that the dual graph will have the same number of nodes as the primal graph's number of edges, since the edges have been converted to nodes. The new edges are created by splitting the original edges at their midpoints, and then welding corresponding segments together across intersections. True representations of street geometries are therefore preserved in the dual graph, allowing for accurate calculations of metric distances and angular deviation.
    """)
    return


@app.cell
def _(G_dual, plot):
    plot.plot_nx(G_dual, plot_geoms=True)
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
    This notebook demonstrated how to convert a primal street network graph into its dual representation using `nx_to_dual`. In the dual graph, edges become nodes and new edges are formed by splitting and welding geometries across intersections, preserving accurate street geometries for metric and angular distance calculations. The dual representation is especially useful for visualising street-level centrality results.

    To continue into analysis, hand the *primal* graph to the high-level API with [`CityNetwork.from_nx`](https://cityseer.benchmarkurbanism.com/api/network#from_nx): it performs this dual conversion automatically and exposes the centrality and land-use methods.

    **Next steps:** The dual representation is recommended for centrality analysis; see [Metric Centrality](https://cityseer.benchmarkurbanism.com/examples/centrality/gpd-metric-centrality).
    """)
    return


if __name__ == "__main__":
    app.run()
