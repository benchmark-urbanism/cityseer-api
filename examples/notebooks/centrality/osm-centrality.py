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
    # Network Centrality from OSM Data

    This notebook demonstrates how to calculate metric (shortest-path) distance centralities for a network loaded from OpenStreetMap (OSM) data. We will create a network from a buffered point, compute centrality measures, and visualise the results.

    The [`CityNetwork.from_osm`](https://cityseer.benchmarkurbanism.com/api/network#from-osm) constructor wraps the download, cleaning, and dual graph conversion behind a single call. For control over the intermediate steps, see the [network preparation recipes](https://cityseer.benchmarkurbanism.com/examples/networks).

    Network data © [OpenStreetMap](https://www.openstreetmap.org/copyright) contributors, available under the Open Database Licence.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    from cityseer.network import CityNetwork
    from cityseer.tools import io

    return CityNetwork, io, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Define the network extents. For example, from a relation id, bounding box, buffered point, or for extents defined from a loaded file. This example uses a buffered point.

    `from_osm` fetches and automatically simplifies the network. Using a simplified representation is recommended, otherwise centrality measures will be distorted for "messier" portions of the network.
    """)
    return


@app.cell
def _(CityNetwork, io):
    lng, lat = -0.13396079424572427, 51.51371088849723
    buffer = 1500
    poly_wgs, _epsg_code = io.buffered_point_poly(lng, lat, buffer)
    cn = CityNetwork.from_osm(poly_wgs)
    return (cn,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Use the [`centrality_shortest`](https://cityseer.benchmarkurbanism.com/api/network#centrality-shortest) method to calculate shortest metric distance centralities. By default it computes a single harmonic closeness and a single betweenness; here the `closeness` expression dictionary and `postprocess` are used to additionally derive the Hillier normalisation, which requires the `density` and `farness` components.

    > Use angular centralities with caution on automatically cleaned OSM networks, preferably only after visual inspection and manual cleaning.
    """)
    return


@app.cell
def _(cn):
    distances = [500, 1000]
    cn.centrality_shortest(
        distances=distances,
        closeness={"density": "1", "farness": "c", "harmonic": "1/c"},
        postprocess={"hillier": "density**2 / farness"},
    )
    nodes_gdf_1 = cn.to_geopandas()
    nodes_gdf_1.head()
    return (nodes_gdf_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exploring the results

    The centrality method adds new columns following the naming convention `cc_{centrality}_{distance}`. Inspect the columns and summary statistics to understand the distribution of values.
    """)
    return


@app.cell
def _(nodes_gdf_1):
    nodes_gdf_1.columns
    return


@app.cell
def _(nodes_gdf_1):
    nodes_gdf_1["cc_hillier_500"].describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualising centrality

    Visualise the results using the `geopandas` `.plot()` method. The first plot shows Hillier closeness at 500m and the second shows betweenness at 1000m.
    """)
    return


@app.cell
def _(nodes_gdf_1, plt):
    _fig, _ax = plt.subplots(1, 1, figsize=(7, 7), dpi=150)
    _g = nodes_gdf_1.copy()
    _g["_r"] = _g["cc_hillier_500"].rank(pct=True)
    _g = _g.sort_values("_r")  # strongest drawn last, on top
    _g.plot(ax=_ax, color=plt.get_cmap("OrRd")(_g["_r"]), linewidth=0.15 + 2.25 * _g["_r"])
    _ax.set_title("Hillier closeness, 500 m", loc="left")
    _ax.set_axis_off()
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(nodes_gdf_1, plt):
    _fig, _ax = plt.subplots(1, 1, figsize=(7, 7), dpi=150)
    _g = nodes_gdf_1.copy()
    _g["_r"] = _g["cc_betweenness_1000"].rank(pct=True)
    _g = _g.sort_values("_r")  # strongest drawn last, on top
    _g.plot(ax=_ax, color=plt.get_cmap("OrRd")(_g["_r"]), linewidth=0.15 + 2.25 * _g["_r"])
    _ax.set_title("Betweenness, 1000 m", loc="left")
    _ax.set_axis_off()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Betweenness highlights the streets that carry the most through-traffic potential. Notice how major roads and bridges tend to score highest, since many routes converge on them.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusion

    This notebook demonstrated how to compute metric distance centralities for a network loaded directly from OpenStreetMap through `CityNetwork.from_osm`, using a buffered point to define the area of interest. Closeness and betweenness centralities were computed and visualised at 500m and 1000m thresholds.

    **Next steps:** To add public transport data, see [GTFS Centrality](https://cityseer.benchmarkurbanism.com/examples/centrality/centrality-metro). For accessibility metrics, see [Accessibility](https://cityseer.benchmarkurbanism.com/examples/accessibility).
    """)
    return


if __name__ == "__main__":
    app.run()
