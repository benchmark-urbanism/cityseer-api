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
    # Boundaries and Live Nodes

    Use a boundary polygon to demarcate the study area from the buffered surrounding extents that are included to avoid edge rolloff. The algorithms continue to route through the parts of the network outside the boundary (thereby preventing edge rolloff), but results are only reported for nodes inside it.

    The high-level [`CityNetwork`](https://cityseer.benchmarkurbanism.com/api/network) class provides the [`set_boundary`](https://cityseer.benchmarkurbanism.com/api/network#set_boundary) method for this purpose. Under the hood it sets a `live` attribute per node: nodes inside the boundary are `live=True` and reported; nodes outside are `live=False` and used for routing only.

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
    When using this approach it is best to work in a projected coordinate reference system. In this example the buffered point polygon is cast to a projected CRS by using the `projected=True` parameter.
    """)
    return


@app.cell
def _(io):
    lng, lat = -0.13396079424572427, 51.51371088849723
    buffer = 500
    poly_utm, epsg_code = io.buffered_point_poly(lng, lat, buffer, projected=True)
    poly_utm
    return epsg_code, poly_utm


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Since the Polygon is in a projected CRS, the buffering step can proceed with a distance defined in metres. Here the Polygon is buffered by 400m since this is the maximum distance considered for the centrality analysis in the next step.
    """)
    return


@app.cell
def _(poly_utm):
    pol_buff = poly_utm.buffer(400)
    pol_buff
    return (pol_buff,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The buffer polygon can then be used as an argument for retrieving the OSM network with the [`from_osm`](https://cityseer.benchmarkurbanism.com/api/network#from_osm) constructor. The `poly_crs_code` and `to_crs_code` parameters need to be set so that the constructor knows which CRS the input geometry is in, and which CRS to convert the network to.
    """)
    return


@app.cell
def _(CityNetwork, epsg_code, pol_buff):
    cn = CityNetwork.from_osm(pol_buff, poly_crs_code=epsg_code, to_crs_code=epsg_code)
    return (cn,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Since the network and the source geometry are in the same CRS, `set_boundary` can now mark the study area with the original, unbuffered polygon: nodes inside it become `live=True`, nodes in the surrounding buffer become `live=False`.
    """)
    return


@app.cell
def _(cn, poly_utm):
    cn_bounded = cn.set_boundary(poly_utm)
    print(f"{int(cn_bounded.nodes_gdf.live.sum())} of {cn_bounded.node_count} nodes are live")
    return (cn_bounded,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Metrics can then be computed based on the target distance without encountering edge roll-off. (`set_boundary` returns the network itself, so it can also be chained directly after the constructor.)
    """)
    return


@app.cell
def _(cn_bounded):
    distances = [400]
    cn_bounded.centrality_shortest(distances=distances)
    nodes_gdf = cn_bounded.to_geopandas()
    nodes_gdf.head()
    return (nodes_gdf,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Values are only reported for live nodes; non-live nodes remain `NaN`.
    """)
    return


@app.cell
def _(nodes_gdf, plt):
    _fig, _ax = plt.subplots(1, 1, figsize=(7, 7), dpi=150)
    # non-live (buffer) nodes have NaN values; draw them as a faint grey context layer
    nodes_gdf[~nodes_gdf.live].plot(ax=_ax, color="#dddddd", linewidth=0.3)
    _g = nodes_gdf[nodes_gdf.live].copy()
    _g["_r"] = _g["cc_harmonic_400"].rank(pct=True)
    _g = _g.sort_values("_r")  # strongest drawn last, on top
    _g.plot(ax=_ax, color=plt.get_cmap("OrRd")(_g["_r"]), linewidth=0.15 + 2.25 * _g["_r"])
    _ax.set_title("Harmonic closeness, 400 m (full network; buffer in grey)", loc="left")
    _ax.set_axis_off()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To print or save only live nodes, filter the data frame accordingly.
    """)
    return


@app.cell
def _(nodes_gdf, plt):
    _fig, _ax = plt.subplots(1, 1, figsize=(7, 7), dpi=150)
    _g = nodes_gdf[nodes_gdf.live].copy()
    _g["_r"] = _g["cc_harmonic_400"].rank(pct=True)
    _g = _g.sort_values("_r")  # strongest drawn last, on top
    _g.plot(ax=_ax, color=plt.get_cmap("OrRd")(_g["_r"]), linewidth=0.15 + 2.25 * _g["_r"])
    _ax.set_title("Harmonic closeness, 400 m (live nodes only)", loc="left")
    _ax.set_axis_off()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    This notebook demonstrated how to demarcate a study area with `CityNetwork.set_boundary`, which sets `live=True` or `live=False` on network nodes based on whether they fall within the original (unbuffered) study area boundary. This prevents edge rolloff effects: the algorithms can route through the buffer zone, but results are only reported for live nodes within the study area.

    **Next steps:**

    - See [Network Preparation](https://cityseer.benchmarkurbanism.com/examples/networks) for different ways to create a network.
    - See [Metric Centrality](https://cityseer.benchmarkurbanism.com/examples/centrality/gpd-metric-centrality) or [Angular Centrality](https://cityseer.benchmarkurbanism.com/examples/centrality/gpd-angular-centrality) for centrality analysis on the prepared network.
    """)
    return


if __name__ == "__main__":
    app.run()
