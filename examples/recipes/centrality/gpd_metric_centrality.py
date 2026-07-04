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
    # Metric Distance Network Centrality

    Calculate metric distance centralities from a `geopandas` `GeoDataFrame`.

    This notebook uses the high-level [`CityNetwork`](https://cityseer.benchmarkurbanism.com/api/network) class, which wraps network preparation (including the dual graph) and centrality behind a single interface.

    The bundled datasets and their source attributions are documented on the [datasets page](https://cityseer.benchmarkurbanism.com/examples/datasets).
    """)
    return


@app.cell
def _():
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from cityseer.network import CityNetwork

    return CityNetwork, gpd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Load the streets and clip them to a buffered study area. The `boundary` argument to [`from_geopandas`](https://cityseer.benchmarkurbanism.com/api/network#from_geopandas) marks the nodes inside the study area as `live`; the surrounding non-live nodes prevent edge rolloff, since they are used for routing but metrics are not computed for them. Because each node's centrality depends only on its catchment within the largest analysis distance, a district plus a sufficiently large buffer reproduces the city-wide values for the live nodes at far lower cost. See the [live nodes example](https://cityseer.benchmarkurbanism.com/examples/recipes/live-nodes) for more details.
    """)
    return


@app.cell
def _(CityNetwork, gpd):
    streets_gpd = gpd.read_file("../../data/madrid_streets/street_network.gpkg")
    streets_gpd = streets_gpd.explode(ignore_index=True)
    # the source data contains a small number of exact duplicate geometries
    streets_gpd = streets_gpd.drop_duplicates(subset="geometry")
    # 3km study area around the city centre, with the network buffered a further 10km
    # (the largest analysis distance used below) to prevent edge rolloff
    centre = gpd.GeoSeries.from_xy([440300], [4474300], crs=streets_gpd.crs)
    study_poly = centre.buffer(3000).iloc[0]
    buffered_poly = centre.buffer(13000).iloc[0]
    streets_clip = streets_gpd[streets_gpd.intersects(buffered_poly)]
    cn = CityNetwork.from_geopandas(streets_clip, boundary=study_poly)
    return (cn,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Use the [`centrality_shortest`](https://cityseer.benchmarkurbanism.com/api/network#centrality_shortest) method to calculate shortest metric distance centralities. It can calculate centralities for numerous distances at once via the `distances` parameter, which accepts a list of distances. By default it computes a single harmonic closeness and a single betweenness; further metrics can be requested through the expression dictionaries shown later in this notebook.

    Results are written to the network's internal nodes `GeoDataFrame` as columns named `cc_{centrality}_{distance}`, and `to_geopandas` returns them joined to the original street LineString geometries. Standard `geopandas` functionality can be used to explore, visualise, or save the results. See the [documentation](https://cityseer.benchmarkurbanism.com/metrics/networks#centrality_shortest) for more information on the available centrality formulations.
    """)
    return


@app.cell
def _(cn):
    _distances = [500, 2000]
    cn.centrality_shortest(distances=_distances)
    nodes_gdf_1 = cn.to_geopandas()
    nodes_gdf_1.head()
    return (nodes_gdf_1,)


@app.cell
def _(nodes_gdf_1):
    nodes_gdf_1.columns
    return


@app.cell
def _(nodes_gdf_1):
    nodes_gdf_1["cc_betweenness_2000"].describe()
    return


@app.cell
def _(nodes_gdf_1, plt):
    _fig, _ax = plt.subplots(1, 1, figsize=(8, 6), dpi=150)
    nodes_gdf_1.plot(
        column="cc_harmonic_500",
        cmap="magma",
        legend=True,
        legend_kwds={"label": "Harmonic closeness, 500 m", "shrink": 0.6},
        ax=_ax,
    )
    _ax.set_title("Harmonic closeness, 500 m")
    _ax.set_xlim(438500, 438500 + 3500)
    _ax.set_ylim(4472500, 4472500 + 3500)
    _fig.tight_layout()
    _ax.axis(False)
    _fig
    return


@app.cell
def _(nodes_gdf_1, plt):
    _fig, _ax = plt.subplots(1, 1, figsize=(8, 6), dpi=150)
    nodes_gdf_1.plot(
        column="cc_betweenness_2000",
        cmap="magma",
        legend=True,
        legend_kwds={"label": "Betweenness, 2000 m", "shrink": 0.6},
        ax=_ax,
    )
    _ax.set_title("Betweenness, 2000 m")
    _ax.set_xlim(438500, 438500 + 3500)
    _ax.set_ylim(4472500, 4472500 + 3500)
    _fig.tight_layout()
    _ax.axis(False)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Alternatively, you can define the distance thresholds using a list of `minutes` instead.
    """)
    return


@app.cell
def _(cn, nodes_gdf_1):
    cn.centrality_shortest(minutes=[15])
    nodes_gdf_2 = cn.to_geopandas()
    nodes_gdf_2[[c for c in nodes_gdf_2.columns if c not in nodes_gdf_1.columns]].head()
    return (nodes_gdf_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The method will map the minutes values into the equivalent distances, which are reported in the logged output.
    """)
    return


@app.cell
def _(nodes_gdf_2):
    nodes_gdf_2.columns
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As per the function logging outputs, 15 minutes has been mapped to 1200m at default `speed_m_s`, so the corresponding outputs can be visualised using the 1200m columns. Use the configurable `speed_m_s` parameter to set a custom metres per second walking speed.
    """)
    return


@app.cell
def _(nodes_gdf_2, plt):
    _fig, _ax = plt.subplots(1, 1, figsize=(8, 6), dpi=150)
    nodes_gdf_2.plot(
        column="cc_harmonic_1200",
        cmap="magma",
        legend=True,
        legend_kwds={"label": "Harmonic closeness, 1200 m", "shrink": 0.6},
        ax=_ax,
    )
    _ax.set_title("Harmonic closeness, 1200 m (15 minutes)")
    _ax.set_xlim(438500, 438500 + 3500)
    _ax.set_ylim(4472500, 4472500 + 3500)
    _fig.tight_layout()
    _ax.axis(False)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Custom metric expressions

    Centrality metrics are defined as `{name: expression}` dictionaries over two variables: `c`, the routing cost (metric distance in metres), and `p`, normalised progress from 0 at the source to 1 at the distance threshold. Passing `{}` skips a category. The `decay` metric, `exp(-4 * p)`, is the continuous equivalent of the historical beta-weighted closeness. The following example computes harmonic and decay-weighted closeness at 400m and skips betweenness.
    """)
    return


@app.cell
def _(cn, nodes_gdf_2):
    cn.centrality_shortest(
        distances=[400],
        closeness={"harmonic": "1/c", "decay": "exp(-4 * p)"},
        betweenness={},
    )
    nodes_gdf_3 = cn.to_geopandas()
    nodes_gdf_3[[c for c in nodes_gdf_3.columns if c not in nodes_gdf_2.columns]].head()
    return (nodes_gdf_3,)


@app.cell
def _(nodes_gdf_3):
    nodes_gdf_3.columns
    return


@app.cell
def _(nodes_gdf_3, plt):
    _fig, _ax = plt.subplots(1, 1, figsize=(8, 6), dpi=150)
    nodes_gdf_3.plot(
        column="cc_decay_400",
        cmap="magma",
        legend=True,
        legend_kwds={"label": "Decay-weighted closeness, 400 m", "shrink": 0.6},
        ax=_ax,
    )
    _ax.set_title("Decay-weighted closeness, 400 m")
    _ax.set_xlim(438500, 438500 + 3500)
    _ax.set_ylim(4472500, 4472500 + 3500)
    _fig.tight_layout()
    _ax.axis(False)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sampled centrality for larger distances
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For larger distance thresholds, the computational cost increases substantially. The `centrality_shortest` method supports a `sample=True` parameter that uses a sampling strategy to compute centralities more efficiently at larger scales while maintaining a relatively high degree of rank performance. Use non-sampled computation for publication-quality results.
    """)
    return


@app.cell
def _(cn, nodes_gdf_3):
    _distances = [500, 10000]
    cn.centrality_shortest(distances=_distances, sample=True)
    nodes_gdf_4 = cn.to_geopandas()
    sorted(c for c in nodes_gdf_4.columns if c not in nodes_gdf_3.columns)
    return (nodes_gdf_4,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For shorter distances where the number of reachable nodes is small, full computation is used. For larger distances, the function applies sampling to achieve the target accuracy efficiently.
    """)
    return


@app.cell
def _(nodes_gdf_4, plt):
    _fig, _ax = plt.subplots(1, 1, figsize=(8, 6), dpi=150)
    nodes_gdf_4.plot(
        column="cc_betweenness_10000",
        cmap="magma",
        legend=True,
        legend_kwds={"label": "Betweenness, 10000 m", "shrink": 0.6},
        ax=_ax,
    )
    _ax.set_title("Betweenness, 10000 m (sampled)")
    _ax.set_xlim(438500, 438500 + 3500)
    _ax.set_ylim(4472500, 4472500 + 3500)
    _fig.tight_layout()
    _ax.axis(False)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tolerance for near-shortest paths
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `tolerance` parameter allows betweenness to count paths that are within a given percentage of the shortest path. For example, `tolerance=2.0` treats paths within 2% of the shortest as equally valid. This can produce more realistic betweenness distributions by accounting for near-optimal route choices.
    """)
    return


@app.cell
def _(cn, nodes_gdf_4):
    _distances = [5000]
    cn.centrality_shortest(distances=_distances, sample=True, tolerance=0.5)
    nodes_gdf_5 = cn.to_geopandas()
    sorted(c for c in nodes_gdf_5.columns if c not in nodes_gdf_4.columns)
    return (nodes_gdf_5,)


@app.cell
def _(nodes_gdf_5, plt):
    _fig, _ax = plt.subplots(1, 1, figsize=(8, 6), dpi=150)
    nodes_gdf_5.plot(
        column="cc_betweenness_5000",
        cmap="magma",
        legend=True,
        legend_kwds={"label": "Betweenness, 5000 m", "shrink": 0.6},
        ax=_ax,
    )
    _ax.set_title("Betweenness, 5000 m (tolerance 0.5%)")
    _ax.set_xlim(438500, 438500 + 3500)
    _ax.set_ylim(4472500, 4472500 + 3500)
    _fig.tight_layout()
    _ax.axis(False)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    This notebook demonstrated how to calculate metric (shortest-path) centralities from a `geopandas` `GeoDataFrame` with the `CityNetwork` class. It covered marking live nodes from a study area polygon to prevent edge rolloff, computing centralities at multiple distance thresholds, using `minutes` as an alternative way to specify thresholds, defining custom metric expressions, and using the `sample` parameter for efficient computation at larger distances. It also showed the `tolerance` parameter for near-shortest path betweenness.

    **Next steps:**

    - To calculate angular (simplest-path) centralities instead, see [Angular Centrality](https://cityseer.benchmarkurbanism.com/examples/centrality/gpd-angular-centrality).
    - To calculate centralities directly from OpenStreetMap data, see [OSM Centrality](https://cityseer.benchmarkurbanism.com/examples/centrality/osm-centrality).
    - To compute accessibility or mixed-use metrics over the same network, see the [Accessibility](https://cityseer.benchmarkurbanism.com/examples/accessibility) recipes.
    """)
    return


if __name__ == "__main__":
    app.run()
