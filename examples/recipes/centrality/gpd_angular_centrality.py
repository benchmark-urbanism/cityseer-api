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
    # Angular Distance Network Centrality

    Calculate angular (geometric or "simplest") distance centralities from a `geopandas` `GeoDataFrame`.

    This notebook uses the high-level [`CityNetwork`](https://cityseer.benchmarkurbanism.com/api/network) class, which builds the dual graph required for angular analysis automatically.

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
    Use the [`centrality_simplest`](https://cityseer.benchmarkurbanism.com/api/network#centrality_simplest) method to calculate angular (geometric or "simplest") distance centralities. Angular analysis requires the dual graph, which `CityNetwork` builds automatically during construction.

    The method can calculate centralities for numerous distances at once via the `distances` parameter, which accepts a list of distances. By default it computes a single angular harmonic closeness and a single angular betweenness.

    Results are written as columns named `cc_{centrality}_{distance}_ang`, and `to_geopandas` returns them joined to the original street LineString geometries. Standard `geopandas` functionality can be used to explore, visualise, or save the results. See the documentation for more information on the available centrality formulations.
    """)
    return


@app.cell
def _(cn):
    _distances = [500, 2000]
    cn.centrality_simplest(distances=_distances)
    nodes_gdf_1 = cn.to_geopandas()
    nodes_gdf_1.head()
    return (nodes_gdf_1,)


@app.cell
def _(nodes_gdf_1):
    nodes_gdf_1.columns
    return


@app.cell
def _(nodes_gdf_1):
    nodes_gdf_1["cc_betweenness_2000_ang"].describe()
    return


@app.cell
def _(nodes_gdf_1, plt):
    _fig, _ax = plt.subplots(1, 1, figsize=(8, 6), dpi=150)
    nodes_gdf_1.plot(
        column="cc_harmonic_500_ang",
        cmap="magma",
        legend=True,
        legend_kwds={"label": "Angular harmonic closeness, 500 m", "shrink": 0.6},
        ax=_ax,
    )
    _ax.set_title("Angular harmonic closeness, 500 m")
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
        column="cc_betweenness_2000_ang",
        cmap="magma",
        legend=True,
        legend_kwds={"label": "Angular betweenness, 2000 m", "shrink": 0.6},
        ax=_ax,
    )
    _ax.set_title("Angular betweenness, 2000 m")
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
    cn.centrality_simplest(minutes=[15])
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
        column="cc_harmonic_1200_ang",
        cmap="magma",
        legend=True,
        legend_kwds={"label": "Angular harmonic closeness, 1200 m", "shrink": 0.6},
        ax=_ax,
    )
    _ax.set_title("Angular harmonic closeness, 1200 m (15 minutes)")
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
    For larger distance thresholds, the computational cost increases substantially. The `centrality_simplest` method supports a `sample=True` parameter that uses a sampling strategy to compute centralities more efficiently at larger scales while maintaining a relatively high degree of rank performance. Use non-sampled computation for publication-quality results.
    """)
    return


@app.cell
def _(cn, nodes_gdf_2):
    _distances = [10000]
    cn.centrality_simplest(distances=_distances, sample=True)
    nodes_gdf_3 = cn.to_geopandas()
    sorted(c for c in nodes_gdf_3.columns if c not in nodes_gdf_2.columns)
    return (nodes_gdf_3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For shorter distances where the number of reachable nodes is small, full computation is used. For larger distances, the function applies sampling to achieve the target accuracy efficiently.
    """)
    return


@app.cell
def _(nodes_gdf_3, plt):
    _fig, _ax = plt.subplots(1, 1, figsize=(8, 6), dpi=150)
    nodes_gdf_3.plot(
        column="cc_betweenness_10000_ang",
        cmap="magma",
        legend=True,
        legend_kwds={"label": "Angular betweenness, 10000 m", "shrink": 0.6},
        ax=_ax,
    )
    _ax.set_title("Angular betweenness, 10000 m (sampled)")
    _ax.set_xlim(438500, 438500 + 3500)
    _ax.set_ylim(4472500, 4472500 + 3500)
    _fig.tight_layout()
    _ax.axis(False)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tolerance for near-simplest paths
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `tolerance` parameter allows betweenness to count paths that are within a given percentage of the simplest (lowest angular cost) path. For angular centrality, a larger tolerance (e.g. `tolerance=20.0`) can be appropriate since angular costs are more ambiguous than metric distances. This produces more distributed betweenness distributions by accounting for near-optimal route choices.
    """)
    return


@app.cell
def _(cn, nodes_gdf_3):
    _distances = [5000]
    cn.centrality_simplest(distances=_distances, sample=True, tolerance=20.0)
    nodes_gdf_4 = cn.to_geopandas()
    sorted(c for c in nodes_gdf_4.columns if c not in nodes_gdf_3.columns)
    return (nodes_gdf_4,)


@app.cell
def _(nodes_gdf_4, plt):
    _fig, _ax = plt.subplots(1, 1, figsize=(8, 6), dpi=150)
    nodes_gdf_4.plot(
        column="cc_betweenness_5000_ang",
        cmap="magma",
        legend=True,
        legend_kwds={"label": "Angular betweenness, 5000 m", "shrink": 0.6},
        ax=_ax,
    )
    _ax.set_title("Angular betweenness, 5000 m (tolerance 20%)")
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

    This notebook demonstrated how to calculate angular (simplest-path) centralities from a `geopandas` `GeoDataFrame` with the `CityNetwork` class. Angular analysis weights paths by cumulative turning angle rather than distance, capturing route directness and identifying streets that form part of straighter, more navigable corridors. It also showed how to use the `sample` parameter for efficient computation at larger distance thresholds, and the `tolerance` parameter for near-simplest path betweenness.

    **Next steps:**

    - To calculate metric (shortest-path) centralities instead, see [Metric Centrality](https://cityseer.benchmarkurbanism.com/examples/centrality/gpd-metric-centrality).
    - To calculate centralities directly from OpenStreetMap data, see [OSM Centrality](https://cityseer.benchmarkurbanism.com/examples/centrality/osm-centrality).
    - To compute accessibility or mixed-use metrics over the same network, see the [Accessibility](https://cityseer.benchmarkurbanism.com/examples/accessibility) recipes.
    """)
    return


if __name__ == "__main__":
    app.run()
