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
    # Calculate Mixed Land-Uses

    Calculate mixed-use diversity measures from a `geopandas` `GeoDataFrame`.

    This recipe uses the high-level [`CityNetwork`](https://cityseer.benchmarkurbanism.com/api/network) class, which wraps network preparation and the mixed-use computation behind a lean interface.

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
    To start, build the network with the [`from_geopandas`](https://cityseer.benchmarkurbanism.com/api/network#from_geopandas) constructor. The full bundled network is clipped to a 2km study area around the city centre, buffered by the maximum analysis distance so that nodes near the study edge keep their full catchments; mixed uses are computed per node from local catchments, so a district gives the same results as the whole city at a fraction of the cost. The `boundary` argument marks the nodes inside the study area as `live`.
    """)
    return


@app.cell
def _(CityNetwork, gpd):
    streets_gpd = gpd.read_file("../../data/madrid_streets/street_network.gpkg")
    streets_gpd = streets_gpd.explode(ignore_index=True)
    # the source data contains a small number of exact duplicate geometries
    streets_gpd = streets_gpd.drop_duplicates(subset="geometry")
    # 2km study area around the city centre, with the network buffered a further 800m
    # (the maximum analysis distance) to prevent edge rolloff
    centre = gpd.GeoSeries.from_xy([440300], [4474300], crs=streets_gpd.crs)
    study_poly = centre.buffer(2000).iloc[0]
    buffered_poly = centre.buffer(2800).iloc[0]
    streets_clip = streets_gpd[streets_gpd.intersects(buffered_poly)]
    cn = CityNetwork.from_geopandas(streets_clip, boundary=study_poly)
    return buffered_poly, cn


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Read the landuse dataset with GeoPandas, for example from a GeoPackage or Shapefile. The premises are clipped to the buffered study area, since only landuses within reach of the clipped network can contribute.
    """)
    return


@app.cell
def _(buffered_poly, gpd):
    prems_gpd = gpd.read_file("../../data/madrid_premises/madrid_premises.gpkg")
    prems_gpd = prems_gpd[prems_gpd.within(buffered_poly)]
    prems_gpd.head()
    return (prems_gpd,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Identify or prepare the column containing the landuse categories over which to compute mixed uses.
    """)
    return


@app.cell
def _(prems_gpd):
    prems_gpd.division_desc.unique()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Once the landuse and network data has been prepared, use the [`compute_mixed_uses`](https://cityseer.benchmarkurbanism.com/api/network#compute_mixed_uses) method to compute mixed-use diversity. The `landuse_column_label` should correspond to the data in the input GeoDataFrame. `to_geopandas` returns the results joined to the original street LineString geometries.
    """)
    return


@app.cell
def _(cn, prems_gpd):
    # compute mixed uses
    distances = [100, 200, 400, 800]
    _cn, prems_gpd_1 = cn.compute_mixed_uses(
        prems_gpd,
        landuse_column_label="division_desc",
        distances=distances,
    )
    nodes_gdf_1 = cn.to_geopandas()
    return nodes_gdf_1, prems_gpd_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The output columns are named `cc_{key}_{distance}`, where the keys correspond to the [form of mixed-use measure](https://cityseer.benchmarkurbanism.com/metrics/layers#compute_mixed_uses) and the distances to the input distances. By default the measures are unweighted; pass a `decay_fn` expression such as `"exp(-4 * p)"` for distance-weighted variants.

    Standard GeoPandas functionality can be used to explore, visualise, or save the results.
    """)
    return


@app.cell
def _(nodes_gdf_1):
    nodes_gdf_1.columns
    return


@app.cell
def _(nodes_gdf_1, plt, prems_gpd_1):
    g = nodes_gdf_1[nodes_gdf_1.live].copy()
    g["_r"] = g["cc_hill_q0_400"].rank(pct=True)
    g = g.sort_values("_r")
    fig, ax = plt.subplots(1, 1, figsize=(7, 7), dpi=150)
    g.plot(ax=ax, color=plt.get_cmap("OrRd")(g["_r"]), linewidth=0.15 + 2.25 * g["_r"])
    # categorical land-use overlay kept as-is (discrete classes)
    prems_gpd_1.plot(column="division_desc", cmap="tab20", markersize=1, edgecolor=None, legend=False, ax=ax)
    ax.set_title("Hill diversity q=0, 400 m", loc="left")
    ax.set_xlim(439000, 439000 + 2500)
    ax.set_ylim(4473000, 4473000 + 2500)
    ax.set_axis_off()
    fig.tight_layout()
    fig
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
    This notebook computed mixed land-use diversity measures from a GeoDataFrame of the bundled premises data using the `CityNetwork.compute_mixed_uses` method. The Hill diversity indices (q0, q1, q2) quantify the richness and evenness of landuse categories at multiple distance thresholds, providing insight into the functional diversity of different neighbourhoods.

    **Next steps:** For statistical aggregation of numeric data, see [Statistics](https://cityseer.benchmarkurbanism.com/examples/stats/gpd-stats).
    """)
    return


if __name__ == "__main__":
    app.run()
