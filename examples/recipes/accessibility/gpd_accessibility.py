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
    # Landuse Accessibility from GeoPandas Data

    Calculate landuse accessibilities from a `geopandas` `GeoDataFrame`.

    This recipe uses the high-level [`CityNetwork`](https://cityseer.benchmarkurbanism.com/api/network) class, which wraps network preparation and the accessibility computation behind a lean interface.

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
    To start, build the network with the [`from_geopandas`](https://cityseer.benchmarkurbanism.com/api/network#from-geopandas) constructor. The full bundled network is clipped to a 2km study area around the city centre, buffered by the maximum analysis distance so that nodes near the study edge keep their full catchments; accessibility is computed per node from local catchments, so a district gives the same results as the whole city at a fraction of the cost. The `boundary` argument marks the nodes inside the study area as `live`.
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
    Identify or prepare any columns and land uses of interest, for which you want to compute accessibilities.
    """)
    return


@app.cell
def _(prems_gpd):
    prems_gpd.division_desc.unique()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Once the landuse and network data has been prepared, use the [`compute_accessibilities`](https://cityseer.benchmarkurbanism.com/api/network#compute-accessibilities) method to compute accessibilities to landuses. The `landuse_column_label` and the target accessibility keys should correspond to the data in the input GeoDataFrame. `to_geopandas` returns the results joined to the original street LineString geometries.
    """)
    return


@app.cell
def _(cn, prems_gpd):
    # compute accessibility
    distances = [100, 200, 400, 800]
    cn.compute_accessibilities(
        prems_gpd,
        landuse_column_label="division_desc",
        accessibility_keys=["food_bev", "creat_entert", "retail"],
        distances=distances,
    )
    nodes_gdf_1 = cn.to_geopandas()
    return (nodes_gdf_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The output columns are named `cc_{key}_{distance}`, where the keys correspond to the input accessibility keys and the distances to the input distances. By default the counts are unweighted; pass a `decay_fn` expression such as `"exp(-4 * p)"` for distance-weighted counts. A further `cc_{key}_nearest_max_{distance}` column, written at the largest distance threshold only, gives the distance to the nearest instance of each landuse.

    Standard GeoPandas functionality can be used to explore, visualise, or save the results.
    """)
    return


@app.cell
def _(nodes_gdf_1):
    nodes_gdf_1.columns
    return


@app.cell
def _(nodes_gdf_1, plt, prems_gpd):
    _g = nodes_gdf_1[nodes_gdf_1.live].copy()
    _g["_r"] = _g["cc_retail_400"].rank(pct=True)
    _g = _g.sort_values("_r")
    _fig, _ax = plt.subplots(1, 1, figsize=(7, 7), dpi=150)
    _g.plot(ax=_ax, color=plt.get_cmap("OrRd")(_g["_r"]), linewidth=0.15 + 2.25 * _g["_r"])
    prems_gpd[prems_gpd["division_desc"] == "retail"].plot(
        markersize=1, edgecolor=None, color="#333333", legend=False, ax=_ax
    )
    _ax.set_title("Retail accessibility, 400 m", loc="left")
    _ax.set_xlim(439000, 439000 + 2500)
    _ax.set_ylim(4473000, 4473000 + 2500)
    _ax.set_axis_off()
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(nodes_gdf_1, plt, prems_gpd):
    _g = nodes_gdf_1[nodes_gdf_1.live].copy()
    _g["_r"] = _g["cc_food_bev_200"].rank(pct=True)
    _g = _g.sort_values("_r")
    _fig, _ax = plt.subplots(1, 1, figsize=(7, 7), dpi=150)
    _g.plot(ax=_ax, color=plt.get_cmap("OrRd")(_g["_r"]), linewidth=0.15 + 2.25 * _g["_r"])
    prems_gpd[prems_gpd["division_desc"] == "food_bev"].plot(
        markersize=1, edgecolor=None, color="#333333", legend=False, ax=_ax
    )
    _ax.set_title("Food and beverage accessibility, 200 m", loc="left")
    _ax.set_xlim(439000, 439000 + 2500)
    _ax.set_ylim(4473000, 4473000 + 2500)
    _ax.set_axis_off()
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(nodes_gdf_1, plt, prems_gpd):
    nodes_gdf_1["cc_creat_entert_nearest_max_800"] = nodes_gdf_1["cc_creat_entert_nearest_max_800"].fillna(800)
    _g = nodes_gdf_1[nodes_gdf_1.live].copy()
    # proximity map: invert the rank so the nearest venues (smallest distance) read boldest
    _g["_r"] = 1 - _g["cc_creat_entert_nearest_max_800"].rank(pct=True)
    _g = _g.sort_values("_r")
    _fig, _ax = plt.subplots(1, 1, figsize=(7, 7), dpi=150)
    _g.plot(ax=_ax, color=plt.get_cmap("OrRd")(_g["_r"]), linewidth=0.15 + 2.25 * _g["_r"])
    prems_gpd[prems_gpd["division_desc"] == "creat_entert"].plot(
        markersize=2, edgecolor=None, color="#333333", legend=False, ax=_ax
    )
    _ax.set_title("Distance to nearest creative or entertainment venue, 800 m max", loc="left")
    _ax.set_xlim(439000, 439000 + 2500)
    _ax.set_ylim(4473000, 4473000 + 2500)
    _ax.set_axis_off()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Brighter areas indicate higher accessibility: more landuses of that type are reachable within the specified network distance.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusion

    This notebook computed landuse accessibility metrics from a GeoDataFrame of premises data through the `CityNetwork` class: reachable counts and nearest-distance measures for food and beverage, creative entertainment, and retail categories at multiple distance thresholds. The results map the spatial distribution and intensity of different landuses across a central study area.

    **Next steps:** To compute accessibility from OSM data instead, see [OSM Accessibility](https://cityseer.benchmarkurbanism.com/examples/accessibility/osm-accessibility).
    """)
    return


if __name__ == "__main__":
    app.run()
