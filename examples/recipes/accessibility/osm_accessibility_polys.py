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
    # Park Accessibility from OSM Data

    Calculate park accessibility for London using OpenStreetMap polygon data.

    This recipe uses the high-level [`CityNetwork`](https://cityseer.benchmarkurbanism.com/api/network) class, whose [`from_osm`](https://cityseer.benchmarkurbanism.com/api/network#from_osm) constructor wraps the OSM download, cleaning, and network preparation behind a single call.

    Network data © [OpenStreetMap](https://www.openstreetmap.org/copyright) contributors, available under the Open Database Licence.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    from cityseer.network import CityNetwork
    from cityseer.tools import io
    from osmnx import features

    return CityNetwork, features, io, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To start, create the network from a buffered point polygon using the `from_osm` constructor.
    """)
    return


@app.cell
def _(CityNetwork, io):
    lng, lat = -0.13396079424572427, 51.51371088849723
    buffer = 1500
    poly_wgs, _epsg_code = io.buffered_point_poly(lng, lat, buffer)
    cn = CityNetwork.from_osm(poly_wgs)
    return cn, poly_wgs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Prepare the parks GeoDataFrame by downloading the data from OpenStreetMap. The `osmnx` [`features_from_polygon`](https://osmnx.readthedocs.io/en/stable/user-reference.html#osmnx.features.features_from_polygon) works well for this purpose. In this instance, we are specifically targeting features that are labelled as a `leisure` type of `park`. You can use the same idea to extract other features or land use types.

    It is important to convert the derivative GeoDataFrame to the same CRS as the network.
    """)
    return


@app.cell
def _(cn, features, poly_wgs):
    data_gdf = features.features_from_polygon(poly_wgs, tags={"leisure": ["park"]})
    data_gdf = data_gdf.to_crs(cn.crs)
    # reset index
    data_gdf = data_gdf.reset_index(level=0, drop=True)

    data_gdf
    return (data_gdf,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Once the data has been prepared, use the [`compute_accessibilities`](https://cityseer.benchmarkurbanism.com/api/network#compute_accessibilities) method to compute accessibilities. The method accepts polygon geometries directly, assigning each polygon to the nearby network nodes. The `landuse_column_label` and the target accessibility keys should correspond to the data in the input GeoDataFrame. Use the `max_netw_assign_dist` parameter to configure the distance for network assignment.
    """)
    return


@app.cell
def _(cn, data_gdf):
    # compute park accessibility
    distances = [100, 200, 400, 800]
    _cn, data_gdf_1 = cn.compute_accessibilities(
        data_gdf,
        landuse_column_label="leisure",
        accessibility_keys=["park"],
        distances=distances,
    )
    nodes_gdf_1 = cn.to_geopandas()
    return data_gdf_1, nodes_gdf_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The output columns are named `cc_{key}_{distance}`, where the keys correspond to the input accessibility keys and the distances to the input distances. By default the counts are unweighted; pass a `decay_fn` expression such as `"exp(-4 * p)"` for distance-weighted counts. A further `cc_{key}_nearest_max_{distance}` column gives the distance to the nearest instance of each landuse.

    Standard GeoPandas functionality can be used to explore, visualise, or save the results.
    """)
    return


@app.cell
def _(nodes_gdf_1):
    nodes_gdf_1.columns
    return


@app.cell
def _(data_gdf_1, nodes_gdf_1, plt):
    fig, ax = plt.subplots(1, 1, figsize=(7, 7), dpi=150)
    _g = nodes_gdf_1.copy()
    # distance-to-nearest: lower is better, so invert the rank to draw the closest streets boldest
    _g["_r"] = 1 - _g["cc_park_nearest_max_800"].rank(pct=True)
    _g = _g.sort_values("_r")
    _g.plot(ax=ax, color=plt.get_cmap("OrRd")(_g["_r"]), linewidth=0.15 + 2.25 * _g["_r"])
    data_gdf_1.plot(color="#cccccc", edgecolor="#bbbbbb", ax=ax)
    ax.set_title("Distance to nearest park, 800 m max", loc="left")
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
    This notebook computed park accessibility from OSM polygon features in London, showing that `CityNetwork.compute_accessibilities` handles polygon geometries as well as points by assigning them to multiple nearby network nodes. The nearest-distance and count outputs reveal how park proximity varies across the street network.

    **Next steps:** To compute mixed land-use diversity, see [Mixed Uses](https://cityseer.benchmarkurbanism.com/examples/accessibility/gpd-mixed-uses).
    """)
    return


if __name__ == "__main__":
    app.run()
