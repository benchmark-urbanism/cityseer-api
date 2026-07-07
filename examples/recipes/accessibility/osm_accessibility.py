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
    # Landuse Accessibility from OSM Data

    Calculate landuse accessibility to pubs and restaurants for London using OpenStreetMap data.

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
    Prepare the amenities GeoDataFrame by downloading the data from OpenStreetMap. The `osmnx` [`features_from_polygon`](https://osmnx.readthedocs.io/en/stable/user-reference.html#osmnx.features.features_from_polygon) works well for this purpose. In this instance, we are specifically targeting features that are labelled as an `amenity` type of either `pub` or `restaurant`. You can use the same idea to extract other features or land use types.

    It is important to convert the derivative GeoDataFrame to the same CRS as the network.
    """)
    return


@app.cell
def _(cn, features, poly_wgs):
    data_gdf = features.features_from_polygon(poly_wgs, tags={"amenity": ["pub", "restaurant"]})
    data_gdf = data_gdf.to_crs(cn.crs)
    data_gdf
    return (data_gdf,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Some preparatory data cleaning is typically necessary. This example extracts the particular rows and columns of interest for the subsequent steps of analysis.
    """)
    return


@app.cell
def _(data_gdf):
    # extract nodes
    data_gdf_1 = data_gdf.loc["node"]
    # reset index
    data_gdf_1 = data_gdf_1.reset_index(level=0, drop=True)
    # extract relevant columns
    data_gdf_1 = data_gdf_1[["amenity", "geometry"]]
    data_gdf_1.head()
    return (data_gdf_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Once the landuse and network data has been prepared, use the [`compute_accessibilities`](https://cityseer.benchmarkurbanism.com/api/network#compute_accessibilities) method to compute accessibilities to landuses. The `landuse_column_label` and the target accessibility keys should correspond to the data in the input GeoDataFrame. `to_geopandas` returns the results joined to the original street LineString geometries.
    """)
    return


@app.cell
def _(cn, data_gdf_1):
    # compute pub accessibility
    distances = [100, 200, 400, 800]
    _cn, data_gdf_2 = cn.compute_accessibilities(
        data_gdf_1,
        landuse_column_label="amenity",
        accessibility_keys=["pub", "restaurant"],
        distances=distances,
    )
    nodes_gdf_1 = cn.to_geopandas()
    return data_gdf_2, nodes_gdf_1


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
def _(data_gdf_2, nodes_gdf_1, plt):
    _fig, _ax = plt.subplots(1, 1, figsize=(8, 8), dpi=150)
    nodes_gdf_1.plot(
        column="cc_restaurant_400",
        cmap="magma",
        legend=True,
        legend_kwds={"label": "Restaurant accessibility, 400 m", "shrink": 0.6},
        ax=_ax,
    )
    data_gdf_2[data_gdf_2["amenity"] == "restaurant"].plot(
        markersize=2, edgecolor=None, color="white", legend=False, ax=_ax
    )
    _ax.set_title("Restaurant accessibility, 400 m")
    _ax.set_axis_off()
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(data_gdf_2, nodes_gdf_1, plt):
    _fig, _ax = plt.subplots(1, 1, figsize=(8, 8), dpi=150)
    nodes_gdf_1.plot(
        column="cc_pub_nearest_max_800",
        cmap="viridis_r",
        legend=True,
        legend_kwds={"label": "Distance to nearest pub (m)", "shrink": 0.6},
        ax=_ax,
    )
    data_gdf_2[data_gdf_2["amenity"] == "pub"].plot(markersize=2, edgecolor=None, color="white", legend=False, ax=_ax)
    _ax.set_title("Distance to nearest pub, 800 m max")
    _ax.set_axis_off()
    _fig.tight_layout()
    _fig
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
    This notebook computed landuse accessibility to pubs and restaurants in London using point features downloaded from OpenStreetMap via `osmnx` and the `CityNetwork` class. The workflow fetches amenity data, assigns it to the network, and produces reachable counts and nearest-distance columns that reveal the spatial concentration of hospitality venues.

    **Next steps:** For polygon-based accessibility, such as parks, see [Park Accessibility](https://cityseer.benchmarkurbanism.com/examples/accessibility/osm-accessibility-polys).
    """)
    return


if __name__ == "__main__":
    app.run()
