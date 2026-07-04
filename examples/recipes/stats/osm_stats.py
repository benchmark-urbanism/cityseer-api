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
    # Statistics from OSM Data

    Calculate building statistics from OpenStreetMap data fetched with `osmnx`.

    This recipe uses the high-level [`CityNetwork`](https://cityseer.benchmarkurbanism.com/api/network) class for the statistics computation. Because this example decomposes the network first, which is a lower-level graph operation without a `CityNetwork` equivalent, the graph is prepared with the `tools` modules and then bridged into the high-level API with `CityNetwork.from_nx`.

    Network data © [OpenStreetMap](https://www.openstreetmap.org/copyright) contributors, available under the Open Database Licence.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import momepy
    import pandas as pd
    import seaborn as sns
    from cityseer.network import CityNetwork
    from cityseer.tools import graphs, io
    from matplotlib import colors
    from osmnx import features

    return CityNetwork, colors, features, graphs, io, momepy, pd, plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To start, follow the same approach as shown in the network examples to create the network. The graph is decomposed with [`nx_decompose`](https://cityseer.benchmarkurbanism.com/tools/graphs#nx_decompose) so that statistics are sampled at a higher spatial resolution, then handed to `CityNetwork.from_nx`.
    """)
    return


@app.cell
def _(CityNetwork, graphs, io):
    lng, lat = -0.13396079424572427, 51.51371088849723
    buffer = 1500
    poly_wgs, _epsg_code = io.buffered_point_poly(lng, lat, buffer)
    G = io.osm_graph_from_poly(poly_wgs)
    G = graphs.nx_decompose(G, 50)
    cn = CityNetwork.from_nx(G)
    return G, cn, poly_wgs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Prepare the buildings GeoDataFrame by downloading the data from OpenStreetMap. The `osmnx` [`features_from_polygon`](https://osmnx.readthedocs.io/en/stable/user-reference.html#osmnx.features.features_from_polygon) function works well for this purpose. In this instance, we target features tagged as `building`.

    It is important to convert the derivative GeoDataFrame to the same CRS as the network.
    """)
    return


@app.cell
def _(G, features, poly_wgs):
    data_gdf = features.features_from_polygon(poly_wgs, tags={"building": True})
    data_gdf = data_gdf.to_crs(G.graph["crs"])
    data_gdf.tail()
    return (data_gdf,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Some preparatory data cleaning is typically necessary. This example extracts the particular rows and columns of interest for the subsequent steps of analysis.
    """)
    return


@app.cell
def _(data_gdf):
    bldg_way = data_gdf.loc["way"]
    bldg_way = bldg_way.reset_index(level=0, drop=True)
    bldg_way = bldg_way[["geometry"]]
    bldg_way.head()
    return (bldg_way,)


@app.cell
def _(data_gdf):
    bldg_rel = data_gdf.loc["relation"]
    bldg_rel = bldg_rel.reset_index(level=0, drop=True)
    bldg_rel = bldg_rel[["geometry"]]
    bldg_rel.head()
    return (bldg_rel,)


@app.cell
def _(bldg_rel, bldg_way, pd):
    bldgs_gpd = pd.concat([bldg_way, bldg_rel], axis=0)
    bldgs_gpd = bldgs_gpd.explode(ignore_index=True)
    bldgs_gpd = bldgs_gpd.reset_index(drop=True)
    bldgs_gpd.head()
    return (bldgs_gpd,)


@app.cell
def _(bldgs_gpd, momepy):
    bldgs_gpd["area"] = bldgs_gpd.geometry.area
    bldgs_gpd["perimeter"] = bldgs_gpd.geometry.length
    bldgs_gpd["compactness"] = momepy.circular_compactness(bldgs_gpd.geometry)
    bldgs_gpd["orientation"] = momepy.orientation(bldgs_gpd.geometry)
    bldgs_gpd["shape_index"] = momepy.shape_index(bldgs_gpd.geometry)
    return


@app.cell
def _(bldgs_gpd):
    bldgs_gpd.geometry.type.unique()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Use the [`compute_stats`](https://cityseer.benchmarkurbanism.com/api/network#compute_stats) method to compute statistics for numeric columns in the `GeoDataFrame`, specified with the `stats_column_labels` argument. The statistics are aggregated over the network using network distances. The `measures` argument selects which statistics to compute; without it, the full set of `count`, `sum`, `min`, `max`, `mean`, `median`, `mad` (median absolute deviation) and `var` is generated.
    """)
    return


@app.cell
def _(bldgs_gpd, cn):
    distances = [100, 200]
    _cn, bldgs_gpd_1 = cn.compute_stats(
        bldgs_gpd,
        stats_column_labels=["area", "perimeter", "compactness", "orientation", "shape_index"],
        distances=distances,
        measures=["mean", "count"],
    )
    nodes_gdf_1 = cn.to_geopandas()
    return bldgs_gpd_1, nodes_gdf_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This generates a `cc_{column}_{measure}_{distance}` column per selected measure for each of the input distance thresholds. By default the aggregations are unweighted; pass a `decay_fn` expression such as `"exp(-4 * p)"` to weight the contribution of each point by its distance from the point of measure.
    """)
    return


@app.cell
def _(nodes_gdf_1):
    nodes_gdf_1.columns
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The result in columns can be explored with conventional Python ecosystem tools such as `seaborn` and `matplotlib`.
    """)
    return


@app.cell
def _(nodes_gdf_1, sns):
    sns.histplot(data=nodes_gdf_1, x="cc_area_mean_100", bins=50)
    return


@app.cell
def _(bldgs_gpd_1, colors, nodes_gdf_1, plt):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=150)
    # log normalisation because building areas span orders of magnitude
    nodes_gdf_1.plot(
        column="cc_area_mean_100",
        cmap="magma",
        legend=True,
        legend_kwds={"label": "Mean building area (m²), 100 m", "shrink": 0.6},
        norm=colors.LogNorm(vmin=1, vmax=1000),
        ax=ax,
    )
    bldgs_gpd_1.plot(column="area", cmap="magma", legend=False, alpha=0.5, norm=colors.LogNorm(vmin=1, vmax=1000), ax=ax)
    ax.set_title("Mean building area, 100 m")
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
    This notebook computed network-distance building statistics from OpenStreetMap data for London, using `osmnx` to fetch building footprints and `momepy` to derive morphological attributes (area, perimeter, compactness, orientation, shape index). The decomposed graph was bridged into the high-level API with `CityNetwork.from_nx`, and the `compute_stats` method aggregates the numeric attributes over the network at specified distance thresholds.

    **Next steps:** For visual enclosure metrics, see [Visibility from OSM](https://cityseer.benchmarkurbanism.com/examples/visibility/vis-osm).
    """)
    return


if __name__ == "__main__":
    app.run()
