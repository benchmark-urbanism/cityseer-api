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
    # Statistics from GeoPandas Data

    Calculate building statistics from a `geopandas` `GeoDataFrame`.

    The bundled datasets and their source attributions are documented on the [datasets page](https://cityseer.benchmarkurbanism.com/examples/datasets).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This recipe uses the high-level [`CityNetwork`](https://cityseer.benchmarkurbanism.com/api/network) class for the statistics computation. Because this example decomposes the network first, which is a lower-level graph operation without a `CityNetwork` equivalent, the graph is prepared with the `tools` modules and then bridged into the high-level API with `CityNetwork.from_nx`.
    """)
    return


@app.cell
def _():
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from cityseer.network import CityNetwork
    from cityseer.tools import graphs, io

    return CityNetwork, gpd, graphs, io, np, plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To start, follow the same approach as shown in the network examples to create the network. The full bundled network is clipped to a 2km study area around the city centre, buffered by the maximum analysis distance so that nodes near the study edge keep their full catchments. Since decomposition multiplies the number of network nodes, this example works with a district rather than the whole city; the statistics are computed per node from local catchments, so the results within the study area are unaffected.

    The graph is decomposed with [`nx_decompose`](https://cityseer.benchmarkurbanism.com/tools/graphs#nx-decompose) so that statistics are sampled at a higher spatial resolution, then handed to `CityNetwork.from_nx`; the `boundary` argument marks the nodes inside the study area as `live`.
    """)
    return


@app.cell
def _(CityNetwork, gpd, graphs, io):
    streets_gpd = gpd.read_file("../../data/madrid_streets/street_network.gpkg")
    streets_gpd = streets_gpd.explode(ignore_index=True)
    # the source data contains a small number of exact duplicate geometries
    streets_gpd = streets_gpd.drop_duplicates(subset="geometry")
    # 2km study area around the city centre, with the network buffered a further 200m
    # (the maximum analysis distance) to prevent edge rolloff
    centre = gpd.GeoSeries.from_xy([440300], [4474300], crs=streets_gpd.crs)
    study_poly = centre.buffer(2000).iloc[0]
    buffered_poly = centre.buffer(2200).iloc[0]
    streets_clip = streets_gpd[streets_gpd.intersects(buffered_poly)]
    G = io.nx_from_generic_geopandas(streets_clip)
    G = graphs.nx_decompose(G, 50)
    cn = CityNetwork.from_nx(G, boundary=study_poly)
    return buffered_poly, cn


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Read the buildings dataset with GeoPandas, for example from a GeoPackage or Shapefile. The buildings are clipped to the buffered study area, since only buildings within reach of the clipped network can contribute to the statistics.
    """)
    return


@app.cell
def _(buffered_poly, gpd):
    bldgs_gpd = gpd.read_file("../../data/madrid_buildings/madrid_bldgs.gpkg")
    bldgs_gpd = bldgs_gpd[bldgs_gpd.intersects(buffered_poly)]
    bldgs_gpd.head()
    return (bldgs_gpd,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Use the [`compute_stats`](https://cityseer.benchmarkurbanism.com/api/network#compute-stats) method to compute statistics for numeric columns in the `GeoDataFrame`, specified with the `stats_column_labels` argument. The statistics are aggregated over the network using network distances. The `measures` argument selects which statistics to compute; without it, the full set of `count`, `sum`, `min`, `max`, `mean`, `median`, `mad` (median absolute deviation) and `var` is generated.
    """)
    return


@app.cell
def _(bldgs_gpd, cn):
    distances = [100, 200]
    cn.compute_stats(
        bldgs_gpd,
        stats_column_labels=["area", "perimeter", "compactness", "orientation", "shape_index"],
        distances=distances,
        measures=["mean", "median", "mad", "var"],
    )
    nodes_gdf_1 = cn.to_geopandas()
    return (nodes_gdf_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This generates a `cc_{column}_{measure}_{distance}` column per selected measure for each of the input distance thresholds. By default the aggregations are unweighted; pass a `decay_fn` expression such as `"exp(-4 * p)"` to weight the contribution of each point by its distance from the point of measure.
    """)
    return


@app.cell
def _(nodes_gdf_1):
    list(nodes_gdf_1.columns)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The result in columns can be explored with conventional Python ecosystem tools such as `seaborn` and `matplotlib`.
    """)
    return


@app.cell
def _(nodes_gdf_1, plt, sns):
    _fig, _ax = plt.subplots(figsize=(9, 5), dpi=150)
    sns.histplot(
        data=nodes_gdf_1,
        x="cc_orientation_mean_200",
        color="#3465a4",
        alpha=0.35,
        edgecolor="#23425f",
        linewidth=0.6,
        label="Mean",
        element="step",
        ax=_ax,
    )
    sns.histplot(
        data=nodes_gdf_1,
        x="cc_orientation_median_200",
        color="#f57900",
        alpha=0.35,
        edgecolor="#a55000",
        linewidth=0.6,
        label="Median",
        element="step",
        ax=_ax,
    )
    _ax.set_xlabel("Orientation (degrees)")
    _ax.set_ylabel("Count")
    _ax.set_title("Orientation Distribution: Mean vs Median (200m network distance)")
    _ax.legend(frameon=False, ncol=2)
    _ax.set_xlim(0, 45)
    sns.despine()
    plt.tight_layout()
    _fig
    return


@app.cell
def _(nodes_gdf_1, np, plt, sns):
    _fig, _ax = plt.subplots(figsize=(9, 5), dpi=150)
    sns.histplot(
        data=nodes_gdf_1,
        x="cc_orientation_mad_200",
        color="#3465a4",
        alpha=0.35,
        edgecolor="#23425f",
        linewidth=0.6,
        label="Median Absolute Deviation",
        element="step",
        ax=_ax,
    )
    nodes_gdf_1["std"] = np.sqrt(nodes_gdf_1["cc_orientation_var_200"])
    sns.histplot(
        data=nodes_gdf_1,
        x="std",
        color="#f57900",
        alpha=0.35,
        edgecolor="#a55000",
        linewidth=0.6,
        label="Standard Deviation",
        element="step",
        ax=_ax,
    )
    _ax.set_xlabel("Orientation (degrees)")
    _ax.set_ylabel("Count")
    _ax.set_title("MAD vs. Standard Deviation (200m network distance)")
    _ax.legend(frameon=False, ncol=2)
    _ax.set_xlim(0, 20)
    sns.despine()
    plt.tight_layout()
    _fig
    return


@app.cell
def _(bldgs_gpd, nodes_gdf_1, plt):
    g = nodes_gdf_1[nodes_gdf_1.live].copy()
    g["_r"] = g["cc_orientation_median_200"].rank(pct=True)
    g = g.sort_values("_r")
    _fig, _ax = plt.subplots(1, 1, figsize=(7, 7), dpi=150)
    g.plot(ax=_ax, color=plt.get_cmap("OrRd")(g["_r"]), linewidth=0.15 + 2.25 * g["_r"])
    # buildings as light-grey context
    bldgs_gpd.plot(color="#cccccc", edgecolor="#bbbbbb", linewidth=0.1, ax=_ax)
    _ax.set_title("Median building orientation, 200 m", loc="left")
    _ax.set_xlim(438500, 438500 + 3500)
    _ax.set_ylim(4472500, 4472500 + 3500)
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
    This notebook computed network-distance statistics for building morphology attributes (area, perimeter, compactness, orientation, and shape index) from a GeoDataFrame using `CityNetwork.compute_stats`, with the decomposed graph bridged into the high-level API via `from_nx`. The method produces summary statistics such as mean, median, variance, and MAD, selectable via `measures`, enabling fine-grained spatial analysis of building characteristics across a district of the bundled street network.

    **Next steps:** To compute statistics from OSM data, see [OSM Statistics](https://cityseer.benchmarkurbanism.com/examples/stats/osm-stats).
    """)
    return


if __name__ == "__main__":
    app.run()
