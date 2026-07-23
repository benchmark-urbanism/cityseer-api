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
    # Adding GTFS Transport Data to Accessibility

    This notebook demonstrates how GTFS (General Transit Feed Specification) public transport data can be integrated into the street network before computing landuse accessibility metrics. By adding metro stops and segment travel times, the accessibility calculations can account for the extended reach provided by public transport.

    Accessibility to retail premises is computed twice: once on the street network alone, and again after adding the metro network. The difference reveals how the metro network extends the catchment area for retail access.

    The analysis uses a 4km study area around the city centre rather than the whole city: this is large enough to span multiple metro lines and interchanges, and since accessibility is computed per node from local catchments, the district reproduces the city-wide values at far lower cost. The network is buffered by the maximum analysis distance so nodes near the study edge keep their full catchments.

    This recipe uses the high-level [`CityNetwork`](https://cityseer.benchmarkurbanism.com/api/network) class, integrating the transit data through its `add_gtfs` method. Two instances are built from the same clipped streets: one street-only baseline and one with the metro added.

    The bundled datasets and their source attributions are documented on the [datasets page](https://cityseer.benchmarkurbanism.com/examples/datasets).
    """)
    return


@app.cell
def _():
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import pandas as pd
    from cityseer.network import CityNetwork

    return CityNetwork, gpd, pd, plt


@app.cell
def _(CityNetwork, gpd):
    streets_gpd = gpd.read_file("../../data/madrid_streets/street_network.gpkg")
    streets_gpd = streets_gpd.explode(ignore_index=True)
    # the source data contains a small number of exact duplicate geometries
    streets_gpd = streets_gpd.drop_duplicates(subset="geometry")
    # 4km study area around the city centre, with the network buffered a further 1600m
    # (the maximum analysis distance) to prevent edge rolloff
    centre = gpd.GeoSeries.from_xy([440300], [4474300], crs=streets_gpd.crs)
    study_poly = centre.buffer(4000).iloc[0]
    buffered_poly = centre.buffer(5600).iloc[0]
    streets_clip = streets_gpd[streets_gpd.intersects(buffered_poly)]
    cn_base = CityNetwork.from_geopandas(streets_clip, boundary=study_poly)
    return buffered_poly, cn_base, streets_clip, study_poly


@app.cell
def _(buffered_poly, gpd):
    prems_gpd = gpd.read_file("../../data/madrid_premises/madrid_premises.gpkg")
    prems_gpd = prems_gpd[prems_gpd.within(buffered_poly)]
    prems_gpd.head()
    return (prems_gpd,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Compute baseline retail accessibility (without GTFS data) and snapshot the results with `to_geopandas` for later comparison.
    """)
    return


@app.cell
def _(cn_base, prems_gpd):
    # compute baseline accessibility
    distances = [800, 1600]
    cn_base.compute_accessibilities(
        prems_gpd,
        landuse_column_label="division_desc",
        accessibility_keys=["retail"],
        distances=distances,
    )
    base_gdf = cn_base.to_geopandas()
    return base_gdf, distances


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Build a second network from the same clipped streets and load the bundled metro GTFS data with [`add_gtfs`](https://cityseer.benchmarkurbanism.com/api/network#add_gtfs). This adds metro stops as nodes and segment travel times as edges.
    """)
    return


@app.cell
def _(CityNetwork, streets_clip, study_poly):
    cn_metro = CityNetwork.from_geopandas(streets_clip, boundary=study_poly)
    cn_metro.add_gtfs("../../data/madrid_gtfs/madrid_metro")
    return (cn_metro,)


@app.cell
def _(cn_metro, gpd, pd):
    # read the GTFS stop locations for plotting
    stops = pd.read_csv("../../data/madrid_gtfs/madrid_metro/stops.txt")
    stops_gdf = gpd.GeoDataFrame(
        stops,
        geometry=gpd.points_from_xy(stops["stop_lon"], stops["stop_lat"]),
        crs=4326,  # GTFS stop coordinates are WGS84
    )
    stops_gdf = stops_gdf.to_crs(cn_metro.crs)
    stops_gdf
    return (stops_gdf,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Compute retail accessibility on the metro-integrated network. The shortest-path algorithms can now route through metro segments, extending the effective catchment area.
    """)
    return


@app.cell
def _(cn_metro, distances, prems_gpd):
    cn_metro.compute_accessibilities(
        prems_gpd,
        landuse_column_label="division_desc",
        accessibility_keys=["retail"],
        distances=distances,
    )
    metro_gdf = cn_metro.to_geopandas()
    return (metro_gdf,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compare results
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Compute the difference between the metro-integrated and baseline retail accessibility. The three-panel plot below shows retail accessibility within 800m without the metro, with the metro, and the difference. Retail premises are shown in white and metro stops in red.
    """)
    return


@app.cell
def _(metro_gdf):
    metro_gdf.columns
    return


@app.cell
def _(base_gdf, metro_gdf):
    metro_gdf["cc_retail_800_diff"] = metro_gdf["cc_retail_800"] - base_gdf["cc_retail_800"]
    return


@app.cell
def _(base_gdf, metro_gdf, plt, prems_gpd, stops_gdf):
    import matplotlib.colors as mcolors

    fig, axes = plt.subplots(3, 1, figsize=(8, 20), dpi=150)
    # baseline and with-metro retail accessibility: OrRd rank style, width and colour by percentile
    for _ax, _gdf, _title in [
        (axes[0], base_gdf, "Retail accessibility without metro, 800 m"),
        (axes[1], metro_gdf, "Retail accessibility with metro, 800 m"),
    ]:
        _g = _gdf[_gdf.live].copy()
        _g["_r"] = _g["cc_retail_800"].rank(pct=True)
        _g = _g.sort_values("_r")
        _g.plot(ax=_ax, color=plt.get_cmap("OrRd")(_g["_r"]), linewidth=0.15 + 2.25 * _g["_r"])
        _ax.set_title(_title, loc="left")
    # difference map: signed values, so keep a diverging scheme centred on zero
    _gd = metro_gdf[metro_gdf.live].copy()
    try:
        _norm = mcolors.TwoSlopeNorm(vcenter=0)
        _gd.plot(
            column="cc_retail_800_diff",
            cmap="coolwarm",
            norm=_norm,
            legend=True,
            legend_kwds={"label": "Difference in retail accessibility, 800 m", "shrink": 0.6},
            ax=axes[2],
        )
    except ValueError:
        _gd.plot(
            column="cc_retail_800_diff",
            cmap="coolwarm",
            legend=True,
            legend_kwds={"label": "Difference in retail accessibility, 800 m", "shrink": 0.6},
            ax=axes[2],
        )
    axes[2].set_title("Difference due to metro, 800 m", loc="left")
    for ax in axes:
        # context overlays on top: retail premises and metro stops in neutral dark
        prems_gpd[prems_gpd["division_desc"] == "retail"].plot(
            markersize=1, edgecolor=None, color="#333333", legend=False, ax=ax
        )
        stops_gdf.plot(ax=ax, color="#333333", markersize=1)
        ax.set_xlim(438500, 438500 + 3500)
        ax.set_ylim(4472500, 4472500 + 3500)
        ax.set_axis_off()
    fig.tight_layout()
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusion

    This notebook integrated GTFS public transport data (the bundled metro feed) into the street network via `CityNetwork.add_gtfs` and compared retail accessibility with and without transit connectivity. The before-and-after comparison reveals how the metro extends the effective catchment area for retail access, particularly around metro stop locations.

    **Next steps:** For centrality with GTFS data, see [GTFS Centrality](https://cityseer.benchmarkurbanism.com/examples/centrality/centrality-metro).
    """)
    return


if __name__ == "__main__":
    app.run()
