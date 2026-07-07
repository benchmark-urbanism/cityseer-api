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
    # Adding GTFS Transport Data to Centrality

    This notebook demonstrates how to integrate GTFS (General Transit Feed Specification) public transport data into the street network before computing centrality metrics. By adding metro stops and segment travel times to the network, centrality calculations can account for the connectivity boost provided by public transport.

    The approach computes centralities twice: once without and once with the GTFS data, so that the effect of the metro network on centrality can be compared.

    The analysis uses a 4km study area around the city centre rather than the whole city: this is large enough to span multiple metro lines and interchanges, and since centralities are computed per node from local catchments, the district reproduces the city-wide values at far lower cost. The network is buffered by the largest analysis distance so nodes near the study edge keep their full catchments.

    This recipe uses the high-level [`CityNetwork`](https://cityseer.benchmarkurbanism.com/api/network) class, integrating the transit data through its `add_gtfs` method. Two instances are built from the same clipped streets: one street-only baseline and one with the metro added.

    > GTFS support is experimental: the GTFS integration is still in development and may change in future releases.

    The bundled datasets and their source attributions are documented on the [datasets page](https://cityseer.benchmarkurbanism.com/examples/datasets).
    """)
    return


@app.cell
def _():
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import pandas as pd
    from cityseer.network import CityNetwork

    streets_gpd = gpd.read_file("../../data/madrid_streets/street_network.gpkg")
    streets_gpd = streets_gpd.explode(ignore_index=True)
    # the source data contains a small number of exact duplicate geometries
    streets_gpd = streets_gpd.drop_duplicates(subset="geometry")
    # 4km study area around the city centre, with the network buffered a further 5000m
    # (the largest analysis distance) to prevent edge rolloff
    centre = gpd.GeoSeries.from_xy([440300], [4474300], crs=streets_gpd.crs)
    study_poly = centre.buffer(4000).iloc[0]
    buffered_poly = centre.buffer(9000).iloc[0]
    streets_clip = streets_gpd[streets_gpd.intersects(buffered_poly)]
    cn_base = CityNetwork.from_geopandas(streets_clip, boundary=study_poly)
    return CityNetwork, cn_base, gpd, pd, plt, streets_clip, study_poly


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Compute baseline centralities (without GTFS data) and snapshot the results with `to_geopandas` for later comparison.
    """)
    return


@app.cell
def _(cn_base):
    distances = [1000, 2000, 5000]
    cn_base.centrality_shortest(distances=distances)
    base_gdf = cn_base.to_geopandas()
    return base_gdf, distances


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Build a second network from the same clipped streets and use [`add_gtfs`](https://cityseer.benchmarkurbanism.com/api/network#add_gtfs) to load the bundled metro GTFS data. This adds metro stops as nodes and segment travel times as edges, enabling the shortest-path algorithms to route through the metro network. The method takes the GTFS directory path; the network CRS is used automatically. For direct access to the derived stop and headway tables, the lower-level [`io.add_transport_gtfs`](https://cityseer.benchmarkurbanism.com/tools/io#add_transport_gtfs) function returns them explicitly.
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
        crs=4326,  # GTFS coordinates are WGS84
    )
    stops_gdf = stops_gdf.to_crs(cn_metro.crs)
    stops_gdf
    return (stops_gdf,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Calculate centralities
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The centrality methods can now be computed on the metro-integrated network.
    """)
    return


@app.cell
def _(cn_metro, distances):
    cn_metro.centrality_shortest(distances=distances)
    metro_gdf = cn_metro.to_geopandas()
    return (metro_gdf,)


@app.cell
def _(metro_gdf):
    metro_gdf.columns
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compare results

    Compute the difference between the metro-integrated and baseline centralities. The three-panel plot below shows harmonic closeness centrality without the metro, with the metro, and the difference. Metro stop locations are overlaid in red.
    """)
    return


@app.cell
def _(base_gdf, metro_gdf):
    metro_gdf["cc_harmonic_1000_diff"] = metro_gdf["cc_harmonic_1000"] - base_gdf["cc_harmonic_1000"]
    return


@app.cell
def _(base_gdf, metro_gdf, plt, stops_gdf):
    import matplotlib.colors as mcolors

    fig, axes = plt.subplots(3, 1, figsize=(6.5, 19.5), dpi=150)
    # sequential closeness panels: OrRd, edge width by percentile rank, no colour bar
    for _ax, _gdf, _title in [
        (axes[0], base_gdf, "Harmonic closeness, 1000 m (without metro)"),
        (axes[1], metro_gdf, "Harmonic closeness, 1000 m (with metro)"),
    ]:
        g = _gdf.copy()
        if "live" in g.columns:
            g = g[g.live].copy()
        g["_r"] = g["cc_harmonic_1000"].rank(pct=True)
        g = g.sort_values("_r")
        g.plot(ax=_ax, color=plt.get_cmap("OrRd")(g["_r"]), linewidth=0.15 + 2.25 * g["_r"])
        _ax.set_title(_title, loc="left")
        # transit stops as neutral dark markers; red now signifies flow intensity
        stops_gdf.plot(ax=_ax, color="#333333", markersize=1)
    # difference panel: signed, diverging, colour bar kept
    try:
        norm = mcolors.TwoSlopeNorm(vcenter=0)
        metro_gdf.plot(
            ax=axes[2],
            column="cc_harmonic_1000_diff",
            cmap="coolwarm",
            norm=norm,
            legend=True,
            legend_kwds={"shrink": 0.6},
        )
    except ValueError:
        metro_gdf.plot(
            ax=axes[2],
            column="cc_harmonic_1000_diff",
            cmap="coolwarm",
            legend=True,
            legend_kwds={"shrink": 0.6},
        )
    axes[2].set_title("Difference due to metro stops, 1000 m", loc="left")
    for ax in axes:
        ax.set_xlim(438500, 438500 + 3500)
        ax.set_ylim(4472500, 4472500 + 3500)
        ax.set_axis_off()
    plt.tight_layout()
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusion

    This notebook demonstrated how to integrate GTFS public transport data (the bundled metro feed) into the street network via `CityNetwork.add_gtfs` and compare centrality metrics with and without transit connectivity. By adding metro stops and segment travel times, the shortest-path algorithms can route through the transit network, revealing how public transport reshapes the pattern of closeness centrality across the district.

    **Next steps:** For GTFS with accessibility metrics, see [GTFS Accessibility](https://cityseer.benchmarkurbanism.com/examples/accessibility/accessibility-metro).
    """)
    return


if __name__ == "__main__":
    app.run()
