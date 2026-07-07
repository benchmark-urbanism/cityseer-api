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
    # Weighted Flows vs Uniform Betweenness

    Standard betweenness assumes uniform demand: every origin-destination pair counts equally. Demand-weighted flows replace that assumption with where people actually are and where they are going. This recipe runs both on the **same network and threshold** so the difference is only the demand model, then maps them side by side. See the [Origin-Destination Flows guide](https://cityseer.benchmarkurbanism.com/guide/flows) for the background.
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
    ## Network and demand data

    A central-Madrid network, with Eurostat census population as **origins** and hospitality premises as **destinations**. `CityNetwork` builds the dual graph internally, so both measures share one basis.
    """)
    return


@app.cell
def _(CityNetwork, gpd, mo):
    data_dir = (mo.notebook_dir() / ".." / ".." / "data").resolve()
    streets_gpd = gpd.read_file(data_dir / "madrid_streets" / "street_network.gpkg")
    streets_gpd = streets_gpd.explode(ignore_index=True).drop_duplicates(subset="geometry")
    centre = gpd.GeoSeries.from_xy([440300], [4474300], crs=streets_gpd.crs)
    study_poly = centre.buffer(3000).iloc[0]
    buffered_poly = centre.buffer(3800).iloc[0]  # network buffered past the 800 m threshold
    cn = CityNetwork.from_geopandas(streets_gpd[streets_gpd.intersects(buffered_poly)], boundary=study_poly)

    census = gpd.read_file(data_dir / "madrid_census" / "eu_stat_clipped.gpkg").to_crs(streets_gpd.crs)
    origins = census[["T", "geometry"]].copy()
    origins["geometry"] = origins.geometry.centroid
    origins = origins[origins.within(buffered_poly)]
    premises = gpd.read_file(data_dir / "madrid_premises" / "madrid_premises.gpkg")
    dests = premises[premises["section_id"] == "I"].copy()
    dests = dests[dests.within(buffered_poly)]
    dests["weight"] = 1.0
    return cn, dests, origins


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compute both measures

    Uniform betweenness needs only the network. Demand-weighted flow additionally takes the weighted origins and destinations and a distance-decay deterrence function. Both write to `nodes_gdf`, and `to_geopandas` projects them onto the street segments for mapping.
    """)
    return


@app.cell
def _(cn, dests, origins):
    cn.centrality_shortest(distances=[800])  # uniform betweenness -> cc_betweenness_800
    cn.betweenness_demand(  # population -> hospitality, gravity-style decay -> cc_demand_800
        origins_gdf=origins,
        destinations_gdf=dests,
        origin_weight_col="T",
        destination_weight_col="weight",
        distances=[800],
        decay_fn="exp(-0.002 * c)",
    )
    streets = cn.to_geopandas()  # both metrics projected onto the street segments
    return (streets,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Map them side by side

    Both use the same styling (line width and orange-red colour scaled by a percentile rank of the value), so the patterns are directly comparable.
    """)
    return


@app.cell
def _(plt, streets):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    live = streets[streets.live]
    for ax, col, title in [
        (axes[0], "cc_betweenness_800", "Uniform betweenness"),
        (axes[1], "cc_demand_800", "Demand-weighted flow"),
    ]:
        g = live.copy()
        g["rank"] = g[col].rank(pct=True)
        g = g.sort_values("rank")  # strongest drawn last
        g.plot(ax=ax, color=plt.get_cmap("OrRd")(g["rank"]), linewidth=0.15 + 2.25 * g["rank"])
        ax.set_title(title, loc="left")
        ax.set_axis_off()
        ax.set_aspect("equal")
    fig.tight_layout()
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interpretation

    Uniform betweenness marks the network's structural through-routes: the streets most likely to lie on a shortest path between any two points. Demand-weighted flow shifts emphasis towards streets that connect where people live to where they go, so residential-to-amenity corridors strengthen and structurally central but lightly-used streets recede. Neither is more correct. Uniform betweenness reflects structural position and needs only the network; demand-weighted flow estimates use but depends on the quality of the origin and destination weights.
    """)
    return


if __name__ == "__main__":
    app.run()
