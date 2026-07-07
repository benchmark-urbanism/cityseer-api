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
    from cityseer.metrics import networks
    from cityseer.tools import graphs, io

    return gpd, graphs, io, networks, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Network and demand data

    A central-Madrid network (as a dual graph, so both measures share one basis), with Eurostat census population as **origins** and hospitality premises as **destinations**.
    """)
    return


@app.cell
def _(gpd, graphs, io, mo):
    from shapely import geometry

    data_dir = (mo.notebook_dir() / ".." / ".." / "data").resolve()
    streets = gpd.read_file(data_dir / "madrid_streets" / "street_network.gpkg")
    streets = streets.explode(ignore_index=True).drop_duplicates(subset="geometry")
    centre = gpd.GeoSeries.from_xy([440300], [4474300], crs=streets.crs)
    study_poly = centre.buffer(3000).iloc[0]
    buffered_poly = centre.buffer(3800).iloc[0]
    streets_clip = streets[streets.intersects(buffered_poly)]
    G = io.nx_from_generic_geopandas(streets_clip)
    for _idx, _data in G.nodes(data=True):
        G.nodes[_idx]["live"] = study_poly.contains(geometry.Point(_data["x"], _data["y"]))
    G_dual = graphs.nx_to_dual(G)
    nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(G_dual)

    census = gpd.read_file(data_dir / "madrid_census" / "eu_stat_clipped.gpkg").to_crs(streets.crs)
    origins = census[["T", "geometry"]].copy()
    origins["geometry"] = origins.geometry.centroid
    origins = origins[origins.within(buffered_poly)]
    premises = gpd.read_file(data_dir / "madrid_premises" / "madrid_premises.gpkg")
    dests = premises[premises["section_id"] == "I"].copy()
    dests = dests[dests.within(buffered_poly)]
    dests["weight"] = 1.0
    return dests, network_structure, nodes_gdf, origins


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compute both measures

    Uniform betweenness needs only the network. Demand-weighted flow additionally takes the weighted origins and destinations and a distance-decay deterrence function.
    """)
    return


@app.cell
def _(dests, network_structure, networks, nodes_gdf, origins):
    # uniform: every pair equal
    nodes_plain = networks.betweenness_shortest(network_structure, nodes_gdf.copy(), distances=[800])
    # demand-weighted: population -> hospitality, gravity-style decay
    nodes_demand = networks.betweenness_demand(
        network_structure,
        nodes_gdf.copy(),
        origins_gdf=origins,
        destinations_gdf=dests,
        origin_weight_col="T",
        destination_weight_col="weight",
        distances=[800],
        decay_fn="exp(-0.002 * c)",
    )
    return nodes_demand, nodes_plain


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Map them side by side

    Both use the same styling (marker size and orange-red colour scaled by a percentile rank of the value), so the patterns are directly comparable.
    """)
    return


@app.cell
def _(nodes_demand, nodes_plain, plt):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    panels = [
        (axes[0], nodes_plain, "cc_betweenness_800", "Uniform betweenness"),
        (axes[1], nodes_demand, "cc_demand_800", "Demand-weighted flow"),
    ]
    for ax, gdf, col, title in panels:
        live = gdf[gdf.live].copy()
        live["rank"] = live[col].rank(pct=True)
        live = live.sort_values("rank")  # strongest drawn last
        live.plot(ax=ax, color=plt.get_cmap("OrRd")(live["rank"]), markersize=0.5 + 12 * live["rank"])
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
