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
    # Directed Networks and One-Way Streets

    By default, `cityseer` builds undirected networks in which every street can be traversed in both directions. That is the correct model for pedestrian analysis. For cycling or vehicular questions, one-way restrictions change the available routes, and with them betweenness and closeness patterns. From v5, `cityseer` supports directed networks: one-way streets are restricted to their designated direction while two-way streets remain bidirectional.

    There are three ways to obtain a directed network:

    - `CityNetwork.from_geopandas(gdf, directed=True)` with a boolean `oneway` column, where one-way features run in their LineString coordinate order;
    - `CityNetwork.from_nx(G)` with a NetworkX `MultiDiGraph`, which enables directed mode automatically;
    - OSM data via [OSMnx](https://osmnx.readthedocs.io/), converted with `io.nx_from_osm_nx(G_osmnx, directed=True)`, shown below.

    Note that the graph cleaning functions in `tools.graphs` do not preserve edge directionality: pass directed graphs directly to the constructors rather than through manual simplification. See the [guide](https://cityseer.benchmarkurbanism.com/guide/networks#directed-graphs-and-one-way-streets).

    Network data © [OpenStreetMap](https://www.openstreetmap.org/copyright) contributors, available under the Open Database Licence.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import osmnx as ox
    from cityseer.network import CityNetwork
    from cityseer.tools import io

    return CityNetwork, io, ox, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fetching a drive network

    We fetch the drivable network for the city centre, where one-way circulation is common, and project it to a metric CRS. OSMnx returns a `MultiDiGraph` whose edges carry `oneway` attributes.
    """)
    return


@app.cell
def _(ox):
    G_osmnx = ox.graph.graph_from_point(
        (40.4169, -3.7035),  # city centre (lat, lng)
        dist=1200,
        network_type="drive",
        simplify=True,
    )
    G_osmnx_proj = ox.projection.project_graph(G_osmnx, to_crs="EPSG:25830")
    print(f"{G_osmnx_proj.number_of_nodes()} nodes, {G_osmnx_proj.number_of_edges()} directed edges")
    return (G_osmnx_proj,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Undirected and directed variants

    We convert the same OSMnx graph twice: once discarding directionality (the default) and once retaining it. Both are wrapped in the high-level `CityNetwork` class, which builds the dual graph and computes centrality with lean defaults (a single harmonic closeness and a single betweenness).
    """)
    return


@app.cell
def _(CityNetwork, G_osmnx_proj, io):
    G_undirected = io.nx_from_osm_nx(G_osmnx_proj, directed=False)
    cn_undirected = CityNetwork.from_nx(G_undirected)
    print("undirected:", cn_undirected.is_directed)

    G_directed = io.nx_from_osm_nx(G_osmnx_proj, directed=True)
    cn_directed = CityNetwork.from_nx(G_directed)
    print("directed:", cn_directed.is_directed)
    return cn_directed, cn_undirected


@app.cell
def _(cn_directed, cn_undirected):
    cn_undirected.centrality_shortest(distances=[800])
    cn_directed.centrality_shortest(distances=[800])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Comparing the results

    One-way restrictions lengthen many round-trip routes, which redistributes through-movement: betweenness concentrates on the streets that carry the permitted directions of flow.

    Two definitional points for directed betweenness:

    - each **ordered** origin-destination pair contributes with weight 1, so on a fully two-way network the directed values are twice the undirected magnitudes (A to B and B to A are counted separately);
    - routes are only counted where a permitted directed path exists within the distance threshold.

    For like-for-like visual comparison we therefore plot each variant on its own scale.
    """)
    return


@app.cell
def _(cn_directed, cn_undirected, plt):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=150)
    for ax, cn, title in (
        (axes[0], cn_undirected, "undirected"),
        (axes[1], cn_directed, "directed (one-way aware)"),
    ):
        streets = cn.to_geopandas()  # values projected back onto the street segments
        streets = streets[streets.live].copy()
        streets["_r"] = streets["cc_betweenness_800"].rank(pct=True)
        streets = streets.sort_values("_r")  # strongest drawn last, on top
        streets.plot(ax=ax, color=plt.get_cmap("OrRd")(streets["_r"]), linewidth=0.15 + 2.25 * streets["_r"])
        ax.set_title(f"Betweenness, 800 m ({title})", loc="left")
        ax.set_axis_off()
        ax.set_aspect("equal")
    fig.tight_layout()
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Practical notes

    - Directed mode composes with the other v5 features: expression dictionaries, [demand-weighted betweenness](https://cityseer.benchmarkurbanism.com/examples/flows/demand-flows), edge impedances, and elevation-aware costs all work on directed networks.
    - For a GeoDataFrame source, set `oneway=True` on the one-way features and check the digitised direction of the geometry: one-way streets flow in LineString coordinate order.
    - Keep pedestrian analyses undirected; footways are not directional, and one-way restrictions apply to vehicles.
    """)
    return


if __name__ == "__main__":
    app.run()
