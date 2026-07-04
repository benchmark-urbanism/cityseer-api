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
    # Demand-Weighted Betweenness: Origin-Destination Flows

    Standard betweenness treats every pair of network nodes as equally important: each pair contributes one unit of flow to the nodes on the shortest path between them. That is a reasonable neutral assumption, but real movement is not uniform. Trips start where people are and end where destinations are.

    `cityseer` provides two functions that route weighted flows instead:

    - [`betweenness_od`](https://cityseer.benchmarkurbanism.com/metrics/networks#betweenness_od) routes an **explicit** origin-destination matrix, built from observed flow data with [`build_od_matrix`](https://cityseer.benchmarkurbanism.com/metrics/networks#build_od_matrix). Use this when you have trip data, for example from a travel survey or mobile data.
    - [`betweenness_demand`](https://cityseer.benchmarkurbanism.com/metrics/networks#betweenness_demand) **models** the matrix from weighted origins and destinations using a singly constrained spatial interaction model. Use this when you have population and attractor data but no observed trips. This notebook demonstrates this function.

    The higher-level [`CityNetwork`](https://cityseer.benchmarkurbanism.com/guide/fundamentals#citynetwork-api) class wraps `betweenness_od` and `build_od_matrix`; `betweenness_demand` is part of the lower-level functional API, which this recipe uses throughout.

    ## The singly constrained model

    "Singly constrained" means the model is constrained at the origin end only: each origin distributes exactly its full weight $W_o$ (no more, no less) across the destinations it can reach within the distance threshold. Destination $d$ receives a share proportional to its attractiveness $W_d$ discounted by the cost of getting there:

    $$T_{od} = W_o \cdot \frac{W_d \, f(c_{od})}{\sum_{d'} W_{d'} \, f(c_{od'})}$$

    where $f$ is a distance-decay (deterrence) function supplied as a `decay_fn` expression. With an exponential decay this is the classic gravity model. Each allocated flow $T_{od}$ is then routed along the shortest network path, and every node traversed accumulates the flow: the result, written to `cc_demand_{distance}`, is the modelled through-movement at each location. The allocation and routing happen in a single traversal, so no explicit matrix is materialised.

    The bundled datasets and their source attributions are documented on the [datasets page](https://cityseer.benchmarkurbanism.com/examples/datasets).
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
    ## Data: population origins and retail destinations

    We model home-to-amenity movement in the city centre using two bundled datasets:

    - **origins**: the Eurostat census grid, whose `T` column is total population per cell. Cells are polygons; we use their centroids as origin points.
    - **destinations**: the bundled premises census, filtered to hospitality (`section_id == "I"`), with each premise weighted equally.

    Origins and destinations are snapped to the nearest network node (within `max_snap_dist`, default 100m). Weights of points snapping to the same node are summed, so total demand is preserved at each junction.
    """)
    return


@app.cell
def _(gpd, graphs, io, mo):
    from shapely import geometry

    data_dir = (mo.notebook_dir() / ".." / ".." / "data").resolve()
    streets_gpd = gpd.read_file(data_dir / "madrid_streets" / "street_network.gpkg")
    streets_gpd = streets_gpd.explode(ignore_index=True)
    # the source data contains a small number of exact duplicate geometries
    streets_gpd = streets_gpd.drop_duplicates(subset="geometry")
    # 3km study area around the centre, with the network buffered a further 800m
    # (the maximum analysis distance) to prevent edge rolloff
    centre = gpd.GeoSeries.from_xy([440300], [4474300], crs=streets_gpd.crs)
    study_poly = centre.buffer(3000).iloc[0]
    buffered_poly = centre.buffer(3800).iloc[0]
    streets_clip = streets_gpd[streets_gpd.intersects(buffered_poly)]
    G = io.nx_from_generic_geopandas(streets_clip)
    for _idx, _data in G.nodes(data=True):
        G.nodes[_idx]["live"] = study_poly.contains(geometry.Point(_data["x"], _data["y"]))
    G_dual = graphs.nx_to_dual(G)
    nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(G_dual)
    print(f"{G_dual.number_of_nodes()} dual nodes, {int(nodes_gdf.live.sum())} live")
    return buffered_poly, data_dir, network_structure, nodes_gdf


@app.cell
def _(buffered_poly, data_dir, gpd):
    # origins: census grid centroids with total population
    census_gpd = gpd.read_file(data_dir / "madrid_census" / "eu_stat_clipped.gpkg")
    census_gpd = census_gpd.to_crs(25830)
    origins_gpd = census_gpd[["T", "geometry"]].copy()
    origins_gpd["geometry"] = origins_gpd.geometry.centroid
    origins_gpd = origins_gpd[origins_gpd.within(buffered_poly)]

    # destinations: hospitality premises, weighted equally
    premises_gpd = gpd.read_file(data_dir / "madrid_premises" / "madrid_premises.gpkg")
    dests_gpd = premises_gpd[premises_gpd["section_id"] == "I"].copy()
    dests_gpd = dests_gpd[dests_gpd.within(buffered_poly)]
    dests_gpd["weight"] = 1.0
    print(f"{len(origins_gpd)} origin cells, {len(dests_gpd)} destination premises")
    return dests_gpd, origins_gpd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Computing demand-weighted betweenness

    The `decay_fn` expression is the deterrence function $f$. It uses the same variables as other cityseer expressions: `c` (network metres) and `p` (progress towards the threshold). `"exp(-0.002 * c)"` is a gravity-style exponential with a beta of 0.002, a gentle decay suited to an 800m walking threshold; see the [decay documentation](https://cityseer.benchmarkurbanism.com/guide/land-use#decay-functions) for alternatives.

    Setting `closest_destination=True` switches from proportional allocation to an all-or-nothing assignment where each origin sends its full weight to the single best destination, appropriate for services chosen by proximity alone (for example, postboxes).
    """)
    return


@app.cell
def _(dests_gpd, network_structure, networks, nodes_gdf, origins_gpd):
    nodes_demand = networks.betweenness_demand(
        network_structure,
        nodes_gdf.copy(),
        origins_gdf=origins_gpd,
        destinations_gdf=dests_gpd,
        origin_weight_col="T",
        destination_weight_col="weight",
        distances=[800],
        decay_fn="exp(-0.002 * c)",
    )
    print(nodes_demand["cc_demand_800"].describe())
    return (nodes_demand,)


@app.cell
def _(nodes_demand, plt):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    live_nodes = nodes_demand[nodes_demand.live]
    live_nodes.plot(
        ax=ax,
        column="cc_demand_800",
        cmap="magma",
        markersize=2,
        legend=True,
        legend_kwds={"label": "Demand-weighted betweenness, 800 m", "shrink": 0.6},
    )
    ax.set_title("Demand-weighted betweenness (population to hospitality, 800 m)")
    ax.set_axis_off()
    fig.tight_layout()
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interpretation and variations

    The map shows modelled pedestrian through-movement between homes and hospitality venues: streets with high values carry many short home-to-amenity trips, weighted by how many people live nearby and how attractive the reachable destinations are. Compare this with unweighted betweenness from the [metric centrality recipe](https://cityseer.benchmarkurbanism.com/examples/centrality/gpd-metric-centrality): the demand-weighted variant concentrates flow where population and destinations actually interact.

    Variations to explore:

    - swap the destination set (parks, schools, transit stops) or weight destinations by a capacity column such as floorspace;
    - adjust the deterrence function: a steeper decay (`"exp(-0.008 * c)"`) localises flows, a flat function (`"1"`) removes distance deterrence entirely;
    - use `betweenness_od` with an explicit matrix when observed flows are available;
    - for one-way street networks, combine with directed mode (see the [directed networks recipe](https://cityseer.benchmarkurbanism.com/examples/networks/directed-networks)).
    """)
    return


if __name__ == "__main__":
    app.run()
