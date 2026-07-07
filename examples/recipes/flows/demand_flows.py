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
    # Modelled Demand Flows

    Standard betweenness treats every origin-destination pair as equally important. This recipe instead **models** the demand between weighted origins and destinations, then routes it across the network so each street accumulates the through-movement that passes along it. See the [Origin-Destination Flows guide](https://cityseer.benchmarkurbanism.com/guide/flows) for the background.

    [`betweenness_demand`](https://cityseer.benchmarkurbanism.com/metrics/networks#betweenness_demand) uses a **singly (origin-)constrained** spatial interaction model. Each origin distributes its full weight $W_o$ across the destinations it can reach, giving each a share proportional to its attractiveness $W_d$ discounted by the network cost of getting there:

    $$W_{od} = W_o \cdot \frac{W_d \, f(c_{od})}{\sum_{d'} W_{d'} \, f(c_{od'})}$$

    where $f$ is the deterrence function `decay_fn` and $c_{od}$ is the network distance. With an exponential decay this is the classic gravity model. Each allocated flow is routed along the shortest path and every street traversed accumulates it, so no explicit matrix is needed.

    This uses the lower-level functional API: `betweenness_demand` is not wrapped by `CityNetwork`. The bundled datasets and their attributions are on the [datasets page](https://cityseer.benchmarkurbanism.com/examples/datasets).
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
    ## Data: population origins and hospitality destinations

    We model home-to-amenity movement in central Madrid: **origins** are the Eurostat census grid (column `T`, total population per cell), and **destinations** are hospitality premises (`section_id == "I"`), each weighted equally. Both are snapped to the nearest network node.
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
    # 3km study area around the centre, buffered a further 800m (the analysis distance)
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

    The `decay_fn` expression is the deterrence function $f$. It uses the same variables as other cityseer expressions: `c` (network metres) and `p` (progress towards the threshold). `"exp(-0.002 * c)"` is a gravity-style exponential suited to an 800 m walking threshold; see the [decay documentation](https://cityseer.benchmarkurbanism.com/api/decay) for alternatives. Setting `closest_destination=True` switches to all-or-nothing assignment, where each origin sends its full weight to the single nearest destination.
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Mapping the flows

    We map intensity by **marker size and colour together**: a percentile rank spreads the heavily-skewed flow values across a hairline-to-bold size ramp, and the same rank drives an orange-red colour. Strong streets are drawn last so they sit on top. The map is legible without a colour bar.
    """)
    return


@app.cell
def _(nodes_demand, plt):
    live = nodes_demand[nodes_demand.live].copy()
    # percentile rank spreads the skewed flow values across [0, 1]
    live["rank"] = live["cc_demand_800"].rank(pct=True)
    live = live.sort_values("rank")  # strongest drawn last, on top
    fig, ax = plt.subplots(figsize=(7, 7))
    live.plot(
        ax=ax,
        color=plt.get_cmap("OrRd")(live["rank"]),
        markersize=0.5 + 12 * live["rank"],
    )
    ax.set_title("Modelled demand flows · population to hospitality · 800 m", loc="left")
    ax.set_axis_off()
    fig.tight_layout()
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interpretation and variations

    High values mark streets carrying many short home-to-amenity trips, weighted by nearby population and the attractiveness of reachable destinations. Compared with unweighted betweenness, the demand-weighted flow concentrates where population and destinations actually interact.

    Variations to explore:

    - swap the destination set (parks, schools, transit stops), or weight destinations by a capacity column such as floorspace;
    - adjust the deterrence: a steeper decay (`"exp(-0.008 * c)"`) localises flows, a flat function (`"1"`) removes distance deterrence entirely;
    - when you have observed flows rather than modelled demand, route an explicit matrix with [`build_od_matrix`](https://cityseer.benchmarkurbanism.com/api/network#build_od_matrix) and [`betweenness_od`](https://cityseer.benchmarkurbanism.com/api/network#betweenness_od);
    - compare against the uniform betweenness baseline from the [metric centrality recipe](https://cityseer.benchmarkurbanism.com/examples/centrality/gpd-metric-centrality) to see how demand weighting shifts the pattern.
    """)
    return


if __name__ == "__main__":
    app.run()
