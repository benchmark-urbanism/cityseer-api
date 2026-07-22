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

    `CityNetwork.betweenness_demand` runs the model, and `to_geopandas` projects the result onto the street segments for mapping. The bundled datasets and their attributions are on the [datasets page](https://cityseer.benchmarkurbanism.com/examples/datasets).
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
    ## Data: population origins and hospitality destinations

    We model home-to-amenity movement in central Madrid: **origins** are the Eurostat census grid (column `T`, total population per cell), and **destinations** are hospitality premises (`section_id == "I"`), each weighted equally. Both are assigned to the network with the library's shared, representation-aware assignment, so the along-street offset from each point to the network enters every routed distance.
    """)
    return


@app.cell
def _(CityNetwork, gpd, mo):
    data_dir = (mo.notebook_dir() / ".." / ".." / "data").resolve()
    streets_gpd = gpd.read_file(data_dir / "madrid_streets" / "street_network.gpkg")
    streets_gpd = streets_gpd.explode(ignore_index=True)
    # the source data contains a small number of exact duplicate geometries
    streets_gpd = streets_gpd.drop_duplicates(subset="geometry")
    # 3km study area around the centre, buffered a further 800m (the analysis distance)
    centre = gpd.GeoSeries.from_xy([440300], [4474300], crs=streets_gpd.crs)
    study_poly = centre.buffer(3000).iloc[0]
    buffered_poly = centre.buffer(3800).iloc[0]
    cn = CityNetwork.from_geopandas(streets_gpd[streets_gpd.intersects(buffered_poly)], boundary=study_poly)
    print(f"{cn.node_count} street segments, {int(cn.nodes_gdf.live.sum())} live")
    return buffered_poly, cn, data_dir


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
def _(cn, dests_gpd, origins_gpd):
    cn.betweenness_demand(
        origins_gdf=origins_gpd,
        destinations_gdf=dests_gpd,
        origin_weight_col="T",
        destination_weight_col="weight",
        distances=[800],
        decay_fn="exp(-0.002 * c)",
    )
    streets = cn.to_geopandas()  # cc_demand_800 projected onto the street segments
    print(streets["cc_demand_800"].describe())
    return (streets,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Mapping the flows

    We map intensity by **line width and colour together**: a percentile rank spreads the heavily-skewed flow values across a hairline-to-bold width ramp, and the same rank drives an orange-red colour. Strong streets are drawn last so they sit on top. The map is legible without a colour bar.
    """)
    return


@app.cell
def _(plt, streets):
    live = streets[streets.live].copy()
    # percentile rank spreads the skewed flow values across [0, 1]
    live["rank"] = live["cc_demand_800"].rank(pct=True)
    live = live.sort_values("rank")  # strongest drawn last, on top
    fig, ax = plt.subplots(figsize=(7, 7))
    live.plot(
        ax=ax,
        color=plt.get_cmap("OrRd")(live["rank"]),
        linewidth=0.15 + 2.25 * live["rank"],
    )
    ax.set_title("Modelled demand flows · population to hospitality · 800 m", loc="left")
    ax.set_axis_off()
    ax.set_aspect("equal")
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
    - lower `participation` below its default of `1.0` (for example `0.2`) to make trip generation accessibility-elastic: a stay-home option enters each origin's choice set, so locations with poorer access generate fewer trips;
    - when you have observed flows rather than modelled demand, route an explicit matrix with [`build_od_matrix`](https://cityseer.benchmarkurbanism.com/api/network#build_od_matrix) and [`betweenness_od`](https://cityseer.benchmarkurbanism.com/api/network#betweenness_od);
    - compare against the uniform betweenness baseline from the [metric centrality recipe](https://cityseer.benchmarkurbanism.com/examples/centrality/gpd-metric-centrality) to see how demand weighting shifts the pattern.
    """)
    return


if __name__ == "__main__":
    app.run()
