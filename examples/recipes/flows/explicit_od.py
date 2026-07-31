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
    # Routing an Explicit OD Matrix

    When you have observed trips between zones, from a travel survey, ticketing records, or mobile traces, you already know the weight for each origin-destination pair. [`build_od_matrix`](https://cityseer.benchmarkurbanism.com/api/network#build-od-matrix) turns a flow table and a set of zones into a matrix, assigning each zone centroid to its nearest network node with the library's shared assignment, and [`betweenness_od`](https://cityseer.benchmarkurbanism.com/api/network#betweenness-od) routes it: each pair's trips are accumulated along the shortest path between its zones. See the [Origin-Destination Flows guide](https://cityseer.benchmarkurbanism.com/guide/flows) for the background.

    No observed-trip dataset is bundled here, so we synthesise one with a gravity model: trips between two zones grow with the population of each and fall away with the distance between them. This is a standard way to estimate a plausible trip table when only zone populations are known. We build two demand scenarios on the same network, route each at a 2 km threshold, and compare the resulting flow maps.
    """)
    return


@app.cell
def _():
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from cityseer.network import CityNetwork

    return CityNetwork, gpd, np, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Network, zones, and population

    A central-Madrid network, with the city's *barrios* (neighbourhoods) as origin-destination zones. Each zone's total population is summed from the Eurostat census grid.
    """)
    return


@app.cell
def _(CityNetwork, gpd, mo):
    data_dir = (mo.notebook_dir() / ".." / ".." / "data").resolve()
    streets = gpd.read_file(data_dir / "madrid_streets" / "street_network.gpkg")
    streets = streets.explode(ignore_index=True).drop_duplicates(subset="geometry")
    centre = gpd.GeoSeries.from_xy([440300], [4474300], crs=streets.crs)
    study_poly = centre.buffer(3000).iloc[0]
    buffered_poly = centre.buffer(5500).iloc[0]  # network buffered past the 2 km threshold
    cn = CityNetwork.from_geopandas(streets[streets.intersects(buffered_poly)], boundary=study_poly)

    zones = gpd.read_file(data_dir / "madrid_nbhds" / "madrid_nbhds.gpkg").to_crs(streets.crs)
    zones = zones[zones.intersects(study_poly)].reset_index(drop=True)
    zones["zone_id"] = zones["COD_DISB"]  # a unique district-barrio code

    # total population per zone, summed from the Eurostat census grid cells within it
    census = gpd.read_file(data_dir / "madrid_census" / "eu_stat_clipped.gpkg").to_crs(streets.crs)
    census["geometry"] = census.geometry.centroid
    counts = gpd.sjoin(census[["T", "geometry"]], zones[["zone_id", "geometry"]], predicate="within")
    zones["pop"] = zones["zone_id"].map(counts.groupby("zone_id")["T"].sum()).fillna(0.0)
    print(f"{len(zones)} zones; total population {zones['pop'].sum():.0f}")
    return cn, study_poly, zones


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Two demand scenarios

    A flow table needs three columns: origin zone, destination zone, and a trip weight. We build the weight for every pair of zones with the same gravity rule, `population_origin * population_destination * exp(-beta * distance)`, and change only where the trips are allowed to go.

    - **Commute to the centre**: trips run to the central barrios only, decaying gently with distance (a monocentric commute).
    - **Local trips everywhere**: trips run between all zones, but a steeper decay means short trips dominate (dispersed, everyday mobility).
    """)
    return


@app.cell
def _(np, study_poly, zones):
    cx, cy = study_poly.centroid.x, study_poly.centroid.y
    zones["px"] = zones.geometry.centroid.x
    zones["py"] = zones.geometry.centroid.y
    zones["central"] = np.hypot(zones.px - cx, zones.py - cy) < 1200  # the inner ring of barrios

    # every ordered pair of zones, with the straight-line distance between their centroids
    left = zones[["zone_id", "px", "py", "pop"]]
    right = zones[["zone_id", "px", "py", "pop", "central"]]
    pairs = left.merge(right, how="cross", suffixes=("_o", "_d"))
    pairs = pairs[pairs.zone_id_o != pairs.zone_id_d].copy()
    pairs["dist"] = np.hypot(pairs.px_o - pairs.px_d, pairs.py_o - pairs.py_d)

    # same gravity rule, different destinations and decay
    commute = pairs[pairs.central].assign(trips=lambda p: p.pop_o * p.pop_d * np.exp(-0.0008 * p.dist))
    local = pairs.assign(trips=lambda p: p.pop_o * p.pop_d * np.exp(-0.0025 * p.dist))

    print(f"{len(commute)} commute pairs, {len(local)} local pairs")
    return commute, local


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Access points within each zone

    A zonal trip table records how many trips run between two zones, not where within each zone they start and end. Passing the zone polygons straight to `build_od_matrix` assigns each zone to the single network node nearest its centroid, so every pair's trips load onto one shortest path and the map reduces to a sparse set of centroid-to-centroid lines. Transport models handle this by connecting each zone to the network at several points (centroid connectors). We do the same here: a 350 m point grid within each zone, clipped to the study area, with the zone's trips divided evenly among its points.
    """)
    return


@app.cell
def _(gpd, np, pd, study_poly, zones):
    clipped = zones[["zone_id", "geometry"]].copy()
    clipped["geometry"] = clipped.geometry.intersection(study_poly)

    # a 350m point grid, each point assigned to the zone that contains it
    minx, miny, maxx, maxy = clipped.total_bounds
    gx, gy = np.meshgrid(np.arange(minx, maxx, 350.0), np.arange(miny, maxy, 350.0))
    grid = gpd.GeoDataFrame(geometry=gpd.points_from_xy(gx.ravel(), gy.ravel()), crs=zones.crs)
    sites = gpd.sjoin(grid, clipped, predicate="within")[["zone_id", "geometry"]]

    # a zone too small to catch a grid point falls back to a single interior point
    missing = clipped[~clipped.zone_id.isin(sites.zone_id)]
    if len(missing):
        fallback = missing.assign(geometry=missing.geometry.representative_point())
        sites = pd.concat([sites, fallback], ignore_index=True)

    sites = sites.reset_index(drop=True)
    sites["site_id"] = sites.index
    n_sites = sites.groupby("zone_id").size()
    print(f"{len(sites)} access points across {len(n_sites)} zones")
    return n_sites, sites


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Route each scenario and snapshot the result

    Each zone-to-zone flow is split evenly across its origin and destination access points, then `build_od_matrix` assigns the points to network nodes and assembles the sparse matrix, and `betweenness_od` routes it at the 2 km threshold. We snapshot each result with `to_geopandas` before running the next, because both write to the same `cc_betweenness_2000` column.
    """)
    return


@app.cell
def _(cn, commute, local, n_sites, sites):
    def expand(od):
        e = od.merge(sites[["zone_id", "site_id"]], left_on="zone_id_o", right_on="zone_id")
        e = e.rename(columns={"site_id": "site_o"}).drop(columns="zone_id")
        e = e.merge(sites[["zone_id", "site_id"]], left_on="zone_id_d", right_on="zone_id")
        e = e.rename(columns={"site_id": "site_d"}).drop(columns="zone_id")
        e["trips"] = e["trips"] / (e["zone_id_o"].map(n_sites) * e["zone_id_d"].map(n_sites))
        return e

    def route(od):
        matrix = cn.build_od_matrix(
            expand(od), sites, origin_col="site_o", destination_col="site_d", weight_col="trips", zone_id_col="site_id"
        )
        cn.betweenness_od(matrix, distances=[2000])
        return cn.to_geopandas()

    gdf_commute = route(commute)
    gdf_local = route(local)
    return gdf_commute, gdf_local


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Mapping the two flow patterns

    Line width and orange-red colour both scale with a percentile rank of the flow, so the map reads without a colour bar. The same styling is used across the flow recipes.
    """)
    return


@app.cell
def _(gdf_commute, gdf_local, plt):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    for ax, gdf, title in [
        (axes[0], gdf_commute, "Commute to the centre"),
        (axes[1], gdf_local, "Local trips everywhere"),
    ]:
        live = gdf[gdf.live].copy()
        live["rank"] = live["cc_betweenness_2000"].rank(pct=True)
        live = live.sort_values("rank")  # strongest drawn last
        live.plot(ax=ax, color=plt.get_cmap("OrRd")(live["rank"]), linewidth=0.15 + 2.25 * live["rank"])
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

    The network, threshold, and routing are identical; only the demand differs. The commute scenario concentrates flow on the streets converging on the central barrios, fading with distance from the centre. The local scenario distributes flow across the densely populated fabric, with cross-town streets carrying more than the centre itself. With real survey or ticketing data in place of the gravity estimate, `betweenness_od` maps where actual trips load the network, instead of the uniform-demand assumption of standard betweenness.

    When you have weighted origins and destinations but no trip table at all, model the demand directly with [`betweenness_demand`](https://cityseer.benchmarkurbanism.com/examples/flows/demand-flows).
    """)
    return


if __name__ == "__main__":
    app.run()
