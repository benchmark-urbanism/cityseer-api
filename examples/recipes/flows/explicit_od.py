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

    When you have observed trips between zones, from a travel survey, ticketing records, or mobile traces, you already know the weight for each origin-destination pair. [`build_od_matrix`](https://cityseer.benchmarkurbanism.com/api/network#build_od_matrix) turns a flow table and a set of zones into a matrix (snapping each zone centroid to the nearest network node), and [`betweenness_od`](https://cityseer.benchmarkurbanism.com/api/network#betweenness_od) routes it: each pair's trips are accumulated along the shortest path between its zones. See the [Origin-Destination Flows guide](https://cityseer.benchmarkurbanism.com/guide/flows) for the background.

    There is no observed-trip dataset bundled here, so we **make one up** to show the mechanics, and then vary it: the same network and distance threshold, driven by two different demand patterns, produce very different flow maps. The busy streets depend on where the trips go, not on the network alone.
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
    ## Network and zones

    A central-Madrid network, with the city's *barrios* (neighbourhoods) as origin-destination zones. Each zone's centroid is snapped to the nearest network node when the matrix is built.
    """)
    return


@app.cell
def _(CityNetwork, gpd, mo):
    data_dir = (mo.notebook_dir() / ".." / ".." / "data").resolve()
    streets = gpd.read_file(data_dir / "madrid_streets" / "street_network.gpkg")
    streets = streets.explode(ignore_index=True).drop_duplicates(subset="geometry")
    centre = gpd.GeoSeries.from_xy([440300], [4474300], crs=streets.crs)
    study_poly = centre.buffer(3000).iloc[0]
    buffered_poly = centre.buffer(5500).iloc[0]  # network buffered past the 2km threshold
    streets_clip = streets[streets.intersects(buffered_poly)]
    cn = CityNetwork.from_geopandas(streets_clip, boundary=study_poly)

    zones = gpd.read_file(data_dir / "madrid_nbhds" / "madrid_nbhds.gpkg").to_crs(streets.crs)
    zones = zones[zones.intersects(study_poly)].reset_index(drop=True)
    zones["zone_id"] = zones["COD_DISB"]  # a unique district-barrio code
    print(f"{len(zones)} zones in the study area")
    return cn, study_poly, zones


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Two made-up demand patterns

    Both are illustrative stand-ins for observed data. The flow table just needs three columns: origin zone, destination zone, and a trip weight.

    - **Radial**: every zone sends trips to the single most central barrio (a commute-to-centre pattern).
    - **Orbital**: each zone sends trips to its neighbour going clockwise around the centre (a ring, or cross-town, pattern).
    """)
    return


@app.cell
def _(np, pd, study_poly, zones):
    cx, cy = study_poly.centroid.x, study_poly.centroid.y
    cent = zones.geometry.centroid
    # the central barrio is the one whose centroid is nearest the study centre
    dist_to_centre = np.hypot(cent.x - cx, cent.y - cy)
    centre_zone = zones.loc[dist_to_centre.idxmin(), "zone_id"]

    # radial: every zone -> the central barrio
    radial = pd.DataFrame({"origin": zones["zone_id"], "destination": centre_zone, "trips": 100.0})
    radial = radial[radial["origin"] != centre_zone]

    # orbital: sort zones by angle around the centre, connect each to the next (a ring)
    angle = np.arctan2(cent.y - cy, cent.x - cx)
    ring = zones.assign(angle=angle).sort_values("angle")["zone_id"].to_numpy()
    orbital = pd.DataFrame({"origin": ring, "destination": np.roll(ring, -1), "trips": 100.0})

    print(f"central barrio: {centre_zone}; {len(radial)} radial pairs, {len(orbital)} orbital pairs")
    return orbital, radial


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Route each pattern and snapshot the result

    `build_od_matrix` snaps zone centroids to network nodes and assembles the sparse matrix; `betweenness_od` routes it at the 2 km threshold. We snapshot each result with `to_geopandas` before running the next, because both write to the same `cc_betweenness_2000` column.
    """)
    return


@app.cell
def _(cn, orbital, radial, zones):
    def route(od_df):
        matrix = cn.build_od_matrix(
            od_df, zones, origin_col="origin", destination_col="destination", weight_col="trips", zone_id_col="zone_id"
        )
        cn.betweenness_od(matrix, distances=[2000])
        return cn.to_geopandas()

    gdf_radial = route(radial)
    gdf_orbital = route(orbital)
    return gdf_orbital, gdf_radial


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Mapping the two flow patterns

    Line width and orange-red colour both scale with a percentile rank of the flow, so busy corridors read as bold and quiet streets fall away to hairlines, no colour bar needed. The same styling is used across the flow recipes.
    """)
    return


@app.cell
def _(gdf_orbital, gdf_radial, plt):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    for ax, gdf, title in [
        (axes[0], gdf_radial, "Radial · to the centre"),
        (axes[1], gdf_orbital, "Orbital · around the ring"),
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

    The network, threshold, and routing are identical; only the demand differs. The radial pattern concentrates flow onto the arteries feeding the centre, while the orbital pattern raises the cross-town streets that the radial pattern barely uses. With real survey or ticketing data in place of these made-up tables, `betweenness_od` maps where actual trips load the network, instead of the uniform-demand assumption of standard betweenness.

    When you have weighted origins and destinations but no observed pairs, model the demand instead with [`betweenness_demand`](https://cityseer.benchmarkurbanism.com/examples/flows/demand-flows).
    """)
    return


if __name__ == "__main__":
    app.run()
