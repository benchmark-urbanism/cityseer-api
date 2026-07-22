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
    # London Cycling Flows: Observed and Modelled

    This recipe maps cycle-commute through-movement across central London two ways, from observed trips and from modelled trips, using only open data.

    - **Observed**: the [Propensity to Cycle Tool](https://www.pct.bike/) (PCT) gives journey-to-work bicycle counts between MSOA zones. [`betweenness_od`](https://cityseer.benchmarkurbanism.com/api/network#betweenness_od) routes that explicit matrix.
    - **Modelled**: [`betweenness_demand`](https://cityseer.benchmarkurbanism.com/api/network#betweenness_demand) allocates trips from weighted origins to weighted destinations with a singly-constrained gravity model, then routes them. No matrix is supplied.

    Modelling the flows lets us check the model against the observed pattern, and calibrate the two decay levers separately: the distance decay `beta` shapes destination choice, and `participation` scales how many trips are generated.

    The observed cycle flows come from the **Propensity to Cycle Tool (PCT)**, developed by **Robin Lovelace** and colleagues (Lovelace et al., 2017), which we gratefully acknowledge as the source of the origin-destination cycling data used here.

    Data and licences: street network © [OpenStreetMap](https://www.openstreetmap.org/copyright) contributors (ODbL); cycle-commute origin-destination flows from the [PCT](https://www.pct.bike/) ([pct-data](https://github.com/Robinlovelace/pct-data)), derived from the 2011 Census travel-to-work table (WU03EW), Open Government Licence; MSOA zone boundaries © ONS, Open Government Licence.

    > Lovelace, R., Goodman, A., Aldred, R., Berkoff, N., Abbas, A. and Woodcock, J. (2017). The Propensity to Cycle Tool: An open source online system for sustainable transport planning. *Journal of Transport and Land Use*, 10(1), 505-528. [doi:10.5198/jtlu.2016.862](https://doi.org/10.5198/jtlu.2016.862)
    """)
    return


@app.cell
def _():
    import tempfile
    import urllib.request
    from pathlib import Path

    import geopandas as gpd
    import numpy as np
    import pandas as pd
    from cityseer.network import CityNetwork
    from cityseer.tools import io
    from matplotlib import pyplot as plt
    from scipy.stats import spearmanr

    return CityNetwork, Path, gpd, io, np, pd, plt, spearmanr, tempfile, urllib


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Network and data

    The network is downloaded from OpenStreetMap for a 4 km study area around the centre. `from_osm` fetches, simplifies, and builds the dual graph in one call, including cycleways by default.

    The PCT files are downloaded once and cached: `l.csv` has one row per (residence MSOA, workplace MSOA) pair with a commuter count per mode; `z.geojson` holds the MSOA boundaries. We keep the zones whose centroids fall inside the study area and the pairs between them.
    """)
    return


@app.cell
def _(CityNetwork, Path, gpd, io, pd, tempfile, urllib):
    # central-London study area; build the OSM network in the British National Grid.
    # green_footways / green_service_roads keep the paths through parks (Regent's Park,
    # Hyde Park), which cyclists use but which a roads-only network omits, forcing detours.
    # cache_path caches the Overpass response to a temp file so re-runs reuse it rather than
    # re-querying (Overpass rate-limits by request volume, so caching is what helps; if a first
    # run hits a busy Overpass, simply retry).
    cache = Path(tempfile.gettempdir()) / "cityseer_pct_london"
    cache.mkdir(exist_ok=True)
    lng, lat = -0.12, 51.51
    poly_wgs, _epsg = io.buffered_point_poly(lng, lat, 4000)
    cn = CityNetwork.from_osm(
        poly_wgs,
        to_crs_code=27700,
        green_footways=True,
        green_service_roads=True,
        cache_path=str(cache / "osm_network.json"),
    )

    # PCT London cycle-commute data, downloaded once to the same temp cache
    base = "https://raw.githubusercontent.com/Robinlovelace/pct-data/master/london"
    for _name in ("l.csv", "z.geojson"):
        if not (cache / _name).exists():
            urllib.request.urlretrieve(f"{base}/{_name}", cache / _name)
    l_df = pd.read_csv(cache / "l.csv")
    zones = gpd.read_file(cache / "z.geojson").to_crs(27700)

    # zones and pairs inside the study area
    centre = gpd.GeoSeries.from_xy([lng], [lat], crs=4326).to_crs(27700).iloc[0]
    study = centre.buffer(4500)
    zones_in = zones[zones.geometry.centroid.within(study)].reset_index(drop=True)
    keep = set(zones_in["geo_code"])
    sub = l_df[l_df["msoa1"].isin(keep) & l_df["msoa2"].isin(keep)]
    print(f"{len(zones_in)} MSOAs, {len(sub)} OD pairs, {sub['bicycle'].sum():.0f} cycle commutes")
    return cn, l_df, sub, zones_in


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Observed flows with `betweenness_od`

    `build_od_matrix` turns the flow table and the zones into a sparse matrix, assigning each MSOA centroid to its nearest network node. `betweenness_od` then routes each pair's `bicycle` count along the shortest path between its zones and accumulates the flow on the streets in between.

    The distance threshold is set to 20 km, beyond the study extent, so that in the modelled version below the decay rather than a hard cutoff governs destination choice. `tolerance` spreads flow across near-equal shortest paths, which avoids the artificial striping that a single arbitrary geodesic produces on a grid.
    """)
    return


@app.cell
def _(cn, sub, zones_in):
    D = 20000
    matrix = cn.build_od_matrix(
        sub[["msoa1", "msoa2", "bicycle"]],
        zones_in,
        origin_col="msoa1",
        destination_col="msoa2",
        weight_col="bicycle",
        zone_id_col="geo_code",
        max_netw_assign_dist=1500.0,
    )
    cn.centrality_shortest(distances=[D], closeness={}, betweenness=None, cycles=False)
    std = cn.to_geopandas()
    cn.betweenness_od(matrix, distances=[D], tolerance=10.0)
    observed = cn.to_geopandas()
    return D, observed, std


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Modelled flows with `betweenness_demand`

    Rather than supplying the matrix, we model it. The inputs are deliberately independent of the cycle counts we are trying to reproduce: each MSOA is weighted by its all-mode commuter mass, residents departing it as origins and workers arriving at it as destinations (the `all` column summed by residence and by workplace).

    The two decay levers are calibrated separately, as intended:

    - `beta` shapes destination choice. We fit it against the observed spatial pattern (Spearman rho of the modelled flow versus the observed flow) and take the best value.
    - `participation` scales trip generation. With all-mode origins, the share who make a *cycle* trip is the cycle mode share, which we read straight from the data (total cycle commuters over total commuters).
    """)
    return


@app.cell
def _(D, cn, np, observed, spearmanr, sub, zones_in):
    # independent inputs: all-mode commuter masses per MSOA
    orig_all = sub.groupby("msoa1")["all"].sum()
    dest_all = sub.groupby("msoa2")["all"].sum()
    cents = zones_in.copy()
    cents["geometry"] = cents.geometry.centroid
    origins_gdf = cents.assign(w=cents["geo_code"].map(orig_all).fillna(0.0))
    origins_gdf = origins_gdf[origins_gdf["w"] > 0][["w", "geometry"]]
    dests_gdf = cents.assign(w=cents["geo_code"].map(dest_all).fillna(0.0))
    dests_gdf = dests_gdf[dests_gdf["w"] > 0][["w", "geometry"]]

    live = observed.live.to_numpy()
    obs_v = observed[f"cc_betweenness_{D}"].to_numpy()

    def demand(beta, participation):
        cn.betweenness_demand(
            origins_gdf=origins_gdf,
            destinations_gdf=dests_gdf,
            origin_weight_col="w",
            destination_weight_col="w",
            distances=[D],
            decay_fn=f"exp(-{beta:.6f} * c)",
            participation=participation,
            tolerance=10.0,
            max_netw_assign_dist=1500.0,
        )
        modelled = cn.to_geopandas()
        rho = spearmanr(modelled[f"cc_demand_{D}"].to_numpy()[live], obs_v[live]).statistic
        return modelled, rho

    # calibrate beta against the observed pattern; participation from the cycle mode share
    betas = [0.0001, 0.0002, 0.0005, 0.001, 0.002]
    rhos = [demand(b, 1.0)[1] for b in betas]
    beta_star = betas[int(np.argmax(rhos))]
    participation = float(sub["bicycle"].sum()) / float(sub["all"].sum())
    modelled, rho = demand(beta_star, participation)
    print("beta sweep (rho vs observed):", {b: round(r, 3) for b, r in zip(betas, rhos, strict=True)})
    print(f"calibrated: beta={beta_star:g}, participation={participation:.3f}, rho={rho:.3f}")
    return beta_star, modelled, participation, rho


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Comparison

    The three panels share the network, the 20 km threshold, and the styling: a light-grey substrate for context, with more-travelled streets drawn darker and thicker on a percentile-scaled log ramp.
    """)
    return


@app.cell
def _(D, beta_star, modelled, np, observed, participation, plt, rho, std):
    def draw(ax, gdf, col, title):
        liv = gdf[gdf.live].copy()
        lv = np.log1p(liv[col].to_numpy())
        lo, hi = np.quantile(lv, 0.20), np.quantile(lv, 0.99)
        t = np.clip((lv - lo) / (hi - lo + 1e-9), 0, 1) ** 1.1
        ax.set_facecolor("white")
        liv.plot(ax=ax, color="#dedede", linewidth=0.2)
        order = np.argsort(t)
        liv.iloc[order].plot(
            ax=ax, color=plt.get_cmap("Reds")(0.10 + 0.90 * t[order]), linewidth=0.05 + 2.6 * t[order] ** 1.5
        )
        ax.set_title(title, color="black", fontsize=12, loc="left")
        ax.set_aspect("equal")
        ax.set_axis_off()

    bkey, dkey = f"cc_betweenness_{D}", f"cc_demand_{D}"
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), facecolor="white")
    draw(axes[0], std, bkey, "Standard betweenness (uniform)")
    draw(axes[1], observed, bkey, "Observed (PCT bicycle)")
    draw(axes[2], modelled, dkey, f"Modelled (calibrated, rho={rho:.2f})")
    fig.suptitle(
        f"Central London cycle flows, 20 km, beta={beta_star:g}, participation={participation:.2f}",
        color="black",
        y=0.04,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interpretation

    The observed and modelled panels concentrate flow on the same corridors, and the modelled flow tracks the observed pattern well above what topology alone gives (uniform betweenness correlates far more weakly with the observed flow). Most of the agreement comes from where commuters live and work, and from the network structure, rather than from fine tuning of the decay: over these central-London distances the fit is broad in `beta`. The `participation` value is the observed cycle mode share, so it sets the volume rather than the shape.

    Variations to try: swap the origins and destinations for population and employment layers from an independent source; lower `participation` further to see trip generation contract where access is poorer; or route the observed matrix at shorter thresholds to isolate local cycling catchments.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(r"""
    **Replicating the modelled flows in QGIS**

    The modelled step maps directly onto the QGIS plugin, so the same result can be produced without code.

    1. Load the street network and build the dual graph with the **Cityseer** network tools.
    2. Load the origin and destination points (here, MSOA centroids carrying the all-mode commuter masses).
    3. Run **Cityseer &rsaquo; Demand Betweenness (OD Flow)** with:
        - *Origins layer* and *Origin weight field* set to the residents mass.
        - *Destinations layer* and *Destination weight field* set to the workplace mass.
        - *Decay function* `exp(-0.0002 * c)` and *Participation* `0.09`, matching the calibrated values above.
        - *Distance thresholds* `20000`.

    The algorithm writes a demand-weighted betweenness column per threshold onto the network, the same quantity `betweenness_demand` returns here. The explicit-matrix path (`betweenness_od`) is library-only for now; in QGIS, model the flows with the demand algorithm.
    """),
        kind="info",
    )
    return


if __name__ == "__main__":
    app.run()
