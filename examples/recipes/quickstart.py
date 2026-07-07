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
    # Quickstart

    This page bridges the Python fundamentals from the [Python 101](https://cityseer.benchmarkurbanism.com/start/1-notebooks) series with the practical [Cityseer Recipes](https://cityseer.benchmarkurbanism.com/examples). It introduces the core concepts and walks through a minimal end-to-end workflow.

    > **Tip:**
    > If you are already familiar with `cityseer` and want to jump into specific analyses, head directly to the [Recipes](https://cityseer.benchmarkurbanism.com/examples).

    ## Core Concepts

    ### The cityseer workflow

    The recommended entry point is the high-level [`CityNetwork`](https://cityseer.benchmarkurbanism.com/api/network) class, which wraps network preparation and metrics behind a lean interface:

    1. **Build a network** with a constructor such as `CityNetwork.from_osm` (download from OpenStreetMap), `from_geopandas` (a `GeoDataFrame` of LineStrings), or `from_nx` (a `networkx` graph). Graph cleaning and dual-graph construction are handled automatically.
    2. **Compute metrics** — centrality, accessibility, mixed uses, or statistics. Methods return `self`, so calls can be chained.
    3. **Export results** — `to_geopandas()` returns a `GeoDataFrame` with the original street geometries and all computed columns, ready for plotting or saving.

    The lower-level API (`cityseer.tools`, `cityseer.metrics`) offers step-by-step control over graph cleaning, network construction, and metric computation. It is covered by the [Network Preparation](https://cityseer.benchmarkurbanism.com/examples/networks) recipes and is useful when integrating `cityseer` into an existing `networkx` pipeline.

    ### Primal vs. dual graphs

    By default, a street network is a **primal graph**: intersections are nodes and streets are edges. This is the representation provided by most data sources.

    `cityseer` can convert a primal graph into a **dual graph**, where the representation is inverted: streets become nodes and intersections become edges. Metrics are then expressed per street segment rather than per intersection, which is usually more intuitive for urban analysis because outputs can be visualised on street geometries. It also corresponds naturally with simplest-path (angular) centralities, which track turning angles from street to street.

    ```
    Primal graph                    Dual graph
    (intersections = nodes)         (streets = nodes)

        A ——— B                        AB
        |     |          →            /  \
        C ——— D                     AC    BD
                                      \  /
                                       CD
    ```

    > **Important:**
    > **Angular centrality measures require the dual graph.** `CityNetwork` builds the dual graph automatically. When using the lower-level API, convert a primal graph with [`graphs.nx_to_dual`](https://cityseer.benchmarkurbanism.com/tools/graphs#nx-to-dual) before computing angular centralities; see the [dual graph recipe](https://cityseer.benchmarkurbanism.com/examples/networks/create-dual-graph).

    ### Where results are stored

    `CityNetwork` exposes a `nodes_gdf` property: one row per street segment (dual node), with computed metrics added as columns. For mapping and export, `to_geopandas()` returns the same columns joined to the original LineString geometries.

    In the lower-level API, [`network_structure_from_nx`](https://cityseer.benchmarkurbanism.com/tools/io#network-structure-from-nx) converts a `networkx` graph into a nodes `GeoDataFrame`, an edges `GeoDataFrame`, and a `NetworkStructure` — the internal Rust data structure used by the metric functions. The `NetworkStructure` can be reused across multiple metric computations as long as the network does not change.

    ### Distance thresholds

    Most `cityseer` functions accept a `distances` parameter specifying network distance thresholds (in metres). Multiple thresholds can be computed at once. See the [distance thresholds table](https://cityseer.benchmarkurbanism.com/guide/fundamentals#distance-thresholds) for common choices and their real-world meaning.

    Alternatively, use the `minutes` parameter to specify walking-time thresholds, converted internally to metres at a default walking speed of 1.33m/s (about 80m per minute).

    ### Reading output columns

    `cityseer` adds result columns to the nodes `GeoDataFrame` using a naming convention:

    | Pattern                               | Example                         | Meaning                                                          |
    | ------------------------------------- | ------------------------------- | ---------------------------------------------------------------- |
    | `cc_{metric}_{distance}`              | `cc_betweenness_800`            | Metric centrality: betweenness at 800m                           |
    | `cc_{metric}_{distance}_ang`          | `cc_harmonic_500_ang`           | Angular centrality: harmonic closeness at 500m                   |
    | `cc_{landuse}_{distance}`             | `cc_restaurant_400`             | Accessibility: count of restaurants within 400m                  |
    | `cc_{landuse}_nearest_max_{distance}` | `cc_restaurant_nearest_max_800` | Accessibility: distance to nearest restaurant (max search 800m)  |
    | `cc_{column}_{stat}_{distance}`       | `cc_height_mean_400`            | Statistics: mean of `height` column within 400m                  |

    The `cc_` prefix identifies columns computed by `cityseer`. For centrality, the `_ang` suffix indicates angular (simplest-path) analysis; its absence indicates metric (shortest-path) analysis. For accessibility and statistics, `CityNetwork` produces plain (unweighted) aggregations by default; pass a `decay_fn` expression for distance-weighted variants, in which case dict labels are embedded in the column names. The lower-level `layers` functions, when `decay_fn` is omitted, retain a legacy default that emits both an unweighted (`_nw`) and a decay-weighted (`_wt`) column. By default, `CityNetwork` centrality emits harmonic closeness (`cc_harmonic_{d}`) and betweenness (`cc_betweenness_{d}`); further metrics can be requested via expression dictionaries, as described in the [guide](https://cityseer.benchmarkurbanism.com/guide/centrality#centrality).

    ## A minimal example

    Here is a complete `cityseer` workflow in a few lines. It downloads a street network, computes centrality, and plots betweenness.

    Network data © [OpenStreetMap](https://www.openstreetmap.org/copyright) contributors, available under the Open Database Licence.
    """)
    return


@app.cell
def _():
    import os

    os.environ["CITYSEER_QUIET_MODE"] = "1"
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    from cityseer.network import CityNetwork
    from cityseer.tools import io

    # 1. Define area of interest (a 1000m buffer around a central point)
    poly_wgs, _ = io.buffered_point_poly(-3.7038, 40.4168, 1000)

    # 2. Build the network: downloads from OSM, cleans, and builds the dual graph
    cn = CityNetwork.from_osm(poly_wgs)

    # 3. Compute shortest-path centrality at 500m, 1000m, and 2000m
    cn.centrality_shortest(distances=[500, 1000, 2000])

    # 4. Export with street geometries and plot betweenness at 1000m
    edges_gdf = cn.to_geopandas()
    # betweenness is heavily skewed: map the percentile rank so the pattern is legible
    edges_gdf["btw_rank"] = edges_gdf["cc_betweenness_1000"].rank(pct=True)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=150)
    edges_gdf.plot(
        ax=ax,
        column="btw_rank",
        cmap="magma",
        linewidth=1,
        legend=True,
        legend_kwds={"label": "Betweenness, 1000 m (percentile)", "shrink": 0.6},
    )
    ax.set_title("Betweenness, 1000 m")
    ax.set_axis_off()
    fig.tight_layout()
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **Note:**
    > Building the network produces log messages showing the cleaning steps. These are normal and can be safely ignored. To suppress them, add `import logging; logging.getLogger("cityseer").setLevel(logging.WARNING)` beforehand.

    ## Choosing a coordinate reference system

    `cityseer` requires a **projected** coordinate reference system (CRS) with metre units for accurate distance calculations. A geographic CRS like WGS 84 (EPSG:4326) uses degrees and will produce incorrect results.

    How to choose the right CRS for your study area:

    - **Let cityseer choose for you:** When using `CityNetwork.from_osm`, `io.buffered_point_poly`, or `io.osm_graph_from_poly` with default settings, `cityseer` automatically selects an appropriate UTM zone based on the input coordinates. This is usually sufficient.
    - **Specify explicitly:** If you need a specific CRS, use the `to_crs_code` parameter. Common choices include UTM zones (search your city at [epsg.io](https://epsg.io)) or regional projections like EPSG:3035 (Europe), EPSG:27700 (Great Britain), or EPSG:2154 (France).
    - **From existing data:** If loading a network from a file, ensure it is already in a projected CRS, or reproject it with `gdf.to_crs(epsg=...)` before passing it to `cityseer`.

    ## Common pitfalls

    If something looks wrong, distances off by orders of magnitude, values fading at the study area edge, missing data points, and so on, see the [Troubleshooting guide](https://cityseer.benchmarkurbanism.com/guide/troubleshooting). It covers the common problems symptom by symptom, with fixes.

    ## Next steps

    With these concepts in hand, you are ready to explore the recipes:

    - [Network Preparation](https://cityseer.benchmarkurbanism.com/examples/networks) — create networks from various data sources.
    - [Network Centrality](https://cityseer.benchmarkurbanism.com/examples/centrality) — metric and angular centrality.
    - [Accessibility](https://cityseer.benchmarkurbanism.com/examples/accessibility) — landuse accessibility and mixed-use metrics.
    - [Statistics](https://cityseer.benchmarkurbanism.com/examples/stats) — aggregate numeric properties over the network.
    - [Visibility](https://cityseer.benchmarkurbanism.com/examples/visibility) — street enclosure and openness.
    - [Continuity](https://cityseer.benchmarkurbanism.com/examples/continuity) — street name and route continuity.
    """)
    return


if __name__ == "__main__":
    app.run()
