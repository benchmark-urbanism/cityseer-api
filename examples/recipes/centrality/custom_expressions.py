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
    # Custom Centrality Expressions and Statistic Selection

    From v5, `cityseer` centrality is expression-based: instead of a fixed metric set, the `centrality_shortest` and `centrality_simplest` methods accept `{name: expression}` dictionaries, so you define exactly which metrics to compute. The same principle extends to the land-use and statistics methods through the `decay_fn` and `measures` parameters.

    This notebook works through the high-level [`CityNetwork`](https://cityseer.benchmarkurbanism.com/api/network) class, which forwards these parameters to the underlying functions unchanged, and covers:

    - the expression model (`c` and `p` variables) and the default metrics;
    - selecting only the metrics you need;
    - defining custom metrics such as gravity-weighted closeness;
    - derived metrics via `postprocess`;
    - selecting statistical measures with `compute_stats(measures=[...])` and applying distance decay to statistics.

    For the underlying concepts see the [guide](https://cityseer.benchmarkurbanism.com/guide/centrality#centrality).

    The bundled datasets and their source attributions are documented on the [datasets page](https://cityseer.benchmarkurbanism.com/examples/datasets).
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
    ## Network preparation

    We load the bundled street network from the bundled datasets and clip it to a 2km study area so that the examples run quickly. `CityNetwork` builds the dual representation automatically, so metrics are expressed relative to street segments. See the [network preparation recipes](https://cityseer.benchmarkurbanism.com/examples/networks) for the lower-level construction steps.
    """)
    return


@app.cell
def _(CityNetwork, gpd, mo):
    data_dir = (mo.notebook_dir() / ".." / ".." / "data").resolve()
    streets_gpd = gpd.read_file(data_dir / "madrid_streets" / "street_network.gpkg")
    streets_gpd = streets_gpd.explode(ignore_index=True)
    # the source data contains a small number of exact duplicate geometries
    streets_gpd = streets_gpd.drop_duplicates(subset="geometry")
    # clip to a 2km buffer around a central point (city centre) for fast examples
    study_area = gpd.GeoSeries.from_xy([440300], [4474300], crs=streets_gpd.crs).buffer(2000).iloc[0]
    streets_clip = streets_gpd[streets_gpd.intersects(study_area)]
    cn = CityNetwork.from_geopandas(streets_clip)
    print(f"{cn.node_count} street segments (dual nodes)")
    return cn, data_dir


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The expression model

    Metric expressions use two variables:

    - `c`: the raw network cost to a reached node - metres for shortest-path analysis, degrees of cumulative turning for simplest-path (angular) analysis;
    - `p`: normalised progress from `0` at the source to `1` at the distance threshold (`p = c / threshold`).

    Expressions may use the functions `exp`, `ln`, `log10`, `sqrt`, `abs`, `floor`, `ceil`, `round`, `sin`, `cos`, `tan`, and the constants `pi` and `e`.

    The two metric categories aggregate differently:

    - **closeness** expressions are evaluated once per reached node and summed: `{"harmonic": "1/c"}` accumulates $\sum_j 1/c_j$ over all nodes $j$ within the threshold;
    - **betweenness** expressions weight each origin-destination pair's contribution to the nodes on the shortest path between them: `{"betweenness": "1"}` counts paths equally.

    Each entry produces a column `cc_{name}_{distance}`. The `CityNetwork` methods default to a lean set, a single harmonic closeness and a single betweenness, while passing `{}` skips a category entirely. Explicitly passing `None` opts into the fuller default set of the underlying functional `networks.centrality_shortest`:

    | Category | Default | Expression |
    |---|---|---|
    | closeness | `density` | `"1"` |
    | closeness | `farness` | `"c"` |
    | closeness | `harmonic` | `"1/c"` |
    | closeness | `decay` | `"exp(-4 * p)"` |
    | betweenness | `betweenness` | `"1"` |
    | betweenness | `betweenness_decay` | `"exp(-4 * p)"` |
    """)
    return


@app.cell
def _(cn):
    _before = set(cn.nodes_gdf.columns)
    cn.centrality_shortest(distances=[800])
    lean_cols = sorted(c for c in set(cn.nodes_gdf.columns) - _before)
    print(lean_cols)
    return (lean_cols,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Selecting only what you need

    Every expression is evaluated during the network traversal, so a smaller metric set runs faster and emits fewer columns. Here we compute a single harmonic closeness across two distances and skip betweenness entirely with `{}` (cycles are already off by default on `CityNetwork`).
    """)
    return


@app.cell
def _(cn, lean_cols):
    _before = set(cn.nodes_gdf.columns)
    cn.centrality_shortest(
        distances=[400, 800],
        closeness={"harmonic": "1/c"},
        betweenness={},
    )
    print(sorted(set(cn.nodes_gdf.columns) - _before), f"(in addition to the existing {lean_cols})")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Custom metrics

    Any expression over `c` and `p` defines a new metric. Three common patterns:

    - **Gravity / spatial interaction**: `exp(-beta * c)` with an explicit beta in metres. `"exp(-0.005 * c)"` corresponds to a beta of 0.005, i.e. roughly an 800m walking tolerance. This replaces the removed `betas=` parameter from earlier versions.
    - **Threshold-relative decay**: `exp(-4 * p)` expresses decay in terms of the threshold rather than absolute metres, so the same expression adapts across distances.
    - **Cumulative reach**: `"1"` simply counts reachable nodes (density).

    The same dictionaries apply to `centrality_simplest`, where `c` is angular cost, and to the betweenness side, where the expression weights each origin-destination pair.
    """)
    return


@app.cell
def _(cn):
    _before = set(cn.nodes_gdf.columns)
    cn.centrality_shortest(
        distances=[800],
        closeness={
            "harmonic": "1/c",
            "gravity": "exp(-0.005 * c)",
        },
        betweenness={
            "betweenness": "1",
            "betweenness_near": "exp(-8 * p)",  # emphasise short-range through-movement
        },
    )
    nodes_custom = cn.to_geopandas()
    print(sorted(set(nodes_custom.columns) - _before))
    return (nodes_custom,)


@app.cell
def _(nodes_custom, plt):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), dpi=150)
    for ax, col, label in zip(
        axes,
        ["cc_harmonic_800", "cc_gravity_800"],
        ["Harmonic closeness, 800 m", "Gravity-weighted closeness, 800 m"],
        strict=False,
    ):
        g = nodes_custom[nodes_custom.live].copy()
        g["_r"] = g[col].rank(pct=True)
        g = g.sort_values("_r")  # strongest drawn last
        g.plot(ax=ax, color=plt.get_cmap("OrRd")(g["_r"]), linewidth=0.15 + 2.25 * g["_r"])
        ax.set_title(label, loc="left")
        ax.set_axis_off()
    fig.tight_layout()
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Derived metrics with `postprocess`

    The `postprocess` parameter derives further columns from computed results using simple arithmetic over the metric names. The default computes Hillier-style normalised integration, `density**2 / farness`, which requires that `density` and `farness` are present in the closeness set. Derived metrics are computed per distance threshold after the traversal, so they add no traversal cost.
    """)
    return


@app.cell
def _(cn):
    _before = set(cn.nodes_gdf.columns)
    cn.centrality_shortest(
        distances=[800],
        closeness={"density": "1", "farness": "c"},
        betweenness={},
        postprocess={"hillier": "density**2 / farness"},
    )
    print(sorted(set(cn.nodes_gdf.columns) - _before))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Selecting statistics: `compute_stats` with `measures`

    The statistics function follows the same select-what-you-need principle. By default, `compute_stats` emits `sum`, `mean`, `count`, `var`, `median`, `mad`, `max`, and `min` for each column and distance. The `measures` parameter restricts this to a subset, which keeps the output frame small and skips the weighted median/MAD sort when neither is requested.

    Distance weighting is controlled by `decay_fn`, an expression over `p` (clamped to `[0, 1]`). Passing a `{label: expression}` dictionary computes several decay variants in a single traversal, with each label suffixed into its column names. The [`cityseer.decay`](https://cityseer.benchmarkurbanism.com/api/decay) module provides helpers that return these expression strings: `decay.flat()` returns `"1"`, `decay.exponential()` returns `"exp(-4.0 * p)"`, and `decay.gaussian(peak=400, cutoff=1600)` returns a Gaussian centred at 400m.
    """)
    return


@app.cell
def _(cn, data_dir, gpd):
    from cityseer import decay

    bldgs_gpd = gpd.read_file(data_dir / "madrid_buildings" / "madrid_bldgs.gpkg")
    _before = set(cn.nodes_gdf.columns)
    _cn, bldgs_assigned = cn.compute_stats(
        bldgs_gpd,
        stats_column_labels=["mean_height"],
        distances=[400, 800],
        measures=["mean", "count"],
        decay_fn={"plain": decay.flat(), "wt": decay.exponential()},
    )
    nodes_stats = cn.to_geopandas()
    print(sorted(set(nodes_stats.columns) - _before))
    return (nodes_stats,)


@app.cell
def _(nodes_stats, plt):
    stat_cols = sorted(c for c in nodes_stats.columns if "mean_height" in c and "_mean_" in c)
    fig_s, axes_s = plt.subplots(1, len(stat_cols), figsize=(6.5 * len(stat_cols), 6.5), dpi=150)
    for ax_s, col_s in zip(axes_s, stat_cols, strict=False):
        # column names follow cc_mean_height_mean_{decay label}_{distance}
        _decay_label, _dist = col_s.split("_")[-2:]
        _title = f"Mean building height, {_dist} m ({'distance-weighted' if _decay_label == 'wt' else 'unweighted'})"
        _g = nodes_stats[nodes_stats.live].copy()
        _g["_r"] = _g[col_s].rank(pct=True)
        _g = _g.sort_values("_r")  # strongest drawn last
        _g.plot(ax=ax_s, color=plt.get_cmap("OrRd")(_g["_r"]), linewidth=0.15 + 2.25 * _g["_r"])
        ax_s.set_title(_title, loc="left")
        ax_s.set_axis_off()
    fig_s.tight_layout()
    fig_s
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Where else expressions apply

    - `compute_accessibilities` and `compute_mixed_uses` accept the same `decay_fn` parameter, including the `{label: expression}` dict form, for distance-weighted variants.
    - `betweenness_demand` uses a `decay_fn` expression as the deterrence function of a spatial-interaction model for origin-destination weighted flows.
    - The lower-level functional `networks` and `layers` modules accept the same dictionaries directly; the [`CityNetwork`](https://cityseer.benchmarkurbanism.com/api/network) methods used here forward them unchanged, differing only in their leaner defaults.

    See the [guide](https://cityseer.benchmarkurbanism.com/guide/land-use#decay-functions) for the full expression syntax and decay reference.
    """)
    return


if __name__ == "__main__":
    app.run()
