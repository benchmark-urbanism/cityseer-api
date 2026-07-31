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
    # Adaptive Sampling for Large Networks

    Exact centrality runs a shortest-path traversal from every node, so cost grows with both network size and distance threshold. For large networks at long thresholds, `cityseer` offers an experimental adaptive sampling mode: pass `sample=True` and a subset of nodes is used as traversal sources, with results corrected by inverse-probability weighting so that estimates remain unbiased.

    The sampling is per-node adaptive. A cheap pilot polls the network with a small number of bounded shortest-path traversals to estimate each node's local reach, and the Hoeffding inequality converts that reach into the minimum sampling probability needed to keep the error within tolerance. Sparse areas are sampled more heavily and dense areas less, so precision is uniform across the network. For each distance threshold, a work test compares the cost of properly powered sampling against exact computation and selects the cheaper: short thresholds typically run exactly, long thresholds on large networks are sampled.

    This notebook demonstrates sampling on the full bundled street network through the high-level [`CityNetwork`](https://cityseer.benchmarkurbanism.com/api/network) class and validates the sampled results against an exact run. Sampling is experimental; the API may change. See the [guide](https://cityseer.benchmarkurbanism.com/guide/centrality#adaptive-sampling) and the [sampling reference](https://cityseer.benchmarkurbanism.com/metrics/sampling).

    The bundled datasets and their source attributions are documented on the [datasets page](https://cityseer.benchmarkurbanism.com/examples/datasets).
    """)
    return


@app.cell
def _():
    import time

    import geopandas as gpd
    import matplotlib.pyplot as plt
    from cityseer.network import CityNetwork
    from scipy.stats import spearmanr

    return CityNetwork, gpd, plt, spearmanr, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Network preparation

    We use the full bundled street network (province clip, roughly 20km around the city) and mark nodes inside the municipal boundary as live via the `boundary` argument. Live status serves two purposes here: results are only reported for live nodes, and for closeness the exact computation skips non-live nodes as sources, which keeps the exact baseline affordable. This is the standard buffered-network setup described in the [edge rolloff](https://cityseer.benchmarkurbanism.com/guide/fundamentals#edge-rolloff) section.
    """)
    return


@app.cell
def _(CityNetwork, gpd, mo):
    data_dir = (mo.notebook_dir() / ".." / ".." / "data").resolve()
    streets_gpd = gpd.read_file(data_dir / "madrid_streets" / "street_network.gpkg")
    streets_gpd = streets_gpd.explode(ignore_index=True)
    # the source data contains a small number of exact duplicate geometries
    streets_gpd = streets_gpd.drop_duplicates(subset="geometry")
    bounds_gpd = gpd.read_file(data_dir / "madrid_bounds" / "madrid_bounds.gpkg")
    boundary_poly = bounds_gpd.geometry.iloc[0]

    cn = CityNetwork.from_geopandas(streets_gpd, boundary=boundary_poly)
    print(f"{cn.node_count} street segments (dual nodes), {int(cn.nodes_gdf.live.sum())} live")
    return (cn,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sampled computation

    Enable sampling with `sample=True`. The log reports, per distance, whether the work test selected exact computation ("Full") or sampling, and in the sampled case the mean source inclusion probability `q`. We compute a single harmonic closeness at 1600m and 10000m; on this network the work test runs 1600m exactly and samples 10000m at a mean inclusion probability of roughly a third. `random_seed` makes the draw reproducible.
    """)
    return


@app.cell
def _(cn, time):
    t0 = time.time()
    cn.centrality_shortest(
        distances=[1600, 10000],
        closeness={"harmonic": "1/c"},
        betweenness={},
        sample=True,
        random_seed=42,
    )
    nodes_sampled = cn.to_geopandas()
    t_sampled = time.time() - t0
    print(f"sampled run: {t_sampled:.1f}s")
    return nodes_sampled, t_sampled


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Validation against an exact run

    To check the approximation quality we recompute the sampled 10000m threshold exactly and compare. Two views of agreement matter in practice:

    - **rank agreement** (Spearman rho): whether the sampled results order locations the same way, which is what most comparative analyses rely on;
    - **relative error**: the per-node deviation of the sampled estimate from the exact value.

    The default tolerance `epsilon=0.05` is calibrated so that rank agreement holds at Spearman rho of at least 0.95 on networks spanning the density range from metropolitan grids to sparse suburbs. Tighten `epsilon` for stricter agreement at higher cost, or loosen it for exploratory work where speed is the priority.
    """)
    return


@app.cell
def _(cn, nodes_sampled, time):
    t0_exact = time.time()
    cn.centrality_shortest(
        distances=[10000],
        closeness={"harmonic": "1/c"},
        betweenness={},
    )
    # to_geopandas snapshots are independent, so nodes_sampled keeps the sampled values
    nodes_exact = cn.to_geopandas()
    assert nodes_exact.index.equals(nodes_sampled.index)
    t_exact = time.time() - t0_exact
    print(f"exact run (10000m): {t_exact:.1f}s")
    return nodes_exact, t_exact


@app.cell
def _(nodes_exact, nodes_sampled, spearmanr, t_exact, t_sampled):
    live_mask = nodes_exact.live
    exact_vals = nodes_exact.loc[live_mask, "cc_harmonic_10000"]
    sampled_vals = nodes_sampled.loc[live_mask, "cc_harmonic_10000"]
    rho = spearmanr(exact_vals, sampled_vals).statistic
    rel_err = ((sampled_vals - exact_vals).abs() / exact_vals).median()
    print(f"Spearman rho (exact vs sampled, 10000m, live nodes): {rho:.4f}")
    print(f"median relative error: {rel_err:.2%}")
    print(f"exact {t_exact:.1f}s vs sampled {t_sampled:.1f}s (sampled run also covers 1600m)")
    return


@app.cell
def _(nodes_sampled, plt):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), dpi=150)
    for ax, col in zip(axes, ["cc_harmonic_1600", "cc_harmonic_10000"], strict=False):
        _dist = col.split("_")[-1]
        _label = f"Harmonic closeness, {_dist} m"
        g = nodes_sampled[nodes_sampled.live].copy()
        g["_r"] = g[col].rank(pct=True)
        g = g.sort_values("_r")  # strongest drawn last
        g.plot(ax=ax, color=plt.get_cmap("OrRd")(g["_r"]), linewidth=0.15 + 2.25 * g["_r"])
        ax.set_title(f"{_label} (sampled run)", loc="left")
        ax.set_axis_off()
    fig.tight_layout()
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Betweenness engages sampling sooner

    The thresholds at which sampling engages depend on what is being computed. Closeness in exact mode only sources from live nodes, so on this network the work test does not find sampling worthwhile until roughly 10000m. Betweenness must source from every node, including the dead buffer nodes, so its exact cost rises much more steeply: with a betweenness expression in the run, the planner already samples the 5000m threshold on this network (at a mean inclusion probability of roughly three quarters). The call is otherwise identical:

    ```python
    cn.centrality_shortest(
        distances=[1600, 5000],
        sample=True,
        random_seed=42,
    )
    ```

    ## Practical notes

    - Sampling is opt-in (`sample=True`) and applies per distance threshold; you can mix short exact thresholds and long sampled thresholds in one call.
    - `epsilon` trades cost against precision; `random_seed` gives reproducible draws.
    - The work test means `sample=True` is safe to leave on: where sampling would not help, the exact path is used automatically.
    - The same parameters are available on [`centrality_simplest`](https://cityseer.benchmarkurbanism.com/api/network#centrality-simplest) and on the lower-level functional `networks` module.
    - Sampling is experimental: behaviour and defaults may change between releases.
    """)
    return


if __name__ == "__main__":
    app.run()
