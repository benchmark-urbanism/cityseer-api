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
    # Elevation Effects on Centrality

    Compare centrality results with and without Z coordinates (elevation). When nodes have `z` attributes, cityseer automatically applies Tobler's hiking function during pathfinding: uphill segments become costlier and gentle downhill segments become slightly cheaper. This reshapes which paths are optimal and therefore changes centrality values.

    The flat/hilly comparison needs direct manipulation of the graph geometry, so each variant is prepared with the lower-level `tools` modules and then bridged into the high-level [`CityNetwork`](https://cityseer.benchmarkurbanism.com/api/network) class with `from_nx`, which builds the dual graph and computes the centralities.

    > 3D elevation support requires cityseer v4.24.0 or later.

    The bundled datasets and their source attributions are documented on the [datasets page](https://cityseer.benchmarkurbanism.com/examples/datasets).
    """)
    return


@app.cell
def _():
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import numpy as np
    import shapely
    from cityseer.network import CityNetwork
    from cityseer.tools import graphs, io
    from scipy.stats import spearmanr

    return CityNetwork, gpd, graphs, io, np, plt, shapely, spearmanr


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load the road network

    Load a hilly subset of the bundled road network with real 3D geometries, from mountainous terrain north of the city. The dataset is loaded twice: once preserving the Z coordinates (hilly) and once stripping them (flat).
    """)
    return


@app.cell
def _(gpd, graphs, io, shapely):
    import copy

    edges_raw = gpd.read_file("../../data/madrid_streets/street_network_3d.gpkg")
    edges_raw = edges_raw.explode(index_parts=False)
    print(f"Loaded {len(edges_raw)} edges, has_z={edges_raw.geometry.iloc[0].has_z}")
    # build the primal graph once (with Z), then simplify
    G_primal = io.nx_from_generic_geopandas(edges_raw)
    G_primal = graphs.nx_remove_filler_nodes(G_primal)
    G_primal = graphs.nx_remove_dangling_nodes(G_primal)
    # report elevation range
    _zs = [d["z"] for _, d in G_primal.nodes(data=True) if "z" in d]
    print(f"Elevation range: {min(_zs):.0f}m to {max(_zs):.0f}m (spread: {max(_zs) - min(_zs):.0f}m)")
    # flat variant: strip Z from a copy of the same primal graph
    G_flat_primal = copy.deepcopy(G_primal)
    for _node_idx, _node_data in G_flat_primal.nodes(data=True):
        _node_data.pop("z", None)
    for _u, _v, _edge_data in G_flat_primal.edges(data=True):
        if "geom" in _edge_data:
            _edge_data["geom"] = shapely.force_2d(_edge_data["geom"])
    return G_flat_primal, G_primal


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Elevation profile

    Visualise the terrain elevation across the network using contours interpolated from the node Z values.
    """)
    return


@app.cell
def _(G_primal, np, plt):
    from scipy.interpolate import griddata

    # extract node positions and elevations from the primal graph
    _xs = np.array([d["x"] for _, d in G_primal.nodes(data=True)])
    _ys = np.array([d["y"] for _, d in G_primal.nodes(data=True)])
    _zs = np.array([d["z"] for _, d in G_primal.nodes(data=True)])
    _xlim = (437400, 442200)
    _ylim = (4519850, 4527150)
    # create grid and interpolate
    _grid_x, _grid_y = np.mgrid[_xlim[0] : _xlim[1] : 200j, _ylim[0] : _ylim[1] : 200j]
    _grid_z = griddata((_xs, _ys), _zs, (_grid_x, _grid_y), method="cubic")
    _fig, _ax = plt.subplots(1, 1, figsize=(7, 7), dpi=150)
    contour_filled = _ax.contourf(_grid_x, _grid_y, _grid_z, levels=20, cmap="terrain", alpha=0.8)
    contour_lines = _ax.contour(_grid_x, _grid_y, _grid_z, levels=20, colors="k", linewidths=0.3, alpha=0.5)
    _ax.clabel(contour_lines, inline=True, fontsize=7, fmt="%.0f m")
    plt.colorbar(contour_filled, ax=_ax, label="Elevation (m)", shrink=0.8)
    _ax.set_xlim(_xlim)
    _ax.set_ylim(_ylim)
    _ax.set_aspect("equal")
    _ax.set_title("Terrain elevation (Sierra de Guadarrama foothills)")
    _ax.axis(False)
    plt.show()
    return (griddata,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compute centrality: flat vs hilly

    Run simplest-path (angular) closeness and betweenness at a 1000m threshold for both versions. Each primal graph variant is bridged into the high-level API with [`CityNetwork.from_nx`](https://cityseer.benchmarkurbanism.com/api/network#from_nx), which builds the dual graph required for angular centrality, and [`centrality_simplest`](https://cityseer.benchmarkurbanism.com/api/network#centrality_simplest) computes the centralities.
    """)
    return


@app.cell
def _(CityNetwork, G_flat_primal, G_primal):
    distances = [1000]

    # flat
    cn_flat = CityNetwork.from_nx(G_flat_primal)
    cn_flat.centrality_simplest(distances=distances)
    nodes_flat = cn_flat.to_geopandas()

    # hilly: the 3D geometries carry the elevation data through from_nx
    cn_hilly = CityNetwork.from_nx(G_primal)
    cn_hilly.centrality_simplest(distances=distances)
    nodes_hilly = cn_hilly.to_geopandas()
    return nodes_flat, nodes_hilly


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compare results

    Align nodes by position and visualise the flat vs hilly centrality results side by side, along with the difference.
    """)
    return


@app.cell
def _(G_primal, griddata, nodes_flat, nodes_hilly, np, plt):
    d = 1000
    harm_key = f"cc_harmonic_{d}_ang"
    betw_key = f"cc_betweenness_{d}_ang"
    diff_gdf = nodes_flat.copy()
    diff_gdf["harm_diff"] = nodes_hilly[harm_key].values - nodes_flat[harm_key].values
    diff_gdf["betw_diff"] = nodes_hilly[betw_key].values - nodes_flat[betw_key].values
    _xlim = (438200, 441600)
    _ylim = (4521000, 4526000)
    lw = 1.5
    _xs = np.array([nd["x"] for _, nd in G_primal.nodes(data=True)])
    _ys = np.array([nd["y"] for _, nd in G_primal.nodes(data=True)])
    _zs = np.array([nd["z"] for _, nd in G_primal.nodes(data=True)])
    _grid_x, _grid_y = np.mgrid[_xlim[0] : _xlim[1] : 200j, _ylim[0] : _ylim[1] : 200j]
    _grid_z = griddata((_xs, _ys), _zs, (_grid_x, _grid_y), method="cubic")
    _fig, axes = plt.subplots(3, 2, figsize=(14, 21), dpi=150, gridspec_kw={"hspace": 0.02, "wspace": 0.02})
    for _ax in axes.flat:
        _ax.contour(_grid_x, _grid_y, _grid_z, levels=15, colors="k", linewidths=0.3, alpha=0.3)
    nodes_flat.plot(
        ax=axes[0, 0],
        column=harm_key,
        cmap="magma",
        linewidth=lw,
        legend=True,
        legend_kwds={"label": f"Angular harmonic closeness, {d} m", "shrink": 0.6},
    )
    axes[0, 0].set_title(f"Harmonic closeness {d}m (flat)")
    axes[0, 0].set_xlim(_xlim)
    axes[0, 0].set_ylim(_ylim)
    axes[0, 0].set_aspect("equal")
    axes[0, 0].axis(False)
    nodes_flat.plot(
        ax=axes[0, 1],
        column=betw_key,
        cmap="magma",
        linewidth=lw,
        legend=True,
        legend_kwds={"label": f"Angular betweenness, {d} m", "shrink": 0.6},
    )
    axes[0, 1].set_title(f"Betweenness {d}m (flat)")
    axes[0, 1].set_xlim(_xlim)
    axes[0, 1].set_ylim(_ylim)
    axes[0, 1].set_aspect("equal")
    axes[0, 1].axis(False)
    nodes_hilly.plot(
        ax=axes[1, 0],
        column=harm_key,
        cmap="magma",
        linewidth=lw,
        legend=True,
        legend_kwds={"label": f"Angular harmonic closeness, {d} m", "shrink": 0.6},
    )
    axes[1, 0].set_title(f"Harmonic closeness {d}m (hilly)")
    axes[1, 0].set_xlim(_xlim)
    axes[1, 0].set_ylim(_ylim)
    axes[1, 0].set_aspect("equal")
    axes[1, 0].axis(False)
    nodes_hilly.plot(
        ax=axes[1, 1],
        column=betw_key,
        cmap="magma",
        linewidth=lw,
        legend=True,
        legend_kwds={"label": f"Angular betweenness, {d} m", "shrink": 0.6},
    )
    axes[1, 1].set_title(f"Betweenness {d}m (hilly)")
    axes[1, 1].set_xlim(_xlim)
    axes[1, 1].set_ylim(_ylim)
    axes[1, 1].set_aspect("equal")
    axes[1, 1].axis(False)
    vmax_h = np.abs(diff_gdf["harm_diff"]).max()
    diff_gdf.plot(
        ax=axes[2, 0],
        column="harm_diff",
        cmap="coolwarm",
        linewidth=lw,
        legend=True,
        legend_kwds={"label": "Closeness difference, hilly minus flat", "shrink": 0.6},
        vmin=-vmax_h,
        vmax=vmax_h,
    )
    axes[2, 0].set_title(f"Harmonic closeness {d}m difference (hilly minus flat)")
    axes[2, 0].set_xlim(_xlim)
    axes[2, 0].set_ylim(_ylim)
    axes[2, 0].set_aspect("equal")
    axes[2, 0].axis(False)
    vmax_b = np.abs(diff_gdf["betw_diff"]).max() / 4
    diff_gdf.plot(
        ax=axes[2, 1],
        column="betw_diff",
        cmap="coolwarm",
        linewidth=lw,
        legend=True,
        legend_kwds={"label": "Betweenness difference, hilly minus flat", "shrink": 0.6},
        vmin=-vmax_b,
        vmax=vmax_b,
    )
    axes[2, 1].set_title(f"Betweenness {d}m difference (hilly minus flat)")
    axes[2, 1].set_xlim(_xlim)
    axes[2, 1].set_ylim(_ylim)
    axes[2, 1].set_aspect("equal")
    axes[2, 1].axis(False)
    plt.show()
    return (d,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Rank correlation

    Compute Spearman rank correlations and mean percentage changes to quantify how much elevation affects the centrality rankings.
    """)
    return


@app.cell
def _(d, nodes_flat, nodes_hilly, np, spearmanr):
    print(f"{'Metric':<35} {'Spearman ρ':>10} {'Mean % change':>15}")
    print("-" * 65)

    for metric, label in [
        ("harmonic", "Angular harmonic closeness"),
        ("betweenness", "Angular betweenness"),
    ]:
        key = f"cc_{metric}_{d}_ang"
        flat_vals = nodes_flat[key].values
        hilly_vals = nodes_hilly[key].values

        mask = (flat_vals > 0) & (hilly_vals > 0)
        rho, _ = spearmanr(flat_vals[mask], hilly_vals[mask])
        pct_change = np.mean(np.abs(hilly_vals[mask] - flat_vals[mask]) / flat_vals[mask]) * 100

        print(f"{label} {d}m{'':<5} {rho:>10.4f} {pct_change:>14.1f}%")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    This notebook demonstrated how elevation (Z coordinates) affects centrality results. When 3D geometries are present, cityseer automatically applies Tobler's hiking function, which penalises uphill travel and slightly rewards gentle downhill slopes. The effect is asymmetric, since walking uphill costs more than walking downhill, and this shifts betweenness patterns in hilly terrain.

    **Next steps:**

    - To calculate angular centralities without elevation, see [Angular Centrality](https://cityseer.benchmarkurbanism.com/examples/centrality/gpd-angular-centrality).
    - To calculate metric (shortest-path) centralities, see [Metric Centrality](https://cityseer.benchmarkurbanism.com/examples/centrality/gpd-metric-centrality).
    - To compute centralities directly from OpenStreetMap data, see [OSM Centrality](https://cityseer.benchmarkurbanism.com/examples/centrality/osm-centrality).
    """)
    return


if __name__ == "__main__":
    app.run()
