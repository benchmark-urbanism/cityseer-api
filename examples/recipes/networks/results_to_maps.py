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
    # From Results to Maps

    This notebook bridges the gap between computing metrics and delivering them: it runs a quick centrality analysis on the bundled street network, exports the results to a GeoPackage ready for QGIS, and builds a publication-quality map figure directly in matplotlib.

    Analysis rarely ends at a `GeoDataFrame`. The two deliverables that most projects need are a GIS layer that collaborators can style and overlay, and a finished figure for a report or paper. Both start from the same exported results.

    The bundled datasets and their source attributions are documented on the [datasets page](https://cityseer.benchmarkurbanism.com/examples/datasets).
    """)
    return


@app.cell
def _():
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import numpy as np

    return gpd, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compute something to map

    We load the bundled street network and clip it to a 2km study area around the city centre. The clip extends a further 800m beyond the study area, matching the maximum analysis distance, so that metrics near the study boundary are not distorted by missing network; the `boundary` argument marks the nodes inside the study area as `live`. The high-level [`CityNetwork`](https://cityseer.benchmarkurbanism.com/api/network) class handles the dual-graph preparation, and its default `centrality_shortest` computes a single harmonic closeness and a single betweenness.
    """)
    return


@app.cell
def _(gpd, mo):
    from cityseer.network import CityNetwork

    data_dir = (mo.notebook_dir() / ".." / ".." / "data").resolve()
    streets_gpd = gpd.read_file(data_dir / "madrid_streets" / "street_network.gpkg")
    streets_gpd = streets_gpd.explode(ignore_index=True)
    # the source data contains a small number of exact duplicate geometries
    streets_gpd = streets_gpd.drop_duplicates(subset="geometry")
    # 2km study area around the city centre, with the network buffered a further
    # 800m (the maximum analysis distance) to prevent edge rolloff
    centre = gpd.GeoSeries.from_xy([440300], [4474300], crs=streets_gpd.crs)
    study_poly = centre.buffer(2000).iloc[0]
    buffered_poly = centre.buffer(2800).iloc[0]
    streets_clip = streets_gpd[streets_gpd.intersects(buffered_poly)]
    cn = CityNetwork.from_geopandas(streets_clip, boundary=study_poly)
    cn.centrality_shortest(distances=[800])
    results_gdf = cn.to_geopandas()
    print(f"{cn.node_count} street segments, {int(results_gdf.live.sum())} live")
    return (results_gdf,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Export to a GeoPackage

    `to_geopandas` returns the results joined back onto the original LineString geometries, so the export is a standard `geopandas` save. The file contains one row per street segment with the computed metric columns (`cc_harmonic_800` and `cc_betweenness_800`), the `live` flag marking segments inside the study area, any columns carried through from the input data, and the geometry in the projected CRS of the source network (EPSG:25830, UTM zone 30N). Because the CRS is embedded in the GeoPackage, GIS applications will position and measure the layer correctly without further configuration.
    """)
    return


@app.cell
def _(results_gdf):
    from pathlib import Path

    out_dir = Path("temp")
    out_dir.mkdir(exist_ok=True)
    results_gdf.to_file(out_dir / "results.gpkg", driver="GPKG")
    print(f"CRS: {results_gdf.crs}")
    print(sorted(c for c in results_gdf.columns if c.startswith("cc_")))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Styling in QGIS

    A typical QGIS workflow from here:

    1. Drag `results.gpkg` into QGIS, or use Layer, then Add Layer, then Add Vector Layer. The layer arrives in EPSG:25830 and aligns with any other projected data without reprojection.
    2. Open Layer Properties, then Symbology, and switch from Single Symbol to Graduated. Set Value to `cc_betweenness_800`, Mode to Equal Count (Quantile), and around 7 classes. Quantile classes suit centrality metrics because their distributions are heavily right-skewed; equal-interval classes would place almost every street in the lowest class. See the [interpretation guide](https://cityseer.benchmarkurbanism.com/guide/interpretation) for why.
    3. Vary line width with the value: in the symbol settings, use a data-defined override on stroke width (or the Size assistant) scaling from roughly 0.2mm for the lowest class to 1mm for the highest, so high-flow streets read at a glance even in greyscale print.
    4. Use a dark, low-saturation basemap or a plain dark canvas so that a bright sequential colour ramp (such as magma or viridis) carries the signal.
    5. Add attribution: as a minimum of good conscience, cite the tools used, for `cityseer` the [associated paper](https://journals.sagepub.com/doi/full/10.1177/23998083221133827), and openly link the [documentation website](https://cityseer.benchmarkurbanism.com); credit the underlying street data per the [datasets page](https://cityseer.benchmarkurbanism.com/examples/datasets).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A publication-quality figure in matplotlib

    The same principles apply when producing the figure in code: quantile class breaks, a single sequential colormap, line width following the value, and a colorbar labelled with the metric and its distance threshold. We filter to `live` segments so the buffer zone does not appear in the figure, and compute the breaks with `numpy` (`np.unique` guards against repeated break values in the zero-heavy tail of the distribution).
    """)
    return


@app.cell
def _(np, plt, results_gdf):
    # betweenness intensity: OrRd rank style, width and colour by percentile, no colour bar
    g = results_gdf[results_gdf.live].copy()
    g["_r"] = g["cc_betweenness_800"].rank(pct=True)
    g = g.sort_values("_r")  # strongest drawn last, on top
    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
    g.plot(ax=ax, color=plt.get_cmap("OrRd")(g["_r"]), linewidth=0.15 + 2.25 * g["_r"])
    ax.set_title("Streets carrying pedestrian through-movement in the study area", loc="left")
    ax.set_axis_off()
    fig.tight_layout()
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next steps

    - The [interpretation guide](https://cityseer.benchmarkurbanism.com/guide/interpretation) covers how to read centrality distributions and choose classification schemes.
    - The [datasets page](https://cityseer.benchmarkurbanism.com/examples/datasets) documents the bundled data, its sources, and licensing.
    """)
    return


if __name__ == "__main__":
    app.run()
