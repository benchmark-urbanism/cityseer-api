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
    # Street Continuity from OS Open Roads

    Compute street continuity metrics from the Ordnance Survey OS Open Roads dataset. Continuity measures how far streets extend as continuous entities based on shared names, route numbers, or road classifications.

    Continuity analysis operates directly on the `networkx` graph through the lower-level functional API; it is not wrapped by the higher-level [`CityNetwork`](https://cityseer.benchmarkurbanism.com/guide/fundamentals#citynetwork-api) class.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data Source
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The following example uses the [OS Open Roads](https://osdatahub.os.uk/downloads/open/OpenRoads) dataset, which is available under the [Open Government License](http://os.uk/opendata/licence).

    _© Crown copyright and database right 2022_
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Preparation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The dataset is not bundled with the repository. This example assumes the OS Open Roads GeoPackage download has been unzipped into `temp/` at the root of a cloned `cityseer-api` repository, giving `temp/oproad_gpkg_gb/Data/oproad_gb.gpkg` (the layout of the official download). Edit the path below if your data lives elsewhere.
    """)
    return


@app.cell
def _(mo):
    from pathlib import Path

    open_roads_path = Path("../../../temp/oproad_gpkg_gb/Data/oproad_gb.gpkg")
    print("data path:", open_roads_path.resolve())
    print("path exists:", open_roads_path.exists())
    # halt gracefully when the manually downloaded dataset is not present
    mo.stop(
        not open_roads_path.exists(),
        "OS Open Roads dataset not found; download it to run the rest of this notebook.",
    )
    return (open_roads_path,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Extents
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Instead of loading the entire dataset, we'll use a bounding box to only load an area of interest.
    """)
    return


@app.cell
def _():
    from cityseer.tools import io
    from pyproj import Transformer
    from shapely import geometry

    # create graph - only UK locations will work for OS Open Roads data
    # stratford-upon-avon
    # lng, lat, buffer_dist, plot_buffer = -1.7063649924889566, 52.19277374082795, 2500, 2000
    # london
    lng, lat, buffer_dist, plot_buffer = (
        -0.13039709427587876,
        51.516434828344366,
        6000,
        5000,
    )
    # transform from WGS to BNG
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700")
    easting, northing = transformer.transform(lat, lng)
    # calculate bbox relative to centroid
    centroid = geometry.Point(easting, northing)
    target_bbox: tuple[float, float, float, float] = centroid.buffer(buffer_dist).bounds  # type: ignore
    plot_bbox: tuple[float, float, float, float] = centroid.buffer(plot_buffer).bounds  # type: ignore
    return io, plot_bbox, target_bbox


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Load
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can now load the OS Open Roads dataset
    """)
    return


@app.cell
def _(io, open_roads_path, target_bbox: tuple[float, float, float, float]):
    # load OS Open Roads data from downloaded geopackage
    G_open = io.nx_from_open_roads(open_roads_path, target_bbox=target_bbox)
    return (G_open,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Observe continuity metrics
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This step runs the continuity analysis using the specified heuristic.
    """)
    return


@app.cell
def _(G_open):
    import matplotlib.pyplot as plt
    from cityseer.metrics import observe
    from cityseer.tools import plot

    print("Continuity by street names")
    G_cont, NamesContReport = observe.street_continuity(G_open, method="names")
    # methods can be "names", "routes", "highways"
    NamesContReport.report_by_count(n_items=5)
    NamesContReport.report_by_length(n_items=5)
    print("Continuity by route numbers")
    G_cont, RoutesContReport = observe.street_continuity(G_cont, method="routes")
    RoutesContReport.report_by_count(n_items=5)
    RoutesContReport.report_by_length(n_items=5)
    print("Continuity by highway types")
    G_cont, HwyContReport = observe.street_continuity(G_cont, method="highways")
    HwyContReport.report_by_count(n_items=5)
    HwyContReport.report_by_length(n_items=5)
    print("Continuity by overlapping routes and names types")
    G_cont, HybridContReport = observe.hybrid_street_continuity(G_cont)
    HybridContReport.report_by_count(n_items=5)
    HybridContReport.report_by_length(n_items=5)
    return G_cont, plot, plt


@app.cell
def _(G_cont, plot, plot_bbox: tuple[float, float, float, float], plt):
    for method, shape_exp, descriptor, cmap, inverse, col_by_categ in zip(
        ["names", "routes", "highways", "hybrid"],  #
        [1, 0.75, 0.5, 1],  #
        ["Street names", "Routes", "Road types", "Hybrid routes & names"],  #
        ["plasma", "viridis", "tab10", "tab10"],  #
        [False, False, True, False],  #
        [False, False, True, True],
        strict=False,
    ):
        print(f"Plotting results for method: {method}")
        # plot
        bg_colour = "#1d1d1d"
        fig, axes = plt.subplots(2, 1, dpi=150, figsize=(8, 12), facecolor=bg_colour, constrained_layout=True)
        fig.suptitle(
            f"OS Open Roads plotted by {descriptor} continuity",
            fontsize="small",
            ha="center",
        )
        # by count
        plot.plot_nx_edges(
            axes[0],  # type: ignore
            nx_multigraph=G_cont,
            edge_metrics_key=f"{method}_cont_by_count",
            bbox_extents=plot_bbox,
            cmap_key=cmap,
            lw_min=0.5,
            lw_max=2,
            edge_label_key=f"{method}_cont_by_label",
            colour_by_categorical=col_by_categ,
            shape_exp=shape_exp,
            face_colour=bg_colour,
            invert_plot_order=inverse,
        )
        axes[0].set_xlabel(f"{descriptor} by count", fontsize="x-small")  # type: ignore
        # by length
        plot.plot_nx_edges(
            axes[1],  # type: ignore
            nx_multigraph=G_cont,
            edge_metrics_key=f"{method}_cont_by_length",
            bbox_extents=plot_bbox,
            cmap_key=cmap,
            lw_min=0.5,
            lw_max=2,
            edge_label_key=f"{method}_cont_by_label",
            colour_by_categorical=col_by_categ,
            shape_exp=shape_exp,
            face_colour=bg_colour,
            invert_plot_order=inverse,
        )
        axes[1].set_xlabel(f"{descriptor} by length (metres)", fontsize="x-small")  # type: ignore
        plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusion

    This notebook demonstrated how to compute street continuity metrics from the OS Open Roads dataset for London, applying the same name, route, highway, and hybrid continuity heuristics as the OSM example but using an authoritative UK national dataset. The OS Open Roads data provides well-structured road classification attributes, producing clean continuity results that highlight the extent of major arterial routes like the A1 and A41.

    **Next steps:** For other analysis types, return to the [Recipes overview](https://cityseer.benchmarkurbanism.com/examples).
    """)
    return


if __name__ == "__main__":
    app.run()
