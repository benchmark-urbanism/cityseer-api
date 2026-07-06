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
    # Street Continuity from OSM Data

    Compute street continuity metrics from OpenStreetMap data. Continuity measures how far streets extend as continuous entities based on shared names, route numbers, or highway classifications.

    Continuity analysis operates directly on the `networkx` graph through the lower-level functional API; it is not wrapped by the higher-level [`CityNetwork`](https://cityseer.benchmarkurbanism.com/guide/fundamentals#citynetwork-api) class.

    Network data © [OpenStreetMap](https://www.openstreetmap.org/copyright) contributors, available under the Open Database Licence.
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
    Set the coordinates and buffer distances to your preferred extents.
    """)
    return


@app.cell
def _():
    from cityseer.tools import io

    # lng, lat, buffer_dist, plot_buffer = -1.7063649924889566, 52.19277374082795, 1500, 1250  # stratford-upon-avon
    lng, lat, buffer_dist, plot_buffer = (
        -0.13039709427587876,
        51.516434828344366,
        6000,
        5000,
    )  # london
    # lng, lat, buffer_dist, plot_buffer = 18.425702641104582, -33.9204746754594, 3000, 2500  # cape town
    poly_wgs, _ = io.buffered_point_poly(lng, lat, buffer_dist)
    poly_utm, _ = io.buffered_point_poly(lng, lat, buffer_dist, projected=True)
    # select extents for plotting
    plot_bbox = poly_utm.centroid.buffer(plot_buffer).bounds
    return io, plot_bbox, poly_wgs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loading data from OSM
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This example uses a custom OSM query that excludes footways, because the continuity analysis needs OSM ways carrying street name or route number information.

    Only basic cleaning is recommended for this form of analysis. Avoid consolidating nodes, since consolidation risks dropping `highway`, `ref`, or `name` attributes. Unlike closeness or betweenness measures, continuity is not particularly sensitive to topological distortions, so aggressive simplification brings little benefit.
    """)
    return


@app.cell
def _(io, poly_wgs):
    query = """
    [out:json];
    (
        way["highway"]
        ["area"!="yes"]
        ["highway"!~"footway|pedestrian|steps|bus_guideway|escape|raceway|proposed|planned|abandoned|platform|construction"]
        ["service"!~"parking_aisle"]
        ["amenity"!~"charging_station|parking|fuel|motorcycle_parking|parking_entrance|parking_space"]
        ["access"!~"private|customers"]
        ["indoor"!="yes"]
        (poly:"{geom_osm}");
    );
    out body;
    >;
    out qt;
    """
    G_osm = io.osm_graph_from_poly(poly_wgs, custom_request=query, simplify=True)
    return (G_osm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Observe continuity metrics

    We can now run the continuity metrics.
    """)
    return


@app.cell
def _(G_osm):
    import matplotlib.pyplot as plt
    from cityseer.metrics import observe
    from cityseer.tools import plot

    print("Continuity by street names")
    G_cont, NamesContReport = observe.street_continuity(G_osm, method="names")
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
def _(G_cont, plot, plot_bbox, plt):
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
            f"OSM network plotted by {descriptor} continuity",
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
    Longer continuous routes (brighter colours) indicate streets that maintain their name or classification over greater distances, suggesting major through-routes. Fragmented streets (darker) are typically local or residential.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusion

    This notebook demonstrated how to compute street continuity metrics from OpenStreetMap data for London, using street names, route numbers, highway types, and a hybrid approach to measure how far streets extend as continuous entities. The analysis reveals the hierarchical structure of the road network, with major routes like the A4 and A501 achieving the longest continuous extents.

    **Next steps:** For continuity from Ordnance Survey data, see [OS Open Roads Continuity](https://cityseer.benchmarkurbanism.com/examples/continuity/continuity-os-open).
    """)
    return


if __name__ == "__main__":
    app.run()
