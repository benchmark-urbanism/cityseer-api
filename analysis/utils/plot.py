import pathlib

import colorcet as cc
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from cityseer.tools import plot as cityseer_plot
from shapely import geometry


def reset_plots():
    """ """
    plt.ioff()
    plt.close("all")
    plt.cla()
    plt.clf()


def plot_poly(polygon: geometry.Polygon, ax: plt.axes) -> None:
    # exterior
    x_arr, y_arr = polygon.exterior.coords.xy
    ax.plot(x_arr, y_arr)
    # interior
    for interior in polygon.interiors:
        x_arr, y_arr = interior.coords.xy
        ax.plot(x_arr, y_arr)


def plot_multipoints(points: list[geometry.Point], ax: plt.axes, size: int, color: str, alpha: float) -> None:
    x_coords = []
    y_coords = []
    for point in points:
        x_coords.append(point.x)
        y_coords.append(point.y)
    ax.scatter(x_coords, y_coords, s=size, c=color, alpha=alpha)


def plot_substrate(
    graph: nx.MultiGraph,
    residential_centroids: list[geometry.Point],
    retail_centroids: list[geometry.Point],
    **kwargs,
) -> None:
    """ """
    # setup
    reset_plots()
    fig, ax = plt.subplots(**kwargs)
    # plot streets
    graph_centroids = []
    for _n, d in graph.nodes(data=True):
        graph_centroids.append(geometry.Point(d["x"], d["y"]))
    plot_multipoints(graph_centroids, ax, 2, "black", 0.5)
    # plot residential entrances
    plot_multipoints(residential_centroids, ax, 2, "g", 1)
    plot_multipoints(retail_centroids, ax, 1, "r", 0.5)
    plt.show()


def simple_plot(nx_graph: nx.MultiGraph, plot_geoms: bool = True):
    """ """
    cityseer_plot.plot_nX(
        nx_graph,
        labels=False,
        plot_geoms=plot_geoms,
        node_size=15,
        edge_width=2,
        figsize=(10, 10),
        dpi=200,
    )


def plot_main(
    ax: plt.axis,
    background_colour: str,
    nx_graph: nx.MultiGraph,
    i_data_map: np.ndarray,
    i_dens: np.ndarray,
    j_data_map: np.ndarray,
    j_lu_vibr_arr: np.ndarray,
    j_lu_spec_arr: np.ndarray,
    netw_flows: np.ndarray,
    prime_locations: np.ndarray = None,
) -> plt.axis:
    """ """
    xs = []
    ys = []
    for _n, d in nx_graph.nodes(data=True):
        xs.append(d["x"])
        ys.append(d["y"])
    xs = np.array(xs)
    ys = np.array(ys)
    ## plot flows
    flows_norm = netw_flows - np.nanmin(netw_flows)
    flows_norm /= np.nanmax(flows_norm)
    flows_norm = flows_norm
    fc = cc.cm.kb(flows_norm)
    # fc[:, 3] = flows_norm
    ax.scatter(
        xs,
        ys,
        c=background_colour,
        edgecolors=fc,
        s=flows_norm * 10,
        lw=0.5,
        marker="D",
    )
    # plot primeness
    if prime_locations is not None:
        primeness_norm = prime_locations - np.nanmin(prime_locations)
        primeness_norm /= np.nanmax(primeness_norm)
        primeness_norm = primeness_norm
        pc = cc.cm.kg(primeness_norm)
        # pc[:, 3] = primeness_norm
        ax.scatter(xs, ys, c=pc, s=primeness_norm, lw=0.1, marker="D")
    # set limits based on network
    buffer = 5
    ax.set_xlim(left=np.nanmin(xs) - buffer, right=np.nanmax(xs) + buffer)
    ax.set_ylim(bottom=np.nanmin(ys) - buffer, top=np.nanmax(ys) + buffer)
    ### PLOT i
    xs = []
    ys = []
    vs = []
    for (x, y, _, _), dens in zip(i_data_map, i_dens, strict=False):
        if not np.isnan(dens):
            xs.append(x)
            ys.append(y)
            vs.append(dens)
    ax.scatter(xs, ys, c="#eee", marker="x", lw=0.1, s=np.array(vs) / np.nanmax(vs) * 1)
    ### plot j
    xs = []
    ys = []
    cs = []
    ss = []
    for (x, y, _, _), type, spec in zip(j_data_map, j_lu_vibr_arr, j_lu_spec_arr, strict=False):
        if not np.isnan(type):
            xs.append(x)
            ys.append(y)
            c = list(cc.cm.rainbow(type))
            cs.append(c)
            ss.append(spec)
    ss = np.array(ss)
    ax.scatter(xs, ys, c=cs, edgecolors="white", lw=0.1, s=ss * 2, marker="s")
    # plot
    ax.axis("off")
    return ax


def plot_lu_box(ax: plt.axis, j_lu_vibr_arr: np.ndarray, j_lu_spec_arr: np.ndarray) -> plt.axis:
    """ """
    # plot landuses scatter
    lu_vibr_col = cc.cm.rainbow(j_lu_vibr_arr)
    lu_vibr_col[:, 3] = 0.9
    lu_spec_col = (j_lu_spec_arr * 0.8 + 0.2) * 0.5
    ax.scatter(
        j_lu_vibr_arr,
        j_lu_spec_arr,
        c=lu_vibr_col,
        s=lu_spec_col,
        edgecolors="white",
        lw=0.1,
        marker="s",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # plot percentiles
    for perc in [25, 50, 75]:
        ax.axvline(np.nanpercentile(j_lu_vibr_arr, perc), c="#eee", ls="-.", lw=0.1)
        ax.axhline(np.nanpercentile(j_lu_spec_arr, perc), c="#eee", ls="-.", lw=0.1)
    return ax


def plot_state(
    nx_graph: nx.MultiGraph,
    i_data_map: np.ndarray,
    i_dens: np.ndarray,
    j_data_map: np.ndarray,
    j_lu_vibr_arr: np.ndarray,
    j_lu_spec_arr: np.ndarray,
    netw_flows: np.ndarray,
    prime_locations: np.ndarray = None,
    path: str = None,
    **kwargs,
):
    """ """
    reset_plots()
    background_colour = "#161616"
    fig = plt.figure(
        figsize=(6, 4),
        dpi=218,
        facecolor=background_colour,
        frameon=False,
        constrained_layout=True,
        **kwargs,
    )
    subfigs = fig.subfigures(
        nrows=1,
        ncols=2,
        wspace=0,
        hspace=0,
        width_ratios=[2, 1],
        facecolor=background_colour,
    )
    main_ax = subfigs[0].subplots(1, 1)
    right_axes = subfigs[1].subplots(2, 1)
    right_top_ax = right_axes[0]
    right_bot_ax = right_axes[1]
    # plot main ax
    main_ax = plot_main(
        ax=main_ax,
        background_colour=background_colour,
        nx_graph=nx_graph,
        i_data_map=i_data_map,
        i_dens=i_dens,
        j_data_map=j_data_map,
        j_lu_vibr_arr=j_lu_vibr_arr,
        j_lu_spec_arr=j_lu_spec_arr,
        netw_flows=netw_flows,
        prime_locations=prime_locations,
    )
    # plot landuses box
    right_top_ax = plot_lu_box(ax=right_top_ax, j_lu_vibr_arr=j_lu_vibr_arr, j_lu_spec_arr=j_lu_spec_arr)
    # setup axes
    for ax in [main_ax, right_top_ax, right_bot_ax]:
        ax.axis("off")
        ax.set_facecolor(background_colour)
    # save or show
    if path is not None:
        plt.savefig(pathlib.Path.cwd().parent / path, facecolor=background_colour)
    else:
        plt.gcf().set_facecolor(background_colour)
        plt.show()
