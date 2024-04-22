"""
Convenience methods for plotting graphs within the cityseer API context.

Custom behaviour can be achieved by directly manipulating the underlying [`NetworkX`](https://networkx.github.io) and
[`matplotlib`](https://matplotlib.org) figures. This module is predominately used for basic plots or visual verification
of behaviour in code tests. Users are encouraged to use matplotlib or other plotting packages directly where possible.
"""

# workaround until networkx adopts types
# pyright: basic
from __future__ import annotations

import logging
from typing import Any, Union

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt
from matplotlib import axes, colors
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch
from shapely import geometry
from sklearn.preprocessing import LabelEncoder, minmax_scale
from tqdm import tqdm

from cityseer import config, rustalgos
from cityseer.tools.graphs import EdgeData, NodeData, NodeKey

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# type hack until networkx supports type-hinting
MultiGraph = Any


class ColourMap:
    """Specifies global colour presets."""

    primary: str = "#0091ea"
    accent: str = "#64c1ff"
    info: str = "#0064b7"
    secondary: str = "#d32f2f"
    warning: str = "#9a0007"
    error: str = "#ffffff"
    background: str = "#19181B"


COLOUR_MAP = ColourMap()

ColourType = Union[str, npt.ArrayLike]


def _open_plots_reset():
    plt.close("all")


def plot_nx_primal_or_dual(  # noqa
    primal_graph: MultiGraph | None = None,
    dual_graph: MultiGraph | None = None,
    path: str | None = None,
    labels: bool = False,
    primal_node_size: int = 30,
    primal_node_colour: ColourType | None = None,
    primal_edge_colour: ColourType | None = None,
    dual_node_size: int = 30,
    dual_node_colour: ColourType | None = None,
    dual_edge_colour: ColourType | None = None,
    primal_edge_width: int | float | None = None,
    dual_edge_width: int | float | None = None,
    plot_geoms: bool = True,
    x_lim: tuple[float, float] | None = None,
    y_lim: tuple[float, float] | None = None,
    ax: axes.Axes | None = None,
    **kwargs: dict[str, Any],
):
    """
    Plot a primal or dual cityseer graph.

    Parameters
    ----------
    primal_graph: MultiGraph
        An optional `NetworkX` MultiGraph to plot in the primal representation. Defaults to None.
    dual_graph: MultiGraph
        An optional `NetworkX` MultiGraph to plot in the dual representation. Defaults to None.
    path: str
        An optional filepath: if provided, the image will be saved to the path instead of being displayed. Defaults to
        None.
    labels: bool
        Whether to display node labels. Defaults to False.
    primal_node_size: int
        The diameter for the primal graph's nodes.
    primal_node_colour: str | float | ndarray
        Primal node colour or colours. When passing an iterable of colours, the number of colours should match the order
        and number of nodes in the MultiGraph. The colours are passed to the underlying
        [`draw_networkx`](https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.drawing.nx_pylab.draw_networkx.html#draw-networkx)  # pylint: disable=line-too-long
        method and should be formatted accordingly. Defaults to None.
    primal_edge_colour: str | float | ndarray
        Primal edge colour or colours. When passing an iterable of colours, the number of colours should match the order
        and number of edges in the MultiGraph. The colours are passed to the underlying
        [`draw_networkx`](https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.drawing.nx_pylab.draw_networkx.html#draw-networkx)  # pylint: disable=line-too-long
        method and should be formatted accordingly. Defaults to None.
    dual_node_size:  int
        The diameter for the dual graph's nodes.
    dual_node_colour: str | float | ndarray
        Dual node colour or colours. When passing a list of colours, the number of colours should match the order and
        number of nodes in the MultiGraph. The colours are passed to the underlying
        [`draw_networkx`](https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.drawing.nx_pylab.draw_networkx.html#draw-networkx)  # pylint: disable=line-too-long
        method and should be formatted accordingly. Defaults to None.
    dual_edge_colour: str | float | ndarray
        Dual edge colour or colours. When passing an iterable of colours, the number of colours should match the order
        and number of edges in the MultiGraph. The colours are passed to the underlying
        [`draw_networkx`](https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.drawing.nx_pylab.draw_networkx.html#draw-networkx)  # pylint: disable=line-too-long
        method and should be formatted accordingly. Defaults to None.
    primal_edge_width: float
        Linewidths for the primal edge. Defaults to None.
    dual_edge_width: float
        Linewidths for the dual edge. Defaults to None.
    plot_geoms: bool
        Whether to plot the edge geometries. If set to `False`, straight lines will be drawn from node-to-node to
        represent edges. Defaults to True.
    x_lim: tuple[float, float]
        A tuple or list with the minimum and maxium `x` extents to be plotted. Defaults to None.
    y_lim: tuple[float, float]
        A tuple or list with the minimum and maxium `y` extents to be plotted. Defaults to None.
    ax: plt.Axes
        An optional `matplotlib` `ax` to which to plot. If not provided, a figure and ax will be generated.
    kwargs
        `kwargs` which will be passed to the `matplotlib` figure parameters. If provided, these will override the
        default figure size or dpi parameters.

    Examples
    --------
    Plot either or both primal and dual representations of a `networkX MultiGraph`. Only call this function directly if
    explicitly printing both primal and dual graphs. Otherwise, use the simplified [`plot_nx`](/tools/plot#plot-nx)
    method instead.

    ```py
    from cityseer.tools import mock, graphs, plot
    G = mock.mock_graph()
    G_simple = graphs.nx_simple_geoms(G)
    G_dual = graphs.nx_to_dual(G_simple)
    plot.plot_nx_primal_or_dual(G_simple,
                                G_dual,
                                plot_geoms=False)
    ```
    ![Example primal and dual graph plot.](/images/graph_dual.png)
    _A dual graph in blue overlaid on the source primal graph in red._

    """
    # cleanup old plots
    if ax is None:
        _open_plots_reset()
        # create new plot
        _fig, target_ax = plt.subplots(1, 1, **kwargs)  # type: ignore
    else:
        target_ax = ax
    # setup params
    alpha = 0.75
    if labels:
        alpha = 0.95

    # setup a function that can be used for either the primal or dual graph
    def _plot_graph(
        _graph: MultiGraph,
        _is_primal: bool,
        _node_size: float,
        _node_colour: ColourType | None,
        _node_shape: str,
        _edge_colour: ColourType | None,
        _edge_style: str,
        _edge_width: int | float | None,
    ) -> None:
        if not len(_graph):  # pylint: disable=len-as-condition
            raise ValueError("Graph contains no nodes to plot.")
        if not isinstance(_node_size, int) or _node_size < 1:
            raise ValueError("Node sizes should be a positive integer.")
        if _node_colour is not None:
            if isinstance(_node_colour, np.ndarray) and len(_node_colour) != len(_graph):
                raise ValueError(
                    "A list, tuple, or array of node colours should match the number of nodes in the graph."
                )
        else:
            if _is_primal:
                _node_colour = COLOUR_MAP.secondary
            else:
                _node_colour = COLOUR_MAP.info
        if _edge_colour is not None:
            if isinstance(_edge_colour, np.ndarray) and len(_edge_colour) != len(_graph.edges()):
                raise ValueError(
                    "A list, tuple, or array of edge colours should match the number of edges in the graph."
                )
        else:
            if _is_primal:
                _edge_colour = "w"
            else:
                _edge_colour = COLOUR_MAP.accent
        # edge width
        if _edge_width is not None:
            if not isinstance(_edge_width, (int, float)):
                raise ValueError("Edge widths should be an int or float.")
        else:
            if _is_primal:
                _edge_width = 1.5
            else:
                _edge_width = 0.5
        pos = {}
        node_list = []
        # setup a node colour list if nodes are individually coloured, this is for filtering by extents
        logger.info("Preparing graph nodes")
        colour_list: list[ColourType] = []
        node_key: NodeKey
        node_data: NodeData
        for n_idx, (node_key, node_data) in enumerate(_graph.nodes(data=True)):
            if x_lim is not None:
                if node_data["x"] < x_lim[0] or node_data["x"] > x_lim[1]:
                    continue
            if y_lim is not None:
                if node_data["y"] < y_lim[0] or node_data["y"] > y_lim[1]:
                    continue
            # add to the pos dictionary regardless (otherwise nx.draw throws an error)
            pos[node_key] = (node_data["x"], node_data["y"])
            # add to the node list
            node_list.append(node_key)
            # and add to the node colour list if node colours are a list or tuple
            if isinstance(_node_colour, np.ndarray):
                colour_list.append(_node_colour[n_idx])
        if not node_list:
            raise ValueError("All nodes have been filtered out by the x_lim / y_lim parameter: check your extents")
        # update the node colours to the filtered list of colours if necessary
        if isinstance(_node_colour, np.ndarray):
            _node_colour = np.array(colour_list)
        # plot edges manually for geoms
        logger.info("Preparing graph edges")
        edge_list = []
        edge_geoms = []
        start_node_key: NodeKey
        end_node_key: NodeKey
        node_data: NodeData
        for start_node_key, end_node_key, node_data in tqdm(_graph.edges(data=True)):  # type: ignore
            # filter out if start and end nodes are not in the active node list
            if start_node_key not in node_list or end_node_key not in node_list:
                continue
            if plot_geoms:
                try:
                    x_arr: npt.ArrayLike
                    y_arr: npt.ArrayLike
                    x_arr, y_arr = node_data["geom"].coords.xy
                except KeyError as err:
                    raise KeyError(
                        f"Can't plot geoms because a 'geom' key can't be found for edge {start_node_key} to "
                        f"{end_node_key}. Use the nx_simple_geoms() method if you need to create geoms for a graph."
                    ) from err
                edge_geoms.append(tuple(zip(x_arr, y_arr)))  # type: ignore
            else:
                edge_list.append((start_node_key, end_node_key))
        # plot geoms manually if required
        if plot_geoms:
            lines = LineCollection(
                edge_geoms,
                colors=_edge_colour,  # type: ignore
                linewidths=_edge_width,  # type: ignore
                linestyles=_edge_style,  # type: ignore
            )
            target_ax.add_collection(lines)  # type: ignore
        # go ahead and plot: note that edge_list will be empty if plotting geoms
        nx.draw(  # type: ignore
            _graph,
            pos=pos,
            ax=target_ax,
            with_labels=labels,
            font_size=5,
            font_color="w",
            font_weight="bold",
            nodelist=node_list,
            node_color=_node_colour,
            node_size=_node_size,
            node_shape=_node_shape,
            edgelist=edge_list,
            edge_color=_edge_colour,
            style=_edge_style,
            width=_edge_width,
            alpha=alpha,
        )

    if primal_graph is not None:
        _plot_graph(
            primal_graph,
            True,
            primal_node_size,
            primal_node_colour,
            "o",
            primal_edge_colour,
            "solid",
            primal_edge_width,
        )
    if dual_graph is not None:
        _plot_graph(
            dual_graph,
            False,
            dual_node_size,
            dual_node_colour,
            "d",
            dual_edge_colour,
            "dashed",
            dual_edge_width,
        )
    if x_lim:
        plt.xlim(*x_lim)
    if y_lim:
        plt.ylim(*y_lim)
    if ax is None:
        plt.tight_layout()
        if path is not None:
            plt.savefig(path, facecolor=COLOUR_MAP.background)
        else:
            plt.gcf().set_facecolor(COLOUR_MAP.background)
            plt.show()


def plot_nx(
    nx_multigraph: MultiGraph,
    path: str | None = None,
    labels: bool = False,
    node_size: int = 20,
    node_colour: ColourType | None = None,
    edge_colour: ColourType | None = None,
    edge_width: int | float | None = None,
    plot_geoms: bool = False,
    x_lim: tuple[float, float] | None = None,
    y_lim: tuple[float, float] | None = None,
    ax: axes.Axes | None = None,
    **kwargs: dict[str, Any],
):  # noqa
    """
    Plot a `networkX` MultiGraph.

    Parameters
    ----------
    nx_multigraph: MultiGraph
        A `NetworkX` MultiGraph.
    path: str
        An optional filepath: if provided, the image will be saved to the path instead of being displayed. Defaults to
        None.
    labels: bool
        Whether to display node labels. Defaults to False.
    node_size: int
        The diameter for the graph's nodes.
    node_colour: str | float | ndarray
        Node colour or colours. When passing an iterable of colours, the number of colours should match the order and
        number of nodes in the MultiGraph. The colours are passed to the underlying
        [`draw_networkx`](https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.drawing.nx_pylab.draw_networkx.html#draw-networkx)  # pylint: disable=line-too-long
        method and should be formatted accordingly. Defaults to None.
    edge_colour: str | float | ndarray
        Edges colour as a `matplotlib` compatible colour string. Defaults to None.
    edge_width: float
        Linewidths for edges. Defaults to None.
    plot_geoms: bool
        Whether to plot the edge geometries. If set to `False`, straight lines will be drawn from node-to-node to
        represent edges. Defaults to True.
    x_lim: tuple[float, float]
        A tuple or list with the minimum and maximum `x` extents to be plotted. Defaults to None.
    y_lim: tuple[float, float]
        A tuple or list with the minimum and maximum `y` extents to be plotted. Defaults to None.
    ax: plt.Axes
        An optional `matplotlib` `ax` to which to plot. If not provided, a figure and ax will be generated.
    kwargs
        `kwargs` which will be passed to the `matplotlib` figure parameters. If provided, these will override the
        default figure size or dpi parameters.

    Examples
    --------
    ```py
    from cityseer.tools import mock, graphs, plot, io
    from cityseer.metrics import networks
    from matplotlib import colors

    # generate a MultiGraph and compute gravity
    G = mock.mock_graph()
    G = graphs.nx_simple_geoms(G)
    G = graphs.nx_decompose(G, 50)
    nodes_gdf, edges_gdf, network_structure = io.network_structure_from_nx(G, crs=3395)
    networks.node_centrality_shortest(
        network_structure=network_structure,
        nodes_gdf=nodes_gdf,
        distances=[800]
    )
    G_after = io.nx_from_cityseer_geopandas(nodes_gdf, edges_gdf)
    # let's extract and normalise the values
    vals = []
    for node, data in G_after.nodes(data=True):
        vals.append(data["cc_beta_800"])
    # let's create a custom colourmap using matplotlib
    cmap = colors.LinearSegmentedColormap.from_list(
        "cityseer", [(100 / 255, 193 / 255, 255 / 255, 255 / 255), (211 / 255, 47 / 255, 47 / 255, 1 / 255)]
    )
    # normalise the values
    vals = colors.Normalize()(vals)
    # cast against the colour map
    cols = cmap(vals)
    # plot
    plot.plot_nx(G_after, node_colour=cols)
    ```

    ![Example Colour Plot.](/images/graph_colour.png)
    _Colour plot of 800m gravity index centrality on a 50m decomposed graph._
    """
    return plot_nx_primal_or_dual(
        primal_graph=nx_multigraph,
        path=path,
        labels=labels,
        primal_node_size=node_size,
        primal_node_colour=node_colour,
        primal_edge_colour=edge_colour,
        primal_edge_width=edge_width,
        plot_geoms=plot_geoms,
        x_lim=x_lim,
        y_lim=y_lim,
        ax=ax,
        **kwargs,  # type: ignore
    )


def plot_assignment(
    network_structure: rustalgos.NetworkStructure,
    nx_multigraph: MultiGraph,
    data_gdf: gpd.GeoDataFrame,
    path: str | None = None,
    node_colour: ColourType | None = None,
    node_labels: bool = False,
    data_labels: npt.ArrayLike | None = None,
    **kwargs: dict[str, Any],
):
    """
    Plot a `network_structure` and `data_gdf` for visualising assignment of data points to respective nodes.

    :::warning
    This method is primarily intended for package testing and development.
    :::

    Parameters
    ----------
    network_structure: rustalgos.NetworkStructure
        A [`rustalgos.NetworkStructure`](/rustalgos/rustalgos#networkstructure) instance.
    nx_multigraph: MultiGraph
        A `NetworkX` MultiGraph.
    data_gdf: GeoDataFrame
        A `data_gdf` `GeoDataFrame` with `nearest_assigned` and `next_neareset_assign` columns.
    path: str
        An optional filepath: if provided, the image will be saved to the path instead of being displayed. Defaults to
        None.
    node_colour: str | float | ndarray
        Node colour or colours. When passing a list of colours, the number of colours should match the order and number
        of nodes in the MultiGraph. The colours are passed to the underlying
        [`draw_networkx`](https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.drawing.nx_pylab.draw_networkx.html#draw-networkx)  # pylint: disable=line-too-long
        method and should be formatted accordingly. Defaults to None.
    node_labels: bool
        Whether to plot the node labels. Defaults to False.
    data_labels: ndarray[int | str]
        An optional iterable of categorical data labels which will be mapped to colours. The number of labels should
        match the number of data points in `data_layer`. Defaults to None.
    kwargs
        `kwargs` which will be passed to the `matplotlib` figure parameters. If provided, these will override the
        default figure size or dpi parameters.

    Examples
    --------
    ![Example assignment plot.](/images/assignment_plot.png)
    _An assignment plot to a 50m decomposed graph, with the data points coloured by categorical labels._

    """
    _open_plots_reset()
    plt.figure(**kwargs)  # type: ignore

    if isinstance(node_colour, (list, tuple, np.ndarray)):
        if not (len(node_colour) == 1 or len(node_colour) == len(nx_multigraph)):
            raise ValueError(
                "Node colours should either be a single colour or a list or tuple of colours matching "
                "the number of nodes in the graph."
            )
    elif node_colour is None:
        node_colour = COLOUR_MAP.secondary

    # do a simple plot - don't provide path
    pos = {}
    node_key: NodeKey
    node_data: NodeData
    for node_key, node_data in nx_multigraph.nodes(data=True):
        pos[node_key] = (node_data["x"], node_data["y"])
    nx.draw(
        nx_multigraph,
        pos,
        with_labels=node_labels,
        font_size=5,
        font_color="w",
        font_weight="bold",
        node_color=node_colour,
        node_size=30,
        node_shape="o",
        edge_color="w",
        width=1,
        alpha=0.75,
    )

    data_colour: ColourType
    if data_labels is None:
        data_colour = COLOUR_MAP.info
        data_cmap = None
    else:
        # generate categorical colormap
        # use sklearn's label encoder
        lab_enc = LabelEncoder()
        lab_enc.fit(data_labels)
        # map the int encodings to the respective classes
        classes_int: npt.ArrayLike = lab_enc.transform(data_labels)
        data_colour = colors.Normalize()(classes_int)
        data_cmap = "Dark2"  # Set1

    # overlay data map
    plt.scatter(
        x=data_gdf.geometry.x,
        y=data_gdf.geometry.y,
        c=data_colour,  # type: ignore
        cmap=data_cmap,  # type: ignore
        s=30,
        edgecolors="white",
        lw=0.5,
        alpha=0.95,
    )
    if "nearest_assign" not in data_gdf.columns:
        raise ValueError(
            "Cannot plot assignment for GeoDataFrame that has not yet been assigned to a NetworkStructure."
        )
    # draw assignment
    for _data_key, data_row in data_gdf.iterrows():
        # if the data points have been assigned network indices
        data_x: float = data_row.geometry.x
        data_y: float = data_row.geometry.y
        nearest_netw_idx: int = data_row.nearest_assign
        next_nearest_netw_idx: int = data_row.next_nearest_assign
        if nearest_netw_idx is not None:
            # plot lines to parents for easier viz
            p_x = network_structure.node_xs[nearest_netw_idx]
            p_y = network_structure.node_ys[nearest_netw_idx]
            plt.plot(
                [p_x, data_x],
                [p_y, data_y],
                c="#64c1ff",
                lw=0.5,
                ls="--",
            )
        if next_nearest_netw_idx is not None:
            p_x = network_structure.node_xs[next_nearest_netw_idx]
            p_y = network_structure.node_ys[next_nearest_netw_idx]
            plt.plot([p_x, data_x], [p_y, data_y], c="#888888", lw=0.5, ls="--")

    plt.tight_layout()
    if path:
        plt.savefig(path, facecolor=COLOUR_MAP.background, dpi=150)
    else:
        plt.gcf().set_facecolor(COLOUR_MAP.background)
        plt.show()


def plot_network_structure(
    network_structure: rustalgos.NetworkStructure,
    data_gdf: gpd.GeoDataFrame,
    poly: geometry.Polygon | None = None,
):
    """
    Plot a graph from raw `cityseer` network structure.

    :::note
    Note that this function is subject to frequent revision pending short-term development requirements. It is used
    mainly to visually confirm the correct behaviour of particular algorithms during the software development cycle.
    :::

    Parameters
    ----------
    network_structure: rustalgos.NetworkStructure
        A [`rustalgos.NetworkStructure`](/rustalgos/rustalgos#networkstructure) instance.
    data_gdf: GeoDataFrame
        A `data_gdf` `GeoDataFrame` with `nearest_assigned` and `next_neareset_assign` columns.
    poly: geometry.Polygon
        An optional polygon. Defaults to None.

    """
    _open_plots_reset()
    # the edges are bi-directional - therefore duplicated per directional from-to edge
    # use two axes to check each copy of edges
    _fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 12))
    # set extents
    xs_arr = np.array(network_structure.node_xs)
    ys_arr = np.array(network_structure.node_ys)
    for ax in (ax1, ax2):
        ax.set_xlim(xs_arr.min() - 100, xs_arr.max() + 100)
        ax.set_ylim(ys_arr.min() - 100, ys_arr.max() + 100)
    if poly:
        x = list(poly.exterior.coords.xy[0])
        y = list(poly.exterior.coords.xy[1])
        ax1.plot(x, y)
        ax2.plot(x, y)
    # plot nodes
    cols = []
    for is_live in network_structure.node_lives:
        if is_live:
            cols.append(COLOUR_MAP.accent)
        else:
            cols.append(COLOUR_MAP.secondary)
    ax1.scatter(
        network_structure.node_xs, network_structure.node_ys, s=30, c=COLOUR_MAP.primary, edgecolor=cols, lw=0.5
    )
    ax2.scatter(
        network_structure.node_xs, network_structure.node_ys, s=30, c=COLOUR_MAP.primary, edgecolor=cols, lw=0.5
    )
    # plot edges
    processed_edges: set[str] = set()
    for start_nd_idx, end_nd_idx, _edge_idx in network_structure.edge_references():
        keys = sorted([start_nd_idx, end_nd_idx])
        se_key = "-".join([str(k) for k in keys])
        # bool indicating whether second copy in opposite direction
        dupe = se_key in processed_edges
        processed_edges.add(se_key)
        # get the start and end coords - this is to check that still within src edge - neighbour range
        s_x, s_y = network_structure.node_xs[start_nd_idx], network_structure.node_ys[start_nd_idx]
        e_x, e_y = network_structure.node_xs[end_nd_idx], network_structure.node_ys[end_nd_idx]
        if not dupe:
            ax1.plot([s_x, e_x], [s_y, e_y], c=COLOUR_MAP.accent, linewidth=1)
        else:
            ax2.plot([s_x, e_x], [s_y, e_y], c=COLOUR_MAP.accent, linewidth=1)
    for node_idx in range(network_structure.node_count()):
        ax2.annotate(node_idx, xy=network_structure.node_xys[node_idx], size=5)
    # plot parents on ax1
    ax1.scatter(
        x=data_gdf.geometry.x,
        y=data_gdf.geometry.y,
        color=COLOUR_MAP.secondary,
        edgecolor=COLOUR_MAP.warning,
        alpha=0.9,
        lw=0.5,
    )
    ax2.scatter(
        x=data_gdf.geometry.x,
        y=data_gdf.geometry.y,
        color=COLOUR_MAP.secondary,
        edgecolor=COLOUR_MAP.warning,
        alpha=0.9,
        lw=0.5,
    )
    for data_idx, data_row in data_gdf.iterrows():
        data_x: float = data_row.geometry.x
        data_y: float = data_row.geometry.y
        nearest_netw_idx: int = data_row.nearest_assign
        next_nearest_netw_idx: int = data_row.next_nearest_assign
        ax2.annotate(str(data_idx), xy=(data_x, data_y), size=8, color="red")
        # if the data points have been assigned network indices
        if nearest_netw_idx is not None:
            # plot lines to parents for easier viz
            p_x, p_y = network_structure.node_xys[int(nearest_netw_idx)]
            ax1.plot([p_x, data_x], [p_y, data_y], c=COLOUR_MAP.warning, lw=0.75, ls="-")
        if next_nearest_netw_idx is not None:
            p_x, p_y = network_structure.node_xys[int(next_nearest_netw_idx)]
            ax1.plot([p_x, data_x], [p_y, data_y], c=COLOUR_MAP.info, lw=0.75, ls="--")
    plt.tight_layout()
    plt.gcf().set_facecolor(COLOUR_MAP.background)
    plt.show()


def plot_scatter(
    ax: axes.Axes,
    xs: list[float] | npt.ArrayLike,
    ys: list[float] | npt.ArrayLike,
    vals: npt.ArrayLike,
    bbox_extents: tuple[int, int, int, int] | tuple[float, float, float, float],
    perc_range: tuple[float, float] = (0.01, 99.99),
    cmap_key: str = "viridis",
    shape_exp: float = 1,
    s_min: float = 0.1,
    s_max: float = 1,
    rasterized: bool = True,
    face_colour: str = "#111",
) -> Any:
    """
    Convenience plotting function for plotting outputs from examples in the Cityseer Guide.

    Parameters
    ----------
    ax: plt.Axes
        A 'matplotlib' `Ax` to which to plot.
    xs: ndarray[float]
        A numpy array of floats representing the `x` coordinates for points to plot.
    ys: ndarray[float]
        A numpy array of floats representing the `y` coordinates for points to plot.
    vals: ndarray[float]
        A numpy array of floats representing the data values for the provided points.
    bbox_extents: tuple[int, int, int, int]
        A tuple or list containing the `[s, w, n, e]` bounding box extents for clipping the plot.
    perc_range: tuple[float, float]
        A tuple of two floats, representing the minimum and maximum percentiles at which to clip the data.
    cmap_key: str
        A `matplotlib` colour map key.
    shape_exp: float
        A float representing an exponential for reshaping the values distribution. Defaults to 1 which returns the
        values as provided. An exponential greater than or less than 1 will shape the values distribution accordingly.
    s_min: float
        A float representing the minimum size for a plotted point.
    s_max: float
        A float representing the maximum size for a plotted point.
    rasterized: bool
        Whether or not to rasterise the output. Recommended for plots with a large number of points.
    face_colour: str
        A hex or other valid `matplotlib` colour value for the ax and figure faces (backgrounds).

    """
    xs = np.array(xs)
    ys = np.array(ys)
    # get extents relative to centre and ax size
    min_x, min_y, max_x, max_y = bbox_extents
    # filter
    select = xs > min_x
    select = np.logical_and(select, xs < max_x)
    select = np.logical_and(select, ys > min_y)
    select = np.logical_and(select, ys < max_y)
    select_idx = np.where(select)[0]
    # remove any extreme outliers
    v_min: float = np.nanpercentile(vals, perc_range[0])  # type: ignore
    v_max: float = np.nanpercentile(vals, perc_range[1])  # type: ignore
    v_shape = np.clip(vals, v_min, v_max)
    # shape if requested
    v_shape = v_shape**shape_exp
    # normalise
    c_norm = mpl.colors.Normalize(vmin=v_min, vmax=v_max, clip=True)  # type: ignore
    colours: npt.ArrayLike = c_norm(v_shape)
    sizes: npt.ArrayLike = minmax_scale(colours, (s_min, s_max))  # type: ignore
    # plot
    img: Any = ax.scatter(
        xs[select_idx],
        ys[select_idx],
        c=colours[select_idx],  # type: ignore
        s=sizes[select_idx],  # type: ignore
        linewidths=0,
        edgecolors="none",
        cmap=plt.get_cmap(cmap_key),  # type: ignore
        rasterized=rasterized,
    )
    # limits
    ax.set_xlim(left=min_x, right=max_x)
    ax.set_ylim(bottom=min_y, top=max_y)
    ax.set_xticks([])  # type: ignore
    ax.set_yticks([])  # type: ignore
    ax.set_aspect(1)
    ax.set_facecolor(face_colour)
    ax.axis("off")

    return img


def plot_nx_edges(
    ax: axes.Axes,
    nx_multigraph: nx.MultiGraph,
    edge_metrics_key: str,
    bbox_extents: tuple[int, int, int, int] | tuple[float, float, float, float],
    perc_range: tuple[float, float] = (0.01, 99.99),
    cmap_key: str = "viridis",
    shape_exp: float = 1,
    lw_min: float = 0.1,
    lw_max: float = 1,
    edge_label_key: str | None = None,
    colour_by_categorical: bool = False,
    max_n_categorical: int = 10,
    rasterized: bool = True,
    face_colour: str = "#111",
    invert_plot_order: bool = False,
):
    """
    Convenience plotting function for plotting outputs from examples in the Cityseer Guide.

    Parameters
    ----------
    ax: plt.Axes
        A 'matplotlib' `Ax` to which to plot.
    nx_multigraph: MultiGraph
        A `NetworkX` MultiGraph.
    edge_metrics_key: str
        An edge key for the provided `nx_multigraph`. Plotted values will be retrieved from this edge key.
    bbox_extents: tuple[int, int, int, int]
        A tuple or list containing the `[s, w, n, e]` bounding box extents for clipping the plot.
    perc_range: tuple[float, float]
        A tuple of two floats, representing the minimum and maximum percentiles at which to clip the data.
    cmap_key: str
        A `matplotlib` colour map key.
    shape_exp: float
        A float representing an exponential for reshaping the values distribution. Defaults to 1 which returns the
        values as provided. An exponential greater than or less than 1 will shape the values distribution accordingly.
    lw_min: float
        A float representing the minimum line width for a plotted edge.
    lw_max: float
        A float representing the maximum line width for a plotted edge.
    edge_label_key: str
        A key for retrieving categorical labels from edges.
    colour_by_categorical: bool
        Whether to plot colours by categorical. This requires an `edge_label_key` parameter.
    max_n_categorical: int
        The number of categorical values (sorted in decreasing order) to plot.
    rasterized: bool
        Whether or not to rasterise the output. Recommended for plots with a large number of edges.
    face_colour: str
        A hex or other valid `matplotlib` colour value for the ax and figure faces (backgrounds).
    invert_plot_order: bool
        Whether to invert the plot order, e.g. if using an inverse colour map.

    """
    min_x, min_y, max_x, max_y = bbox_extents
    cmap = plt.get_cmap(cmap_key)  # type: ignore
    # extract data for shaping
    vals: list[str] = []
    edge_geoms: list[geometry.LineString] = []
    labels_info: dict[str, dict[str, Any]] = {}
    logger.info("Extracting edge geometries")
    edge_data: EdgeData
    for idx, (_, _, edge_data) in tqdm(  # type: ignore
        enumerate(nx_multigraph.edges(data=True)), disable=config.QUIET_MODE
    ):
        vals.append(edge_data[edge_metrics_key])
        edge_geoms.append(edge_data["geom"])
        # if label key provided
        if edge_label_key is not None:
            label_val: str | None = edge_data[edge_label_key]
            if label_val is None or label_val.lower() in [
                "",
                " ",
                "not classified",
                "unclassified",
                "service",
                "residential",
                "unknown",
                "local road",
            ]:
                label_val = "other"
            if label_val not in labels_info:
                labels_info[label_val] = {"count": 1, "idxs": []}
            else:
                labels_info[label_val]["count"] += 1
                labels_info[label_val]["idxs"].append(idx)
    vals_arr: npt.ArrayLike = np.array(vals)
    # remove any extreme outliers
    v_min: float = np.nanpercentile(vals_arr, perc_range[0])  # type: ignore
    v_max: float = np.nanpercentile(vals_arr, perc_range[1])  # type: ignore
    v_shape = np.clip(vals_arr, v_min, v_max)
    # shape if requested
    v_shape = v_shape**shape_exp
    # normalise
    c_norm = mpl.colors.Normalize(vmin=v_shape.min(), vmax=v_shape.max())  # type: ignore
    colours: npt.ArrayLike = c_norm(v_shape)
    sizes: npt.ArrayLike = minmax_scale(colours, (lw_min, lw_max))  # type: ignore
    # sort so that larger lines plot over smaller lines
    sort_idx: npt.ArrayLike = np.argsort(colours)
    if invert_plot_order:
        sort_idx = sort_idx[::-1]
    # plot using geoms
    logger.info("Generating plot")
    if not colour_by_categorical:
        plot_geoms = []
        plot_colours = []
        plot_lws = []
        idx: int
        for idx in tqdm(sort_idx, disable=config.QUIET_MODE):
            xs = np.array(edge_geoms[idx].coords.xy[0])
            ys = np.array(edge_geoms[idx].coords.xy[1])
            if np.any(xs < min_x) or np.any(xs > max_x):
                continue
            if np.any(ys < min_y) or np.any(ys > max_y):
                continue
            plot_geoms.append(tuple(zip(xs, ys)))
            plot_colours.append(cmap(colours[idx]))  # type: ignore
            plot_lws.append(sizes[idx])  # type: ignore
        lines = LineCollection(
            plot_geoms,
            colors=plot_colours,
            linewidths=plot_lws,
            rasterized=rasterized,
            alpha=0.9,
        )
        ax.add_collection(lines)  # type: ignore
    else:
        plot_handles = []
        plot_geoms = []
        plot_colours = []
        plot_lws = []
        # extract sorted counts by decreasing order
        labels_info = dict(sorted(labels_info.items(), key=lambda item: item[1]["count"], reverse=True))
        label_keys = [k for k in labels_info.keys() if k != "other"]
        label_counts = [v["count"] for k, v in labels_info.items() if k != "other"]
        # clip by maximum categoricals
        if len(label_keys) > max_n_categorical:
            label_keys = label_keys[:max_n_categorical]
            label_counts = label_counts[:max_n_categorical]
        # sort by increasing
        label_keys.reverse()
        label_counts.reverse()
        # iterate label info
        for label_key in tqdm(["other"] + label_keys, disable=config.QUIET_MODE):
            if label_key not in labels_info:
                continue
            # if label count not clipped
            label_info = labels_info[label_key]
            label_count = label_info["count"]
            label_idxs = label_info["idxs"]
            if label_count in label_counts:
                item_wt = (label_counts.index(label_count)) / (max_n_categorical - 1)
                item_c = cmap(item_wt)
                s_range = lw_max - lw_min
                item_lw = lw_min + item_wt * s_range
                plot_handles.append(Patch(facecolor=item_c, edgecolor=item_c, label=label_key))
            else:
                item_c = "#444"
                item_lw = lw_min
            for idx in label_idxs:
                xs = np.array(edge_geoms[idx].coords.xy[0])
                ys = np.array(edge_geoms[idx].coords.xy[1])
                if np.any(xs < min_x) or np.any(xs > max_x):
                    continue
                if np.any(ys < min_y) or np.any(ys > max_y):
                    continue
                plot_geoms.append(tuple(zip(xs, ys)))
                plot_colours.append(item_c)
                plot_lws.append(item_lw)
        lines = LineCollection(
            plot_geoms,
            colors=plot_colours,
            linewidths=plot_lws,
            rasterized=rasterized,
            alpha=0.9,
        )
        ax.add_collection(lines)  # type: ignore
        ax.legend(handles=plot_handles)

    ax.set_xlim(left=min_x, right=max_x)
    ax.set_ylim(bottom=min_y, top=max_y)
    ax.set_xticks([])  # type: ignore
    ax.set_yticks([])  # type: ignore
    ax.set_aspect(1)
    ax.set_facecolor(face_colour)
