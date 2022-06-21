"""
Convenience methods for plotting graphs within the cityseer API context.

Custom behaviour can be achieved by directly manipulating the underlying [`NetworkX`](https://networkx.github.io) and
[`matplotlib`](https://matplotlib.org) figures. This module is predominately used for basic plots or visual verification
of behaviour in code tests. Users are encouraged to use matplotlib or other plotting packages directly where possible.
See the demos section for examples.
"""
from __future__ import annotations

import logging
from typing import Any, Optional, Union

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt
from matplotlib import colors
from matplotlib.collections import LineCollection
from shapely import geometry
from sklearn.preprocessing import LabelEncoder

from cityseer import structures
from cityseer.tools.graphs import NodeData, NodeKey

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


plt.tight_layout()

# type hack until networkx supports type-hinting
MultiGraph = Any


class ColourMap:  # pylint: disable=too-few-public-methods
    """Specifies global colour presets."""

    primary: str = "#0091ea"
    accent: str = "#64c1ff"
    info: str = "#0064b7"
    secondary: str = "#d32f2f"
    warning: str = "#9a0007"
    error: str = "#ffffff"
    background: str = "#19181B"


COLOUR_MAP = ColourMap()

ColourType = Union[str, npt.NDArray[np.float_], npt.NDArray[np.float_]]


def _open_plots_reset():
    plt.close("all")


def plot_nx_primal_or_dual(  # noqa
    primal_graph: Optional[MultiGraph] = None,
    dual_graph: Optional[MultiGraph] = None,
    path: Optional[str] = None,
    labels: bool = False,
    primal_node_size: int = 30,
    primal_node_colour: Optional[ColourType] = None,
    primal_edge_colour: Optional[ColourType] = None,
    dual_node_size: int = 30,
    dual_node_colour: Optional[ColourType] = None,
    dual_edge_colour: Optional[ColourType] = None,
    primal_edge_width: Optional[Union[int, float]] = None,
    dual_edge_width: Optional[Union[int, float]] = None,
    plot_geoms: bool = True,
    x_lim: Optional[Union[tuple[float, float], list[float]]] = None,
    y_lim: Optional[Union[tuple[float, float], list[float]]] = None,
    ax: Optional[plt.Axes] = None,
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
        Primal edge colour as a `matplotlib` compatible colour string. Defaults to None.
    dual_node_size:  int
        The diameter for the dual graph's nodes.
    dual_node_colour: str | float | ndarray
        Dual node colour or colours. When passing a list of colours, the number of colours should match the order and
        number of nodes in the MultiGraph. The colours are passed to the underlying
        [`draw_networkx`](https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.drawing.nx_pylab.draw_networkx.html#draw-networkx)  # pylint: disable=line-too-long
        method and should be formatted accordingly. Defaults to None.
    dual_edge_colour: str | float | ndarray
        Dual edge colour as a `matplotlib` compatible colour string. Defaults to None.
    primal_edge_width: float
        Linewidths for the primal edge. Defaults to None.
    dual_edge_width: float
        Linewidths for the dual edge. Defaults to None.
    plot_geoms: bool
        Whether to plot the edge geometries. If set to `False`, straight lines will be drawn from node-to-node to
        represent edges. Defaults to True.
    x_lim: tuple[float, float]
        A tuple or list with the minimum and maxium `x` extents to be plotted.
        Defaults to None.
    y_lim: tuple[float, float]
        A tuple or list with the minimum and maxium `y` extents to be plotted.
        Defaults to None.
    ax: plt.Axes
        An optional `matplotlib` `ax` to which to plot. If not provided, a figure and ax will be generated.
    kwargs
        `kwargs` which will be passed to the `matplotlib` figure parameters. If
        provided, these will override the default figure size or dpi parameters.

    Examples
    --------
    Plot either or both primal and dual representations of a `networkX MultiGraph`. Only call this function directly if
    explicitly printing both primal and dual graphs. Otherwise, use the simplified [`plot_nx`](/tools/plot/#plot-nx)
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
        _node_colour: Optional[ColourType],
        _node_shape: str,
        _edge_colour: Optional[ColourType],
        _edge_style: str,
        _edge_width: Optional[Union[int, float]],
    ) -> None:
        if not len(_graph):  # pylint: disable=len-as-condition
            raise ValueError("Graph contains no nodes to plot.")
        if not isinstance(_node_size, int) or _node_size < 1:
            raise ValueError("Node sizes should be a positive integer.")
        if _node_colour is not None:
            if isinstance(_node_colour, np.ndarray) and len(_node_colour) != len(_graph):
                raise ValueError("A list, tuple, or array of colours should match the number of nodes in the graph.")
        else:
            if _is_primal:
                _node_colour = COLOUR_MAP.secondary
            else:
                _node_colour = COLOUR_MAP.info
        if _edge_colour is not None:
            if not isinstance(_edge_colour, str):
                raise ValueError("Edge colours should be a string representing a single colour.")
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
        colour_list: list[ColourType] = []
        node_key: NodeKey
        node_data: NodeData
        for n_idx, (node_key, node_data) in enumerate(_graph.nodes(data=True)):
            x: float = node_data["x"]
            y: float = node_data["y"]
            # add to the pos dictionary regardless (otherwise nx.draw throws an error)
            pos[node_key] = (node_data["x"], node_data["y"])
            # filter out nodes not within the extnets
            if x_lim and (x < x_lim[0] or x > x_lim[1]):
                continue
            if y_lim and (y < y_lim[0] or y > y_lim[1]):
                continue
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
        edge_list = []
        edge_geoms = []
        start_node_key: NodeKey
        end_node_key: NodeKey
        node_data: NodeData
        for start_node_key, end_node_key, node_data in _graph.edges(data=True):  # type: ignore
            # filter out if start and end nodes are not in the active node list
            if start_node_key not in node_list and end_node_key not in node_list:
                continue
            if plot_geoms:
                try:
                    x_arr: npt.NDArray[np.float_]
                    y_arr: npt.NDArray[np.float_]
                    x_arr, y_arr = node_data["geom"].coords.xy
                except KeyError as err:
                    raise KeyError(
                        f"Can't plot geoms because a 'geom' key can't be found for edge {start_node_key} to "
                        f"{end_node_key}. Use the nx_simple_geoms() method if you need to create geoms for a graph."
                    ) from err
                edge_geoms.append(tuple(zip(x_arr, y_arr)))
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
            target_ax.add_collection(lines)
        # go ahead and plot: note that edge_list will be empty if plotting geoms
        nx.draw(
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
        if path is not None:
            plt.savefig(path, facecolor=COLOUR_MAP.background)
        else:
            plt.gcf().set_facecolor(COLOUR_MAP.background)
            plt.show()


def plot_nx(
    nx_multigraph: MultiGraph,
    path: Optional[str] = None,
    labels: bool = False,
    node_size: int = 20,
    node_colour: Optional[ColourType] = None,
    edge_colour: Optional[ColourType] = None,
    edge_width: Optional[Union[int, float]] = None,
    plot_geoms: bool = False,
    x_lim: Optional[Union[tuple[float, float], list[float]]] = None,
    y_lim: Optional[Union[tuple[float, float], list[float]]] = None,
    ax: Optional[plt.Axes] = None,
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
    ```py
    from cityseer.tools import mock, graphs, plot
    from cityseer.metrics import networks
    from matplotlib import colors

    # generate a MultiGraph and compute gravity
    G = mock.mock_graph()
    G = graphs.nx_simple_geoms(G)
    G = graphs.nx_decompose(G, 50)
    nodes_gdf, network_structure = graphs.network_structure_from_nx(G, crs=3395)
    networks.node_centrality(
        measures=["node_beta"], network_structure=network_structure, nodes_gdf=nodes_gdf, distances=[800]
    )
    G_after = graphs.nx_from_network_structure(nodes_gdf, network_structure, G)
    # let's extract and normalise the values
    vals = []
    for node, data in G_after.nodes(data=True):
        vals.append(data["cc_metric_node_beta_800"])
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
        **kwargs,
    )


def plot_assignment(  # noqa
    network_structure: structures.NetworkStructure,
    nx_multigraph: MultiGraph,
    data_gdf: gpd.GeoDataFrame,
    path: Optional[str] = None,
    node_colour: Optional[ColourType] = None,
    node_labels: bool = False,
    data_labels: Optional[Union[npt.NDArray[np.int_], npt.NDArray[np.unicode_]]] = None,
    **kwargs: dict[str, Any],
):
    """
    Plot a `network_structure` and `data_gdf` for visualising assignment of data points to respective nodes.

    :::warning
    This method is primarily intended for package testing and development.
    :::

    Parameters
    ----------
    network_structure: structures.NetworkStructure
        A [`structures.NetworkStructure`](/structures/#networkstructure) instance.
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
    plt.figure(**kwargs)

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
        classes_int: npt.NDArray[np.int_] = lab_enc.transform(data_labels)  # type: ignore
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
    for _data_key, data_row in data_gdf.iterrows():  # type: ignore
        # if the data points have been assigned network indices
        data_x: float = data_row.geometry.x
        data_y: float = data_row.geometry.y
        nearest_netw_idx: int = data_row.nearest_assign
        next_nearest_netw_idx: int = data_row.next_nearest_assign
        if nearest_netw_idx != -1:
            # plot lines to parents for easier viz
            p_x = network_structure.nodes.xs[nearest_netw_idx]
            p_y = network_structure.nodes.ys[nearest_netw_idx]
            plt.plot(
                [p_x, data_x],
                [p_y, data_y],
                c="#64c1ff",
                lw=0.5,
                ls="--",
            )
        if next_nearest_netw_idx != -1:
            p_x = network_structure.nodes.xs[next_nearest_netw_idx]
            p_y = network_structure.nodes.ys[next_nearest_netw_idx]
            plt.plot([p_x, data_x], [p_y, data_y], c="#888888", lw=0.5, ls="--")

    if path:
        plt.savefig(path, facecolor=COLOUR_MAP.background, dpi=150)
    else:
        plt.gcf().set_facecolor(COLOUR_MAP.background)
        plt.show()


def plot_network_structure(
    network_structure: structures.NetworkStructure,
    data_gdf: gpd.GeoDataFrame,
    poly: Optional[geometry.Polygon] = None,
):
    """
    Plot a graph from raw `cityseer` data structures.

    :::note
    Note that this function is subject to frequent revision pending short-term development requirements. It is used
    mainly to visually confirm the correct behaviour of particular algorithms during the software development cycle.
    :::

    Parameters
    ----------
    network_structure: structures.NetworkStructure
        A [`structures.NetworkStructure`](/structures/#networkstructure) instance.
    data_gdf: GeoDataFrame
        A `data_gdf` `GeoDataFrame` with `nearest_assigned` and `next_neareset_assign` columns.
    poly: geometry.Polygon
        An optional polygon. Defaults to None.

    """
    _open_plots_reset()
    # the edges are bi-directional - therefore duplicated per directional from-to edge
    # use two axes to check each copy of edges
    _fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 12))  # type: ignore
    # set extents
    for ax in (ax1, ax2):  # type: ignore
        ax.set_xlim(network_structure.nodes.xs.min() - 100, network_structure.nodes.xs.max() + 100)
        ax.set_ylim(network_structure.nodes.ys.min() - 100, network_structure.nodes.ys.max() + 100)
    if poly:
        x = list(poly.exterior.coords.xy[0])  # type: ignore
        y = list(poly.exterior.coords.xy[1])  # type: ignore
        ax1.plot(x, y)
        ax2.plot(x, y)
    # plot nodes
    cols = []
    for is_live in network_structure.nodes.live:
        if is_live:
            cols.append(COLOUR_MAP.accent)
        else:
            cols.append(COLOUR_MAP.secondary)
    ax1.scatter(
        network_structure.nodes.xs, network_structure.nodes.ys, s=30, c=COLOUR_MAP.primary, edgecolor=cols, lw=0.5
    )
    ax2.scatter(
        network_structure.nodes.xs, network_structure.nodes.ys, s=30, c=COLOUR_MAP.primary, edgecolor=cols, lw=0.5
    )
    # plot edges
    processed_edges: set[str] = set()
    for edge_idx in range(network_structure.edges.count):
        start_idx = network_structure.edges.start[edge_idx]
        end_idx = network_structure.edges.end[edge_idx]
        keys = sorted([start_idx, end_idx])
        se_key = "-".join([str(k) for k in keys])
        # bool indicating whether second copy in opposite direction
        dupe = se_key in processed_edges
        processed_edges.add(se_key)
        # get the start and end coords - this is to check that still within src edge - neighbour range
        s_x, s_y = network_structure.nodes.xs[start_idx], network_structure.nodes.ys[start_idx]
        e_x, e_y = network_structure.nodes.xs[end_idx], network_structure.nodes.ys[end_idx]
        if not dupe:
            ax1.plot([s_x, e_x], [s_y, e_y], c=COLOUR_MAP.accent, linewidth=1)
        else:
            ax2.plot([s_x, e_x], [s_y, e_y], c=COLOUR_MAP.accent, linewidth=1)
    for node_idx in range(network_structure.nodes.count):
        ax2.annotate(node_idx, xy=network_structure.nodes.x_y(node_idx), size=5)
    if data_gdf is not None:
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
        for data_idx, data_row in data_gdf.iterrows():  # type: ignore
            data_x: float = data_row.geometry.x
            data_y: float = data_row.geometry.y
            nearest_netw_idx: int = data_row.nearest_assign
            next_nearest_netw_idx: int = data_row.next_nearest_assign
            ax2.annotate(str(data_idx), xy=(data_x, data_y), size=8, color="red")
            # if the data points have been assigned network indices
            if nearest_netw_idx != -1:
                # plot lines to parents for easier viz
                p_x, p_y = network_structure.nodes.x_y(nearest_netw_idx)
                ax1.plot([p_x, data_x], [p_y, data_y], c=COLOUR_MAP.warning, lw=0.75, ls="-")
            if next_nearest_netw_idx != -1:
                p_x, p_y = network_structure.nodes.x_y(next_nearest_netw_idx)
                ax1.plot([p_x, data_x], [p_y, data_y], c=COLOUR_MAP.info, lw=0.75, ls="--")
    plt.gcf().set_facecolor(COLOUR_MAP.background)
    plt.show()
