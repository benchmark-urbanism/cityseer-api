"""
Convenience methods for plotting graphs within the cityseer API context. This module is predominately used for basic
plots or visual verification of behaviour in code tests. Custom behaviour can be achieved by directly manipulating the
underlying [`NetworkX`](https://networkx.github.io) and [`matplotlib`](https://matplotlib.org) figures.
"""
import logging
from typing import Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import colors
from matplotlib.collections import LineCollection
from shapely import geometry
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

primary = '#0091ea'
accent = '#64c1ff'
info = '#0064b7'
secondary = '#d32f2f'
warning = '#9a0007'
error = '#ffffff'
success = '#2e7d32'
background = '#2e2e2e'


def plot_nX_primal_or_dual(primal_graph: nx.MultiGraph = None,
                           dual_graph: nx.MultiGraph = None,
                           path: str = None,
                           labels: bool = False,
                           primal_node_colour: Union[str, tuple, list] = None,
                           primal_edge_colour: str = None,
                           dual_node_colour: Union[str, tuple, list] = None,
                           dual_edge_colour: str = None,
                           primal_edge_width: Union[int, float] = None,
                           dual_edge_width: Union[int, float] = None,
                           plot_geoms: bool = True,
                           x_lim: Union[tuple, list] = None,
                           y_lim: Union[tuple, list] = None,
                           **figure_kwargs):
    """
    Parameters
    ----------
    primal_graph
        An optional `NetworkX` MultiGraph to plot in the primal representation. Defaults to None.
    dual_graph
        An optional `NetworkX` MultiGraph to plot in the dual representation. Defaults to None.
    path
        An optional filepath: if provided, the image will be saved to the path instead of being displayed. Defaults to
        None.
    labels
        Whether to display node labels. Defaults to False.
    primal_node_colour
        Primal node colour or colours. When passing an iterable of colours, the number of colours should match the order
        and number of nodes in the MultiGraph. The colours are passed to the underlying [`draw_networkx`](https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.drawing.nx_pylab.draw_networkx.html#draw-networkx)
        method and should be formatted accordingly. Defaults to None.
    primal_edge_colour
        Primal edge colour as a `matplotlib` compatible colour string. Defaults to None.
    dual_node_colour
        Dual node colour or colours. When passing a list of colours, the number of colours should match the order and
        number of nodes in the MultiGraph. The colours are passed to the underlying [`draw_networkx`](https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.drawing.nx_pylab.draw_networkx.html#draw-networkx)
        method and should be formatted accordingly. Defaults to None.
    dual_edge_colour
        Dual edge colour as a `matplotlib` compatible colour string. Defaults to None.
    primal_edge_width
        Linewidths for the primal edge. Defaults to None.
    dual_edge_width
        Linewidths for the dual edge. Defaults to None.
    plot_geoms
        Whether to plot the edge geometries. If set to `False`, straight lines will be drawn from node-to-node to
        represent edges. Defaults to True.
    x_lim
        A tuple or list with the minimum and maxium `x` extents to be plotted.
        Defaults to None.
    y_lim
        A tuple or list with the minimum and maxium `y` extents to be plotted.
        Defaults to None.
    figure_kwargs
        `kwargs` which will be passed to the `matplotlib` figure parameters. If
        provided, these will override the default figure size or dpi parameters.

    Notes
    -----
    Plot either or both primal and dual representations of a `networkX MultiGraph`. Only call this function directly if
    explicitly printing both primal and dual graphs. Otherwise, use the simplified [`plot_nX`](/tools/plot/#plot_nx) method
    instead.

    ```py
    from cityseer.tools import mock, graphs, plot
    G = mock.mock_graph()
    G_simple = graphs.nX_simple_geoms(G)
    G_dual = graphs.nX_to_dual(G_simple)
    plot.plot_nX_primal_or_dual(G_simple,
                                G_dual,
                                plot_geoms=False)
    ```
    ![Example primal and dual graph plot.](../../src/assets/plots/images/graph_dual.png)
    _A dual graph in blue overlaid on the source primal graph in red._
    """
    # cleanup old plots
    plt.ioff()
    plt.close('all')
    plt.cla()
    plt.clf()
    # create new plot
    fig, ax = plt.subplots(1, 1, **figure_kwargs)
    # setup params
    alpha = 0.75
    node_size = 30
    if labels:
        alpha = 0.95
        node_size = 70

    # setup a function that can be used for either the primal or dual graph
    def plot_graph(_graph, _is_primal, _node_colour, _node_shape, _edge_colour, _edge_style, _edge_width):
        if not len(_graph.nodes()):
            raise ValueError('Graph contains no nodes to plot.')
        iterables = (list, tuple, np.ndarray)
        if _node_colour is not None:
            if isinstance(_node_colour, iterables) and len(_node_colour) != len(_graph):
                raise ValueError('A list, tuple, or array of colours should match the number of nodes in the graph.')
        else:
            if _is_primal:
                _node_colour = secondary
            else:
                _node_colour = info
        if _edge_colour is not None:
            if not isinstance(_edge_colour, str):
                raise ValueError('Edge colours should be a string representing a single colour.')
        else:
            if _is_primal:
                _edge_colour = 'w'
            else:
                _edge_colour = accent
        # edge width
        if _edge_width is not None:
            if not isinstance(_edge_width, (int, float)):
                raise ValueError('Edge widths should be an int or float.')
        else:
            if _is_primal:
                _edge_width = 1.5
            else:
                _edge_width = 0.5
        pos = {}
        node_list = []
        # setup a node colour list if nodes are individually coloured, this is for filtering by extents
        colour_list = []
        for n_idx, (n, d) in enumerate(_graph.nodes(data=True)):
            x = d['x']
            y = d['y']
            # add to the pos dictionary regardless (otherwise nx.draw throws an error)
            pos[n] = (d['x'], d['y'])
            # filter out nodes not within the extnets
            if x_lim and (x < x_lim[0] or x > x_lim[1]):
                continue
            if y_lim and (y < y_lim[0] or y > y_lim[1]):
                continue
            # add to the node list
            node_list.append(n)
            # and add to the node colour list if node colours are a list or tuple
            if isinstance(_node_colour, iterables):
                colour_list.append(_node_colour[n_idx])
        if not len(node_list):
            raise ValueError('All nodes have been filtered out by the x_lim / y_lim parameter: check your extents')
        # update the node colours to the filtered list of colours if necessary
        if isinstance(_node_colour, iterables):
            _node_colour = colour_list
        # plot edges manually for geoms
        edge_list = []
        edge_geoms = []
        for s, e, data in _graph.edges(data=True):
            # filter out if start and end nodes are not in the active node list
            if s not in node_list and e not in node_list:
                continue
            if plot_geoms:
                try:
                    x_arr, y_arr = data['geom'].coords.xy
                except KeyError:
                    raise KeyError(f"Can't plot geoms because a 'geom' key can't be found for edge {s} to {e}. "
                                   f"Use the nX_simple_geoms() method if you need to create geoms for a graph.")
                edge_geoms.append(tuple(zip(x_arr, y_arr)))
            else:
                edge_list.append((s, e))
        # plot geoms manually if required
        if plot_geoms:
            lines = LineCollection(edge_geoms, colors=_edge_colour, linewidths=_edge_width, linestyles=_edge_style)
            ax.add_collection(lines)
        # go ahead and plot: note that edge_list will be empty if plotting geoms
        nx.draw(_graph,
                pos=pos,
                ax=ax,
                with_labels=labels,
                font_size=5,
                font_color='w',
                font_weight='bold',
                nodelist=node_list,
                node_color=_node_colour,
                node_size=node_size,
                node_shape=_node_shape,
                edgelist=edge_list,
                edge_color=_edge_colour,
                style=_edge_style,
                width=_edge_width,
                alpha=alpha)

    if primal_graph is not None:
        plot_graph(primal_graph, True, primal_node_colour, 'o', primal_edge_colour, 'solid', primal_edge_width)
    if dual_graph is not None:
        plot_graph(dual_graph, False, dual_node_colour, 'd', dual_edge_colour, 'dashed', dual_edge_width)
    if x_lim:
        plt.xlim(*x_lim)
    if y_lim:
        plt.ylim(*y_lim)
    if path:
        plt.savefig(path, facecolor=background)
    else:
        plt.gcf().set_facecolor(background)
        plt.show()


def plot_nX(networkX_graph: nx.MultiGraph,
            path: str = None,
            labels: bool = False,
            node_colour: Union[str, tuple, list] = None,
            edge_colour: Union[str, tuple, list] = None,
            edge_width: Union[int, float] = None,
            plot_geoms: bool = False,
            x_lim: Union[tuple, list] = None,
            y_lim: Union[tuple, list] = None,
            **figure_kwargs):
    """
    Plot a `networkX` MultiGraph.

    Parameters
    ----------
    networkX_graph
        A `NetworkX` MultiGraph.
    path
        An optional filepath: if provided, the image will be saved to the path instead of being displayed. Defaults to
        None.
    labels
        Whether to display node labels. Defaults to False.
    node_colour
        Node colour or colours. When passing an iterable of colours, the number of colours should match the order and
        number of nodes in the MultiGraph. The colours are passed to the underlying [`draw_networkx`](https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.drawing.nx_pylab.draw_networkx.html#draw-networkx)
        method and should be formatted accordingly. Defaults to None.
    edge_colour
        Edges colour as a `matplotlib` compatible colour string. Defaults to None.
    edge_width
        Linewidths for edges. Defaults to None.
    plot_geoms
        Whether to plot the edge geometries. If set to `False`, straight lines will be drawn from node-to-node to
        represent edges. Defaults to True.
    x_lim
        A tuple or list with the minimum and maxium `x` extents to be plotted. Defaults to None.
    y_lim
        A tuple or list with the minimum and maxium `y` extents to be plotted. Defaults to None.
    figure_kwargs
        `kwargs` which will be passed to the `matplotlib` figure parameters. If provided, these will override the
        default figure size or dpi parameters.

    Notes
    -----
    ```py
    from cityseer.tools import mock, graphs, plot
    from cityseer.metrics import networks
    from matplotlib import colors
    # generate a MultiGraph and compute gravity
    G = mock.mock_graph()
    G = graphs.nX_simple_geoms(G)
    G = graphs.nX_decompose(G, 50)
    N = networks.NetworkLayerFromNX(G, distances=[800])
    N.node_centrality(measures=['node_beta'])
    G_after = N.to_networkX()
    # let's extract and normalise the values
    vals = []
    for node, data in G_after.nodes(data=True):
        vals.append(data['metrics']['centrality']['node_beta'][800])
    # let's create a custom colourmap using matplotlib
    cmap = colors.LinearSegmentedColormap.from_list('cityseer',
                                                    [(100/255, 193/255, 255/255, 255/255),
                                                    (211/255, 47/255, 47/255, 1/255)])
    # normalise the values
    vals = colors.Normalize()(vals)
    # cast against the colour map
    cols = cmap(vals)
    # plot
    plot.plot_nX(G_after, node_colour=cols)
    ```

    ![Example Colour Plot.](../../src/assets/plots/images/graph_colour.png)
    _Colour plot of 800m gravity index centrality on a 50m decomposed graph._
    """
    return plot_nX_primal_or_dual(primal_graph=networkX_graph,
                                  path=path,
                                  labels=labels,
                                  primal_node_colour=node_colour,
                                  primal_edge_colour=edge_colour,
                                  primal_edge_width=edge_width,
                                  plot_geoms=plot_geoms,
                                  x_lim=x_lim,
                                  y_lim=y_lim,
                                  **figure_kwargs)


def plot_assignment(Network_Layer,
                    Data_Layer,
                    path: str = None,
                    node_colour: Union[list, tuple, np.ndarray] = None,
                    node_labels: bool = False,
                    data_labels: Union[list, tuple, np.ndarray] = None,
                    **figure_kwargs):
    """
    Plot a `NetworkLayer` and `DataLayer` for the purpose of visualising assignment of data points to respective nodes.

    Parameters
    ----------
    Network_Layer
        A [`NetworkLayer`](/metrics/networks/#class-networklayer).
    Data_Layer
        A [`DataLayer`](/metrics/layers/#class-datalayer).
    path
        An optional filepath: if provided, the image will be saved to the path instead of being displayed. Defaults to
        None.
    node_colour
        Node colour or colours. When passing a list of colours, the number of colours should match the order and number
        of nodes in the MultiGraph. The colours are passed to the underlying [`draw_networkx`](https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.drawing.nx_pylab.draw_networkx.html#draw-networkx)
        method and should be formatted accordingly. Defaults to None.
    node_labels
        Whether to plot the node labels. Defaults to False.
    data_labels
        An optional iterable of categorical data labels which will be mapped to colours. The number of labels should
        match the number of data points in `DataLayer`. Defaults to None.
    figure_kwargs
        `kwargs` which will be passed to the `matplotlib` figure parameters. If provided, these will override the
        default figure size or dpi parameters.

    Notes
    -----
    ![Example assignment plot.](../../src/assets/plots/images/assignment_plot.png)
    _An assignment plot to a 50m decomposed graph, with the data points coloured by categorical labels._
    """
    plt.figure(**figure_kwargs)

    # extract NetworkX
    Graph = Network_Layer.to_networkX()

    if node_colour is not None:
        if not (len(node_colour) == 1 or len(node_colour) == len(Graph)):
            raise ValueError('Node colours should either be a single colour or a list or tuple of colours matching '
                             'the number of nodes in the graph.')
        node_colour = node_colour
    else:
        node_colour = secondary

    # do a simple plot - don't provide path
    pos = {}
    for n, d in Graph.nodes(data=True):
        pos[n] = (d['x'], d['y'])
    nx.draw(Graph, pos,
            with_labels=node_labels,
            font_size=5,
            font_color='w',
            font_weight='bold',
            node_color=node_colour,
            node_size=30,
            node_shape='o',
            edge_color='w',
            width=1,
            alpha=0.75)

    if data_labels is None:
        data_colour = info
        data_cmap = None
    else:
        # generate categorical colormap
        # use sklearn's label encoder
        le = LabelEncoder()
        le.fit(data_labels)
        # map the int encodings to the respective classes
        classes_int = le.transform(data_labels)
        data_colour = colors.Normalize()(classes_int)
        data_cmap = 'Dark2'  # Set1

    # overlay data map
    plt.scatter(x=Data_Layer._data[:, 0],
                y=Data_Layer._data[:, 1],
                c=data_colour,
                cmap=data_cmap,
                s=30,
                edgecolors='white',
                lw=0.5,
                alpha=0.95)

    # draw assignment
    for i, (x, y, nearest_netw_idx, next_n_netw_idx) in \
            enumerate(zip(Data_Layer._data[:, 0],
                          Data_Layer._data[:, 1],
                          Data_Layer._data[:, 2],
                          Data_Layer._data[:, 3])):

        # if the data points have been assigned network indices
        if not np.isnan(nearest_netw_idx):
            # plot lines to parents for easier viz
            p_x = Network_Layer._node_data[int(nearest_netw_idx)][0]
            p_y = Network_Layer._node_data[int(nearest_netw_idx)][1]
            plt.plot([p_x, x], [p_y, y], c='#64c1ff', lw=0.5, ls='--')

        if not np.isnan(next_n_netw_idx):
            p_x = Network_Layer._node_data[int(next_n_netw_idx)][0]
            p_y = Network_Layer._node_data[int(next_n_netw_idx)][1]
            plt.plot([p_x, x], [p_y, y], c='#888888', lw=0.5, ls='--')

    if path:
        plt.savefig(path, facecolor=background, dpi=150)
    else:
        plt.gcf().set_facecolor(background)
        plt.show()


def plot_graph_maps(node_data: np.ndarray,
                    edge_data: np.ndarray,
                    data_map: np.ndarray = None,
                    poly: geometry.Polygon = None):
    """
    Plot a graph from raw `cityseer` data structures. Note that this function is subject to frequent revision pending
    short-term development requirements. It is used mainly to visually confirm the correct behaviour of particular
    algorithms during the software development cycle.

    Parameters
    ----------
    node_data
        `cityseer` node map.
    edge_data
        `cityseer` edge map.
    data_map
        An optional data map. Defaults to None.
    poly
        An optional polygon. Defaults to None.
    """
    # the edges are bi-directional - therefore duplicated per directional from-to edge
    # use two axes to check each copy of edges
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 10))
    # set extents
    for ax in (ax1, ax2):
        ax.set_xlim(node_data[:, 0].min() - 100, node_data[:, 0].max() + 100)
        ax.set_ylim(node_data[:, 1].min() - 100, node_data[:, 1].max() + 100)
    if poly:
        x = [x for x in poly.exterior.coords.xy[0]]
        y = [y for y in poly.exterior.coords.xy[1]]
        ax1.plot(x, y)
        ax2.plot(x, y)
    # plot nodes
    cols = []
    for n in node_data[:, 2]:
        if bool(n):
            cols.append(secondary)
        else:
            cols.append(accent)
    ax1.scatter(node_data[:, 0], node_data[:, 1], s=30, c=primary, edgecolor=cols, lw=0.5)
    ax2.scatter(node_data[:, 0], node_data[:, 1], s=30, c=primary, edgecolor=cols, lw=0.5)
    # plot edges
    processed_edges = set()
    for edge_idx, edge_data in enumerate(edge_data):
        s, e = edge_data[:2]
        se_key = '-'.join(sorted([str(s), str(e)]))
        # bool indicating whether second copy in opposite direction
        dupe = se_key in processed_edges
        processed_edges.add(se_key)
        # get the start and end coords - this is to check that still within src edge - neighbour range
        s_x, s_y = node_data[int(s), :2]
        e_x, e_y = node_data[int(e), :2]
        if not dupe:
            ax1.plot([s_x, e_x], [s_y, e_y], c=accent, linewidth=1)
        else:
            ax2.plot([s_x, e_x], [s_y, e_y], c=accent, linewidth=1)
    for node_idx, n_data in enumerate(node_data):
        ax2.annotate(node_idx, xy=n_data[:2], size=5)
    '''
    DATA MAP:
    0 - x
    1 - y
    2 - assigned network index - nearest
    3 - assigned network index - next-nearest
    '''
    if data_map is not None:
        # plot parents on ax1
        ax1.scatter(x=data_map[:, 0],
                    y=data_map[:, 1],
                    color=secondary,
                    edgecolor=warning,
                    alpha=0.9,
                    lw=0.5)
        ax2.scatter(x=data_map[:, 0],
                    y=data_map[:, 1],
                    color=secondary,
                    edgecolor=warning,
                    alpha=0.9,
                    lw=0.5)
        for i, assignment_data in enumerate(data_map):
            x, y, nearest_netw_idx, next_n_netw_idx = assignment_data
            ax2.annotate(str(int(i)), xy=(x, y), size=8, color='red')
            # if the data points have been assigned network indices
            if not np.isnan(nearest_netw_idx):
                # plot lines to parents for easier viz
                p_x = node_data[int(nearest_netw_idx)][0]
                p_y = node_data[int(nearest_netw_idx)][1]
                ax1.plot([p_x, x], [p_y, y], c=warning, lw=0.75, ls='-')
            if not np.isnan(next_n_netw_idx):
                p_x = node_data[int(next_n_netw_idx)][0]
                p_y = node_data[int(next_n_netw_idx)][1]
                ax1.plot([p_x, x], [p_y, y], c=info, lw=0.75, ls='--')
    plt.gcf().set_facecolor(background)
    plt.show()
