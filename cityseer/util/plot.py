'''
These plot methods are mainly for testing and debugging
'''
from matplotlib import colors
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from shapely import geometry
from sklearn.preprocessing import LabelEncoder

primary = '#0091ea'
accent = '#64c1ff'
info = '#0064b7'
secondary = '#d32f2f'
warning = '#9a0007'
error = '#ffffff'
success = '#2e7d32'
background = '#2e2e2e'


def plot_nX_primal_or_dual(primal_graph: nx.Graph = None,
                           dual_graph: nx.Graph = None,
                           path: str = None,
                           labels: bool = False,
                           primal_node_colour: (str, tuple, list) = None,
                           primal_edge_colour: (str, tuple, list) = None,
                           dual_node_colour: (str, tuple, list) = None,
                           dual_edge_colour: (str, tuple, list) = None,
                           primal_edge_width: (int, float) = None,
                           dual_edge_width: (int, float) = None,
                           plot_geoms: bool = False,
                           x_lim: (tuple, list) = None,
                           y_lim: (tuple, list) = None,
                           **kwargs):
    # cleanup old plots
    plt.ioff()
    plt.close('all')
    plt.cla()
    plt.clf()
    # create new plot
    fig, ax = plt.subplots(1, 1, **kwargs)
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
                x_arr, y_arr = data['geom'].coords.xy
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


def plot_nX(networkX_graph: nx.Graph,
            path: str = None,
            labels: bool = False,
            node_colour: (str, tuple, list) = None,
            edge_colour: (str, tuple, list) = None,
            edge_width: (int, float) = None,
            plot_geoms: bool = False,
            x_lim: (tuple, list) = None,
            y_lim: (tuple, list) = None,
            **kwargs):
    return plot_nX_primal_or_dual(primal_graph=networkX_graph,
                                  path=path,
                                  labels=labels,
                                  primal_node_colour=node_colour,
                                  primal_edge_colour=edge_colour,
                                  primal_edge_width=edge_width,
                                  plot_geoms=plot_geoms,
                                  x_lim=x_lim,
                                  y_lim=y_lim,
                                  **kwargs)


def plot_assignment(Network_Layer,
                    Data_Layer,
                    path: str = None,
                    node_colour: (list, tuple, np.ndarray) = None,
                    node_labels: bool = False,
                    data_labels: (list, tuple, np.ndarray) = None,
                    **kwargs):
    plt.figure(**kwargs)

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


def plot_graph_maps(node_uids: [list, tuple, np.ndarray],
                    node_data: np.ndarray,
                    edge_data: np.ndarray,
                    data_map: np.ndarray = None,
                    poly: geometry.Polygon = None):
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

    # check for duplicate edges
    edges = set()

    # plot edges - requires iteration through maps
    for src_idx, src_data in enumerate(node_data):
        # get the starting edge index
        # isolated nodes don't have edges
        edge_idx = src_data[3]
        if np.isnan(edge_idx):
            continue
        edge_idx = int(edge_idx)
        # iterate the neighbours
        # don't use while True because last node's index increment won't be caught
        while edge_idx < len(edge_data):
            # get the corresponding edge data
            edge_data = edge_data[edge_idx]
            # get the src node - this is to check that still within src edge - neighbour range
            fr_idx = edge_data[0]
            # break once all neighbours visited
            if fr_idx != src_idx:
                break
            # get the neighbour node's index
            to_idx = edge_data[1]
            # fetch the neighbour node's data
            nb_data = node_data[int(to_idx)]
            # check for duplicates
            k = str(sorted([fr_idx, to_idx]))
            if k not in edges:
                edges.add(k)
                ax1.plot([src_data[0], nb_data[0]], [src_data[1], nb_data[1]], c=accent, linewidth=1)
            else:
                ax2.plot([src_data[0], nb_data[0]], [src_data[1], nb_data[1]], c=accent, linewidth=1)
            edge_idx += 1

    for idx, (x, y) in enumerate(zip(node_data[:, 0], node_data[:, 1])):
        ax2.annotate(idx, xy=(x, y), size=5)

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

        for i, (x, y, nearest_netw_idx, next_n_netw_idx) in \
                enumerate(zip(data_map[:, 0],
                              data_map[:, 1],
                              data_map[:, 2],
                              data_map[:, 3])):

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
