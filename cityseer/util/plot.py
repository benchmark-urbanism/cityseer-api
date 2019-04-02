'''
These plot methods are mainly for testing and debugging
'''
import matplotlib.pyplot as plt
from matplotlib import colors
import networkx as nx
import numpy as np
from shapely import geometry
from cityseer.metrics import layers

primary = '#0091ea'
accent = '#64c1ff'
info = '#0064b7'
secondary = '#d32f2f'
warning = '#9a0007'
error = '#ffffff'
success = '#2e7d32'
background = '#2e2e2e'


def plot_nX_primal_or_dual(primal: nx.Graph = None,
                           dual: nx.Graph = None,
                           path: str = None,
                           labels: bool = False,
                           primal_colour: (tuple, list, np.ndarray) = None,
                           dual_colour: (tuple, list, np.ndarray) = None,
                           **kwargs):

    plt.figure(**kwargs)

    alpha = 0.75
    node_size = 30
    if labels:
        alpha = 0.95
        node_size = 70

    if primal is not None:

        pos_primal = {}
        for n, d in primal.nodes(data=True):
            pos_primal[n] = (d['x'], d['y'])

        if primal_colour is not None:
            if not (len(primal_colour) == 1 or len(primal_colour) == len(primal)):
                raise ValueError('Node colours should either be a single colour or a list or tuple of colours matching '
                                 'the number of nodes in the graph.')
            node_colour = primal_colour
        else:
            node_colour = secondary

        nx.draw(primal, pos_primal,
                with_labels=labels,
                font_size=5,
                font_color='w',
                font_weight='bold',
                node_color=node_colour,
                node_size=node_size,
                node_shape='o',
                edge_color='w',
                width=1,
                alpha=alpha)

    if dual is not None:

        pos_dual = {}
        for n, d in dual.nodes(data=True):
            pos_dual[n] = (d['x'], d['y'])

        if dual_colour is not None:
            if not (len(dual_colour) == 1 or len(dual_colour) == len(dual)):
                raise ValueError('Node colours should either be a single colour or a list or tuple of colours matching '
                                 'the number of nodes in the graph.')
            node_colour = dual_colour
        else:
            node_colour = info

        nx.draw(dual, pos_dual,
                with_labels=labels,
                font_size=5,
                font_color='w',
                font_weight='bold',
                node_color=node_colour,
                node_size=node_size,
                node_shape='d',
                edge_color='#999999',
                style='dashed',
                width=1,
                alpha=alpha)

    if path:
        plt.savefig(path, facecolor=background)
    else:
        plt.gcf().set_facecolor(background)
        plt.show()


def plot_nX(networkX_graph: nx.Graph, path: str = None, labels: bool = False, colour: (list, tuple, np.ndarray) = None, **kwargs):
    return plot_nX_primal_or_dual(primal=networkX_graph, path=path, labels=labels, primal_colour=colour, **kwargs)


def plot_assignment(Network_Layer,
                    Data_Layer,
                    path: str = None,
                    node_colour: (list, tuple, np.ndarray) = None,
                    node_labels: bool = False,
                    data_labels: (list, tuple, np.ndarray) = None):

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
        d_classes, d_encodings = layers.encode_categorical(data_labels)
        data_colour = colors.Normalize()(d_encodings)
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
            p_x = Network_Layer._nodes[int(nearest_netw_idx)][0]
            p_y = Network_Layer._nodes[int(nearest_netw_idx)][1]
            plt.plot([p_x, x], [p_y, y], c='#64c1ff', lw=0.5, ls='--')

        if not np.isnan(next_n_netw_idx):
            p_x = Network_Layer._nodes[int(next_n_netw_idx)][0]
            p_y = Network_Layer._nodes[int(next_n_netw_idx)][1]
            plt.plot([p_x, x], [p_y, y], c='#888888', lw=0.5, ls='--')

    if path:
        plt.savefig(path, facecolor=background, dpi=150)
    else:
        plt.gcf().set_facecolor(background)
        plt.show()


def plot_graph_maps(node_uids: [list, tuple, np.ndarray],
                    node_map: np.ndarray,
                    edge_map: np.ndarray,
                    data_map: np.ndarray = None,
                    poly: geometry.Polygon = None):
    # the edges are bi-directional - therefore duplicated per directional from-to edge
    # use two axes to check each copy of edges
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 10))

    # set extents
    for ax in (ax1, ax2):
        ax.set_xlim(node_map[:, 0].min() - 100, node_map[:, 0].max() + 100)
        ax.set_ylim(node_map[:, 1].min() - 100, node_map[:, 1].max() + 100)

    if poly:
        x = [x for x in poly.exterior.coords.xy[0]]
        y = [y for y in poly.exterior.coords.xy[1]]
        ax1.plot(x, y)
        ax2.plot(x, y)

    # plot nodes
    cols = []
    for n in node_map[:, 2]:
        if bool(n):
            cols.append(secondary)
        else:
            cols.append(accent)
    ax1.scatter(node_map[:, 0], node_map[:, 1], s=30, c=primary, edgecolor=cols, lw=0.5)
    ax2.scatter(node_map[:, 0], node_map[:, 1], s=30, c=primary, edgecolor=cols, lw=0.5)

    # check for duplicate edges
    edges = set()

    # plot edges - requires iteration through maps
    for src_idx, src_data in enumerate(node_map):
        # get the starting edge index
        # isolated nodes don't have edges
        edge_idx = src_data[3]
        if np.isnan(edge_idx):
            continue
        edge_idx = int(edge_idx)
        # iterate the neighbours
        # don't use while True because last node's index increment won't be caught
        while edge_idx < len(edge_map):
            # get the corresponding edge data
            edge_data = edge_map[edge_idx]
            # get the src node - this is to check that still within src edge - neighbour range
            fr_idx = edge_data[0]
            # break once all neighbours visited
            if fr_idx != src_idx:
                break
            # get the neighbour node's index
            to_idx = edge_data[1]
            # fetch the neighbour node's data
            nb_data = node_map[int(to_idx)]
            # check for duplicates
            k = str(sorted([fr_idx, to_idx]))
            if k not in edges:
                edges.add(k)
                ax1.plot([src_data[0], nb_data[0]], [src_data[1], nb_data[1]], c=accent, linewidth=1)
            else:
                ax2.plot([src_data[0], nb_data[0]], [src_data[1], nb_data[1]], c=accent, linewidth=1)
            edge_idx += 1

    for idx, (x, y) in enumerate(zip(node_map[:, 0], node_map[:, 1])):
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
                p_x = node_map[int(nearest_netw_idx)][0]
                p_y = node_map[int(nearest_netw_idx)][1]
                ax1.plot([p_x, x], [p_y, y], c=warning, lw=0.75, ls='-')

            if not np.isnan(next_n_netw_idx):
                p_x = node_map[int(next_n_netw_idx)][0]
                p_y = node_map[int(next_n_netw_idx)][1]
                ax1.plot([p_x, x], [p_y, y], c=info, lw=0.75, ls='--')

    plt.gcf().set_facecolor(background)
    plt.show()
