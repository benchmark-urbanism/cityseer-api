'''
These plot methods are mainly for testing and debugging
'''
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def plot_graphs(primal: nx.Graph = None, dual: nx.Graph = None):
    if primal is not None:
        pos_primal = {}
        for n, d in primal.nodes(data=True):
            pos_primal[n] = (d['x'], d['y'])
        nx.draw(primal, pos_primal,
                with_labels=True,
                font_size=8,
                node_color='y',
                node_size=100,
                node_shape='o',
                edge_color='g',
                width=1,
                alpha=0.6)

    if dual is not None:
        pos_dual = {}
        for n, d in dual.nodes(data=True):
            pos_dual[n] = (d['x'], d['y'])
        nx.draw(dual, pos_dual,
                with_labels=True,
                font_size=8,
                node_color='r',
                node_size=50,
                node_shape='d',
                edge_color='b',
                width=1,
                alpha=0.6)

    plt.show()


def plot_graph_maps(node_labels, node_map, edge_map, data_map=None, poly=None):
    # the links are bi-directional - therefore duplicated per directional from-to edge
    # use two axes to check each copy of links
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
    ax1.scatter(node_map[:, 0], node_map[:, 1], s=7, c=node_map[:, 2])
    ax2.scatter(node_map[:, 0], node_map[:, 1], s=7, c=node_map[:, 2])

    # check for duplicate edges
    edges = set()

    # plot edges - requires iteration through maps
    for src_idx, src_data in enumerate(node_map):
        # get the starting edge index
        edge_idx = int(src_data[3])
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
                ax1.plot([src_data[0], nb_data[0]], [src_data[1], nb_data[1]], c='grey', linewidth=1)
            else:
                ax2.plot([src_data[0], nb_data[0]], [src_data[1], nb_data[1]], c='grey', linewidth=1)
            edge_idx += 1

    for ax in (ax1, ax2):
        for label, x, y in zip(node_labels, node_map[:, 0], node_map[:, 1]):
            ax.annotate(label, xy=(x, y))

    '''
    DATA MAP:
    0 - x
    1 - y
    2 - live
    3 - data class
    4 - assigned network index - nearest
    5 - assigned network index - next-nearest
    '''

    if data_map is not None:

        # plot parents on ax1
        ax1.scatter(x=data_map[:, 0], y=data_map[:, 1], c=data_map[:, 3])
        ax2.scatter(x=data_map[:, 0], y=data_map[:, 1], c=data_map[:, 2])
        for idx, (x, y, cl, nearest_netw_idx, next_n_netw_idx) in \
                enumerate(zip(data_map[:, 0], data_map[:, 1], data_map[:, 3], data_map[:, 4], data_map[:, 5])):

            ax1.annotate('cl:' + str(int(cl)), xy=(x, y), size=8, color='red')
            ax2.annotate('idx:' + str(int(idx)), xy=(x, y), size=8, color='red')

            # if the data points have been assigned network indices:
            if not np.isnan(nearest_netw_idx):
                # plot lines to parents for easier viz
                p_x = node_map[int(nearest_netw_idx)][0]
                p_y = node_map[int(nearest_netw_idx)][1]
                ax1.plot([p_x, x], [p_y, y], c='blue', linewidth=0.5)

                # ax-2
                p_x = node_map[int(next_n_netw_idx)][0]
                p_y = node_map[int(next_n_netw_idx)][1]
                ax2.plot([p_x, x], [p_y, y], c='blue', linewidth=0.5)

    plt.show()
