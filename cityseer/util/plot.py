'''
These plot methods are mainly for testing and debugging
'''
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from shapely import geometry


def plot_nX_primal_or_dual(primal: nx.Graph = None,
                           dual: nx.Graph = None,
                           path: str = None,
                           labels: bool = False):
    if primal is not None:
        pos_primal = {}
        for n, d in primal.nodes(data=True):
            pos_primal[n] = (d['x'], d['y'])
        nx.draw(primal, pos_primal,
                with_labels=labels,
                font_size=5,
                font_color='w',
                font_weight='bold',
                node_color='#d32f2f',
                node_size=75,
                node_shape='o',
                edge_color='w',
                width=1,
                alpha=0.95)

    if dual is not None:
        pos_dual = {}
        for n, d in dual.nodes(data=True):
            pos_dual[n] = (d['x'], d['y'])
        nx.draw(dual, pos_dual,
                with_labels=labels,
                font_size=5,
                node_color='#0064b7',
                node_size=75,
                node_shape='d',
                edge_color='w',
                width=1,
                alpha=0.95)

    if path:
        plt.savefig(path, facecolor='#2e2e2e', dpi=150)
    else:
        plt.gcf().set_facecolor("#2e2e2e")
        plt.show(facecolor='#2e2e2e')


def plot_nX(networkX_graph: nx.Graph, path: str = None, labels: bool = False):
    return plot_nX_primal_or_dual(primal=networkX_graph, path=path, labels=labels)


def plot_graph_maps(node_uids: [list, tuple, np.ndarray],
                    node_map: np.ndarray,
                    edge_map: np.ndarray,
                    data_map: np.ndarray = None,
                    poly: geometry.Polygon = None):
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
                ax1.plot([src_data[0], nb_data[0]], [src_data[1], nb_data[1]], c='grey', linewidth=1)
            else:
                ax2.plot([src_data[0], nb_data[0]], [src_data[1], nb_data[1]], c='grey', linewidth=1)
            edge_idx += 1

    for ax in (ax1, ax2):
        for label, x, y in zip(node_uids, node_map[:, 0], node_map[:, 1]):
            ax.annotate(label, xy=(x, y))

    '''
    DATA MAP:
    0 - x
    1 - y
    2 - live
    3 - assigned network index - nearest
    4 - assigned network index - next-nearest
    '''

    if data_map is not None:

        # plot parents on ax1
        ax1.scatter(x=data_map[:, 0], y=data_map[:, 1], c=data_map[:, 3])
        ax2.scatter(x=data_map[:, 0], y=data_map[:, 1], c=data_map[:, 2])

        cm = plt.get_cmap('hsv')
        cols = [cm(i / len(data_map)) for i in range(len(data_map))]
        for i, (x, y, nearest_netw_idx, next_n_netw_idx) in \
                enumerate(zip(data_map[:, 0],
                              data_map[:, 1],
                              data_map[:, 3],
                              data_map[:, 4])):

            ax2.annotate(str(int(i)), xy=(x, y), size=8, color='red')

            # if the data points have been assigned network indices
            if not np.isnan(nearest_netw_idx):
                # plot lines to parents for easier viz
                p_x = node_map[int(nearest_netw_idx)][0]
                p_y = node_map[int(nearest_netw_idx)][1]
                ax1.plot([p_x, x], [p_y, y], c=cols[i], linewidth=0.5)

            if not np.isnan(next_n_netw_idx):
                p_x = node_map[int(next_n_netw_idx)][0]
                p_y = node_map[int(next_n_netw_idx)][1]
                ax1.plot([p_x, x], [p_y, y], c=cols[i], linewidth=0.5)

    plt.show()
