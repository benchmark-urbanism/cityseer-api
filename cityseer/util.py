'''
Generate a graph for testing and documentation purposes.
'''
import logging
import matplotlib.pyplot as plt
import networkx as nx
import random
import utm


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def tutte_graph(wgs84_coords=False):
    '''
    Prepares a Tutte graph per https://en.wikipedia.org/wiki/Tutte_graph
    :return: NetworkX graph
    '''

    G = nx.Graph()

    nodes = [
        (0, {'x': 6000700, 'y': 600700}),
        (1, {'x': 6000610, 'y': 600780}),
        (2, {'x': 6000460, 'y': 600700}),
        (3, {'x': 6000520, 'y': 600820}),
        (4, {'x': 6000620, 'y': 600905}),
        (5, {'x': 6000260, 'y': 600700}),
        (6, {'x': 6000320, 'y': 600850}),
        (7, {'x': 6000420, 'y': 600880}),
        (8, {'x': 6000460, 'y': 600980}),
        (9, {'x': 6000580, 'y': 601030}),
        (10, {'x': 6000100, 'y': 600810}),
        (11, {'x': 6000280, 'y': 600980}),
        (12, {'x': 6000400, 'y': 601030}),
        (13, {'x': 6000460, 'y': 601130}),
        (14, {'x': 6000190, 'y': 601050}),
        (15, {'x': 6000350, 'y': 601200}),
        (16, {'x': 6000800, 'y': 600750}),
        (17, {'x': 6000800, 'y': 600920}),
        (18, {'x': 6000900, 'y': 600820}),
        (19, {'x': 6000910, 'y': 600690}),
        (20, {'x': 6000905, 'y': 601080}),
        (21, {'x': 6001000, 'y': 600870}),
        (22, {'x': 6001040, 'y': 600660}),
        (23, {'x': 6001050, 'y': 600760}),
        (24, {'x': 6001000, 'y': 600980}),
        (25, {'x': 6001130, 'y': 600950}),
        (26, {'x': 6001130, 'y': 600805}),
        (27, {'x': 6001170, 'y': 600700}),
        (28, {'x': 6001100, 'y': 601200}),
        (29, {'x': 6001240, 'y': 600990}),
        (30, {'x': 6001300, 'y': 600760}),
        (31, {'x': 6000690, 'y': 600590}),
        (32, {'x': 6000570, 'y': 600530}),
        (33, {'x': 6000820, 'y': 600500}),
        (34, {'x': 6000700, 'y': 600480}),
        (35, {'x': 6000490, 'y': 600440}),
        (36, {'x': 6000580, 'y': 600360}),
        (37, {'x': 6000690, 'y': 600370}),
        (38, {'x': 6000920, 'y': 600330}),
        (39, {'x': 6000780, 'y': 600300}),
        (40, {'x': 6000680, 'y': 600200}),
        (41, {'x': 6000560, 'y': 600280}),
        (42, {'x': 6000450, 'y': 600300}),
        (43, {'x': 6000440, 'y': 600150}),
        (44, {'x': 6000650, 'y': 600080}),
        (45, {'x': 6000930, 'y': 600110})
    ]

    G.add_nodes_from(nodes)

    edges = [
        (0, 1),
        (0, 16),
        (0, 31),
        (1, 2),
        (1, 4),
        (2, 3),
        (2, 5),
        (3, 4),
        (3, 7),
        (4, 9),
        (5, 6),
        (5, 10),
        (6, 7),
        (6, 11),
        (7, 8),
        (8, 9),
        (8, 12),
        (9, 13),
        (10, 14),
        (10, 43),
        (11, 12),
        (11, 14),
        (12, 13),
        (13, 15),
        (14, 15),
        (15, 28),
        (16, 17),
        (16, 19),
        (17, 18),
        (17, 20),
        (18, 19),
        (18, 21),
        (19, 22),
        (20, 24),
        (20, 28),
        (21, 23),
        (21, 24),
        (22, 23),
        (22, 27),
        (23, 26),
        (24, 25),
        (25, 26),
        (25, 29),
        (26, 27),
        (27, 30),
        (28, 29),
        (29, 30),
        (30, 45),
        (31, 32),
        (31, 33),
        (32, 34),
        (32, 35),
        (33, 34),
        (33, 38),
        (34, 37),
        (35, 36),
        (35, 42),
        (36, 37),
        (36, 41),
        (37, 39),
        (38, 39),
        (38, 45),
        (39, 40),
        (40, 41),
        (40, 44),
        (41, 42),
        (42, 43),
        (43, 44),
        (44, 45)
    ]

    G.add_edges_from(edges)

    pos = {}
    for n, d in G.nodes(data=True):
        x = d['x']
        y = d['y']
        if wgs84_coords:
            y, x = utm.to_latlon(d['y'], d['x'], 30, 'U')
        G.nodes[n]['x'] = x
        G.nodes[n]['y'] = y

    return G, pos


def mock_data(G):

    min_x = None
    max_x = None
    min_y = None
    max_y = None

    for n, d in G.nodes(data=True):

        if not min_x:
            min_x = d['x']
        elif d['x'] < min_x:
            min_x = d['x']

        if not max_x:
            max_x = d['x']
        elif d['x'] > max_x:
            max_x = d['x']

        if not min_y:
            min_y = d['y']
        elif d['y'] < min_y:
            min_y = d['y']

        if not max_y:
            max_y = d['y']
        elif d['y'] > max_y:
            max_y = d['y']

    data_dict = {}
    for i in range(100):
        data_dict[i] = {
            'x': random.uniform(min_x, max_x),
            'y': random.uniform(min_y, max_y),
            'live': bool(random.getrandbits(1)),
            'class': random.uniform(0, 10)
        }

    return data_dict


def plot_graph_maps(node_map, edge_map, geom=None):

    # the links are undirected and therefore duplicate per edge
    # use two axes to check each copy of links
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

    # set extents
    for ax in (ax1, ax2):
        ax.set_xlim(node_map[:, 0].min() - 100, node_map[:, 0].max() + 100)
        ax.set_ylim(node_map[:, 1].min() - 100, node_map[:, 1].max() + 100)

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
                ax1.plot([src_data[0], nb_data[0]], [src_data[1], nb_data[1]], c='grey')
            else:
                ax2.plot([src_data[0], nb_data[0]], [src_data[1], nb_data[1]], c='grey')
            edge_idx += 1

    if geom:
        ax1.plot(geom.exterior.coords.xy[0], geom.exterior.coords.xy[1])
        ax2.plot(geom.exterior.coords.xy[0], geom.exterior.coords.xy[1])

    plt.show()
