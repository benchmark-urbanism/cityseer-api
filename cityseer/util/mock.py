'''
Generate a graph for testing and documentation purposes.
'''
import logging
from typing import Tuple

import networkx as nx
import numpy as np
import utm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def mock_graph(wgs84_coords: bool = False) -> nx.Graph:
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
        (45, {'x': 6000930, 'y': 600110}),
        # cul-de-sacs
        (46, {'x': 6001015, 'y': 600535}),
        (47, {'x': 6001100, 'y': 600480}),
        (48, {'x': 6000917, 'y': 600517}),
        # isolated node
        (49, {'x': 6000400, 'y': 600550}),
        # isolated edge
        (50, {'x': 6000700, 'y': 601100}),
        (51, {'x': 6000700, 'y': 600900})
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
        (44, 45),
        # cul-de-sacs
        (22, 46),
        (46, 47),
        (46, 48),
        # isolated edge
        (50, 51)
    ]

    G.add_edges_from(edges)

    for n, d in G.nodes(data=True):
        x = d['x']
        y = d['y']
        if wgs84_coords:
            y, x = utm.to_latlon(d['y'], d['x'], 30, 'U')
        G.nodes[n]['x'] = x
        G.nodes[n]['y'] = y

    return G


def get_graph_extents(G: nx.Graph) -> Tuple[float, float, float, float]:

    # get min and maxes for x and y
    min_x = np.inf
    max_x = -np.inf
    min_y = np.inf
    max_y = -np.inf

    for n, d in G.nodes(data=True):
        if d['x'] < min_x:
            min_x = d['x']
        if d['x'] > max_x:
            max_x = d['x']
        if d['y'] < min_y:
            min_y = d['y']
        if d['y'] > max_y:
            max_y = d['y']

    return min_x, min_y, max_x, max_y


def mock_data_dict(G: nx.Graph, length: int = 50, random_seed: int = None) -> dict:
    if random_seed is not None:
        np.random.seed(seed=random_seed)

    min_x, min_y, max_x, max_y = get_graph_extents(G)

    data_dict = {}

    for i in range(length):
        data_dict[i] = {
            'x': np.random.uniform(min_x, max_x),
            'y': np.random.uniform(min_y, max_y),
            'live': bool(np.random.randint(0, 1))
        }

    return data_dict


def mock_categorical_data(length: int, random_seed: int = None) -> np.ndarray:
    if random_seed is not None:
        np.random.seed(seed=random_seed)

    random_class_str = 'abcdefghijk'
    d = []

    for i in range(length):
        d.append(random_class_str[np.random.randint(0, len(random_class_str) - 1)])

    return np.array(d)


def mock_numerical_data(length: int, num_arrs: int = 1, random_seed: int = None) -> np.ndarray:
    if random_seed is not None:
        np.random.seed(seed=random_seed)

    num_data = []

    for i in range(num_arrs):
        data = []

        for i in range(length):
            data.append(np.random.uniform(low=0, high=100000))

        num_data.append(data)

    # return a 2d array
    return np.array(num_data)


def mock_species_data(random_seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    if random_seed is not None:
        np.random.seed(seed=random_seed)

    for n in range(1, 50, 5):
        data = np.random.randint(1, 10, n)
        unique = np.unique(data)
        counts = np.zeros_like(unique)
        for i, u in enumerate(unique):
            counts[i] = (data == u).sum()
        probs = counts / len(data)

        yield counts, probs
