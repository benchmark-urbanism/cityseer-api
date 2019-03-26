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
        (0, {'x': 700700, 'y': 5719700}),
        (1, {'x': 700610, 'y': 5719780}),
        (2, {'x': 700460, 'y': 5719700}),
        (3, {'x': 700520, 'y': 5719820}),
        (4, {'x': 700620, 'y': 5719905}),
        (5, {'x': 700260, 'y': 5719700}),
        (6, {'x': 700320, 'y': 5719850}),
        (7, {'x': 700420, 'y': 5719880}),
        (8, {'x': 700460, 'y': 5719980}),
        (9, {'x': 700580, 'y': 5720030}),
        (10, {'x': 700100, 'y': 5719810}),
        (11, {'x': 700280, 'y': 5719980}),
        (12, {'x': 700400, 'y': 5720030}),
        (13, {'x': 700460, 'y': 5720130}),
        (14, {'x': 700190, 'y': 5720050}),
        (15, {'x': 700350, 'y': 5720200}),
        (16, {'x': 700800, 'y': 5719750}),
        (17, {'x': 700800, 'y': 5719920}),
        (18, {'x': 700900, 'y': 5719820}),
        (19, {'x': 700910, 'y': 5719690}),
        (20, {'x': 700905, 'y': 5720080}),
        (21, {'x': 701000, 'y': 5719870}),
        (22, {'x': 701040, 'y': 5719660}),
        (23, {'x': 701050, 'y': 5719760}),
        (24, {'x': 701000, 'y': 5719980}),
        (25, {'x': 701130, 'y': 5719950}),
        (26, {'x': 701130, 'y': 5719805}),
        (27, {'x': 701170, 'y': 5719700}),
        (28, {'x': 701100, 'y': 5720200}),
        (29, {'x': 701240, 'y': 5719990}),
        (30, {'x': 701300, 'y': 5719760}),
        (31, {'x': 700690, 'y': 5719590}),
        (32, {'x': 700570, 'y': 5719530}),
        (33, {'x': 700820, 'y': 5719500}),
        (34, {'x': 700700, 'y': 5719480}),
        (35, {'x': 700490, 'y': 5719440}),
        (36, {'x': 700580, 'y': 5719360}),
        (37, {'x': 700690, 'y': 5719370}),
        (38, {'x': 700920, 'y': 5719330}),
        (39, {'x': 700780, 'y': 5719300}),
        (40, {'x': 700680, 'y': 5719200}),
        (41, {'x': 700560, 'y': 5719280}),
        (42, {'x': 700450, 'y': 5719300}),
        (43, {'x': 700440, 'y': 5719150}),
        (44, {'x': 700650, 'y': 5719080}),
        (45, {'x': 700930, 'y': 5719110}),
        # cul-de-sacs
        (46, {'x': 701015, 'y': 5719535}),
        (47, {'x': 701100, 'y': 5719480}),
        (48, {'x': 700917, 'y': 5719517}),
        # isolated node
        (49, {'x': 700400, 'y': 5719550}),
        # isolated edge
        (50, {'x': 700700, 'y': 5720100}),
        (51, {'x': 700700, 'y': 5719900}),
        # disconnected looping component
        (52, {'x': 700400, 'y': 5719650}),
        (53, {'x': 700500, 'y': 5719550}),
        (54, {'x': 700400, 'y': 5719450}),
        (55, {'x': 700300, 'y': 5719550})
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
        (50, 51),
        # disconnected looping component
        (52, 53),
        (53, 54),
        (54, 55),
        (55, 52)
    ]

    G.add_edges_from(edges)

    for n, d in G.nodes(data=True):
        x = d['x']
        y = d['y']
        if wgs84_coords:
            easting = x
            northing = y
            # be cognisant of parameter and return order
            # returns in lat, lng order
            lat, lng = utm.to_latlon(easting, northing, 30, 'U')
            x = lng
            y = lat
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
