'''
Generate a graph for testing and documentation purposes.
'''
import logging
import networkx as nx
import numpy as np
import pytest
import requests
from shapely import geometry
import string
from typing import Tuple
import utm

from cityseer.tools import graphs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def mock_graph(wgs84_coords: bool = False) -> nx.MultiGraph:
    '''
    Prepares a Tutte graph per https://en.wikipedia.org/wiki/Tutte_graph
    :return: NetworkX graph
    '''

    G = nx.MultiGraph()

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
        (55, {'x': 700300, 'y': 5719550}),
        # add a parallel edge
        (56, {'x': 701300, 'y': 5719110})
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
        (55, 52),
        # parallel edge
        (45, 56),
        (30, 56)
    ]

    G.add_edges_from(edges)

    if wgs84_coords:
        for n, d in G.nodes(data=True):
            easting = d['x']
            northing = d['y']
            # be cognisant of parameter and return order
            # returns in lat, lng order
            lat, lng = utm.to_latlon(easting, northing, 30, 'U')
            G.nodes[n]['x'] = lng
            G.nodes[n]['y'] = lat

    return G


@pytest.fixture
def primal_graph():
    G_primal = mock_graph()
    G_primal = graphs.nX_simple_geoms(G_primal)
    return G_primal


@pytest.fixture
def dual_graph():
    G_dual = mock_graph()
    G_dual = graphs.nX_simple_geoms(G_dual)
    G_dual = graphs.nX_to_dual(G_dual)
    return G_dual


@pytest.fixture
def diamond_graph():
    '''
    For manual checks of all node and segmentised methods
       3
      / \
     /   \
    /  a  \
   1-------2
    \  |  /
     \ |b/ c
      \|/
       0
    a = 100m = 2 * 50m
    b = 86.60254m
    c = 100m
    all inner angles = 60º
    '''
    G_diamond = nx.MultiGraph()
    G_diamond.add_nodes_from([
        (0, {'x': 50, 'y': 0}),
        (1, {'x': 0, 'y': 86.60254}),
        (2, {'x': 100, 'y': 86.60254}),
        (3, {'x': 50, 'y': 86.60254 * 2})
    ])
    G_diamond.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)])
    G_diamond = graphs.nX_simple_geoms(G_diamond)
    return G_diamond


def get_graph_extents(G: nx.MultiGraph) -> Tuple[float, float, float, float]:
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


def mock_data_dict(G: nx.MultiGraph, length: int = 50, random_seed: int = None) -> dict:
    if random_seed is not None:
        np.random.seed(seed=random_seed)

    min_x, min_y, max_x, max_y = get_graph_extents(G)

    data_dict = {}

    for i in range(length):
        data_dict[i] = {
            'x': np.random.uniform(min_x, max_x),
            'y': np.random.uniform(min_y, max_y)
        }

    return data_dict


def mock_categorical_data(length: int, num_classes: int = 10, random_seed: int = None) -> np.ndarray:
    if random_seed is not None:
        np.random.seed(seed=random_seed)

    random_class_str = string.ascii_lowercase
    if num_classes > len(random_class_str):
        raise ValueError(
            f'The requested {num_classes} classes exceeds the max available categorical classes: {len(random_class_str)}')
    random_class_str = random_class_str[:num_classes]

    d = []
    for i in range(length):
        d.append(random_class_str[np.random.randint(0, len(random_class_str))])

    return np.array(d)


def mock_numerical_data(length: int, min: int = 0, max: int = 100000, num_arrs: int = 1,
                        random_seed: int = None) -> np.ndarray:
    if random_seed is not None:
        np.random.seed(seed=random_seed)

    num_data = []
    for i in range(num_arrs):
        num_data.append(np.random.randint(min, high=max, size=length))
    # return a 2d array
    return np.array(num_data, dtype=float)


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


def fetch_osm_response(geom_osm: str, timeout: int = 30, max_tries: int = 3) -> requests.Response:
    request = f'''
    /* https://wiki.openstreetmap.org/wiki/Overpass_API/Overpass_QL
    */
    [out:json][timeout:{timeout}];
    /*
    build spatial_set from highways based on extent
    */
    way["highway"]
      ["area"!="yes"]
      ["highway"!~"motorway|motorway_link|bus_guideway|escape|raceway|proposed|abandoned|platform|construction"]
      ["service"!~"parking_aisle"]
      (if:
       /* don't fetch roads that don't have sidewalks */
       (t["sidewalk"] != "none" && t["sidewalk"] != "no")
       /* unless foot or bicycles permitted */
       || t["foot"]!="no"
       || (t["bicycle"]!="no" && t["bicycle"]!="unsuitable")
      )
      ["amenity"!~"charging_station|parking|fuel|motorcycle_parking|parking_entrance|parking_space"]
      ["access"!~"private|customers"]
      ["indoor"!="yes"]
      (poly:"{geom_osm}") -> .spatial_set;
    /*
    build union_set from spatial_set
    */
    (
      way.spatial_set["highway"];
      way.spatial_set["foot"~"yes|designated"];
      way.spatial_set["bicycle"~"yes|designated"];
    ) -> .union_set;
    /*
    filter union_set
    */
    way.union_set -> .filtered_set;
    /*
    union filtered_set ways with nodes via recursion
    */
    (
      .filtered_set;
      >;
    );
    /*
    return only basic info
    */
    out skel qt;
    '''

    osm_response = None
    while max_tries:
        osm_response = requests.get('https://overpass-api.de/api/interpreter',
                                    timeout=timeout,
                                    params={'data': request})
        # break if OK response
        if osm_response is not None and osm_response.status_code == 200:
            break
        # otherwise try until max_tries is exhausted
        logger.warning(f'Unsuccessful OSM API request response, trying again...')
        max_tries -= 1

    if osm_response is None or not osm_response.status_code == 200:
        raise('Unsuccessful OSM API request.')

    return osm_response


def make_buffered_osm_graph(lng: float, lat: float, buffer: float) -> nx.MultiGraph:
    """

    """
    # cast the WGS coordinates to UTM prior to buffering
    easting, northing, utm_zone_number, utm_zone_letter = utm.from_latlon(lat, lng)
    # create a point, and then buffer
    pt = geometry.Point(easting, northing)
    poly_utm = pt.buffer(buffer)
    # convert back to WGS
    # the polygon is too big for the OSM server, so have to use convex hull then later prune
    geom = [utm.to_latlon(east, north, utm_zone_number, utm_zone_letter) for east, north in
            poly_utm.convex_hull.exterior.coords]
    poly_wgs = geometry.Polygon(geom)
    # format for OSM query
    geom_osm = str.join(' ', [f'{lat} {lng}' for lat, lng in poly_wgs.exterior.coords])
    # generate the query
    osm_response = fetch_osm_response(geom_osm)
    # build graph
    G_wgs = graphs.nX_from_osm(osm_json=osm_response.text)
    # cast to UTM
    G_utm = graphs.nX_wgs_to_utm(G_wgs)

    return G_utm
