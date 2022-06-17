"""
A collection of functions for the generation of mock data.

This module is intended for project development and writing code tests, but may otherwise be useful for demonstration
and utility purposes.
"""

import logging
import string
from typing import Any, Generator

import networkx as nx
import numpy as np
import numpy.typing as npt
import requests
import utm
from shapely import geometry

from cityseer import types
from cityseer.tools import graphs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# type hack until networkx supports type-hinting
MultiGraph = Any


def mock_graph(wgs84_coords: bool = False) -> MultiGraph:
    """
    Generate a `NetworkX` `MultiGraph` for testing or experimentation purposes.

    Parameters
    ----------
    wgs84_coords: bool
        If set to `True`, the `x` and `y` attributes will be in [WGS84](https://epsg.io/4326) geographic coordinates
        instead of a projected cartesion coordinate system. By default False

    Returns
    -------
    MultiGraph
        A `NetworkX` `MultiGraph` with `x` and `y` node attributes.

    Examples
    --------
    ```python
    from cityseer.tools import mock, plot
    nx_multigraph = mock.mock_graph()
    plot.plot_nx(nx_multigraph)
    ```

    ![Example graph](/images/graph_example.png)
    _Mock graph._

    """
    nx_multigraph = nx.MultiGraph()

    nodes = [
        (0, {"x": 700700, "y": 5719700}),
        (1, {"x": 700610, "y": 5719780}),
        (2, {"x": 700460, "y": 5719700}),
        (3, {"x": 700520, "y": 5719820}),
        (4, {"x": 700620, "y": 5719905}),
        (5, {"x": 700260, "y": 5719700}),
        (6, {"x": 700320, "y": 5719850}),
        (7, {"x": 700420, "y": 5719880}),
        (8, {"x": 700460, "y": 5719980}),
        (9, {"x": 700580, "y": 5720030}),
        (10, {"x": 700100, "y": 5719810}),
        (11, {"x": 700280, "y": 5719980}),
        (12, {"x": 700400, "y": 5720030}),
        (13, {"x": 700460, "y": 5720130}),
        (14, {"x": 700190, "y": 5720050}),
        (15, {"x": 700350, "y": 5720200}),
        (16, {"x": 700800, "y": 5719750}),
        (17, {"x": 700800, "y": 5719920}),
        (18, {"x": 700900, "y": 5719820}),
        (19, {"x": 700910, "y": 5719690}),
        (20, {"x": 700905, "y": 5720080}),
        (21, {"x": 701000, "y": 5719870}),
        (22, {"x": 701040, "y": 5719660}),
        (23, {"x": 701050, "y": 5719760}),
        (24, {"x": 701000, "y": 5719980}),
        (25, {"x": 701130, "y": 5719950}),
        (26, {"x": 701130, "y": 5719805}),
        (27, {"x": 701170, "y": 5719700}),
        (28, {"x": 701100, "y": 5720200}),
        (29, {"x": 701240, "y": 5719990}),
        (30, {"x": 701300, "y": 5719760}),
        (31, {"x": 700690, "y": 5719590}),
        (32, {"x": 700570, "y": 5719530}),
        (33, {"x": 700820, "y": 5719500}),
        (34, {"x": 700700, "y": 5719480}),
        (35, {"x": 700490, "y": 5719440}),
        (36, {"x": 700580, "y": 5719360}),
        (37, {"x": 700690, "y": 5719370}),
        (38, {"x": 700920, "y": 5719330}),
        (39, {"x": 700780, "y": 5719300}),
        (40, {"x": 700680, "y": 5719200}),
        (41, {"x": 700560, "y": 5719280}),
        (42, {"x": 700450, "y": 5719300}),
        (43, {"x": 700440, "y": 5719150}),
        (44, {"x": 700650, "y": 5719080}),
        (45, {"x": 700930, "y": 5719110}),
        # cul-de-sacs
        (46, {"x": 701015, "y": 5719535}),
        (47, {"x": 701100, "y": 5719480}),
        (48, {"x": 700917, "y": 5719517}),
        # isolated node
        (49, {"x": 700400, "y": 5719550}),
        # isolated edge
        (50, {"x": 700700, "y": 5720100}),
        (51, {"x": 700700, "y": 5719900}),
        # disconnected looping component
        (52, {"x": 700400, "y": 5719650}),
        (53, {"x": 700500, "y": 5719550}),
        (54, {"x": 700400, "y": 5719450}),
        (55, {"x": 700300, "y": 5719550}),
        # add a parallel edge
        (56, {"x": 701300, "y": 5719110}),
    ]

    nx_multigraph.add_nodes_from(nodes)

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
        (30, 56),
    ]

    nx_multigraph.add_edges_from(edges)

    if wgs84_coords:
        for node_idx, node_data in nx_multigraph.nodes(data=True):  # type: ignore
            easting = node_data["x"]  # type: ignore
            northing = node_data["y"]  # type: ignore
            # be cognisant of parameter and return order
            # returns in lat, lng order
            lat, lng = utm.to_latlon(easting, northing, 30, "U")  # type: ignore
            nx_multigraph.nodes[node_idx]["x"] = lng  # type: ignore
            nx_multigraph.nodes[node_idx]["y"] = lat  # type: ignore

    return nx_multigraph


def get_graph_extents(
    nx_multigraph: MultiGraph,
) -> tuple[float, float, float, float]:
    """
    Derive geographic bounds for a given networkX graph.

    Parameters
    ----------
    nx_multigraph: MultiGraph
        A `NetworkX` `MultiGraph` with `x` and `y` node parameters.

    Returns
    -------
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    """
    # get min and maxes for x and y
    min_x: float = np.inf
    max_x: float = -np.inf
    min_y: float = np.inf
    max_y: float = -np.inf

    _node_idx: int | str
    node_data: dict[str, Any]
    for _node_idx, node_data in nx_multigraph.nodes(data=True):
        if node_data["x"] < min_x:
            min_x = node_data["x"]
        if node_data["x"] > max_x:
            max_x = node_data["x"]
        if node_data["y"] < min_y:
            min_y = node_data["y"]
        if node_data["y"] > max_y:
            max_y = node_data["y"]

    return min_x, min_y, max_x, max_y


def mock_data_dict(nx_multigraph: MultiGraph, length: int = 50, random_seed: int = 0) -> types.DataDictType:
    """
    Generate a dictionary containing mock data for testing or experimentation purposes.

    Parameters
    ----------
    nx_multigraph: MultiGraph
        A `NetworkX` graph with `x` and `y` attributes. This is used in order to determine the spatial extents of the
        network. The returned data will be within these extents.
    length: int
        The number of data elements to return in the dictionary, by default 50.
    random_seed: int
        An optional random seed, by default None.

    Returns
    -------
    dict
        A dictionary where each entry consists of a distinct `data_key`, and corresponding `x` and `y`.

    """
    np.random.seed(seed=random_seed)  # pylint: disable=no-member
    min_x, min_y, max_x, max_y = get_graph_extents(nx_multigraph)
    data_dict: types.DataDictType = {}
    for data_idx in range(length):
        data_dict[data_idx] = {
            "x": np.random.uniform(min_x, max_x),  # pylint: disable=no-member
            "y": np.random.uniform(min_y, max_y),  # pylint: disable=no-member
        }
    return data_dict


def mock_categorical_data(length: int, num_classes: int = 10, random_seed: int = 0) -> list[str]:
    """
    Generate a `numpy` array containing mock categorical data for testing or experimentation purposes.

    Parameters
    ----------
    length: int
        The number of categorical elements to return in the array.
    num_classes: int
        The maximum number of unique classes to return in the randomly assigned categorical data. The classes are
        randomly generated from a pool of unique class labels of length `num_classes`. The number of returned unique
        classes will be less than or equal to `num_classes`. By default 10
    random_seed: int
        An optional random seed, by default None

    Returns
    -------
    list[str]
        A `list`containing categorical data elements. The number of elements will match the `length` parameter.
        The categorical data will consist of randomly selected characters.

    """
    np.random.seed(seed=random_seed)

    random_class_str = string.ascii_lowercase
    if num_classes > len(random_class_str):
        raise ValueError(
            f"The requested {num_classes} classes exceeds max available categorical classes of {len(random_class_str)}"
        )
    random_class_str = random_class_str[:num_classes]

    cl_codes: list[str] = []
    for _ in range(length):
        random_idx = int(np.random.randint(0, len(random_class_str)))
        cl_codes.append(random_class_str[random_idx])

    return cl_codes


def mock_numerical_data(
    length: int,
    val_min: int = 0,
    val_max: int = 100000,
    num_arrs: int = 1,
    floating_pt: int = 3,
    random_seed: int = 0,
) -> npt.NDArray[np.float32]:
    """
    Generate a 2d `numpy` array containing mock numerical data for testing or experimentation purposes.

    Parameters
    ----------
    length: int
        The number of numerical elements to return in the array.
    val_min: int
        The (inclusive) minimum value in the `val_min`, `val_max` range of randomly generated integers.
    val_max: int
        The (exclusive) maximum value in the `val_min`, `val_max` range of randomly generated integers.
    num_arrs: int
        The number of arrays to nest in the returned 2d array.
    floating_pt: int
        The floating point precision
    random_seed: int
        An optional random seed, by default None

    Returns
    -------
    ndarray[float]
        A 2d `numpy` array containing numerical data elements. The first dimension corresponds to the number of data
        arrays, and is set with the `num_arrs` parameter. The length of the second dimension will represents the number
        of data elements and will match the `length` parameter. The numerical data will consist of randomly selected
        integers.

    """
    np.random.seed(seed=random_seed)  # pylint: disable=no-member

    num_data = []
    for _ in range(num_arrs):
        num_data.append(np.random.randint(val_min, high=val_max, size=length))  # pylint: disable=no-member
    # return a 2d array
    num_arr: npt.NDArray[np.float32] = np.array(num_data, dtype=np.float32)
    num_arr /= 10**floating_pt

    return num_arr


def mock_species_data(
    random_seed: int = 0,
) -> Generator[tuple[npt.NDArray[np.int_], npt.NDArray[np.float32]], None, None]:
    """
    Generate a series of randomly generated counts and corresponding probabilities.

    This function is used for testing diversity measures. The data is generated in varying lengths from randomly
    assigned integers between 1 and 10. Matching integers are then collapsed into species "classes" with probabilities
    computed accordingly.

    Parameters
    ----------
    random_seed: int
        An optional random seed, by default None

    Yields
    ------
    counts: ndarray[int]
        The number of members for each species class.
    probs: ndarray[float]
        The probability of encountering the respective species classes.

    Examples
    --------
    ```python
    from cityseer.tools import mock

    for counts, probs in mock.mock_species_data():
        cs = [c for c in counts]
        print(f'c = {cs}')
        ps = [round(p, 3) for p in probs]
        print(f'p = {ps}')

    # c = [1]
    # p = [1.0]

    # c = [1, 1, 2, 2]
    # p = [0.167, 0.167, 0.333, 0.333]

    # c = [3, 2, 1, 1, 1, 3]
    # p = [0.273, 0.182, 0.091, 0.091, 0.091, 0.273]

    # c = [3, 3, 2, 2, 1, 1, 1, 2, 1]
    # p = [0.188, 0.188, 0.125, 0.125, 0.062, 0.062, 0.062, 0.125, 0.062]

    # etc.
    ```

    """
    np.random.seed(seed=random_seed)  # pylint: disable=no-member

    for n in range(1, 50, 5):
        data = np.random.randint(1, 10, n)  # pylint: disable=no-member
        unique: npt.NDArray[np.int_] = np.unique(data)
        counts: npt.NDArray[np.int_] = np.zeros_like(unique, dtype=np.int_)
        for idx, uniq in enumerate(unique):
            counts[idx] = (data == uniq).sum()
        probs = counts / len(data)

        yield counts, probs


def fetch_osm_response(geom_osm: str, timeout: int = 30, max_tries: int = 3) -> requests.Response | None:
    """
    Fetch and parse an OSM response.

    Parameters
    ----------
    geom_osm: str
        A string representing a polygon for the request extents, formatting according to OSM conventions.
    timeout: int
        An optional timeout, by default 30s
    max_tries: int
        The number of attempts to fetch a response before raising, by default 3

    Returns
    -------
    requests.Response
        An OSM API response.

    """
    request = f"""
    /* https://wiki.openstreetmap.org/wiki/Overpass_API/Overpass_QL */
    [out:json][timeout:{timeout}];
    /* build spatial_set from highways based on extent */
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
    /* build union_set from spatial_set */
    (
      way.spatial_set["highway"];
      way.spatial_set["foot"~"yes|designated"];
      way.spatial_set["bicycle"~"yes|designated"];
    ) -> .union_set;
    /* filter union_set */
    way.union_set -> .filtered_set;
    /* union filtered_set ways with nodes via recursion */
    (
      .filtered_set;
      >;
    );
    /* return only basic info */
    out skel qt;
    """
    osm_response: requests.Response | None = None
    while max_tries:
        osm_response = requests.get(
            "https://overpass-api.de/api/interpreter",
            timeout=timeout,
            params={"data": request},
        )
        # break if OK response
        if osm_response is not None and osm_response.status_code == 200:
            break
        # otherwise try until max_tries is exhausted
        logger.warning("Unsuccessful OSM API request response, trying again...")
        max_tries -= 1

    if osm_response is None or not osm_response.status_code == 200:
        raise requests.RequestException("Unsuccessful OSM API request.")

    return osm_response


def make_buffered_osm_graph(lng: float, lat: float, buffer: float) -> MultiGraph:  # noqa
    """

    Prepares a `networkX` `MultiGraph` from an OSM request for a buffered region around a given `lng` and `lat`
    parameter.

    Parameters
    ----------
    lng: float
        The longitude argument for the request.
    lat: float
        The latitude argument for the request.
    buffer: float
        The buffer distance.

    Returns
    -------
    MultiGraph
        A `networkX` `MultiGraph` with `x` and `y` node attributes that have been converted to UTM.
    """
    # cast the WGS coordinates to UTM prior to buffering
    easting, northing, utm_zone_number, utm_zone_letter = utm.from_latlon(lat, lng)
    # create a point, and then buffer
    pnt = geometry.Point(easting, northing)
    poly_utm: geometry.Polygon = pnt.buffer(buffer)  # type: ignore
    # convert back to WGS
    # the polygon is too big for the OSM server, so have to use convex hull then later prune
    geom = [
        utm.to_latlon(east, north, utm_zone_number, utm_zone_letter)  # type: ignore
        for east, north in poly_utm.convex_hull.exterior.coords  # type: ignore # pylint: disable=no-member
    ]
    poly_wgs = geometry.Polygon(geom)
    # format for OSM query
    geom_osm = str.join(" ", [f"{lat} {lng}" for lat, lng in poly_wgs.exterior.coords])  # type: ignore
    # generate the query
    osm_response = fetch_osm_response(geom_osm)
    # build graph
    graph_wgs = graphs.nx_from_osm(osm_json=osm_response.text)  # type: ignore
    # cast to UTM
    graph_utm = graphs.nx_wgs_to_utm(graph_wgs)

    return graph_utm
