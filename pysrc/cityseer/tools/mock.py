"""
A collection of functions for the generation of mock data.

This module is intended for project development and writing code tests, but may otherwise be useful for demonstration
and utility purposes.
"""

from __future__ import annotations

import logging
import string
from typing import Any, Generator, cast

import geopandas as gpd
import networkx as nx
import numpy as np
import numpy.typing as npt
from shapely import geometry

from cityseer.tools import util

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
        instead of a projected cartesion coordinate system.

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
    nx_multigraph: MultiGraph = nx.MultiGraph()

    nodes = [
        ("0", {"x": 700700, "y": 5719700}),
        ("1", {"x": 700610, "y": 5719780}),
        ("2", {"x": 700460, "y": 5719700}),
        ("3", {"x": 700520, "y": 5719820}),
        ("4", {"x": 700620, "y": 5719905}),
        ("5", {"x": 700260, "y": 5719700}),
        ("6", {"x": 700320, "y": 5719850}),
        ("7", {"x": 700420, "y": 5719880}),
        ("8", {"x": 700460, "y": 5719980}),
        ("9", {"x": 700580, "y": 5720030}),
        ("10", {"x": 700100, "y": 5719810}),
        ("11", {"x": 700280, "y": 5719980}),
        ("12", {"x": 700400, "y": 5720030}),
        ("13", {"x": 700460, "y": 5720130}),
        ("14", {"x": 700190, "y": 5720050}),
        ("15", {"x": 700350, "y": 5720200}),
        ("16", {"x": 700800, "y": 5719750}),
        ("17", {"x": 700800, "y": 5719920}),
        ("18", {"x": 700900, "y": 5719820}),
        ("19", {"x": 700910, "y": 5719690}),
        ("20", {"x": 700905, "y": 5720080}),
        ("21", {"x": 701000, "y": 5719870}),
        ("22", {"x": 701040, "y": 5719660}),
        ("23", {"x": 701050, "y": 5719760}),
        ("24", {"x": 701000, "y": 5719980}),
        ("25", {"x": 701130, "y": 5719950}),
        ("26", {"x": 701130, "y": 5719805}),
        ("27", {"x": 701170, "y": 5719700}),
        ("28", {"x": 701100, "y": 5720200}),
        ("29", {"x": 701240, "y": 5719990}),
        ("30", {"x": 701300, "y": 5719760}),
        ("31", {"x": 700690, "y": 5719590}),
        ("32", {"x": 700570, "y": 5719530}),
        ("33", {"x": 700820, "y": 5719500}),
        ("34", {"x": 700700, "y": 5719480}),
        ("35", {"x": 700490, "y": 5719440}),
        ("36", {"x": 700580, "y": 5719360}),
        ("37", {"x": 700690, "y": 5719370}),
        ("38", {"x": 700920, "y": 5719330}),
        ("39", {"x": 700780, "y": 5719300}),
        ("40", {"x": 700680, "y": 5719200}),
        ("41", {"x": 700560, "y": 5719280}),
        ("42", {"x": 700450, "y": 5719300}),
        ("43", {"x": 700440, "y": 5719150}),
        ("44", {"x": 700650, "y": 5719080}),
        ("45", {"x": 700930, "y": 5719110}),
        # cul-de-sacs
        ("46", {"x": 701015, "y": 5719535}),
        ("47", {"x": 701100, "y": 5719480}),
        ("48", {"x": 700917, "y": 5719517}),
        # isolated node
        ("49", {"x": 700400, "y": 5719550}),
        # isolated edge
        ("50", {"x": 700700, "y": 5720100}),
        ("51", {"x": 700700, "y": 5719900}),
        # disconnected looping component
        # don't make exactly diamond shaped to remove ambiguity about shortest path route
        ("52", {"x": 700400, "y": 5719650}),
        ("53", {"x": 700550, "y": 5719550}),
        ("54", {"x": 700410, "y": 5719450}),
        ("55", {"x": 700300, "y": 5719550}),
        # add a parallel edge
        ("56", {"x": 701300, "y": 5719110}),
    ]

    nx_multigraph.add_nodes_from(nodes)

    edges = [
        ("0", "1"),
        ("0", "16"),
        ("0", "31"),
        ("1", "2"),
        ("1", "4"),
        ("2", "3"),
        ("2", "5"),
        ("3", "4"),
        ("3", "7"),
        ("4", "9"),
        ("5", "6"),
        ("5", "10"),
        ("6", "7"),
        ("6", "11"),
        ("7", "8"),
        ("8", "9"),
        ("8", "12"),
        ("9", "13"),
        ("10", "14"),
        ("10", "43"),
        ("11", "12"),
        ("11", "14"),
        ("12", "13"),
        ("13", "15"),
        ("14", "15"),
        ("15", "28"),
        ("16", "17"),
        ("16", "19"),
        ("17", "18"),
        ("17", "20"),
        ("18", "19"),
        ("18", "21"),
        ("19", "22"),
        ("20", "24"),
        ("20", "28"),
        ("21", "23"),
        ("21", "24"),
        ("22", "23"),
        ("22", "27"),
        ("23", "26"),
        ("24", "25"),
        ("25", "26"),
        ("25", "29"),
        ("26", "27"),
        ("27", "30"),
        ("28", "29"),
        ("29", "30"),
        ("30", "45"),
        ("31", "32"),
        ("31", "33"),
        ("32", "34"),
        ("32", "35"),
        ("33", "34"),
        ("33", "38"),
        ("34", "37"),
        ("35", "36"),
        ("35", "42"),
        ("36", "37"),
        ("36", "41"),
        ("37", "39"),
        ("38", "39"),
        ("38", "45"),
        ("39", "40"),
        ("40", "41"),
        ("40", "44"),
        ("41", "42"),
        ("42", "43"),
        ("43", "44"),
        ("44", "45"),
        # " cul-de-sacs
        ("22", "46"),
        ("46", "47"),
        ("46", "48"),
        # " isolated edge
        ("50", "51"),
        # " disconnected looping component
        ("52", "53"),
        ("53", "54"),
        ("54", "55"),
        ("55", "52"),
        # " parallel edge
        ("45", "56"),
        ("30", "56"),
    ]

    nx_multigraph.add_edges_from(edges)

    if wgs84_coords:
        for node_idx, node_data in nx_multigraph.nodes(data=True):
            easting = node_data["x"]
            northing = node_data["y"]
            wgs_pnt = util.project_geom(geometry.Point(easting, northing), 32630, 4326)  # type: ignore
            nx_multigraph.nodes[node_idx]["x"] = wgs_pnt.x
            nx_multigraph.nodes[node_idx]["y"] = wgs_pnt.y

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


def mock_data_gdf(nx_multigraph: MultiGraph, length: int = 50, random_seed: int = 0) -> gpd.GeoDataFrame:
    """
    Generate a `GeoDataFrame` containing mock data for testing or experimentation purposes.

    Parameters
    ----------
    nx_multigraph: MultiGraph
        A `NetworkX` graph with `x` and `y` attributes. This is used in order to determine the spatial extents of the
        network. The returned data will be within these extents.
    length: int
        The number of data elements to return in the `GeoDataFrame`.
    random_seed: int
        An optional random seed.

    Returns
    -------
    GeoDataFrame
        A `GeoDataFrame` with data points for testing purposes.

    """
    np.random.seed(seed=random_seed)
    min_x, min_y, max_x, max_y = get_graph_extents(nx_multigraph)
    xs = np.random.uniform(min_x, max_x, length)
    ys = np.random.uniform(min_y, max_y, length)
    data_gpd = gpd.GeoDataFrame(
        {
            "uid": [str(i) for i in np.arange(length)],
            "geometry": gpd.points_from_xy(xs, ys),
            "data_id": np.arange(length),
        }
    )
    data_gpd = data_gpd.set_index("uid")
    # last 5 datapoints are a cluster of nodes where the nodes share the same data_id for deduplication checks
    for idx, loc_idx in enumerate(range(length - 5, length)):
        data_gpd.loc[str(loc_idx), "data_id"] = length - 5
        data_gpd.loc[str(loc_idx), "geometry"] = geometry.Point(700100 + idx * 10, 5719100 + idx * 10)  # type: ignore
    data_gpd = cast(gpd.GeoDataFrame, data_gpd)
    return data_gpd


def mock_landuse_categorical_data(
    nx_multigraph: MultiGraph, length: int = 50, num_classes: int = 10, random_seed: int = 0
) -> gpd.GeoDataFrame:
    """
    Generate a `numpy` array containing mock categorical data for testing or experimentation purposes.

    Parameters
    ----------
    nx_multigraph: MultiGraph
        A `NetworkX` graph with `x` and `y` attributes. This is used in order to determine the spatial extents of the
        network. The returned data will be within these extents.
    length: int
        The number of categorical elements to return in the array.
    num_classes: int
        The maximum number of unique classes to return in the randomly assigned categorical data. The classes are
        randomly generated from a pool of unique class labels of length `num_classes`. The number of returned unique
        classes will be less than or equal to `num_classes`.
    random_seed: int
        An optional random seed.

    Returns
    -------
    GeoDataFrame
        A `GeoDataFrame` with a "categorical_landuses" data column for testing purposes. The number of rows will match
        the `length` parameter. The categorical data will consist of randomly selected characters from `num_classes`.

    """
    np.random.seed(seed=random_seed)
    random_class_str: list[str] = list(string.ascii_lowercase)
    if num_classes > len(random_class_str):
        raise ValueError(
            f"The requested {num_classes} classes exceeds max available categorical classes of {len(random_class_str)}"
        )
    data_gpd = mock_data_gdf(nx_multigraph, length=length, random_seed=random_seed)
    random_class_str = random_class_str[: num_classes - 1]
    cl_codes: list[str] = []
    for idx in range(len(data_gpd)):
        # set last 5 items to z - to correspond to deduplication checks
        if idx >= length - 5:
            cl_codes.append("z")
        else:
            class_key = int(np.random.randint(0, len(random_class_str)))
            cl_codes.append(random_class_str[class_key])
    data_gpd["categorical_landuses"] = cl_codes  # pylint: disable=unsupported-assignment-operation

    return data_gpd


def mock_numerical_data(
    nx_multigraph: MultiGraph,
    length: int = 50,
    val_min: int = 0,
    val_max: int = 100000,
    num_arrs: int = 1,
    floating_pt: int = 3,
    random_seed: int = 0,
) -> gpd.GeoDataFrame:
    """
    Generate a 2d `numpy` array containing mock numerical data for testing or experimentation purposes.

    Parameters
    ----------
    nx_multigraph: MultiGraph
        A `NetworkX` graph with `x` and `y` attributes. This is used in order to determine the spatial extents of the
        network. The returned data will be within these extents.
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
        An optional random seed.

    Returns
    -------
    GeoDataFrame
        A `GeoDataFrame` with a "mock_numerical_x" data columns for testing purposes. The number of rows will match
        the `length` parameter. The numer of numerical columns will match the `num_arrs` paramter.

    """
    np.random.seed(seed=random_seed)
    data_gpd = mock_data_gdf(nx_multigraph, length=length, random_seed=random_seed)
    for idx in range(1, num_arrs + 1):
        num_arr: npt.NDArray[np.float32] = np.array(
            np.random.randint(val_min, high=val_max, size=length), dtype=np.float32
        )
        num_arr /= 10**floating_pt
        # set last five items to max - this is for duplicate checking
        num_max = np.nanmax(num_arr)
        num_arr[-5:] = num_max
        data_gpd[f"mock_numerical_{idx}"] = num_arr  # pylint: disable=unsupported-assignment-operation
    return data_gpd


def mock_species_data(
    random_seed: int = 0,
) -> Generator[tuple[list[int], list[float]], None, None]:
    """
    Generate a series of randomly generated counts and corresponding probabilities.

    This function is used for testing diversity measures. The data is generated in varying lengths from randomly
    assigned integers between 1 and 10. Matching integers are then collapsed into species "classes" with probabilities
    computed accordingly.

    Parameters
    ----------
    random_seed: int
        An optional random seed.

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
    np.random.seed(seed=random_seed)

    for n in range(1, 50, 5):
        data = np.random.randint(1, 10, n)
        unique: npt.NDArray[np.int_] = np.unique(data)
        counts: npt.NDArray[np.int_] = np.zeros_like(unique, dtype=np.int_)
        for idx, uniq in enumerate(unique):
            counts[idx] = (data == uniq).sum()
        probs = counts / len(data)

        yield counts.tolist(), probs.tolist()
