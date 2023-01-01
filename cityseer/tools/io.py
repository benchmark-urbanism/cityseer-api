"""
Functions for fetching and cleaning OSM data.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Union, cast

import fiona
import networkx as nx
import numpy as np
import requests
import utm
from pyproj import Transformer
from shapely import geometry
from tqdm import tqdm

from cityseer import config
from cityseer.tools import graphs
from cityseer.tools.graphs import EdgeData, MultiDiGraph, NodeKey

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# type hack until networkx supports type-hinting
MultiGraph = Any


def buffered_point_poly(lng: float, lat: float, buffer: int) -> tuple[geometry.Polygon, geometry.Polygon, int, str]:
    """
    Buffer a point and return a `shapely` Polygon in WGS and UTM coordinates.

    This function can be used to prepare a `poly_wgs` `Polygon` for passing to
    [`osm_graph_from_poly()`](#osm-graph-from-poly).

    Parameters
    ----------
    lng: float
        The longitudinal WGS coordinate in degrees.
    lat: float
        The latitudinal WGS coordinate in degrees.
    buffer: int
        The buffer distance in metres.

    Returns
    -------
    poly_wgs: Polygon
        A `shapely` `Polygon` in WGS coordinates.
    poly_utm: Polygon
        A `shapely` `Polygon` in UTM coordinates.
    utm_zone_number: int
        The UTM zone number used for conversion.
    utm_zone_letter: str
        The UTM zone letter used for conversion.

    """
    # cast the WGS coordinates to UTM prior to buffering
    easting, northing, utm_zone_number, utm_zone_letter = utm.from_latlon(lat, lng)
    logger.info(f"UTM conversion info: UTM zone number: {utm_zone_number}, UTM zone letter: {utm_zone_letter}")
    # create a point, and then buffer
    pnt = geometry.Point(easting, northing)
    poly_utm: geometry.Polygon = pnt.buffer(buffer)  # type: ignore
    # convert back to WGS
    # the polygon is too big for the OSM server, so have to use convex hull then later prune
    coords = []
    for easting, northing in poly_utm.convex_hull.exterior.coords:  # type: ignore  # pylint: disable=no-member
        lat, lng = utm.to_latlon(easting, northing, utm_zone_number, utm_zone_letter)  # type: ignore
        coords.append((lng, lat))
    poly_wgs = geometry.Polygon(coords)

    return poly_wgs, poly_utm, utm_zone_number, utm_zone_letter


def fetch_osm_network(osm_request: str, timeout: int = 300, max_tries: int = 3) -> Optional[requests.Response]:
    """
    Fetches an OSM response.

    :::note
    This function requires a valid OSM request. If you prepare a polygonal extents then it may be easier to use
    [`osm_graph_from_poly()`](#osm-graph-from-poly), which would call this method on your behalf and then
    builds a graph automatically.
    :::

    Parameters
    ----------
    osm_request: str
        A valid OSM request as a string. Use
        [OSM Overpass](https://wiki.openstreetmap.org/wiki/Overpass_API/Overpass_QL) for testing custom queries.
    timeout: int
        Timeout duration for API call in seconds.
    max_tries: int
        The number of attempts to fetch a response before raising.

    Returns
    -------
    requests.Response
        An OSM API response.

    """
    osm_response: Optional[requests.Response] = None
    while max_tries:
        osm_response = requests.get(
            "https://overpass-api.de/api/interpreter",
            timeout=timeout,
            params={"data": osm_request},
        )
        # break if OK response
        if osm_response is not None and osm_response.status_code == 200:
            break
        # otherwise try until max_tries is exhausted
        logger.warning("Unsuccessful OSM API request response, trying again...")
        max_tries -= 1
    if osm_response is None:
        raise requests.RequestException("None response. Unsuccessful OSM API request.")
    if not osm_response.status_code == 200:
        osm_response.raise_for_status()

    return osm_response


def osm_graph_from_poly_wgs() -> None:
    """Deprecated. Please use [`osm_graph_from_poly()`](#osm-graph-from-poly) instead."""
    raise DeprecationWarning("This method is deprecated. Please use osm_graph_from_poly instead.")


def osm_graph_from_poly(
    poly_geom: geometry.Polygon,
    poly_epsg_code: int = 4326,
    to_epsg_code: Optional[int] = None,
    custom_request: Optional[str] = None,
    simplify: bool = True,
    remove_parallel: bool = True,
    iron_edges: bool = True,
    remove_disconnected: bool = True,
    timeout: int = 300,
    max_tries: int = 3,
) -> MultiGraph:  # noqa
    """

    Prepares a `networkX` `MultiGraph` from an OSM request for the specified shapely polygon. This function will
    retrieve the OSM response and will automatically unpack this into a `networkX` graph. Simplification will be applied
    by default, but can be disabled.

    Parameters
    ----------
    poly_geom: shapely.Polygon
        A shapely Polygon representing the extents for which to fetch the OSM network.
    poly_epsg_code: int
        An integer representing a valid EPSG code for the provided polygon. For example, [4326](https://epsg.io/4326) if
        using WGS lng / lat, or [27700](https://epsg.io/27700) if using the British National Grid.
    to_epsg_code: int
        An optional integer representing a valid EPSG code for the generated network returned from this function. If
        this parameter is provided, then the network will be converted to the specified EPSG coordinate reference
        system. If not provided, then the OSM network will be projected into a local UTM coordinate reference system.
    custom_request: str
        An optional custom OSM request. If provided, this must include a "geom_osm" string formatting key for inserting
        the geometry passed to the OSM API query. See the discussion below.
    simplify: bool
        Whether to automatically simplify the OSM graph. Set to False for manual cleaning.
    remove_parallel: bool
        Ignored if simplify is False. Whether to remove parallel roadway segments.
    iron_edges: bool
        Ignored if simplify is False.  Whether to straighten the ends of street segments. This can help to reduce the
        number of artefacts from segment kinks from merging `LineStrings`.
    remove_disconnected: bool
        Ignored if simplify is False.  Whether to remove disconnected components from the network.
    timeout: int
        Timeout duration for API call in seconds.
    max_tries: int
        The number of attempts to fetch a response before raising.

    Returns
    -------
    MultiGraph
        A `networkX` `MultiGraph` with `x` and `y` node attributes that have been converted to UTM. The network will be
        simplified if the `simplify` parameter is `True`.

    Examples
    --------
    The default OSM request will attempt to find all walkable routes. It will ignore motorways and will try to work with
    pedestrianised routes and walkways.

    If you wish to provide your own OSM request, then provide a valid OSM API request as a string. The string must
    contain a `{geom_osm}` string formatting key. This allows for the geometry parameter passed to the OSM API to be
    injected into the request. It is also recommended to not use the `skel` output option so that `cityseer` can use
    street name and highway reference information for cleaning purposes. See
    [OSM Overpass](https://wiki.openstreetmap.org/wiki/Overpass_API/Overpass_QL) for experimenting with custom queries.

    For example, to return only drivable roads, then use a request similar to the following. Notice the `{geom_osm}`
    formatting key and the use of `out qt;` instead of `out skel qt;`.

    ```python
    custom_request = f'''
    [out:json];
    (
    way["highway"]
    ["area"!="yes"]
    ["highway"!~"footway|pedestrian|steps|bus_guideway|escape|raceway|proposed|planned|abandoned|platform|construction"]
    (poly:"{geom_osm}");
    );
    out body;
    >;
    out qt;
    '''
    ```

    """
    # format for OSM query
    in_transformer = Transformer.from_crs(poly_epsg_code, 4326, always_xy=True)
    coords = [in_transformer.transform(lng, lat) for lng, lat in poly_geom.exterior.coords]  # type: ignore
    geom_osm = str.join(" ", [f"{lat} {lng}" for lng, lat in coords])
    if custom_request is not None:
        if "geom_osm" not in custom_request:
            raise ValueError(
                'The provided custom_request does not contain a "geom_osm" formatting key, i.e. (poly:"{geom_osm}") '
                "This key is required for interpolating the generated geometry into the request."
            )
        request = custom_request.format(geom_osm=geom_osm)
    else:
        request = f"""
        /* https://wiki.openstreetmap.org/wiki/Overpass_API/Overpass_QL */
        [out:json];
        (
            way["highway"]
            ["area"!="yes"]
            ["highway"!~"motorway|motorway_link|bus_guideway|escape|raceway|proposed|planned|abandoned|platform|construction"]
            ["service"!~"parking_aisle"]
            ["amenity"!~"charging_station|parking|fuel|motorcycle_parking|parking_entrance|parking_space"]
            ["access"!~"private|customers"]
            ["indoor"!="yes"]
            (poly:"{geom_osm}");
        );
        out body;
        >;
        out qt;
        """
    # generate the query
    osm_response = fetch_osm_network(request, timeout=timeout, max_tries=max_tries)
    # build graph
    graph_wgs = graphs.nx_from_osm(osm_json=osm_response.text)  # type: ignore
    # cast to UTM
    if to_epsg_code is not None:
        graph_crs = graphs.nx_epsg_conversion(graph_wgs, 4326, to_epsg_code)
    else:
        graph_crs = graphs.nx_wgs_to_utm(graph_wgs)
    # simplify
    if simplify:
        graph_crs = graphs.nx_simple_geoms(graph_crs)
        graph_crs = graphs.nx_remove_filler_nodes(graph_crs)
        graph_crs = graphs.nx_remove_dangling_nodes(graph_crs, despine=20, remove_disconnected=remove_disconnected)
        graph_crs = graphs.nx_consolidate_nodes(
            graph_crs, buffer_dist=15, crawl=True, min_node_group=4, cent_min_degree=4, cent_min_names=4
        )
        if remove_parallel:
            graph_crs = graphs.nx_split_opposing_geoms(graph_crs, buffer_dist=15)
            graph_crs = graphs.nx_consolidate_nodes(
                graph_crs, buffer_dist=15, crawl=False, min_node_degree=2, cent_min_degree=4, cent_min_names=4
            )
        if iron_edges:
            graph_crs = graphs.nx_iron_edges(graph_crs)

    return graph_crs


def nx_from_osm_nx(
    nx_multidigraph: MultiDiGraph,
    node_attributes: Optional[Union[list[str], tuple[str]]] = None,
    edge_attributes: Optional[Union[list[str], tuple[str]]] = None,
    tolerance: float = config.ATOL,
) -> MultiGraph:
    """
    Copy an [`OSMnx`](https://osmnx.readthedocs.io/) directed `MultiDiGraph` to an undirected `cityseer` `MultiGraph`.

    See the [`OSMnx`](/guide#osm-and-networkx) section of the guide for a more general discussion (and example) on
    workflows combining `OSMnx` with `cityseer`.

    `x` and `y` node attributes will be copied directly and `geometry` edge attributes will be copied to a `geom` edge
    attribute. The conversion process will snap the `shapely` `LineString` endpoints to the corresponding start and end
    node coordinates.

    Note that `OSMnx` `geometry` attributes only exist for simplified edges: if a `geometry` edge attribute is not
    found, then a simple (straight) `shapely` `LineString` geometry will be inferred from the respective start and end
    nodes.

    Other attributes will be ignored to avoid potential downstream misinterpretations of the attributes as a consequence
    of subsequent steps of graph manipulation, i.e. to avoid situations where attributes may fall out of lock-step with
    the state of the graph. If particular attributes need to be copied across, and assuming cognisance of downstream
    implications, then these can be manually specified by providing a list of node attributes keys per the
    `node_attributes` parameter or edge attribute keys per the `edge_attributes` parameter.

    Parameters
    ----------
    nx_multidigraph: MultiDiGraph
        A `OSMnx` derived `networkX` `MultiDiGraph` containing `x` and `y` node attributes, with optional `geometry`
        edge attributes containing `LineString` geoms (for simplified edges).
    node_attributes: tuple[str]
        Optional node attributes to copy to the new MultiGraph. (In addition to the default `x` and `y` attributes.)
    edge_attributes: tuple[str]
        Optional edge attributes to copy to the new MultiGraph. (In addition to the optional `geometry` attribute.)
    tolerance: float
        Tolerance at which to raise errors for mismatched geometry end-points vis-a-vis corresponding node coordinates.
        Prior to conversion, this method will check edge geometry end-points for alignment with the corresponding
        end-point nodes. Where these don't align within the given tolerance an exception will be raised. Otherwise, if
        within the tolerance, the conversion function will snap the geometry end-points to the corresponding node
        coordinates so that downstream exceptions are not subsequently raised. It is preferable to minimise graph
        manipulation prior to conversion to a `cityseer` compatible `MultiGraph` otherwise particularly large tolerances
        may be required, and this may lead to some unexpected or undesirable effects due to aggressive snapping.

    Returns
    -------
    MultiGraph
        A `cityseer` compatible `networkX` graph with `x` and `y` node attributes and `geom` edge attributes.

    """
    if not isinstance(nx_multidigraph, nx.MultiDiGraph):  # type: ignore
        raise TypeError("This method requires a directed networkX MultiDiGraph as derived from `OSMnx`.")
    if node_attributes is not None and not isinstance(node_attributes, (list, tuple)):
        raise TypeError("Node attributes to be copied should be provided as either a list or tuple of attribute keys.")
    if edge_attributes is not None and not isinstance(edge_attributes, (list, tuple)):
        raise TypeError("Edge attributes to be copied should be provided as either a list or tuple of attribute keys.")
    logger.info("Converting OSMnx MultiDiGraph to cityseer MultiGraph.")
    # target MultiGraph
    g_multi: MultiGraph = nx.MultiGraph()  # type: ignore

    def _process_node(nd_key: NodeKey) -> tuple[float, float]:
        # x
        if "x" not in nx_multidigraph.nodes[nd_key]:
            raise KeyError(f'Encountered node missing "x" coordinate attribute for node {nd_key}.')
        x: float = nx_multidigraph.nodes[nd_key]["x"]
        # y
        if "y" not in nx_multidigraph.nodes[nd_key]:
            raise KeyError(f'Encountered node missing "y" coordinate attribute for node {nd_key}.')
        y: float = nx_multidigraph.nodes[nd_key]["y"]
        # add attributes if necessary
        if nd_key not in g_multi:
            g_multi.add_node(nd_key, x=x, y=y)
            if node_attributes is not None:
                for node_att in node_attributes:
                    if node_att not in nx_multidigraph.nodes[nd_key]:
                        raise ValueError(f"Specified attribute {node_att} is not available for node {nd_key}.")
                    g_multi.nodes[nd_key][node_att] = nx_multidigraph.nodes[nd_key][node_att]

        return x, y

    # copy nodes and edges
    start_nd_key: NodeKey
    end_nd_key: NodeKey
    edge_idx: int
    edge_data: EdgeData
    for start_nd_key, end_nd_key, edge_idx, edge_data in tqdm(  # type: ignore
        nx_multidigraph.edges(data=True, keys=True), disable=config.QUIET_MODE
    ):
        edge_data = cast(EdgeData, edge_data)  # type: ignore
        s_x, s_y = _process_node(start_nd_key)
        e_x, e_y = _process_node(end_nd_key)
        # copy edge if present
        if "geometry" in edge_data:
            line_geom: geometry.LineString = edge_data["geometry"]
        # otherwise create
        else:
            line_geom = geometry.LineString([[s_x, s_y], [e_x, e_y]])
        # check for LineString validity
        if line_geom.type != "LineString":
            raise TypeError(
                f"Expecting LineString geometry but found {line_geom.type} geometry for "
                f"edge {start_nd_key}-{end_nd_key}."
            )
        # orient LineString
        geom_coords = line_geom.coords
        if not np.allclose((s_x, s_y), geom_coords[0][:2], atol=tolerance, rtol=0):
            geom_coords = graphs.align_linestring_coords(geom_coords, (s_x, s_y))
        # check starting and ending tolerances
        if not np.allclose((s_x, s_y), geom_coords[0][:2], atol=tolerance, rtol=0):
            raise ValueError("Starting node coordinates don't match LineString geometry starting coordinates.")
        if not np.allclose((e_x, e_y), geom_coords[-1][:2], atol=tolerance, rtol=0):
            raise ValueError("Ending node coordinates don't match LineString geometry ending coordinates.")
        # snap starting and ending coords to avoid rounding error issues
        geom_coords = graphs.snap_linestring_startpoint(geom_coords, (s_x, s_y))
        geom_coords = graphs.snap_linestring_endpoint(geom_coords, (e_x, e_y))
        g_multi.add_edge(start_nd_key, end_nd_key, key=edge_idx, geom=geometry.LineString(geom_coords))
        if edge_attributes is not None:
            for edge_att in edge_attributes:
                if edge_att not in edge_data:
                    raise ValueError(f"Attribute {edge_att} is not available for edge {start_nd_key}-{end_nd_key}.")
                g_multi[start_nd_key][end_nd_key][edge_idx][edge_att] = edge_data[edge_att]

    return g_multi


BboxType = Union[tuple[int, int, int, int], tuple[float, float, float, float]]


def nx_from_open_roads(
    open_roads_path: Union[str, Path],
    target_bbox: Optional[BboxType] = None,
) -> nx.MultiGraph:
    """
    Generates a `networkX` `MultiGraph` from an OS Open Roads dataset.

    Parameters
    ----------
    open_roads_path: str | Path
        A valid relative filepath from which to load the OS Open Roads dataset.
    target_bbox: tuple[int]
        A tuple of integers or floats representing the `[s, w, n, e]` bounding box extents for which to load the
        dataset. Set to `None` for no bounding box.

    Returns
    -------
    MultiGraph
        A `cityseer` compatible `networkX` graph with `x` and `y` node attributes and `geom` edge attributes.

    """
    # create a networkX multigraph
    g_multi = nx.MultiGraph()

    # load the nodes
    with fiona.open(open_roads_path, layer="RoadNode") as nodes:  # type: ignore
        for node_data in nodes.values(bbox=target_bbox):  # type: ignore
            node_id: str = node_data["properties"]["id"]
            x: float
            y: float
            x, y = node_data["geometry"]["coordinates"]
            g_multi.add_node(node_id, x=x, y=y)  # type: ignore

    # load the edges
    n_dropped = 0
    with fiona.open(open_roads_path, layer="RoadLink") as edges:  # type: ignore
        for edge_data in edges.values(bbox=target_bbox):  # type: ignore
            # x, y = edge_data['geometry']['coordinates']
            props: dict = edge_data["properties"]  # type: ignore
            start_nd: str = props["startNode"]
            end_nd: str = props["endNode"]
            names: set[str] = set()
            for name_key in ["name1", "name2"]:
                name: str | None = props[name_key]
                if name is not None:
                    names.add(name)
            routes: set[str] = set()
            for ref_key in ["roadClassificationNumber"]:
                ref: str | None = props[ref_key]
                if ref is not None:
                    routes.add(ref)
            highways: set[str] = set()
            for highway_key in ["roadFunction", "roadClassification"]:  # 'formOfWay'
                highway: str = props[highway_key]
                if highway is not None:
                    highways.add(highway)
            if props["trunkRoad"]:
                highways.add("Trunk Road")
            if props["primaryRoute"]:
                highways.add("Primary Road")
            # filter out unwanted highway tags
            highways.difference_update(
                {
                    "Not Classified",
                    "Unclassified",
                    "Unknown",
                    "Restricted Local Access Road",
                    "Local Road",
                    "Classified Unnumbered",
                }
            )
            # create the geometry
            geom = geometry.LineString(edge_data["geometry"]["coordinates"])  # type: ignore
            geom = geom.simplify(5)
            # do not add edges to clipped extents
            if start_nd not in g_multi or end_nd not in g_multi:
                n_dropped += 1
                continue
            g_multi.add_edge(
                start_nd, end_nd, names=list(names), routes=list(routes), highways=list(highways), geom=geom
            )

    logger.info(f"Nodes: {g_multi.number_of_nodes()}")
    logger.info(f"Edges: {g_multi.number_of_edges()}")
    logger.info(f"Dropped {n_dropped} edges where not both start and end nodes were present.")
    logger.info("Running basic graph cleaning")
    g_multi = graphs.nx_remove_filler_nodes(g_multi)
    g_multi = graphs.merge_parallel_edges(g_multi, True, 10)

    return g_multi
