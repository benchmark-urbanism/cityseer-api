"""
Functions for fetching and converting graphs and network structures.
"""
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional, Union, cast

import fiona
import geopandas as gpd
import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd
import requests
import utm
from pyproj import CRS, Transformer
from shapely import geometry
from tqdm import tqdm

from cityseer import config, rustalgos
from cityseer.tools import graphs, util
from cityseer.tools.util import EdgeData, ListCoordsType, MultiDiGraph, NodeData, NodeKey

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# type hack until networkx supports type-hinting
MultiGraph = Any


def nx_epsg_conversion(nx_multigraph: MultiGraph, from_epsg_code: int, to_epsg_code: int) -> MultiGraph:
    """
    Convert a graph from the `from_epsg_code` EPSG CRS to the `to_epsg_code` EPSG CRS.

    The `to_epsg_code` must be for a projected CRS. If edge `geom` attributes are found, the associated `LineString`
    geometries will also be converted.

    Parameters
    ----------
    nx_multigraph: MultiGraph
        A `networkX` `MultiGraph` with `x` and `y` node attributes in the `from_epsg_code` coordinate system. Optional
        `geom` edge attributes containing `LineString` geoms to be converted.
    from_epsg_code: int
        An integer representing a valid EPSG code specifying the CRS from which the graph must be converted. For
        example, [4326](https://epsg.io/4326) if converting data from an OpenStreetMap response.
    to_epsg_code: int
        An integer representing a valid EPSG code specifying the CRS into which the graph must be projected. For
        example, [27700](https://epsg.io/27700) if converting to British National Grid.

    Returns
    -------
    MultiGraph
        A `networkX` `MultiGraph` with `x` and `y` node attributes converted to the specified `to_epsg_code` coordinate
        system. Edge `geom` attributes will also be converted if found.

    """
    if not isinstance(nx_multigraph, nx.MultiGraph):
        raise TypeError("This method requires an undirected networkX MultiGraph.")
    logger.info(f"Converting networkX graph from EPSG code {from_epsg_code} to EPSG code {to_epsg_code}.")
    g_multi_copy: MultiGraph = nx_multigraph.copy()
    test_crs = CRS.from_epsg(to_epsg_code)
    if not test_crs.is_projected:
        raise ValueError("The to_epsg_code parameter must be for a projected CRS")
    transformer = Transformer.from_crs(from_epsg_code, to_epsg_code, always_xy=True)
    logger.info("Processing node x, y coordinates.")
    nd_key: NodeKey
    node_data: NodeData
    for nd_key, node_data in tqdm(g_multi_copy.nodes(data=True), disable=config.QUIET_MODE):
        # x coordinate
        if "x" not in node_data:
            raise KeyError(f'Encountered node missing "x" coordinate attribute at node {nd_key}.')
        x: float = node_data["x"]
        # y coordinate
        if "y" not in node_data:
            raise KeyError(f'Encountered node missing "y" coordinate attribute at node {nd_key}.')
        y: float = node_data["y"]
        # be cognisant of parameter and return order, using always_xy for transformer
        easting, northing = transformer.transform(x, y)  # pylint: disable=unpacking-non-sequence
        # write back to graph
        g_multi_copy.nodes[nd_key]["x"] = easting
        g_multi_copy.nodes[nd_key]["y"] = northing
    # if line geom property provided, then convert as well
    logger.info("Processing edge geom coordinates, if present.")
    start_nd_key: NodeKey
    end_nd_key: NodeKey
    edge_idx: int
    edge_data: EdgeData
    for start_nd_key, end_nd_key, edge_idx, edge_data in tqdm(
        g_multi_copy.edges(data=True, keys=True), disable=config.QUIET_MODE
    ):
        # check if geom present - optional step
        if "geom" in edge_data:
            line_geom: geometry.LineString = edge_data["geom"]
            if line_geom.geom_type != "LineString":
                raise TypeError(f"Expecting LineString geometry but found {line_geom.geom_type} geometry.")
            # convert
            edge_coords: ListCoordsType = [transformer.transform(x, y) for x, y in line_geom.coords]
            # snap ends
            edge_coords = util.snap_linestring_endpoints(g_multi_copy, start_nd_key, end_nd_key, edge_coords)
            # write back to edge
            g_multi_copy[start_nd_key][end_nd_key][edge_idx]["geom"] = geometry.LineString(edge_coords)

    return g_multi_copy


def nx_wgs_to_utm(nx_multigraph: MultiGraph, force_zone_number: Optional[int] = None) -> MultiGraph:
    """
    Convert a graph from WGS84 geographic coordinates to UTM projected coordinates.

    Converts `x` and `y` node attributes from [WGS84](https://epsg.io/4326) `lng`, `lat` geographic coordinates to the
    local UTM projected coordinate system. If edge `geom` attributes are found, the associated `LineString` geometries
    will also be converted. The UTM zone derived from the first processed node will be used for the conversion of all
    other nodes and geometries contained in the graph. This ensures consistent behaviour in cases where a graph spans
    a UTM boundary.

    Parameters
    ----------
    nx_multigraph: MultiGraph
        A `networkX` `MultiGraph` with `x` and `y` node attributes in the WGS84 coordinate system. Optional `geom` edge
        attributes containing `LineString` geoms to be converted.
    force_zone_number: int
        An optional UTM zone number for coercing all conversions to an explicit UTM zone. Use with caution: mismatched
        UTM zones may introduce substantial distortions in the results.

    Returns
    -------
    MultiGraph
        A `networkX` `MultiGraph` with `x` and `y` node attributes converted to the local UTM coordinate system. If edge
         `geom` attributes are present, these will also be converted.

    """
    # sample the first node for UTM
    utm_zone_number = force_zone_number
    nd_key = list(nx_multigraph.nodes())[0]
    lng = nx_multigraph.nodes[nd_key]["x"]
    lat = nx_multigraph.nodes[nd_key]["y"]
    is_north = lat >= 0
    is_south = lat < 0
    utm_zone_number, _utm_zone_letter = utm.from_latlon(lat, lng)[2:]  # zone number is position 2
    if force_zone_number is not None:
        utm_zone_number = force_zone_number
    # or dictionary
    crs = CRS.from_dict({"proj": "utm", "zone": utm_zone_number, "north": is_north, "south": is_south})
    target_epsg = crs.to_epsg()
    if not isinstance(target_epsg, int):
        raise ValueError("Unable to extract an EPSG code from the provided network.")
    return nx_epsg_conversion(nx_multigraph, 4326, target_epsg)


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
    poly_utm: geometry.Polygon = pnt.buffer(buffer)
    # convert back to WGS
    # the polygon is too big for the OSM server, so have to use convex hull then later prune
    coords = []
    for easting, northing in poly_utm.convex_hull.exterior.coords:
        lat, lng = utm.to_latlon(easting, northing, utm_zone_number, utm_zone_letter)
        coords.append((lng, lat))
    poly_wgs = geometry.Polygon(coords)

    return poly_wgs, poly_utm, utm_zone_number, utm_zone_letter  # type: ignore


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
    osm_response: Optional[requests.Response] | None = None
    while max_tries:
        osm_response = requests.get(
            "https://overpass-api.de/api/interpreter",
            timeout=timeout,
            params={"data": osm_request},
        )
        # break if OK response
        if osm_response is not None and osm_response.status_code == 200:  # type: ignore
            break
        # otherwise try until max_tries is exhausted
        logger.warning("Unsuccessful OSM API request response, trying again...")
        max_tries -= 1
    if osm_response is None:
        raise requests.RequestException("None response. Unsuccessful OSM API request.")
    if not osm_response.status_code == 200:
        osm_response.raise_for_status()

    return osm_response


def osm_graph_from_poly(
    poly_geom: geometry.Polygon,
    poly_epsg_code: int = 4326,
    to_epsg_code: Optional[int] = None,
    buffer_dist: int = 15,
    custom_request: str | None = None,
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
    buffer_dist: int
        A distance to use for buffering and cleaning operations. 15m by default.
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
    if poly_epsg_code is not None and not isinstance(poly_epsg_code, int):  # type: ignore
        raise TypeError('Please provide "poly_epsg_code" parameter as int')
    if to_epsg_code is not None and not isinstance(to_epsg_code, int):
        raise TypeError('Please provide "to_epsg_code" parameter as int')
    # format for OSM query
    in_transformer = Transformer.from_crs(poly_epsg_code, 4326, always_xy=True)
    coords = [in_transformer.transform(lng, lat) for lng, lat in poly_geom.exterior.coords]
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
        ["highway"!~"motorway|motorway_link|bus_guideway|busway|escape|raceway|proposed|planned|abandoned|platform|construction|emergency_bay|rest_area"]
        ["service"!~"parking_aisle|driveway|drive-through|slipway"] ["amenity"!~"charging_station|parking|fuel|motorcycle_parking|parking_entrance|parking_space"]
        ["footway"!~"sidewalk|traffic_island|crossing"]
        ["indoor"!="yes"]
        ["level"!="-2"]
        ["level"!="-3"]
        ["level"!="-4"]
        ["level"!="-5"]
        (poly:"{geom_osm}");
        );
        out body;
        >;
        out qt;
        """
    # generate the query
    osm_response = fetch_osm_network(request, timeout=timeout, max_tries=max_tries)
    # build graph
    graph_wgs = nx_from_osm(osm_json=osm_response.text)  # type: ignore
    # cast to UTM
    if to_epsg_code is not None:
        graph_crs = nx_epsg_conversion(graph_wgs, 4326, to_epsg_code)
    else:
        graph_crs = nx_wgs_to_utm(graph_wgs)
    # simplify
    if simplify:
        graph_crs = graphs.nx_simple_geoms(graph_crs)
        graph_crs = graphs.nx_remove_filler_nodes(graph_crs)
        graph_crs = graphs.nx_remove_dangling_nodes(
            graph_crs, despine=buffer_dist, remove_disconnected=remove_disconnected
        )
        graph_crs = graphs.nx_consolidate_nodes(graph_crs, buffer_dist=buffer_dist, crawl=True)
        if remove_parallel:
            graph_crs = graphs.nx_split_opposing_geoms(graph_crs, buffer_dist=buffer_dist)
            graph_crs = graphs.nx_consolidate_nodes(
                graph_crs, buffer_dist=buffer_dist, crawl=False, neighbour_policy="indirect"
            )
        if iron_edges:
            graph_crs = graphs.nx_iron_edges(graph_crs)

    return graph_crs


def nx_from_osm(osm_json: str) -> MultiGraph:
    """
    Generate a `NetworkX` `MultiGraph` from [Open Street Map](https://www.openstreetmap.org) data.

    Parameters
    ----------
    osm_json: str
        A `json` string response from the [OSM overpass API](https://wiki.openstreetmap.org/wiki/Overpass_API),
        consisting of `nodes` and `ways`.

    Returns
    -------
    MultiGraph
        A `NetworkX` `MultiGraph` with `x` and `y` attributes in [WGS84](https://epsg.io/4326) `lng`, `lat` geographic
        coordinates.

    """
    osm_network_data = json.loads(osm_json)
    nx_multigraph: MultiGraph = nx.MultiGraph()
    # deduplicate nodes based on x, y matches
    xy_nd_map: dict[str, str] = {}  # from x y to original node index
    nd_merge_map: dict[str, str] = {}  # from original node index to deduplicated node index

    for elem in osm_network_data["elements"]:
        if elem["type"] == "node":
            # all nodes should be string type
            nd_idx = str(elem["id"])
            x = elem["lon"]
            y = elem["lat"]
            nd_x_y = f"{x}-{y}"
            # when deduplicating, sometimes overlapping stairways / underpasses might share an x, y space...
            if "tags" in elem and "level" in elem["tags"]:
                nd_x_y = f'{nd_x_y}-{elem["tags"]["level"]}'
            # only add if non-duplicate
            if nd_x_y not in xy_nd_map:
                xy_nd_map[nd_x_y] = nd_idx
            else:
                logger.warning(f"Merging node {nd_idx} into {xy_nd_map[nd_x_y]} due to identical x, y coords.")
            # ok_nd_idx will correspond to deduplicated node index
            ok_nd_idx = xy_nd_map[nd_x_y]
            # track original node idx to deduplicated idx for reference when building edges
            nd_merge_map[nd_idx] = ok_nd_idx
            nx_multigraph.add_node(ok_nd_idx, x=x, y=y)

    def get_merged_nd_keys(_idx: int) -> tuple[str, str]:
        start_nd_key = str(elem["nodes"][_idx])
        end_nd_key = str(elem["nodes"][_idx + 1])
        return nd_merge_map[start_nd_key], nd_merge_map[end_nd_key]

    for elem in osm_network_data["elements"]:
        if elem["type"] == "way":
            count = len(elem["nodes"])
            if "tags" in elem:
                tags = elem["tags"]
                name = tags["name"] if "name" in tags else None
                ref = tags["ref"] if "ref" in tags else None
                highway = tags["highway"] if "highway" in tags else None
                for idx in range(count - 1):
                    start_nd_key, end_nd_key = get_merged_nd_keys(idx)
                    nx_multigraph.add_edge(
                        start_nd_key,
                        end_nd_key,
                        names=[name],
                        routes=[ref],
                        highways=[highway],
                    )
            else:
                for idx in range(count - 1):
                    start_nd_key, end_nd_key = get_merged_nd_keys(idx)
                    nx_multigraph.add_edge(start_nd_key, end_nd_key)

    return nx_multigraph


def nx_from_osm_nx(
    nx_multidigraph: MultiDiGraph,
    node_attributes: Optional[list[str] | tuple[str]] = None,
    edge_attributes: Optional[list[str] | tuple[str]] = None,
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
    if not isinstance(nx_multidigraph, nx.MultiDiGraph):
        raise TypeError("This method requires a directed networkX MultiDiGraph as derived from `OSMnx`.")
    if node_attributes is not None and not isinstance(node_attributes, (list, tuple)):
        raise TypeError("Node attributes to be copied should be provided as either a list or tuple of attribute keys.")
    if edge_attributes is not None and not isinstance(edge_attributes, (list, tuple)):
        raise TypeError("Edge attributes to be copied should be provided as either a list or tuple of attribute keys.")
    logger.info("Converting OSMnx MultiDiGraph to cityseer MultiGraph.")
    # target MultiGraph
    g_multi: MultiGraph = nx.MultiGraph()

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
        if line_geom.geom_type != "LineString":
            raise TypeError(
                f"Expecting LineString geometry but found {line_geom.geom_type} geometry for "
                f"edge {start_nd_key}-{end_nd_key}."
            )
        # orient LineString
        geom_coords = line_geom.coords
        if not np.allclose((s_x, s_y), geom_coords[0][:2], atol=tolerance, rtol=0):
            geom_coords = util.align_linestring_coords(geom_coords, (s_x, s_y))
        # check starting and ending tolerances
        if not np.allclose((s_x, s_y), geom_coords[0][:2], atol=tolerance, rtol=0):
            raise ValueError("Starting node coordinates don't match LineString geometry starting coordinates.")
        if not np.allclose((e_x, e_y), geom_coords[-1][:2], atol=tolerance, rtol=0):
            raise ValueError("Ending node coordinates don't match LineString geometry ending coordinates.")
        # snap starting and ending coords to avoid rounding error issues
        geom_coords = util.snap_linestring_startpoint(geom_coords, (s_x, s_y))
        geom_coords = util.snap_linestring_endpoint(geom_coords, (e_x, e_y))
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
    with fiona.open(open_roads_path, layer="RoadNode") as nodes:
        for node_data in nodes.values(bbox=target_bbox):
            node_id: str = node_data["properties"]["id"]
            x: float
            y: float
            x, y = node_data["geometry"]["coordinates"]
            g_multi.add_node(node_id, x=x, y=y)

    # load the edges
    n_dropped = 0
    with fiona.open(open_roads_path, layer="RoadLink") as edges:
        for edge_data in edges.values(bbox=target_bbox):
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
                highway: str | None = props[highway_key]
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
            geom = geometry.LineString(edge_data["geometry"]["coordinates"])
            geom: geometry.LineString = geom.simplify(5)
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
    g_multi = graphs.nx_merge_parallel_edges(g_multi, True, 10)

    return g_multi


def network_structure_from_nx(
    nx_multigraph: MultiGraph,
    crs: Union[str, int],
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, rustalgos.NetworkStructure]:
    """
    Transpose a `networkX` `MultiGraph` into a `GeoDataFrame` and `NetworkStructure` for use by `cityseer`.

    Calculates length and angle attributes, as well as in and out bearings, and stores this information in the returned
    data maps.

    Parameters
    ----------
    nx_multigraph: MultiGraph
        A `networkX` `MultiGraph` in a projected coordinate system, containing `x` and `y` node attributes, and `geom`
        edge attributes containing `LineString` geoms.
    crs: str | int
        CRS for initialising the returned structures. This is used for initialising the GeoPandas
        [`GeoDataFrame`](https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoDataFrame.html#geopandas-geodataframe).  # pylint: disable=line-too-long

    Returns
    -------
    nodes_gdf: GeoDataFrame
        A `GeoDataFrame` with `live`, `weight`, and `geometry` attributes. The original `networkX` graph's node keys
        will be used for the `GeoDataFrame` index.
    edges_gdf: GeoDataFrame
        A `GeoDataFrame` with `ns_edge_idx`, `start_ns_node_idx`, `end_ns_node_idx`, `edge_idx`, `nx_start_node_key`,
        `nx_end_node_key`, `length`, `angle_sum`, `imp_factor`, `in_bearing`, `out_bearing`, `total_bearing`, `geom`
        attributes.
    network_structure: rustalgos.NetworkStructure
        A [`rustalgos.NetworkStructure`](/rustalgos#networkstructure) instance.

    """
    if not isinstance(nx_multigraph, nx.MultiGraph):
        raise TypeError("This method requires an undirected networkX MultiGraph.")
    logger.info("Preparing node and edge arrays from networkX graph.")
    g_multi_copy: MultiGraph = nx_multigraph.copy()
    # prepare the network structure
    network_structure = rustalgos.NetworkStructure()
    # generate the network information
    agg_node_data: dict[str, tuple[Any, ...]] = {}
    agg_node_dual_data: dict[str, tuple[Any, Any, Any]] = {}
    agg_edge_data: dict[str, tuple[Any, ...]] = {}
    agg_edge_dual_data: list[str] = []
    node_data: NodeData
    # set nodes
    for node_key, node_data in tqdm(g_multi_copy.nodes(data=True), disable=config.QUIET_MODE):
        # node_key must be string
        if not isinstance(node_key, str):
            raise TypeError(f"Node key must be of type string but encountered {type(node_key)}")
        if "x" not in node_data:
            raise KeyError(f'Encountered node missing "x" coordinate attribute at node {node_key}.')
        node_x: float = node_data["x"]
        if "y" not in node_data:
            raise KeyError(f'Encountered node missing "y" coordinate attribute at node {node_key}.')
        node_y: float = node_data["y"]
        is_live: bool = True
        if "live" in node_data:
            is_live = bool(node_data["live"])
        weight = 1
        if "weight" in node_data:
            weight = node_data["weight"]
        # set node
        ns_node_idx = network_structure.add_node(node_key, node_x, node_y, is_live, weight)
        agg_node_data[node_key] = (ns_node_idx, node_x, node_y, is_live, weight, geometry.Point(node_x, node_y))
        if "is_dual" in g_multi_copy.graph and g_multi_copy.graph["is_dual"]:
            agg_node_dual_data[node_key] = (
                node_data["primal_edge_node_a"],
                node_data["primal_edge_node_b"],
                node_data["primal_edge_idx"],
            )
    # set edges
    for start_node_key in tqdm(g_multi_copy.nodes(), disable=config.QUIET_MODE):
        # build edges
        start_ns_node_idx, start_node_x, start_node_y, _, _, _ = agg_node_data[start_node_key]
        end_node_key: str
        for end_node_key in g_multi_copy.neighbors(start_node_key):
            end_ns_node_idx, _, _, _, _, _ = agg_node_data[end_node_key]
            # add the new edge index to the node's out edges
            nx_edge_data: EdgeData
            for edge_idx, nx_edge_data in g_multi_copy[start_node_key][end_node_key].items():
                if not "geom" in nx_edge_data:
                    raise KeyError(
                        f"No edge geom found for edge {start_node_key}-{end_node_key}: Please add an edge 'geom' "
                        "attribute consisting of a shapely LineString. Simple (straight) geometries can be inferred "
                        "automatically through the nx_simple_geoms() method."
                    )
                line_geom = nx_edge_data["geom"]
                if line_geom.geom_type != "LineString":
                    raise TypeError(
                        f"Expecting LineString geometry but found {line_geom.geom_type} geom for edge "
                        f"{start_node_key}-{end_node_key}."
                    )
                # cannot have zero or negative length - division by zero
                line_len = line_geom.length
                if not np.isfinite(line_len) or line_len <= 0:
                    raise ValueError(
                        f"Length {line_len} for edge {start_node_key}-{end_node_key} must be finite and positive."
                    )
                # check geom coordinates directionality (for bearings at index 5 / 6)
                # flip if facing backwards direction
                line_geom_coords = util.align_linestring_coords(line_geom.coords, (start_node_x, start_node_y))
                # iterate the coordinates and calculate the angular change
                angle_sum = util.measure_cumulative_angle(line_geom_coords)
                if not np.isfinite(angle_sum) or angle_sum < 0:
                    raise ValueError(
                        f"Angle sum {angle_sum} for edge {start_node_key}-{end_node_key} must be finite and positive."
                    )
                # if imp_factor is set explicitly, then use
                # fallback imp_factor of 1
                imp_factor: float = 1
                if "imp_factor" in nx_edge_data:
                    # cannot have imp_factor less than zero (but == 0 is OK)
                    imp_factor = nx_edge_data["imp_factor"]
                    if not (np.isfinite(imp_factor) or np.isinf(imp_factor)) or imp_factor < 0:
                        raise ValueError(
                            f"Impedance factor: {imp_factor} for edge {start_node_key}-{end_node_key} must be finite "
                            " and positive or positive infinity."
                        )
                # in bearing
                xy_1: npt.NDArray[np.float_] = np.array(line_geom_coords[0])
                xy_2: npt.NDArray[np.float_] = np.array(line_geom_coords[1])
                in_bearing: float = util.measure_bearing(xy_1, xy_2)
                # out bearing
                xy_3: npt.NDArray[np.float_] = np.array(line_geom_coords[-2])
                xy_4: npt.NDArray[np.float_] = np.array(line_geom_coords[-1])
                out_bearing: float = util.measure_bearing(xy_3, xy_4)
                total_bearing = util.measure_bearing(xy_1, xy_4)
                # set edge
                ns_edge_idx = network_structure.add_edge(
                    start_ns_node_idx,
                    end_ns_node_idx,
                    edge_idx,
                    start_node_key,
                    end_node_key,
                    line_len,
                    angle_sum,
                    imp_factor,
                    in_bearing,
                    out_bearing,
                )
                # add to edge data
                agg_edge_data[f"{start_node_key}-{end_node_key}"] = (
                    ns_edge_idx,
                    start_ns_node_idx,
                    end_ns_node_idx,
                    edge_idx,
                    start_node_key,
                    end_node_key,
                    line_len,
                    angle_sum,
                    imp_factor,
                    in_bearing,
                    out_bearing,
                    total_bearing,
                    line_geom,
                )
                if "is_dual" in g_multi_copy.graph and g_multi_copy.graph["is_dual"]:
                    agg_edge_dual_data.append(nx_edge_data["primal_node_id"])
    # create geopandas for node keys and data state
    nodes_gdf = gpd.GeoDataFrame.from_dict(
        agg_node_data,
        orient="index",
        columns=["ns_node_idx", "x", "y", "live", "weight", "geom"],
        geometry="geom",
        crs=crs,
    )
    edges_gdf = gpd.GeoDataFrame.from_dict(
        agg_edge_data,
        orient="index",
        columns=[
            "ns_edge_idx",
            "start_ns_node_idx",
            "end_ns_node_idx",
            "edge_idx",
            "nx_start_node_key",
            "nx_end_node_key",
            "length",
            "angle_sum",
            "imp_factor",
            "in_bearing",
            "out_bearing",
            "total_bearing",
            "geom",
        ],
        geometry="geom",
        crs=crs,
    )
    if "is_dual" in g_multi_copy.graph and g_multi_copy.graph["is_dual"]:
        nodes_dual_gdf = pd.DataFrame.from_dict(
            agg_node_dual_data,
            orient="index",
            columns=[
                "primal_edge_node_a",
                "primal_edge_node_b",
                "primal_edge_idx",
            ],
        )
        nodes_gdf: gpd.GeoDataFrame = nodes_gdf.join(nodes_dual_gdf)  # type: ignore
        edges_gdf["primal_node_id"] = agg_edge_dual_data
    return nodes_gdf, edges_gdf, network_structure


def nx_from_geopandas(
    nodes_gdf: gpd.GeoDataFrame,
    edges_gdf: gpd.GeoDataFrame,
) -> MultiGraph:
    """
    Write nodes and edges `GeoDataFrames` to a `networkX` `MultiGraph`.

    Parameters
    ----------
    nodes_gdf: GeoDataFrame
        A `GeoDataFrame` with `live`, `weight`, and Point `geometry` attributes. The index will be used for the returned
        `networkX` graph's node keys.
    edges_gdf: GeoDataFrame
        An edges `GeoDataFrame` as derived from [`network_structure_from_nx`](#network-structure-from-nx).

    Returns
    -------
    nx_multigraph: MultiGraph
        A `networkX` graph with geometries and attributes as copied from the input `GeoDataFrames`.

    """
    logger.info("Populating node and edge map data to a networkX graph.")
    g_multi_copy: MultiGraph = nx.MultiGraph()
    g_multi_copy.add_nodes_from(nodes_gdf.index.values.tolist())
    # after above so that errors caught first
    logger.info("Unpacking node data.")
    for nd_key, nd_data in tqdm(nodes_gdf.iterrows(), disable=config.QUIET_MODE):
        g_multi_copy.nodes[nd_key]["x"] = nd_data.x
        g_multi_copy.nodes[nd_key]["y"] = nd_data.y
        g_multi_copy.nodes[nd_key]["live"] = nd_data.live
        g_multi_copy.nodes[nd_key]["weight"] = nd_data.weight
    logger.info("Unpacking edge data.")
    for _, row_data in tqdm(edges_gdf.iterrows(), disable=config.QUIET_MODE):
        g_multi_copy.add_edge(
            row_data.nx_start_node_key,
            row_data.nx_end_node_key,
            row_data.edge_idx,
            length=row_data.length,
            angle_sum=row_data.angle_sum,
            imp_factor=row_data.imp_factor,
            in_bearing=row_data.in_bearing,
            out_bearing=row_data.out_bearing,
            total_bearing=row_data.total_bearing,
            geom=row_data.geom,
        )
    # unpack any metrics written to the nodes
    metrics_column_labels: list[str] = [c for c in nodes_gdf.columns if c.startswith("cc_metric")]
    if metrics_column_labels:
        logger.info("Unpacking metrics to nodes.")
        for metrics_column_label in metrics_column_labels:
            for nd_key, node_row in tqdm(nodes_gdf.iterrows(), disable=config.QUIET_MODE):
                g_multi_copy.nodes[nd_key][metrics_column_label] = node_row[metrics_column_label]

    return g_multi_copy
