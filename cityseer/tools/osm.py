"""
Functions for fetching and cleaning OSM data.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import requests
import utm
from shapely import geometry

from cityseer.tools import graphs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# type hack until networkx supports type-hinting
MultiGraph = Any


def buffered_point_poly(lng: float, lat: float, buffer: int) -> tuple[geometry.Polygon, geometry.Polygon, int, str]:
    """
    Buffer a point and return a `shapely` Polygon in WGS and UTM coordinates.

    This function can be used to prepare a `poly_wgs` `Polygon` for passing to
    [`osm_graph_from_poly_wgs()`](#osm_graph_from_poly_wgs).

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
    geom = [
        utm.to_latlon(east, north, utm_zone_number, utm_zone_letter)  # type: ignore
        for east, north in poly_utm.convex_hull.exterior.coords  # type: ignore # pylint: disable=no-member
    ]
    poly_wgs = geometry.Polygon(geom)

    return poly_wgs, poly_utm, utm_zone_number, utm_zone_letter


def fetch_osm_network(osm_request: str, timeout: int = 30, max_tries: int = 3) -> Optional[requests.Response]:
    """
    Fetches an OSM response.

    Parameters
    ----------
    osm_request: str
        A valid OSM request as a string. Use
        [OSM Overpass](https://wiki.openstreetmap.org/wiki/Overpass_API/Overpass_QL) for testing custom queries.
    timeout: int
        Timeout duration for API call in seconds.
    max_tries: int
        The number of attempts to fetch a response before raising, by default 3

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

    if osm_response is None or not osm_response.status_code == 200:
        raise requests.RequestException("Unsuccessful OSM API request.")

    return osm_response


def osm_graph_from_poly_wgs(
    poly_wgs: geometry.Polygon,
    custom_request: Optional[str] = None,
    simplify: bool = True,
    remove_parallel: bool = True,
    iron_edges: bool = True,
) -> MultiGraph:  # noqa
    """

    Prepares a `networkX` `MultiGraph` from an OSM request for a buffered region around a given `lng` and `lat`
    parameter.

    Parameters
    ----------
    poly_wgs: shapely.Polygon
        A shapely Polygon representing the extents for which to fetch the OSM network. Must be in WGS (EPSG 4326)
        coordinates.
    custom_request: str
        An optional custom OSM request. None by default. If provided, this must include a "geom_osm" string formatting
        key for inserting the geometry passed to the OSM API query. See the discussion below.
    simplify: bool
        Whether to automatically simplify the OSM graph. True by default. Set to False for manual cleaning.
    remove_parallel: bool
        Whether to remove parallel roadway segments. True by default. Only has an effect if `simplify` is `True`.
    iron_edges: bool
        Whether to straighten the ends of street segments. This can help to reduce the number of artefacts from
        segment kinks from merging `LineStrings`. Only has an effect if `simplify` is `True`.

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
    geom_osm = str.join(" ", [f"{lat} {lng}" for lat, lng in poly_wgs.exterior.coords])  # type: ignore
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
    osm_response = fetch_osm_network(request)
    # build graph
    graph_wgs = graphs.nx_from_osm(osm_json=osm_response.text)  # type: ignore
    # cast to UTM
    graph_utm = graphs.nx_wgs_to_utm(graph_wgs)
    # simplify
    if simplify:
        graph_utm = graphs.nx_simple_geoms(graph_utm)
        graph_utm = graphs.nx_remove_filler_nodes(graph_utm)
        graph_utm = graphs.nx_remove_dangling_nodes(graph_utm, despine=20, remove_disconnected=True)
        graph_utm = graphs.nx_remove_filler_nodes(graph_utm)
        graph_utm = graphs.nx_consolidate_nodes(
            graph_utm, crawl=True, buffer_dist=10, min_node_group=3, cent_min_degree=4, cent_min_names=4
        )

        if remove_parallel:
            graph_utm = graphs.nx_split_opposing_geoms(graph_utm, buffer_dist=15)
            graph_utm = graphs.nx_consolidate_nodes(
                graph_utm, buffer_dist=15, crawl=False, min_node_degree=2, cent_min_degree=4, cent_min_names=4
            )
            graph_utm = graphs.nx_remove_filler_nodes(graph_utm)

        if iron_edges:
            graph_utm = graphs.nx_iron_edge_ends(graph_utm)

    return graph_utm
