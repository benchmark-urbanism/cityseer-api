"""
Convenience functions for the preparation and conversion of `networkX` graphs to and from `cityseer` data structures.

Note that the `cityseer` network data structures can be created and manipulated directly, if so desired.

"""

# workaround until networkx adopts types
# pyright: basic

from __future__ import annotations

import logging
from typing import Any

import networkx as nx
import numpy as np
from shapely import BufferCapStyle, geometry, ops
from tqdm import tqdm

from cityseer import config
from cityseer.tools import util
from cityseer.tools.util import EdgeData, ListCoordsType, MultiGraph, NodeData, NodeKey

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def nx_simple_geoms(nx_multigraph: MultiGraph) -> MultiGraph:
    """
    Inferring geometries from node to node.

    Infers straight-lined geometries connecting the `x` and `y` coordinates of each node-pair. The resultant edge
    geometry will be stored to each edge's `geom` attribute.

    Parameters
    ----------
    nx_multigraph: MultiGraph
        A `networkX` `MultiGraph` with `x` and `y` node attributes.

    Returns
    -------
    MultiGraph
        A `networkX` `MultiGraph` with `shapely`
        [`Linestring`](https://shapely.readthedocs.io/en/latest/manual.html#linestrings) geometries assigned to the edge
        `geom` attributes.

    """
    if not isinstance(nx_multigraph, nx.MultiGraph):
        raise TypeError("This method requires an undirected networkX MultiGraph.")
    logger.info("Generating interpolated edge geometries.")
    g_multi_copy: MultiGraph = nx_multigraph.copy()

    def _process_node(nd_key: NodeKey):
        # x coordinate
        if "x" not in g_multi_copy.nodes[nd_key]:
            raise KeyError(f'Encountered node missing "x" coordinate attribute at node {nd_key}.')
        x: float = g_multi_copy.nodes[nd_key]["x"]
        # y coordinate
        if "y" not in g_multi_copy.nodes[nd_key]:
            raise KeyError(f'Encountered node missing "y" coordinate attribute at node {nd_key}.')
        y: float = g_multi_copy.nodes[nd_key]["y"]

        return x, y

    # unpack coordinates and build simple edge geoms
    remove_edges: list[tuple[NodeKey, NodeKey, int]] = []
    start_nd_key: NodeKey
    end_nd_key: NodeKey
    edge_idx: int
    for start_nd_key, end_nd_key, edge_idx in tqdm(g_multi_copy.edges(keys=True), disable=config.QUIET_MODE):
        s_x, s_y = _process_node(start_nd_key)
        e_x, e_y = _process_node(end_nd_key)
        seg = geometry.LineString([[s_x, s_y], [e_x, e_y]])
        if start_nd_key == end_nd_key and seg.length == 0:
            remove_edges.append((start_nd_key, end_nd_key, edge_idx))
        else:
            g_multi_copy[start_nd_key][end_nd_key][edge_idx]["geom"] = seg
    for start_nd_key, end_nd_key, edge_idx in remove_edges:
        logger.warning(f"Found zero length looped edge for node {start_nd_key}, removing from graph.")
        g_multi_copy.remove_edge(start_nd_key, end_nd_key, key=edge_idx)

    return g_multi_copy


def nx_remove_filler_nodes(nx_multigraph: MultiGraph) -> MultiGraph:
    """
    Remove nodes of degree=2.

    Nodes of degree=2 represent no route-choice options other than traversal to the next edge. These are frequently
    found on network topologies as a means of describing roadway geometry, but are meaningless from a network topology
    point of view. This method will find and deleted these nodes, and replaces the two edges on either side with a new
    spliced edge. The new edge's `geom` attribute will retain the geometric properties of the original edges.

    :::note
    Filler nodes may be prevalent in poor quality datasets, or in situations where curved roadways have been represented
    through the addition of nodes to describe arced geometries. `cityseer` uses `shapely` `Linestrings` to describe
    arbitrary road geometries without the need for filler nodes. Filler nodes can therefore be removed, thus reducing
    side-effects as a function of varied node intensities when computing network centralities.
    :::

    Parameters
    ----------
    nx_multigraph: MultiGraph
        A `networkX` `MultiGraph` in a projected coordinate system, containing `x` and `y` node attributes, and `geom`
        edge attributes containing `LineString` geoms.

    Returns
    -------
    MultiGraph
        A `networkX` `MultiGraph` with nodes of degree=2 removed. Adjacent edges will be combined into a unified new
        edge with associated `geom` attributes spliced together.

    """
    if not isinstance(nx_multigraph, nx.MultiGraph):
        raise TypeError("This method requires an undirected networkX MultiGraph.")
    logger.info("Removing filler nodes.")
    g_multi_copy: MultiGraph = nx_multigraph.copy()
    removed_nodes: set[NodeKey] = set()
    # iterates the original graph, but changes are written to the copied version (to avoid in-place snafus)
    nd_key: NodeKey
    for nd_key in tqdm(sorted(nx_multigraph.nodes()), disable=config.QUIET_MODE):
        # some nodes will already have been removed
        if nd_key in removed_nodes:
            continue
        # proceed if a "simple" node is discovered, i.e. degree = 2
        if nx.degree(nx_multigraph, nd_key) == 2:
            # pick the first neighbour and follow the chain until a non-simple node is encountered
            # this will become the starting point of the chain of simple nodes to be consolidated
            nbs: list[NodeKey] = sorted(list(nx.neighbors(nx_multigraph, nd_key)))
            # catch the edge case where the a single dead-end node has two out-edges to a single neighbour
            if len(nbs) == 1:
                continue
            # otherwise randomly select one side and find a non-simple node as a starting point.
            nb_nd_key = nbs[0]
            # anchor_nd should be the first node of the chain of nodes to be merged, and should be a non-simple node
            anchor_nd: NodeKey | None = None
            # next_link_nd should be a direct neighbour of anchor_nd and must be a simple node
            next_link_nd: NodeKey = nd_key
            # find the non-simple start node
            while anchor_nd is None:
                # follow the chain of neighbours and break once a non-simple node is found
                # catch disconnected looping components by checking for re-encountering start-node
                if nx.degree(nx_multigraph, nb_nd_key) != 2 or nb_nd_key == nd_key:
                    anchor_nd = nb_nd_key
                    break
                # probe neighbours in one-direction only - i.e. don't backtrack
                nb_a: NodeKey
                nb_b: NodeKey
                nb_a, nb_b = sorted(list(nx.neighbors(nx_multigraph, nb_nd_key)))
                if nb_a == next_link_nd:
                    next_link_nd = nb_nd_key
                    nb_nd_key = nb_b
                else:
                    next_link_nd = nb_nd_key
                    nb_nd_key = nb_a
            # from anchor_nd, proceed along the chain in the next_link_nd direction
            # accumulate and weld geometries along the way
            # break once finding another non-simple node
            trailing_nd: NodeKey = anchor_nd
            end_nd: NodeKey | None = None
            drop_nodes: list[NodeKey] = []
            agg_geom: ListCoordsType = []
            edge_info = util.EdgeInfo()
            while True:
                # cast to list and take first in cases where key at index 0 may have been deleted
                edge_data: EdgeData = list(nx_multigraph[trailing_nd][next_link_nd].values())[0]
                edge_info.gather_edge_info(edge_data)
                # aggregate the geom
                try:
                    # there is ordinarily a single edge from trailing to next
                    # however, there is an edge case where next is a dead-end with two edges linking back to trailing
                    # (i.e. where one of those edges is longer than the maximum length discrepancy for merging edges)
                    # in either case, use the first geom
                    geom: geometry.LineString = edge_data["geom"]
                except KeyError as err:
                    raise KeyError(f'Missing "geom" attribute for edge {trailing_nd}-{next_link_nd}') from err
                if geom.geom_type != "LineString":
                    raise TypeError(f"Expecting LineString geometry but found {geom.geom_type} geometry.")
                # welds can be done automatically, but there are edge cases, e.g.:
                # looped roadways or overlapping edges such as stairways don't know which sides of two segments to join
                # i.e. in these cases the edges can sometimes be matched from one of two possible configurations
                # since the x_y join is known for all cases it is used here regardless
                trailing_nd_data: NodeData = nx_multigraph.nodes[trailing_nd]
                override_xy = (trailing_nd_data["x"], trailing_nd_data["y"])
                # weld
                agg_geom = util.weld_linestring_coords(agg_geom, geom.coords, force_xy=override_xy)
                # if the next node has a degree other than 2, then break
                # for circular components, break if the next node matches the start node
                if nx.degree(nx_multigraph, next_link_nd) != 2 or next_link_nd == anchor_nd:
                    end_nd = next_link_nd
                    break
                # otherwise, follow the chain
                # add next_link_nd to drop list
                drop_nodes.append(next_link_nd)
                # get the next set of neighbours
                # in the above-mentioned edge-case, a single dead-end node with two edges back to a start node
                # will only have one neighbour
                new_nbs: list[NodeKey] = sorted(list(nx.neighbors(nx_multigraph, next_link_nd)))
                if len(new_nbs) == 1:
                    trailing_nd = next_link_nd
                    next_link_nd = new_nbs[0]
                # but in almost all cases there will be two neighbours, one of which will be the previous node
                else:
                    nb_a, nb_b = sorted(list(nx.neighbors(nx_multigraph, next_link_nd)))
                    # proceed to the new_next node
                    if nb_a == trailing_nd:
                        trailing_nd = next_link_nd
                        next_link_nd = nb_b
                    else:
                        trailing_nd = next_link_nd
                        next_link_nd = nb_a
            # checks and snapping
            agg_geom = util.snap_linestring_endpoints(g_multi_copy, anchor_nd, end_nd, agg_geom)
            # create a new linestring
            new_geom = geometry.LineString(agg_geom)
            if new_geom.geom_type != "LineString":
                raise TypeError(
                    f'Found {new_geom.geom_type} geometry instead of "LineString" for new geom {new_geom.wkt}.'
                    f"Check that the adjacent LineStrings in the vicinity of {nd_key} are not corrupted."
                )
            # add a new edge from anchor_nd to end_nd
            edge_idx = g_multi_copy.add_edge(
                anchor_nd,
                end_nd,
                geom=new_geom,
            )
            edge_info.set_edge_info(g_multi_copy, anchor_nd, end_nd, edge_idx)
            # drop the removed nodes, which will also implicitly drop the related edges
            g_multi_copy.remove_nodes_from(drop_nodes)
            removed_nodes.update(drop_nodes)

    return g_multi_copy


def nx_remove_dangling_nodes(
    nx_multigraph: MultiGraph, despine: int = 15, remove_disconnected: int = 100
) -> MultiGraph:
    """
    Remove disconnected components and optionally removes short dead-end street stubs.

    Parameters
    ----------
    nx_multigraph: MultiGraph
        A `networkX` `MultiGraph` in a projected coordinate system, containing `x` and `y` node attributes, and `geom`
        edge attributes containing `LineString` geoms.
    despine: int
        The maximum cutoff distance for removal of dead-ends. Use `0` where no despining should occur.
    remove_disconnected: int
        Remove disconnected components with fewer nodes than specified by this parameter. Defaults to 100. Set to 0 to
        keep all disconnected components.

    Returns
    -------
    MultiGraph
        A `networkX` `MultiGraph` with disconnected components optionally removed, and dead-ends removed where less than
         the `despine` parameter distance.

    """
    logger.info("Removing dangling nodes.")
    if remove_disconnected > len(nx_multigraph):
        logger.warning(
            f"An empty graph will be returned because the graph contains {len(nx_multigraph)} nodes, which is fewer "
            f"specified by the remove_disconnected parameter, which is currently set to: {remove_disconnected}. "
            "Decrease the remove_disconnected parameter or set to zero to retain graph components."
        )
    g_multi_copy = nx_multigraph.copy()

    # remove danglers
    if despine > 0:
        remove_nodes = []
        nd_key: NodeKey
        for nd_key in tqdm(g_multi_copy.nodes(data=False), disable=config.QUIET_MODE):
            if nx.degree(g_multi_copy, nd_key) == 1:
                # only a single neighbour, so index-in directly and update at first neighbour
                nb_nd_key: NodeKey = list(nx.neighbors(g_multi_copy, nd_key))[0]
                # cast to list and take first in cases where key at index 0 may have been deleted
                edge_data = list(g_multi_copy[nd_key][nb_nd_key].values())[0]
                if (
                    edge_data["geom"].length <= despine
                    or ("is_tunnel" in edge_data and edge_data["is_tunnel"] is True)
                    or ("is_bridge" in edge_data and edge_data["is_bridge"] is True)
                ):
                    remove_nodes.append(nd_key)
        g_multi_copy.remove_nodes_from(remove_nodes)

    # clean up nodes at ex-dangler intersections
    g_multi_copy = nx_remove_filler_nodes(g_multi_copy)

    # finds connected components - this behaviour changed with networkx v2.4
    # do this after to prevent creation of new isolated components after dropping tunnels
    connected_components = list(nx.algorithms.components.connected_components(g_multi_copy))
    # keep connected components greater than remove_disconnected param
    large_components = [component for component in connected_components if len(component) >= remove_disconnected]
    large_subgraphs = [g_multi_copy.subgraph(component).copy() for component in large_components]
    if not large_subgraphs:
        logger.warning(
            f"An empty graph will be returned because all graph components had fewer than {remove_disconnected} nodes. "
            "Decrease the remove_disconnected parameter or set to zero to retain graph components."
        )
    # make a copy of the graph using the largest component
    g_multi_large = nx.MultiGraph()
    for subgraph in large_subgraphs:
        g_multi_large.add_nodes_from(subgraph.nodes(data=True))
        g_multi_large.add_edges_from(subgraph.edges(data=True))

    return g_multi_large


def _extract_tags_to_set(
    tags_list: list[str] | None = None,
) -> set[str | int]:
    """Converts a `list` of `str` tags to a `set` of small caps `str`."""
    tags = set()
    if tags_list is not None:
        if not isinstance(tags_list, list | set | tuple):
            raise ValueError(f"Tags should be provided as a `list` of `str` instead of {type(tags_list)}.")
        cleaned_tags_list = []
        for t in tags_list:
            if isinstance(t, str):
                if t not in ["", " ", None]:
                    cleaned_tags_list.append(t.strip().lower())
            else:
                cleaned_tags_list.append(t)
        tags.update(tags_list)
    return tags


def _tags_from_edge_key(edge_data: EdgeData, edge_key: str) -> set[str | int]:
    """Fetches tags from a given edge key and returns as `set` of `str`."""
    if edge_key in edge_data:
        return _extract_tags_to_set(edge_data[edge_key])
    return set()


def _gather_nb_tags(nx_multigraph: MultiGraph, nd_key: NodeKey, edge_key: str) -> set[str | int]:
    """Fetches tags from edges neighbouring a node and returns as a `set` of `str`."""
    nb_tags = set()
    for nb_nd_key in nx_multigraph.neighbors(nd_key):
        for edge_data in nx_multigraph[nd_key][nb_nd_key].values():
            nb_tags.update(_tags_from_edge_key(edge_data, edge_key))
    return nb_tags


def _gather_name_tags(edge_data: EdgeData) -> set[str | int]:
    """Fetches `names` and `routes` tags from the provided edge and returns as a `set` of `str`."""
    names_tags = _tags_from_edge_key(edge_data, "names")
    routes_tags = _tags_from_edge_key(edge_data, "routes")
    return names_tags.union(routes_tags)


def _gather_nb_name_tags(nx_multigraph: MultiGraph, nd_key: NodeKey) -> set[str | int]:
    """Fetches `names` and `routes` tags from edges neighbouring a node and returns as a `set` of `str`."""
    names_tags = _gather_nb_tags(nx_multigraph, nd_key, "names")
    routes_tags = _gather_nb_tags(nx_multigraph, nd_key, "routes")
    return names_tags.union(routes_tags)


def nx_merge_parallel_edges(
    nx_multigraph: MultiGraph,
    merge_edges_by_midline: bool,
    contains_buffer_dist: int,
    osm_hwy_target_tags: list[str] | None = None,
    osm_matched_tags_only: bool = False,
) -> MultiGraph:
    """
    Check a MultiGraph for duplicate edges; which, if found, will be merged.

    The shortest of these parallel edges is selected and buffered by `contains_buffer_dist`. If this buffer contains an
    adjacent edge, then the adjacent edge is merged. Edges falling outside this buffer are retained.

    When candidate edges are found for merging, they are replaced by a single new edge. The new geometry selected from
    either:
    - An imaginary centreline of the combined edges if `merge_edges_by_midline` is set to `True`;
    - Else, the shortest edge is retained, with longer edges discarded.

    Parameters
    ----------
    nx_multigraph: MultiGraph
        A `networkX` `MultiGraph` in a projected coordinate system, containing `x` and `y` node attributes, and `geom`
        edge attributes containing `LineString` geoms.
    merge_edges_by_midline: bool
        Whether to merge parallel edges by an imaginary centreline. If set to False, then the shortest edge will be
        retained as the new geometry and the longer edges will be discarded. Defaults to True.
    contains_buffer_dist: int
        The buffer distance to consider when checking if parallel edges sharing the same start and end nodes are
        sufficiently adjacent to be merged.
    osm_hwy_target_tags: list[str]
        An optional list of OpenStreetMap target highway tags. If provided, only nodes with neighbouring edges
        containing a tag matching one of the target OSM highway tags will be consolidated. Requires graph prepared with
        via [`io.osm_graph_from_poly`](/io#osm-graph-from-poly).
    osm_matched_tags_only: bool
        Whether to only merge edges with shared OSM `name` or `ref` tags. False by default. Requires graph prepared with
        via [`io.osm_graph_from_poly`](/io#osm-graph-from-poly).

    Returns
    -------
    MultiGraph
        A `networkX` `MultiGraph` with consolidated nodes.

    """
    if not isinstance(nx_multigraph, nx.MultiGraph):
        raise TypeError("This method requires an undirected networkX MultiGraph (for multiple edges).")
    if contains_buffer_dist < 1:
        raise TypeError("contains_buffer_dist should be greater or equal to 1. ")
    logger.info(f"Merging parallel edges within buffer of {contains_buffer_dist}.")
    # don't use copy() - add nodes only
    deduped_graph: MultiGraph = nx.MultiGraph()
    deduped_graph.add_nodes_from(nx_multigraph.nodes(data=True))
    # if using OSM tags heuristic
    hwy_tags = _extract_tags_to_set(osm_hwy_target_tags)
    # iter the edges
    start_nd_key: NodeKey
    end_nd_key: NodeKey
    edge_data: EdgeData
    for start_nd_key, end_nd_key, edge_data in tqdm(nx_multigraph.edges(data=True), disable=config.QUIET_MODE):
        # if only one edge is associated with this node pair, then add
        if nx_multigraph.number_of_edges(start_nd_key, end_nd_key) == 1:
            deduped_graph.add_edge(start_nd_key, end_nd_key, **edge_data)
        # otherwise, add if not already added from another (parallel) edge
        elif not deduped_graph.has_edge(start_nd_key, end_nd_key):
            # there are normally max two edges, but sometimes three or more
            edges_data: list[EdgeData] = []
            for edge_data in nx_multigraph.get_edge_data(start_nd_key, end_nd_key).values():  # type: ignore
                edges_data.append(edge_data)
            edge_info = util.EdgeInfo()
            # find the shortest of the geoms
            edge_geoms = [edge["geom"] for edge in edges_data]
            edge_lens = [geom.length for geom in edge_geoms]
            shortest_idx = edge_lens.index(min(edge_lens))
            shortest_geom = edge_geoms.pop(shortest_idx)
            shortest_data = edges_data.pop(shortest_idx)
            # extract names and routes
            shortest_name_tags = _gather_name_tags(shortest_data)
            # start by gathering shortest's data
            edge_info.gather_edge_info(shortest_data)
            # process longer geoms
            longer_geoms: list[geometry.LineString] = []
            for edge_geom, edge_data in zip(edge_geoms, edges_data, strict=False):
                # where the buffer of the shorter contains the longer
                is_contained = shortest_geom.buffer(contains_buffer_dist).contains(edge_geom)
                # hwy tags
                hwy_tags_ok = True
                if osm_hwy_target_tags:
                    edge_hwy_tags = _tags_from_edge_key(edge_data, "highways")
                    if not hwy_tags.intersection(edge_hwy_tags):
                        hwy_tags_ok = False
                # names tags
                name_tags_ok = True
                if osm_matched_tags_only:
                    edge_name_tags = _gather_name_tags(edge_data)
                    if not shortest_name_tags.intersection(edge_name_tags):
                        name_tags_ok = False
                # collapse into shorter edge if conditions met
                if is_contained and hwy_tags_ok is True and name_tags_ok is True:
                    edge_info.gather_edge_info(edge_data)
                    longer_geoms.append(edge_geom)
                else:
                    edge_data_copy = {k: v for k, v in edge_data.items() if k != "geom"}
                    deduped_graph.add_edge(start_nd_key, end_nd_key, geom=edge_geom, **edge_data_copy)
            # otherwise, if not merging on a midline basis
            # or, if no other edges to process (in cases where longer geom has been retained per above)
            # then use the shortest geom
            if not merge_edges_by_midline or len(longer_geoms) == 0:
                edge_idx = deduped_graph.add_edge(start_nd_key, end_nd_key, geom=shortest_geom)
                edge_info.set_edge_info(deduped_graph, start_nd_key, end_nd_key, edge_idx)
            # otherwise weld the geoms, using the shortest as a yardstick
            else:
                # iterate the coordinates along the shorter geom
                # starting and endpoint geoms already match
                new_coords = []
                for coord in shortest_geom.coords:
                    # from the current short_geom coordinate
                    short_point = geometry.Point(coord)
                    # find the nearest points on the longer geoms
                    multi_coords = [short_point]
                    for longer_geom in longer_geoms:
                        # get the nearest point on the longer geom
                        # returns a tuple of nearest geom for respective input geoms
                        longer_point: geometry.Point = ops.nearest_points(short_point, longer_geom)[-1]
                        # only use for new coord centroid if not the starting or end point
                        lg_start_point = geometry.Point(longer_geom.coords[0])
                        lg_end_point = geometry.Point(longer_geom.coords[-1])
                        if lg_start_point.distance(longer_point) < 1 or lg_end_point.distance(longer_point) < 1:
                            continue
                        # aggregate
                        multi_coords.append(longer_point)
                    # create a midpoint between the geoms and add to the new coordinate array
                    mid_point: geometry.Point = geometry.MultiPoint(multi_coords).centroid
                    new_coords.append((mid_point.x, mid_point.y))
                # generate the new mid-line geom
                new_coords = util.snap_linestring_endpoints(deduped_graph, start_nd_key, end_nd_key, new_coords)
                new_geom = geometry.LineString(new_coords)
                # add to the graph
                edge_idx = deduped_graph.add_edge(start_nd_key, end_nd_key, geom=new_geom)
                edge_info.set_edge_info(deduped_graph, start_nd_key, end_nd_key, edge_idx)

    return deduped_graph


def nx_snap_endpoints(nx_multigraph: MultiGraph) -> MultiGraph:
    """
    Snaps geom endpoints to adjacent node coordinates.

    Parameters
    ----------
    nx_multigraph: MultiGraph
        A `networkX` `MultiGraph` in a projected coordinate system, containing `x` and `y` node attributes, and `geom`
        edge attributes containing `LineString` geoms.

    Returns
    -------
    MultiGraph
        A `networkX` `MultiGraph`.

    """
    logger.info("Snapping edge endpoints.")
    g_multi_copy: MultiGraph = nx_multigraph.copy()
    start_nd_key: NodeKey
    end_nd_key: NodeKey
    for start_nd_key, end_nd_key, edge_idx, edge_data in tqdm(
        g_multi_copy.edges(keys=True, data=True), disable=config.QUIET_MODE
    ):
        edge_geom: geometry.LineString = edge_data["geom"]
        edge_geom = geometry.LineString(
            util.snap_linestring_endpoints(g_multi_copy, start_nd_key, end_nd_key, edge_geom.coords)
        )
        g_multi_copy[start_nd_key][end_nd_key][edge_idx]["geom"] = edge_geom
    return g_multi_copy


def _simplify_line_by_max_angle(coords, max_angle):
    # continue simplifying until no angles exceed the threshold
    while True:
        angles_exceeding_threshold = False
        # start with the first point
        new_coords = [coords[0]]
        # iterate through the points
        for i in range(1, len(coords) - 1):
            p1, p2, p3 = coords[i - 1], coords[i], coords[i + 1]
            angle = util.measure_coords_angle(p1, p2, p3)
            # if angle exceeds the threshold
            if angle > max_angle:
                angles_exceeding_threshold = True
                # skip adding this point
                continue
            # otherwise, keep the point
            new_coords.append(p2)
        # always keep the last point
        new_coords.append(coords[-1])
        # exit loop if no angles exceeded the threshold
        if not angles_exceeding_threshold:
            break
        # update coords with modified list
        coords = new_coords

    # Return a new LineString with simplified coordinates
    return new_coords


def nx_iron_edges(
    nx_multigraph: MultiGraph,
    simplify_by_max_angle: int = 120,
    min_self_loop_length: int = 100,
    max_foot_tunnel_length: int = 50,
) -> MultiGraph:
    """
    Simplifies edges.

    Parameters
    ----------
    nx_multigraph: MultiGraph
        A `networkX` `MultiGraph` in a projected coordinate system, containing `x` and `y` node attributes, and `geom`
        edge attributes containing `LineString` geoms.
    simplify_by_max_angle: int
        The maximum angle to permit for a given edge. Angles greater than this will be reduced.
    min_self_loop_length: int
        Maximum self loop length to permit for a given edge.
    max_foot_tunnel_length: int
        Maximum tunnel length to permit for non motorised edges.

    Returns
    -------
    MultiGraph
        A `networkX` `MultiGraph` with simplified edges.

    """
    logger.info("Ironing edges.")
    g_multi_copy: MultiGraph = nx_multigraph.copy()
    start_nd_key: NodeKey
    end_nd_key: NodeKey
    remove_edges = []
    for start_nd_key, end_nd_key, edge_idx, edge_data in tqdm(
        g_multi_copy.edges(keys=True, data=True), disable=config.QUIET_MODE
    ):
        edge_geom: geometry.LineString = edge_data["geom"]
        # only apply to non looping geoms otherwise issues occur
        if start_nd_key == end_nd_key and edge_geom.length < min_self_loop_length:
            remove_edges.append((start_nd_key, end_nd_key, edge_idx))
            continue
        # drop long foot tunnels
        if (
            "is_tunnel" in edge_data
            and edge_data["is_tunnel"] is True
            and edge_data["geom"].length > max_foot_tunnel_length
        ):
            hwy_tags = _tags_from_edge_key(edge_data, "highways")
            if not hwy_tags.intersection(
                [
                    "motorway",
                    "motorway_link",
                    "trunk",
                    "trunk_link",
                    "primary",
                    "primary_link",
                    "secondary",
                    "secondary_link",
                    "tertiary",
                    "tertiary_link",
                    "residential",
                    "service",
                ]
            ):
                remove_edges.append((start_nd_key, end_nd_key, edge_idx))
                continue
        # simplify
        line_coords = _simplify_line_by_max_angle(edge_geom.coords, simplify_by_max_angle)
        g_multi_copy[start_nd_key][end_nd_key][edge_idx]["geom"] = geometry.LineString(line_coords)
    g_multi_copy.remove_edges_from(remove_edges)
    # straightening parallel edges can create duplicates
    g_multi_copy = nx_merge_parallel_edges(g_multi_copy, False, 1)

    return g_multi_copy


_MOTORISED_HWYS = set(
    [
        "motorway",
        "motorway_link",
        "trunk",
        "trunk_link",
        "primary",
        "primary_link",
        "secondary",
        "secondary_link",
        "tertiary",
        "tertiary_link",
        "residential",
        "living_street",
        # "service",  # intentional
    ]
)


def nx_deduplicate_edges(
    nx_multigraph: MultiGraph,
    dissolve_distance: int = 12,
    max_ang_diff: int = 20,
) -> MultiGraph:
    """
    Deduplicates non-motorised edges where parallel to nearby motorised edges.

    Parameters
    ----------
    nx_multigraph: MultiGraph
        A `networkX` `MultiGraph` in a projected coordinate system, containing `x` and `y` node attributes, and `geom`
        edge attributes containing `LineString` geoms.
    dissolve_distance: int
        A distance to use when searching for adjacent edges. 12m by default.
    max_ang_diff: int
         Only count a nearby adjacent edge as duplicitous if the angular difference between edges is less than
         `max_ang_diff`. 20 degrees by default.

    Returns
    -------
    MultiGraph
        A `networkX` graph. The nodes will have a new `weight` parameter indicating the node's contribution given the
        locally 'dissolved' context.

    """
    g_multi_copy: MultiGraph = nx_multigraph.copy()
    # generate STR tree
    edges_tree, edge_lookups = util.create_edges_strtree(g_multi_copy)
    # edges to remove
    edges_to_remove = set()
    # first iterate edges to save number of iters
    for start_nd_key, end_nd_key, edge_idx, edge_data in tqdm(
        g_multi_copy.edges(data=True, keys=True), disable=config.QUIET_MODE
    ):
        # find nearby edges
        edge_geom = edge_data["geom"]
        # edge_hierarchy = len(_ROAD_HIERARCHY)
        candidate = False
        for hwy_key in edge_data["highways"]:
            if hwy_key in _MOTORISED_HWYS:
                candidate = True
                break
        if candidate is False:
            continue
        edges_hits: list[int] = edges_tree.query(
            edge_geom,
            predicate="dwithin",
            distance=dissolve_distance,
        )  # type: ignore
        # buffer once outside of loop
        edge_geom_buff = edge_geom.buffer(dissolve_distance, cap_style=geometry.CAP_STYLE.flat)
        # review hits
        for edge_hit_idx in edges_hits:
            edge_lookup = edge_lookups[edge_hit_idx]
            nearby_start_nd_key = edge_lookup["start_nd_key"]
            nearby_end_nd_key = edge_lookup["end_nd_key"]
            nearby_edge_idx = edge_lookup["edge_idx"]
            # continue if already removed
            if (nearby_start_nd_key, nearby_end_nd_key, nearby_edge_idx) in edges_to_remove:
                continue
            # or current edge
            if nearby_start_nd_key == start_nd_key and nearby_end_nd_key == end_nd_key and edge_idx == nearby_edge_idx:
                continue
            # get edge data
            nearby_edge_data = g_multi_copy[nearby_start_nd_key][nearby_end_nd_key][nearby_edge_idx]
            # only remove if non motorised
            bail = False
            for nearby_hwy_key in nearby_edge_data["highways"]:
                if nearby_hwy_key in _MOTORISED_HWYS:
                    bail = True
                    break
            if bail is True:
                continue
            # fetch geom and check for intersection
            nearby_edge_geom: geometry.LineString = nearby_edge_data["geom"]
            # remove contained geoms but only if dead-ends
            if edge_geom_buff.contains(nearby_edge_geom) and (
                nx.degree(nx_multigraph, nearby_start_nd_key) == 1 or nx.degree(nx_multigraph, nearby_end_nd_key) == 1
            ):
                edges_to_remove.add((nearby_start_nd_key, nearby_end_nd_key, nearby_edge_idx))
            else:
                edge_itx = nearby_edge_geom.intersection(edge_geom_buff)
                if edge_itx and edge_itx.geom_type == "LineString" and edge_itx.length > 5:
                    # check for angle
                    ang_diff = util.measure_angle_diff_betw_linestrings(edge_geom.coords, edge_itx.coords)
                    if ang_diff < max_ang_diff:
                        # remove if duplicitous
                        edges_to_remove.add((nearby_start_nd_key, nearby_end_nd_key, nearby_edge_idx))
    # remove edges from graph
    g_multi_copy.remove_edges_from(edges_to_remove)
    # remove orphaned nodes
    g_multi_copy = nx_remove_filler_nodes(g_multi_copy)

    return g_multi_copy


def _squash_adjacent(
    nx_multigraph: MultiGraph,
    node_group: set[NodeKey],
    centroid_by_itx: bool,
    prioritise_by_hwy_tag: bool,
    simplify_by_max_angle: int | None = None,
) -> MultiGraph:
    """
    Squash nodes from a specified node group down to a new node.

    The new node can either be based on:
    - The centroid of all nodes;
    - else, all nodes of degree greater or equal to cent_min_degree;
    - and / else, all nodes with cumulative adjacent OSM street names or routes greater than cent_min_names;
    - and / else, all nodes with aggregate adjacent edge lengths greater than centroid_by_min_len_factor as a factor of
    the node with the greatest overall aggregate lengths. Edges are adjusted from the old nodes to the new combined
    node.
    """
    if not isinstance(nx_multigraph, nx.MultiGraph):
        raise TypeError("This method requires an undirected networkX MultiGraph (for multiple edges).")
    # nothing to do for single node group
    if len(node_group) < 2:
        return nx_multigraph
    # remove any node keys no longer in the graph
    centroid_nodes_filter = [nd_key for nd_key in node_group if nd_key in nx_multigraph]
    # find highest priority OSM highway tag
    if prioritise_by_hwy_tag:
        prioritise_tag = None
        for osm_hwy_tag in [
            "motorway",
            "motorway_link",
            "trunk",
            "trunk_link",
            "primary",
            "primary_link",
            "secondary",
            "secondary_link",
            "tertiary",
            "tertiary_link",
            "residential",
        ]:
            for nd_key in node_group:
                nb_hwy_tags = _gather_nb_tags(nx_multigraph, nd_key, "highways")
                if osm_hwy_tag in nb_hwy_tags:
                    prioritise_tag = osm_hwy_tag
                    break
            if prioritise_tag is not None:
                break
        # # filter by hwy tags if provided
        if prioritise_tag is not None:
            # extract nodes intersecting prioritised tag
            hwy_tags_filtered = []
            for nd_key in node_group:
                nb_hwy_tags = _gather_nb_tags(nx_multigraph, nd_key, "highways")
                if prioritise_tag in nb_hwy_tags:
                    # count edges which explicitly have tag
                    nb_tag_count = 0
                    for nb_nd_key in nx.all_neighbors(nx_multigraph, nd_key):
                        # can be multiple edges
                        for edge_data in nx_multigraph[nd_key][nb_nd_key].values():
                            edges_tags = _tags_from_edge_key(edge_data, "highways")
                            if prioritise_tag in edges_tags:
                                nb_tag_count += 1
                    hwy_tags_filtered.append((nd_key, nb_tag_count))
            # if also prioritising by itx
            if hwy_tags_filtered and centroid_by_itx:
                max_nb_count = max([n[1] for n in hwy_tags_filtered])
                centroid_nodes_filter = [n[0] for n in hwy_tags_filtered if n[1] == max_nb_count]
            elif hwy_tags_filtered:
                centroid_nodes_filter = [n[0] for n in hwy_tags_filtered]
    # if using intersections, find straight-through routes and count
    elif centroid_by_itx:
        crossings_2 = []
        crossings_1 = []
        for nd_key in centroid_nodes_filter:
            crossings = 0
            # compute node straight-through-angles
            nd_x_y = (nx_multigraph.nodes[nd_key]["x"], nx_multigraph.nodes[nd_key]["y"])
            for nb_nd_key_a in nx.neighbors(nx_multigraph, nd_key):
                for nb_edge_data_a in nx_multigraph[nd_key][nb_nd_key_a].values():
                    geom_a = nb_edge_data_a["geom"]
                    geom_a_coords = util.align_linestring_coords(geom_a.coords, nd_x_y, reverse=True)
                    for nb_nd_key_b in nx.neighbors(nx_multigraph, nd_key):
                        if nb_nd_key_b == nb_nd_key_a:
                            continue
                        for nb_edge_data_b in nx_multigraph[nd_key][nb_nd_key_b].values():
                            geom_b = nb_edge_data_b["geom"]
                            geom_b_coords = util.align_linestring_coords(geom_b.coords, nd_x_y, reverse=False)
                            angle_sum = util.measure_coords_angle(
                                geom_a_coords[0][:2],  # type: ignore
                                nd_x_y,  # type: ignore
                                geom_b_coords[-1][:2],  # type: ignore
                            )
                            if angle_sum < 10:
                                crossings += 1
            if crossings / 2 >= 2:
                crossings_2.append(nd_key)
            elif crossings / 2 >= 1:
                crossings_1.append(nd_key)
        # favour nodes with two through routes
        if crossings_2:
            centroid_nodes_filter = crossings_2
        elif crossings_1:
            centroid_nodes_filter = crossings_1
    # prepare the names and geoms for filtered points to be used for the new centroid
    node_geoms = []
    coords_set: set[str] = set()
    for nd_key in centroid_nodes_filter:
        x: float = nx_multigraph.nodes[nd_key]["x"]
        y: float = nx_multigraph.nodes[nd_key]["y"]
        # in rare cases opposing geom splitting can cause overlaying nodes
        # these can swing the gravity of multipoint centroids so screen these out
        xy_key: str = f"{round(x)}-{round(y)}"
        if xy_key in coords_set:
            continue
        coords_set.add(xy_key)
        node_geoms.append(geometry.Point(x, y))
    # set the new centroid from the centroid of the node group's Multipoint:
    new_cent: geometry.Point = geometry.MultiPoint(node_geoms).centroid
    # now that the centroid is known, add the new node
    new_nd_name, is_dupe = util.add_node(nx_multigraph, node_group, x=new_cent.x, y=new_cent.y)  # type: ignore
    if is_dupe:
        # an edge case: if the potential duplicate was one of the node group then it doesn't need adding
        if new_nd_name in node_group:
            # but remove from the node group since it doesn't need to be removed and replumbed
            node_group.remove(new_nd_name)
        else:
            raise ValueError(f"Attempted to add a duplicate node for node_group {node_group}.")
    # iterate the nodes to be removed and connect their existing edge geometries to the new centroid
    for nd_key in node_group:
        nd_data: NodeData = nx_multigraph.nodes[nd_key]
        nd_xy = (nd_data["x"], nd_data["y"])
        # iterate the node's existing neighbours
        for nb_nd_key in nx.neighbors(nx_multigraph, nd_key):
            # no need to rewire the edge if the neighbour is the same as the new node
            # this would otherwise result in a zero length edge
            # the edge will be dropped once the nd_key is removed
            if nb_nd_key == new_nd_name:
                continue
            # if a neighbour is also going to be dropped, then no need to create new between edges
            # an exception exists when a geom is looped, in which case the neighbour is also the current node
            if nb_nd_key in node_group and nb_nd_key != nd_key:
                continue
            # MultiGraph - so iter edges
            edge_data: EdgeData
            for edge_data in nx_multigraph[nd_key][nb_nd_key].values():
                if "geom" not in edge_data:
                    raise KeyError(f'Missing "geom" attribute for edge {nd_key}-{nb_nd_key}')
                line_geom: geometry.LineString = edge_data["geom"]
                if line_geom.geom_type != "LineString":
                    raise TypeError(
                        f"Expecting LineString geometry but found {line_geom.geom_type} geometry "
                        f"for edge {nd_key}-{nb_nd_key}."
                    )
                # orient the LineString so that the geom starts from the node's x_y
                line_coords = util.align_linestring_coords(line_geom.coords, nd_xy)
                # update geom starting point to new parent node's coordinates
                line_coords = util.snap_linestring_startpoint(line_coords, (new_cent.x, new_cent.y))
                # if self-loop, then the end also needs updating to the new centroid
                if nd_key == nb_nd_key:
                    line_coords = util.snap_linestring_endpoint(line_coords, (new_cent.x, new_cent.y))
                    target_nd_key = new_nd_name
                else:
                    target_nd_key = nb_nd_key
                # simplify to handle new kinks
                if simplify_by_max_angle is not None:
                    line_coords = _simplify_line_by_max_angle(line_coords, simplify_by_max_angle)
                # build the new geom
                new_edge_geom = geometry.LineString(line_coords)
                if new_edge_geom.length == 0:
                    continue
                # bail if short self loop
                if new_nd_name == target_nd_key and new_edge_geom.length < 100:
                    continue
                # check that a duplicate is not being added
                dupe = False
                if nx_multigraph.has_edge(new_nd_name, target_nd_key):
                    # only add parallel edges if substantially different from any existing edges
                    n_edges: int = nx_multigraph.number_of_edges(new_nd_name, target_nd_key)
                    for edge_idx in range(n_edges):
                        exist_geom: geometry.LineString = nx_multigraph[new_nd_name][target_nd_key][edge_idx]["geom"]
                        # don't add if the edges have the same number of coords and the coords are similar
                        # 5m x and y tolerance across all coordinates
                        if len(new_edge_geom.coords) == len(exist_geom.coords) and np.allclose(
                            new_edge_geom.coords,
                            exist_geom.coords,
                            atol=5,
                            rtol=0,
                        ):
                            dupe = True
                            logger.debug(
                                f"Not adding edge {new_nd_name} to {target_nd_key}: a similar edge already exists. "
                                f"Length: {new_edge_geom.length} vs. {exist_geom.length}. "
                                f"Num coords: {len(new_edge_geom.coords)} vs. {len(exist_geom.coords)}."
                            )
                if not dupe:
                    # add the new edge
                    edge_idx = nx_multigraph.add_edge(
                        new_nd_name,
                        target_nd_key,
                        geom=new_edge_geom,
                    )
                    edge_info = util.EdgeInfo()
                    edge_info.gather_edge_info(edge_data)
                    edge_info.set_edge_info(nx_multigraph, new_nd_name, target_nd_key, edge_idx)
        # drop the node, this will also implicitly drop the old edges
        nx_multigraph.remove_node(nd_key)

    return nx_multigraph


def nx_consolidate_nodes(
    nx_multigraph: MultiGraph,
    buffer_dist: float = 12,
    neighbour_policy: str | None = None,
    crawl: bool = False,
    centroid_by_itx: bool = True,
    prioritise_by_hwy_tag: bool = False,
    merge_edges_by_midline: bool = True,
    contains_buffer_dist: int = 25,
    osm_hwy_target_tags: list[str] | None = None,
    osm_matched_tags_only: bool = False,
    simplify_by_max_angle: int | None = None,
) -> MultiGraph:
    """
    Consolidates nodes if they are within a buffer distance of each other.

    Several parameters provide more control over the conditions used for deciding whether or not to merge nodes. The
    algorithm proceeds in two steps:

    Nodes within the buffer distance of each other are merged. If selecting `centroid_by_itx` then the new centroid
    will try to use intersections to determine the new centroid for the nodes. It will first attempt to find
    intersections with two through-routes, else will use intersections with one through-route.

    The merging of nodes can create parallel edges with mutually shared nodes on either side. These edges are replaced
    by a single new edge, with the new geometry selected from either:
    - An imaginary centreline of the combined edges if `merge_edges_by_midline` is set to `True`;
    - Else, the shortest edge, with longer edges discarded;
    See [`nx_merge_parallel_edges`](#nx-merge-parallel-edges) for more information.

    Parameters
    ----------
    nx_multigraph: MultiGraph
        A `networkX` `MultiGraph` in a projected coordinate system, containing `x` and `y` node attributes, and `geom`
        edge attributes containing `LineString` geoms.
    buffer_dist: float
        The buffer distance to be used for consolidating nearby nodes. Defaults to 5.
    neighbour_policy: str
        Whether all nodes within the buffer distance are merged, or only "direct" or "indirect" neighbours. Defaults to
        None which will consider all nodes.
    crawl: bool
        Whether the algorithm will recursively explore neighbours of neighbours if those neighbours are within the
        buffer distance from the prior node. Defaults to False.
    centroid_by_itx: bool
        Whether to favour intersections when selecting the combined centroid of merged nodes. Intersections with two
        straight through-routes will be favoured if found, otherwise intersections with one straight through-route are
        used where available. True by default.
    prioritise_by_hwy_tag: bool
        Whether to prioritise centroid locations by OSM highway tags. For example, trunk roads will have higher priority
        than residential roads. Requires graph prepared with via [`io.osm_graph_from_poly`](/io#osm-graph-from-poly).
        Defaults to False.
    merge_edges_by_midline: bool
        Whether to merge parallel edges by an imaginary centreline. If set to False, then the shortest edge will be
        retained as the new geometry and the longer edges will be discarded. Defaults to True.
    contains_buffer_dist: int
        The buffer distance to consider when checking if parallel edges sharing the same start and end nodes are
        sufficiently adjacent to be merged. This is run after node consolidation has completed.
    osm_hwy_target_tags: list[str]
        An optional list of OpenStreetMap target highway tags. If provided, only nodes with neighbouring edges
        containing a tag matching one of the target OSM highway tags will be consolidated. Requires graph prepared with
        via [`io.osm_graph_from_poly`](/io#osm-graph-from-poly).
    osm_matched_tags_only: bool
        Whether to only merge edges with shared OSM `name` or `ref` tags. False by default. Requires graph prepared with
        via [`io.osm_graph_from_poly`](/io#osm-graph-from-poly).
    simplify_by_max_angle: int
        The optional maximum angle to permit for a given edge. Angles greater than this will be reduced.

    Returns
    -------
    nx.MultiGraph
        A `networkX` `MultiGraph` with consolidated nodes.

    Examples
    --------
    See the guide on [graph cleaning](/guide#graph-cleaning) for more information.

    ![Example raw graph from OSM](/images/graph_raw.png)
    _The pre-consolidation OSM street network for Soho, London. © OpenStreetMap contributors._

    ![Example cleaned graph](/images/graph_clean.png)
    _The consolidated OSM street network for Soho, London. © OpenStreetMap contributors._

    """
    if not isinstance(nx_multigraph, nx.MultiGraph):
        raise TypeError("This method requires an undirected networkX MultiGraph.")
    if neighbour_policy is not None and neighbour_policy not in ("direct", "indirect"):
        raise ValueError('Neighbour policy should be "direct", "indirect", or the default of "None"')
    if crawl and buffer_dist >= 20:
        logger.warning("Be cautious with large buffer distances when using crawl!")
    _multi_graph: MultiGraph = nx_multigraph.copy()
    # create a nodes STRtree
    nodes_tree, node_lookups = util.create_nodes_strtree(_multi_graph)
    # iter
    logger.info("Consolidating nodes.")
    # keep track of removed nodes
    removed_nodes: set[NodeKey] = set()

    def recursive_squash(
        _nd_key: NodeKey,
        x: float,
        y: float,
        node_group: set[NodeKey],
        processed_nodes: set[NodeKey],
        _hwy_tags: set,
        _name_tags: set,
        _levels_tags: set,
        recursive: bool = False,
    ) -> set[NodeKey]:
        # keep track of which nodes have been processed as part of recursion
        processed_nodes.add(_nd_key)
        # get all other nodes within buffer distance - the self-node and previously processed nodes are also returned
        j_hits: list[int] = nodes_tree.query(geometry.Point(x, y).buffer(buffer_dist))  # type: ignore
        # review each node within the buffer
        j_nd_key: NodeKey
        for j_hit_idx in j_hits:
            j_lookup: dict[str, Any] = node_lookups[j_hit_idx]
            j_nd_key = j_lookup["nd_key"]
            if j_nd_key in removed_nodes or j_nd_key in processed_nodes:
                continue
            # check neighbour policy
            if neighbour_policy is not None:
                # use the original graph prior to in-place modifications
                neighbours: list[NodeKey] = nx.neighbors(nx_multigraph, _nd_key)
                if neighbour_policy == "indirect" and j_nd_key in neighbours:
                    continue
                if neighbour_policy == "direct" and j_nd_key not in neighbours:
                    continue
            # levels
            if _levels_tags:
                _nb_level_tags = _gather_nb_tags(nx_multigraph, j_nd_key, "levels")
                if not _levels_tags.intersection(_nb_level_tags):
                    continue
            # hwy tags
            if osm_hwy_target_tags:
                _nb_hwy_tags = _gather_nb_tags(nx_multigraph, j_nd_key, "highways")
                if not _hwy_tags.intersection(_nb_hwy_tags):
                    continue
            # names tags
            if osm_matched_tags_only is True:
                _nb_name_tags = _gather_nb_name_tags(nx_multigraph, j_nd_key)
                if not _name_tags.intersection(_nb_name_tags):
                    continue
            # otherwise add the node
            node_group.add(j_nd_key)
            # if recursive, follow the chain
            if recursive:
                j_nd_data: NodeData = nx_multigraph.nodes[j_nd_key]
                return recursive_squash(
                    j_nd_key,
                    j_nd_data["x"],
                    j_nd_data["y"],
                    node_group,
                    processed_nodes,
                    _hwy_tags,
                    _name_tags,
                    _levels_tags,
                    recursive=crawl,
                )
        return node_group

    # if using OSM tags heuristic
    hwy_tags = _extract_tags_to_set(osm_hwy_target_tags)
    # iterate origin graph (else node structure changes in place)
    nd_key: NodeKey
    nd_data: NodeData
    for nd_key, nd_data in tqdm(nx_multigraph.nodes(data=True), disable=config.QUIET_MODE):
        # skip if already consolidated from an adjacent node, or if the node's degree doesn't meet min_node_degree
        if nd_key in removed_nodes:
            continue
        # get this nodes neighbouring edge hwy tags
        nb_hwy_tags = _gather_nb_tags(nx_multigraph, nd_key, "highways")
        if osm_hwy_target_tags and not hwy_tags.intersection(nb_hwy_tags):
            continue
        # get levels info for matching against potential nodes
        nb_levels_tags = _gather_nb_tags(nx_multigraph, nd_key, "levels")
        # get name tags for matching against potential matches
        nb_name_tags = _gather_nb_name_tags(nx_multigraph, nd_key)
        # recurse
        node_group = recursive_squash(
            nd_key,  # node nd_key
            nd_data["x"],  # x point for buffer
            nd_data["y"],  # y point for buffer
            set([nd_key]),  # node group for consolidation (with starting node)
            set(),  # processed nodes tracked through recursion
            hwy_tags,
            nb_name_tags,
            nb_levels_tags,
            crawl,
        )  # whether to recursively probe neighbours per distance
        # update removed nodes
        removed_nodes.update(node_group)
        # consolidate
        if len(node_group) > 1:
            _multi_graph = _squash_adjacent(
                _multi_graph,
                node_group,
                centroid_by_itx=centroid_by_itx,
                prioritise_by_hwy_tag=prioritise_by_hwy_tag,
                simplify_by_max_angle=simplify_by_max_angle,
            )
    # remove parallel edges resulting from squashing nodes
    _multi_graph = nx_merge_parallel_edges(
        _multi_graph,
        merge_edges_by_midline,
        contains_buffer_dist,
    )

    return _multi_graph


def nx_snap_gapped_endings(
    nx_multigraph: nx.MultiGraph,
    buffer_dist: float = 12,
    osm_hwy_target_tags: list[str] | None = None,
    osm_matched_tags_only: bool = False,
) -> nx.MultiGraph:
    if not isinstance(nx_multigraph, nx.MultiGraph):
        raise TypeError("This method requires an undirected networkX MultiGraph.")
    _multi_graph: nx.MultiGraph
    _multi_graph = nx_multigraph.copy()  # type: ignore
    # if using OSM tags heuristic
    hwy_tags = _extract_tags_to_set(osm_hwy_target_tags)
    # create an edges STRtree (nodes and edges)
    nodes_tree, node_lookups = util.create_nodes_strtree(_multi_graph)
    # create an edges STRtree (nodes and edges)
    edges_tree, edge_lookups = util.create_edges_strtree(_multi_graph)
    # iter
    logger.info("Snapping gapped endings.")
    # iterate origin graph
    for nd_key, nd_data in tqdm(nx_multigraph.nodes(data=True), disable=config.QUIET_MODE):
        # don't split opposing geoms from nodes of degree 1
        nd_degree = nx.degree(nx_multigraph, nd_key)
        if nd_degree != 1:
            continue
        # check tags
        if osm_hwy_target_tags:
            nb_hwy_tags = _gather_nb_tags(nx_multigraph, nd_key, "highways")
            if not hwy_tags.intersection(nb_hwy_tags):
                continue
        nb_levels_tags = _gather_nb_tags(nx_multigraph, nd_key, "levels")
        # get name tags for matching against potential gapped edges
        nb_name_tags = _gather_nb_name_tags(nx_multigraph, nd_key)
        # get all other nodes within the buffer distance
        # the spatial index using bounding boxes, so further filtering is required (see further down)
        n_point = geometry.Point(nd_data["x"], nd_data["y"])
        # spatial query from point returns all buffers with buffer_dist
        node_hits: list[dict] = nodes_tree.query(n_point.buffer(buffer_dist))  # type: ignore
        # extract the start node, end node, geom
        node_keys: list = []
        for node_hit_idx in node_hits:
            j_nd_key = node_lookups[node_hit_idx]["nd_key"]  # type: ignore
            if j_nd_key == nd_key:
                continue
            j_nd_degree = node_lookups[node_hit_idx]["nd_degree"]  # type: ignore
            if j_nd_degree == 1:
                node_keys.append(j_nd_key)
        # abort if no gapped nodes
        if not node_keys:
            continue
        # prepare the root node's point geom
        n_geom = geometry.Point(nd_data["x"], nd_data["y"])
        # iter gapped edges
        for j_nd_key in node_keys:
            # check distance
            j_nd_data = nx_multigraph.nodes[j_nd_key]
            j_geom = geometry.Point(j_nd_data["x"], j_nd_data["y"])
            if n_geom.distance(j_geom) > buffer_dist:
                continue
            # hwy tags
            if osm_hwy_target_tags:
                edge_hwy_tags = _gather_nb_tags(nx_multigraph, j_nd_key, "highways")
                if not hwy_tags.intersection(edge_hwy_tags):
                    continue
            # levels
            if nb_levels_tags:
                edge_level_tags = _gather_nb_tags(nx_multigraph, j_nd_key, "levels")
                if not nb_levels_tags.intersection(edge_level_tags):
                    continue
            # names tags
            if osm_matched_tags_only is True:
                edge_name_tags = _gather_nb_name_tags(nx_multigraph, j_nd_key)
                if not nb_name_tags.intersection(edge_name_tags):
                    continue
            # create new geom
            new_geom = geometry.LineString(
                [
                    [nd_data["x"], nd_data["y"]],
                    [j_nd_data["x"], j_nd_data["y"]],
                ]
            )
            # don't add new edges that would criss cross existing
            bail = False
            edge_hits = edges_tree.query(new_geom)
            for edge_hit_idx in edge_hits:
                edge_lookup = edge_lookups[edge_hit_idx]
                start_nd_key = edge_lookup["start_nd_key"]
                end_nd_key = edge_lookup["end_nd_key"]
                edge_idx = edge_lookup["edge_idx"]
                edge_geom: dict = nx_multigraph[start_nd_key][end_nd_key][edge_idx]["geom"]
                if edge_geom.crosses(new_geom):  # type: ignore
                    bail = True
                    break
            if bail:
                continue
            # add new edge
            if not _multi_graph.has_edge(nd_key, j_nd_key):
                _multi_graph.add_edge(
                    nd_key,
                    j_nd_key,
                    names=[],
                    routes=[],
                    highways=[],
                    levels=[],
                    geom=new_geom,
                )

    return _multi_graph


def nx_split_opposing_geoms(
    nx_multigraph: nx.MultiGraph,
    buffer_dist: float = 12,
    merge_edges_by_midline: bool = True,
    contains_buffer_dist: int = 25,
    prioritise_by_hwy_tag: bool = False,
    osm_hwy_target_tags: list[str] | None = None,
    osm_matched_tags_only: bool = False,
    min_node_degree: int = 2,
    max_node_degree: int | None = None,
    squash_nodes: bool = True,
    centroid_by_itx: bool = False,
    simplify_by_max_angle: int | None = None,
) -> nx.MultiGraph:
    """
    Split edges in near proximity to nodes, then weld the resultant node group together, updating edges in the process.

    This is primarily intended for merging parallel roadways when used with the default `min_node_degree=2`. When an
    edge geometry is within the specified `buffer_dist` of a node (with the specified `min_node_degree`) then the edge
    geom is split and a new node is inserted. The new node is then merged with the node which triggered the split, with
    edge geometries updated accordingly.

    Dead-end segments can be projected to nearby edge geometries if lowering `min_node_degree` to `1`. This is useful
    for connecting disjointed OSM pedestrian segments to nearby roadway geometries. Consider using with
    `max_node_degree=1` and `osm_hwy_target_tags` set to `['footway']` to restrict the behaviour to dead-end pedestrian
    routes.

    The merging of nodes can create parallel edges with mutually shared nodes on either side. These edges are replaced
    by a single new edge, with the new geometry selected from either:
        - An imaginary centreline of the combined edges if `merge_edges_by_midline` is set to `True`;
        - Else, the shortest edge, with longer edges discarded.
    See [`nx_merge_parallel_edges`](#nx-merge-parallel-edges) for more information.

    Parameters
    ----------
    nx_multigraph: MultiGraph
        A `networkX` `MultiGraph` in a projected coordinate system, containing `x` and `y` node attributes, and `geom`
        edge attributes containing `LineString` geoms.
    buffer_dist: int
        The buffer distance to be used for splitting nearby nodes. Defaults to 5.
    merge_edges_by_midline: bool
        Whether to merge parallel edges by an imaginary centreline. If set to False, then the shortest edge will be
        retained as the new geometry and the longer edges will be discarded. Defaults to True.
    contains_buffer_dist: int
        The buffer distance to consider when checking if parallel edges sharing the same start and end nodes are
        sufficiently adjacent to be merged.
    prioritise_by_hwy_tag: bool
        Whether to prioritise centroid locations by OSM highway tags. For example, trunk roads will have higher priority
        than residential roads. Requires graph prepared with via [`io.osm_graph_from_poly`](/io#osm-graph-from-poly).
        Defaults to False.
    osm_hwy_target_tags: list[str]
        An optional list of OpenStreetMap target highway tags. If provided, only nodes with neighbouring edges
        containing a tag matching one of the target OSM highway tags will be consolidated. Requires graph prepared with
        via [`io.osm_graph_from_poly`](/io#osm-graph-from-poly).
    osm_matched_tags_only: bool
        Whether to only merge edges with shared OSM `name` or `ref` tags. False by default. Requires graph prepared with
        via [`io.osm_graph_from_poly`](/io#osm-graph-from-poly).
    min_node_degree: int
        Only project nodes with at least node degree of `min_node_degree`.
    max_node_degree: int
        Only project nodes with at most node degree of `max_node_degree`.
    squash_nodes: bool
        Whether to automatically squash new node pairings resulting from splitting a nearby edge. If set to `False` then
        a line will be added instead. Defaults to `True`.
    simplify_by_max_angle: int
        The optional maximum angle to permit for a given edge. Angles greater than this will be reduced.

    Returns
    -------
    MultiGraph
        A `networkX` `MultiGraph` with consolidated nodes.

    """

    def make_edge_key(start_nd_key: NodeKey, end_nd_key: NodeKey, edge_idx: int) -> str:
        return "-".join(sorted([str(start_nd_key), str(end_nd_key)])) + f"-k{edge_idx}"

    # where edges are deleted, keep track of new children edges
    edge_children: dict[str, list] = {}

    # recursive function for retrieving nested layers of successively replaced edges
    def recurse_child_keys(
        _start_nd_key: NodeKey,
        _end_nd_key: NodeKey,
        _edge_idx: int,
        _edge_data: dict,
        current_edges: list,
    ):
        """
        Recursively checks if an edge has been replaced by children, if so, use children instead.
        """
        edge_key = make_edge_key(_start_nd_key, _end_nd_key, _edge_idx)
        # if an edge does not have children, add to current_edges and return
        if edge_key not in edge_children:
            current_edges.append((_start_nd_key, _end_nd_key, _edge_idx, _edge_data))
        # otherwise recursively drill-down until newest edges are found
        else:
            for child_s, child_e, child_k, child_data in edge_children[edge_key]:
                recurse_child_keys(child_s, child_e, child_k, child_data, current_edges)

    if not isinstance(nx_multigraph, nx.MultiGraph):
        raise TypeError("This method requires an undirected networkX MultiGraph.")
    _multi_graph = nx_multigraph.copy()
    # if using OSM tags heuristic
    hwy_tags = _extract_tags_to_set(osm_hwy_target_tags)
    # create an edges STRtree (nodes and edges)
    edges_tree, edge_lookups = util.create_edges_strtree(_multi_graph)
    # node groups
    node_groups: list[set] = []
    # iter
    logger.info("Splitting opposing edges.")
    # iterate origin graph (else node structure changes in place)
    nd_key: NodeKey
    for nd_key, nd_data in tqdm(nx_multigraph.nodes(data=True), disable=config.QUIET_MODE):
        nd_degree = nx.degree(_multi_graph, nd_key)
        if nd_degree < min_node_degree:
            continue
        if max_node_degree is not None and nd_degree > max_node_degree:
            continue
        # check tags
        if osm_hwy_target_tags:
            nb_hwy_tags = _gather_nb_tags(nx_multigraph, nd_key, "highways")
            if not hwy_tags.intersection(nb_hwy_tags):
                continue
        # get name tags for matching against potential gapped edges
        nb_name_tags = _gather_nb_name_tags(nx_multigraph, nd_key)
        # get levels info for matching against potential gapped edges
        nb_levels_tags = _gather_nb_tags(nx_multigraph, nd_key, "levels")
        # only split from ground level nodes
        if nb_levels_tags and 0 not in nb_levels_tags:
            continue
        # neighbours for filtering out
        neighbours = list(nx.neighbors(nx_multigraph, nd_key))
        # get all other edges within the buffer distance
        # the spatial index using bounding boxes, so further filtering is required (see further down)
        # furthermore, successive iterations may remove old edges
        # so keep track of removed parent vs new child edges
        n_point = geometry.Point(nd_data["x"], nd_data["y"])
        # spatial query from point returns all buffers with buffer_dist
        edge_hits: list[int] = edges_tree.query(n_point.buffer(buffer_dist))  # type: ignore
        # extract the start node, end node, geom
        edges: list = []
        for edge_hit_idx in edge_hits:
            edge_lookup = edge_lookups[edge_hit_idx]
            start_nd_key = edge_lookup["start_nd_key"]
            end_nd_key = edge_lookup["end_nd_key"]
            edge_idx = edge_lookup["edge_idx"]
            edge_data: dict = nx_multigraph[start_nd_key][end_nd_key][edge_idx]
            # don't add attached edge
            if nd_key in (start_nd_key, end_nd_key):
                continue
            # don't add neighbouring edges
            if start_nd_key in neighbours or end_nd_key in neighbours:
                continue
            edges.append((start_nd_key, end_nd_key, edge_idx, edge_data))
        # review gapped edges
        # if already removed, get the new child edges
        current_edges: list = []
        for start_nd_key, end_nd_key, edge_idx, edge_data in edges:
            recurse_child_keys(start_nd_key, end_nd_key, edge_idx, edge_data, current_edges)
        # check that edges are within buffer
        gapped_edges: list = []
        for start_nd_key, end_nd_key, edge_idx, edge_data in current_edges:
            edge_geom = edge_data["geom"]
            # check whether the geom is truly within the buffer distance
            if edge_geom.distance(n_point) > buffer_dist:  # type: ignore
                continue
            gapped_edges.append((start_nd_key, end_nd_key, edge_idx, edge_data))
        # abort if no gapped edges
        if not gapped_edges:
            continue
        # prepare the root node's point geom
        n_geom = geometry.Point(nd_data["x"], nd_data["y"])
        # nodes for squashing
        node_group = [nd_key]
        # sort gapped edges by distance
        gapped_edges = sorted(gapped_edges, key=lambda edge: n_point.distance(edge[3]["geom"]))
        # unique edges not sharing a node - i.e. only pierce nearest rather than in multiple directions
        shared_nodes = set()
        distinct_edges = []
        for start_nd_key, end_nd_key, edge_idx, edge_data in gapped_edges:
            if start_nd_key in shared_nodes or end_nd_key in shared_nodes:
                continue
            shared_nodes.add(start_nd_key)
            shared_nodes.add(end_nd_key)
            distinct_edges.append((start_nd_key, end_nd_key, edge_idx, edge_data))
        # iter gapped edges
        for start_nd_key, end_nd_key, edge_idx, edge_data in distinct_edges:
            edge_geom = edge_data["geom"]
            # don't split on tunnels
            if "is_tunnel" in edge_data and edge_data["is_tunnel"] is True:
                continue
            # level tags
            if nb_levels_tags:
                # only split on ground levels
                edge_levels_tags = _tags_from_edge_key(edge_data, "levels")
                if edge_levels_tags and 0 not in edge_levels_tags:
                    continue
            # hwy tags
            if osm_hwy_target_tags:
                edge_hwy_tags = _tags_from_edge_key(edge_data, "highways")
                if not hwy_tags.intersection(edge_hwy_tags):
                    continue
            # names tags
            if osm_matched_tags_only is True:
                edge_name_tags = _gather_name_tags(edge_data)
                if not nb_name_tags.intersection(edge_name_tags):
                    continue
            # project a point and split the opposing geom
            # ops.nearest_points returns tuple of nearest from respective input geoms
            # want the nearest point on the line at index 1
            nearest_point: geometry.Point = ops.nearest_points(n_geom, edge_geom)[-1]  # type: ignore
            # if a valid nearest point has been found, go ahead and split the geom
            # use a snap because rounding precision errors will otherwise cause issues
            split_geoms: geometry.GeometryCollection = ops.split(
                ops.snap(edge_geom, nearest_point, 0.01),  # type: ignore
                nearest_point,  # type: ignore
            )
            # in some cases the line will be pointing away, but is still near enough to be within max
            # in these cases a single geom will be returned
            if len(split_geoms.geoms) < 2:
                continue
            new_edge_geom_a: geometry.LineString
            new_edge_geom_b: geometry.LineString
            new_edge_geom_a, new_edge_geom_b = split_geoms.geoms  # type: ignore
            # add the new node and edges to _multi_graph (don't modify nx_multigraph because of iter in place)
            new_nd_name, is_dupe = util.add_node(
                _multi_graph,
                [start_nd_key, nd_key, end_nd_key],
                x=nearest_point.x,
                y=nearest_point.y,
            )
            # continue if a node already exists at this location
            if is_dupe:
                continue
            node_group.append(new_nd_name)
            # copy edge data
            edge_data_copy = {k: v for k, v in edge_data.items() if k != "geom"}
            _multi_graph.add_edge(start_nd_key, new_nd_name, **edge_data_copy)
            _multi_graph.add_edge(end_nd_key, new_nd_name, **edge_data_copy)
            # get starting geom for orientation
            s_nd_data = _multi_graph.nodes[start_nd_key]
            s_nd_geom = geometry.Point(s_nd_data["x"], s_nd_data["y"])
            if np.allclose(
                s_nd_geom.coords,
                new_edge_geom_a.coords[0][:2],
                atol=config.ATOL,
                rtol=0,
            ) or np.allclose(
                s_nd_geom.coords,
                new_edge_geom_a.coords[-1][:2],
                atol=config.ATOL,
                rtol=0,
            ):
                s_new_geom = new_edge_geom_a
                e_new_geom = new_edge_geom_b
            else:
                # double check matching geoms
                if not np.allclose(
                    s_nd_geom.coords,
                    new_edge_geom_b.coords[0][:2],
                    atol=config.ATOL,
                    rtol=0,
                ) and not np.allclose(
                    s_nd_geom.coords,
                    new_edge_geom_b.coords[-1][:2],
                    atol=config.ATOL,
                    rtol=0,
                ):
                    raise ValueError("Unable to match split geoms to existing nodes")
                s_new_geom = new_edge_geom_b
                e_new_geom = new_edge_geom_a
            # if splitting a looped component, then both new edges will have the same starting and ending nodes
            # in these cases, there will be multiple edges
            if start_nd_key == end_nd_key:
                if _multi_graph.number_of_edges(start_nd_key, new_nd_name) != 2:
                    raise ValueError(f"Number of edges between {start_nd_key} and {new_nd_name} does not equal 2")
                s_k = 0
                e_k = 1
            else:
                if _multi_graph.number_of_edges(start_nd_key, new_nd_name) != 1:
                    raise ValueError(f"Number of edges between {start_nd_key} and {new_nd_name} does not equal 1.")
                if _multi_graph.number_of_edges(end_nd_key, new_nd_name) != 1:
                    raise ValueError(f"Number of edges between {end_nd_key} and {new_nd_name} does not equal 1.")
                s_k = e_k = 0
            # write the new edges
            _multi_graph[start_nd_key][new_nd_name][s_k]["geom"] = s_new_geom
            _multi_graph[end_nd_key][new_nd_name][e_k]["geom"] = e_new_geom
            # add the new edges to the edge_children dictionary
            edge_key = make_edge_key(start_nd_key, end_nd_key, edge_idx)
            edge_children[edge_key] = [
                (
                    start_nd_key,
                    new_nd_name,
                    s_k,
                    _multi_graph[start_nd_key][new_nd_name][s_k],
                ),
                (
                    end_nd_key,
                    new_nd_name,
                    e_k,
                    _multi_graph[end_nd_key][new_nd_name][e_k],
                ),
            ]
            # drop the old edge from _multi_graph
            if _multi_graph.has_edge(start_nd_key, end_nd_key, edge_idx):  # type: ignore
                _multi_graph.remove_edge(start_nd_key, end_nd_key, edge_idx)  # type: ignore
        node_groups.append(list(node_group))  # type: ignore
    # iter and squash
    if squash_nodes is True:
        logger.info("Squashing opposing nodes")
        for node_group in node_groups:
            _multi_graph = _squash_adjacent(
                _multi_graph,
                node_group,
                centroid_by_itx=centroid_by_itx,
                prioritise_by_hwy_tag=prioritise_by_hwy_tag,
                simplify_by_max_angle=simplify_by_max_angle,
            )
    else:
        for node_group in node_groups:
            origin_nd_key = node_group.pop(0)  # type: ignore
            template = None
            origin_nd_data = _multi_graph.nodes[origin_nd_key]
            for new_nd_key in node_group:
                new_nd_data = _multi_graph.nodes[new_nd_key]
                new_geom = geometry.LineString(
                    [
                        [origin_nd_data["x"], origin_nd_data["y"]],
                        [new_nd_data["x"], new_nd_data["y"]],
                    ]
                )
                # don't add overly similar new edges
                if template is None:
                    template = new_geom.buffer(5, cap_style=BufferCapStyle.flat)
                elif template.contains(new_geom):
                    continue
                else:
                    template = template.union(new_geom.buffer(10, cap_style=BufferCapStyle.flat))
                # don't add new edges that would criss cross existing
                bail = False
                new_end_pnt = geometry.Point(new_nd_data["x"], new_nd_data["y"])
                edge_hits = edges_tree.query(new_geom)  # type: ignore
                for edge_hit_idx in edge_hits:
                    edge_lookup = edge_lookups[edge_hit_idx]
                    start_nd_key = edge_lookup["start_nd_key"]
                    end_nd_key = edge_lookup["end_nd_key"]
                    edge_idx = edge_lookup["edge_idx"]
                    edge_geom: dict = nx_multigraph[start_nd_key][end_nd_key][edge_idx]["geom"]
                    # use distance to catch "crossing" where curved geoms lead to issues
                    if new_geom.crosses(edge_geom) and round(new_end_pnt.distance(edge_geom), 3) > 0:  # type: ignore
                        bail = True
                        break
                if bail:
                    continue
                # add
                if not _multi_graph.has_edge(origin_nd_key, new_nd_key):
                    _multi_graph.add_edge(
                        origin_nd_key,
                        new_nd_key,
                        names=[],
                        routes=[],
                        highways=[],
                        levels=[],
                        geom=new_geom,
                    )
    # squashing nodes can result in edge duplicates
    deduped_graph = nx_merge_parallel_edges(
        _multi_graph,
        merge_edges_by_midline,
        contains_buffer_dist,
    )

    return deduped_graph


def nx_decompose(
    nx_multigraph: MultiGraph,
    decompose_max: float,
    osm_hwy_target_tags: list[str] | None = None,
) -> MultiGraph:
    """
    Decomposes a graph so that no edge is longer than a set maximum.

    Decomposition provides a more granular representation of potential variations along street lengths, while reducing
    network centrality side-effects that arise as a consequence of varied node densities.

    :::note
    Setting the `decompose` parameter too small in relation to the size of the graph may increase the computation time
    unnecessarily for subsequent analysis. For larger-scale urban analysis, it is generally not necessary to go smaller
    20m, and 50m may already be sufficient for the majority of cases.
    :::

    Parameters
    ----------
    nx_multigraph: MultiGraph
        A `networkX` `MultiGraph` in a projected coordinate system, containing `x` and `y` node attributes, and `geom`
        edge attributes containing `LineString` geoms.
    decompose_max: float
        The maximum length threshold for decomposed edges.
    osm_hwy_target_tags: list[str]
        An optional list of OpenStreetMap target highway tags. If provided, only nodes with neighbouring edges
        containing a tag matching one of the target OSM highway tags will be decomposed. Requires graph prepared with
        via [`io.osm_graph_from_poly`](/io#osm-graph-from-poly).

    Returns
    -------
    MultiGraph
        A decomposed `networkX` graph with no edge longer than the `decompose_max` parameter. If `live` node attributes
        were provided, then the `live` attribute for child-nodes will be set to `True` if either or both parent nodes
        were `live`. Otherwise, all nodes wil be set to `live=True`. The `length` and `imp_factor` edge attributes will
        be set to match the lengths of the new edges.

    Examples
    --------
    ```python
    from cityseer.tools import mock, graphs, plot

    G = mock.mock_graph()
    G_simple = graphs.nx_simple_geoms(G)
    G_decomposed = graphs.nx_decompose(G_simple, 100)
    plot.plot_nx(G_decomposed)
    ```

    ![Example graph](/images/graph_simple.png)
    _Example graph prior to decomposition._

    ![Example decomposed graph](/images/graph_decomposed.png)
    _Example graph after decomposition._

    """
    if not isinstance(nx_multigraph, nx.MultiGraph):
        raise TypeError("This method requires an undirected networkX MultiGraph.")
    logger.info(f"Decomposing graph to maximum edge lengths of {decompose_max}.")
    g_multi_copy: MultiGraph = nx_multigraph.copy()
    # if using OSM tags heuristic
    hwy_tags = _extract_tags_to_set(osm_hwy_target_tags)
    # note -> write to a duplicated graph to avoid in-place errors
    start_nd_key: NodeKey
    end_nd_key: NodeKey
    edge_data: EdgeData
    for start_nd_key, end_nd_key, edge_data in tqdm(nx_multigraph.edges(data=True), disable=config.QUIET_MODE):
        # hwy tags
        if osm_hwy_target_tags:
            edge_hwy_tags = _tags_from_edge_key(edge_data, "highways")
            if not hwy_tags.intersection(edge_hwy_tags):
                continue
        # test for x, y in start coordinates
        if "x" not in nx_multigraph.nodes[start_nd_key] or "y" not in nx_multigraph.nodes[start_nd_key]:
            raise KeyError(f'Encountered node missing "x" or "y" coordinate attributes at node {start_nd_key}.')
        # test for x, y in end coordinates
        if "x" not in nx_multigraph.nodes[end_nd_key] or "y" not in nx_multigraph.nodes[end_nd_key]:
            raise KeyError(f'Encountered node missing "x" or "y" coordinate attributes at node {end_nd_key}.')
        # test for geom
        if "geom" not in edge_data:
            raise KeyError(
                f"No edge geom found for edge {start_nd_key}-{end_nd_key}: "
                f'Please add an edge "geom" attribute consisting of a shapely LineString.'
            )
        # get edge geometry
        line_geom: geometry.LineString = edge_data["geom"]
        if line_geom.geom_type != "LineString":
            raise TypeError(
                f"Expected LineString geometry but found {line_geom.geom_type} for edge {start_nd_key}-{end_nd_key}."
            )
        # check geom coordinates directionality - flip if facing backwards direction
        line_geom_coords = util.snap_linestring_endpoints(
            nx_multigraph,
            start_nd_key,
            end_nd_key,
            line_geom.coords,  # type:ignore
        )
        line_geom: geometry.LineString = geometry.LineString(line_geom_coords)
        # see how many segments are necessary so as not to exceed decomposition max distance
        # note that a length less than the decompose threshold will result in a single 'sub'-string
        cuts: int = int(np.ceil(line_geom.length / decompose_max))
        step_size: float = line_geom.length / cuts
        # since decomposing, remove the prior edge... but only after properties have been read
        g_multi_copy.remove_edge(start_nd_key, end_nd_key)
        # then add the new sub-edge/s
        step = 0
        prior_node_id = start_nd_key
        sub_node_counter = 0
        # everything inside this loop is a new node - i.e. this loop is effectively skipped if cuts = 1
        for sub_node_counter in range(cuts - 1):
            # create the split LineString geom for measuring the new length
            # switch back to shapely once bug resolved
            line_segment: geometry.LineString = ops.substring(line_geom, step, step + step_size)  # type: ignore
            # get the x, y of the new end node
            x, y = line_segment.coords[-1]
            # add the new node and edge
            new_nd_name, is_dupe = util.add_node(
                g_multi_copy,
                [start_nd_key, sub_node_counter, end_nd_key],  # type: ignore
                x=x,
                y=y,  # type:ignore
            )
            if is_dupe:
                raise ValueError(
                    f"Attempted to add a duplicate node at x: {x}, y:{y}. "
                    f"Check for existence of duplicate edges in the vicinity of {start_nd_key}-{end_nd_key}."
                )
            sub_node_counter += 1
            # add and set live property if present in parent graph
            if "live" in nx_multigraph.nodes[start_nd_key] and "live" in nx_multigraph.nodes[end_nd_key]:
                live = True
                # if BOTH parents are not live, then set child to not live
                if not nx_multigraph.nodes[start_nd_key]["live"] and not nx_multigraph.nodes[end_nd_key]["live"]:
                    live = False
                g_multi_copy.nodes[new_nd_name]["live"] = live
            # add the edge
            edge_data_copy = {k: v for k, v in edge_data.items() if k != "geom"}
            g_multi_copy.add_edge(prior_node_id, new_nd_name, geom=line_segment, **edge_data_copy)
            # increment the step and node id
            prior_node_id = new_nd_name
            step += step_size
        # set the last edge manually to avoid rounding errors at end of LineString
        # the nodes already exist, so just add edge
        # switch back to shapely once bug resolved
        line_segment = ops.substring(line_geom, step, line_geom.length)  # type: ignore
        edge_data_copy = {k: v for k, v in edge_data.items() if k != "geom"}
        g_multi_copy.add_edge(prior_node_id, end_nd_key, geom=line_segment, **edge_data_copy)

    return g_multi_copy


def nx_to_dual(nx_multigraph: MultiGraph) -> MultiGraph:
    """
    Convert a primal graph representation to the dual representation.

    Primal graphs represent intersections as nodes and streets as edges. This method will invert this representation
    so that edges are converted to nodes and intersections become edges. Primal edge `geom` attributes will be welded to
    adjacent edges and split into the new dual edge `geom` attributes.

    :::note
    Note that a `MultiGraph` is useful for primal but not for dual, so the output `MultiGraph` will have single edges.
    e.g. a crescent street that spans the same intersections as parallel straight street requires multiple edges in
    primal. The same type of situation does not arise in the dual because the nodes map to distinct edges regardless.
    :::

    Parameters
    ----------
    nx_multigraph: MultiGraph
        A `networkX` `MultiGraph` in a projected coordinate system, containing `x` and `y` node attributes, and `geom`
        edge attributes containing `LineString` geoms.

    Returns
    -------
    MultiGraph
        A dual representation `networkX` graph. The new dual nodes will have `x` and `y` node attributes corresponding
        to the mid-points of the original primal edges. If `live` node attributes were provided, then the `live`
        attribute for the new dual nodes will be set to `True` if either or both of the adjacent primal nodes were set
        to `live=True`. Otherwise, all dual nodes wil be set to `live=True`. The primal edges will be split and welded
        to form the new dual `geom` edges. The primal `LineString` `geom` will be saved to the dual node's `primal_edge`
        attribute. `primal_edge_node_a`, `primal_edge_node_b`, and `primal_edge_idx` attributes will be added to the new
        (dual) nodes, and a `primal_node_id` edge attribute will be added to the new (dual) edges.

    Examples
    --------
    ```python
    from cityseer.tools import graphs, mock, plot

    G = mock.mock_graph()
    G_simple = graphs.nx_simple_geoms(G)
    G_dual = graphs.nx_to_dual(G_simple)
    plot.plot_nx_primal_or_dual(G_simple, G_dual, plot_geoms=False)
    ```

    ![Example dual graph](/images/graph_dual.png)
    _Dual graph (blue) overlaid on the source primal graph (red)._

    """
    if not isinstance(nx_multigraph, nx.MultiGraph):
        raise TypeError("This method requires an undirected networkX MultiGraph.")
    logger.info("Converting graph to dual.")
    g_dual: MultiGraph = nx.MultiGraph()
    g_dual.graph["is_dual"] = True

    def get_half_geoms(nx_multigraph_ref: MultiGraph, a_node: NodeKey, b_node: NodeKey, edge_idx: int):  # type: ignore
        """
        Split geom and orient half-geoms.
        """
        # get edge data
        edge_data: EdgeData = nx_multigraph_ref[a_node][b_node][edge_idx]
        # test for x coordinates
        if "x" not in nx_multigraph_ref.nodes[a_node] or "y" not in nx_multigraph_ref.nodes[a_node]:
            raise KeyError(f'Encountered node missing "x" or "y" coordinate attributes at node {a_node}.')
        # test for y coordinates
        if "x" not in nx_multigraph_ref.nodes[b_node] or "y" not in nx_multigraph_ref.nodes[b_node]:
            raise KeyError(f'Encountered node missing "x" or "y" coordinate attributes at node {b_node}.')
        a_node_data: NodeData = nx_multigraph_ref.nodes[a_node]
        a_xy = (a_node_data["x"], a_node_data["y"])
        b_node_data: NodeData = nx_multigraph_ref.nodes[b_node]
        b_xy = (b_node_data["x"], b_node_data["y"])
        # test for geom
        if "geom" not in edge_data:
            raise KeyError(
                f"No edge geom found for edge {a_node}-{b_node}: "
                f'Please add an edge "geom" attribute consisting of a shapely LineString.'
            )
        # get edge geometry
        line_geom = edge_data["geom"]
        if line_geom.geom_type != "LineString":
            raise TypeError(
                f"Expecting LineString geometry but found {line_geom.geom_type} geometry for edge {a_node}-{b_node}."
            )
        # align geom coordinates to start from A side
        line_geom_coords = util.align_linestring_coords(line_geom.coords, a_xy)
        line_geom = geometry.LineString(line_geom_coords)
        # generate the two half geoms
        # switch back to shapely once bug resolved
        a_half_geom: geometry.LineString = ops.substring(line_geom, 0.0, 0.5, normalized=True)  # type: ignore
        b_half_geom: geometry.LineString = ops.substring(line_geom, 0.5, 1.0, normalized=True)  # type: ignore
        # check that nothing odd happened with new midpoint
        if not np.allclose(
            a_half_geom.coords[-1][:2],
            b_half_geom.coords[0][:2],
            atol=config.ATOL,
            rtol=0,
        ):
            raise ValueError("Nodes of half geoms don't match")
        # snap to prevent creeping tolerance issues
        # A side geom starts at node A and ends at new midpoint
        a_half_geom_coords = util.snap_linestring_startpoint(a_half_geom.coords, a_xy)
        # snap new midpoint to geom A's endpoint (i.e. no need to snap endpoint of geom A)
        mid_xy = a_half_geom_coords[-1][:2]
        # B side geom starts at mid and ends at B node
        b_half_geom_coords = util.snap_linestring_startpoint(b_half_geom.coords, mid_xy)  # type: ignore
        b_half_geom_coords = util.snap_linestring_endpoint(b_half_geom_coords, b_xy)
        # double check coords
        if (
            a_half_geom_coords[0][:2] != a_xy
            or a_half_geom_coords[-1][:2] != mid_xy
            or b_half_geom_coords[0][:2] != mid_xy
            or b_half_geom_coords[-1][:2] != b_xy
        ):
            raise ValueError("Nodes of half geoms don't match")

        return geometry.LineString(a_half_geom_coords), geometry.LineString(b_half_geom_coords)

    def set_live(start_nd_key: NodeKey, end_nd_key: NodeKey, dual_node_key: NodeKey):
        # add and set live property to dual node if present in adjacent primal graph nodes
        if "live" in nx_multigraph.nodes[start_nd_key] and "live" in nx_multigraph.nodes[end_nd_key]:
            live = True
            # if BOTH parents are not live, then set child to not live
            if not nx_multigraph.nodes[start_nd_key]["live"] and not nx_multigraph.nodes[end_nd_key]["live"]:
                live = False
            g_dual.nodes[dual_node_key]["live"] = live

    def prepare_dual_node_key(start_nd_key: NodeKey, end_nd_key: NodeKey, edge_idx: int) -> str:
        s_e = sorted([str(start_nd_key), str(end_nd_key)])
        return f"{s_e[0]}_{s_e[1]}_k{edge_idx}"

    # add dual nodes
    logger.info("Preparing dual nodes")
    for start_nd_key, end_nd_key, edge_idx, edge_data in tqdm(  # type: ignore
        nx_multigraph.edges(data=True, keys=True),  # type: ignore
        disable=config.QUIET_MODE,  # type: ignore
    ):
        primal_geom = edge_data["geom"]
        mid_point = primal_geom.interpolate(0.5, normalized=True)  # type: ignore
        dual_node_key = prepare_dual_node_key(start_nd_key, end_nd_key, edge_idx)
        # create a new dual node corresponding to the current primal edge
        g_dual.add_node(
            dual_node_key,
            x=mid_point.x,
            y=mid_point.y,
            primal_edge=primal_geom,
            primal_edge_node_a=start_nd_key,
            primal_edge_node_b=end_nd_key,
            primal_edge_idx=edge_idx,
        )
        # add and set live property if present in parent graph
        set_live(start_nd_key, end_nd_key, dual_node_key)
    # add dual edges
    logger.info("Preparing dual edges (splitting and welding geoms)")
    for start_nd_key, end_nd_key, edge_idx in tqdm(  # type: ignore
        nx_multigraph.edges(data=False, keys=True),  # type: ignore
        disable=config.QUIET_MODE,  # type: ignore
    ):
        hub_node_dual = prepare_dual_node_key(start_nd_key, end_nd_key, edge_idx)
        # get the first and second half geoms
        s_half_geom, e_half_geom = get_half_geoms(nx_multigraph, start_nd_key, end_nd_key, edge_idx)
        # process either side
        for n_side, m_side, half_geom in zip(
            [start_nd_key, end_nd_key],
            [end_nd_key, start_nd_key],
            [s_half_geom, e_half_geom],
            strict=False,
        ):
            # add the spoke edges on the dual
            nb_nd_key: NodeKey
            for nb_nd_key in nx.neighbors(nx_multigraph, n_side):
                # don't follow neighbour back to current edge combo
                if nb_nd_key == m_side:
                    continue
                # add the neighbouring primal edges as dual nodes
                for edge_idx in nx_multigraph[n_side][nb_nd_key]:
                    spoke_node_dual = prepare_dual_node_key(n_side, nb_nd_key, edge_idx)
                    # skip if the edge has already been processed from another direction
                    if g_dual.has_edge(hub_node_dual, spoke_node_dual):
                        continue
                    # get the near and far half geoms
                    spoke_half_geom, _discard_geom = get_half_geoms(nx_multigraph, n_side, nb_nd_key, edge_idx)
                    # weld the lines
                    merged_line: geometry.LineString = ops.linemerge([half_geom, spoke_half_geom])  # type: ignore
                    if merged_line.geom_type != "LineString":
                        raise TypeError(
                            f'Found {merged_line.geom_type} instead of "LineString" for new geom {merged_line.wkt}. '
                            f"Check that the LineStrings for {start_nd_key}-{end_nd_key} & {n_side}-{nb_nd_key} touch."
                        )
                    # add the dual edge
                    g_dual.add_edge(
                        hub_node_dual,
                        spoke_node_dual,
                        primal_node_id=n_side,
                        geom=merged_line,
                    )

    return g_dual


def nx_weight_by_dissolved_edges(
    nx_multigraph: MultiGraph, dissolve_distance: int = 20, max_ang_diff: int = 45
) -> MultiGraph:
    """
    Generates graph node weightings based on the ratio of directly adjacent edges to total nearby edges.

    This is used to control for unintended amplification of centrality measures where redundant network representations
    (e.g. duplicitious segments such as adjacent street, sidewalk, cycleway, busway) tend to inflate centrality scores.
    This method is intended for 'messier' network representations (e.g. OSM).

    > This method is only recommended for primal graph representations.

    Parameters
    ----------
    nx_multigraph: MultiGraph
        A `networkX` `MultiGraph` in a projected coordinate system, containing `x` and `y` node attributes, and `geom`
        edge attributes containing `LineString` geoms.
    dissolve_distance: int
        A distance to use when buffering edges to calculate the weighting. 20m by default.
    max_ang_diff: int
         Only count a nearby adjacent edge as duplicitous if the angular difference between edges is less than
         `max_ang_diff`. 45 degrees by default.

    Returns
    -------
    MultiGraph
        A `networkX` graph. The nodes will have a new `weight` parameter indicating the node's contribution given the
        locally 'dissolved' context.

    """
    # note it is better to weight via edges than via nodes this is because offset / staggered nodes
    # (intersections on one side of parallel road) might not otherwise trigger de-duplication via weights
    if not isinstance(nx_multigraph, nx.MultiGraph):
        raise TypeError("This method requires an undirected networkX MultiGraph.")
    logger.info(f"Generating node weights based on locally dissolved edges using a buffer of {dissolve_distance}m.")
    g_multi_copy: MultiGraph = nx_multigraph.copy()
    # generate STR tree
    edges_tree, edge_lookups = util.create_edges_strtree(g_multi_copy)
    # first iterate edges to save number of iters
    for start_nd_key, end_nd_key, edge_idx, edge_data in tqdm(
        g_multi_copy.edges(data=True, keys=True), disable=config.QUIET_MODE
    ):
        # start scratch variable
        g_multi_copy[start_nd_key][end_nd_key][edge_idx]["nearby_itx_lens"] = 0
        # find nearby edges
        edge_geom = edge_data["geom"]
        edge_geom_buff = edge_geom.buffer(dissolve_distance, cap_style=geometry.CAP_STYLE.square)
        edges_hits: list[int] = edges_tree.query(edge_geom_buff)  # type: ignore
        for edge_hit_idx in edges_hits:
            edge_lookup = edge_lookups[edge_hit_idx]
            nearby_start_nd_key = edge_lookup["start_nd_key"]
            nearby_end_nd_key = edge_lookup["end_nd_key"]
            nearby_edge_idx = edge_lookup["edge_idx"]
            # don't add edges which share common nodes (directly adjacent)
            # this is an important line because otherwise the measure becomes indiscriminate i.e. would apply to regular
            # intersections without duplicitous edges - working against intention of this method
            if nearby_start_nd_key in [start_nd_key, end_nd_key] or nearby_end_nd_key in [
                start_nd_key,
                end_nd_key,
            ]:
                continue
            # get linestring
            nearby_edge_data = g_multi_copy[nearby_start_nd_key][nearby_end_nd_key][nearby_edge_idx]
            nearby_edge_geom: geometry.LineString = nearby_edge_data["geom"]
            # get angular difference
            ang_diff = util.measure_angle_diff_betw_linestrings(edge_geom.coords, nearby_edge_geom.coords)
            if ang_diff > max_ang_diff:
                continue
            # find length of geom intersecting buff
            edge_itx = nearby_edge_geom.intersection(edge_geom_buff)
            if edge_itx and edge_itx.geom_type == "LineString":
                g_multi_copy[start_nd_key][end_nd_key][edge_idx]["nearby_itx_lens"] += edge_itx.length
    # gather out edges
    for nd_key in tqdm(g_multi_copy.nodes(), disable=config.QUIET_MODE):
        adjacent_lens = 0
        total_lens = 0
        for nb_nd_key in nx.neighbors(g_multi_copy, nd_key):
            for nb_edge_data in g_multi_copy[nd_key][nb_nd_key].values():
                edge_geom = nb_edge_data["geom"]
                adjacent_lens += edge_geom.length
                total_lens += edge_geom.length
                total_lens += nb_edge_data["nearby_itx_lens"]
        # calculate ratio
        weight = 1
        if total_lens > dissolve_distance:
            weight = adjacent_lens / total_lens
        g_multi_copy.nodes[nd_key]["weight"] = weight

    return g_multi_copy


def nx_generate_vis_lines(nx_multigraph: MultiGraph) -> MultiGraph:
    """
    Generates a `line_geom` property for nodes consisting `MultiLineString` geoms for visualisation purposes.

    This method can be used if preferring to visualise the outputs as lines instead of points. The lines are assembled
    from the adjacent half segments.

    Parameters
    ----------
    nx_multigraph: MultiGraph
        A `networkX` `MultiGraph` in a projected coordinate system, containing `x` and `y` node attributes, and `geom`
        edge attributes containing `LineString` geoms.

    Returns
    -------
    MultiGraph
        A `networkX` graph. The nodes will have a new `line_geom` parameter containing `shapely` `MultiLineString`
        geoms.

    """
    if not isinstance(nx_multigraph, nx.MultiGraph):
        raise TypeError("This method requires an undirected networkX MultiGraph.")
    logger.info("Preparing LineStrings for node visualisation.")
    g_multi_copy: MultiGraph = nx_multigraph.copy()
    # gather out edges
    for nd_key, nd_data in tqdm(g_multi_copy.nodes(data=True), disable=config.QUIET_MODE):
        line_geoms: list[geometry.LineString] = []
        for nb_nd_key in nx.neighbors(g_multi_copy, nd_key):
            for nb_edge_data in g_multi_copy[nd_key][nb_nd_key].values():
                # slice nearest halves and gather
                edge_geom = nb_edge_data["geom"]
                edge_geom = geometry.LineString(
                    util.align_linestring_coords(edge_geom.coords, (nd_data["x"], nd_data["y"]))
                )
                edge_slice = ops.substring(edge_geom, 0, 0.5, normalized=True)
                line_geoms.append(edge_slice)  # type: ignore
        # build line geom
        g_multi_copy.nodes[nd_key]["line_geom"] = geometry.MultiLineString(line_geoms)

    return g_multi_copy
