# %%
from typing import Any, cast

import geopandas as gpd
import networkx as nx
import numpy as np
from cityseer.tools import io
from overturemaps import core
from pyproj import Transformer
from shapely import geometry, ops
from shapely.ops import transform

Connector = tuple[Any, geometry.Point]


def split_street_segment(
    line_string: geometry.LineString, connector_infos: list[Connector]
) -> list[tuple[geometry.LineString, Connector, Connector]]:
    """ """
    # overture segments can span multiple intersections
    # sort through and split until pairings are ready for insertion to the graph
    node_segment_pairs: list[tuple[geometry.LineString, Connector, Connector]] = []
    node_segment_lots: list[tuple[geometry.LineString, list[Connector]]] = [(line_string, connector_infos)]
    # start iterating
    while node_segment_lots:
        old_line_string, old_connectors = node_segment_lots.pop()
        # filter down connectors
        new_connectors: list[tuple[str, geometry.Point]] = []
        # if the point doesn't touch the line, discard
        for _fid, _point in old_connectors:
            if _point.distance(old_line_string) > 0:
                continue
            new_connectors.append((_fid, _point))
        # if only two connectors
        if len(new_connectors) == 2:
            node_segment_pairs.append((old_line_string, new_connectors[0], new_connectors[1]))
            continue
        # look for splits
        for _fid, _point in new_connectors:
            splits = ops.split(old_line_string, _point)
            # continue if an endpoint
            if len(splits.geoms) == 1:
                continue
            # otherwise unpack
            line_string_a, line_string_b = splits.geoms
            # otherwise split into two bundles and reset
            node_segment_lots.append((cast(geometry.LineString, line_string_a), new_connectors))
            node_segment_lots.append((cast(geometry.LineString, line_string_b), new_connectors))
            break
    return node_segment_pairs


def generate_graph(
    nodes_gdf: gpd.GeoDataFrame,
    edges_gdf: gpd.GeoDataFrame,
    drop_road_types: list[str] | None = None,
) -> nx.MultiGraph:
    """ """
    if drop_road_types is None:
        drop_road_types = []
    # create graph
    multigraph = nx.MultiGraph()
    # filter by boundary and build nx
    # dedupe nodes by coordinate while keeping a lookup back to original ids
    xy_to_id: dict[str, Any] = {}
    id_to_merged: dict[Any, Any] = {}
    for node_idx, node_geom in nodes_gdf["geom"].items():
        node_geom_point = cast(geometry.Point, node_geom)
        x = node_geom_point.x
        y = node_geom_point.y
        xy_key = f"{x}-{y}"
        merged_key = xy_to_id.setdefault(xy_key, node_idx)
        id_to_merged[node_idx] = merged_key
        id_to_merged[str(node_idx)] = merged_key
        if not multigraph.has_node(merged_key):
            multigraph.add_node(
                merged_key,
                x=x,
                y=y,
            )
    dropped_road_types = set()
    kept_road_types = set()
    for edge_idx, edges_data in edges_gdf.iterrows():
        road_class = edges_data["class"]
        if road_class in drop_road_types:
            dropped_road_types.add(road_class)
            continue
        kept_road_types.add(road_class)
        connectors_data = edges_data["connectors"]
        if not isinstance(connectors_data, (list, tuple, np.ndarray)) or len(connectors_data) == 0:
            continue
        uniq_fids = set()
        connector_fids: list[Any] = []
        for connector in connectors_data:
            connector_fid = connector.get("connector_id")
            if connector_fid is not None:
                connector_fids.append(connector_fid)
        connector_infos: list[Connector] = []
        missing_connectors = False
        for connector_fid in connector_fids:
            # skip malformed edges - this happens at boundary thresholds with missing nodes in relation to edges
            merged_key = id_to_merged.get(connector_fid)
            if merged_key is None or not multigraph.has_node(merged_key):
                missing_connectors = True
                break
            # deduplicate
            x, y = multigraph.nodes[merged_key]["x"], multigraph.nodes[merged_key]["y"]
            if merged_key in uniq_fids:
                continue
            uniq_fids.add(merged_key)
            # track
            connector_point = geometry.Point(x, y)
            connector_infos.append((merged_key, connector_point))
        if missing_connectors is True:
            continue
        if len(connector_infos) < 2:
            # logger.warning("Only one connector pair for edge")
            continue
        # extract levels, names, routes, highways
        # do this once instead of for each new split segment
        levels = set([])
        if edges_data["level_rules"] is not None:
            for level_info in edges_data["level_rules"]:
                levels.add(level_info["value"])
        names = []  # takes list form for nx
        if edges_data["names"] is not None and "primary" in edges_data["names"]:
            names.append(edges_data["names"]["primary"])
        routes = set([])
        if edges_data["routes"] is not None:
            for routes_info in edges_data["routes"]:
                if "ref" in routes_info:
                    routes.add(routes_info["ref"])
        is_tunnel = False
        is_bridge = False
        if edges_data["road_flags"] is not None:
            for flags_info in edges_data["road_flags"]:
                if "is_tunnel" in flags_info["values"]:
                    is_tunnel = True
                if "is_bridge" in flags_info["values"]:
                    is_bridge = True
        highways = []  # takes list form for nx
        if road_class is not None and road_class not in ["unknown"]:
            highways.append(road_class)
        # split segments and build
        edge_geom = cast(geometry.LineString, edges_data["geom"])
        street_segs = split_street_segment(edge_geom, connector_infos)
        for seg_geom, node_info_a, node_info_b in street_segs:
            if not node_info_a[1].touches(seg_geom) or not node_info_b[1].touches(seg_geom):
                raise ValueError(
                    "Edge and endpoint connector are not touching. "
                    f"See connectors: {node_info_a[0]} and {node_info_b[0]}"
                )
            # don't add duplicates
            dupe = False
            if multigraph.has_edge(node_info_a[0], node_info_b[0]):
                edges = multigraph[node_info_a[0]][node_info_b[0]]
                for _edge_idx, edge_val in dict(edges).items():
                    if edge_val["geom"].buffer(1).contains(seg_geom):
                        dupe = True
                        break
            if dupe is False:
                multigraph.add_edge(
                    node_info_a[0],
                    node_info_b[0],
                    edge_idx=edge_idx,
                    geom=seg_geom,
                    levels=list(levels),
                    names=names,
                    routes=list(routes),
                    highways=highways,
                    is_bridge=is_bridge,
                    is_tunnel=is_tunnel,
                )

    return multigraph


# %%
WORKING_CRS = 3035

# BOUNDARY
# bounds_geom = gpd.read_file("boundaries.gpkg")
# bounds_geom_wgs = bounds_geom.to_crs(4326).iloc[336]["geometry"]
p = geometry.Point(4062780.86, 4152161.71)
buff = p.buffer(300)
transformer = Transformer.from_crs(WORKING_CRS, 4326, always_xy=True)
buff_wgs = transform(transformer.transform, buff)
buff_wgs

# %%
# NODES
nodes_gdf = core.geodataframe("connector", buff_wgs.bounds)  # type:ignore
nodes_gdf.set_crs(4326, inplace=True)
nodes_gdf.to_crs(WORKING_CRS, inplace=True)
nodes_gdf.set_index("id", inplace=True)
nodes_gdf.rename(columns={"geometry": "geom"}, inplace=True)
nodes_gdf.set_geometry("geom", inplace=True)
nodes_gdf.drop(columns=["bbox"], inplace=True)
nodes_gdf.head()
nodes_gdf.to_file("tttest_raw_nodes.gpkg", driver="GPKG")

# %%
# EDGES
edges_gdf = core.geodataframe("segment", buff_wgs.bounds)  # type:ignore
edges_gdf.set_crs(4326, inplace=True)
edges_gdf.to_crs(WORKING_CRS, inplace=True)
edges_gdf.set_index("id", inplace=True)
edges_gdf.rename(columns={"geometry": "geom"}, inplace=True)
edges_gdf.set_geometry("geom", inplace=True)
edges_gdf.drop(columns=["bbox"], inplace=True)
edges_gdf.head()
edges_gdf.to_file("tttest_raw_edges.gpkg", driver="GPKG")

# %%
# CLEAN
edges_gdf = edges_gdf[edges_gdf["subtype"] == "road"]  # type: ignore
multigraph = generate_graph(
    nodes_gdf=nodes_gdf,  # type: ignore
    edges_gdf=edges_gdf,  # type: ignore
    # not dropping "parking_aisle" because this sometimes removes important links
)
multigraph.graph["crs"] = WORKING_CRS
edges_split = io.geopandas_from_nx(multigraph)
edges_split.to_file("tttest_cleaned_edges.gpkg", driver="GPKG")

# %%
multigraph = io._auto_clean_network(
    multigraph,
    geom_wgs=buff_wgs,
    to_crs_code=WORKING_CRS,
    final_clean_distances=(8,),
    remove_disconnected=100,
    green_footways=True,
    green_service_roads=False,
)

# %%
edges = io.geopandas_from_nx(multigraph)
edges.to_file("tttest_edges.gpkg", driver="GPKG")

# %%
agg_nodes_data = []
for nd_key, nd_data in multigraph.nodes(data=True):  # type: ignore
    agg_nodes_data.append((nd_key, geometry.Point(nd_data["x"], nd_data["y"])))
nodes_gdf = gpd.GeoDataFrame(agg_nodes_data, columns=["node_key", "geometry"], crs=multigraph.graph["crs"])
nodes_gdf.to_file("tttest_nodes.gpkg")

# %%
