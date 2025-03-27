# %%
from pathlib import Path

import networkx as nx
import pandas as pd
from cityseer.tools import graphs, io, mock
from cityseer.metrics import networks
from cityseer import rustalgos
from pyproj import Transformer

# %%
# prepare GTFS data
G: nx.MultiGraph = mock.mock_graph()
G = graphs.nx_simple_geoms(G)
gtfs_data_path = "temp"
mock.mock_gtfs_stops_txt(gtfs_data_path)

graph_crs = 32630
nodes_gdf, edges_gdf, network_structure = io.network_structure_from_nx(G, graph_crs)

# %%
# add GTFS data to network structure
# network_structure = io.add_transport_gtfs(gtfs_data_path, network_structure, graph_crs)
gtfs_path = Path(gtfs_data_path)
if not gtfs_path.exists():
    raise FileNotFoundError(f"GTFS data not found at {gtfs_data_path}")
if not (gtfs_path / "stops.txt").exists():
    raise FileNotFoundError(f"GTFS stops.txt not found at {gtfs_data_path}")
if not (gtfs_path / "stop_times.txt").exists():
    raise FileNotFoundError(f"GTFS stop_times.txt not found at {gtfs_data_path}")

# %%
# Load GTFS data
stops = pd.read_csv(gtfs_path / "stops.txt")
stops["stop_id"] = stops["stop_id"].astype(str)
stops

# %%
stop_times = pd.read_csv(gtfs_path / "stop_times.txt")
stop_times["stop_id"] = stop_times["stop_id"].astype(str)
stop_times

# %%
stop_times["arrival_time"] = pd.to_datetime(stop_times["arrival_time"], format="%H:%M:%S").dt.time
stop_times["arrival_seconds"] = stop_times["arrival_time"].apply(lambda t: t.hour * 3600 + t.minute * 60 + t.second)
stop_times.sort_values(by=["stop_id", "arrival_seconds"], inplace=True)
stop_times

# %%
# Compute gaps between consecutive arrivals
stop_times["wait_time"] = stop_times.groupby("stop_id")["arrival_seconds"].diff()
stop_times

# %%
avg_wait_time = stop_times.groupby("stop_id")["wait_time"].mean().fillna(0)
avg_wait_time = avg_wait_time / 2
avg_wait_time

#%%
# Merge avg_wait_time into stop_times
stops = stops.merge(avg_wait_time.rename("avg_wait_time"), on="stop_id", how="left")
stops

# %%
stop_lookups = {}
max_netw_assign_dist = 400
# transformer to convert lat/lon to graph crs
transformer = Transformer.from_crs(4326, graph_crs, always_xy=True)
# add nodes for stops
for _, row in stops.iterrows():
    # wait_time = avg_wait_time.get(row["stop_id"], 0)  # Default to 0 if missing
    e, n = transformer.transform(row["stop_lon"], row["stop_lat"])
    station_coord = rustalgos.Coord(e, n)
    nearest_idx, next_nearest_idx = network_structure.assign_to_network(station_coord, max_netw_assign_dist)
    new_stop_key = str(row["stop_id"])
    new_stop_idx = network_structure.add_node(
        new_stop_key,
        float(e),
        float(n),
        True,  # live
        float(1),  # weight
    )
    # add to lookups
    stop_lookups[new_stop_key] = new_stop_idx
    # add edges between stops and pedestrian network
    for near_node_idx in [nearest_idx, next_nearest_idx]:
        if near_node_idx is not None:
            netw_node = network_structure.get_node_payload(near_node_idx)
            dist = netw_node.coord.hypot(station_coord)
            print(near_node_idx, dist)
            # to direction
            network_structure.add_edge(
                near_node_idx,
                new_stop_idx,
                0,  # edge_idx
                "na-gtfs",  # nx_start_node_key
                "na-gtfs",  # nx_end_node_key
                dist,  # length - don't use zero otherwise short-cutting will occur
                180,  # angle_sum - don't use zero otherwise short-cutting will occur
                1,  # imp_factor
                None,  # in_bearing
                None,  # out_bearing
                float(row['avg_wait_time']),  # seconds
            )
            # from direction
            network_structure.add_edge(
                new_stop_idx,
                near_node_idx,
                1,  # edge_idx
                "na-gtfs",  # nx_start_node_key
                "na-gtfs",  # nx_end_node_key
                dist,  # length - don't use zero otherwise short-cutting will occur
                180,  # angle_sum - don't use zero otherwise short-cutting will occur
                1,  # imp_factor
                None,  # in_bearing
                None,  # out_bearing
                float(0),  # seconds
            )

# %%
# Create a column for the previous stop in each trip
stop_times["prev_stop_id"] = stop_times.groupby("trip_id")["stop_id"].shift()

# Sort stop_times for proper sequencing
stop_times.sort_values(by=["trip_id", "stop_sequence"], inplace=True)
stop_times

# %%
# Compute travel time between consecutive stops
stop_times["segment_time"] = stop_times.groupby("trip_id")["arrival_seconds"].diff()
stop_times

# %%
avg_stop_pairs = (
    stop_times
    .dropna(subset=["prev_stop_id"])  # remove rows where prev_stop_id is NaN
    .groupby(["prev_stop_id", "stop_id"])["segment_time"]
    .mean()
    .reset_index(name="avg_segment_time")
)
avg_stop_pairs

#%%
# add edges between stops
for _, row in avg_stop_pairs.iterrows():
    # prev stop
    prev_stop = row["prev_stop_id"]
    prev_stop_idx = stop_lookups.get(str(prev_stop), None)
    # next stop
    next_stop = row["stop_id"]
    next_stop_idx = stop_lookups.get(str(next_stop), None)
    # segment time
    avg_seg_time = row["avg_segment_time"]
    # add edge
    network_structure.add_edge(
        prev_stop_idx,
        next_stop_idx,
        0,  # edge_idx
        "na-gtfs",  # nx_start_node_key
        "na-gtfs",  # nx_end_node_key
        0,  # length - don't use zero otherwise short-cutting will occur
        0,  # angle_sum - don't use zero otherwise short-cutting will occur
        1,  # imp_factor
        None,  # in_bearing
        None,  # out_bearing
        float(avg_seg_time),  # seconds
    )

# %%
from pathlib import Path

import networkx as nx
import pandas as pd
from cityseer.tools import graphs, io, mock
from cityseer.metrics import networks
from cityseer import rustalgos
from pyproj import Transformer


# %%
# prepare GTFS data
G: nx.MultiGraph = mock.mock_graph()
G = graphs.nx_simple_geoms(G)
gtfs_data_path = "temp"
mock.mock_gtfs_stops_txt(gtfs_data_path)
graph_crs = 32630
nodes_gdf, edges_gdf, network_structure = io.network_structure_from_nx(G, graph_crs)
distances = [500, 1000]
#
nodes_gdf = networks.node_centrality_shortest(
    network_structure=network_structure,
    nodes_gdf=nodes_gdf,
    distances=distances,
)
nodes_gdf_w_trans = nodes_gdf.copy()
#
nodes_gdf_w_trans, edges_gdf, network_structure = io.add_transport_gtfs(
    gtfs_data_path, nodes_gdf_w_trans, edges_gdf, network_structure, graph_crs
)
nodes_gdf_w_trans = networks.node_centrality_shortest(
    network_structure=network_structure,
    nodes_gdf=nodes_gdf_w_trans,
    distances=distances,
)
nodes_gdf_w_trans

# %%
