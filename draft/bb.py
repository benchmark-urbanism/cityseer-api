# %%
from pathlib import Path

import networkx as nx
import pandas as pd
from cityseer.tools import graphs, io, mock
from pyproj import Transformer

# %%
# prepare GTFS data
G: nx.MultiGraph = mock.mock_graph()
G = graphs.nx_simple_geoms(G)
gtfs_data_path = "temp"
mock.mock_gtfs_stops_txt(gtfs_data_path)

graph_crs = 3395
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
stops

# %%
stop_times = pd.read_csv(gtfs_path / "stop_times.txt")
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
stop_times

# %%
# Sort stop_times for proper sequencing
stop_times.sort_values(by=["trip_id", "stop_sequence"], inplace=True)
stop_times

# %%
# Compute travel time between consecutive stops
stop_times["segment_time"] = stop_times.groupby("trip_id")["arrival_seconds"].diff()
stop_times

# %%
# Drop NaN values (first stop in each trip has no previous stop)
stop_times.dropna(subset=["segment_time"], inplace=True)
stop_times

# %%
# Compute average travel time between stops
avg_segment_time = stop_times.groupby(["stop_id"]).mean(numeric_only=True)["segment_time"].fillna(0)
avg_segment_time

# %%
# transformer to convert lat/lon to graph crs
transformer = Transformer.from_crs(4326, graph_crs, always_xy=True)
# add nodes for stops
for _, row in stops.iterrows():
    # wait_time = avg_wait_time.get(row["stop_id"], 0)  # Default to 0 if missing
    e, n = transformer.transform(row["stop_lon"], row["stop_lat"])
    nearest_idx, next_nearest_idx = network_structure.assign_to_network((e, n), max_netw_assign_dist)
    new_stop_key = str(row["stop_id"])
    new_stop_idx = network_structure.add_node(
        new_stop_key,
        float(e),
        float(n),
        True,  # live
        float(1),  # weight
    )
    # add edges between stops and pedestrian network
    for near_node_idx in [nearest_idx, next_nearest_idx]:
        if near_node_idx is not None:
            network_structure.add_edge(
                new_stop_idx,
                near_node_idx,
                0,  # edge_idx
                "na-gtfs",  # nx_start_node_key
                "na-gtfs",  # nx_end_node_key
                0,  # length
                0,  # angle_sum
                0,  # imp_factor
                0,  # in_bearing
                0,  # out_bearing
                None,  # minutes
            )
# add edges between stops
for _, trip_df in stop_times.groupby("trip_id"):
    prev_stop = None
    for _, row in trip_df.iterrows():
        if prev_stop:
            segment_time = avg_segment_time.get(row["stop_id"], 0)  # Default to 0 if missing
            print(segment_time)
            # G.add_edge(prev_stop, row["stop_id"], segment_time=segment_time)
        prev_stop = row["stop_id"]

# %%
