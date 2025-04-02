#%%
from pathlib import Path

import networkx as nx
import pandas as pd
from cityseer.tools import graphs, io, mock
from cityseer.metrics import networks
from cityseer import rustalgos
from pyproj import Transformer
import pathlib
from shapely import geometry
import matplotlib.pyplot as plt
import numpy as np
from cityseer.metrics import layers, networks
from cityseer.tools import graphs, io, mock, plot
from matplotlib import colors
import geopandas as gpd

# %%
# prepare GTFS data
streets_gpd = gpd.read_file("../temp/madrid_streets/street_network.gpkg")
streets_gpd = streets_gpd.explode(reset_index=True)
G = io.nx_from_generic_geopandas(streets_gpd)
G_dual = graphs.nx_to_dual(G)

#%%
nodes_gdf, edges_gdf, network_structure = io.network_structure_from_nx(G)
max_netw_assign_dist = 400
speed_m_s = 1.3333
distances = [500, 1000]

#%%
gtfs_data_path = "../temp/madrid_metro"
gtfs_path = Path(gtfs_data_path)
if not gtfs_path.exists():
    raise FileNotFoundError(f"GTFS data not found at {gtfs_data_path}")
if not (gtfs_path / "stops.txt").exists():
    raise FileNotFoundError(f"GTFS stops.txt not found at {gtfs_data_path}")
if not (gtfs_path / "stop_times.txt").exists():
    raise FileNotFoundError(f"GTFS stop_times.txt not found at {gtfs_data_path}")
# load GTFS stops data - rename stop_id to include gtfs_data_path
stops = pd.read_csv(gtfs_path / "stops.txt")
stops["stop_id"] = stops["stop_id"].apply(lambda sid: f"gtfs-{gtfs_data_path}-{sid}")
# load GTFS stop times data - rename stop_id to include gtfs_data_path
stop_times = pd.read_csv(gtfs_path / "stop_times.txt")
stop_times["stop_id"] = stop_times["stop_id"].apply(lambda sid: f"gtfs-{gtfs_data_path}-{sid}")
stop_times

#%%
# prepare arrival time
stop_times["arrival_time"] = pd.to_datetime(stop_times["arrival_time"], format="%H:%M:%S").dt.time
## first of each trip will be NaN
stop_times["arrival_seconds"] = stop_times["arrival_time"].apply(lambda t: t.hour * 3600 + t.minute * 60 + t.second)
stop_times

#%%
stop_times.sort_values(by=["stop_id", "arrival_seconds"], inplace=True)
# compute gaps between consecutive arrivals
stop_times["wait_time"] = stop_times.groupby("stop_id")["arrival_seconds"].diff()
stop_times

#%%
##
stop_times["wait_time"] = stop_times.groupby("stop_id")["wait_time"].transform(lambda x: x.fillna(x.mean()))

stop_times

#%%
# average wait time at stations
avg_wait_time = stop_times.groupby("stop_id")["wait_time"].mean().fillna(0)
avg_wait_time = avg_wait_time / 2
# merge avg_wait_time into stop times data
stops = stops.merge(avg_wait_time.rename("avg_wait_time"), on="stop_id", how="left")
stops

#%%
nodes_gdf_w_trans, edges_gdf_w_trans, network_structure, _, _ = io.add_transport_gtfs(
    gtfs_data_path, nodes_gdf_w_trans, edges_gdf_w_trans, network_structure, graph_crs
)
nodes_gdf_w_trans = networks.node_centrality_shortest(
    network_structure=network_structure,
    nodes_gdf=nodes_gdf_w_trans,
    distances=distances,
)
nodes_gdf_w_trans

#%%
edges_gdf_w_trans

# %%
for nodes, edges in [(nodes_gdf, edges_gdf), (nodes_gdf_w_trans, edges_gdf_w_trans)]:
    fig, ax = plt.subplots()
    nodes.plot(column='cc_harmonic_1000', cmap='magma', ax=ax)
    edges.plot(ax=ax)

# %%
