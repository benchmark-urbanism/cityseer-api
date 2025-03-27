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


# %%
# prepare GTFS data
G: nx.MultiGraph = mock.mock_graph()
G = graphs.nx_simple_geoms(G)
# G = graphs.nx_decompose(G, 100)
gtfs_data_path = "temp"
mock.mock_gtfs_stops_txt(gtfs_data_path)
graph_crs = 32630
nodes_gdf, edges_gdf, network_structure = io.network_structure_from_nx(G, graph_crs)
max_netw_assign_dist = 400
speed_m_s = 1.3333
distances = [500, 1000]
#
nodes_gdf = networks.node_centrality_shortest(
    network_structure=network_structure,
    nodes_gdf=nodes_gdf,
    distances=distances,
)
nodes_gdf_w_trans = nodes_gdf.copy()
edges_gdf_w_trans = edges_gdf.copy()

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
