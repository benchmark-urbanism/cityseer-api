#%%
from matplotlib import colors
import numpy as np
from shapely import geometry
import utm

from cityseer.metrics import networks
from cityseer.tools import graphs, plot, mock

# 81, 312, 339
#%% -1.43884,53.24618
lng, lat = -1.43884, 53.24618
G_utm = mock.make_buffered_osm_graph(lng, lat, 15000)
# select extents for plotting
easting, northing = utm.from_latlon(lat, lng)[:2]
# buffer
buff = geometry.Point(easting, northing).buffer(500)
# extract extents
min_x, min_y, max_x, max_y = buff.bounds
# plot
plot.plot_nX(G_utm,
             labels=False,
             plot_geoms=False,
             x_lim=(min_x, max_x),
             y_lim=(min_y, max_y),
             figsize=(20, 20),
             dpi=200)

# %%
# basic initial cleanup
G = graphs.nX_simple_geoms(G_utm)
G = graphs.nX_remove_filler_nodes(G)
G = graphs.nX_remove_dangling_nodes(G, despine=20, remove_disconnected=True)
G = graphs.nX_remove_filler_nodes(G)
plot.plot_nX(G,
             labels=False,
             plot_geoms=True,
             x_lim=(min_x, max_x),
             y_lim=(min_y, max_y),
             figsize=(20, 20),
             dpi=200)

#%%
# initial pass of spatial consolidation to cleanup intersections
G1 = graphs.nX_consolidate_nodes(G,
                                 buffer_dist=10,
                                 min_node_group=3,
                                 min_node_degree=1,
                                 crawl=True,
                                 cent_min_degree=3,
                                 merge_edges_by_midline=True,
                                 multi_edge_len_factor=1.25,
                                 multi_edge_min_len=100)
# plot.plot_nX(G1, x_lim=(min_x, max_x), y_lim=(min_y, max_y), plot_geoms=True, figsize=(20, 20), dpi=200)
plot.plot_nX(G1,
             labels=False,
             plot_geoms=True,
             x_lim=(min_x, max_x),
             y_lim=(min_y, max_y),
             figsize=(20, 20),
             dpi=200)

# %%
from importlib import reload
reload(graphs)


# split opposing line geoms to facilitate parallel merging
G2 = graphs.nX_split_opposing_geoms(G1,
                                    buffer_dist=15,
                                    merge_edges_by_midline=True,
                                    multi_edge_len_factor=1.25,
                                    multi_edge_min_len=100)
# plot.plot_nX(G2, x_lim=(min_x, max_x), y_lim=(min_y, max_y), plot_geoms=True, figsize=(20, 20), dpi=200)
plot.plot_nX(G2,
             labels=False,
             plot_geoms=True,
             x_lim=(min_x, max_x),
             y_lim=(min_y, max_y),
             figsize=(20, 20),
             dpi=200)

# %%
from importlib import reload
reload(graphs)

G3 = graphs.nX_consolidate_nodes(G2,
                                 buffer_dist=15,
                                 crawl=True,
                                 merge_edges_by_midline=True,
                                 cent_min_degree=4,
                                 multi_edge_len_factor=1.25,
                                 multi_edge_min_len=100)
# plot.plot_nX(G3, x_lim=(min_x, max_x), y_lim=(min_y, max_y), plot_geoms=True, figsize=(20, 20), dpi=200)
plot.plot_nX(G3,
             labels=False,
             plot_geoms=True,
             x_lim=(min_x, max_x),
             y_lim=(min_y, max_y),
             figsize=(20, 20),
             dpi=200)


#%%
from importlib import reload
reload(graphs)

decomp = graphs.nX_decompose(G3, decompose_max=50)
plot.plot_nX(decomp,
             plot_geoms=True,
             x_lim=(min_x, max_x),
             y_lim=(min_y, max_y),
             figsize=(20, 20),
             dpi=200)

#%%
for gr in [G3, decomp]:
    for s, e, d in gr.edges(data=True):
        if not d['geom'].is_valid:
            print('not valid', s, e, d)
        if d['geom'].type == 'Point':
            print(d['geom'].type, s, e, d)
        if d['geom'].length < 2:
            print(d['geom'].length, s, e, d)


# %%
# create a Network layer from the networkX graph
N = networks.Network_Layer_From_nX(G3, distances=[1000, 5000, 10000])
# the underlying method allows the computation of various centralities simultaneously, e.g.
N.compute_node_centrality(measures=['node_harmonic', 'node_betweenness'])
#
G_metrics = N.to_networkX()

#  %%
# plot centrality
cent = []
for node, data in G_metrics.nodes(data=True):
    cent.append(data['metrics']['centrality']['node_harmonic'][10000])

# custom colourmap
cmap = colors.LinearSegmentedColormap.from_list('cityseer', ['#64c1ff', '#d32f2f'])
# mask outliers
cent = np.array(cent)
cent = np.clip(cent, cent.min(), np.percentile(cent, 99.5))
# normalise the values
segment_harmonic_vals = colors.Normalize()(cent)
# cast against the colour map
segment_harmonic_cols = cmap(segment_harmonic_vals)
# plot
plot.plot_nX(G3,
             node_colour=segment_harmonic_cols,
             x_lim=(min_x, max_x),
             y_lim=(min_y, max_y),
             plot_geoms=True,
             figsize=(20, 20),
             dpi=200)
