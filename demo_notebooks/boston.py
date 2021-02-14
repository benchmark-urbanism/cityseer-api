#%%
from matplotlib import colors
import numpy as np
from shapely import geometry
import utm

from cityseer.metrics import networks
from cityseer.util import graphs, plot, mock


# %% create a test graph
lng, lat = -0.13435325883010577, 51.51197802142709
G_utm = mock.make_buffered_osm_graph(lng, lat, 2000)
# select extents for plotting
easting, northing = utm.from_latlon(lat, lng)[:2]
# buffer
buff = geometry.Point(easting, northing).buffer(1000)
# extract extents
min_x, min_y, max_x, max_y = buff.bounds
# plot
plot.plot_nX(G_utm, x_lim=(min_x, max_x), y_lim=(min_y, max_y), plot_geoms=False, figsize=(20, 20), dpi=200)

# %%
# basic initial cleanup
G = graphs.nX_simple_geoms(G_utm)
G = graphs.nX_remove_filler_nodes(G)
G = graphs.nX_remove_dangling_nodes(G, despine=10, remove_disconnected=True)
G = graphs.nX_remove_filler_nodes(G)
plot.plot_nX(G, x_lim=(min_x, max_x), y_lim=(min_y, max_y), plot_geoms=True, figsize=(20, 20), dpi=200)

# %%
# initial pass of spatial consolidation to cleanup major intersections
G1 = graphs.nX_consolidate_spatial(G,
                                   buffer_dist=10,
                                   min_node_threshold=2,
                                   min_node_degree=1,
                                   crawl=True,
                                   centroid_policy_min_degree=3,
                                   # centroid_policy_min_len_factor=0.9,
                                   merge_edges_by_midline=True,
                                   max_length_discrepancy_ratio=1.25,
                                   discrepancy_ratio_min_length=100)
plot.plot_nX(G1, x_lim=(min_x, max_x), y_lim=(min_y, max_y), plot_geoms=True, figsize=(20, 20), dpi=200)

# %%
# split opposing line geoms to facilitate parallel merging
G2 = graphs.nX_split_opposing_geoms(G1,
                                    buffer_dist=15,
                                    use_midline=True,
                                    max_length_discrepancy_ratio=1.25,
                                    discrepancy_ratio_min_length=100)
plot.plot_nX(G2, x_lim=(min_x, max_x), y_lim=(min_y, max_y), plot_geoms=True, figsize=(20, 20), dpi=200)

# %%
G3 = graphs.nX_consolidate_spatial(G2,
                                   buffer_dist=15,
                                   min_node_threshold=2,
                                   min_node_degree=2,
                                   max_cumulative_degree=5,
                                   neighbour_policy='indirect',
                                   crawl=False,
                                   merge_edges_by_midline=True,
                                   max_length_discrepancy_ratio=1.25,
                                   discrepancy_ratio_min_length=100)
plot.plot_nX(G3, x_lim=(min_x, max_x), y_lim=(min_y, max_y), plot_geoms=True, figsize=(20, 20), dpi=200)

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
